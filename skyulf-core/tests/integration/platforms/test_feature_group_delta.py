"""Real Delta feature materialization creates, updates and replays stable inputs."""

import uuid
from datetime import datetime

import pytest

from skyulf.integrations.databricks.features import delta
from skyulf.integrations.databricks.features.config import FeatureGroup, FeaturePlan
from skyulf.integrations.databricks.features.delta import publish_feature_table, read_version
from skyulf.integrations.databricks.features.runtime import (
    build_feature_group,
    initialize_features,
    merge_feature_groups,
)


def test_feature_upsert_preserves_history_and_replays(delta_spark):
    """Feature corrections update keys without deleting earlier observation history."""
    spark = delta_spark
    table = "feature_test_" + uuid.uuid4().hex
    schema = "id long, at timestamp, value double"
    first = spark.createDataFrame([(1, datetime(2026, 1, 1), 10.0)], schema)
    second = spark.createDataFrame(
        [(1, datetime(2026, 1, 1), 11.0), (2, datetime(2026, 1, 2), 20.0)], schema
    )
    try:
        receipt = publish_feature_table(spark, first, table, ("id",), "at")
        assert receipt["rows"] == 1
        publish_feature_table(spark, second, table, ("id",), "at")
        publish_feature_table(spark, second, table, ("id",), "at")
        assert [row.value for row in spark.table(table).orderBy("id").collect()] == [11.0, 20.0]
        assert read_version(spark, table, receipt["version"]).first().value == 10.0
    finally:
        spark.sql(f"DROP TABLE IF EXISTS `{table}`")


def test_separate_group_tasks_refuse_superseded_pinned_merge(delta_spark, tmp_path):
    """Newer partial group publication must prevent a stale final table overwrite."""
    spark = delta_spark
    suffix = uuid.uuid4().hex
    namespace = spark.sql("SELECT current_catalog(), current_database()").first()
    assert namespace is not None
    base, source, output, merged = [
        f"{namespace[0]}.{namespace[1]}.feature_{name}_{suffix}"
        for name in ("base", "raw", "group", "merged")
    ]
    path = tmp_path / "src/features/groups/company.py"
    path.parent.mkdir(parents=True)
    path.write_text("def compute(frame):\n    return frame.select('id', 'at', 'value')\n")
    group = FeatureGroup(
        "company", source, output, "src/features/groups/company.py:compute", ("value",)
    )
    plan = FeaturePlan(base, merged, ("id",), "at", (group,))
    schema = "id long, at timestamp, value double"
    rows = spark.createDataFrame([(1, datetime(2026, 1, 1), 10.0)], schema)
    try:
        rows.selectExpr("id", "at", "1.0 AS target").write.format("delta").saveAsTable(base)
        rows.write.format("delta").saveAsTable(source)
        snapshot = initialize_features(spark, tmp_path, plan, "*")
        receipt = build_feature_group(spark, tmp_path, plan, snapshot, "company")
        merge_feature_groups(spark, plan, snapshot, {"company": receipt})
        reused = initialize_features(spark, tmp_path, plan, "")
        changed = spark.createDataFrame([(1, datetime(2026, 1, 1), 999.0)], schema)
        with pytest.raises(ValueError, match="publication"):
            publish_feature_table(spark, changed, output, ("id",), "at")
        pinned = build_feature_group(spark, tmp_path, plan, reused, "company")
        merge_feature_groups(spark, plan, reused, {"company": pinned})
        changed.write.format("delta").mode("overwrite").saveAsTable(source)
        newer = initialize_features(spark, tmp_path, plan, "*")
        build_feature_group(spark, tmp_path, plan, newer, "company")
        with pytest.raises(ValueError, match="superseded"):
            merge_feature_groups(spark, plan, reused, {"company": pinned})
        with pytest.raises(ValueError, match="superseded"):
            build_feature_group(spark, tmp_path, plan, snapshot, "company")
        assert pinned["reused"] is True
        assert spark.table(output).first().value == 999.0
        assert spark.table(merged).first().value == 10.0
    finally:
        for table in (base, source, output, merged):
            spark.sql("DROP TABLE IF EXISTS " + ".".join(f"`{part}`" for part in table.split(".")))


def test_feature_publication_retry_after_lost_merge_receipt(delta_spark, monkeypatch):
    """A real committed MERGE is safely replayed using its already durable intent."""
    spark = delta_spark
    table = "feature_retry_" + uuid.uuid4().hex
    schema = "id long, at timestamp, value double"
    first = spark.createDataFrame([(1, datetime(2026, 1, 1), 10.0)], schema)
    second = spark.createDataFrame([(1, datetime(2026, 1, 1), 20.0)], schema)
    publication = {
        "generation": 1,
        "snapshot_sha256": "a" * 64,
        "expected": {"table_id": None, "version": None},
    }
    try:
        original_receipt = publish_feature_table(
            spark, first, table, ("id",), "at", publication=publication
        )
        state = delta.publication_state(spark, table)
        assert state is not None
        newer = {
            "generation": 2,
            "snapshot_sha256": "b" * 64,
            "expected": {key: state[key] for key in ("table_id", "version")},
        }
        merge = delta._merge_rows

        def lose_response(*args):
            """Simulate transport failure after Delta committed the exact update."""
            merge(*args)
            raise RuntimeError("lost committed MERGE response")

        monkeypatch.setattr(delta, "_merge_rows", lose_response)
        with pytest.raises(RuntimeError, match="lost committed"):
            publish_feature_table(spark, second, table, ("id",), "at", publication=newer)
        monkeypatch.setattr(delta, "_merge_rows", merge)
        receipt = publish_feature_table(spark, second, table, ("id",), "at", publication=newer)
        with pytest.raises(ValueError, match="superseded"):
            publish_feature_table(spark, first, table, ("id",), "at", publication=publication)
        assert delta.read_version(spark, table, receipt["version"]).first().value == 20.0
        assert delta.read_version(spark, table, original_receipt["version"]).first().value == 10.0
    finally:
        spark.sql(f"DROP TABLE IF EXISTS `{table}`")
