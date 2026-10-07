"""Real Delta feature materialization creates, updates and replays stable inputs."""

import uuid
from datetime import datetime

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


def test_separate_group_tasks_reuse_pinned_versions(delta_spark, tmp_path):
    """A later feature commit must not replace the version chosen for a selective run."""
    spark = delta_spark
    suffix = uuid.uuid4().hex
    base, source, output, merged = [
        f"spark_catalog.default.feature_{name}_{suffix}"
        for name in ("base", "raw", "group", "merged")
    ]
    path = tmp_path / "src/feature_groups/company.py"
    path.parent.mkdir(parents=True)
    path.write_text("def compute(frame):\n    return frame.select('id', 'at', 'value')\n")
    group = FeatureGroup(
        "company", source, output, "src/feature_groups/company.py:compute", ("value",)
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
        publish_feature_table(spark, changed, output, ("id",), "at")
        pinned = build_feature_group(spark, tmp_path, plan, reused, "company")
        merge_feature_groups(spark, plan, reused, {"company": pinned})
        assert pinned["reused"] is True
        assert spark.table(output).first().value == 999.0
        assert spark.table(merged).first().value == 10.0
    finally:
        for table in (base, source, output, merged):
            spark.sql("DROP TABLE IF EXISTS " + ".".join(f"`{part}`" for part in table.split(".")))
