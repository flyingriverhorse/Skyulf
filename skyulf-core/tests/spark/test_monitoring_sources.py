"""Distributed monitoring retains original receipt and label snapshot identities."""

from datetime import UTC, datetime, timedelta

import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_config import MonitorConfig
from skyulf.integrations.databricks.observability.monitoring.spark import (
    spark_monitoring_sources as sources,
)


def config(**options):
    """Keep deliberately tiny local limits to catch accidental driver materialization."""
    return MonitorConfig(
        environment="test",
        project="spark",
        model_name="a.b.model",
        model_version="2",
        source_table="a.b.source",
        prediction_table="a.b.predictions",
        max_rows=1,
        max_bytes=1,
        **options,
    )


def test_prediction_source_join_remains_distributed_and_pinned(spark, monkeypatch):
    """Large Spark populations must ignore local row caps and use the original source."""
    end = datetime(2026, 10, 5, tzinfo=UTC)
    predictions = spark.createDataFrame(
        [(1, "r", "a.b.model", "2", 4.0), (2, "r", "a.b.model", "2", 5.0)],
        "id long, run_id string, model_name string, model_version string, prediction double",
    )
    features = spark.createDataFrame([(1, 3.0), (2, 4.0)], "id long, x double")
    reads = []

    def read(session, table, version):
        """Expose accidental use of a mutable latest source version."""
        reads.append((table, version))
        return predictions if table.endswith("predictions") else features

    monkeypatch.setattr(sources, "read_snapshot", read)
    monkeypatch.setattr(sources, "table_identity", lambda session, table: table)
    monkeypatch.setattr(sources, "snapshot_at", lambda *args: 8)
    monkeypatch.setattr(
        sources,
        "_window_receipts",
        lambda *args: {
            "r": {
                "receipt": {
                    "run_id": "r",
                    "source_table_id": "a.b.source",
                    "target_table_id": "a.b.predictions",
                    "model_name": "a.b.model",
                    "model_version": "2",
                    "source_version": 3,
                },
                "commit_version": 8,
                "committed_us": int((end - timedelta(hours=1)).timestamp() * 1e6),
            }
        },
    )
    current, saved, evidence, observed = sources.read_spark_observation(
        spark,
        config(),
        "2",
        ("id",),
        ("x",),
        probabilities=0,
        as_of=end,
        start=end - timedelta(days=1),
        end=end,
    )
    assert current.count() == saved.count() == 2
    assert ("a.b.source", 3) in reads
    assert evidence["batches"][0]["source_version"] == 3
    assert observed == end - timedelta(hours=1)


def test_labels_are_joined_on_spark_and_replacement_rejected(spark, monkeypatch):
    """A recreated labels table must not silently change the observation population."""
    cutoff = datetime(2026, 10, 5, tzinfo=UTC)
    predictions = spark.createDataFrame([(1, 3.0), (2, 4.0)], "id long, prediction double")
    labels = spark.createDataFrame(
        [(1, 3.0, cutoff), (2, 4.0, cutoff), (3, 8.0, cutoff)],
        "id long, target double, available_at timestamp",
    )
    monkeypatch.setattr(sources, "read_snapshot", lambda *args: labels)
    monkeypatch.setattr(sources, "snapshot_at", lambda *args: 4)
    monkeypatch.setattr(sources, "table_identity", lambda *args: "original")
    enrollment = config(label_table="a.b.labels", result_available_at_column="available_at")
    joined, evidence = sources.read_spark_labels(
        spark,
        enrollment,
        ("id",),
        "target",
        predictions,
        cutoff,
    )
    assert joined is not None
    assert joined.count() == 2
    assert evidence["label_table_id"] == "original"
    identities = iter(["original", "replacement"])
    monkeypatch.setattr(sources, "table_identity", lambda *args: next(identities))
    with pytest.raises(ValueError, match="replaced"):
        sources.read_spark_labels(spark, enrollment, ("id",), "target", predictions, cutoff)
