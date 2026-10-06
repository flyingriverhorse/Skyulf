"""Prepared reference reads must never replay local training populations."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_config import MonitorConfig
from skyulf.integrations.databricks.observability.monitoring.spark import (
    spark_monitoring_reference as reference,
)


def test_reference_read_uses_prepared_snapshot_and_rejects_different_artifact(monkeypatch):
    """Saved immutable populations must belong to the exact registered artifact."""
    config = MonitorConfig(
        environment="test",
        project="spark",
        model_name="a.b.model",
        model_version="1",
        source_table="a.b.source",
        prediction_table="a.b.predictions",
        execution_engine="spark",
        reference_namespace="a.monitoring",
    )
    artifact = SimpleNamespace(manifest=SimpleNamespace(pipeline_sha256="digest"))
    evidence = {"model_version": "1", "model_digest": "digest", "dataset_id": "dataset"}
    spec = SimpleNamespace()
    monkeypatch.setattr(
        reference,
        "load_monitoring_artifact",
        lambda *args, **kwargs: (artifact, spec, {}, evidence.copy()),
    )
    prepared = {
        "evidence": evidence.copy(),
        "tables": {"train": {"name": "a.b.ref", "id": "id", "version": 0}},
    }
    monkeypatch.setattr(reference, "read_reference_metadata", lambda *args: prepared)
    frame = Mock()
    read = Mock(return_value=frame)
    monkeypatch.setattr(reference, "read_reference_population", read)
    result = reference.load_spark_monitoring_reference(
        None, config, tracking_uri=None, registry_uri=None
    )
    assert result[2] is frame
    assert read.call_count == 1
    prepared["evidence"]["model_digest"] = "other"
    with pytest.raises(ValueError, match="identity"):
        reference.load_spark_monitoring_reference(
            None, config, tracking_uri=None, registry_uri=None
        )
