"""Slow monitoring preparation must expose its last operation before completion."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_config import MonitorConfig
from skyulf.integrations.databricks.observability.monitoring.monitoring_source_evidence import (
    source_evidence,
)
from skyulf.integrations.databricks.observability.monitoring.spark import (
    spark_monitoring_reference as reference,
)
from skyulf.integrations.databricks.training.fitting.candidate import TrainingSpec


@pytest.fixture(autouse=True)
def reference_admission(monkeypatch):
    """Keep progress tests at the artifact boundary; admission has separate concurrency tests."""
    monkeypatch.setattr(reference, "ensure_monitoring_store", lambda *a: "a.monitoring")
    monkeypatch.setattr(reference, "hold_reference_mutation", lambda *a: nullcontext("owner"))
    monkeypatch.setattr(reference, "require_reference_mutation", Mock())


def _config():
    """Use one concrete component to bind every visible stage to its model version."""
    return MonitorConfig(
        environment="test",
        project="progress",
        model_name="a.b.model",
        model_version="1",
        source_table="a.b.source",
        prediction_table="a.b.predictions",
        execution_engine="spark",
        reference_namespace="a.monitoring",
    )


def test_reference_load_failure_identifies_stage_and_preserves_exception(monkeypatch, capsys):
    """A blocked or failing download must be distinguishable from Spark table writes."""
    failure = RuntimeError("artifact transport failed")
    monkeypatch.setattr(reference, "load_monitoring_artifact", Mock(side_effect=failure))
    with pytest.raises(RuntimeError) as caught:
        reference.prepare_spark_monitoring_reference(Mock(), _config())
    output = capsys.readouterr().out
    assert caught.value is failure
    assert "STARTED | reference.load_model a.b.model/1" in output
    assert "FAILED | reference.load_model a.b.model/1" in output
    assert "reference.write_" not in output


def test_reference_progress_covers_each_population_and_holdout(monkeypatch, capsys):
    """Combined population, shared source and holdout measurement need separate timings."""
    frame = pd.DataFrame({"id": [10, 20, 30], "x": [1, 2, 3], "y": [2, 4, 6]})
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(fitted_engine="pandas", project_source_sha256="source")
    )
    spec = TrainingSpec(
        table="a.b.training",
        version=1,
        record_key_columns=("id",),
        max_rows=100,
        max_bytes=10000,
        split_strategy="random",
        input_columns=("x",),
        target_column="y",
    )
    evidence = {"training_run_id": "run", "model_version": "1"}
    monkeypatch.setattr(
        reference, "load_monitoring_artifact", lambda *a, **k: (artifact, spec, {}, evidence)
    )
    monkeypatch.setattr(reference, "ensure_owned_object", lambda *a: False)
    monkeypatch.setattr(reference, "table_identity", lambda *a: "table-id")
    monkeypatch.setattr(reference, "read_training_snapshot", lambda *a: frame)
    monkeypatch.setattr(reference, "make_registry_client", Mock())
    monkeypatch.setattr(reference, "require_mlflow", Mock())
    monkeypatch.setattr(
        reference,
        "reference_document",
        lambda *a: source_evidence(frame, spec.source_columns, spec.dataset_id),
    )
    monkeypatch.setattr(reference, "validate_training_evidence", Mock())
    monkeypatch.setattr(
        reference, "split_labeled_snapshot", lambda *a, **k: (frame.head(2), frame.tail(1), None)
    )
    monkeypatch.setattr(reference, "read_snapshot", lambda *a: SimpleNamespace(schema=None))
    monkeypatch.setattr(
        reference,
        "_prepared_frame",
        lambda spark, data, *a: SimpleNamespace(
            schema=SimpleNamespace(jsonValue=lambda: {"columns": list(data.columns)})
        ),
    )
    write = Mock(side_effect=lambda spark, name, data: {"name": name, "id": "id", "version": 0})
    monkeypatch.setattr(reference, "_write_population", write)
    monkeypatch.setattr(reference, "measure_holdout_values", lambda *a: {"values": {"rmse": 0.1}})

    receipt = reference.prepare_spark_monitoring_reference(Mock(), _config())

    output = capsys.readouterr().out
    for stage in (
        "load_model",
        "read_training",
        "write_population",
        "write_source",
        "measure_holdout",
        "write_receipt",
    ):
        assert f"STARTED | reference.{stage} a.b.model/1" in output
        assert f"COMPLETED | reference.{stage} a.b.model/1" in output
    assert write.call_count == 3
    assert receipt["holdout"]["values"]["rmse"] == 0.1


def test_quality_failure_identifies_component_before_remote_work(monkeypatch, capsys):
    """One slow component must not make the entire set look silently stuck."""
    from skyulf.inference.model_set import ModelSetArtifact
    from skyulf.integrations.databricks.model_sets import model_set_quality as quality

    artifact = Mock(
        spec=ModelSetArtifact,
        manifest=SimpleNamespace(
            quality_evidence={"expected_champion_version": None, "comparisons": {"risk": "digest"}},
            components=[SimpleNamespace(branch="risk")],
        ),
    )
    failure = ValueError("cannot load candidate")
    monkeypatch.setattr(quality, "_evaluate_component", Mock(side_effect=failure))
    with pytest.raises(ValueError) as caught:
        quality.evaluate_model_set_quality(
            Mock(), artifact, None, expected_champion_version=None, max_rows=100, max_bytes=1024
        )
    output = capsys.readouterr().out
    assert caught.value is failure
    assert "STARTED | model_set_quality risk" in output
    assert "FAILED | model_set_quality risk" in output
