"""Optional training diagnostics observe saved preprocessing without changing training."""

import json
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import polars as pl
import pytest

pytest.importorskip("mlflow")
from mlflow import MlflowClient

from skyulf.data.dataset import SplitDataset
from skyulf.integrations.databricks.lifecycle.local_workflow import training_spec
from skyulf.integrations.databricks.observability.reports.training_node_output import (
    render_training_node,
    training_report_document,
)
from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config
from skyulf.integrations.databricks.training.fitting import local_retraining as retraining
from skyulf.integrations.databricks.training.tuning.local_cv import LocalCVSpec
from skyulf.integrations.mlflow.runs.tracking import TrackingConfig, track_run
from skyulf.pipeline.seal import artifact_digest


def _spec():
    """Use a pinned minimal random split with a pre-change dataset identity."""
    return retraining.LocalTrainingSpec(
        table="workspace.test.labels",
        version=4,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=2000,
        max_bytes=1048576,
    )


@pytest.fixture
def local_tracking(tmp_path):
    """Keep actual MLflow writes local even when tests run inside Databricks."""
    uri = f"sqlite:///{(tmp_path / 'runs.db').as_posix()}"
    client = MlflowClient(tracking_uri=uri)
    client.create_experiment("probe", artifact_location=(tmp_path / "artifacts").as_uri())
    return TrackingConfig(enabled=True, tracking_uri=uri, experiment_name="probe")


def test_training_probe_preserves_legacy_identity_and_payload():
    """A diagnostic toggle must not invalidate pinned data or older saved evidence."""
    disabled = _spec()
    enabled = replace(disabled, preprocessing_probe=True)
    payload = retraining.training_spec_payload(enabled, "pandas")
    assert retraining.LocalTrainingSpec.from_payload(payload) == enabled
    payload.pop("preprocessing_probe")
    assert retraining.LocalTrainingSpec.from_payload(payload) == disabled
    assert disabled.preprocessing_probe is False
    assert (
        enabled.dataset_id
        == disabled.dataset_id
        == (
            "workspace.test.labels@4/random/"
            "6bc58cd1cb7f093da1979997e18737fe3dada247e718d2e4b5dd274320e2af27"
        )
    )


@pytest.mark.parametrize("value", [None, 0, 1, "true", {}, []])
def test_training_probe_rejects_non_boolean_settings(workflow_config, value):
    """Config must never enable diagnostics through truthy strings or numbers."""
    workflow_config["preprocessing_probe"] = value
    with pytest.raises(ValueError, match="preprocessing_probe must be boolean"):
        validate_workflow_config(workflow_config, action="train")


def test_training_probe_config_reaches_spec(workflow_config):
    """Generated workflow configuration must reach every shared candidate fit."""
    workflow_config["preprocessing_probe"] = True
    validate_workflow_config(workflow_config, action="train")
    assert training_spec(workflow_config).preprocessing_probe is True


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("enabled", [False, True])
def test_training_probe_logs_real_saved_apply_and_report(
    tmp_path, monkeypatch, local_tracking, engine, enabled
):
    """Actual fitting and MLflow reports must share bounded, redacted saved-state evidence."""
    frame = pd.DataFrame(
        {"id": range(1400), "x": [float(i % 13) for i in range(1400)], "target": range(1400)}
    )
    monkeypatch.setattr(retraining, "read_training_snapshot", lambda *_: frame)
    spec = replace(_spec(), preprocessing_probe=enabled)
    config = {
        "preprocessing": [
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"columns": ["x"], "strategy": "mean"},
            }
        ],
        "modeling": {"type": "linear_regression"},
    }
    original = deepcopy(config)
    with track_run(local_tracking, run_name="fit") as run:
        assert run.run_id is not None
        fitted = retraining.fit_candidate(
            None,
            spec,
            config,
            run=run,
            pipeline_config=config,
            artifact_path=tmp_path / "model",
            engine=engine,
            cv=LocalCVSpec(),
            risk_category=None,
        )
        assert config == original
        paths = {item.path for item in run.client.list_artifacts(run.run_id)}
        assert ("preprocessing_probe.json" in paths) is enabled
        if enabled:
            report = training_report_document(run.client, run.run_id, "preprocessing_probe.json")
            assert report["status"] == "passed", report
            assert report["sample_rows"] == 256
            assert report["admission"] == "diagnostic_only"
            assert report["fitted_engine"] == engine
            assert report["pipeline_sha256"] == fitted.artifact.manifest.pipeline_sha256
            assert report["steps"][0]["status"] == "passed"
            assert '"target"' not in json.dumps(report)
            run.client.log_dict(
                run.run_id,
                retraining.training_spec_payload(fitted.spec, engine),
                "candidate_training_spec.json",
            )
            html = render_training_node(run.client, run.run_id, {})
            assert "Preprocessing diagnostics" in html and "diagnostic_only" in html
        assert fitted.training_rows == 1120 and fitted.holdout_rows == 280


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_training_probe_redacts_failure_and_preserves_artifact(
    tmp_path, monkeypatch, local_tracking, engine
):
    """Diagnostic exceptions cannot leak sample values or fail a successful model fit."""
    from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow
    from skyulf.integrations.databricks.training.shared import preprocessing_checks

    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    artifact = fit_local_workflow(
        {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        SplitDataset(train=native, test=native.head(0)),
        target_column="target",
        artifact_path=tmp_path / "saved",
        max_rows=10,
        max_bytes=1048576,
    )
    before = artifact_digest(artifact.pipeline.feature_engineer.fitted_steps)

    def fail_probe(*args, **kwargs):
        """Model a callback error containing private row content."""
        raise ValueError("private row secret-value")

    monkeypatch.setattr(preprocessing_checks, "probe_fitted_preprocessing", fail_probe)
    with track_run(local_tracking, run_name="failure") as run:
        assert run.run_id is not None
        preprocessing_checks.log_preprocessing_probe(run, artifact, native, enabled=False)
        assert not run.client.list_artifacts(run.run_id)
        preprocessing_checks.log_preprocessing_probe(run, artifact, native, enabled=True)
        report = training_report_document(run.client, run.run_id, "preprocessing_probe.json")
        assert report["status"] == "failed" and report["error_type"] == "ValueError"
        assert "secret-value" not in json.dumps(report)
        preprocessing_checks.log_preprocessing_probe(run, artifact, native.head(0), enabled=True)
        empty = training_report_document(run.client, run.run_id, "preprocessing_probe.json")
        assert empty["status"] == "not_run" and empty["reason"] == "empty_holdout"
    assert artifact_digest(artifact.pipeline.feature_engineer.fitted_steps) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_training_probe_records_window_context_without_partitioning(
    tmp_path, local_tracking, engine
):
    """A successfully fitted window model must remain a context requirement in reports."""
    from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow
    from skyulf.integrations.databricks.training.shared.preprocessing_checks import (
        log_preprocessing_probe,
    )

    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    artifact = fit_local_workflow(
        {
            "preprocessing": [
                {
                    "name": "rolling",
                    "transformer": "RollingAggregate",
                    "params": {"columns": ["x"], "window": 2},
                }
            ],
            "modeling": {"type": "linear_regression"},
        },
        SplitDataset(train=native, test=native.head(0)),
        target_column="target",
        artifact_path=tmp_path / "saved",
        max_rows=10,
        max_bytes=1048576,
    )
    before = artifact_digest(artifact.pipeline.feature_engineer.fitted_steps)
    with track_run(local_tracking, run_name="context") as run:
        assert run.run_id is not None
        log_preprocessing_probe(run, artifact, native, enabled=True)
        report = training_report_document(run.client, run.run_id, "preprocessing_probe.json")
        assert report["status"] == "requires_context"
        assert report["steps"][0]["context"] == "window"
        assert report["steps"][0]["checks"] == []
    assert artifact_digest(artifact.pipeline.feature_engineer.fitted_steps) == before
