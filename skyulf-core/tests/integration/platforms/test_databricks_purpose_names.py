"""Purpose names define the Databricks API without legacy local aliases."""

import importlib
import json
from hashlib import sha256

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pydantic import ValidationError

from skyulf.data.dataset import SplitDataset
from skyulf.pipeline import SkyulfPipeline


@pytest.mark.parametrize(
    ("new", "old"),
    [
        ("TrainingSpec", "LocalTrainingSpec"),
        ("CandidateResult", "LocalCandidateResult"),
        ("train_candidate", "train_local_candidate"),
        ("WorkflowConfig", "LocalWorkflowConfig"),
        ("PreparedWorkflow", "PreparedLocalWorkflow"),
        ("prepare_workflow", "prepare_local_workflow"),
        ("preflight", "preflight_local"),
        ("SourceSpec", "LocalSourceSpec"),
        ("ScoreResult", "LocalScoreResult"),
        ("fit_workflow", "fit_local_workflow"),
        ("read_source", "read_local_source"),
        ("score_source", "score_local_source"),
        ("evaluate_holdout", "evaluate_local_holdout"),
        ("run_frame_batch", "run_local_batch"),
        ("run_incremental_batch", "run_incremental_local_batch"),
        ("train_branches", "train_local_branches"),
    ],
)
def test_public_api_exposes_only_canonical_names(new, old):
    """One public spelling avoids duplicate APIs and misleading location names."""
    import skyulf.integrations.databricks as db

    assert hasattr(db, new), f"Missing purpose-based public API: {new}"
    assert not hasattr(db, old)
    assert new in db.__all__ and old not in db.__all__


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("training.fitting.local_retraining", "training.fitting.candidate"),
        ("scoring.local_sdk", "scoring.workflow"),
        ("scoring.batch.local_batch", "scoring.batch.frame_batch"),
        ("scoring.local_publish", "scoring.publish"),
        ("scoring.incremental.local_history", "scoring.incremental.history"),
        ("scoring.incremental.local_incremental", "scoring.incremental.incremental_batch"),
        ("lifecycle.local_workflow", "lifecycle.workflow"),
        ("lifecycle.local_approval", "lifecycle.approval"),
        ("training.local_branches", "training.branches"),
        ("training.tuning.local_cv", "training.tuning.cv"),
        ("training.tuning.local_search", "training.tuning.search"),
        ("training.tuning.local_search_results", "training.tuning.search_results"),
        ("training.fitting.local_pre_split", "training.fitting.pre_split"),
        ("training.fitting.local_ensemble", "training.fitting.ensemble"),
        ("training.weights.local_weights", "training.weights.weights"),
        ("training.shared.local_training_evidence", "training.shared.training_evidence"),
        ("training.competition.local_competition", "training.competition.competition"),
        ("observability.reports.local_explanations", "observability.reports.explanations"),
    ],
)
def test_canonical_modules_own_code_without_legacy_paths(old, new):
    """The clean migration must remove old files, not replace them with import hooks."""
    prefix = "skyulf.integrations.databricks."
    canonical = importlib.import_module(prefix + new)
    assert canonical.__name__ == prefix + new
    for legacy in (old, old.rsplit(".", 1)[-1], "_compat." + old.rsplit(".", 1)[-1]):
        with pytest.raises(ModuleNotFoundError) as error:
            importlib.import_module(prefix + legacy)
        assert error.value.name == prefix + legacy


def _workflow(path, **changes):
    """Build a caller-frame config whose runtime remains a declared policy."""
    from skyulf.integrations.databricks import (
        InputSource,
        ModelSelection,
        OutputSink,
        WorkflowConfig,
    )

    values = {
        "runtime": "standalone",
        "engine": "pandas",
        "source": InputSource(kind="caller_frame"),
        "model": ModelSelection(kind="local_pipeline", path=str(path)),
        "sink": OutputSink(kind="return_frame"),
    }
    values.update(changes)
    return WorkflowConfig.model_validate(values)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_runtime_policy_does_not_detect_machine_location(tmp_path, monkeypatch, engine):
    """Standalone execution works on cloud hosts without enabling Databricks publication."""
    from skyulf.inference.fitted_pipeline import save_pipeline
    from skyulf.integrations.databricks import OutputSink, preflight, prepare_workflow

    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "test-driver")
    frame = pd.DataFrame({"x": np.arange(8, dtype=float), "target": np.arange(8) * 2.0})
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    pipeline.fit(SplitDataset(train=native, test=native.head(0)), target_column="target")
    save_pipeline(pipeline, tmp_path / "pipeline")
    outputs = []
    for runtime in ("standalone", "databricks"):
        prepared = prepare_workflow(
            _workflow(tmp_path / "pipeline", runtime=runtime, engine=engine)
        )
        assert prepared.preflight.ready
        inputs = pd.DataFrame({"x": [2.0, 4.0]})
        output = prepared.predict(pl.from_pandas(inputs) if engine == "polars" else inputs)
        outputs.append(output.to_pandas() if isinstance(output, pl.DataFrame) else output)
    pd.testing.assert_frame_equal(outputs[0], outputs[1])
    np.testing.assert_allclose(outputs[0]["prediction"], [4.0, 8.0])
    sink_config = _workflow(
        tmp_path / "pipeline",
        engine=engine,
        sink=OutputSink(kind="uc_delta", table="c.s.predictions"),
    )
    assert "sink_runtime_mismatch" in {issue.code for issue in preflight(sink_config).issues}


def test_spark_inference_still_requires_explicit_policy():
    """Renaming cannot bypass distributed runtime, source, registry or engine admission."""
    from skyulf.integrations.databricks import InputSource, ModelSelection, OutputSink

    with pytest.raises(ValidationError, match="runtime='databricks'"):
        _workflow("model", inference_mode="spark")
    with pytest.raises(ValidationError, match="incremental UC source"):
        _workflow("model", runtime="databricks", inference_mode="spark")
    values = {
        "runtime": "databricks",
        "inference_mode": "spark",
        "source": InputSource(kind="uc_table", table="c.s.source", read_mode="incremental"),
        "model": ModelSelection(kind="local_pipeline", name="c.s.model", version="1"),
        "sink": OutputSink(kind="uc_delta", table="c.s.predictions"),
    }
    with pytest.raises(ValidationError, match="engine='pandas'"):
        _workflow("model", engine="polars", **values)
    assert _workflow("model", **values).inference_mode == "spark"


def test_training_spec_serialization_keeps_saved_keys():
    """The renamed dataclass must reproduce the pre-move persisted selection payload."""
    from skyulf.integrations.databricks import TrainingSpec
    from skyulf.integrations.databricks.training.fitting.candidate import training_spec_payload

    spec = TrainingSpec(
        table="c.s.training",
        version=3,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="y",
        max_rows=100,
        max_bytes=1048576,
        preprocessing_probe=True,
    )
    payload = training_spec_payload(spec, "pandas")
    restored = TrainingSpec.from_payload(json.loads(json.dumps(payload)))
    assert restored == spec
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == "4bb82749d44838f7c6e438f195784e2ff1efa79b91423516ab71600f0ec3e147"


def test_runtime_schema_separates_integration_policy_from_frame_engine():
    """Runtime names must describe integration policy while engine selects the frame type."""
    from skyulf.integrations.databricks import WorkflowConfig

    schema = WorkflowConfig.model_json_schema()
    assert schema["title"] == "WorkflowConfig"
    assert schema["properties"]["runtime"]["enum"] == ["standalone", "databricks", "spark"]
    assert schema["properties"]["engine"]["enum"] == ["pandas", "polars"]
    assert schema["properties"]["inference_mode"]["enum"] == ["local", "spark"]
    assert schema["properties"]["spark_udf_env_manager"]["enum"] == ["local", "virtualenv"]
    with pytest.raises(ValidationError, match="standalone"):
        _workflow("model", runtime="local")


def test_workflow_rejection_preserves_existing_registry_hook(monkeypatch):
    """The high-level workflow must still use the injectable low-level registry operation."""
    from skyulf.integrations.databricks.lifecycle import approval

    report = object()
    receipt = object()
    calls = []

    def reject(candidate_report, **options):
        """Record the existing injectable operation without changing any registry state."""
        calls.append((candidate_report, options))
        return receipt

    monkeypatch.setattr(approval, "require_mlflow", lambda: object())
    monkeypatch.setattr(approval, "make_registry_client", lambda *args: object())
    monkeypatch.setattr(
        approval, "load_candidate_evidence", lambda *args, **kwargs: (report, None, None, None)
    )
    monkeypatch.setattr(approval, "controlled_champion_version", lambda *args, **kwargs: None)
    monkeypatch.setattr(approval, "reject_candidate", reject)
    result = approval.reject_workflow_candidate(
        {"promotion_policy": "manual_approval", "model_name": "c.s.model"},
        candidate_version="1",
        comparison_sha256="a" * 64,
        expected_champion_version=None,
        rejection_reason="Operator decision",
    )
    assert result is receipt
    assert calls[0][0] is report
    assert calls[0][1]["reason"] == "Operator decision"
