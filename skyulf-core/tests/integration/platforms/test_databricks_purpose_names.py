"""Purpose names retain the existing Databricks integration contracts."""

import importlib
import json
from hashlib import sha256

import numpy as np
import pandas as pd
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
def test_public_names_are_same_objects(new, old):
    """Both supported spellings must share implementation and type identity."""
    import skyulf.integrations.databricks as db

    assert hasattr(db, new), f"Missing purpose-based public API: {new}"
    assert getattr(db, new) is getattr(db, old)
    assert new in db.__all__ and old in db.__all__


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
def test_old_hierarchical_and_flat_imports_share_canonical_module(old, new):
    """Compatibility paths must preserve globals used by saved code and caller hooks."""
    prefix = "skyulf.integrations.databricks."
    legacy = importlib.import_module(prefix + old)
    canonical = importlib.import_module(prefix + new)
    flat = importlib.import_module(prefix + old.rsplit(".", 1)[-1])
    assert legacy is canonical is flat


def _workflow(path, **changes):
    """Build a caller-frame config whose runtime remains a declared policy."""
    from skyulf.integrations.databricks import (
        InputSource,
        ModelSelection,
        OutputSink,
        WorkflowConfig,
    )

    values = {
        "runtime": "local",
        "engine": "pandas",
        "source": InputSource(kind="caller_frame"),
        "model": ModelSelection(kind="local_pipeline", path=str(path)),
        "sink": OutputSink(kind="return_frame"),
    }
    values.update(changes)
    return WorkflowConfig.model_validate(values)


def test_runtime_policy_does_not_detect_machine_location(tmp_path, monkeypatch):
    """Driver environment markers cannot turn local runtime into a workstation restriction."""
    from skyulf.inference.local_pipeline import save_local_pipeline
    from skyulf.integrations.databricks import OutputSink, preflight, prepare_workflow

    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "test-driver")
    frame = pd.DataFrame({"x": np.arange(8, dtype=float), "target": np.arange(8) * 2.0})
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    save_local_pipeline(pipeline, tmp_path / "pipeline")
    outputs = []
    for runtime in ("local", "databricks"):
        prepared = prepare_workflow(_workflow(tmp_path / "pipeline", runtime=runtime))
        assert prepared.preflight.ready
        outputs.append(prepared.predict(pd.DataFrame({"x": [2.0, 4.0]})))
    pd.testing.assert_frame_equal(outputs[0], outputs[1])
    np.testing.assert_allclose(outputs[0]["prediction"], [4.0, 8.0])
    sink_config = _workflow(
        tmp_path / "pipeline", sink=OutputSink(kind="uc_delta", table="c.s.predictions")
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


def test_workflow_schema_only_changes_its_public_title():
    """New Python names cannot silently alter persisted workflow validation contracts."""
    from skyulf.integrations.databricks import WorkflowConfig

    schema = WorkflowConfig.model_json_schema()
    assert schema["title"] == "WorkflowConfig"
    schema["title"] = "LocalWorkflowConfig"
    digest = sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == "800e70c7240282be3c7ff4752c70bc62c9a44cfcd2f9a3d3de3269c0da492e66"


def test_workflow_rejection_preserves_existing_registry_hook(monkeypatch):
    """Legacy rejection callers must still intercept the low-level registry operation."""
    from skyulf.integrations.databricks.lifecycle import local_approval as approval

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
    result = approval.reject_local_candidate(
        {"promotion_policy": "manual_approval", "model_name": "c.s.model"},
        candidate_version="1",
        comparison_sha256="a" * 64,
        expected_champion_version=None,
        rejection_reason="Operator decision",
    )
    assert result is receipt
    assert calls[0][0] is report
    assert calls[0][1]["reason"] == "Operator decision"
    assert approval.reject_local_candidate is approval.reject_workflow_candidate
