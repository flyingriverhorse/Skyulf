"""Saved preprocessing probes exercise apply without changing fitted state."""

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
from skyulf.pipeline import SkyulfPipeline
from skyulf.pipeline.seal import artifact_digest
from skyulf.registry import NodeRegistry


def _artifact(tmp_path, engine="pandas", steps=None):
    """Persist actual learned parameters and schemas before diagnostic execution."""
    frame = pd.DataFrame(
        {"value": [0.0, 2.0, 0.0, 2.0, 0.0, 2.0], "target": [0.0, 4.0, 0.0, 4.0, 0.0, 4.0]}
    )
    if engine == "polars":
        frame = pl.from_pandas(frame)
    config = {
        "preprocessing": steps
        if steps is not None
        else [
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"columns": ["value"], "strategy": "mean"},
            },
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["value"]}},
        ],
        "modeling": {"type": "linear_regression"},
    }
    pipeline = SkyulfPipeline(config)
    pipeline.fit(SplitDataset(train=frame, test=frame[:0]), target_column="target")
    path = tmp_path / "artifact"
    save_local_pipeline(pipeline, path)
    return load_local_pipeline(path)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_probe_uses_saved_apply_without_refit_or_mutation(tmp_path, monkeypatch, engine):
    """Learned values must survive all batch strategies and stay unchanged."""
    artifact = _artifact(tmp_path, engine)
    sample = pd.DataFrame({"value": [None, 17.0, -3.0, 4.0, 8.0]}, index=[8, 3, 3, 9, 1])
    if engine == "polars":
        sample = pl.from_pandas(sample)
    before = artifact_digest(artifact.pipeline.feature_engineer.fitted_steps)
    original = deepcopy(sample)

    def forbidden(*args, **kwargs):
        """Inference must not learn anything from the diagnostic sample."""
        raise AssertionError("Unexpected fit")

    for node in ("SimpleImputer", "StandardScaler"):
        monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    result = probe_fitted_preprocessing(artifact, sample, chunk_sizes=(2, 3))
    assert result["status"] == "passed"
    assert result["admission"] == "diagnostic_only"
    assert result["pipeline_sha256"] == artifact.manifest.pipeline_sha256
    assert [step["status"] for step in result["steps"]] == ["passed", "passed"]
    checks = {check["name"] for check in result["steps"][0]["checks"]}
    assert {"full", "repeat", "chunks:1", "chunks:2", "chunks:3", "reverse"} <= checks
    assert artifact_digest(artifact.pipeline.feature_engineer.fitted_steps) == before
    if isinstance(sample, pd.DataFrame):
        assert isinstance(original, pd.DataFrame) and sample.equals(original)
    else:
        assert isinstance(original, pl.DataFrame) and sample.equals(original)
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rows": 1},
        {"max_bytes": 1},
        {"chunk_sizes": (0,)},
        {"chunk_sizes": (True,)},
        {"max_rows": True},
    ],
)
def test_probe_rejects_invalid_or_exceeded_budgets_before_apply(tmp_path, kwargs):
    """Diagnostics must be bounded explicitly rather than silently sampling data."""
    artifact = _artifact(tmp_path)
    with pytest.raises(ValueError):
        probe_fitted_preprocessing(artifact, pd.DataFrame({"value": [1.0, 2.0]}), **kwargs)


def test_probe_rejects_manifest_schema_disagreement(tmp_path):
    """A diagnostic must use the same saved schema contract as worker admission."""
    artifact = _artifact(tmp_path)
    artifact.pipeline._inference_schemas = None
    with pytest.raises(ValueError, match="Missing fitted inference schemas"):
        probe_fitted_preprocessing(artifact, pd.DataFrame({"value": [1.0, 2.0]}))


def test_probe_reports_prediction_skips(tmp_path):
    """Training filters must not be executed by row-preserving inference diagnostics."""
    artifact = _artifact(
        tmp_path, steps=[{"name": "unique", "transformer": "Deduplicate", "params": {}}]
    )
    result = probe_fitted_preprocessing(artifact, pd.DataFrame({"value": [1.0, 1.0, 2.0]}))
    assert result["status"] == "passed"
    assert result["steps"][0]["action"] == "skip_preserve_rows"
    assert result["steps"][0]["status"] == "skipped"
    assert result["steps"][0]["checks"] == []


def test_probe_rejects_nested_mutable_cells(tmp_path):
    """A shallow pandas object copy must never expose caller-owned mutable cells."""
    artifact = _artifact(tmp_path)
    with pytest.raises(ValueError, match="immutable scalar"):
        probe_fitted_preprocessing(artifact, pd.DataFrame({"value": [[1], [2]]}))


def test_probe_can_observe_local_state_outside_worker_validator_subset(tmp_path):
    """Legitimate local defaults must be probed even when remote validation abstains."""
    artifact = _artifact(
        tmp_path,
        steps=[
            {
                "name": "encode",
                "transformer": "OneHotEncoder",
                "params": {"columns": ["value"]},
            }
        ],
    )
    report = probe_fitted_preprocessing(artifact, pd.DataFrame({"value": [0.0, 2.0, 0.0]}))
    assert report["status"] == "passed", report
    assert report["steps"][0]["state_validation"] == "unavailable"
    assert report["steps"][0]["context"] == "unknown"


@pytest.mark.parametrize(
    "axis",
    [
        "index",
        "columns",
        "categories",
        "categorical_index",
        "categorical_columns",
        "multi_categorical_index",
    ],
)
def test_probe_detaches_pandas_metadata_before_calling_custom_code(axis):
    """Pandas deep-copy metadata must not expose caller arrays or hide mutation."""
    from skyulf.inference._probe_frames import ProbeFailure
    from skyulf.inference.preprocessing_probe import _apply_checked

    class Mutator:
        """Represent a custom callback changing pandas metadata in place."""

        def apply(self, frame, state):
            """Modify shared backing storage that DataFrame.copy alone retains."""
            if axis == "categories":
                frame["value"].cat.categories.values[0] = "changed"
            elif axis == "multi_categorical_index":
                frame.index.levels[0].categories.values[0] = 999
            elif axis.startswith("categorical_"):
                getattr(frame, axis.removeprefix("categorical_")).categories.values[0] = (
                    999 if axis == "categorical_index" else "changed"
                )
            else:
                getattr(frame, axis).values[0] = 999 if axis == "index" else "changed"
            return frame

    frame = pd.DataFrame({"value": pd.Categorical(["A", "B"])}, index=[10, 20])
    if axis == "categorical_index":
        frame.index = pd.CategoricalIndex([10, 20])
    elif axis == "categorical_columns":
        frame.columns = pd.CategoricalIndex(["value"])
    elif axis == "multi_categorical_index":
        frame.index = pd.MultiIndex.from_arrays([pd.CategoricalIndex([10, 20])])
    with pytest.raises(ProbeFailure, match="input_mutation"):
        _apply_checked(
            {"applier": Mutator(), "artifact": {}, "name": "mutate", "type": "custom"},
            frame,
            10,
            10000,
        )
    assert frame.index.tolist() == (
        [(10,), (20,)] if axis == "multi_categorical_index" else [10, 20]
    )
    assert frame.columns.tolist() == ["value"]
    assert frame["value"].cat.categories.tolist() == ["A", "B"]


def test_probe_rejects_mutable_numpy_structured_cells():
    """Structured numpy scalars are mutable despite containing no Python objects."""
    from skyulf.inference._probe_frames import validate_frame

    cell = np.array([(1, 2)], dtype=[("a", "i4"), ("b", "i4")])[0]
    frame = pd.DataFrame({"value": pd.Series([cell], dtype=object)})
    with pytest.raises(ValueError, match="immutable scalar"):
        validate_frame(frame, 10, 10000)
    assert cell["a"] == 1


def test_probe_detaches_recipe_before_custom_validation(tmp_path, monkeypatch):
    """An optional validator must not mutate the caller's live recipe configuration."""
    artifact = _artifact(tmp_path)
    recipe = deepcopy(artifact.pipeline.feature_engineer.steps_config)
    owner = NodeRegistry.get_applier("SimpleImputer")

    def mutate(params, state):
        """Expose a custom validator accidentally annotating input configuration."""
        params["visited"] = True
        return params

    monkeypatch.setattr(owner, "resolve_fitted_config", staticmethod(mutate))
    report = probe_fitted_preprocessing(artifact, pd.DataFrame({"value": [0.0, 2.0]}))
    assert report["status"] == "failed"
    assert artifact.pipeline.feature_engineer.steps_config == recipe


@pytest.mark.parametrize("method", ["transform", "_transform_steps"])
def test_probe_rejects_overridden_orchestration(tmp_path, method):
    """Exact class identity alone cannot guarantee the pipeline uses the tested path."""
    artifact = _artifact(tmp_path)
    setattr(artifact.pipeline.feature_engineer, method, lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="Overridden inference methods"):
        probe_fitted_preprocessing(artifact, pd.DataFrame({"value": [0.0, 2.0]}))


def test_shared_local_fallback_never_weakens_partition_admission(tmp_path, monkeypatch):
    """Worker admission must fail if a required node-owned validator disappears."""
    from skyulf.core.capabilities import UnsupportedExecutionError
    from skyulf.inference.partition_safety import require_partition_safe_pipeline

    artifact = _artifact(
        tmp_path,
        steps=[
            {
                "name": "scale",
                "transformer": "StandardScaler",
                "params": {"columns": ["value"], "with_mean": True, "with_std": True},
            }
        ],
    )
    monkeypatch.setattr(NodeRegistry.get_applier("StandardScaler"), "validate_fitted_state", None)
    with pytest.raises(UnsupportedExecutionError, match="Required fitted validation"):
        require_partition_safe_pipeline(artifact)


@pytest.mark.parametrize("context", ["polars", "spark", "frame_spec"])
def test_probe_uses_actual_engine_context_validation(tmp_path, context):
    """A probe must reject contexts that the saved engineer itself cannot execute."""
    from skyulf.core.execution import ExecutionOptions, FrameSpec

    artifact = _artifact(tmp_path)
    engineer = artifact.pipeline.feature_engineer
    if context == "frame_spec":
        engineer.frame_spec = FrameSpec(record_key_columns=("value",))
    else:
        engineer.execution_options = ExecutionOptions(engine=context)
    with pytest.raises(ValueError, match="engine|frame_spec"):
        probe_fitted_preprocessing(artifact, pd.DataFrame({"value": [0.0, 2.0]}))


def test_probe_reports_window_requirement_without_running_independent_chunks(tmp_path, monkeypatch):
    """A saved rolling recipe must require neighboring data before any apply runs."""
    artifact = _artifact(
        tmp_path,
        steps=[
            {
                "name": "rolling",
                "transformer": "RollingAggregate",
                "params": {"columns": ["value"], "window": 2},
            },
            {"name": "later", "transformer": "StandardScaler", "params": {"columns": ["value"]}},
        ],
    )

    def forbidden(*args, **kwargs):
        """Independent requests cannot supply the required temporal context."""
        raise AssertionError("Context-dependent apply must not run")

    monkeypatch.setattr(NodeRegistry.get_applier("RollingAggregate"), "apply", forbidden)
    report = probe_fitted_preprocessing(artifact, pd.DataFrame({"value": [0.0, 2.0]}))
    assert report["status"] == "requires_context"
    assert report["steps"][0]["context"] == "window"
    assert report["steps"][0]["checks"] == []
    assert report["steps"][1]["status"] == "not_run"


@pytest.mark.parametrize("mode", ["valid", "invalid", "mutate"])
def test_probe_inspects_local_state_without_worker_validation(tmp_path, monkeypatch, mode):
    """Local state hooks must run even when a node has no portable worker contract."""
    artifact = _artifact(tmp_path)
    owner = NodeRegistry.get_applier("StandardScaler")
    before = artifact_digest(artifact.pipeline.feature_engineer.fitted_steps)

    def inspect(state):
        """Expose validation and mutation evidence before saved apply executes."""
        if mode == "invalid":
            raise ValueError("Invalid local fitted state")
        if mode == "mutate":
            state["columns"].append("unexpected")
        return state

    monkeypatch.setattr(owner, "validate_fitted_state", None)
    monkeypatch.setattr(owner, "resolve_fitted_config", None)
    monkeypatch.setattr(owner, "validate_inference_state", staticmethod(inspect), raising=False)
    report = probe_fitted_preprocessing(artifact, pd.DataFrame({"value": [0.0, 2.0]}))
    step = report["steps"][1]
    if mode == "valid":
        assert step["state_validation"] == "node_owned"
        assert step["status"] == "passed"
    else:
        assert step["status"] == "failed"
        assert step["reason"] == "invalid_step_contract"
        assert step["checks"] == []
    assert artifact_digest(artifact.pipeline.feature_engineer.fitted_steps) == before
