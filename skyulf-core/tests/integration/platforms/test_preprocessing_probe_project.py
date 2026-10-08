"""Saved custom preprocessing diagnostics must replay state without granting admission."""

import json
import subprocess
import sys
from copy import deepcopy
from importlib import import_module, util

import pandas as pd
import polars as pl
import pytest

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import load_local_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow
from skyulf.preprocessing.function_steps import FittedFunctionCalculator

SOURCE = '''
import polars as pl
from skyulf.inference.project_code import custom_step
from skyulf.preprocessing import fitted_step
from skyulf.preprocessing.base import apply_method, fit_method

MODE = "__MODE__"
LEARN_CALLS = 0


def learn(df, y):
    """Learn one training mean, recording whether replay accidentally fits again."""
    global LEARN_CALLS
    LEARN_CALLS += 1
    return {"mean": float(df["x"].mean())}


def apply_saved(df, state):
    """Keep training valid and expose the selected defect only on scoring rows."""
    if df["x"].gt(100).any():
        if MODE in ("mutate", "mutate_raise"):
            state["calls"] = state.get("calls", 0) + 1
            if MODE == "mutate_raise":
                raise ValueError(f"SAMPLE_SECRET={df['x'].iloc[0]}")
        if MODE == "batch_mean":
            return df["x"] - df["x"].mean()
        if MODE == "sort":
            return df["x"].sort_values().reset_index(drop=True)
    return df["x"] - state["mean"]


class Calculator:
    """Use the ordinary custom-class fit protocol."""

    @fit_method
    def fit(self, X, y, config):
        """Freeze a constant so the custom fixture needs no alternative fit path."""
        return {"mean": 2.5}


class Applier:
    """Apply saved custom state with optional instance or row-order defects."""

    @apply_method
    def apply(self, X, y, params):
        """Preserve the native engine and mutate only on held-out scoring inputs."""
        native = X.to_native() if hasattr(X, "to_native") else X
        frame = native.to_pandas() if isinstance(native, pl.DataFrame) else native
        result = frame.assign(z=frame["x"] - params["mean"])
        if frame["x"].gt(100).any():
            if MODE == "class_counter":
                self.calls = getattr(self, "calls", 0) + 1
            if MODE == "class_sort":
                result = result.sort_values("x").reset_index(drop=True)
        return pl.from_pandas(result) if isinstance(native, pl.DataFrame) else result


def build_preprocessing(recipe="default"):
    """Expose a single saved step through the actual project factory."""
    if MODE.startswith("class_"):
        return [custom_step("custom", Calculator, Applier)]
    return [fitted_step("custom", learn, apply_saved, output="z")]
'''


def _probe(artifact, sample, **kwargs):
    """Fail explicitly while the diagnostic API is still being implemented."""
    module = "skyulf.inference.preprocessing_probe"
    assert util.find_spec(module) is not None, "Missing fitted preprocessing diagnostic"
    return import_module(module).probe_fitted_preprocessing(artifact, sample, **kwargs)


def _native(frame, engine):
    """Use the artifact's actual fit engine for both replay paths."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _saved_project(tmp_path, mode="fitted", engine="pandas"):
    """Capture relative imports, fit once, and reload through the real artifact loader."""
    root = tmp_path / "features"
    root.mkdir()
    (root / "__init__.py").write_text("from .steps import build_preprocessing\n", encoding="utf-8")
    source = SOURCE.replace("__MODE__", mode) + f"\n# Project: {tmp_path.name}\n"
    (root / "steps.py").write_text(source, encoding="utf-8")
    config = load_project_workflow(
        {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}},
        root,
    )
    rows = _native(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]}), engine
    )
    path = tmp_path / "artifact"
    fit_local_workflow(
        config["pipeline"],
        SplitDataset(train=rows, test=rows[:0]),
        target_column="target",
        artifact_path=path,
        max_rows=10,
        max_bytes=10000,
    )
    return load_local_pipeline(path), root, path


def _sample(engine="pandas"):
    """Use distinct unsorted values so chunking cannot conceal batch/order dependence."""
    return _native(pd.DataFrame({"x": [103.125, 101.25, 104.375, 102.5]}), engine)


def _forbid_fit(*args, **kwargs):
    """Make any accidental learning during the diagnostic an observable failure."""
    raise AssertionError("Diagnostic must not fit")


def _checks(report):
    """Read the custom step's public checks without depending on report internals."""
    step = next(step for step in report["steps"] if step["name"] == "custom")
    return {check["name"]: check for check in step["checks"]}


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_probe_reuses_saved_learned_state_without_fitting(tmp_path, monkeypatch, engine):
    """Diagnostics must use the learned training mean and leave the saved state unchanged."""
    artifact, _, _ = _saved_project(tmp_path, engine=engine)
    record = artifact.pipeline.feature_engineer.fitted_steps[0]
    module = import_module(record["artifact"]["apply"].split(":")[0])
    assert module.LEARN_CALLS == 1
    assert record["artifact"]["state"] == {"mean": 2.5}
    before = deepcopy(record["artifact"])
    monkeypatch.setattr(FittedFunctionCalculator, "fit", _forbid_fit)
    report = _probe(artifact, _sample(engine), chunk_sizes=(1, 2, 3))
    assert report["status"] == "passed", report
    assert report["admission"] == "diagnostic_only"
    assert _checks(report)["chunks:1"]["status"] == "passed"
    assert module.LEARN_CALLS == 1
    assert record["artifact"] == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("mode", ["fitted", "class_safe"])
def test_probe_reloads_captured_project_in_a_fresh_process(tmp_path, engine, mode):
    """Saved function and class packages must be sufficient after editable sources change."""
    artifact, root, path = _saved_project(tmp_path, mode, engine)
    assert artifact.manifest.project_source_sha256
    (root / "steps.py").write_text("raise RuntimeError('live source loaded')\n", encoding="utf-8")
    code = """
import json, sys, pandas as pd, polars as pl
from skyulf.inference.local_pipeline import load_local_pipeline
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
from skyulf.preprocessing.function_steps import FittedFunctionCalculator
def forbidden(*args, **kwargs):
    raise AssertionError("Diagnostic must not fit")
artifact = load_local_pipeline(sys.argv[1])
FittedFunctionCalculator.fit = forbidden
frame = pd.DataFrame({"x": [103.125, 101.25, 104.375, 102.5]})
sample = pl.from_pandas(frame) if sys.argv[2] == "polars" else frame
report = probe_fitted_preprocessing(artifact, sample, chunk_sizes=(1, 2, 3))
print(json.dumps(report))
"""
    loaded = subprocess.run(
        [sys.executable, "-c", code, str(path), engine],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert loaded.returncode == 0, loaded.stderr
    report = json.loads(loaded.stdout)
    assert report["status"] == "passed", report
    assert report["admission"] == "diagnostic_only"
    assert _checks(report)["repeat"]["status"] == "passed"


@pytest.mark.parametrize("mode", ["fitted", "class_safe"])
def test_passing_unknown_custom_probe_does_not_grant_partition_admission(tmp_path, mode):
    """Finite sample parity must never authorize an unreviewed custom implementation."""
    artifact, _, _ = _saved_project(tmp_path, mode)
    report = _probe(artifact, _sample())
    assert report["status"] == "passed", report
    assert report["steps"][0]["context"] == "unknown"
    assert report["admission"] == "diagnostic_only"
    with pytest.raises(UnsupportedExecutionError):
        require_partition_safe_pipeline(artifact)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_probe_detects_batch_statistics_recomputed_during_apply(tmp_path, engine):
    """A callback using the scoring batch mean must fail singleton parity."""
    artifact, _, _ = _saved_project(tmp_path, "batch_mean", engine)
    report = _probe(artifact, _sample(engine))
    assert report["status"] == "failed"
    assert _checks(report)["chunks:1"]["status"] == "failed"
    assert _checks(report)["chunks:1"]["reason"] == "output_mismatch"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("mode", ["mutate", "mutate_raise"])
def test_probe_reports_state_mutation_even_when_callback_raises(tmp_path, engine, mode):
    """Callback exceptions must not hide mutation or leak sample values into evidence."""
    artifact, _, _ = _saved_project(tmp_path, mode, engine)
    record = artifact.pipeline.feature_engineer.fitted_steps[0]
    before = deepcopy(record["artifact"])
    report = _probe(artifact, _sample(engine))
    encoded = json.dumps(report)
    assert report["status"] == "failed"
    assert _checks(report)["full"]["reason"] == "state_mutation"
    assert "103.125" not in encoded
    assert "SAMPLE_SECRET" not in encoded
    assert record["artifact"] == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("mode", ["sort", "class_sort"])
def test_probe_detects_same_length_sorting_with_reset_index(tmp_path, engine, mode):
    """Preserving row count and resetting labels must not disguise a changed row order."""
    artifact, _, _ = _saved_project(tmp_path, mode, engine)
    report = _probe(artifact, _sample(engine))
    assert report["status"] == "failed"
    assert any(check["status"] == "failed" for check in _checks(report).values())


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_probe_detects_custom_applier_counter_mutation(tmp_path, engine):
    """Class instance state must be checked in addition to the learned artifact mapping."""
    artifact, _, _ = _saved_project(tmp_path, "class_counter", engine)
    applier = artifact.pipeline.feature_engineer.fitted_steps[0]["applier"]
    assert vars(applier) == {}
    report = _probe(artifact, _sample(engine))
    assert report["status"] == "failed"
    assert _checks(report)["full"]["reason"] == "state_mutation"
    assert vars(applier) == {}


@pytest.mark.parametrize("budget", [{"max_rows": 3}, {"max_bytes": 1}])
def test_probe_rejects_excess_sample_before_custom_apply(tmp_path, monkeypatch, budget):
    """A bounded diagnostic must reject excessive input before invoking user code."""
    artifact, _, _ = _saved_project(tmp_path, "class_safe")
    applier = artifact.pipeline.feature_engineer.fitted_steps[0]["applier"]
    monkeypatch.setattr(type(applier), "apply", _forbid_fit)
    with pytest.raises(ValueError, match="(?i)(row|byte|budget|limit)"):
        _probe(artifact, _sample(), **budget)
