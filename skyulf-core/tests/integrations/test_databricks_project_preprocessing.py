"""Project Python preprocessing must survive fitting, CV and isolated artifact loading."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest

SOURCE = """
import pandas as pd
import polars as pl
from skyulf.preprocessing.base import BaseCalculator, BaseApplier, fit_method, apply_method
from skyulf.inference.project_code import custom_step

class CenterApplier(BaseApplier):
    @apply_method
    def apply(self, X, y, params):
        frame = X.to_native() if hasattr(X, "to_native") else X
        if isinstance(frame, pl.DataFrame):
            return frame.with_columns((pl.col("x") - params["mean"]).alias("x"))
        return frame.assign(x=frame["x"] - params["mean"])

class CenterCalculator(BaseCalculator):
    @fit_method
    def fit(self, X, y, config):
        frame = X.to_native() if hasattr(X, "to_native") else X
        return {"mean": float(frame["x"].mean())}

def build_preprocessing():
    return [
        {"name": "impute", "transformer": "SimpleImputer",
         "params": {"columns": ["x"], "strategy": "mean"}},
        custom_step("center", CenterCalculator, CenterApplier),
    ]
"""


def _project(tmp_path):
    """Resolve the editable Python source into a normal Core pipeline config."""
    from skyulf.integrations.databricks.project import load_project_workflow

    source = tmp_path / "preprocessing.py"
    source.write_text(SOURCE, encoding="utf-8")
    config = {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}
    return load_project_workflow(config, source), source


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_project_code_and_fitted_state_survive_without_the_project_file(tmp_path, engine):
    """Inference must use saved training code/state even after the project file changes."""
    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.local_pipeline import predict_local_pipeline
    from skyulf.integrations.databricks.local_batch import fit_local_workflow

    config, source = _project(tmp_path)
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    artifact = fit_local_workflow(
        config["pipeline"],
        SplitDataset(train=frame, test=frame[:0]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=10,
        max_bytes=10000,
    )
    assert artifact.pipeline.feature_engineer.fitted_steps[1]["artifact"]["mean"] == 2.5
    assert artifact.manifest.project_source_sha256
    expected = predict_local_pipeline(pd.DataFrame({"x": [5.0, 6.0]}), artifact)["prediction"]
    source.write_text(
        "raise RuntimeError('edited project must not run during inference')", encoding="utf-8"
    )
    code = """
import json, sys, pandas as pd
from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline
artifact = load_local_pipeline(sys.argv[1])
print(json.dumps(predict_local_pipeline(pd.DataFrame({"x": [5., 6.]}), artifact)["prediction"].tolist()))
"""
    loaded = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "artifact")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert loaded.returncode == 0, loaded.stderr
    np.testing.assert_allclose(json.loads(loaded.stdout), expected)


def test_project_rejects_two_preprocessing_sources(tmp_path):
    """A JSON chain cannot be silently replaced by the Python chain."""
    from skyulf.integrations.databricks.project import load_project_workflow

    source = tmp_path / "preprocessing.py"
    source.write_text(SOURCE, encoding="utf-8")
    with pytest.raises(ValueError, match="preprocessing"):
        load_project_workflow(
            {"pipeline": {"preprocessing": [{"transformer": "StandardScaler"}]}}, source
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_project_custom_fit_is_repeated_inside_cv_folds(tmp_path, monkeypatch, engine):
    """Custom learned preprocessing must never reuse a full-data mean inside CV."""
    from skyulf.integrations.databricks.local_cv import LocalCVSpec, evaluate_training_cv
    from skyulf.preprocessing.base import BaseCalculator
    from skyulf.registry import NodeRegistry

    config, _ = _project(tmp_path)
    calculator = NodeRegistry.get_calculator(config["pipeline"]["preprocessing"][1]["transformer"])
    assert issubclass(calculator, BaseCalculator)
    original = calculator.fit
    observed = []

    def record_fit(self, frame, params):
        """Record actual fitted statistics without changing the transformation."""
        state = original(self, frame, params)
        observed.append(state["mean"])
        return state

    monkeypatch.setattr(calculator, "fit", record_fit)
    frame = pd.DataFrame(
        {"x": np.arange(12, dtype=float), "target": np.arange(12, dtype=float) * 2}
    )
    if engine == "polars":
        frame = pl.from_pandas(frame)
    result = evaluate_training_cv(
        frame,
        config["pipeline"],
        LocalCVSpec(enabled=True, folds=3, shuffle=False),
        target_column="target",
    )
    assert result is not None
    assert sorted(observed) == [3.5, 5.5, 7.5]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_mlflow_custom_code_loads_in_a_fresh_process(tmp_path, monkeypatch, engine):
    """The complete MLflow package must predict without access to the project source."""
    mlflow = pytest.importorskip("mlflow")
    from skyulf.data.dataset import SplitDataset
    from skyulf.integrations.databricks.local_batch import fit_local_workflow
    from skyulf.integrations.mlflow.local_model import log_local_model
    from skyulf.integrations.mlflow.tracking import TrackingConfig, track_run

    monkeypatch.chdir(tmp_path)
    config, source = _project(tmp_path)
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    fit_local_workflow(
        config["pipeline"],
        SplitDataset(train=frame, test=frame[:0]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=10,
        max_bytes=10000,
    )
    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    with track_run(
        TrackingConfig(enabled=True, tracking_uri=uri, experiment_name="project"), run_name="custom"
    ) as run:
        assert run.run_id is not None
        model_uri = log_local_model(
            tmp_path / "artifact", run_id=run.run_id, artifact_path="model", tracking_uri=uri
        )
    downloaded = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, tracking_uri=uri)
    source.write_text("raise RuntimeError('project unavailable')", encoding="utf-8")
    code = """
import json, sys, mlflow, pandas as pd
model = mlflow.pyfunc.load_model(sys.argv[1])
print(json.dumps(model.predict(pd.DataFrame({"x": [5.,6.]}))["prediction"].tolist()))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, downloaded],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(json.loads(result.stdout), [11.0, 13.0])


def test_changed_packaged_source_is_rejected_before_import(tmp_path):
    """A damaged code snapshot cannot execute before its checksum is checked."""
    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.local_pipeline import load_local_pipeline
    from skyulf.integrations.databricks.local_batch import fit_local_workflow

    config, _ = _project(tmp_path)
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "target": [3.0, 5.0, 7.0]})
    path = tmp_path / "artifact"
    fit_local_workflow(
        config["pipeline"],
        SplitDataset(train=frame, test=frame[:0]),
        target_column="target",
        artifact_path=path,
        max_rows=10,
        max_bytes=10000,
    )
    (path / "preprocessing.py").write_text(
        "raise RuntimeError('must not execute')", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source checksum"):
        load_local_pipeline(path)


def test_two_saved_code_versions_keep_their_own_transformations(tmp_path):
    """Loading another model's same-named classes cannot replace the first model's code."""
    from skyulf.data.dataset import SplitDataset
    from skyulf.integrations.databricks.local_batch import fit_local_workflow
    from skyulf.integrations.databricks.project import load_project_workflow

    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    source = tmp_path / "preprocessing.py"
    variants = [
        SOURCE,
        SOURCE.replace(
            'return frame.assign(x=frame["x"] - params["mean"])',
            'return frame.assign(x=(frame["x"] - params["mean"]) * 2)',
        ),
    ]
    for version, code in enumerate(variants):
        source.write_text(code, encoding="utf-8")
        config = load_project_workflow(
            {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}},
            source,
        )
        fit_local_workflow(
            config["pipeline"],
            SplitDataset(train=frame, test=frame[:0]),
            target_column="target",
            artifact_path=tmp_path / f"v{version}",
            max_rows=10,
            max_bytes=10000,
        )
    code = """
import json, sys, pandas as pd
from skyulf.inference.local_pipeline import load_local_pipeline
first = load_local_pipeline(sys.argv[1])
second = load_local_pipeline(sys.argv[2])
print(json.dumps([
    float(model.pipeline.feature_engineer.transform(pd.DataFrame({"x": [3.]}))["x"].iloc[0])
    for model in (first, second, first)
]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "v0"), str(tmp_path / "v1")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [0.5, 1.0, 0.5]


@pytest.mark.parametrize(
    "action", ["train", "train_monthly", "approve", "reject", "rollback", "score"]
)
def test_notebook_uses_editable_code_only_for_training(
    tmp_path, monkeypatch, workflow_config, action
):
    """Editing the project must affect new training, never existing-model operator actions."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from skyulf.integrations.databricks import job_runtime

    path = tmp_path / "workflow.json"
    workflow_config["pipeline"]["preprocessing"] = []
    path.write_text(json.dumps(workflow_config), encoding="utf-8")
    source = tmp_path / "preprocessing.py"
    source.write_text(
        SOURCE
        if action.startswith("train")
        else "raise RuntimeError('must not read current file')",
        encoding="utf-8",
    )
    values = {
        "config_path": str(path),
        "workflow_contract": "1",
        "deployed_score_handoff": "after_alias_change",
        "lifecycle_action": action,
        "catalog": "workspace",
        "input_schema": "test",
        "output_schema": "test",
        "metadata_schema": "test",
        "resource_suffix": "",
    }
    monkeypatch.setattr(job_runtime, "resolve_target_config", lambda config, bindings: config)
    execute = Mock(return_value=job_runtime.BundleActionResult(action, {}, False, {}))
    monkeypatch.setattr(job_runtime, "run_bundle_action", execute)
    dbutils = SimpleNamespace(
        widgets=SimpleNamespace(getAll=lambda: values), notebook=Mock(), jobs=Mock()
    )
    job_runtime.run_notebook(
        None,
        dbutils,
        task_role="score" if action == "score" else "lifecycle",
        preprocessing_path="preprocessing.py",
        exit_notebook=False,
    )
    used = execute.call_args.args[1]["pipeline"]
    if action.startswith("train"):
        assert len(used["preprocessing"]) == 2 and used["project_python_source"]
    else:
        assert used["preprocessing"] == [] and "project_python_source" not in used
