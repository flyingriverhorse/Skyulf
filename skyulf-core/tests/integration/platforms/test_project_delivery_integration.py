"""Saved project assets and dependencies survive full pipeline and MLflow delivery."""

import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline
from skyulf.integrations.databricks.local_batch import fit_local_workflow
from skyulf.integrations.databricks.project import load_project_workflow

_FEATURE_SOURCE = '''"""A saved asset and external library define the nonlinear feature."""
import json
import polars as pl
from packaging.version import Version
from skyulf.inference.project_code import custom_step
from skyulf.inference.project_package import read_project_asset
from skyulf.preprocessing.base import BaseCalculator, BaseApplier, fit_method, apply_method

class PowerCalculator(BaseCalculator):
    """Keep this deterministic transform free of learned data."""
    @fit_method
    def fit(self, X, y, config):
        """Return only the configured column identity."""
        return {"column": "x"}

class PowerApplier(BaseApplier):
    """Read the saved asset whenever the fitted transform is applied."""
    @apply_method
    def apply(self, X, y, params):
        """Preserve the native dataframe while deriving the asset-defined power."""
        power = Version(json.loads(read_project_asset(__package__, "power.json"))["power"]).major
        frame = X.to_native() if hasattr(X, "to_native") else X
        column = params["column"]
        if isinstance(frame, pl.DataFrame):
            return frame.with_columns((pl.col(column) ** power).alias(column))
        return frame.assign(**{column: frame[column] ** power})

def build_preprocessing():
    """Register the exact project classes included in the saved source snapshot."""
    return [custom_step("asset_power", PowerCalculator, PowerApplier)]
'''


def _fit_saved_package(tmp_path, engine="pandas"):
    """Fit a nonlinear feature whose replay genuinely needs its asset and dependency."""
    pytest.importorskip("packaging")
    root = tmp_path / "features"
    root.mkdir()
    (root / "__init__.py").write_text(_FEATURE_SOURCE, encoding="utf-8")
    (root / "power.json").write_text('{"power": "2.0"}', encoding="utf-8")
    (root / "assets.json").write_text('["power.json"]', encoding="utf-8")
    pin = f"packaging=={importlib.metadata.version('packaging')}"
    (root / "requirements.txt").write_text(pin, encoding="utf-8")
    config = load_project_workflow(
        {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}, root
    )
    rows = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 9.0, 19.0, 33.0]})
    native = pl.from_pandas(rows) if engine == "polars" else rows
    artifact_path = tmp_path / "local-artifact"
    artifact = fit_local_workflow(
        config["pipeline"],
        SplitDataset(train=native, test=native.head(0)),
        target_column="target",
        artifact_path=artifact_path,
        max_rows=10,
        max_bytes=10000,
    )
    return root, artifact_path, artifact, pin


def _remove_original_package(root):
    """Remove only the known flat test package so replay cannot consult original files."""
    for path in root.iterdir():
        path.unlink()
    root.rmdir()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_saved_pipeline_assets_and_pins_replay_in_fresh_process(tmp_path, engine):
    """Both fit engines must replay project transforms after the original package is gone."""
    root, path, artifact, pin = _fit_saved_package(tmp_path, engine)
    assert artifact.manifest.project_requirements == (pin,)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["project_requirements"] == [pin]
    query = pd.DataFrame({"x": [5.0, 6.0]})
    np.testing.assert_allclose(predict_local_pipeline(query, artifact)["prediction"], [51.0, 73.0])
    _remove_original_package(root)
    code = (
        "import json, sys, pandas as pd\n"
        "from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline\n"
        "artifact = load_local_pipeline(sys.argv[1])\n"
        "result = predict_local_pipeline(pd.DataFrame({'x': [5., 6.]}), artifact)\n"
        "print(json.dumps({'predictions': result['prediction'].tolist(), "
        "'pins': artifact.manifest.project_requirements}))\n"
    )
    child = subprocess.run(
        [sys.executable, "-I", "-c", code, str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr
    delivered = json.loads(child.stdout)
    assert delivered["pins"] == [pin]
    np.testing.assert_allclose(delivered["predictions"], [51.0, 73.0])


@pytest.mark.parametrize("available", ["changed", "missing"])
def test_saved_project_dependency_rejected_before_unpickle(tmp_path, monkeypatch, available):
    """Even a cached project module cannot bypass dependency verification during loading."""
    from skyulf.inference import local_pipeline, project_dependencies

    _, path, _, _ = _fit_saved_package(tmp_path)
    original_version = project_dependencies.version

    def installed_version(name):
        """Simulate only the declared external dependency changing after training."""
        if name == "packaging":
            if available == "missing":
                raise importlib.metadata.PackageNotFoundError(name)
            return "0.0.0"
        return original_version(name)

    def forbid_unpickle(payload):
        """Any pickle decoding would violate the dependency preflight boundary."""
        raise AssertionError("unpickle ran before project dependency verification")

    monkeypatch.setattr(project_dependencies, "version", installed_version)
    monkeypatch.setattr(local_pipeline.pickle, "loads", forbid_unpickle)
    with pytest.raises(ValueError, match="Project dependency packaging"):
        load_local_pipeline(path)


def test_preloaded_mlflow_adapter_unpickles_before_project_context(tmp_path):
    """MLflow may serialize after loading context, but fresh workers need context first."""
    pytest.importorskip("mlflow")
    cloudpickle = pytest.importorskip("cloudpickle")
    from skyulf.integrations.mlflow.local_model import SkyulfLocalPythonModel

    root, path, _, _ = _fit_saved_package(tmp_path)
    model = SkyulfLocalPythonModel()
    model.load_context(SimpleNamespace(artifacts={"local_pipeline": str(path)}))
    model.__dict__["_delivery_metadata"] = {"marker": "preserved"}
    saved = tmp_path / "python_model.pkl"
    saved.write_bytes(cloudpickle.dumps(model))
    assert model._artifact is not None
    _remove_original_package(root)
    code = (
        "import json, sys, cloudpickle, pandas as pd\n"
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "model = cloudpickle.loads(Path(sys.argv[1]).read_bytes())\n"
        "assert model._artifact is None, 'adapter retained a preloaded project artifact'\n"
        "assert model._delivery_metadata == {'marker': 'preserved'}\n"
        "model.load_context(SimpleNamespace(artifacts={'local_pipeline': sys.argv[2]}))\n"
        "result = model.predict(None, pd.DataFrame({'x': [5., 6.]}))\n"
        "print(json.dumps(result['prediction'].tolist()))\n"
    )
    child = subprocess.run(
        [sys.executable, "-I", "-c", code, str(saved), str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr
    np.testing.assert_allclose(json.loads(child.stdout), [51.0, 73.0])


def test_mlflow_delivers_saved_assets_and_exact_external_pins(tmp_path, monkeypatch):
    """A real logged pyfunc must carry dependency pins and replay without original assets."""
    mlflow = pytest.importorskip("mlflow")
    from skyulf.integrations.mlflow.local_model import log_local_model
    from skyulf.integrations.mlflow.tracking import TrackingConfig, track_run

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setenv("TMP", str(tmp_path))
    root, path, _, pin = _fit_saved_package(tmp_path)
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    config = TrackingConfig(enabled=True, tracking_uri=tracking_uri, experiment_name="assets")
    previous = mlflow.get_tracking_uri()
    try:
        with track_run(config, run_name="project-delivery") as run:
            assert run.run_id is not None
            model_uri = log_local_model(
                path, run_id=run.run_id, artifact_path="model", tracking_uri=tracking_uri
            )
        mlflow.set_tracking_uri(tracking_uri)
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri, dst_path=str(tmp_path / "download")
            )
        )
        requirements = (downloaded / "requirements.txt").read_text(encoding="utf-8")
        assert pin in requirements.splitlines()
        assert (downloaded / "MLmodel").is_file()
        _remove_original_package(root)
        code = (
            "import json, sys, mlflow, pandas as pd\n"
            "model = mlflow.pyfunc.load_model(sys.argv[1])\n"
            "result = model.predict(pd.DataFrame({'x': [5., 6.]}))\n"
            "print(json.dumps(result['prediction'].tolist()))\n"
        )
        child = subprocess.run(
            [sys.executable, "-I", "-c", code, str(downloaded)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert child.returncode == 0, child.stderr
        np.testing.assert_allclose(json.loads(child.stdout), [51.0, 73.0])
    finally:
        mlflow.set_tracking_uri(previous)
