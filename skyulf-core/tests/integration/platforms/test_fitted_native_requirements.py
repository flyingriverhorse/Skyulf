"""Native fitted dependencies must accompany artifacts and fail before unpickling."""

import builtins
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier

from skyulf.inference import fitted_pipeline, project_dependencies
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.pipeline import SkyulfPipeline


@pytest.fixture
def frame():
    """Use deterministic real fits without external data or network access."""
    values = np.arange(80, dtype=float)
    return pd.DataFrame({"x": values, "y": values % 7, "target": (values > 39).astype(int)})


def _fit(frame, package=None):
    """Fit the actual Core calculator before persisting its native estimator."""
    if package is not None:
        pytest.importorskip(package)
    model_type = {
        None: "logistic_regression",
        "xgboost": "xgboost_classifier",
        "lightgbm": "lgbm_classifier",
    }[package]
    params = {"n_estimators": 3, "n_jobs": 1, "random_state": 7} if package else {"max_iter": 200}
    if package == "lightgbm":
        params.update(verbosity=-1, min_child_samples=2)
    pipeline = SkyulfPipeline(
        {"preprocessing": [], "modeling": {"type": model_type, "params": params}}
    )
    pipeline.fit(frame, target_column="target")
    assert pipeline.is_fitted()
    return pipeline


@pytest.mark.parametrize("package", ["xgboost", "lightgbm"])
def test_native_artifact_records_exact_pin_and_replays(package, frame, tmp_path):
    """A source-free fitted native model carries its actual installed version on save/load."""
    pipeline = _fit(frame, package)
    path = tmp_path / "native"
    save_pipeline(pipeline, path)
    restored = load_pipeline(path)
    assert restored.manifest.project_requirements == (f"{package}=={version(package)}",)
    assert restored.manifest.project_source_sha256 is None
    pd.testing.assert_frame_equal(
        predict_pipeline(frame[["x", "y"]], restored),
        predict_pipeline(
            frame[["x", "y"]], fitted_pipeline.FittedPipelineArtifact(restored.manifest, pipeline)
        ),
        check_exact=True,
    )


@pytest.mark.parametrize("package", ["xgboost", "lightgbm"])
def test_actual_mlflow_package_contains_native_pin(package, frame, tmp_path, monkeypatch):
    """Real MLflow packaging must install the native dependency without a template project."""
    mlflow = pytest.importorskip("mlflow")
    from skyulf.integrations.mlflow.models.pipeline_model import log_pipeline_model

    monkeypatch.chdir(tmp_path)
    pipeline = _fit(frame, package)
    path = tmp_path / "native"
    save_pipeline(pipeline, path)
    previous = mlflow.get_tracking_uri()
    tracking = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    try:
        mlflow.set_tracking_uri(tracking)
        experiment = mlflow.create_experiment(
            "native-pins", artifact_location=(tmp_path / "runs").as_uri()
        )
        with mlflow.start_run(experiment_id=experiment) as run:
            uri = log_pipeline_model(
                path, run_id=run.info.run_id, artifact_path="model", tracking_uri=tracking
            )
        downloaded = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
        assert (
            f"{package}=={version(package)}"
            in (downloaded / "requirements.txt").read_text().splitlines()
        )
        prediction = mlflow.pyfunc.load_model(uri).predict(frame[["x", "y"]])
        pd.testing.assert_frame_equal(
            prediction, predict_pipeline(frame[["x", "y"]], load_pipeline(path)), check_exact=True
        )
    finally:
        mlflow.set_tracking_uri(previous)


@pytest.mark.parametrize("wrapper", ["voting", "stacking", "calibrated"])
@pytest.mark.parametrize("package", ["xgboost", "lightgbm"])
def test_nested_native_members_are_pinned(wrapper, package, frame, tmp_path):
    """Fitted sklearn wrappers cannot hide their retained native base estimators from packaging."""
    pipeline = _fit(frame, package)
    base = pipeline.model_estimator.model
    if wrapper == "calibrated":
        model = CalibratedClassifierCV(estimator=base, cv=2, n_jobs=1)
    else:
        estimators = [("native", base), ("tree", DecisionTreeClassifier(max_depth=2))]
        model = (
            VotingClassifier(estimators, voting="soft", n_jobs=1)
            if wrapper == "voting"
            else StackingClassifier(
                estimators, final_estimator=DecisionTreeClassifier(max_depth=2), cv=2, n_jobs=1
            )
        )
    model.fit(frame[["x", "y"]], frame.target)
    pipeline.model_estimator.model = model
    path = tmp_path / "nested"
    save_pipeline(pipeline, path)
    restored = load_pipeline(path)
    assert restored.manifest.project_requirements == (f"{package}=={version(package)}",)
    assert restored.pipeline.model_estimator is not None
    assert restored.pipeline.model_estimator.model is not None
    np.testing.assert_array_equal(
        restored.pipeline.model_estimator.model.predict(frame[["x", "y"]]),
        model.predict(frame[["x", "y"]]),
    )


@pytest.mark.parametrize("package", ["xgboost", "lightgbm"])
@pytest.mark.parametrize("installed", [None, "0.0.0"])
def test_native_runtime_error_precedes_payload_read(
    package, installed, frame, tmp_path, monkeypatch
):
    """Missing or mismatched native libraries must fail clearly before unpickle or certificate checks."""
    path = tmp_path / "native"
    save_pipeline(_fit(frame, package), path)
    (path / "pipeline.pkl").unlink()
    original = project_dependencies.version

    def runtime_version(name):
        """Emulate only the worker's distribution metadata, retaining the real saved artifact."""
        if name != package:
            return original(name)
        if installed is None:
            raise PackageNotFoundError(name)
        return installed

    monkeypatch.setattr(project_dependencies, "version", runtime_version)
    message = f"Project dependency {package}.*" + (
        "not installed" if installed is None else "found 0.0.0"
    )
    with pytest.raises(ValueError, match=message):
        load_pipeline(path)


@pytest.mark.parametrize("package", ["xgboost", "lightgbm"])
def test_conflicting_project_native_pin_is_rejected(package, frame, tmp_path):
    """Custom source cannot override the actual native library used to fit the retained model."""
    pipeline = _fit(frame, package)
    pipeline.config["project_python_source"] = (
        f"install_project_package(requirements=('{package}==0.0.0',))"
    )
    with pytest.raises(ValueError, match="Duplicate project dependency"):
        save_pipeline(pipeline, tmp_path / "conflict")


@pytest.mark.parametrize("package", ["xgboost", "lightgbm"])
def test_matching_project_native_pin_is_deduplicated(package, frame):
    """A template's matching explicit pin must coexist with automatic fitted-state metadata."""
    pipeline = _fit(frame, package)
    pin = f"{package}=={version(package)}"
    pipeline.config["project_python_source"] = f"install_project_package(requirements=({pin!r},))"
    assert fitted_pipeline._project_contract(pipeline) == (pin,)


@pytest.mark.parametrize("package", ["xgboost", "lightgbm"])
def test_retained_selector_state_identifies_native_estimator(package, frame):
    """A real fitted selector retains native code even when wrapped inside preprocessing state."""
    from skyulf.inference._model_requirements import native_model_requirements

    native = _fit(frame, package).model_estimator.model
    selector = SelectFromModel(native).fit(frame[["x", "y"]], frame.target)
    retained_state = {"selector": selector}
    # Core ModelBasedSelection currently discards this selector; only retained objects need pins.
    assert native_model_requirements(retained_state) == (f"{package}=={version(package)}",)


def test_mixed_native_ensemble_collects_both_packages(frame, tmp_path):
    """One fitted ensemble must package every native library represented by its members."""
    xgb = _fit(frame, "xgboost").model_estimator.model
    lgb = _fit(frame, "lightgbm").model_estimator.model
    pipeline = _fit(frame)
    pipeline.model_estimator.model = VotingClassifier(
        [("xgb", xgb), ("lgb", lgb)], voting="soft", n_jobs=1
    ).fit(frame[["x", "y"]], frame.target)
    path = tmp_path / "mixed"
    save_pipeline(pipeline, path)
    assert load_pipeline(path).manifest.project_requirements == (
        f"lightgbm=={version('lightgbm')}",
        f"xgboost=={version('xgboost')}",
    )


def test_plain_sklearn_does_not_import_or_pin_native_libraries(frame, tmp_path, monkeypatch):
    """Installed but unused native libraries must not become mandatory serving dependencies."""
    pipeline = _fit(frame)
    original = builtins.__import__

    def import_without_native(name, *args, **kwargs):
        """Reject any optional import attempted while saving and restoring a sklearn artifact."""
        if name.split(".")[0] in {"xgboost", "lightgbm"}:
            raise AssertionError(f"Unused native import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_native)
    path = tmp_path / "plain"
    save_pipeline(pipeline, path)
    assert load_pipeline(path).manifest.project_requirements == ()
    assert json.loads((path / "manifest.json").read_text())["project_requirements"] == []


def test_same_python_minor_accepts_patch_difference(frame, tmp_path, monkeypatch):
    """Manifest reconstruction must honor the runtime guard's Python major/minor contract."""
    path = tmp_path / "python-patch"
    save_pipeline(_fit(frame), path)
    before = load_pipeline(path)
    original = fitted_pipeline.runtime_requirements()
    current = dict(original)["python"].split(".")
    different_patch = ".".join([*current[:2], str(int(current[2]) + 1)])
    monkeypatch.setattr(
        fitted_pipeline,
        "runtime_requirements",
        lambda: tuple(
            (name, different_patch if name == "python" else value) for name, value in original
        ),
    )
    restored = load_pipeline(path)
    assert restored.manifest == before.manifest
    pd.testing.assert_frame_equal(
        predict_pipeline(frame[["x", "y"]], restored),
        predict_pipeline(frame[["x", "y"]], before),
        check_exact=True,
    )


def test_different_python_minor_fails_before_payload_read(frame, tmp_path, monkeypatch):
    """Accepting patch differences must not relax Python major/minor compatibility."""
    path = tmp_path / "python-minor"
    save_pipeline(_fit(frame), path)
    (path / "pipeline.pkl").unlink()
    original = fitted_pipeline.runtime_requirements()
    current = dict(original)["python"].split(".")
    different_minor = f"{current[0]}.{int(current[1]) + 1}.0"
    monkeypatch.setattr(
        fitted_pipeline,
        "runtime_requirements",
        lambda: tuple(
            (name, different_minor if name == "python" else value) for name, value in original
        ),
    )
    with pytest.raises(fitted_pipeline.ArtifactRuntimeMismatchError, match="mismatch for python"):
        load_pipeline(path)


@pytest.mark.parametrize("change", ["duplicate", "reorder"])
def test_runtime_pin_shape_is_not_relaxed(frame, tmp_path, change):
    """Preserving validated Python patch metadata must not accept noncanonical runtime tuples."""
    path = tmp_path / "runtime-shape"
    save_pipeline(_fit(frame), path)
    manifest_path = path / "manifest.json"
    document = json.loads(manifest_path.read_text())
    pins = document["requirements"]
    document["requirements"] = [*pins, pins[-1]] if change == "duplicate" else pins[::-1]
    manifest_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="runtime requirements"):
        load_pipeline(path)
