"""Contract tests for the optional MLflow pyfunc bundle adapter."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import accuracy_score

from skyulf.data.dataset import SplitDataset
from skyulf.inference.bundle import build_bundle, predict_local
from skyulf.integrations.mlflow import tracking
from skyulf.pipeline import SkyulfPipeline

mlflow = pytest.importorskip("mlflow")
from skyulf.integrations.mlflow.model import log_model  # noqa: E402
from skyulf.integrations.mlflow.tracking import TrackingConfig, track_run  # noqa: E402


@pytest.fixture(autouse=True)
def restore_tracking_uri():
    """MLflow configuration from one test must not leak into the next test."""
    previous = mlflow.get_tracking_uri()
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous)


def _regression_pipeline(engine: str) -> tuple[SkyulfPipeline, pd.DataFrame]:
    """Fit the same small regression pipeline through pandas or Polars."""
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({"x": rng.normal(100, 20, 40), "z": rng.normal(5, 2, 40)})
    frame["target"] = 2 * frame.x - 3 * frame.z + 7
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "fill",
                    "transformer": "SimpleImputer",
                    "params": {"columns": ["x", "z"]},
                },
                {
                    "name": "scale",
                    "transformer": "StandardScaler",
                    "params": {"columns": ["x", "z"]},
                },
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    data = pl.from_pandas(frame) if engine == "polars" else frame
    pipeline.fit(SplitDataset(train=data, test=data.head(0)), target_column="target")
    query = pd.DataFrame({"x": [150.0, 80.0], "z": [9.0, 3.0]}, index=[17, 4])
    return pipeline, query


def _classification_pipeline() -> tuple[SkyulfPipeline, pd.DataFrame]:
    """Fit a binary string-label pipeline with a non-default threshold."""
    train = pd.DataFrame(
        {
            "x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            "label": ["no", "no", "no", "yes", "yes", "yes"],
        }
    )
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "logistic_regression"}})
    pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="label")
    query = pd.DataFrame({"x": [-2.5, -0.25, 0.25, 2.5]})
    pipeline.optimize_thresholds(
        query,
        np.array(["no", "yes", "yes", "yes"]),
        accuracy_score,
        grid_points=5,
    )
    return pipeline, query


def _config(tmp_path: Path, name: str) -> TrackingConfig:
    """Create an isolated SQLite tracking configuration for one test."""
    db_path = (tmp_path / f"{name}.db").resolve()
    return TrackingConfig(
        enabled=True,
        tracking_uri=f"sqlite:///{db_path.as_posix()}",
        experiment_name=name,
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("stage", ["raw", "features"])
def test_pyfunc_save_load_matches_bundle(
    engine: str, stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MLflow save/load must preserve raw/features local regression predictions."""
    monkeypatch.chdir(tmp_path)
    pipeline, raw = _regression_pipeline(engine)
    bundle = build_bundle(pipeline, input_stage=stage, feature_order=("x", "z"))
    query = raw if stage == "raw" else pipeline.feature_engineer.transform(raw)
    expected = predict_local(query, bundle)
    config = _config(tmp_path, f"regression-{engine}-{stage}")

    with track_run(config, run_name="package-regression") as run:
        model_uri = log_model(
            bundle,
            run_id=run.run_id,
            artifact_path="skyulf-model",
            tracking_uri=config.tracking_uri,
        )

    mlflow.set_tracking_uri(config.tracking_uri)
    loaded = mlflow.pyfunc.load_model(model_uri)
    actual = loaded.predict(query)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=True)
    assert loaded.metadata.signature is not None
    mlmodel_path = mlflow.artifacts.download_artifacts(
        f"{model_uri}/MLmodel", dst_path=str(tmp_path / "download")
    )
    mlmodel_text = Path(mlmodel_path).read_text(encoding="utf-8")
    assert "uri: bundle" in mlmodel_text
    assert "skyulf-mlflow-" not in mlmodel_text
    assert str(tmp_path) not in mlmodel_text


def test_pyfunc_preserves_class_order_and_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pyfunc classification must retain classes, probabilities, and saved threshold decisions."""
    monkeypatch.chdir(tmp_path)
    pipeline, query = _classification_pipeline()
    bundle = build_bundle(
        pipeline,
        input_stage="raw",
        feature_order=("x",),
        use_tuned_thresholds=True,
    )
    expected = predict_local(query, bundle).reset_index(drop=True)
    config = _config(tmp_path, "classification")

    with track_run(config, run_name="package-classification") as run:
        model_uri = log_model(
            bundle,
            run_id=run.run_id,
            artifact_path="skyulf-model",
            tracking_uri=config.tracking_uri,
        )

    mlflow.set_tracking_uri(config.tracking_uri)
    loaded = mlflow.pyfunc.load_model(model_uri)
    actual = loaded.predict(query).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=True)
    assert list(actual.columns) == ["prediction", *bundle.probability_columns]
    assert bundle.classes == ("no", "yes")
    assert bundle.manifest.thresholds.values != (0.5, 0.5)


def test_log_model_uses_requested_run_with_unrelated_active_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit client upload must leave an unrelated fluent run without model artifacts."""
    monkeypatch.chdir(tmp_path)
    pipeline, query = _regression_pipeline("pandas")
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x", "z"))
    config = _config(tmp_path, "explicit-target")
    caller_config = _config(tmp_path, "caller-store")
    mlflow.set_tracking_uri(caller_config.tracking_uri)
    caller_client = mlflow.MlflowClient(tracking_uri=caller_config.tracking_uri)
    caller_experiment = caller_client.create_experiment("caller")
    client = tracking._make_client(config.tracking_uri)

    with (
        mlflow.start_run(experiment_id=caller_experiment, run_name="caller") as caller,
        track_run(config, run_name="target") as target,
    ):
        model_uri = log_model(
            bundle,
            run_id=target.run_id,
            artifact_path="skyulf-model",
            tracking_uri=config.tracking_uri,
        )
        assert mlflow.active_run().info.run_id == caller.info.run_id
        assert mlflow.get_tracking_uri() == caller_config.tracking_uri
        target_artifacts = {item.path for item in client.list_artifacts(target.run_id)}
        caller_artifacts = {item.path for item in caller_client.list_artifacts(caller.info.run_id)}

    assert model_uri == f"runs:/{target.run_id}/skyulf-model"
    assert "skyulf-model" in target_artifacts
    assert "skyulf-model" not in caller_artifacts
    mlflow.set_tracking_uri(config.tracking_uri)
    pd.testing.assert_frame_equal(
        mlflow.pyfunc.load_model(model_uri).predict(query).reset_index(drop=True),
        predict_local(query, bundle).reset_index(drop=True),
        check_dtype=True,
    )


def test_pyfunc_aligns_by_name_and_rejects_positional_or_missing_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MLflow transport aligns names while the estimator always sees the saved feature order."""
    monkeypatch.chdir(tmp_path)
    pipeline, query = _regression_pipeline("pandas")
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x", "z"))
    config = _config(tmp_path, "order-contract")

    with track_run(config, run_name="package-order") as run:
        model_uri = log_model(
            bundle,
            run_id=run.run_id,
            artifact_path="skyulf-model",
            tracking_uri=config.tracking_uri,
        )

    mlflow.set_tracking_uri(config.tracking_uri)
    loaded = mlflow.pyfunc.load_model(model_uri)
    expected = predict_local(query, bundle)
    pd.testing.assert_frame_equal(loaded.predict(query[["z", "x"]]), expected)
    pd.testing.assert_frame_equal(loaded.predict(query.assign(unused_id=123)), expected)
    pd.testing.assert_frame_equal(loaded.predict(query.astype("int32")), expected)
    for invalid in (query.to_numpy(), query.to_numpy().tolist(), query.drop(columns="x")):
        with pytest.raises(mlflow.exceptions.MlflowException, match="missing inputs"):
            loaded.predict(invalid)
    with pytest.raises(ValueError, match="order"):
        predict_local(query[["z", "x"]], bundle)


def test_log_model_requires_explicit_run_id() -> None:
    """Packaging must associate the model with the requested run instead of an active caller run."""
    pipeline, _ = _regression_pipeline("pandas")
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x", "z"))
    with pytest.raises(ValueError, match="run_id"):
        log_model(bundle, run_id=None, artifact_path="skyulf-model")


@pytest.mark.parametrize("artifact_path", ["model#candidate", "model?candidate"])
def test_log_model_rejects_uri_delimiters(
    artifact_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Artifact paths must remain unambiguous when embedded in a runs URI."""
    pipeline, _ = _regression_pipeline("pandas")
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x", "z"))
    monkeypatch.setattr(
        "skyulf.integrations.mlflow.model._make_client",
        lambda *args, **kwargs: pytest.fail("invalid artifact path contacted MLflow"),
    )
    with pytest.raises(ValueError, match="URI delimiters"):
        log_model(bundle, run_id="run-id", artifact_path=artifact_path)


@pytest.mark.parametrize("dtype", ["int8", "int16", "uint8", "uint16", "uint32", "uint64"])
def test_signature_rejects_dtypes_mlflow_cannot_preserve(
    dtype: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Packaging must fail rather than silently widen unsupported integer inputs."""
    train = pd.DataFrame({"x": np.arange(6, dtype=dtype), "target": [0.0] * 6})
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="target")
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x",))

    def forbidden_client(*args: object, **kwargs: object) -> object:
        raise AssertionError("unsupported dtype must fail before contacting MLflow")

    monkeypatch.setattr("skyulf.integrations.mlflow.model._make_client", forbidden_client)
    with pytest.raises(ValueError, match="preserve this bundle dtype exactly"):
        log_model(bundle, run_id="never-contact-store", artifact_path="model")


@pytest.mark.parametrize("dtype", ["bool", "int32", "int64", "float32", "float64"])
def test_pyfunc_preserves_supported_dtypes(
    dtype: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each supported MLflow scalar type must remain usable after a real save/load."""
    monkeypatch.chdir(tmp_path)
    train = pd.DataFrame({"x": pd.Series([0, 1, 0, 1, 0, 1], dtype=dtype), "y": [0.0, 2.0] * 3})
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="y")
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x",))
    query = train[["x"]]
    config = _config(tmp_path, "types")
    with track_run(config, run_name=dtype) as run:
        uri = log_model(
            bundle, run_id=run.run_id, artifact_path="model", tracking_uri=config.tracking_uri
        )
    mlflow.set_tracking_uri(config.tracking_uri)
    pd.testing.assert_frame_equal(
        mlflow.pyfunc.load_model(uri).predict(query), predict_local(query, bundle)
    )


def test_package_excludes_producer_project_and_records_synthetic_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Packaging must not auto-capture the producer's project files or real input rows."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "producer-private"\n', encoding="utf-8"
    )
    pipeline, _ = _regression_pipeline("pandas")
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x", "z"))
    config = _config(tmp_path, "privacy")
    with track_run(config, run_name="package") as run:
        uri = log_model(
            bundle, run_id=run.run_id, artifact_path="model", tracking_uri=config.tracking_uri
        )
    path = Path(
        mlflow.artifacts.download_artifacts(
            uri, tracking_uri=config.tracking_uri, dst_path=str(tmp_path / "download")
        )
    )
    assert not (path / "uv.lock").exists()
    assert not (path / "pyproject.toml").exists()
    example = json.loads((path / "input_example.json").read_text(encoding="utf-8"))
    assert example["columns"] == ["x", "z"]
    assert example["data"] == [[0.0, 0.0]]
    requirements = (path / "requirements.txt").read_text(encoding="utf-8")
    for package, version in bundle.manifest.requirements:
        if package != "python":
            assert f"{package}=={version}" in requirements
    assert f"mlflow=={mlflow.__version__}" in requirements


@pytest.mark.skipif(
    not os.environ.get("SKYULF_MLFLOW_WHEEL_PYTHON"), reason="Requires wheel-installed Python"
)
def test_wheel_subprocess_loads_bundle_without_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A separate wheel-installed interpreter must load and score the uploaded artifact in isolation."""
    monkeypatch.chdir(tmp_path)
    pipeline, query = _regression_pipeline("polars")
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x", "z"))
    config = _config(tmp_path, "wheel")
    with track_run(config, run_name="package-wheel") as run:
        uri = log_model(
            bundle, run_id=run.run_id, artifact_path="model", tracking_uri=config.tracking_uri
        )
    script = """
import sys
import sysconfig
from pathlib import Path
import mlflow
import pandas as pd
import skyulf
assert Path(skyulf.__file__).resolve().is_relative_to(Path(sysconfig.get_path("purelib")).resolve())
mlflow.set_tracking_uri(sys.argv[1])
query = pd.DataFrame({"x": [150.0, 80.0], "z": [9.0, 3.0]}, index=[17, 4])
mlflow.pyfunc.load_model(sys.argv[2]).predict(query).to_json(sys.argv[3], orient="split")
"""
    output = tmp_path / "predictions.json"
    wheel_python = os.environ.get("SKYULF_MLFLOW_WHEEL_PYTHON")
    assert wheel_python is not None
    tracking_uri = config.tracking_uri
    assert tracking_uri is not None
    completed = subprocess.run(
        [
            wheel_python,
            "-I",
            "-c",
            script,
            tracking_uri,
            uri,
            str(output),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = pd.read_json(output, orient="split", dtype={"prediction": "float64"})
    pd.testing.assert_frame_equal(actual, predict_local(query, bundle))
