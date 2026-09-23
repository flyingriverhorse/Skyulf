"""MLflow transport tests for whole-frame local pandas/Polars pipelines."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import accuracy_score

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import (
    load_local_pipeline,
    predict_local_pipeline,
    save_local_pipeline,
)
from skyulf.pipeline import SkyulfPipeline

mlflow = pytest.importorskip("mlflow")
from skyulf.integrations.mlflow.local_model import log_local_model  # noqa: E402
from skyulf.integrations.mlflow.registry import register_model, resolve_model  # noqa: E402
from skyulf.integrations.mlflow.tracking import TrackingConfig, track_run  # noqa: E402


@pytest.fixture(autouse=True)
def restore_tracking_uri():
    """Each pyfunc test must leave the caller's MLflow tracking URI unchanged."""
    previous = mlflow.get_tracking_uri()
    try:
        yield
    finally:
        mlflow.set_tracking_uri(previous)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_local_pyfunc_matches_saved_categorical_pipeline(
    engine: str, tmp_path, monkeypatch
) -> None:
    """MLflow must restore fitted categorical FE and the recorded engine without refitting."""
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame(
        {
            "city": ["Riga", "Vilnius", "Riga", "Tallinn", "Vilnius", "Tallinn"],
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": [11.0, 24.0, 13.0, 36.0, 27.0, 38.0],
        }
    )
    native = pl.from_pandas(frame) if engine == "polars" else frame
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "city",
                    "transformer": "OneHotEncoder",
                    "params": {
                        "columns": ["city"],
                        "drop_original": True,
                        "handle_unknown": "ignore",
                    },
                }
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    if isinstance(native, pl.DataFrame):
        training, heldout = native.slice(0, 5), native.slice(5, 1)
    else:
        training, heldout = native.iloc[:5], native.iloc[5:]
    pipeline.fit(SplitDataset(train=training, test=heldout), target_column="target")
    query = pd.DataFrame({"city": ["Riga", None], "amount": [8.0, 9.0]})
    expected = np.asarray(pipeline.predict(pl.from_pandas(query) if engine == "polars" else query))
    artifact_path = tmp_path / "local"
    save_local_pipeline(pipeline, artifact_path)
    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    config = TrackingConfig(enabled=True, tracking_uri=uri, experiment_name=f"local-{engine}")
    assert config.tracking_uri is not None

    with track_run(config, run_name="local-package") as run:
        assert run.run_id is not None
        model_uri = log_local_model(
            artifact_path,
            run_id=run.run_id,
            artifact_path="model",
            tracking_uri=config.tracking_uri,
        )

    mlflow.set_tracking_uri(config.tracking_uri)
    loaded = mlflow.pyfunc.load_model(model_uri)
    result = loaded.predict(query)
    saved = predict_local_pipeline(query, load_local_pipeline(artifact_path))

    np.testing.assert_allclose(result["prediction"], expected, rtol=0, atol=1e-10)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), saved.reset_index(drop=True))
    assert loaded.metadata.signature is not None
    assert loaded.metadata.metadata["skyulf_fitted_engine"] == engine
    mlmodel_file = mlflow.artifacts.download_artifacts(
        f"{model_uri}/MLmodel", dst_path=str(tmp_path / "download")
    )
    mlmodel_text = Path(mlmodel_file).read_text(encoding="utf-8")
    assert "uri: local_pipeline" in mlmodel_text
    assert "skyulf-local-mlflow-" not in mlmodel_text

    registered = register_model(
        model_uri,
        f"local_{engine}",
        tracking_uri=config.tracking_uri,
        registry_uri=config.tracking_uri,
    )
    resolved = resolve_model(
        f"local_{engine}",
        version=registered.version,
        tracking_uri=config.tracking_uri,
        registry_uri=config.tracking_uri,
    )
    assert resolved.model_uri == f"models:/local_{engine}/{registered.version}"
    assert resolved.digest == load_local_pipeline(artifact_path).manifest.pipeline_sha256

    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import json, mlflow, pandas as pd, sys; "
            "mlflow.set_tracking_uri(sys.argv[1]); "
            "frame = pd.DataFrame({'city': ['Riga', None], 'amount': [8.0, 9.0]}); "
            "print(json.dumps(mlflow.pyfunc.load_model(sys.argv[2]).predict(frame).to_dict('list')))",
            config.tracking_uri,
            model_uri,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    np.testing.assert_allclose(json.loads(child.stdout)["prediction"], expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_local_pyfunc_preserves_tuned_classification(engine: str, tmp_path, monkeypatch) -> None:
    """Class probabilities and saved threshold decisions survive MLflow loading."""
    monkeypatch.chdir(tmp_path)
    train = pd.DataFrame(
        {
            "x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            "label": ["no", "no", "no", "yes", "yes", "yes"],
        }
    )
    native = pl.from_pandas(train) if engine == "polars" else train
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "logistic_regression"}})
    pipeline.fit(SplitDataset(train=native, test=native.head(0)), target_column="label")
    query = pd.DataFrame({"x": [-2.5, -0.25, 0.25, 2.5]})
    native_query = pl.from_pandas(query) if engine == "polars" else query
    pipeline.optimize_thresholds(
        native_query, np.array(["no", "yes", "yes", "yes"]), accuracy_score, grid_points=5
    )
    artifact_path = tmp_path / "local"
    save_local_pipeline(pipeline, artifact_path, use_tuned_thresholds=True)
    expected = predict_local_pipeline(query, load_local_pipeline(artifact_path))
    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    config = TrackingConfig(enabled=True, tracking_uri=uri, experiment_name=f"classes-{engine}")
    assert config.tracking_uri is not None

    with track_run(config, run_name="class-package") as run:
        assert run.run_id is not None
        model_uri = log_local_model(
            artifact_path,
            run_id=run.run_id,
            artifact_path="model",
            tracking_uri=config.tracking_uri,
        )

    mlflow.set_tracking_uri(config.tracking_uri)
    actual = mlflow.pyfunc.load_model(model_uri).predict(query)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.columns.tolist() == ["prediction", "probability_0", "probability_1"]
