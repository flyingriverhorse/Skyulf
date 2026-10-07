"""Transported temporal models must keep causal batch continuity and receipt identity."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import (
    load_local_pipeline,
    predict_local_pipeline,
    save_local_pipeline,
)
from skyulf.integrations.databricks.scoring.incremental.local_history import (
    bind_period_history,
    history_receipt,
    incremental_history,
)
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.time_series.history import TemporalHistorySession


def fitted_temporal_pipeline(engine="pandas"):
    """Fit on observed features with an isolated later holdout and explicit clock removal."""
    frame = pd.DataFrame({"t": np.arange(20, dtype=np.int64), "v": np.arange(20, dtype=float)})
    frame["target"] = frame.v.rolling(3, min_periods=1).mean()
    if engine == "polars":
        frame = pl.from_pandas(frame)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "rolling",
                    "transformer": "RollingAggregate",
                    "params": {
                        "columns": ["v"],
                        "window": 3,
                        "sort_by": "t",
                        "history_mode": "carry",
                    },
                },
                {
                    "name": "drop_clock",
                    "transformer": "DropMissingColumns",
                    "params": {"columns": ["t", "v"], "missing_threshold": None},
                },
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=frame[:16], test=frame[16:]), target_column="target")
    return pipeline


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_artifact_and_databricks_receipt_continue_real_predictions(tmp_path, engine):
    """A transported artifact and committed receipt reproduce uninterrupted predictions."""
    pipeline = fitted_temporal_pipeline(engine)
    save_local_pipeline(pipeline, tmp_path / "model")
    artifact = load_local_pipeline(tmp_path / "model")
    prepared = SimpleNamespace(artifact=artifact)
    rows = pd.DataFrame(
        {"t": np.arange(20, 24, dtype=np.int64), "v": np.arange(20, 24, dtype=float)}
    )
    expected = predict_local_pipeline(rows, artifact)
    state = None
    outputs = []
    for batch in (rows[:2], rows[2:]):
        with TemporalHistorySession(artifact.manifest.pipeline_sha256, state) as session:
            outputs.append(predict_local_pipeline(batch, artifact))
        committed = history_receipt(session)
        state = json.loads(json.dumps(committed["temporal_history"]))
        assert incremental_history(prepared, committed).previous == state["steps"]
    np.testing.assert_allclose(pd.concat(outputs)["prediction"], expected["prediction"])
    assert (
        artifact.pipeline.feature_engineer.fitted_steps[0]["artifact"]["history_seed"][-1]["t"]
        == 15
    )


def test_bootstrap_reconstructs_history_and_period_identity_binds_context(tmp_path):
    """A complete source snapshot must not double-count the training seed on first publish."""
    save_local_pipeline(fitted_temporal_pipeline(), tmp_path / "model")
    artifact = load_local_pipeline(tmp_path / "model")
    prepared = SimpleNamespace(artifact=artifact)
    with incremental_history(prepared, None) as session:
        result = predict_local_pipeline(
            pd.DataFrame({"t": [0, 1, 2], "v": [0.0, 1.0, 2.0]}), artifact
        )
    assert len(result) == 3
    receipt = bind_period_history({"request_digest": "original"}, session, None)
    altered = bind_period_history({"request_digest": "original"}, session, {"different": True})
    assert receipt["request_digest"] != altered["request_digest"]
    assert receipt["temporal_history"]["steps"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_mlflow_roundtrip_keeps_temporal_seed_and_continuation(tmp_path, monkeypatch, engine):
    """Real MLflow logging/loading must transport the seed and allow explicit later batches."""
    mlflow = pytest.importorskip("mlflow")
    import tempfile

    from skyulf.integrations.mlflow.models.local_model import log_local_model, prepare_pyfunc_input
    from skyulf.integrations.mlflow.runs.tracking import TrackingConfig, track_run

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    path = tmp_path / "local"
    save_local_pipeline(fitted_temporal_pipeline(engine), path)
    artifact = load_local_pipeline(path)
    config = TrackingConfig(
        enabled=True,
        tracking_uri=f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}",
        experiment_name="temporal",
    )
    original_uri = mlflow.get_tracking_uri()
    try:
        with track_run(config, run_name="temporal-package") as run:
            assert run.run_id is not None
            model_uri = log_local_model(
                path, run_id=run.run_id, artifact_path="model", tracking_uri=config.tracking_uri
            )
        mlflow.set_tracking_uri(config.tracking_uri)
        loaded = mlflow.pyfunc.load_model(model_uri)
        rows = pd.DataFrame({"t": [20, 21], "v": [20.0, 21.0]})
        expected = predict_local_pipeline(rows, artifact)
        request = prepare_pyfunc_input(rows, loaded)
        with TemporalHistorySession(artifact.manifest.pipeline_sha256) as first:
            a = loaded.predict(request.iloc[:1])
        with TemporalHistorySession(artifact.manifest.pipeline_sha256, first.state):
            b = loaded.predict(request.iloc[1:])
        np.testing.assert_allclose(pd.concat([a, b])["prediction"], expected["prediction"])
    finally:
        mlflow.set_tracking_uri(original_uri)
