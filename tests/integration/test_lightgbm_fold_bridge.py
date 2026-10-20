"""Canvas training must bridge native fold outputs before fitting LightGBM."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("polars")

from backend.config import get_settings
from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.deployment.service import DeploymentService


@pytest.mark.parametrize("model_type", ["lgbm_classifier", "lgbm_regressor"])
def test_polars_imputation_then_lightgbm_trains_and_replays(tmp_path, monkeypatch, model_type):
    """A normal canvas learner before LightGBM must not trigger a read-only feature-name error."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "polars")
    rng = np.random.default_rng(42)
    x = rng.normal(size=120)
    frame = pd.DataFrame({"x": x, "z": rng.normal(size=120), "target": (x > 0).astype(int)})
    if model_type == "lgbm_regressor":
        frame["target"] = 2 * x + frame["z"]
    frame.loc[::9, "x"] = np.nan
    source = tmp_path / "source.csv"
    frame.to_csv(source, index=False)
    store = LocalArtifactStore(str(tmp_path / "models"))
    result = PipelineEngine(store, FileSystemCatalog(str(tmp_path))).run(
        PipelineConfig(
            "lightgbm-fold-bridge",
            [
                NodeConfig("source", "data_loader", {"path": str(source)}),
                NodeConfig("split", "TrainTestSplitter", {"target_column": "target"}, ["source"]),
                NodeConfig(
                    "fill", "SimpleImputer", {"columns": ["x", "z"], "strategy": "mean"}, ["split"]
                ),
                NodeConfig(
                    "model",
                    "training",
                    {
                        "model_type": model_type,
                        "target_column": "target",
                        "cv_enabled": False,
                        "hyperparameters": {"n_estimators": 5, "n_jobs": 1, "min_child_samples": 2},
                    },
                    ["fill"],
                ),
            ],
        ),
        job_id="lightgbm-fold-job",
    )
    assert result.status == "success", result.node_results["model"].error
    bundle = store.load("lightgbm-fold-job")
    rows = frame.drop(columns="target").iloc[:8]
    predictions, _ = DeploymentService._predict_with_bundled_artifact(bundle, rows)
    assert len(predictions) == len(rows)
    assert np.isfinite(predictions).all()
