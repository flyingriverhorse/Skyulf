"""The backend must enforce the same explicit temporal feature contract as Core."""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from backend.config import get_settings
from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.deployment.service import DeploymentService


@pytest.mark.parametrize("engine_name", ["pandas", "polars"])
@pytest.mark.parametrize("model_type", ["linear_regression", "decision_tree_regressor"])
@pytest.mark.parametrize("date_mode", ["raw", "keep", "drop"])
def test_backend_date_features_model_contract(
    tmp_path, monkeypatch, engine_name, model_type, date_mode
):
    """Real job fitting and bundle reload must use extracted dates or reject the raw column."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", engine_name)
    monkeypatch.setattr(get_settings(), "TUNING_N_JOBS", 1)
    frame = pd.DataFrame(
        {
            "when": pd.date_range("2024-01-01", periods=60),
            "x": np.arange(60, dtype=float),
            "target": np.arange(60, dtype=float) * 2,
        }
    )
    source = tmp_path / "data.parquet"
    frame.to_parquet(source, index=False)
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    nodes = [
        NodeConfig("source", "data_loader", {"path": str(source)}),
        NodeConfig(
            "split",
            "TrainTestSplitter",
            {"target_column": "target", "test_size": 0.2, "random_state": 42},
            ["source"],
        ),
    ]
    if date_mode != "raw":
        nodes.append(
            NodeConfig(
                "dates",
                "DateFeatures",
                {
                    "columns": ["when"],
                    "features": ["day", "month"],
                    "drop_original": date_mode == "drop",
                },
                ["split"],
            )
        )
    nodes.append(
        NodeConfig(
            "trainer",
            "training",
            {
                "model_type": model_type,
                "target_column": "target",
                "cv_enabled": True,
                "cv_folds": 2,
            },
            [nodes[-1].node_id],
        )
    )
    result = PipelineEngine(store, FileSystemCatalog(str(tmp_path))).run(
        PipelineConfig("dates", nodes), job_id="date-job"
    )
    if date_mode == "drop":
        assert result.status == "success", result.node_results["trainer"].error
        bundle = store.load("date-job")
        transformed = bundle["feature_engineer"].transform(frame.drop(columns="target").head(3))
        assert "when" not in transformed.columns
        assert len(bundle["model"].predict(transformed)) == 3
    else:
        assert result.status == "failed"
        assert "Raw temporal features" in result.node_results["trainer"].error
        assert not store.exists("date-job")


@pytest.mark.parametrize("legacy", [False, True])
def test_old_temporal_estimators_require_explicit_features_at_deployment(legacy):
    """Reloading an old estimator must not restore implicit datetime-to-number conversion."""
    frame = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=5)})
    estimator = DecisionTreeRegressor().fit(frame.to_numpy(), np.arange(5))
    with pytest.raises(ValueError, match="Raw temporal features"):
        if legacy:
            DeploymentService._predict_with_legacy_artifact(estimator, frame)
        else:
            DeploymentService._predict_and_decode(estimator, frame, None, None)
