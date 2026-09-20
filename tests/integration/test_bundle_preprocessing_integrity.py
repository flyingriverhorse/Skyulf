"""A model must not be published when required fitted preprocessing is unavailable."""

from pathlib import Path

import pandas as pd
import pytest

from backend.config import get_settings
from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore


@pytest.mark.parametrize("frame_engine", ["pandas", "polars"])
@pytest.mark.parametrize("retained_transform", [False, True])
@pytest.mark.parametrize("damage", [None, "missing", "corrupt", "all_missing", "invalid"])
def test_real_training_requires_complete_preprocessing_bundle(
    tmp_path, monkeypatch, damage, frame_engine, retained_transform
):
    """A lost scaler cannot turn a successfully trained model into an unsafe raw-data bundle."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", frame_engine)
    source = tmp_path / "source.csv"
    pd.DataFrame({"x": list(range(80)), "target": [0] * 40 + [1] * 40}).to_csv(source, index=False)
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    original_save = store.save

    def save_with_storage_fault(key, data):
        """Simulate loss or corruption after the actual fitted scaler was persisted."""
        original_save(key, data)
        if damage == "all_missing" and key.endswith("_pipeline"):
            Path(store.get_artifact_uri(key)).unlink()
        elif key == "exec_scale_pipeline" and damage:
            path = Path(store.get_artifact_uri(key))
            if damage == "missing":
                path.unlink()
            elif damage == "invalid":
                original_save(key, {"not": "a fitted pipeline"})
            else:
                path.write_bytes(b"corrupt fitted preprocessing")

    monkeypatch.setattr(store, "save", save_with_storage_fault)
    extra = (
        [NodeConfig("impute", "SimpleImputer", {"columns": ["x"], "strategy": "median"}, ["split"])]
        if retained_transform
        else []
    )
    config = PipelineConfig(
        "integrity",
        [
            NodeConfig("source", "data_loader", {"path": str(source)}),
            NodeConfig(
                "split",
                "TrainTestSplitter",
                {"target_column": "target", "test_size": 0.2, "random_state": 42},
                ["source"],
            ),
            *extra,
            NodeConfig(
                "scale", "StandardScaler", {"columns": ["x"]}, ["impute" if extra else "split"]
            ),
            NodeConfig(
                "trainer",
                "training",
                {
                    "model_type": "logistic_regression",
                    "target_column": "target",
                    "cv_enabled": False,
                },
                ["scale"],
            ),
        ],
    )
    engine = PipelineEngine(store, FileSystemCatalog(str(tmp_path)))
    result = engine.run(config, job_id="integrity-job")
    if damage:
        assert result.status == "failed"
        missing_node = "split" if damage == "all_missing" else "scale"
        assert missing_node in result.node_results["trainer"].error
        assert not store.exists("integrity-job")
    else:
        assert result.status == "success"
        bundle = store.load("integrity-job")
        assert "feature_engineer" in bundle
        transformed = bundle["feature_engineer"].transform(pd.DataFrame({"x": [10, 70]}))
        assert abs(float(transformed["x"][0])) < 3
        assert len(bundle["model"].predict(transformed)) == 2
