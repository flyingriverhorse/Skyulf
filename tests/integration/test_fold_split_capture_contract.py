"""Keep composite fold snapshots exact and fail closed when they are unavailable."""

import numpy as np
import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from skyulf.modeling.base import extract_xy


def _engine(tmp_path, *, repeated=False):
    """Execute an unseeded split inside a real composite and retain its artifacts."""
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    logs = []
    engine = PipelineEngine(store, FileSystemCatalog(), log_callback=logs.append)
    store.save(
        "load",
        pd.DataFrame({"x": np.arange(96, dtype=float), "target": np.arange(96) % 2}),
    )
    steps = [
        {
            "name": "split",
            "transformer": "TrainTestSplitter",
            "params": {"target_column": "target", "random_state": None},
        },
        {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
    ]
    if repeated:
        steps.append({"name": "again", "transformer": "Split", "params": {}})
    features = NodeConfig(
        "features", "feature_engineering", inputs=["load"], params={"steps": steps}
    )
    model = NodeConfig("model", "training", inputs=["features"], params={"target_column": "target"})
    engine._node_configs = {
        node.node_id: node for node in [NodeConfig("load", "data_loader"), features, model]
    }
    engine._run_feature_engineering(features)
    return engine, model, store, logs


@pytest.mark.parametrize("mode", ["raise", "warn", "ignore"])
@pytest.mark.parametrize("failure", ["missing", "invalid_type"])
def test_missing_or_invalid_split_snapshot_obeys_leakage_policy(
    tmp_path, monkeypatch, mode, failure
):
    """An unavailable raw snapshot must never be replaced by a newly randomized split."""
    engine, model, store, logs = _engine(tmp_path)
    engine._on_leakage = mode
    original_load = store.load

    def load_artifact(key):
        """Fail only the saved raw partition while leaving processed artifacts usable."""
        if key == "exec_features_split":
            if failure == "missing":
                raise FileNotFoundError("missing raw split")
            return pd.DataFrame({"x": [1.0], "target": [0]})
        return original_load(key)

    monkeypatch.setattr(store, "load", load_artifact)
    if mode == "raise":
        with pytest.raises(ValueError, match="payload reconstruction failed"):
            engine._resolve_fold_preprocessing(model, "target")
    else:
        resolved, fallback = engine._resolve_fold_preprocessing(model, "target")
        assert resolved is None
        assert fallback == "payload_reconstruction_failed"
        assert any("Per-fold preprocessing refit skipped" in log for log in logs) is (
            mode == "warn"
        )


def test_repeated_split_keeps_the_original_pre_transform_snapshot(tmp_path):
    """A no-op splitter after scaling cannot overwrite the first partition with scaled values."""
    engine, model, store, _logs = _engine(tmp_path, repeated=True)
    resolved, fallback = engine._resolve_fold_preprocessing(model, "target")
    assert fallback is None
    assert resolved is not None
    _adapter, (raw_x, raw_y), _validation = resolved
    processed_x, processed_y = extract_xy(store.load("features").train, "target")
    np.testing.assert_array_equal(raw_x["x"], raw_x.index.to_numpy(dtype=float))
    np.testing.assert_array_equal(raw_y, processed_y)
    assert abs(processed_x["x"].mean()) < 1e-12
