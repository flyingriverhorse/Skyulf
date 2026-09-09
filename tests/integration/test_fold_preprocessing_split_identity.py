"""Pin the original split membership when constructing raw fold-refit payloads."""

import numpy as np
import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from skyulf.modeling.base import extract_xy


@pytest.mark.parametrize("composite", [False, True], ids=["separate", "composite"])
@pytest.mark.parametrize("random_state", [42, None], ids=["seeded", "unseeded"])
@pytest.mark.parametrize("validation_size", [0.0, 0.2], ids=["test-only", "validation"])
def test_fold_payload_preserves_original_split_membership(
    tmp_path, composite, random_state, validation_size
):
    """Reconstructing raw features must never move original holdout rows into a CV fit."""
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    engine = PipelineEngine(store, FileSystemCatalog())
    store.save(
        "load",
        pd.DataFrame(
            {
                "row_id": np.arange(96),
                "x": np.arange(96, dtype=float),
                "target": np.arange(96) % 2,
            }
        ),
    )
    split_step = {
        "name": "split",
        "transformer": "TrainTestSplitter",
        "params": {
            "target_column": "target",
            "test_size": 0.2,
            "validation_size": validation_size,
            "random_state": random_state,
        },
    }
    scale_step = {
        "name": "scale",
        "transformer": "StandardScaler",
        "params": {"columns": ["x"]},
    }
    loader = NodeConfig("load", "data_loader")
    split = NodeConfig("split", "TrainTestSplitter", inputs=["load"], params=split_step["params"])
    features = NodeConfig(
        "features",
        "feature_engineering",
        inputs=["load" if composite else "split"],
        params={"steps": [split_step, scale_step] if composite else [scale_step]},
    )
    model = NodeConfig(
        "model",
        "training",
        inputs=["features"],
        params={"target_column": "target", "cv_enabled": True},
    )
    nodes = [loader, features, model] if composite else [loader, split, features, model]
    engine._node_configs = {node.node_id: node for node in nodes}
    original_rng = np.random.get_state()
    try:
        np.random.seed(1)
        if not composite:
            engine._run_transformer(split)
        engine._run_feature_engineering(features)
        eager = store.load("features")
        np.random.seed(2)
        resolved, fallback = engine._resolve_fold_preprocessing(model, "target")
    finally:
        np.random.set_state(original_rng)

    assert fallback is None
    assert resolved is not None
    _adapter, (raw_train, _target), raw_validation = resolved
    train_ids = set(extract_xy(eager.train, "target")[0]["row_id"])
    test_ids = set(extract_xy(eager.test, "target")[0]["row_id"])
    reconstructed_ids = set(raw_train["row_id"])
    assert reconstructed_ids.isdisjoint(test_ids), (
        f"{len(reconstructed_ids & test_ids)} original test rows entered the raw CV payload"
    )
    assert reconstructed_ids == train_ids
    if eager.validation is None:
        assert raw_validation is None
    else:
        assert raw_validation is not None
        validation_ids = set(extract_xy(eager.validation, "target")[0]["row_id"])
        assert set(raw_validation[0]["row_id"]) == validation_ids
