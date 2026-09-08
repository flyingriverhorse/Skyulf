"""Probe exact split snapshots across supported multi-node preprocessing graphs."""

import numpy as np
import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import extract_xy
from skyulf.preprocessing.pipeline import FeatureEngineer


def _step(name, transformer, **params):
    """Represent the same ordered steps consumed by real backend execution."""
    return {"name": name, "transformer": transformer, "params": params}


def _frame():
    """Keep immutable row IDs independent of transformed features and encoded labels."""
    return pd.DataFrame(
        {
            "row_id": np.arange(96),
            "x": np.arange(1, 97, dtype=float),
            "target": ["class_a" if index % 2 else "class_b" for index in range(96)],
            "other_target": np.arange(96) % 3,
        }
    )


@pytest.mark.parametrize("boundary", ["separate", "composite", "stateless_fan_in"])
@pytest.mark.parametrize("encode_target", [False, True], ids=["string-target", "encoded-target"])
def test_noop_internal_split_uses_the_actual_upstream_boundary(
    tmp_path, monkeypatch, boundary, encode_target
):
    """A later no-op splitter must not request its own snapshot or change outer holdout rows."""
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    engine = PipelineEngine(store, FileSystemCatalog())
    saved_keys = []
    loaded_keys = []
    original_save = store.save
    original_load = store.load

    def record_save(key, artifact):
        """Observe actual persistence while keeping real serialization active."""
        saved_keys.append(key)
        return original_save(key, artifact)

    def record_load(key):
        """Record the boundary selected by reconstruction without substituting its data."""
        loaded_keys.append(key)
        return original_load(key)

    monkeypatch.setattr(store, "save", record_save)
    monkeypatch.setattr(store, "load", record_load)
    store.save("load", _frame())
    prefix_steps = [
        _step(
            "square", "GeneralTransformation", transformations=[{"column": "x", "method": "square"}]
        )
    ]
    if encode_target:
        prefix_steps.append(_step("target", "OrdinalEncoder", columns=["target"]))
    prefix = NodeConfig(
        "prefix", "feature_engineering", inputs=["load"], params={"steps": prefix_steps}
    )
    split_params = {
        "target_column": "target",
        "test_size": 0.2,
        "validation_size": 0.2,
        "random_state": None,
    }
    middle = [prefix]
    first_inputs = ["prefix"]
    if boundary == "stateless_fan_in":
        middle.extend(
            [
                NodeConfig(
                    "left", "DropMissingColumns", inputs=["prefix"], params={"columns": ["absent"]}
                ),
                NodeConfig(
                    "right",
                    "ValueReplacement",
                    inputs=["prefix"],
                    params={"columns": ["x"], "mapping": {}},
                ),
            ]
        )
        first_inputs = ["left", "right"]
    if boundary == "separate":
        middle.extend(
            [
                NodeConfig("first", "TrainTestSplitter", inputs=first_inputs, params=split_params),
                NodeConfig("scale", "StandardScaler", inputs=["first"], params={"columns": ["x"]}),
            ]
        )
        later_inputs = ["scale"]
        expected_boundary = "first"
    else:
        middle.append(
            NodeConfig(
                "first",
                "feature_engineering",
                inputs=first_inputs,
                params={
                    "steps": [
                        _step("split", "TrainTestSplitter", **split_params),
                        _step("scale", "StandardScaler", columns=["x"]),
                    ],
                },
            )
        )
        later_inputs = ["first"]
        expected_boundary = "exec_first_split"
    later = NodeConfig(
        "later",
        "feature_engineering",
        inputs=later_inputs,
        params={
            "steps": [
                _step(
                    "no_op_split", "TrainTestSplitter", target_column="target", random_state=None
                ),
                _step("scale_again", "MinMaxScaler", columns=["x"]),
            ],
        },
    )
    model = NodeConfig("model", "training", inputs=["later"], params={"target_column": "target"})
    sibling = NodeConfig(
        "sibling", "training", inputs=["load"], params={"target_column": "other_target"}
    )
    nodes = [NodeConfig("load", "data_loader"), *middle, later, model, sibling]
    engine._node_configs = {node.node_id: node for node in nodes}
    for node in [*middle, later]:
        if node.step_type == "feature_engineering":
            engine._run_feature_engineering(node)
        else:
            engine._run_transformer(node)
    expected_x, expected_y = extract_xy(store.load("prefix"), "target")
    eager = store.load("later")
    loaded_keys.clear()
    resolved, fallback = engine._resolve_fold_preprocessing(model, "target")
    assert fallback is None
    assert resolved is not None
    assert expected_boundary in loaded_keys
    assert "exec_later_split" not in saved_keys
    assert "exec_later_split" not in loaded_keys
    adapter, (raw_x, raw_y), raw_validation = resolved
    assert raw_validation is not None
    source = expected_x.set_index("row_id")
    targets = pd.Series(np.asarray(expected_y), index=expected_x["row_id"])
    np.testing.assert_array_equal(raw_x["x"], source.loc[raw_x["row_id"], "x"])
    np.testing.assert_array_equal(raw_y, targets.loc[raw_x["row_id"]])
    train_ids = set(raw_x["row_id"])
    test_ids = set(extract_xy(eager.test, "target")[0]["row_id"])
    validation_ids = set(extract_xy(eager.validation, "target")[0]["row_id"])
    assert train_ids == set(extract_xy(eager.train, "target")[0]["row_id"])
    assert train_ids.isdisjoint(test_ids | validation_ids)
    assert set(raw_validation[0]["row_id"]) == validation_ids
    half = len(raw_x) // 2
    fold_x, fold_y = adapter.fit_transform(raw_x.iloc[:half].copy(), raw_y.iloc[:half].copy())
    np.testing.assert_array_equal(fold_y, raw_y.iloc[:half])
    assert fold_x["x"].min() == pytest.approx(0.0)
    assert fold_x["x"].max() == pytest.approx(1.0)


def test_snapshot_is_persisted_before_a_downstream_applier_mutates_its_input(tmp_path, monkeypatch):
    """Persistence must finish at the boundary even when a later step mutates the split object."""
    original_run = FeatureEngineer._run_step

    def mutate_then_run(engineer, **kwargs):
        """Exercise the adversarial in-place mutation allowed by a transformer implementation."""
        if kwargs["name"] == "mutate_after_split":
            current = kwargs["current_data"]
            assert isinstance(current, SplitDataset)
            current.train[0].loc[:, "x"] += 1000.0
        return original_run(engineer, **kwargs)

    monkeypatch.setattr(FeatureEngineer, "_run_step", mutate_then_run)
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    engine = PipelineEngine(store, FileSystemCatalog())
    store.save("load", _frame())
    features = NodeConfig(
        "features",
        "feature_engineering",
        inputs=["load"],
        params={
            "steps": [
                _step("split", "TrainTestSplitter", target_column="target", random_state=None),
                _step("mutate_after_split", "StandardScaler", columns=["x"]),
            ],
        },
    )
    model = NodeConfig("model", "training", inputs=["features"], params={"target_column": "target"})
    engine._node_configs = {
        node.node_id: node for node in [NodeConfig("load", "data_loader"), features, model]
    }
    engine._run_feature_engineering(features)
    resolved, fallback = engine._resolve_fold_preprocessing(model, "target")
    assert fallback is None
    assert resolved is not None
    _adapter, (raw_x, _raw_y), _validation = resolved
    np.testing.assert_array_equal(raw_x["x"], raw_x["row_id"] + 1.0)
