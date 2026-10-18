"""Multivariate detectors learn jointly and keep positional exclusion contracts."""

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.registry import NodeRegistry


def _train():
    """Correlated features expose behavior that separate per-column filters cannot learn."""
    rng = np.random.default_rng(19)
    x = rng.normal(size=60)
    return pd.DataFrame({"x": x, "z": x + rng.normal(scale=0.08, size=60)}, index=[7] * 60)


def _native(frame, engine):
    """Preserve the selected dataframe engine and pandas duplicate indexes."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _reference(node):
    """Build sklearn with exactly the public recipe used in these comparisons."""
    if node == "IsolationForest":
        return IsolationForest(contamination=0.1, random_state=42, n_estimators=20)
    return LocalOutlierFactor(contamination=0.1, n_neighbors=8, novelty=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["IsolationForest", "LocalOutlierFactor"])
def test_multivariate_saved_predictions_match_sklearn_and_chunks(engine, node):
    """Saved joint detectors must score new rows independently of request partitions."""
    train = _train()
    config = {"columns": ["x", "z"], "contamination": 0.1, "n_estimators": 20, "n_neighbors": 8}
    calculator = NodeRegistry.get_calculator(node)()
    applier = NodeRegistry.get_applier(node)()
    state = calculator.fit(_native(train, engine), config)
    reference = _reference(node).fit(train.to_numpy())
    sample = pd.DataFrame(
        {"x": [0.1, 0.3, 8.0, np.nan], "z": [0.1, -3.0, -8.0, 0.1]}, index=[2] * 4
    )
    keep = np.r_[reference.predict(sample.iloc[:3].to_numpy()) == 1, True]
    restored = pickle.loads(pickle.dumps(state))
    for other in ("pandas", "polars"):
        native = _native(sample, other)
        result, positions = applier.apply((native, np.arange(4)), restored)
        np.testing.assert_array_equal(positions, np.flatnonzero(keep))
        assert len(result) == int(keep.sum())
        parts = [applier.apply((native[i : i + 1], np.array([i])), restored)[1] for i in range(4)]
        np.testing.assert_array_equal(np.concatenate(parts), positions)
        empty = applier.apply(native.head(0), restored)
        assert empty.shape == native.head(0).shape
    assert applier.validate_inference_state(restored) is restored


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["IsolationForest", "LocalOutlierFactor"])
def test_multivariate_training_and_weights_follow_real_training_mask(engine, node):
    """LOF training scores and positional weights must not be replaced by novelty predictions."""
    frame = _train()
    labels = np.arange(len(frame))
    weights = labels + 0.5
    reference = _reference(node).fit(frame.to_numpy())
    keep = (
        reference.negative_outlier_factor_ >= reference.offset_
        if node == "LocalOutlierFactor"
        else reference.predict(frame.to_numpy()) == 1
    )
    config = {"columns": ["x", "z"], "contamination": 0.1, "n_estimators": 20, "n_neighbors": 8}
    engineer = FeatureEngineer([{"name": "joint", "transformer": node, "params": config}])
    (result, target), _ = engineer.fit_transform(
        (_native(frame, engine), labels), sample_weight=weights
    )
    np.testing.assert_array_equal(target, labels[keep])
    np.testing.assert_array_equal(engineer.train_sample_weight_, weights[keep])
    assert len(result) == int(keep.sum())
    if node == "LocalOutlierFactor":
        assert not np.array_equal(keep, reference.predict(frame.to_numpy()) == 1)


@pytest.mark.parametrize("node", ["IsolationForest", "LocalOutlierFactor"])
def test_multivariate_noop_target_exclusion_missing_axes_and_invalid_fit(node):
    """Explicit no-op remains valid while unusable data and missing fitted axes fail visibly."""
    calc = NodeRegistry.get_calculator(node)()
    app = NodeRegistry.get_applier(node)()
    train = _train()
    assert calc.fit(train, {"columns": []}) == {}
    state = calc.fit(train, {"target_column": "z", "n_estimators": 5, "n_neighbors": 3})
    assert state["columns"] == ["x"]
    with pytest.raises(ValueError, match="columns"):
        app.apply(train.drop(columns="x"), state)
    with pytest.raises(ValueError, match="finite"):
        calc.fit(pd.DataFrame({"x": [np.nan, np.inf]}), {"columns": ["x"]})
    with pytest.raises(ValueError):
        calc.fit(train, {"columns": ["x", "z"], "contamination": 0.8})


@pytest.mark.parametrize("node", ["IsolationForest", "LocalOutlierFactor"])
@pytest.mark.parametrize("damage", ["width", "extra", "offset"])
def test_multivariate_corrupt_saved_state_rejected(node, damage):
    """Malformed estimator axes and offsets cannot acquire row-filter admission."""
    state = NodeRegistry.get_calculator(node)().fit(_train(), {"n_estimators": 5, "n_neighbors": 3})
    if damage == "width":
        state["detector_object"].n_features_in_ = 99
    elif damage == "extra":
        state["extra"] = True
    else:
        state["detector_object"].offset_ = np.nan
    with pytest.raises(ValueError):
        NodeRegistry.get_applier(node)().validate_inference_state(state)


@pytest.mark.parametrize("node", ["IsolationForest", "LocalOutlierFactor"])
def test_multivariate_training_retains_nonfinite_rows_and_their_weights(node):
    """Rows without a detector decision must retain their original target and weight positions."""
    frame = pd.concat([_train(), pd.DataFrame({"x": [np.nan], "z": [1.0]})], ignore_index=True)
    labels = np.arange(len(frame))
    weights = labels + 1.0
    engineer = FeatureEngineer(
        [
            {
                "name": "joint",
                "transformer": node,
                "params": {"columns": ["x", "z"], "n_neighbors": 8, "n_estimators": 10},
            }
        ]
    )
    (out, target), _ = engineer.fit_transform((frame, labels), sample_weight=weights)
    assert target[-1] == 60 and pd.isna(out.x.iloc[-1])
    np.testing.assert_array_equal(engineer.train_sample_weight_, target + 1.0)


@pytest.mark.parametrize(
    "node,option", [("IsolationForest", "n_estimators"), ("LocalOutlierFactor", "n_neighbors")]
)
def test_multivariate_count_options_reject_boolean(node, option):
    """Boolean configuration must not silently become a one-tree or one-neighbor recipe."""
    with pytest.raises(ValueError, match="booleans"):
        NodeRegistry.get_calculator(node)().fit(_train(), {"columns": ["x", "z"], option: True})


@pytest.mark.parametrize("damage", ["callback", "width", "tree_type"])
def test_isolation_forest_rejects_corrupted_child_tree_state(damage):
    """A nested tree override can change predictions despite an intact top-level forest type."""
    state = NodeRegistry.get_calculator("IsolationForest")().fit(_train(), {"n_estimators": 3})
    child = state["detector_object"].estimators_[0]
    if damage == "callback":
        child.apply = lambda values, check_input=True: np.zeros(len(values), dtype=np.intp)
    elif damage == "width":
        child.n_features_in_ = 99
    else:
        child.tree_ = object()
    with pytest.raises(ValueError, match="tree"):
        NodeRegistry.get_applier("IsolationForest")().validate_inference_state(state)
