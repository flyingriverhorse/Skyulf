"""Require unambiguous raw row weights when refitting preprocessing."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from skyulf.data.dataset import SplitDataset
from skyulf.modeling._tuning.engine import TuningApplier, TuningCalculator
from skyulf.modeling.base import StatefulEstimator
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter
from skyulf.registry import NodeRegistry


def _payload(engine, shortened=False, weighted=True):
    """Build raw rows and already processed rows in a distinct positional order."""
    raw = pd.DataFrame({"x": np.arange(12, dtype=float)})
    target = pd.Series(
        [0.0, 1.0, 4.0, 2.0, 3.0, 7.0, 2.0, 9.0, 5.0, 15.0, 8.0, 40.0], name="target"
    )
    weights = np.arange(1, 13, dtype=float)
    processed = pd.DataFrame(StandardScaler().fit_transform(raw), columns=["x"])
    order = np.arange(12)[::-1][1:] if shortened else np.arange(12)[::-1]
    train = processed.assign(target=target).iloc[order]
    if engine == "polars":
        raw, target, train = pl.from_pandas(raw), pl.Series("target", target), pl.from_pandas(train)
    dataset = SplitDataset(
        train=train, test=train[:0], train_sample_weight=weights[order] if weighted else None
    )
    adapter = FeatureEngineerFoldAdapter(
        [{"name": "scale", "transformer": "StandardScaler", "params": {}}], "target"
    )
    calculator = NodeRegistry.get_calculator("ridge_regression")()
    estimator = StatefulEstimator(
        TuningCalculator(calculator),
        TuningApplier(NodeRegistry.get_applier("ridge_regression")()),
        "weighted",
    )
    config = {"strategy": "grid", "search_space": {"alpha": [1.0]}, "cv_folds": 2, "metric": "mse"}
    return raw, target, weights, dataset, adapter, estimator, config


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("shortened", [False, True])
def test_weighted_raw_hook_requires_explicit_original_weights(engine, shortened):
    """Equal-length reordering must be rejected just as clearly as a row-count mismatch."""
    raw, target, weights, dataset, adapter, estimator, config = _payload(engine, shortened)
    with pytest.raises(ValueError, match="preprocessing_sample_weight"):
        estimator.fit_predict(
            dataset, "target", config, preprocessing=adapter, preprocessing_train=(raw, target)
        )
    assert estimator.model is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("weighted", [False, True])
def test_explicit_raw_weights_reach_correct_rows_without_changing_predictions_order(
    engine, weighted
):
    """The refit must match a raw-weight sklearn oracle and keep processed row predictions aligned."""
    raw, target, weights, dataset, adapter, estimator, config = _payload(engine, weighted=weighted)
    supplied = {"preprocessing_sample_weight": weights} if weighted else {}
    predictions = estimator.fit_predict(
        dataset,
        "target",
        config,
        preprocessing=adapter,
        preprocessing_train=(raw, target),
        **supplied,
    )
    raw_array = np.asarray(raw)
    scaled = StandardScaler().fit_transform(raw_array)
    expected = Ridge(alpha=1.0).fit(
        scaled, np.asarray(target), sample_weight=weights if weighted else None
    )
    np.testing.assert_allclose(predictions["train"], expected.predict(scaled[::-1]))
    np.testing.assert_array_equal(weights, np.arange(1, 13, dtype=float))
    assert set(predictions) == {"train"}
