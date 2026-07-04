"""Tests for skyulf.modeling.cross_validation."""

from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.classification import (
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
)
from skyulf.modeling.cross_validation import (
    _aggregate_metrics,
    _build_splitter,
    _sort_by_time,
    perform_cross_validation,
)
from skyulf.modeling.regression import (
    RandomForestRegressorApplier,
    RandomForestRegressorCalculator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_classification_xy(n: int = 120, seed: int = 0) -> tuple:
    """Build a clean numeric classification DataFrame and Series."""
    X_arr, y_arr = make_classification(
        n_samples=n, n_features=4, n_informative=3, n_redundant=1, random_state=seed
    )
    X = pd.DataFrame(X_arr, columns=pd.Index(["a", "b", "c", "d"]))
    y = pd.Series(y_arr, name="target")
    return X, y


def _make_regression_xy(n: int = 120, seed: int = 0) -> tuple:
    """Build a clean numeric regression DataFrame and Series."""
    X_arr, y_arr = make_regression(n_samples=n, n_features=4, noise=0.1, random_state=seed)
    X = pd.DataFrame(X_arr, columns=pd.Index(["a", "b", "c", "d"]))
    y = pd.Series(y_arr, name="target")
    return X, y


# ---------------------------------------------------------------------------
# _aggregate_metrics
# ---------------------------------------------------------------------------


def test_aggregate_metrics_returns_mean_std():
    """Mean and std should be computed correctly across folds."""
    fold_metrics = [
        {"accuracy": 0.8, "f1": 0.75},
        {"accuracy": 0.9, "f1": 0.85},
        {"accuracy": 0.7, "f1": 0.65},
    ]
    result = _aggregate_metrics(fold_metrics)
    assert "accuracy" in result
    assert result["accuracy"]["mean"] == pytest.approx(0.8, abs=1e-6)
    assert "std" in result["accuracy"]
    assert "min" in result["accuracy"]
    assert "max" in result["accuracy"]


def test_aggregate_metrics_empty_list():
    """Empty fold list should return empty dict."""
    assert _aggregate_metrics([]) == {}


def test_aggregate_metrics_filters_nan():
    """NaN values in individual folds should be excluded from aggregation."""
    fold_metrics = [
        {"accuracy": 0.9},
        {"accuracy": float("nan")},
        {"accuracy": 0.8},
    ]
    result = _aggregate_metrics(fold_metrics)
    # mean should be computed only from finite values
    assert result["accuracy"]["mean"] == pytest.approx(0.85, abs=1e-6)


def test_aggregate_metrics_single_fold_zero_std():
    """Single fold should produce std of 0.0."""
    result = _aggregate_metrics([{"r2": 0.95}])
    assert result["r2"]["std"] == 0.0


# ---------------------------------------------------------------------------
# _build_splitter
# ---------------------------------------------------------------------------


def test_build_splitter_kfold():
    """k_fold strategy should produce a KFold splitter."""
    from sklearn.model_selection import KFold

    splitter = _build_splitter("k_fold", 5, "classification")
    assert isinstance(splitter, KFold)


def test_build_splitter_stratified():
    """stratified_k_fold + classification should produce StratifiedKFold."""
    from sklearn.model_selection import StratifiedKFold

    splitter = _build_splitter("stratified_k_fold", 5, "classification")
    assert isinstance(splitter, StratifiedKFold)


def test_build_splitter_stratified_regression_falls_back():
    """stratified_k_fold for regression should fall back to KFold."""
    from sklearn.model_selection import KFold

    splitter = _build_splitter("stratified_k_fold", 5, "regression")
    assert isinstance(splitter, KFold)


def test_build_splitter_time_series():
    """time_series_split should produce TimeSeriesSplit."""
    from sklearn.model_selection import TimeSeriesSplit

    splitter = _build_splitter("time_series_split", 4, "regression")
    assert isinstance(splitter, TimeSeriesSplit)


def test_build_splitter_shuffle_split():
    """shuffle_split should produce ShuffleSplit."""
    from sklearn.model_selection import ShuffleSplit

    splitter = _build_splitter("shuffle_split", 3, "classification")
    assert isinstance(splitter, ShuffleSplit)


# ---------------------------------------------------------------------------
# perform_cross_validation — classification
# ---------------------------------------------------------------------------


def test_perform_cv_kfold_classification_returns_aggregated():
    """k_fold CV on classification data should have aggregated accuracy."""
    X, y = _make_classification_xy()
    calc = LogisticRegressionCalculator()
    appl = LogisticRegressionApplier()

    result = perform_cross_validation(calc, appl, X, y, config={}, n_folds=3, cv_type="k_fold")

    assert "aggregated_metrics" in result
    assert "accuracy" in result["aggregated_metrics"]
    assert len(result["folds"]) == 3


def test_perform_cv_fold_count_matches_n_folds():
    """Number of fold entries should exactly equal n_folds."""
    X, y = _make_classification_xy()
    calc = LogisticRegressionCalculator()
    appl = LogisticRegressionApplier()

    for n_folds in (2, 4, 5):
        result = perform_cross_validation(calc, appl, X, y, config={}, n_folds=n_folds)
        assert len(result["folds"]) == n_folds


def test_perform_cv_cv_config_recorded():
    """The returned cv_config block should mirror the input parameters."""
    X, y = _make_classification_xy()
    calc = LogisticRegressionCalculator()
    appl = LogisticRegressionApplier()

    result = perform_cross_validation(
        calc, appl, X, y, config={}, n_folds=4, cv_type="shuffle_split", random_state=7
    )
    cfg = result["cv_config"]
    assert cfg["n_folds"] == 4
    assert cfg["cv_type"] == "shuffle_split"
    assert cfg["random_state"] == 7


def test_perform_cv_stratified_classification():
    """stratified_k_fold CV on classification should succeed and return folds."""
    X, y = _make_classification_xy()
    calc = LogisticRegressionCalculator()
    appl = LogisticRegressionApplier()

    result = perform_cross_validation(
        calc, appl, X, y, config={}, n_folds=3, cv_type="stratified_k_fold"
    )
    assert len(result["folds"]) == 3


def test_perform_cv_shuffle_split():
    """shuffle_split CV should work and return the expected number of folds."""
    X, y = _make_classification_xy()
    calc = LogisticRegressionCalculator()
    appl = LogisticRegressionApplier()

    result = perform_cross_validation(
        calc, appl, X, y, config={}, n_folds=3, cv_type="shuffle_split"
    )
    assert len(result["folds"]) == 3


def test_perform_cv_shuffle_false():
    """shuffle=False should still run without error."""
    X, y = _make_classification_xy()
    calc = LogisticRegressionCalculator()
    appl = LogisticRegressionApplier()

    result = perform_cross_validation(calc, appl, X, y, config={}, n_folds=3, shuffle=False)
    assert len(result["folds"]) == 3


# ---------------------------------------------------------------------------
# perform_cross_validation — regression
# ---------------------------------------------------------------------------


def test_perform_cv_regression_has_r2():
    """k_fold CV on regression data should expose r2 in aggregated metrics."""
    X, y = _make_regression_xy()
    calc = RandomForestRegressorCalculator()
    appl = RandomForestRegressorApplier()

    result = perform_cross_validation(
        calc,
        appl,
        X,
        y,
        config={"params": {"n_estimators": 5}},
        n_folds=3,
    )
    assert "r2" in result["aggregated_metrics"]


def test_perform_cv_time_series_split():
    """time_series_split should run without error on sequential data."""
    X, y = _make_regression_xy(n=100)
    calc = RandomForestRegressorCalculator()
    appl = RandomForestRegressorApplier()

    result = perform_cross_validation(
        calc,
        appl,
        X,
        y,
        config={"params": {"n_estimators": 5}},
        n_folds=3,
        cv_type="time_series_split",
    )
    assert len(result["folds"]) == 3


# ---------------------------------------------------------------------------
# perform_cross_validation — callbacks
# ---------------------------------------------------------------------------


def test_perform_cv_progress_callback_called_per_fold():
    """progress_callback should be called once per fold."""
    X, y = _make_classification_xy()
    calc = LogisticRegressionCalculator()
    appl = LogisticRegressionApplier()

    calls: List[tuple] = []
    perform_cross_validation(
        calc,
        appl,
        X,
        y,
        config={},
        n_folds=3,
        progress_callback=lambda cur, total: calls.append((cur, total)),
    )
    assert len(calls) == 3
    # fold numbers should be 1, 2, 3
    assert [c[0] for c in calls] == [1, 2, 3]


def test_perform_cv_log_callback_receives_messages():
    """log_callback should receive at least one message per fold."""
    X, y = _make_classification_xy()
    calc = LogisticRegressionCalculator()
    appl = LogisticRegressionApplier()

    messages: List[str] = []
    perform_cross_validation(calc, appl, X, y, config={}, n_folds=2, log_callback=messages.append)
    assert len(messages) >= 2


# ---------------------------------------------------------------------------
# perform_cross_validation — nested_cv
# ---------------------------------------------------------------------------


def test_perform_nested_cv_classification():
    """nested_cv should return inner_cv_mean on each fold entry."""
    X, y = _make_classification_xy(n=150)
    calc = LogisticRegressionCalculator()
    appl = LogisticRegressionApplier()

    result = perform_cross_validation(calc, appl, X, y, config={}, n_folds=3, cv_type="nested_cv")
    assert result["cv_config"]["cv_type"] == "nested_cv"
    assert len(result["folds"]) == 3
    for fold in result["folds"]:
        assert "inner_cv_mean" in fold


def test_perform_nested_cv_regression():
    """nested_cv for regression should aggregate r2 across outer folds."""
    X, y = _make_regression_xy(n=150)
    calc = RandomForestRegressorCalculator()
    appl = RandomForestRegressorApplier()

    result = perform_cross_validation(
        calc,
        appl,
        X,
        y,
        config={"params": {"n_estimators": 5}},
        n_folds=3,
        cv_type="nested_cv",
    )
    assert "r2" in result["aggregated_metrics"]


# ---------------------------------------------------------------------------
# _sort_by_time
# ---------------------------------------------------------------------------


def test_sort_by_time_explicit_column():
    """Explicit time_column should sort X and y and drop the time column."""
    import logging

    logger = logging.getLogger(__name__)

    dates = pd.to_datetime(["2023-03-01", "2023-01-01", "2023-02-01"])
    X = pd.DataFrame({"date": dates, "val": [3, 1, 2]})
    y = pd.Series([3, 1, 2])

    X_out, y_out = _sort_by_time(X, y, "date", None, logger)

    assert "date" not in X_out.columns
    # Values should be in chronological order
    assert list(X_out["val"]) == [1, 2, 3]


def test_sort_by_time_no_datetime_column():
    """Without a datetime column and no time_column, data should be returned as-is."""
    import logging

    logger = logging.getLogger(__name__)

    X = pd.DataFrame({"a": [3, 1, 2], "b": [1, 2, 3]})
    y = pd.Series([3, 1, 2])

    X_out, y_out = _sort_by_time(X, y, None, None, logger)

    # No datetime column — data unchanged
    assert list(X_out["a"]) == [3, 1, 2]


def test_sort_by_time_auto_detect_datetime():
    """Auto-detection should find the first datetime64 column."""
    import logging

    logger = logging.getLogger(__name__)

    dates = pd.to_datetime(["2023-03-01", "2023-01-01", "2023-02-01"])
    X = pd.DataFrame({"ts": dates, "val": [30, 10, 20]})
    y = pd.Series([30, 10, 20])

    X_out, y_out = _sort_by_time(X, y, None, None, logger)

    # ts should be dropped and rows sorted
    assert "ts" not in X_out.columns
    assert list(X_out["val"]) == [10, 20, 30]


def test_sort_by_time_missing_specified_column():
    """Specifying a non-existent time column should return data as-is with a warning."""
    import logging

    logger = logging.getLogger(__name__)

    X = pd.DataFrame({"a": [3, 1, 2]})
    y = pd.Series([3, 1, 2])

    X_out, y_out = _sort_by_time(X, y, "nonexistent", None, logger)

    # Data unchanged
    assert list(X_out["a"]) == [3, 1, 2]


# ---------------------------------------------------------------------------
# perform_cross_validation — time_series with explicit time column
# ---------------------------------------------------------------------------


def test_perform_cv_time_series_with_datetime_column():
    """time_series_split should sort by datetime column when provided."""
    np.random.seed(1)
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    X = pd.DataFrame({"ts": dates, "x1": np.random.randn(90), "x2": np.random.randn(90)})
    y = pd.Series(np.random.randint(0, 2, 90), name="target")

    calc = LogisticRegressionCalculator()
    appl = LogisticRegressionApplier()

    result = perform_cross_validation(
        calc, appl, X, y, config={}, n_folds=3, cv_type="time_series_split", time_column="ts"
    )
    assert len(result["folds"]) == 3
