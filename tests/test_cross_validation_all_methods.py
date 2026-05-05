"""
Comprehensive cross-validation tests for all 5 CV methods across
both Simple (Basic Training) and Advanced (Tuning) flows.

Tests:
  - K-Fold
  - Stratified K-Fold
  - Shuffle Split
  - Time Series Split (with auto-sort and explicit time column)
  - Nested CV

Each method is tested for both classification and regression.
Advanced Tuning flow tests each CV method inside hyperparameter search.
"""

import numpy as np
import pandas as pd
import pytest
from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.classification import (
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
)
from skyulf.modeling.regression import (
    RidgeRegressionApplier,
    RidgeRegressionCalculator,
)
from skyulf.modeling._tuning import TuningCalculator, TuningConfig

# ---------------------------------------------------------------------------
# Fixtures — larger datasets so multi-fold CV doesn't starve
# ---------------------------------------------------------------------------


@pytest.fixture
def classification_dataset():
    """Binary classification: 50 samples, two features."""
    rng = np.random.RandomState(42)
    n = 50
    X = rng.randn(n, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1], "target": y})
    train = df.iloc[:40].reset_index(drop=True)
    test = df.iloc[40:].reset_index(drop=True)
    return SplitDataset(train=train, test=test)


@pytest.fixture
def regression_dataset():
    """Linear regression: y = 2*x1 + x2 + noise, 50 samples."""
    rng = np.random.RandomState(42)
    n = 50
    X = rng.randn(n, 2)
    y = 2 * X[:, 0] + X[:, 1] + rng.randn(n) * 0.1
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "target": y})
    train = df.iloc[:40].reset_index(drop=True)
    test = df.iloc[40:].reset_index(drop=True)
    return SplitDataset(train=train, test=test)


@pytest.fixture
def timeseries_classification_dataset():
    """Classification dataset with a datetime column, intentionally unsorted."""
    rng = np.random.RandomState(42)
    n = 50
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    X = rng.randn(n, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame({"date": dates, "f1": X[:, 0], "f2": X[:, 1], "target": y})
    # Shuffle rows to simulate unsorted data
    df = df.sample(frac=1, random_state=99).reset_index(drop=True)
    train = df.iloc[:40].reset_index(drop=True)
    test = df.iloc[40:].reset_index(drop=True)
    return SplitDataset(train=train, test=test)


@pytest.fixture
def timeseries_regression_dataset():
    """Regression dataset with a datetime column, intentionally unsorted."""
    rng = np.random.RandomState(42)
    n = 50
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    X = rng.randn(n, 2)
    y = 2 * X[:, 0] + X[:, 1] + rng.randn(n) * 0.1
    df = pd.DataFrame({"date": dates, "x1": X[:, 0], "x2": X[:, 1], "target": y})
    df = df.sample(frac=1, random_state=99).reset_index(drop=True)
    train = df.iloc[:40].reset_index(drop=True)
    test = df.iloc[40:].reset_index(drop=True)
    return SplitDataset(train=train, test=test)


def _make_classification_estimator() -> StatefulEstimator:
    return StatefulEstimator(
        LogisticRegressionCalculator(),
        LogisticRegressionApplier(),
        "test_node",
    )


def _make_regression_estimator() -> StatefulEstimator:
    return StatefulEstimator(
        RidgeRegressionCalculator(),
        RidgeRegressionApplier(),
        "test_node",
    )


def _assert_cv_result(result: dict, n_folds: int, problem_type: str) -> None:
    """Shared assertions for any CV result."""
    assert "aggregated_metrics" in result, "Missing aggregated_metrics key"
    assert "folds" in result, "Missing folds key"
    assert "cv_config" in result, "Missing cv_config key"
    assert len(result["folds"]) == n_folds, f"Expected {n_folds} folds, got {len(result['folds'])}"

    agg = result["aggregated_metrics"]
    assert len(agg) > 0, "No aggregated metrics returned"

    # Each metric should have mean/std/min/max
    for metric_name, stats in agg.items():
        assert "mean" in stats, f"{metric_name} missing mean"
        assert "std" in stats, f"{metric_name} missing std"
        assert "min" in stats, f"{metric_name} missing min"
        assert "max" in stats, f"{metric_name} missing max"

    if problem_type == "classification":
        assert "accuracy" in agg, "Classification should have accuracy"
        assert 0.0 <= agg["accuracy"]["mean"] <= 1.0
    else:
        assert "r2" in agg or "mse" in agg, "Regression should have r2 or mse"

    # Per-fold checks
    for fold in result["folds"]:
        assert "fold" in fold
        assert "metrics" in fold
        assert isinstance(fold["metrics"], dict)
        assert len(fold["metrics"]) > 0


# =========================================================================
# SIMPLE FLOW — Basic Training cross_validate (all 5 CV methods)
# =========================================================================


class TestSimpleCVClassification:
    """Basic Training CV — Classification with all 5 methods."""

    def test_k_fold(self, classification_dataset: SplitDataset) -> None:
        est = _make_classification_estimator()
        result = est.cross_validate(
            classification_dataset,
            "target",
            {},
            n_folds=3,
            cv_type="k_fold",
            shuffle=True,
        )
        _assert_cv_result(result, n_folds=3, problem_type="classification")
        assert result["cv_config"]["cv_type"] == "k_fold"

    def test_stratified_k_fold(self, classification_dataset: SplitDataset) -> None:
        est = _make_classification_estimator()
        result = est.cross_validate(
            classification_dataset,
            "target",
            {},
            n_folds=3,
            cv_type="stratified_k_fold",
            shuffle=True,
        )
        _assert_cv_result(result, n_folds=3, problem_type="classification")
        assert result["cv_config"]["cv_type"] == "stratified_k_fold"

    def test_shuffle_split(self, classification_dataset: SplitDataset) -> None:
        est = _make_classification_estimator()
        result = est.cross_validate(
            classification_dataset,
            "target",
            {},
            n_folds=4,
            cv_type="shuffle_split",
        )
        _assert_cv_result(result, n_folds=4, problem_type="classification")
        assert result["cv_config"]["cv_type"] == "shuffle_split"

    def test_time_series_split_auto_detect(
        self, timeseries_classification_dataset: SplitDataset
    ) -> None:
        """Time Series Split with auto-detected datetime column."""
        est = _make_classification_estimator()
        logs: list[str] = []
        result = est.cross_validate(
            timeseries_classification_dataset,
            "target",
            {},
            n_folds=3,
            cv_type="time_series_split",
            log_callback=logs.append,
        )
        _assert_cv_result(result, n_folds=3, problem_type="classification")
        assert result["cv_config"]["cv_type"] == "time_series_split"
        # Should have auto-detected and sorted
        assert any(
            "auto-detected" in log or "sorted" in log for log in logs
        ), f"Expected auto-detect log, got: {logs}"

    def test_time_series_split_explicit_column(
        self, timeseries_classification_dataset: SplitDataset
    ) -> None:
        """Time Series Split with user-specified time column."""
        est = _make_classification_estimator()
        logs: list[str] = []
        result = est.cross_validate(
            timeseries_classification_dataset,
            "target",
            {},
            n_folds=3,
            cv_type="time_series_split",
            time_column="date",
            log_callback=logs.append,
        )
        _assert_cv_result(result, n_folds=3, problem_type="classification")
        assert any(
            "sorted by 'date'" in log for log in logs
        ), f"Expected sort-by-date log, got: {logs}"

    def test_nested_cv(self, classification_dataset: SplitDataset) -> None:
        est = _make_classification_estimator()
        result = est.cross_validate(
            classification_dataset,
            "target",
            {},
            n_folds=3,
            cv_type="nested_cv",
            shuffle=True,
        )
        _assert_cv_result(result, n_folds=3, problem_type="classification")
        assert result["cv_config"]["cv_type"] == "nested_cv"
        assert "inner_folds" in result["cv_config"]
        # Each fold should have inner_cv_mean
        for fold in result["folds"]:
            assert "inner_cv_mean" in fold, "Nested CV fold missing inner_cv_mean"


class TestSimpleCVRegression:
    """Basic Training CV — Regression with all 5 methods."""

    def test_k_fold(self, regression_dataset: SplitDataset) -> None:
        est = _make_regression_estimator()
        result = est.cross_validate(
            regression_dataset,
            "target",
            {},
            n_folds=3,
            cv_type="k_fold",
            shuffle=True,
        )
        _assert_cv_result(result, n_folds=3, problem_type="regression")

    def test_stratified_k_fold_fallback(self, regression_dataset: SplitDataset) -> None:
        """Stratified K-Fold on regression should fall back to regular K-Fold."""
        est = _make_regression_estimator()
        result = est.cross_validate(
            regression_dataset,
            "target",
            {},
            n_folds=3,
            cv_type="stratified_k_fold",
        )
        _assert_cv_result(result, n_folds=3, problem_type="regression")

    def test_shuffle_split(self, regression_dataset: SplitDataset) -> None:
        est = _make_regression_estimator()
        result = est.cross_validate(
            regression_dataset,
            "target",
            {},
            n_folds=4,
            cv_type="shuffle_split",
        )
        _assert_cv_result(result, n_folds=4, problem_type="regression")

    def test_time_series_split_auto_detect(
        self, timeseries_regression_dataset: SplitDataset
    ) -> None:
        est = _make_regression_estimator()
        logs: list[str] = []
        result = est.cross_validate(
            timeseries_regression_dataset,
            "target",
            {},
            n_folds=3,
            cv_type="time_series_split",
            log_callback=logs.append,
        )
        _assert_cv_result(result, n_folds=3, problem_type="regression")
        assert any("auto-detected" in log or "sorted" in log for log in logs)

    def test_time_series_split_explicit_column(
        self, timeseries_regression_dataset: SplitDataset
    ) -> None:
        est = _make_regression_estimator()
        logs: list[str] = []
        result = est.cross_validate(
            timeseries_regression_dataset,
            "target",
            {},
            n_folds=3,
            cv_type="time_series_split",
            time_column="date",
            log_callback=logs.append,
        )
        _assert_cv_result(result, n_folds=3, problem_type="regression")
        assert any("sorted by 'date'" in log for log in logs)

    def test_time_series_split_no_datetime_column(self, regression_dataset: SplitDataset) -> None:
        """Time Series Split without any datetime column — should warn and use row order."""
        est = _make_regression_estimator()
        logs: list[str] = []
        result = est.cross_validate(
            regression_dataset,
            "target",
            {},
            n_folds=3,
            cv_type="time_series_split",
            log_callback=logs.append,
        )
        _assert_cv_result(result, n_folds=3, problem_type="regression")
        assert any(
            "no datetime column" in log.lower() or "already sorted" in log.lower() for log in logs
        )

    def test_nested_cv(self, regression_dataset: SplitDataset) -> None:
        est = _make_regression_estimator()
        result = est.cross_validate(
            regression_dataset,
            "target",
            {},
            n_folds=3,
            cv_type="nested_cv",
        )
        _assert_cv_result(result, n_folds=3, problem_type="regression")
        assert result["cv_config"]["cv_type"] == "nested_cv"
        for fold in result["folds"]:
            assert "inner_cv_mean" in fold


# =========================================================================
# ADVANCED FLOW — Tuning with each CV method × each search strategy
# =========================================================================

# All 5 tuning strategies
TUNING_STRATEGIES = ["grid", "random", "halving_grid", "halving_random", "optuna"]

# All 5 CV methods
CV_METHODS = ["k_fold", "stratified_k_fold", "shuffle_split", "time_series_split", "nested_cv"]


def _run_tuning_test(
    dataset: SplitDataset,
    problem_type: str,
    cv_type: str,
    strategy: str,
) -> None:
    """Run a tuning session with the given strategy and cv_type.

    Halving strategies may fail on small data — we accept that as expected.
    """
    if problem_type == "classification":
        calculator = LogisticRegressionCalculator()
        metric = "accuracy"
        search_space = {"C": [0.1, 1.0], "solver": ["lbfgs"]}
    else:
        calculator = RidgeRegressionCalculator()
        metric = "r2"
        search_space = {"alpha": [0.01, 1.0]}

    tuner = TuningCalculator(calculator)
    config = TuningConfig(
        strategy=strategy,
        metric=metric,
        n_trials=2,
        cv_folds=3,
        cv_type=cv_type,
        cv_shuffle=True,
        cv_random_state=42,
        search_space=search_space,
    )

    X = dataset.train.drop(columns=["target"])
    y = dataset.train["target"]

    # Drop datetime columns (not features)
    datetime_cols = X.select_dtypes(include=["datetime64", "datetimetz"]).columns
    if len(datetime_cols) > 0:
        X = X.drop(columns=datetime_cols)

    try:
        result = tuner.tune(X.values, y.values, config)
        assert result.best_params is not None
        assert result.best_score != 0.0
        assert result.n_trials >= 1
        assert len(result.trials) >= 1
    except ValueError as e:
        # Halving strategies can legitimately fail on small datasets
        err = str(e)
        if "min_resources_" in err and "max_resources_" in err:
            pytest.skip(f"Halving strategy needs more data: {err}")
        raise


class TestAdvancedTuningCV:
    """Grid Search — verify each CV method works."""

    @pytest.mark.parametrize("cv_type", CV_METHODS)
    def test_grid_classification(self, classification_dataset: SplitDataset, cv_type: str) -> None:
        _run_tuning_test(classification_dataset, "classification", cv_type, "grid")

    @pytest.mark.parametrize("cv_type", CV_METHODS)
    def test_grid_regression(self, regression_dataset: SplitDataset, cv_type: str) -> None:
        _run_tuning_test(regression_dataset, "regression", cv_type, "grid")


class TestRandomSearchCV:
    """Random Search — verify each CV method works."""

    @pytest.mark.parametrize("cv_type", CV_METHODS)
    def test_random_classification(
        self, classification_dataset: SplitDataset, cv_type: str
    ) -> None:
        _run_tuning_test(classification_dataset, "classification", cv_type, "random")

    @pytest.mark.parametrize("cv_type", CV_METHODS)
    def test_random_regression(self, regression_dataset: SplitDataset, cv_type: str) -> None:
        _run_tuning_test(regression_dataset, "regression", cv_type, "random")


class TestHalvingGridCV:
    """Halving Grid Search — verify each CV method works."""

    @pytest.mark.parametrize("cv_type", CV_METHODS)
    def test_halving_grid_classification(
        self, classification_dataset: SplitDataset, cv_type: str
    ) -> None:
        _run_tuning_test(classification_dataset, "classification", cv_type, "halving_grid")

    @pytest.mark.parametrize("cv_type", CV_METHODS)
    def test_halving_grid_regression(self, regression_dataset: SplitDataset, cv_type: str) -> None:
        _run_tuning_test(regression_dataset, "regression", cv_type, "halving_grid")


class TestHalvingRandomCV:
    """Halving Random Search — verify each CV method works."""

    @pytest.mark.parametrize("cv_type", CV_METHODS)
    def test_halving_random_classification(
        self, classification_dataset: SplitDataset, cv_type: str
    ) -> None:
        _run_tuning_test(classification_dataset, "classification", cv_type, "halving_random")

    @pytest.mark.parametrize("cv_type", CV_METHODS)
    def test_halving_random_regression(
        self, regression_dataset: SplitDataset, cv_type: str
    ) -> None:
        _run_tuning_test(regression_dataset, "regression", cv_type, "halving_random")


class TestOptunaCV:
    """Optuna Search — verify each CV method works."""

    @pytest.mark.parametrize("cv_type", CV_METHODS)
    def test_optuna_classification(
        self, classification_dataset: SplitDataset, cv_type: str
    ) -> None:
        try:
            _run_tuning_test(classification_dataset, "classification", cv_type, "optuna")
        except ImportError:
            pytest.skip("Optuna not installed")

    @pytest.mark.parametrize("cv_type", CV_METHODS)
    def test_optuna_regression(self, regression_dataset: SplitDataset, cv_type: str) -> None:
        try:
            _run_tuning_test(regression_dataset, "regression", cv_type, "optuna")
        except ImportError:
            pytest.skip("Optuna not installed")


# =========================================================================
# EDGE CASES
# =========================================================================


class TestCVEdgeCases:
    """Edge cases and special scenarios."""

    def test_progress_callback_called(self, classification_dataset: SplitDataset) -> None:
        """Verify progress_callback is invoked with correct fold numbers."""
        est = _make_classification_estimator()
        progress_calls: list[tuple[int, int]] = []

        def progress_cb(current: int, total: int) -> None:
            progress_calls.append((current, total))

        est.cross_validate(
            classification_dataset,
            "target",
            {},
            n_folds=3,
            cv_type="k_fold",
            progress_callback=progress_cb,
        )
        assert len(progress_calls) == 3
        assert progress_calls == [(1, 3), (2, 3), (3, 3)]

    def test_nested_cv_inner_folds_capped(self, classification_dataset: SplitDataset) -> None:
        """With n_folds=3, inner folds should be min(3, 3-1) = 2."""
        est = _make_classification_estimator()
        result = est.cross_validate(
            classification_dataset,
            "target",
            {},
            n_folds=3,
            cv_type="nested_cv",
        )
        assert result["cv_config"]["inner_folds"] == 2

    def test_nested_cv_inner_folds_for_5_folds(self, classification_dataset: SplitDataset) -> None:
        """With n_folds=5, inner folds should be min(3, 5-1) = 3."""
        est = _make_classification_estimator()
        result = est.cross_validate(
            classification_dataset,
            "target",
            {},
            n_folds=5,
            cv_type="nested_cv",
        )
        assert result["cv_config"]["inner_folds"] == 3

    def test_time_series_missing_column_warns(self, classification_dataset: SplitDataset) -> None:
        """Specifying a nonexistent time column should warn and use row order."""
        est = _make_classification_estimator()
        logs: list[str] = []
        result = est.cross_validate(
            classification_dataset,
            "target",
            {},
            n_folds=3,
            cv_type="time_series_split",
            time_column="nonexistent_column",
            log_callback=logs.append,
        )
        _assert_cv_result(result, n_folds=3, problem_type="classification")
        assert any("not found" in log for log in logs), f"Expected 'not found' warning, got: {logs}"

    def test_cv_2_folds_minimum(self, classification_dataset: SplitDataset) -> None:
        """Minimum 2 folds should work for all methods."""
        est = _make_classification_estimator()
        for cv_type in [
            "k_fold",
            "stratified_k_fold",
            "shuffle_split",
            "time_series_split",
            "nested_cv",
        ]:
            result = est.cross_validate(
                classification_dataset,
                "target",
                {},
                n_folds=2,
                cv_type=cv_type,
            )
            _assert_cv_result(result, n_folds=2, problem_type="classification")
