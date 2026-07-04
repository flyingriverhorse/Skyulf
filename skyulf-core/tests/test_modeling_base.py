"""Tests for skyulf.modeling.base (BaseModelCalculator, BaseModelApplier, StatefulEstimator)."""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import BaseModelApplier, BaseModelCalculator, StatefulEstimator
from skyulf.modeling.classification import (
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
)
from skyulf.modeling.regression import (
    RandomForestRegressorApplier,
    RandomForestRegressorCalculator,
)

# ---------------------------------------------------------------------------
# Minimal concrete implementations for ABC tests
# ---------------------------------------------------------------------------


class _DummyCalculator(BaseModelCalculator):
    """Minimal calculator that echoes a constant artifact."""

    @property
    def problem_type(self) -> str:
        """Returns classification."""
        return "classification"

    def fit(self, X, y, config, progress_callback=None, log_callback=None, validation_data=None):
        """Return a simple dict as the model artifact."""
        return {"fitted": True, "n_samples": len(X)}


class _DummyApplier(BaseModelApplier):
    """Minimal applier that returns all-zeros predictions."""

    def predict(self, df, model_artifact) -> pd.Series:
        """Return zeros for every row."""
        return pd.Series(np.zeros(len(df), dtype=int))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _classification_dataset() -> tuple:
    """Binary classification split (160 train / 40 test)."""
    X_arr, y_arr = make_classification(
        n_samples=200, n_features=5, n_informative=3, random_state=42
    )
    df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(5)])  # ty: ignore[invalid-argument-type]
    df["target"] = y_arr
    return SplitDataset(train=df.iloc[:160], test=df.iloc[160:], validation=None), df


def _regression_dataset() -> tuple:
    """Regression split (160 train / 40 test)."""
    X_arr, y_arr = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(5)])  # ty: ignore[invalid-argument-type]
    df["target"] = y_arr
    return SplitDataset(train=df.iloc[:160], test=df.iloc[160:], validation=None), df


# ---------------------------------------------------------------------------
# BaseModelCalculator / BaseModelApplier contracts
# ---------------------------------------------------------------------------


def test_base_calculator_abstract_raises():
    """Instantiating BaseModelCalculator directly should raise TypeError."""
    with pytest.raises(TypeError):
        BaseModelCalculator()  # type: ignore[abstract]


def test_base_applier_abstract_raises():
    """Instantiating BaseModelApplier directly should raise TypeError."""
    with pytest.raises(TypeError):
        BaseModelApplier()  # type: ignore[abstract]


def test_dummy_calculator_problem_type():
    """Concrete calculator should return the declared problem_type."""
    calc = _DummyCalculator()
    assert calc.problem_type == "classification"


def test_dummy_calculator_default_params_empty():
    """Default params should be empty dict unless overridden."""
    calc = _DummyCalculator()
    assert calc.default_params == {}


def test_dummy_applier_predict_zeros():
    """Dummy applier should return a zero-filled Series."""
    appl = _DummyApplier()
    X = pd.DataFrame({"a": [1, 2, 3]})
    preds = appl.predict(X, model_artifact=None)
    assert list(preds) == [0, 0, 0]


def test_base_applier_predict_proba_default_none():
    """Default predict_proba should return None."""
    appl = _DummyApplier()
    X = pd.DataFrame({"a": [1]})
    assert appl.predict_proba(X, model_artifact=None) is None


def test_base_calculator_prepare_tuning_params_noop():
    """prepare_tuning_params should return None by default."""
    calc = _DummyCalculator()
    assert calc.prepare_tuning_params({}) is None


def test_base_calculator_build_tuning_search_space_empty():
    """build_tuning_search_space should return empty dict by default."""
    calc = _DummyCalculator()
    assert calc.build_tuning_search_space({}, "grid") == {}


# ---------------------------------------------------------------------------
# StatefulEstimator._extract_xy
# ---------------------------------------------------------------------------


def test_extract_xy_from_dataframe_with_target():
    """_extract_xy should split DataFrame into features and target."""
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="n1"
    )
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
    X, y = estimator._extract_xy(df, "target")
    assert "target" not in X.columns
    assert list(y) == [0, 1]


def test_extract_xy_missing_target_raises():
    """_extract_xy should raise ValueError if target column is absent."""
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="n1"
    )
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="not found in data"):
        estimator._extract_xy(df, "missing_col")


def test_extract_xy_from_tuple_xy():
    """_extract_xy with a (X, y) tuple should return it unchanged."""
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="n1"
    )
    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])
    X_out, y_out = estimator._extract_xy((X, y), "target")
    assert len(X_out) == 2
    assert list(y_out) == [0, 1]


# ---------------------------------------------------------------------------
# StatefulEstimator.fit_predict
# ---------------------------------------------------------------------------


def test_fit_predict_train_test_split():
    """fit_predict should return predictions for both train and test splits."""
    dataset, _ = _classification_dataset()
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e1",
    )
    preds = estimator.fit_predict(dataset, "target", config={})
    assert "train" in preds
    assert "test" in preds
    assert len(preds["train"]) == 160
    assert len(preds["test"]) == 40


def test_fit_predict_stores_model():
    """After fit_predict the model attribute should be populated."""
    dataset, _ = _classification_dataset()
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e2",
    )
    estimator.fit_predict(dataset, "target", config={})
    assert estimator.model is not None


def test_fit_predict_with_validation_split():
    """fit_predict should return validation predictions when validation split is present."""
    dataset, df = _classification_dataset()
    val_df = df.sample(20, random_state=99)
    dataset_with_val = SplitDataset(train=df.iloc[:160], test=df.iloc[160:], validation=val_df)
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e3",
    )
    preds = estimator.fit_predict(dataset_with_val, "target", config={})
    assert "validation" in preds
    assert len(preds["validation"]) == 20


def test_fit_predict_raw_dataframe_input():
    """fit_predict should accept a raw DataFrame (no test split) and return train preds."""
    _, df = _classification_dataset()
    train_only = df.iloc[:80].copy()
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e4",
    )
    preds = estimator.fit_predict(train_only, "target", config={})
    assert "train" in preds
    assert len(preds["train"]) == 80


def test_fit_predict_regression():
    """fit_predict for a regression problem should return numeric predictions."""
    dataset, _ = _regression_dataset()
    estimator = StatefulEstimator(
        calculator=RandomForestRegressorCalculator(),
        applier=RandomForestRegressorApplier(),
        node_id="e5",
    )
    preds = estimator.fit_predict(dataset, "target", config={"params": {"n_estimators": 5}})
    assert "train" in preds
    assert pd.api.types.is_float_dtype(preds["train"]) or pd.api.types.is_numeric_dtype(
        preds["train"]
    )


# ---------------------------------------------------------------------------
# StatefulEstimator.evaluate
# ---------------------------------------------------------------------------


def test_evaluate_before_fit_raises():
    """evaluate() before fit_predict() should raise ValueError."""
    dataset, _ = _classification_dataset()
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e6",
    )
    with pytest.raises(ValueError, match="Model has not been trained"):
        estimator.evaluate(dataset, "target")


def test_evaluate_classification_returns_report():
    """evaluate() on classification should include 'accuracy' in test metrics."""
    dataset, _ = _classification_dataset()
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e7",
    )
    estimator.fit_predict(dataset, "target", config={})
    report = estimator.evaluate(dataset, "target")
    assert report["problem_type"] == "classification"
    assert "accuracy" in report["splits"]["test"].metrics


def test_evaluate_regression_returns_report():
    """evaluate() on regression should include 'mse' in test metrics."""
    dataset, _ = _regression_dataset()
    estimator = StatefulEstimator(
        calculator=RandomForestRegressorCalculator(),
        applier=RandomForestRegressorApplier(),
        node_id="e8",
    )
    estimator.fit_predict(dataset, "target", config={"params": {"n_estimators": 5}})
    report = estimator.evaluate(dataset, "target")
    assert report["problem_type"] == "regression"
    assert "mse" in report["splits"]["test"].metrics


# ---------------------------------------------------------------------------
# StatefulEstimator.refit
# ---------------------------------------------------------------------------


def test_refit_without_validation_is_noop_fit():
    """refit() without a validation split should still train the model."""
    dataset, _ = _classification_dataset()
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e9",
    )
    estimator.refit(dataset, "target", config={})
    assert estimator.model is not None


def test_refit_with_validation_combines_train_val():
    """refit() with validation should train on train+val combined."""
    dataset, df = _classification_dataset()
    val_df = df.sample(20, random_state=77)
    dataset_with_val = SplitDataset(train=df.iloc[:160], test=df.iloc[160:], validation=val_df)
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e10",
    )
    estimator.fit_predict(dataset_with_val, "target", config={})
    estimator.refit(dataset_with_val, "target", config={})
    assert estimator.model is not None


# ---------------------------------------------------------------------------
# StatefulEstimator.cross_validate
# ---------------------------------------------------------------------------


def test_cross_validate_via_stateful_estimator():
    """cross_validate() on StatefulEstimator should return aggregated_metrics."""
    dataset, _ = _classification_dataset()
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e11",
    )
    result = estimator.cross_validate(dataset, "target", config={}, n_folds=3)
    assert "aggregated_metrics" in result
    assert len(result["folds"]) == 3
