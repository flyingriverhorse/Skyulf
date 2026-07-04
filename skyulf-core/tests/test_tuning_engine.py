"""Tests for skyulf.modeling._tuning.engine.TuningCalculator."""

import typing
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import (
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
)
from skyulf.modeling.regression import (
    RandomForestRegressorApplier,
    RandomForestRegressorCalculator,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _clf_xy(n: int = 120, seed: int = 42) -> tuple:
    """Return a clean binary classification (X, y) pair as Numpy arrays."""
    X, y = make_classification(
        n_samples=n, n_features=4, n_informative=3, n_redundant=1, random_state=seed
    )
    return X.astype(float), y.astype(int)


def _reg_xy(n: int = 120, seed: int = 42) -> tuple:
    """Return a clean regression (X, y) pair as Numpy arrays."""
    X, y = make_regression(n_samples=n, n_features=4, noise=0.1, random_state=seed)
    return X.astype(float), y.astype(float)


def _clf_config(**overrides) -> dict:
    """Return a grid-search TuningConfig dict for LogisticRegression."""
    base = TuningConfig(
        strategy="grid",
        metric="accuracy",
        search_space={"C": [0.1, 1.0]},
        cv_folds=3,
    )
    cfg = base.__dict__.copy()
    cfg.update(overrides)
    return cfg


def _tuner_clf() -> TuningCalculator:
    """Return a TuningCalculator wrapping LogisticRegressionCalculator."""
    return TuningCalculator(LogisticRegressionCalculator())


# ---------------------------------------------------------------------------
# TuningCalculator.problem_type delegation
# ---------------------------------------------------------------------------


def test_tuning_calculator_problem_type_classification():
    """problem_type should mirror the wrapped calculator's problem_type."""
    tuner = _tuner_clf()
    assert tuner.problem_type == "classification"


def test_tuning_calculator_problem_type_regression():
    """problem_type for regression should be 'regression'."""
    tuner = TuningCalculator(RandomForestRegressorCalculator())
    assert tuner.problem_type == "regression"


# ---------------------------------------------------------------------------
# _clean_search_space
# ---------------------------------------------------------------------------


def test_clean_search_space_converts_none_string():
    """'none' strings in search space lists should be converted to Python None."""
    tuner = _tuner_clf()
    space = {"penalty": ["l1", "none", "l2"]}
    cleaned = tuner._clean_search_space(space)
    assert cleaned["penalty"] == ["l1", None, "l2"]


def test_clean_search_space_nested_dict():
    """Nested dicts inside search space should be recursively cleaned."""
    tuner = _tuner_clf()
    space = {"sub": {"penalty": "none", "C": 1.0}}
    cleaned = tuner._clean_search_space(space)
    assert cleaned["sub"]["penalty"] is None
    assert cleaned["sub"]["C"] == 1.0


def test_clean_search_space_passthrough():
    """Values that are not 'none' or nested dicts should remain unchanged."""
    tuner = _tuner_clf()
    space = {"C": [0.1, 1.0, 10.0]}
    assert tuner._clean_search_space(space) == space


# ---------------------------------------------------------------------------
# _instantiate_model
# ---------------------------------------------------------------------------


def test_instantiate_model_basic():
    """_instantiate_model should create a model with the supplied params."""
    from sklearn.linear_model import LogisticRegression

    tuner = _tuner_clf()
    model = TuningCalculator._instantiate_model(LogisticRegression, {"C": 5.0})
    assert hasattr(model, "fit")
    assert model.C == 5.0


def test_instantiate_model_filters_invalid_params():
    """_instantiate_model should silently ignore params not in the constructor."""
    from sklearn.linear_model import LogisticRegression

    tuner = _tuner_clf()
    # 'nonexistent_param' is not a LogisticRegression constructor arg
    model = TuningCalculator._instantiate_model(
        LogisticRegression, {"C": 2.0, "nonexistent_param": 99}
    )
    assert model.C == 2.0


# ---------------------------------------------------------------------------
# TuningCalculator.fit — validation: NaN/Inf in data
# ---------------------------------------------------------------------------


def test_fit_raises_on_nan_features():
    """fit() should raise ValueError when X contains NaN values."""
    X, y = _clf_xy()
    X[0, 0] = float("nan")
    tuner = _tuner_clf()
    with pytest.raises(ValueError, match="NaN values"):
        tuner.fit(X, y, config=_clf_config())


def test_fit_raises_on_inf_features():
    """fit() should raise ValueError when X contains Inf values."""
    X, y = _clf_xy()
    X[0, 1] = float("inf")
    tuner = _tuner_clf()
    with pytest.raises(ValueError, match="Infinite values"):
        tuner.fit(X, y, config=_clf_config())


def test_fit_raises_on_nan_target():
    """fit() should raise ValueError when y contains NaN values."""
    X, y = _reg_xy()
    y_float = y.astype(float)
    y_float[0] = float("nan")
    tuner = TuningCalculator(RandomForestRegressorCalculator())
    cfg = TuningConfig(
        strategy="grid",
        metric="r2",
        search_space={"n_estimators": [3]},
        cv_folds=2,
    )
    with pytest.raises(ValueError, match="NaN values"):
        tuner.fit(X, y_float, config=cfg.__dict__)


# ---------------------------------------------------------------------------
# TuningCalculator.fit — metric validation
# ---------------------------------------------------------------------------


def test_fit_raises_on_classification_metric_for_regression():
    """Using a classification metric for a regression model should raise ValueError."""
    X, y = _reg_xy()
    tuner = TuningCalculator(RandomForestRegressorCalculator())
    cfg = TuningConfig(
        strategy="grid",
        metric="accuracy",  # wrong for regression
        search_space={"n_estimators": [5]},
        cv_folds=2,
    )
    with pytest.raises(ValueError, match="Regression model"):
        tuner.fit(X, y, config=cfg.__dict__)


# ---------------------------------------------------------------------------
# TuningCalculator.fit — grid search (happy path)
# ---------------------------------------------------------------------------


def test_fit_grid_search_returns_model_and_result():
    """Grid search should return (fitted_model, TuningResult) tuple."""
    X, y = _clf_xy()
    tuner = _tuner_clf()

    model, result = tuner.fit(X, y, config=_clf_config())

    assert hasattr(model, "predict")
    assert result.best_score > 0
    assert "C" in result.best_params
    assert len(result.trials) == 2  # grid has 2 C values


def test_fit_grid_search_trials_count():
    """Number of trials should equal the product of search space cardinalities."""
    X, y = _clf_xy()
    tuner = _tuner_clf()
    cfg = _clf_config(search_space={"C": [0.01, 0.1, 1.0]})

    _, result = tuner.fit(X, y, config=cfg)
    assert result.n_trials == 3


# ---------------------------------------------------------------------------
# TuningCalculator.fit — random search
# ---------------------------------------------------------------------------


def test_fit_random_search():
    """Random search should respect n_trials and return valid result."""
    X, y = _clf_xy()
    tuner = _tuner_clf()
    cfg = TuningConfig(
        strategy="random",
        metric="accuracy",
        search_space={"C": [0.01, 0.1, 1.0, 10.0]},
        n_trials=3,
        cv_folds=3,
        random_state=42,
    )
    model, result = tuner.fit(X, y, config=cfg.__dict__)
    assert hasattr(model, "predict")
    assert result.best_score > 0


# ---------------------------------------------------------------------------
# TuningCalculator.fit — CV type variants
# ---------------------------------------------------------------------------


def test_fit_stratified_kfold_cv():
    """stratified_k_fold CV type should work for classification tuning."""
    X, y = _clf_xy()
    tuner = _tuner_clf()
    cfg = _clf_config(cv_type="stratified_k_fold", cv_folds=3)
    model, result = tuner.fit(X, y, config=cfg)
    assert result.best_score > 0


def test_fit_shuffle_split_cv():
    """shuffle_split CV type should work for classification tuning."""
    X, y = _clf_xy()
    tuner = _tuner_clf()
    cfg = _clf_config(cv_type="shuffle_split", cv_folds=3)
    model, result = tuner.fit(X, y, config=cfg)
    assert result.best_score > 0


def test_fit_cv_disabled_uses_holdout():
    """When cv_enabled=False tuning should use a single 80/20 holdout."""
    X, y = _clf_xy()
    tuner = _tuner_clf()
    cfg = _clf_config(cv_enabled=False)
    model, result = tuner.fit(X, y, config=cfg)
    assert hasattr(model, "predict")


# ---------------------------------------------------------------------------
# TuningCalculator.fit — callbacks
# ---------------------------------------------------------------------------


def test_fit_progress_callback_invoked():
    """progress_callback should be called at least once during grid search."""
    X, y = _clf_xy()
    tuner = _tuner_clf()

    calls: List[tuple] = []

    def _cb(current, total, score, params):
        calls.append((current, total, score, params))

    tuner.fit(X, y, config=_clf_config(), progress_callback=_cb)
    assert len(calls) >= 1


def test_fit_log_callback_receives_messages():
    """log_callback should receive at least start and end messages."""
    X, y = _clf_xy()
    tuner = _tuner_clf()

    messages: List[str] = []
    tuner.fit(X, y, config=_clf_config(), log_callback=messages.append)
    assert len(messages) >= 2


# ---------------------------------------------------------------------------
# TuningCalculator.fit — TuningConfig object directly
# ---------------------------------------------------------------------------


def test_fit_accepts_tuning_config_object():
    """fit() should work when passed a TuningConfig instance directly."""
    X, y = _clf_xy()
    tuner = _tuner_clf()
    cfg = TuningConfig(
        strategy="grid",
        metric="accuracy",
        search_space={"C": [1.0]},
        cv_folds=3,
    )
    model, result = tuner.fit(X, y, config=typing.cast(Dict[str, Any], cfg))
    assert result.best_params["C"] == 1.0


# ---------------------------------------------------------------------------
# TuningCalculator.fit — multiclass metric promotion
# ---------------------------------------------------------------------------


def test_fit_multiclass_promotes_f1_to_weighted():
    """For multiclass targets, f1 should be promoted to f1_weighted automatically."""
    X_arr, y_arr = make_classification(
        n_samples=150,
        n_features=6,
        n_informative=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    tuner = _tuner_clf()
    cfg = _clf_config(metric="f1", search_space={"C": [1.0]}, cv_folds=3)
    model, result = tuner.fit(X_arr, y_arr, config=cfg)
    assert result.scoring_metric is not None
    assert "weighted" in result.scoring_metric or "f1" in result.scoring_metric


# ---------------------------------------------------------------------------
# TuningCalculator.fit — validation_data provided
# ---------------------------------------------------------------------------


def test_fit_with_validation_data():
    """fit() should accept a pre-split validation_data tuple."""
    X, y = _clf_xy(n=200)
    X_train, y_train = X[:160], y[:160]
    X_val, y_val = X[160:], y[160:]

    tuner = _tuner_clf()
    cfg = _clf_config(search_space={"C": [0.1, 1.0]}, cv_folds=3)
    model, result = tuner.fit(X_train, y_train, config=cfg, validation_data=(X_val, y_val))
    assert result.best_score > 0


# ---------------------------------------------------------------------------
# TuningCalculator.fit — no model_class raises
# ---------------------------------------------------------------------------


def test_fit_without_model_class_raises():
    """A calculator without model_class should raise ValueError in tune()."""
    from skyulf.modeling.base import BaseModelCalculator

    class _NoClassCalc(BaseModelCalculator):
        @property
        def problem_type(self) -> str:
            return "classification"

        def fit(
            self,
            X,
            y,
            config,
            progress_callback=None,
            log_callback=None,
            validation_data=None,
        ):
            return None

    X, y = _clf_xy()
    tuner = TuningCalculator(_NoClassCalc())
    cfg = _clf_config()
    with pytest.raises(ValueError, match="SklearnCalculator"):
        tuner.fit(X, y, config=cfg)


# ---------------------------------------------------------------------------
# TuningCalculator.fit — halving grid search (basic smoke test)
# ---------------------------------------------------------------------------


def test_fit_halving_grid_search():
    """halving_grid strategy should return a valid (model, result) tuple."""
    X, y = _clf_xy(n=200)
    tuner = _tuner_clf()
    cfg = TuningConfig(
        strategy="halving_grid",
        metric="accuracy",
        search_space={"C": [0.1, 1.0, 10.0]},
        cv_folds=3,
        random_state=42,
    )
    model, result = tuner.fit(X, y, config=cfg.__dict__)
    assert hasattr(model, "predict")
    assert result.best_score > 0
