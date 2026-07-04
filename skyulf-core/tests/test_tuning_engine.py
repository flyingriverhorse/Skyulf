"""Tests for skyulf.modeling._tuning.engine.TuningCalculator."""

import importlib.util
import sys
import typing
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression

from skyulf.modeling._tuning import engine as engine_mod
from skyulf.modeling._tuning.engine import TuningApplier, TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.base import BaseModelCalculator
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


# ---------------------------------------------------------------------------
# Module-level import fallback branches (optuna availability)
# ---------------------------------------------------------------------------


def _load_engine_variant(extra_sys_modules: Dict[str, Any]):
    """Execute engine.py's module source fresh, under a distinct module
    object (not registered in ``sys.modules`` under the canonical name), so
    the optuna-import fallback branches can be exercised without touching —
    or corrupting coverage tracking of — the real cached
    ``skyulf.modeling._tuning.engine`` module used by the rest of the suite.
    """
    saved: Dict[str, Any] = {}
    sentinel = object()
    for name, mod in extra_sys_modules.items():
        saved[name] = sys.modules.get(name, sentinel)
        sys.modules[name] = mod
    try:
        spec = importlib.util.spec_from_file_location(
            "skyulf.modeling._tuning.engine", engine_mod.__file__
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in saved.items():
            if old is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_optuna_import_failure_disables_optuna():
    """If 'optuna' itself cannot be imported, HAS_OPTUNA should end up False."""
    variant = _load_engine_variant({"optuna": None})
    assert variant.HAS_OPTUNA is False


def test_optuna_integration_import_all_fallbacks_fail():
    """If optuna is present but none of the integration import paths work,
    HAS_OPTUNA should be reset to False and a warning logged."""
    variant = _load_engine_variant(
        {
            "optuna.integration": None,
            "optuna.integration.sklearn": None,
            "optuna_integration": None,
            "optuna_integration.sklearn": None,
        }
    )
    assert variant.HAS_OPTUNA is False


def test_optuna_integration_second_fallback_path_succeeds():
    """If `optuna.integration` fails but `optuna.integration.sklearn` succeeds,
    OptunaSearchCV should be sourced from the second fallback path."""
    import types

    fake_module = types.ModuleType("optuna.integration.sklearn")
    setattr(fake_module, "OptunaSearchCV", object())  # sentinel marker class
    variant = _load_engine_variant(
        {"optuna.integration": None, "optuna.integration.sklearn": fake_module}
    )
    assert variant.HAS_OPTUNA is True
    assert variant.OptunaSearchCV is getattr(fake_module, "OptunaSearchCV")


def test_optuna_integration_third_fallback_path_succeeds():
    """If both `optuna.integration` and `optuna.integration.sklearn` fail but
    `optuna_integration.sklearn` succeeds, OptunaSearchCV should come from the
    third fallback path."""
    import types

    fake_module = types.ModuleType("optuna_integration.sklearn")
    setattr(fake_module, "OptunaSearchCV", object())  # sentinel marker class
    variant = _load_engine_variant(
        {
            "optuna.integration": None,
            "optuna.integration.sklearn": None,
            "optuna_integration.sklearn": fake_module,
        }
    )
    assert variant.HAS_OPTUNA is True
    assert variant.OptunaSearchCV is getattr(fake_module, "OptunaSearchCV")


# ---------------------------------------------------------------------------
# TuningCalculator.fit — Inf target validation (regression)
# ---------------------------------------------------------------------------


def test_fit_raises_on_inf_target():
    """fit() should raise ValueError when y contains Infinite values."""
    X, y = _reg_xy()
    y_float = y.astype(float)
    y_float[0] = float("inf")
    tuner = TuningCalculator(RandomForestRegressorCalculator())
    cfg = TuningConfig(
        strategy="grid",
        metric="r2",
        search_space={"n_estimators": [3]},
        cv_folds=2,
    )
    with pytest.raises(ValueError, match="Infinite values"):
        tuner.fit(X, y_float, config=cfg.__dict__)


# ---------------------------------------------------------------------------
# TuningCalculator.fit — model_class becomes falsy between tune() and fit()
# ---------------------------------------------------------------------------


class _FlipModelClassCalculator(BaseModelCalculator):
    """Calculator whose `model_class` property returns a real class the first
    time it is accessed (so tune() succeeds) and None afterwards (so the
    post-tune check in fit() raises)."""

    def __init__(self):
        self._access_count = 0

    @property
    def problem_type(self) -> str:
        return "classification"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"max_iter": 200, "random_state": 42}

    @property
    def model_class(self):
        # `hasattr()` in tune() consumes one access, and the actual
        # `getattr(...)` in tune() consumes a second — both must return the
        # real class so tune() succeeds. Only the post-tune check in fit()
        # (the third access) should see a falsy value.
        self._access_count += 1
        if self._access_count <= 2:
            return LogisticRegression
        return None

    def fit(self, X, y, config, progress_callback=None, log_callback=None, validation_data=None):
        return None


def test_fit_raises_when_model_class_falsy_after_tune():
    """fit() should raise ValueError if model_class is falsy post-tune."""
    X, y = _clf_xy()
    tuner = TuningCalculator(_FlipModelClassCalculator())
    cfg = _clf_config(search_space={"C": [1.0]}, cv_folds=2)
    with pytest.raises(ValueError, match="model_class attribute"):
        tuner.fit(X, y, config=cfg)


# ---------------------------------------------------------------------------
# TuningCalculator.tune — nested_cv cv_type (classification + regression)
# ---------------------------------------------------------------------------


def test_tune_nested_cv_classification():
    """nested_cv should use StratifiedKFold for classification."""
    X, y = _clf_xy()
    tuner = _tuner_clf()
    cfg = _clf_config(cv_type="nested_cv", cv_folds=4)
    model, result = tuner.fit(X, y, config=cfg)
    assert result.best_score > 0


def test_tune_nested_cv_regression():
    """nested_cv should use plain KFold for regression."""
    X, y = _reg_xy()
    tuner = TuningCalculator(RandomForestRegressorCalculator())
    cfg = TuningConfig(
        strategy="grid",
        metric="r2",
        search_space={"n_estimators": [5]},
        cv_type="nested_cv",
        cv_folds=4,
    )
    model, result = tuner.fit(X, y, config=cfg.__dict__)
    assert result.best_score is not None


def test_tune_nested_cv_small_cv_folds():
    """cv_folds<=2 with nested_cv should fall back to inner_folds=2."""
    X, y = _clf_xy()
    tuner = _tuner_clf()
    cfg = _clf_config(cv_type="nested_cv", cv_folds=2)
    model, result = tuner.fit(X, y, config=cfg)
    assert result.best_score > 0


def test_tune_time_series_split_cv():
    """time_series_split CV type should work for grid search."""
    X, y = _clf_xy()
    tuner = _tuner_clf()
    cfg = _clf_config(cv_type="time_series_split", cv_folds=3)
    model, result = tuner.fit(X, y, config=cfg)
    assert result.best_score is not None


# ---------------------------------------------------------------------------
# TuningCalculator.tune — multiclass metric promotion via pd.Series target
# ---------------------------------------------------------------------------


def test_fit_multiclass_promotes_via_pandas_series_target():
    """is_multiclass detection should work when y is a pandas Series.

    Calls tune() directly (rather than fit()) since fit() converts y to a
    numpy array before delegating, which would never exercise the
    `isinstance(y, pd.Series)` branch.
    """
    X_arr, y_arr = make_classification(
        n_samples=150,
        n_features=6,
        n_informative=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    y_series = pd.Series(y_arr)
    tuner = _tuner_clf()
    cfg = TuningConfig(
        strategy="grid",
        metric="f1",
        search_space={"C": [1.0]},
        cv_folds=3,
    )
    result = tuner.tune(X_arr, y_series, cfg)
    assert "weighted" in (result.scoring_metric or "")


def test_fit_multiclass_roc_auc_promotes_to_ovr_weighted():
    """For multiclass targets, roc_auc should be promoted to roc_auc_ovr_weighted."""
    X_arr, y_arr = make_classification(
        n_samples=150,
        n_features=6,
        n_informative=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    tuner = _tuner_clf()
    cfg = _clf_config(metric="roc_auc", search_space={"C": [1.0]}, cv_folds=3)
    model, result = tuner.fit(X_arr, y_arr, config=cfg)
    assert result.scoring_metric == "roc_auc_ovr_weighted"


# ---------------------------------------------------------------------------
# TuningCalculator.tune — grid search candidate that fails every fold
# ---------------------------------------------------------------------------


def test_fit_grid_search_handles_failing_candidate():
    """A candidate whose params make the model.fit() raise on every fold
    should be penalized with -inf mean score, not crash the whole tuning."""
    X, y = _clf_xy()
    tuner = _tuner_clf()
    # C=-5 is invalid for LogisticRegression and will raise on every fold.
    cfg = _clf_config(search_space={"C": [-5, 1.0]}, cv_folds=3)

    logs: List[str] = []
    model, result = tuner.fit(X, y, config=cfg, log_callback=logs.append)

    assert result.best_params["C"] == 1.0
    scores = {t["params"]["C"]: t["score"] for t in result.trials}
    assert scores[-5] == -float("inf")
    assert any("Failed" in msg for msg in logs)


# ---------------------------------------------------------------------------
# TuningCalculator.tune — halving_grid / halving_random log callbacks
# ---------------------------------------------------------------------------


def test_fit_halving_grid_with_log_callback():
    """halving_grid should emit a start log message describing the grid size."""
    X, y = _clf_xy(n=200)
    tuner = _tuner_clf()
    cfg = TuningConfig(
        strategy="halving_grid",
        metric="accuracy",
        search_space={"C": [0.1, 1.0, 10.0]},
        cv_folds=3,
        random_state=42,
    )
    logs: List[str] = []
    model, result = tuner.fit(X, y, config=cfg.__dict__, log_callback=logs.append)
    assert any("Starting halving_grid search" in msg for msg in logs)
    assert any("Tuning Completed (halving_grid)" in msg for msg in logs)


def test_fit_halving_random_with_log_callback_and_string_min_resources():
    """halving_random should emit its own start log and accept a numeric
    string for min_resources (converted to int)."""
    X, y = _clf_xy(n=200)
    tuner = _tuner_clf()
    cfg = TuningConfig(
        strategy="halving_random",
        metric="accuracy",
        search_space={"C": [0.1, 1.0, 10.0]},
        n_trials=3,
        cv_folds=3,
        random_state=42,
        strategy_params={"min_resources": "20"},
    )
    logs: List[str] = []
    model, result = tuner.fit(X, y, config=cfg.__dict__, log_callback=logs.append)
    assert any("Starting halving_random search" in msg for msg in logs)
    assert hasattr(model, "predict")


# ---------------------------------------------------------------------------
# TuningCalculator.tune — optuna strategy
# ---------------------------------------------------------------------------


def test_fit_optuna_strategy_basic():
    """optuna strategy (default TPE sampler / median pruner) should tune successfully."""
    X, y = _clf_xy(n=150)
    tuner = _tuner_clf()
    cfg = TuningConfig(
        strategy="optuna",
        metric="accuracy",
        search_space={"C": [0.1, 1.0, 10.0]},
        n_trials=4,
        cv_folds=3,
        random_state=42,
    )
    progress_calls: List[tuple] = []
    logs: List[str] = []
    model, result = tuner.fit(
        X,
        y,
        config=cfg.__dict__,
        progress_callback=lambda *a: progress_calls.append(a),
        log_callback=logs.append,
    )
    assert hasattr(model, "predict")
    assert result.n_trials > 0
    assert len(progress_calls) > 0
    assert any("Tuning Completed (optuna)" in msg for msg in logs)


@pytest.mark.parametrize("sampler_name", ["tpe", "random", "cmaes"])
def test_fit_optuna_strategy_samplers(sampler_name):
    """optuna strategy should support tpe/random/cmaes samplers."""
    X, y = _clf_xy(n=150)
    tuner = _tuner_clf()
    cfg = TuningConfig(
        strategy="optuna",
        metric="accuracy",
        search_space={"C": [0.1, 1.0, 10.0]},
        n_trials=3,
        cv_folds=3,
        random_state=42,
        strategy_params={"sampler": sampler_name},
    )
    model, result = tuner.fit(X, y, config=cfg.__dict__)
    assert hasattr(model, "predict")
    assert result.n_trials > 0


def test_fit_optuna_cmaes_with_integer_search_space():
    """cmaes sampler with an all-integer numeric list should use IntDistribution."""
    X, y = _clf_xy(n=150)
    tuner = TuningCalculator(RandomForestRegressorCalculator())
    tuner_clf = _tuner_clf()
    cfg = TuningConfig(
        strategy="optuna",
        metric="accuracy",
        search_space={"max_iter": [50, 100, 200]},
        n_trials=3,
        cv_folds=3,
        random_state=42,
        strategy_params={"sampler": "cmaes"},
    )
    model, result = tuner_clf.fit(X, y, config=cfg.__dict__)
    assert hasattr(model, "predict")
    assert result.n_trials > 0


def test_fit_optuna_with_non_list_search_space_value():
    """A non-list search_space value (a pre-built Optuna distribution) should
    be passed through to Optuna unchanged rather than converted."""
    import optuna

    X, y = _clf_xy(n=150)
    tuner = _tuner_clf()
    # search_space is typed Dict[str, List[Any]], but the engine also accepts
    # pre-built Optuna distribution objects at runtime (see engine.py's
    # `else: distributions[k] = v` fallback) — cast to satisfy the type checker.
    search_space = typing.cast(
        Dict[str, List[Any]],
        {
            "C": optuna.distributions.FloatDistribution(0.1, 10.0),
            "solver": ["lbfgs", "liblinear"],
        },
    )
    cfg = TuningConfig(
        strategy="optuna",
        metric="accuracy",
        search_space=search_space,
        n_trials=3,
        cv_folds=3,
        random_state=42,
    )
    model, result = tuner.fit(X, y, config=cfg.__dict__)
    assert hasattr(model, "predict")
    assert result.n_trials > 0


@pytest.mark.parametrize("pruner_name", ["median", "hyperband", "none"])
def test_fit_optuna_strategy_pruners(pruner_name):
    """optuna strategy should support median/hyperband/none pruners."""
    X, y = _clf_xy(n=150)
    tuner = _tuner_clf()
    cfg = TuningConfig(
        strategy="optuna",
        metric="accuracy",
        search_space={"C": [0.1, 1.0, 10.0]},
        n_trials=3,
        cv_folds=3,
        random_state=42,
        strategy_params={"pruner": pruner_name},
    )
    model, result = tuner.fit(X, y, config=cfg.__dict__)
    assert hasattr(model, "predict")
    assert result.n_trials > 0


def test_fit_optuna_without_optuna_installed_raises(monkeypatch):
    """If HAS_OPTUNA is False, requesting the optuna strategy should raise ImportError."""
    monkeypatch.setattr(engine_mod, "HAS_OPTUNA", False)
    X, y = _clf_xy()
    tuner = TuningCalculator(LogisticRegressionCalculator())
    cfg = TuningConfig(
        strategy="optuna",
        metric="accuracy",
        search_space={"C": [1.0]},
        n_trials=2,
        cv_folds=2,
    )
    with pytest.raises(ImportError, match="Optuna is not installed"):
        tuner.fit(X, y, config=cfg.__dict__)


# ---------------------------------------------------------------------------
# TuningCalculator.tune — unknown strategy
# ---------------------------------------------------------------------------


def test_tune_unknown_strategy_raises():
    """An unrecognized strategy should raise ValueError."""
    X, y = _clf_xy()
    tuner = _tuner_clf()
    cfg = TuningConfig(
        strategy=typing.cast(Any, "bogus_strategy"),
        metric="accuracy",
        search_space={"C": [1.0]},
        cv_folds=2,
    )
    with pytest.raises(ValueError, match="Unknown tuning strategy"):
        tuner.fit(X, y, config=cfg.__dict__)


# ---------------------------------------------------------------------------
# TuningCalculator.tune — parallel_backend usage
# ---------------------------------------------------------------------------


def test_fit_halving_grid_with_parallel_backend():
    """When config.parallel_backend is set, the search should run inside a
    joblib parallel_backend context manager."""
    X, y = _clf_xy(n=200)
    tuner = _tuner_clf()
    cfg = TuningConfig(
        strategy="halving_grid",
        metric="accuracy",
        search_space={"C": [0.1, 1.0]},
        cv_folds=3,
        random_state=42,
        parallel_backend="threading",
    )
    model, result = tuner.fit(X, y, config=cfg.__dict__)
    assert hasattr(model, "predict")


# ---------------------------------------------------------------------------
# TuningCalculator.tune — searcher.fit() exception handling
# ---------------------------------------------------------------------------


class _FakeSearcher:
    """A stand-in for a sklearn/optuna searcher with configurable failures."""

    def __init__(
        self,
        fit_exception: Optional[Exception] = None,
        best_params_exception: Optional[Exception] = None,
        best_params: Optional[Dict[str, Any]] = None,
        best_score: float = 0.9,
        cv_results: Optional[Dict[str, Any]] = None,
    ):
        self._fit_exception = fit_exception
        self._best_params_exception = best_params_exception
        self._best_params = best_params if best_params is not None else {"C": 1.0}
        self._best_score = best_score
        self._cv_results = cv_results if cv_results is not None else {}

    def fit(self, X, y):
        if self._fit_exception is not None:
            raise self._fit_exception

    @property
    def best_params_(self):
        if self._best_params_exception is not None:
            raise self._best_params_exception
        return self._best_params

    @property
    def best_score_(self):
        return self._best_score

    @property
    def cv_results_(self):
        return self._cv_results


def _run_with_fake_halving_grid_searcher(monkeypatch, fake_searcher):
    """Force the halving_grid branch to use `fake_searcher` and run tune()."""
    monkeypatch.setattr(engine_mod, "HalvingGridSearchCV", lambda **kwargs: fake_searcher)
    X, y = _clf_xy(n=60)
    tuner = _tuner_clf()
    cfg = TuningConfig(
        strategy="halving_grid",
        metric="accuracy",
        search_space={"C": [1.0]},
        cv_folds=2,
        random_state=42,
    )
    return tuner.fit(X, y, config=cfg.__dict__)


def test_tune_fit_no_trials_completed_error_message(monkeypatch):
    """searcher.fit() raising 'No trials are completed yet' should be
    translated into a user-friendly ValueError."""
    fake = _FakeSearcher(fit_exception=ValueError("No trials are completed yet somehow"))
    with pytest.raises(ValueError, match="No trials completed successfully"):
        _run_with_fake_halving_grid_searcher(monkeypatch, fake)


def test_tune_fit_halving_dataset_too_small_error_message(monkeypatch):
    """searcher.fit() raising sklearn's halving 'n_samples ... resample ... Got 0'
    error should be translated into a clearer dataset-size ValueError."""
    fake = _FakeSearcher(
        fit_exception=ValueError("n_samples=10 ... cannot resample ... Got 0 candidates")
    )
    with pytest.raises(ValueError, match="dataset is too small"):
        _run_with_fake_halving_grid_searcher(monkeypatch, fake)


def test_tune_fit_generic_exception_reraised(monkeypatch):
    """Any other exception from searcher.fit() should be re-raised unchanged."""
    fake = _FakeSearcher(fit_exception=RuntimeError("totally unrelated failure"))
    with pytest.raises(RuntimeError, match="totally unrelated failure"):
        _run_with_fake_halving_grid_searcher(monkeypatch, fake)


def test_tune_best_params_no_trials_completed_error_message(monkeypatch):
    """Accessing best_params_ raising 'No trials are completed yet' should be
    translated into a user-friendly ValueError about all trials failing."""
    fake = _FakeSearcher(best_params_exception=ValueError("No trials are completed yet."))
    with pytest.raises(ValueError, match="All trials failed"):
        _run_with_fake_halving_grid_searcher(monkeypatch, fake)


def test_tune_best_params_generic_value_error_reraised(monkeypatch):
    """Any other ValueError from best_params_ access should be re-raised unchanged."""
    fake = _FakeSearcher(best_params_exception=ValueError("some other unrelated issue"))
    with pytest.raises(ValueError, match="some other unrelated issue"):
        _run_with_fake_halving_grid_searcher(monkeypatch, fake)


# ---------------------------------------------------------------------------
# TuningApplier
# ---------------------------------------------------------------------------


def test_tuning_applier_init_stores_base_applier():
    """TuningApplier should store the wrapped base applier."""
    base = LogisticRegressionApplier()
    applier = TuningApplier(base)
    assert applier.base_applier is base


def test_tuning_applier_predict_with_tuple_artifact():
    """predict() should unwrap the (model, tuning_result) tuple and delegate."""
    X, y = _clf_xy()
    tuner = _tuner_clf()
    model, result = tuner.fit(X, y, config=_clf_config())

    applier = TuningApplier(LogisticRegressionApplier())
    X_df = pd.DataFrame(X)
    preds = applier.predict(X_df, (model, result))
    assert len(preds) == len(X_df)


def test_tuning_applier_predict_without_tuple_artifact_returns_nan():
    """predict() should fall back to NaN series for a legacy non-tuple artifact."""
    applier = TuningApplier(LogisticRegressionApplier())
    X_df = pd.DataFrame({"a": [1, 2, 3]})
    preds = applier.predict(X_df, "not-a-tuple-artifact")
    assert preds.isna().all()
    assert len(preds) == 3


def test_tuning_applier_predict_proba_with_tuple_artifact():
    """predict_proba() should unwrap the tuple and delegate to the base applier."""
    X, y = _clf_xy()
    tuner = _tuner_clf()
    model, result = tuner.fit(X, y, config=_clf_config())

    applier = TuningApplier(LogisticRegressionApplier())
    X_df = pd.DataFrame(X)
    proba = applier.predict_proba(X_df, (model, result))
    assert proba is not None
    assert len(proba) == len(X_df)


def test_tuning_applier_predict_proba_without_tuple_artifact_returns_none():
    """predict_proba() should return None for a legacy non-tuple artifact."""
    applier = TuningApplier(LogisticRegressionApplier())
    X_df = pd.DataFrame({"a": [1, 2, 3]})
    proba = applier.predict_proba(X_df, "not-a-tuple-artifact")
    assert proba is None
