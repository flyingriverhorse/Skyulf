"""LightGBM sampling must survive ensemble construction, cloning, and refits."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression

lgb = pytest.importorskip("lightgbm")

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.ensemble import (
    StackingClassifierCalculator,
    StackingRegressorCalculator,
    VotingClassifierCalculator,
    VotingRegressorCalculator,
)

pytestmark = pytest.mark.filterwarnings("ignore:.*valid feature names.*:UserWarning")


@pytest.fixture(
    params=[
        VotingClassifierCalculator,
        StackingClassifierCalculator,
        VotingRegressorCalculator,
        StackingRegressorCalculator,
    ],
    ids=["voting-classifier", "stacking-classifier", "voting-regressor", "stacking-regressor"],
)
def ensemble_case(request):
    """Exercise the same sampling contract across all four ensemble calculators."""
    calculator = request.param()
    classification = calculator.problem_type == "classification"
    maker = make_classification if classification else make_regression
    values, labels = maker(n_samples=300, n_features=6, random_state=7)
    params = {"n_estimators": 20, "n_jobs": 1, "random_state": 7}
    config = {
        "base_estimators": ["lightgbm"],
        "base_estimator_params": {"lightgbm": params},
        "cv": 2,
        "n_jobs": 1,
    }
    return (
        calculator,
        pd.DataFrame(values),
        pd.Series(labels),
        config,
        "predict_proba" if classification else "predict",
        lgb.LGBMClassifier if classification else lgb.LGBMRegressor,
    )


@pytest.mark.parametrize("boosting_type", ["gbdt", "dart"])
def test_ensemble_lightgbm_subsample_changes_predictions(ensemble_case, boosting_type):
    """Base learners must actually sample rows when only the fraction is configured."""
    calculator, X, y, config, prediction, native_class = ensemble_case
    common = {**config, "lightgbm__boosting_type": boosting_type}
    full = calculator.fit(X, y, {"params": {**common, "lightgbm__subsample": 1.0}})
    sampled = calculator.fit(X, y, {"params": {**common, "lightgbm__subsample": 0.4}})
    native = native_class(
        **config["base_estimator_params"]["lightgbm"],
        boosting_type=boosting_type,
        subsample=0.4,
        subsample_freq=1,
        verbose=-1,
    ).fit(X.to_numpy(), y.to_numpy())

    assert not np.allclose(getattr(full, prediction)(X), getattr(sampled, prediction)(X))
    np.testing.assert_allclose(
        getattr(sampled.named_estimators_["lightgbm"], prediction)(X),
        getattr(native, prediction)(X),
    )


@pytest.mark.parametrize(
    "frequency",
    [{"subsample_freq": 0}, {"subsample_freq": 3}, {"bagging_freq": 0}, {"bagging_freq": 3}],
)
def test_ensemble_lightgbm_preserves_explicit_frequency(ensemble_case, frequency):
    """User frequencies and native aliases must override automatic ensemble sampling."""
    calculator, X, y, config, prediction, native_class = ensemble_case
    params = {**config["base_estimator_params"]["lightgbm"], "subsample": 0.4, **frequency}
    actual = calculator.fit(X, y, {**config, "base_estimator_params": {"lightgbm": params}})
    native = native_class(**params, verbose=-1).fit(X.to_numpy(), y.to_numpy())

    np.testing.assert_allclose(
        getattr(actual.named_estimators_["lightgbm"], prediction)(X),
        getattr(native, prediction)(X),
    )


def test_ensemble_lightgbm_clone_can_switch_from_goss_to_bagging(ensemble_case):
    """Cloning a GOSS ensemble must retain automatic sampling for a later GBDT fit."""
    calculator, X, y, config, prediction, native_class = ensemble_case
    goss = calculator.fit(
        X, y, {**config, "lightgbm__boosting_type": "goss", "lightgbm__subsample": 0.4}
    )
    bagged = clone(goss).set_params(lightgbm__boosting_type="gbdt").fit(X.to_numpy(), y.to_numpy())
    native = native_class(
        **config["base_estimator_params"]["lightgbm"],
        subsample=0.4,
        subsample_freq=1,
        verbose=-1,
    ).fit(X.to_numpy(), y.to_numpy())

    np.testing.assert_allclose(
        getattr(bagged.named_estimators_["lightgbm"], prediction)(X),
        getattr(native, prediction)(X),
    )


def test_ensemble_lightgbm_tuning_and_refit_sample_rows(ensemble_case):
    """Nested tuning candidates and post-tuning calculator refits must preserve bagging."""
    calculator, X, y, config, prediction, native_class = ensemble_case
    calculator.prepare_tuning_params(config)
    model, result = TuningCalculator(calculator).fit(
        X,
        y,
        TuningConfig(
            strategy="grid",
            metric="neg_log_loss" if prediction == "predict_proba" else "neg_mean_squared_error",
            cv_folds=2,
            search_space={"lightgbm__subsample": [0.4]},
        ),
    )
    refitted = calculator.fit(X, y, {"params": result.best_params})
    native = native_class(
        **config["base_estimator_params"]["lightgbm"],
        subsample=0.4,
        subsample_freq=1,
        verbose=-1,
    ).fit(X.to_numpy(), y.to_numpy())

    np.testing.assert_allclose(
        getattr(model.named_estimators_["lightgbm"], prediction)(X),
        getattr(native, prediction)(X),
    )
    np.testing.assert_allclose(getattr(refitted, prediction)(X), getattr(model, prediction)(X))


@pytest.mark.parametrize(
    "calculator_type", [StackingClassifierCalculator, StackingRegressorCalculator]
)
def test_stacking_lightgbm_final_estimator_samples_rows(calculator_type):
    """A LightGBM meta-learner must honor its configured subsample fraction too."""
    calculator = calculator_type()
    classification = calculator.problem_type == "classification"
    maker = make_classification if classification else make_regression
    values, labels = maker(n_samples=300, n_features=6, random_state=7)
    X, y = pd.DataFrame(values), pd.Series(labels)
    config = {
        "base_estimators": ["random_forest"],
        "base_estimator_params": {"random_forest": {"n_estimators": 10, "n_jobs": 1}},
        "final_estimator": "lightgbm",
        "final_estimator_params": {"n_estimators": 20, "n_jobs": 1, "random_state": 7},
        "cv": 2,
        "n_jobs": 1,
    }
    full = calculator.fit(X, y, {**config, "final_estimator__subsample": 1.0})
    sampled = calculator.fit(X, y, {**config, "final_estimator__subsample": 0.4})
    prediction = "predict_proba" if classification else "predict"

    assert not np.allclose(getattr(full, prediction)(X), getattr(sampled, prediction)(X))
