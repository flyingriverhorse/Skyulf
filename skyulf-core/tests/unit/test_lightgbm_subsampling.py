"""LightGBM row sampling must affect direct fits and cloned tuning candidates."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression

from skyulf.modeling import classification, regression
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.hyperparameters import get_default_search_space

lgb = pytest.importorskip("lightgbm")

pytestmark = pytest.mark.filterwarnings("ignore:.*valid feature names.*:UserWarning")


@pytest.fixture(params=["classification", "regression"])
def lightgbm_case(request):
    """Exercise identical sampling contracts for classification and regression."""
    if request.param == "classification":
        calculator = classification.LGBMClassifierCalculator()
        applier = classification.LGBMClassifierApplier()
        native_class = lgb.LGBMClassifier
        maker = make_classification
        prediction = "predict_proba"
        metric = "neg_log_loss"
    else:
        calculator = regression.LGBMRegressorCalculator()
        applier = regression.LGBMRegressorApplier()
        native_class = lgb.LGBMRegressor
        maker = make_regression
        prediction = "predict"
        metric = "neg_mean_squared_error"
    X, y = maker(n_samples=300, n_features=6, random_state=7)
    params = {"n_estimators": 20, "n_jobs": 1, "random_state": 7}
    return (
        calculator,
        applier,
        native_class,
        pd.DataFrame(X),
        pd.Series(y),
        params,
        prediction,
        metric,
    )


@pytest.mark.parametrize("boosting_type", ["gbdt", "dart"])
def test_lightgbm_subsample_changes_fitted_predictions(lightgbm_case, boosting_type):
    """The advertised fraction must change fitted predictions in bagging-compatible modes."""
    calculator, _, _, X, y, params, prediction, _ = lightgbm_case
    params = {**params, "boosting_type": boosting_type}
    full = calculator.fit(X, y, {"params": {**params, "subsample": 1.0}})
    sampled = calculator.fit(X, y, {"params": {**params, "subsample": 0.4}})

    assert not np.allclose(getattr(full, prediction)(X), getattr(sampled, prediction)(X))


def test_lightgbm_subsample_changes_tuning_scores_and_refitted_predictions(lightgbm_case):
    """Candidate cloning and final refits must retain sampling across all advertised modes."""
    calculator, _, _, X, y, params, prediction, metric = lightgbm_case
    model_key = (
        "lgbm_classifier" if calculator.problem_type == "classification" else "lgbm_regressor"
    )
    advertised_space = get_default_search_space(model_key)
    search_space = {
        **{key: [value] for key, value in params.items()},
        "boosting_type": advertised_space["boosting_type"],
        "subsample": advertised_space["subsample"],
    }
    model, result = TuningCalculator(calculator).fit(
        X,
        y,
        TuningConfig(
            strategy="grid", metric=metric, cv_folds=2, random_state=7, search_space=search_space
        ),
    )
    scores = {
        mode: [
            trial["score"] for trial in result.trials if trial["params"]["boosting_type"] == mode
        ]
        for mode in ["gbdt", "dart", "goss"]
    }
    assert len(result.trials) == 9
    assert all(len(values) == 3 and np.isfinite(values).all() for values in scores.values())
    assert not np.allclose(scores["gbdt"], scores["gbdt"][0])
    assert not np.allclose(scores["dart"], scores["dart"][0])
    np.testing.assert_allclose(scores["goss"], scores["goss"][0])
    direct = calculator.fit(X, y, {"params": result.best_params})
    np.testing.assert_allclose(getattr(model, prediction)(X), getattr(direct, prediction)(X))


@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "halving_random", "optuna"])
def test_lightgbm_tuning_refit_uses_sampled_rows(lightgbm_case, strategy):
    """Every searcher must refit a selected sampling fraction with actual row bagging."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
    calculator, _, native_class, X, y, params, prediction, metric = lightgbm_case
    params = {**params, "subsample": 0.4}
    model, _ = TuningCalculator(calculator).fit(
        X,
        y,
        TuningConfig(
            strategy=strategy,
            n_trials=1,
            metric=metric,
            cv_folds=2,
            random_state=7,
            search_space={key: [value] for key, value in params.items()},
        ),
    )
    native = native_class(**params, subsample_freq=1, verbosity=-1).fit(X.to_numpy(), y.to_numpy())

    np.testing.assert_allclose(getattr(model, prediction)(X), getattr(native, prediction)(X))


@pytest.mark.parametrize(
    "frequency",
    [
        {"subsample_freq": 0},
        {"bagging_freq": 0},
        {"subsample_freq": 3},
        {"bagging_freq": 3},
        {"subsample_freq": 1, "bagging_freq": 0},
        {"subsample_freq": 0, "bagging_freq": 3},
    ],
)
def test_lightgbm_explicit_sampling_frequency_matches_native_training(lightgbm_case, frequency):
    """Explicit zero, positive frequencies, and the native alias must keep their semantics."""
    calculator, _, native_class, X, y, params, prediction, _ = lightgbm_case
    params = {**params, "subsample": 0.4, **frequency}
    actual = calculator.fit(X, y, {"params": params})
    native = native_class(**params, verbosity=-1).fit(X.to_numpy(), y.to_numpy())

    np.testing.assert_allclose(getattr(actual, prediction)(X), getattr(native, prediction)(X))


@pytest.mark.parametrize("frequency", [{"subsample_freq": 0}, {"bagging_freq": 0}])
def test_lightgbm_tuning_preserves_explicit_disabled_bagging(lightgbm_case, frequency):
    """Explicit disabled sampling must survive searcher parameter updates and final refit."""
    calculator, _, native_class, X, y, params, prediction, metric = lightgbm_case
    params = {**params, "subsample": 0.4, **frequency}
    model, _ = TuningCalculator(calculator).fit(
        X,
        y,
        TuningConfig(
            strategy="grid",
            metric=metric,
            cv_folds=2,
            random_state=7,
            search_space={key: [value] for key, value in params.items()},
        ),
    )
    native = native_class(**params, verbosity=-1).fit(X.to_numpy(), y.to_numpy())

    np.testing.assert_allclose(getattr(model, prediction)(X), getattr(native, prediction)(X))


@pytest.mark.parametrize(
    "goss_params",
    [
        {"boosting_type": "goss"},
        {"boosting": "goss"},
        {"boost": "goss"},
        {"data_sample_strategy": "goss"},
        {"boosting_type": "GOSS"},
        {"boosting": "GOSS"},
        {"boost": "GOSS"},
        {"data_sample_strategy": "GOSS"},
    ],
)
def test_lightgbm_automatic_sampling_preserves_goss_training(lightgbm_case, goss_params):
    """GOSS retains gradient sampling for canonical and alias configurations."""
    calculator, _, native_class, X, y, params, prediction, _ = lightgbm_case
    params = {**params, "subsample": 0.4, **goss_params}
    actual = calculator.fit(X, y, {"params": params})
    native = native_class(**params, verbosity=-1).fit(X.to_numpy(), y.to_numpy())

    np.testing.assert_allclose(getattr(actual, prediction)(X), getattr(native, prediction)(X))


def test_lightgbm_clone_can_switch_from_goss_to_bagging(lightgbm_case):
    """A fitted GOSS candidate must not freeze the automatic frequency on later clones."""
    calculator, applier, native_class, X, y, params, prediction, _ = lightgbm_case
    params = {**params, "subsample": 0.4}
    goss = calculator.fit(X, y, {"params": {**params, "boosting_type": "goss"}})
    bagged = clone(goss).set_params(boosting_type="gbdt").fit(X.to_numpy(), y.to_numpy())
    restored = pickle.loads(pickle.dumps(bagged))
    native = native_class(**params, subsample_freq=1, verbosity=-1).fit(X.to_numpy(), y.to_numpy())

    np.testing.assert_allclose(
        getattr(applier, prediction)(X, restored), getattr(native, prediction)(X)
    )


def test_lightgbm_explicit_goss_bagging_conflict_is_not_silently_overridden(lightgbm_case):
    """Automatic defaults must not erase an explicitly incompatible native frequency."""
    calculator, _, _, X, y, params, _, _ = lightgbm_case
    params = {**params, "subsample": 0.4, "boosting_type": "goss", "subsample_freq": 1}

    with pytest.raises(lgb.basic.LightGBMError, match="Cannot use bagging in GOSS"):
        calculator.fit(X, y, {"params": params})


def test_lightgbm_sampling_preserves_tuning_with_missing_features(lightgbm_case):
    """The sampling estimator must retain LightGBM's native missing-feature support."""
    calculator, _, _, X, y, params, prediction, metric = lightgbm_case
    X.iloc[::5, 0] = np.nan
    model, result = TuningCalculator(calculator).fit(
        X,
        y,
        TuningConfig(
            strategy="grid",
            metric=metric,
            cv_folds=2,
            search_space={**{key: [value] for key, value in params.items()}, "subsample": [0.4]},
        ),
    )

    assert np.isfinite(result.best_score)
    assert np.isfinite(getattr(model, prediction)(X)).all()


def test_lightgbm_legacy_fitted_artifact_preserves_predictions(lightgbm_case):
    """Loading a previously fitted native estimator must not enable bagging retroactively."""
    _, applier, native_class, X, y, params, prediction, _ = lightgbm_case
    legacy = native_class(**params, subsample=0.4, verbosity=-1).fit(X.to_numpy(), y.to_numpy())
    restored = pickle.loads(pickle.dumps(legacy))

    np.testing.assert_allclose(
        getattr(applier, prediction)(X, restored), getattr(legacy, prediction)(X)
    )
