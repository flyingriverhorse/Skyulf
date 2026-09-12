"""Regression coverage for the calibrated classifier's advertised seed control."""

import logging
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import CalibratedClassifierCalculator
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter


@pytest.fixture
def calibration_data() -> tuple[pd.DataFrame, pd.Series]:
    """Keep folds small while exposing stochastic differences in forest probabilities."""
    X, y = make_classification(
        n_samples=96, n_features=6, n_informative=4, flip_y=0.15, random_state=11
    )
    return pd.DataFrame(X), pd.Series(y)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "base_estimator",
    ["logistic_regression", "random_forest", "gradient_boosting", "decision_tree", "svc"],
)
def test_configured_seed_reaches_every_fitted_base_estimator(
    calibration_data, base_estimator, nested, caplog
):
    """Both accepted payload shapes must seed all fitted calibration folds without dropping it."""
    X, y = calibration_data
    params = {"base_estimator": base_estimator, "random_state": 7, "cv": 3}
    config = {"params": params} if nested else params
    with caplog.at_level(logging.WARNING, logger="skyulf.modeling"):
        model = CalibratedClassifierCalculator().fit(X, y, config)

    assert isinstance(model, CalibratedClassifierCV)
    assert model.cv == 3
    assert len(model.calibrated_classifiers_) == 3
    assert {fold.estimator.random_state for fold in model.calibrated_classifiers_} == {7}
    assert params == {"base_estimator": base_estimator, "random_state": 7, "cv": 3}
    assert not caplog.records


@pytest.mark.parametrize(
    "seed_params,expected", [({}, 42), ({"random_state": 0}, 0), ({"random_state": None}, None)]
)
def test_default_zero_and_unseeded_configuration(calibration_data, seed_params, expected):
    """Missing seeds retain reproducibility while explicit zero and None remain authoritative."""
    X, y = calibration_data
    model = CalibratedClassifierCalculator().fit(
        X, y, {"params": {"base_estimator": "decision_tree", "cv": 3, **seed_params}}
    )
    assert {fold.estimator.random_state for fold in model.calibrated_classifiers_} == {expected}


def test_implicit_default_base_estimator_receives_seed(calibration_data):
    """A payload may set only the seed and rely on the default logistic-regression base."""
    X, y = calibration_data
    model = CalibratedClassifierCalculator().fit(X, y, {"params": {"random_state": 19}})
    assert model.cv == 5
    assert {fold.estimator.random_state for fold in model.calibrated_classifiers_} == {19}


def test_clone_and_set_params_seed_without_mutating_supplied_estimator(calibration_data):
    """Sklearn search clones must apply late seed overrides without changing caller-owned models."""
    X, y = calibration_data
    supplied = DecisionTreeClassifier(random_state=97)
    calculator = CalibratedClassifierCalculator()
    model = calculator.model_class(estimator=supplied, cv=3, random_state=7)
    candidate = clone(model).set_params(random_state=17)
    candidate.fit(X.to_numpy(), y.to_numpy())
    model.fit(X.to_numpy(), y.to_numpy())

    assert supplied.random_state == 97
    assert not hasattr(supplied, "tree_")
    assert candidate.get_params()["random_state"] == 17
    assert {fold.estimator.random_state for fold in candidate.calibrated_classifiers_} == {17}
    assert {fold.estimator.random_state for fold in model.calibrated_classifiers_} == {7}


def test_failed_fit_preserves_supplied_estimator(calibration_data):
    """Rejected calibration settings must leave the caller's base estimator untouched."""
    X, y = calibration_data
    supplied = DecisionTreeClassifier(random_state=97)
    model = CalibratedClassifierCalculator().model_class(
        estimator=supplied, method="invalid", random_state=7
    )
    with pytest.raises(ValueError, match="method"):
        model.fit(X.to_numpy(), y.to_numpy())

    assert supplied.random_state == 97
    assert not hasattr(supplied, "tree_")


def test_direct_constructor_without_estimator_preserves_sklearn_default(calibration_data):
    """Direct estimator construction must still resolve sklearn's default base and seed it."""
    X, y = calibration_data
    model = CalibratedClassifierCalculator().model_class(cv=3, random_state=19)
    model.fit(X.to_numpy(), y.to_numpy())

    assert all(isinstance(fold.estimator, LinearSVC) for fold in model.calibrated_classifiers_)
    assert {fold.estimator.random_state for fold in model.calibrated_classifiers_} == {19}


def test_fitted_calibration_pickle_round_trip(calibration_data):
    """Persisting the seeded estimator must preserve fitted probabilities and fold seeds."""
    X, y = calibration_data
    model = CalibratedClassifierCalculator().fit(
        X, y, {"params": {"base_estimator": "decision_tree", "cv": 3, "random_state": 19}}
    )
    restored = pickle.loads(pickle.dumps(model))

    np.testing.assert_array_equal(
        model.predict_proba(X.to_numpy()), restored.predict_proba(X.to_numpy())
    )
    assert {fold.estimator.random_state for fold in restored.calibrated_classifiers_} == {19}


def test_random_forest_seed_controls_repeatable_predictions(calibration_data):
    """A real stochastic fit must repeat for one seed and respond when that seed changes."""
    X, y = calibration_data
    predictions = []
    for seed in [7, 7, 19]:
        model = CalibratedClassifierCalculator().fit(
            X, y, {"params": {"base_estimator": "random_forest", "cv": 3, "random_state": seed}}
        )
        predictions.append(model.predict_proba(X.to_numpy()))

    np.testing.assert_array_equal(predictions[0], predictions[1])
    assert not np.allclose(predictions[0], predictions[2])


def test_deterministic_base_accepts_seed_without_changing_calibration_splits(
    calibration_data, caplog
):
    """GaussianNB has no seed parameter and must retain the existing unshuffled integer folds."""
    X, y = calibration_data
    reference = CalibratedClassifierCV(estimator=GaussianNB(), cv=3).fit(X.to_numpy(), y.to_numpy())
    with caplog.at_level(logging.WARNING, logger="skyulf.modeling"):
        model = CalibratedClassifierCalculator().fit(
            X, y, {"params": {"base_estimator": "gaussian_nb", "cv": 3, "random_state": 7}}
        )

    assert model.cv == 3
    assert all(isinstance(fold.estimator, GaussianNB) for fold in model.calibrated_classifiers_)
    np.testing.assert_array_equal(
        model.predict_proba(X.to_numpy()), reference.predict_proba(X.to_numpy())
    )
    assert not caplog.records


def test_tuning_seed_reaches_search_clones_and_final_refit(calibration_data, caplog):
    """The tuning path constructs estimators directly and must preserve the seed through refit."""
    X, y = calibration_data
    calculator = CalibratedClassifierCalculator()
    calculator.prepare_tuning_params({"base_estimator": "decision_tree"})
    config = TuningConfig(
        strategy="grid",
        search_space={"method": ["sigmoid", "isotonic"], "cv": [3]},
        cv_folds=2,
        random_state=19,
    )
    with caplog.at_level(logging.WARNING, logger="skyulf.modeling"):
        model, result = TuningCalculator(calculator).fit(X, y, config.__dict__)

    assert result.n_trials == 2
    assert np.isfinite(result.best_score)
    assert {fold.estimator.random_state for fold in model.calibrated_classifiers_} == {19}
    assert not caplog.records


@pytest.mark.parametrize("preprocess", [False, True])
@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "halving_random", "optuna"])
def test_search_space_base_selection_reaches_trials_and_refit(
    calibration_data, monkeypatch, strategy, preprocess
):
    """Canvas base selections must reach real trial fits, including nested fold pipelines."""
    X, y = calibration_data
    X = X.rename(columns=lambda col: f"x{col}")
    fitted_bases = []
    original_fit = CalibratedClassifierCV.fit

    def observe_fit(self, X, y, **kwargs):
        """Record the actual fitted fold models while retaining sklearn's real training."""
        fitted = original_fit(self, X, y, **kwargs)
        fitted_bases.extend(fold.estimator for fold in fitted.calibrated_classifiers_)
        return fitted

    monkeypatch.setattr(CalibratedClassifierCV, "fit", observe_fit)
    search_space = {"base_estimator": ["random_forest"], "method": ["isotonic"], "cv": [3]}
    original_space = deepcopy(search_space)
    config = TuningConfig(
        strategy=strategy,
        n_trials=1,
        cv_folds=2,
        random_state=7,
        search_space=search_space,
        strategy_params={"min_resources": 48, "factor": 2},
    )
    adapter = (
        FeatureEngineerFoldAdapter(
            [
                {
                    "name": "scale",
                    "transformer": "StandardScaler",
                    "params": {"columns": list(X.columns)},
                }
            ],
            target_column="target",
        )
        if preprocess
        else None
    )
    logs = []
    model, result = TuningCalculator(CalibratedClassifierCalculator()).fit(
        X, y, config, preprocessing=adapter, log_callback=logs.append
    )

    assert np.isfinite(result.best_score)
    assert result.best_params == {"base_estimator": "random_forest", "method": "isotonic", "cv": 3}
    assert len(fitted_bases) > len(model.calibrated_classifiers_)
    assert all(isinstance(base, RandomForestClassifier) for base in fitted_bases)
    assert {base.random_state for base in fitted_bases} == {7}
    assert all(np.isfinite(trial["score"]) for trial in result.trials)
    assert all(trial["params"]["base_estimator"] == "random_forest" for trial in result.trials)
    assert search_space == original_space
    if preprocess and strategy in {"halving_grid", "halving_random", "optuna"}:
        assert any("fold-aware estimator" in message for message in logs)
    assert model.get_params()["base_estimator"] == "random_forest"


def test_base_estimator_search_compares_candidates_and_refits_winner(calibration_data, monkeypatch):
    """A multiselect search must train both requested base families and retain the winner."""
    X, y = calibration_data
    fitted_types = set()
    original_fit = CalibratedClassifierCV.fit

    def observe_fit(self, X, y, **kwargs):
        """Observe actual candidate fits without substituting their scores or results."""
        fitted = original_fit(self, X, y, **kwargs)
        fitted_types.update(type(fold.estimator) for fold in fitted.calibrated_classifiers_)
        return fitted

    monkeypatch.setattr(CalibratedClassifierCV, "fit", observe_fit)
    model, result = TuningCalculator(CalibratedClassifierCalculator()).fit(
        X,
        y,
        TuningConfig(
            strategy="grid",
            cv_folds=2,
            random_state=19,
            search_space={"base_estimator": ["random_forest", "gaussian_nb"], "cv": [3]},
        ),
    )
    expected_type = {"random_forest": RandomForestClassifier, "gaussian_nb": GaussianNB}[
        result.best_params["base_estimator"]
    ]

    assert result.n_trials == 2
    assert fitted_types == {RandomForestClassifier, GaussianNB}
    assert all(isinstance(fold.estimator, expected_type) for fold in model.calibrated_classifiers_)
    assert model.get_params()["base_estimator"] == result.best_params["base_estimator"]


def test_base_selection_is_cloneable_and_can_change_between_fits(calibration_data):
    """The symbolic choice must override stale estimators after cloning or changing candidates."""
    X, y = calibration_data
    supplied = DecisionTreeClassifier(random_state=97)
    estimator_class = CalibratedClassifierCalculator().model_class
    model = estimator_class(
        estimator=supplied, base_estimator="random_forest", cv=3, random_state=7
    )
    model.fit(X.to_numpy(), y.to_numpy())
    changed = clone(model).set_params(base_estimator="gaussian_nb")
    changed.fit(X.to_numpy(), y.to_numpy())
    restored = pickle.loads(pickle.dumps(model))

    assert all(
        isinstance(fold.estimator, RandomForestClassifier) for fold in model.calibrated_classifiers_
    )
    assert all(isinstance(fold.estimator, GaussianNB) for fold in changed.calibrated_classifiers_)
    assert supplied.random_state == 97
    assert not hasattr(supplied, "tree_")
    assert restored.get_params()["base_estimator"] == "random_forest"
    np.testing.assert_array_equal(
        model.predict_proba(X.to_numpy()), restored.predict_proba(X.to_numpy())
    )
