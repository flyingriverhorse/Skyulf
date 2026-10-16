"""Exact executable identities eligible for evidence-based model admission."""

from importlib import import_module
from typing import Any

import numpy as np
from sklearn._loss.loss import (
    AbsoluteError,
    HalfBinomialLoss,
    HalfMultinomialLoss,
    HalfSquaredError,
    HuberLoss,
    PinballLoss,
)
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, _CalibratedClassifier, _SigmoidCalibration
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor

from ..modeling import classification, ensemble, naive_bayes, regression
from ..registry import NodeRegistry

# Resolve explicitly named Core classes, never registry-provided implementations.
_SPECS = (
    (classification, "adaboost_classifier", "AdaBoostClassifier"),
    (classification, "calibrated_classifier", "CalibratedClassifier"),
    (classification, "gaussian_nb", "GaussianNB"),
    (classification, "gradient_boosting_classifier", "GradientBoostingClassifier"),
    (classification, "hist_gradient_boosting_classifier", "HistGradientBoostingClassifier"),
    (classification, "k_neighbors_classifier", "KNeighborsClassifier"),
    (classification, "lgbm_classifier", "LGBMClassifier"),
    (classification, "sgd_classifier", "SGDClassifier"),
    (classification, "svc", "SVC"),
    (classification, "xgboost_classifier", "XGBClassifier"),
    (naive_bayes, "bernoulli_nb", "BernoulliNB"),
    (naive_bayes, "multinomial_nb", "MultinomialNB"),
    (regression, "adaboost_regressor", "AdaBoostRegressor"),
    (regression, "elasticnet_regression", "ElasticNetRegression"),
    (regression, "gradient_boosting_regressor", "GradientBoostingRegressor"),
    (regression, "hist_gradient_boosting_regressor", "HistGradientBoostingRegressor"),
    (regression, "k_neighbors_regressor", "KNeighborsRegressor"),
    (regression, "lasso_regression", "LassoRegression"),
    (regression, "lgbm_regressor", "LGBMRegressor"),
    (regression, "ridge_regression", "RidgeRegression"),
    (regression, "svr", "SVR"),
    (ensemble, "voting_classifier", "VotingClassifier"),
    (ensemble, "voting_regressor", "VotingRegressor"),
    (ensemble, "stacking_classifier", "StackingClassifier"),
    (ensemble, "stacking_regressor", "StackingRegressor"),
)


def _contracts() -> dict[type, tuple[str, type, type]]:
    """Include optional implementations only when their Core classes are installed."""
    result = {}
    for module, node, stem in _SPECS:
        calculator = getattr(module, stem + "Calculator", None)
        if calculator is not None:
            result[calculator().model_class] = (node, calculator, getattr(module, stem + "Applier"))
    return result


CONTRACTS = _contracts()


def _boosters() -> set[type]:
    """Pin native booster helpers only for installed optional model implementations."""
    result = set()
    for module, node in (("lightgbm", "lgbm_classifier"), ("xgboost", "xgboost_classifier")):
        if node in {item[0] for item in CONTRACTS.values()}:
            result.add(import_module(module).Booster)
    return result


_BOOSTERS = _boosters()
_REVIEWED = (
    (classification, "logistic_regression", "LogisticRegression"),
    (regression, "linear_regression", "LinearRegression"),
    (regression, "xgboost_regressor", "XGBRegressor"),
)
_LOSSES = {
    HalfBinomialLoss,
    HalfMultinomialLoss,
    HalfSquaredError,
    AbsoluteError,
    PinballLoss,
    HuberLoss,
}


def class_name(value: Any) -> str:
    """Record the concrete executable class, including Core's fitted wrappers."""
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _member_contracts() -> dict[type, tuple[str, type, type]]:
    """Reuse reviewed tree identities and allow plain sklearn calibration in ensembles."""
    from ._partition_trees import _CONTRACTS  # noqa: PLC0415

    result = {**CONTRACTS, **_CONTRACTS}
    for module, node, stem in _REVIEWED:
        calculator = getattr(module, stem + "Calculator", None)
        if calculator is not None:
            result[calculator().model_class] = (node, calculator, getattr(module, stem + "Applier"))
    result[CalibratedClassifierCV] = CONTRACTS[classification._SeededCalibratedClassifierCV]
    return result


def _registration(contract: tuple[str, type, type]) -> None:
    """Reject registration replacement without executing the replacement."""
    node, calculator, applier = contract
    if (
        NodeRegistry.get_calculator(node) is not calculator
        or NodeRegistry.get_applier(node) is not applier
    ):
        raise ValueError("Unreviewed model registration.")


def inspect_model(model: Any) -> list[str]:
    """Check every nested fitted estimator and inference helper without predicting."""
    classes: list[str] = []
    _inspect_value(model, _member_contracts(), classes, set())
    return classes


def _check_callback(value: Any) -> None:
    """Permit sklearn's known loss objects while rejecting callbacks at any nesting depth."""
    if callable(value) and type(value) not in _LOSSES:
        raise ValueError("Nested model inference callbacks are unsupported.")


def _inspect_value(value: Any, contracts: dict, classes: list[str], seen: set[int]) -> None:
    """Walk estimator state, containers and object arrays, rejecting executable overrides."""
    if id(value) in seen:
        return
    seen.add(id(value))
    _check_callback(value)
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = value
    elif isinstance(value, np.ndarray) and value.dtype.hasobject:
        children = value.flat
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        _inspect_object(value, contracts, classes)
        children = vars(value).values()
    else:
        return
    for child in children:
        _inspect_value(child, contracts, classes, seen)


def _inspect_object(value: Any, contracts: dict, classes: list[str]) -> None:
    """Pin model and sklearn helper types used by composed inference paths."""
    helpers = {
        DummyClassifier,
        DummyRegressor,
        LabelEncoder,
        _CalibratedClassifier,
        _SigmoidCalibration,
        IsotonicRegression,
        TreePredictor,
        _BinMapper,
        ExtraTreeClassifier,
        ExtraTreeRegressor,
        *_LOSSES,
        *_BOOSTERS,
    }
    contract = contracts.get(type(value))
    executable = _executable(value)
    if executable and contract is None and type(value) not in helpers:
        raise ValueError(f"Unaccepted nested estimator class: {class_name(value)}.")
    if contract is not None:
        _registration(contract)
        classes.append(class_name(value))
        _check_composition(value, contracts)
    if executable and _overridden(value):
        raise ValueError("Overridden model inference methods or callbacks are unsupported.")


def _overridden(value: Any) -> bool:
    """Allow only sklearn's exact loss state, never a loss substituted for a method."""
    return any(
        callable(getattr(type(value), name, None))
        or (callable(item) and not (name == "_loss" and type(item) in _LOSSES))
        for name, item in vars(value).items()
    )


def _check_composition(model: Any, contracts: dict) -> None:
    """Require admitted model identities in every composition slot, not helper types."""
    composed = {
        ensemble.VotingClassifier,
        ensemble.VotingRegressor,
        ensemble.StackingClassifier,
        ensemble.StackingRegressor,
    }
    if type(model) in composed and hasattr(model, "estimators_"):
        children = list(model.estimators_)
        if hasattr(model, "final_estimator_"):
            children.append(model.final_estimator_)
    elif type(model) in {CalibratedClassifierCV, classification._SeededCalibratedClassifierCV}:
        children = [child.estimator for child in getattr(model, "calibrated_classifiers_", [])]
    else:
        return
    for child in children:
        if type(child) not in contracts:
            raise ValueError(f"Unaccepted nested estimator class: {class_name(child)}.")


def _executable(value: Any) -> bool:
    """Recognize inference-bearing objects, including sklearn private helper classes."""
    return isinstance(value, BaseEstimator) or any(
        hasattr(value, name) for name in ("predict", "predict_proba", "decision_function")
    )


def model_contract(model: Any, applier: Any) -> tuple[str, list[str]]:
    """Require an exact eligible model and its unchanged Core prediction wrapper."""
    contract = CONTRACTS.get(type(model))
    if contract is None or type(applier) is not contract[2] or vars(applier):
        raise ValueError("Model requires an eligible exact class and unchanged applier.")
    _registration(contract)
    return contract[0], inspect_model(model)
