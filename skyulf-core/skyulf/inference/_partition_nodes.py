"""Explicit built-in identities and fitted XGBoost model admission.

Preprocessing state and configuration are validated by the node itself.
"""

import json
from importlib import import_module
from typing import Any

from ..core.portable_state import _normalize
from ..preprocessing.encoding.one_hot import OneHotEncoderApplier, OneHotEncoderCalculator
from ..preprocessing.feature_generation.interaction import (
    FeatureInteractionApplier,
    FeatureInteractionCalculator,
)
from ..preprocessing.imputation.group import GroupImputerApplier, GroupImputerCalculator
from ..preprocessing.outliers.clip_values import (
    ClipValuesApplier,
    ClipValuesCalculator,
)
from ..preprocessing.scaling.minmax import MinMaxScalerApplier, MinMaxScalerCalculator
from ..registry import NodeRegistry

APPLIERS = {
    "FeatureInteraction": FeatureInteractionApplier,
    "ClipValues": ClipValuesApplier,
    "GroupImputer": GroupImputerApplier,
    "OneHotEncoder": OneHotEncoderApplier,
    "MinMaxScaler": MinMaxScalerApplier,
}
CALCULATORS = {
    "FeatureInteraction": FeatureInteractionCalculator,
    "ClipValues": ClipValuesCalculator,
    "GroupImputer": GroupImputerCalculator,
    "OneHotEncoder": OneHotEncoderCalculator,
    "MinMaxScaler": MinMaxScalerCalculator,
}


def xgboost_applier(model: Any) -> type | None:
    """Lazily admit exact CPU gbtree regressors and their registered built-in applier."""
    if type(model).__module__ != "xgboost.sklearn" or type(model).__name__ != "XGBRegressor":
        return None
    xgb = import_module("xgboost")
    regression = import_module("skyulf.modeling.regression")
    if type(model) is not xgb.XGBRegressor:
        raise ValueError("Expected exact XGBRegressor.")
    if NodeRegistry.get_calculator("xgboost_regressor") is not regression.XGBRegressorCalculator:
        raise ValueError("Unreviewed XGBoost calculator registration.")
    applier = regression.XGBRegressorApplier
    if NodeRegistry.get_applier("xgboost_regressor") is not applier:
        raise ValueError("Unreviewed XGBoost applier registration.")
    _check_xgboost_state(model, xgb.Booster)
    return applier


def _check_xgboost_state(model: Any, booster_type: type) -> None:
    """Reject custom objectives/callbacks, DART and substituted native booster wrappers."""
    if any(callable(value) for value in vars(model).values()):
        raise ValueError("Overridden XGBoost inference methods are unsupported.")
    if model.callbacks is not None or model.booster not in (None, "gbtree"):
        raise ValueError("XGBoost callbacks and non-gbtree boosters are unsupported.")
    if model.objective not in ("reg:squarederror", "reg:logistic"):
        raise ValueError("Only squared-error and logistic regression objectives are admitted.")
    _normalize(getattr(model, "kwargs", {}))
    _check_booster(model, booster_type)


def _check_booster(model: Any, booster_type: type) -> None:
    """Bind the trusted estimator wrapper to its actual native tree/objective payload."""
    booster = model.get_booster()
    if type(booster) is not booster_type or any(
        callable(value) for value in vars(booster).values()
    ):
        raise ValueError("Overridden XGBoost booster methods are unsupported.")
    learner = json.loads(booster.save_config())["learner"]
    if learner["gradient_booster"]["name"] != "gbtree":
        raise ValueError("Only fitted gbtree boosters are admitted.")
    if learner["objective"]["name"] != model.objective:
        raise ValueError("Fitted XGBoost objective disagrees with estimator configuration.")
