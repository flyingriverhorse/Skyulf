"""Modeling module for Skyulf."""

from .base import BaseModelApplier, BaseModelCalculator, StatefulEstimator
from .classification import (
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
    RandomForestClassifierApplier,
    RandomForestClassifierCalculator,
)
from .cross_validation import perform_cross_validation
from .hyperparameters import (
    HyperparameterField,
    get_default_search_space,
    get_hyperparameters,
)
from .regression import (
    RandomForestRegressorApplier,
    RandomForestRegressorCalculator,
    RidgeRegressionApplier,
    RidgeRegressionCalculator,
)
from .sklearn_wrapper import SklearnApplier, SklearnCalculator

__all__ = [
    "BaseModelCalculator",
    "BaseModelApplier",
    "StatefulEstimator",
    "SklearnCalculator",
    "SklearnApplier",
    "LogisticRegressionCalculator",
    "LogisticRegressionApplier",
    "RandomForestClassifierCalculator",
    "RandomForestClassifierApplier",
    "RidgeRegressionCalculator",
    "RidgeRegressionApplier",
    "RandomForestRegressorCalculator",
    "RandomForestRegressorApplier",
    "perform_cross_validation",
    "HyperparameterField",
    "get_hyperparameters",
    "get_default_search_space",
]

# Auto-import any submodule added to this package so its @NodeRegistry.register
# decorators run at import time.  New node files need no __init__.py edits.
import importlib as _importlib
import pkgutil as _pkgutil

for _mi in _pkgutil.iter_modules(__path__, __name__ + "."):  # type: ignore[name-defined]
    _importlib.import_module(_mi.name)
