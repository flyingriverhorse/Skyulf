"""Modeling module for Skyulf."""

from .base import BaseModelApplier, BaseModelCalculator, StatefulEstimator
from .classification import (
    CalibratedClassifierApplier,
    CalibratedClassifierCalculator,
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
    RandomForestClassifierApplier,
    RandomForestClassifierCalculator,
)
from .cross_validation import perform_cross_validation
from .ensemble import (
    StackingClassifierApplier,
    StackingClassifierCalculator,
    StackingRegressorApplier,
    StackingRegressorCalculator,
    VotingClassifierApplier,
    VotingClassifierCalculator,
    VotingRegressorApplier,
    VotingRegressorCalculator,
)
from .hyperparameters import (
    HyperparameterField,
    get_default_search_space,
    get_hyperparameters,
)
from .naive_bayes import (
    BernoulliNBApplier,
    BernoulliNBCalculator,
    MultinomialNBApplier,
    MultinomialNBCalculator,
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
    "CalibratedClassifierCalculator",
    "CalibratedClassifierApplier",
    "RandomForestClassifierCalculator",
    "RandomForestClassifierApplier",
    "RidgeRegressionCalculator",
    "RidgeRegressionApplier",
    "RandomForestRegressorCalculator",
    "RandomForestRegressorApplier",
    "VotingClassifierCalculator",
    "VotingClassifierApplier",
    "StackingClassifierCalculator",
    "StackingClassifierApplier",
    "VotingRegressorCalculator",
    "VotingRegressorApplier",
    "StackingRegressorCalculator",
    "StackingRegressorApplier",
    "perform_cross_validation",
    "HyperparameterField",
    "get_hyperparameters",
    "get_default_search_space",
    "MultinomialNBCalculator",
    "MultinomialNBApplier",
    "BernoulliNBCalculator",
    "BernoulliNBApplier",
]

# NOTE: Imports above are intentionally explicit. Every node module is imported
# by name so its ``@NodeRegistry.register`` decorators run at import time. We do
# NOT auto-discover submodules with ``pkgutil.iter_modules``; explicit imports
# keep the registry deterministic and prevent stray/duplicate files from being
# silently registered. Adding a new model node requires one import line here.
