"""Modeling module for Skyulf."""

from ._evaluation import (
    apply_thresholds,
    calculate_classification_metrics,
    calculate_clustering_metrics,
    calculate_regression_metrics,
    optimize_thresholds,
)
from ._explainability import compute_shap_explanation
from .base import BaseModelApplier, BaseModelCalculator, StatefulEstimator
from .classification import (
    CalibratedClassifierApplier,
    CalibratedClassifierCalculator,
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
    RandomForestClassifierApplier,
    RandomForestClassifierCalculator,
    SGDClassifierApplier,
    SGDClassifierCalculator,
)
from .clustering import KMeansApplier, KMeansCalculator
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
from .fold_preprocessing import FoldPreprocessor
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
    "BaseModelApplier",
    "BaseModelCalculator",
    "BernoulliNBApplier",
    "BernoulliNBCalculator",
    "CalibratedClassifierApplier",
    "CalibratedClassifierCalculator",
    "FoldPreprocessor",
    "HyperparameterField",
    "KMeansApplier",
    "KMeansCalculator",
    "LogisticRegressionApplier",
    "LogisticRegressionCalculator",
    "MultinomialNBApplier",
    "MultinomialNBCalculator",
    "RandomForestClassifierApplier",
    "RandomForestClassifierCalculator",
    "RandomForestRegressorApplier",
    "RandomForestRegressorCalculator",
    "RidgeRegressionApplier",
    "RidgeRegressionCalculator",
    "SGDClassifierApplier",
    "SGDClassifierCalculator",
    "SklearnApplier",
    "SklearnCalculator",
    "StackingClassifierApplier",
    "StackingClassifierCalculator",
    "StackingRegressorApplier",
    "StackingRegressorCalculator",
    "StatefulEstimator",
    "VotingClassifierApplier",
    "VotingClassifierCalculator",
    "VotingRegressorApplier",
    "VotingRegressorCalculator",
    "apply_thresholds",
    "calculate_classification_metrics",
    "calculate_clustering_metrics",
    "calculate_regression_metrics",
    "compute_shap_explanation",
    "get_default_search_space",
    "get_hyperparameters",
    "optimize_thresholds",
    "perform_cross_validation",
]

# NOTE: Imports above are intentionally explicit. Every node module is imported
# by name so its ``@NodeRegistry.register`` decorators run at import time. We do
# NOT auto-discover submodules with ``pkgutil.iter_modules``; explicit imports
# keep the registry deterministic and prevent stray/duplicate files from being
# silently registered. Adding a new model node requires one import line here.
