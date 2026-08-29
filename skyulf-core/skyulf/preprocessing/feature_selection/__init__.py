"""Feature-selection nodes package.

Importing this package registers all feature-selection nodes (variance,
correlation, univariate, model-based, and the unified facade) and re-exports
their public classes.
"""

from .correlation import CorrelationThresholdApplier, CorrelationThresholdCalculator
from .facade import FeatureSelectionApplier, FeatureSelectionCalculator
from .model_based import ModelBasedSelectionApplier, ModelBasedSelectionCalculator
from .univariate import UnivariateSelectionApplier, UnivariateSelectionCalculator
from .variance import VarianceThresholdApplier, VarianceThresholdCalculator

__all__ = [
    "CorrelationThresholdApplier",
    "CorrelationThresholdCalculator",
    "FeatureSelectionApplier",
    "FeatureSelectionCalculator",
    "ModelBasedSelectionApplier",
    "ModelBasedSelectionCalculator",
    "UnivariateSelectionApplier",
    "UnivariateSelectionCalculator",
    "VarianceThresholdApplier",
    "VarianceThresholdCalculator",
]
