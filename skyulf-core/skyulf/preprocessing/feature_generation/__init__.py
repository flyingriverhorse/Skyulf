"""Feature-generation nodes package.

Importing this package registers the PolynomialFeatures and FeatureGeneration
nodes and re-exports their public classes plus shared constants for backward
compatibility.
"""

from ._common import (
    ALLOWED_DATETIME_FEATURES,
    DEFAULT_EPSILON,
    FEATURE_MATH_ALLOWED_TYPES,
)
from ._pandas_ops import _featgen_apply_pandas
from ._polars_ops import _featgen_apply_polars
from .generation import FeatureGenerationApplier, FeatureGenerationCalculator
from .polynomial import PolynomialFeaturesApplier, PolynomialFeaturesCalculator

__all__ = [
    "PolynomialFeaturesApplier",
    "PolynomialFeaturesCalculator",
    "FeatureGenerationApplier",
    "FeatureGenerationCalculator",
    "DEFAULT_EPSILON",
    "FEATURE_MATH_ALLOWED_TYPES",
    "ALLOWED_DATETIME_FEATURES",
    "_featgen_apply_pandas",
    "_featgen_apply_polars",
]
