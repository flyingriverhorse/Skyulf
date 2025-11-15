"""Statistics-focused preprocessing modules (scaling, outliers, skewness, imputation)."""

from .outliers_removal import (
    DEFAULT_METHOD_PARAMETERS,
    OUTLIER_DEFAULT_METHOD,
    OUTLIER_METHODS,
    _apply_outlier_removal,
    _build_outlier_recommendations,
    _outlier_method_details,
)
from .skewness import (
    SKEWNESS_METHODS,
    SKEWNESS_THRESHOLD,
    _apply_skewness_transformations,
    _build_skewness_recommendations,
    _skewness_method_details,
)
from .imputation import (
    METHOD_LABELS,
    SIMPLE_METHODS,
    ADVANCED_METHODS,
    apply_imputation_methods,
)
from .scaling import (
    SCALING_DEFAULT_METHOD,
    SCALING_METHODS,
    SCALING_METHOD_ORDER,
    NormalizedScalingConfig,
    _apply_scale_numeric_features,
    _build_scaling_recommendations,
    _scaling_method_details,
)

__all__ = [
    "DEFAULT_METHOD_PARAMETERS",
    "OUTLIER_DEFAULT_METHOD",
    "OUTLIER_METHODS",
    "_apply_outlier_removal",
    "_build_outlier_recommendations",
    "_outlier_method_details",
    "SKEWNESS_METHODS",
    "SKEWNESS_THRESHOLD",
    "_apply_skewness_transformations",
    "_build_skewness_recommendations",
    "_skewness_method_details",
    "METHOD_LABELS",
    "SIMPLE_METHODS",
    "ADVANCED_METHODS",
    "apply_imputation_methods",
    "SCALING_DEFAULT_METHOD",
    "SCALING_METHODS",
    "SCALING_METHOD_ORDER",
    "NormalizedScalingConfig",
    "_apply_scale_numeric_features",
    "_build_scaling_recommendations",
    "_scaling_method_details",
]
