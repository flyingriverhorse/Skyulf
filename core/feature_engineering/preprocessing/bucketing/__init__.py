"""Bucketing and discretization preprocessing modules."""

from .binning import (
    BINNING_DEFAULT_EQUAL_FREQUENCY_BINS,
    BINNING_DEFAULT_EQUAL_WIDTH_BINS,
    BINNING_DEFAULT_MISSING_LABEL,
    BINNING_DEFAULT_PRECISION,
    BINNING_DEFAULT_SUFFIX,
    BINNING_STRATEGIES,
    KBINS_ENCODE_TYPES,
    KBINS_STRATEGIES,
    _apply_binning_discretization,
    _build_binned_distribution,
    _build_binning_recommendations,
    _normalize_binning_config,
)

__all__ = [
    "BINNING_DEFAULT_EQUAL_FREQUENCY_BINS",
    "BINNING_DEFAULT_EQUAL_WIDTH_BINS",
    "BINNING_DEFAULT_MISSING_LABEL",
    "BINNING_DEFAULT_PRECISION",
    "BINNING_DEFAULT_SUFFIX",
    "BINNING_STRATEGIES",
    "KBINS_ENCODE_TYPES",
    "KBINS_STRATEGIES",
    "_apply_binning_discretization",
    "_build_binned_distribution",
    "_build_binning_recommendations",
    "_normalize_binning_config",
]
