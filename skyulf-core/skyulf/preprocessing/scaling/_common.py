"""Shared numeric-column selection and Decimal conversion for scaling nodes.

Automatic selection includes pandas Decimal object columns and Polars Decimal
dtypes. Selected Decimal values use float64 for scaling; source frames and
unselected columns keep their original values. Refit older artifacts to include
Decimal columns that their automatic selection previously omitted.
"""

from decimal import Decimal
from math import isfinite
from numbers import Real
from typing import Any

from ...utils import detect_numeric_columns, resolve_columns
from .._helpers import decimal_columns_to_float


def validate_scaling_range(value: Any, name: str) -> tuple[Any, Any]:
    """Normalize two finite bounds and validate the scaler's interval contract."""
    try:
        bounds = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly two finite numbers.") from exc
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain exactly two finite numbers.")
    if not all(_finite_bound(bound) for bound in bounds):
        raise ValueError(f"{name} must contain exactly two finite numbers.")
    lower, upper = bounds
    if name == "feature_range" and lower >= upper:
        raise ValueError("feature_range minimum must be less than maximum.")
    if name == "quantile_range" and not 0 <= lower <= upper <= 100:
        raise ValueError("quantile_range must satisfy 0 <= minimum <= maximum <= 100.")
    return lower, upper


def _finite_bound(value: Any) -> bool:
    """Accept real numeric endpoints without coercing booleans or nulls."""
    return isinstance(value, (Real, Decimal)) and not isinstance(value, bool) and isfinite(value)


def _select_subset_polars(X: Any, config: dict[str, Any]) -> tuple[list[str], Any]:
    """Resolve numeric columns and return (cols, X[cols]) for a Polars frame."""
    cols = resolve_columns(X, config, detect_numeric_columns)
    if not cols:
        return [], None
    return cols, X.select(cols)


def _select_subset_pandas(X: Any, config: dict[str, Any]) -> tuple[list[str], Any]:
    """Resolve numeric columns and return (cols, X[cols]) for a Pandas frame."""
    cols = resolve_columns(X, config, detect_numeric_columns)
    if not cols:
        return [], None
    return cols, decimal_columns_to_float(X[cols], cols)
