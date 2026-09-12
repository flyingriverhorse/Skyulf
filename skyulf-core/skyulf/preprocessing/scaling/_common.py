"""Shared numeric-column selection and Decimal conversion for scaling nodes.

Automatic selection includes pandas Decimal object columns and Polars Decimal
dtypes. Selected Decimal values use float64 for scaling; source frames and
unselected columns keep their original values. Refit older artifacts to include
Decimal columns that their automatic selection previously omitted.
"""

from typing import Any

from ...utils import detect_numeric_columns, resolve_columns
from .._helpers import decimal_columns_to_float


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
