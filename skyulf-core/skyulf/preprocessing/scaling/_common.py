"""Shared helpers for scaling nodes (engine-specific subset selection)."""

from typing import Any

from ...utils import detect_numeric_columns, resolve_columns


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
    return cols, X[cols]
