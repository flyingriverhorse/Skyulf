"""Shared helpers for drop / missing-value nodes."""

from typing import Any

import numpy as np
import polars as pl


def _polars_with_row_positions(X: Any) -> tuple[Any, str]:
    """Add row positions under a name that cannot overwrite a user column."""
    index_name = "__idx__"
    while index_name in X.columns:
        index_name += "_"
    return X.with_row_index(index_name), index_name


def _polars_filter_y_by_kept_indices(y: Any, kept_indices: Any) -> Any:
    """Filter native Polars or engine-neutral targets to the rows kept in ``X``.

    ``kept_indices`` is a Polars Series of integer row indices that survived
    a filter on ``X``. Used by Deduplicate + DropMissingRows so dropping rows
    in ``X`` propagates to a paired ``y``.
    """
    if y is None:
        return None
    if isinstance(y, (pl.DataFrame, pl.Series)):
        return y.gather(kept_indices)
    if isinstance(y, np.ndarray):
        return y[kept_indices.to_numpy()]
    if isinstance(y, list):
        return [y[position] for position in kept_indices]
    raise TypeError(
        f"Cannot filter y of type {type(y).__name__} by kept row indices on the Polars "
        "engine; expected a polars DataFrame or Series, numpy array, list, or None."
    )


def _pandas_filter_y_by_kept_positions(y: Any, kept_positions: Any) -> Any:
    """Select rows of native pandas or engine-neutral targets by positional index.

    Positional (``.iloc``) selection is required because label-based ``.loc``
    selection returns every row matching a duplicated index label, which
    desynchronizes ``y`` from the cleaned ``X``.
    """
    if y is None:
        return None
    if isinstance(y, np.ndarray):
        return y[kept_positions]
    if isinstance(y, list):
        return [y[position] for position in kept_positions]
    return y.iloc[kept_positions]


def _normalize_subset(subset: Any, existing_cols: list) -> list | None:
    """Filter ``subset`` to columns that actually exist; return ``None`` if empty."""
    if not subset:
        return None
    filtered = [c for c in subset if c in existing_cols]
    return filtered if filtered else None
