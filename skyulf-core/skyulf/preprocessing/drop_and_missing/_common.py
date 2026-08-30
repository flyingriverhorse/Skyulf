"""Shared helpers for drop / missing-value nodes."""

from typing import Any

import polars as pl


def _polars_filter_y_by_kept_indices(y: Any, kept_indices: Any) -> Any:
    """Filter ``y`` (Polars Series / DataFrame) to the rows kept in ``X``.

    ``kept_indices`` is a Polars Series of integer row indices that survived
    a filter on ``X``. Used by Deduplicate + DropMissingRows so dropping rows
    in ``X`` propagates to a paired ``y``.
    """
    if y is None:
        return None
    if isinstance(y, pl.DataFrame):
        return (
            y.with_row_index("__idx__")
            .filter(pl.col("__idx__").is_in(kept_indices))
            .drop("__idx__")
        )
    if isinstance(y, pl.Series):
        return y.gather(kept_indices)
    raise TypeError(
        f"Cannot filter y of type {type(y).__name__} by kept row indices on the Polars "
        "engine; expected a polars DataFrame or Series (or None)."
    )


def _normalize_subset(subset: Any, existing_cols: list) -> list | None:
    """Filter ``subset`` to columns that actually exist; return ``None`` if empty."""
    if not subset:
        return None
    filtered = [c for c in subset if c in existing_cols]
    return filtered if filtered else None
