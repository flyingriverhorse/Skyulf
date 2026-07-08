"""Shared mask helpers for outlier nodes."""

from typing import Any

import pandas as pd


def _filter_y_polars(y: Any, mask_series: Any) -> Any:
    """Apply a Polars boolean mask to ``y`` if it is a Polars Series/DataFrame."""
    import polars as pl

    if y is None:
        return None
    if isinstance(y, (pl.Series, pl.DataFrame)):
        return y.filter(mask_series)
    return y


def _apply_pandas_mask(X_pd: Any, y: Any, mask: pd.Series) -> tuple[Any, Any]:
    """Apply a Pandas boolean mask to X (and y if non-null)."""
    X_filtered = X_pd[mask]
    if y is None:
        return X_filtered, y
    y_pd: Any = y
    return X_filtered, y_pd[mask]
