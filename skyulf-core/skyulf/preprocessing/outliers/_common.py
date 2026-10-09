"""Shared mask helpers for outlier nodes.

Both helpers select rows *positionally* and route ``y`` through
``select_rows_by_position``, so every target shape the dispatcher accepts is
filtered rather than silently passed through. A helper that returns ``y``
unchanged for a shape it does not recognise is what desynchronised ``X`` and
``y`` here (OC-166): numpy targets came back with all their rows while ``X``
lost the outliers, and list targets crashed on one engine and no-oped on the
other.
"""

from decimal import Decimal
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from .._helpers import select_rows_by_position


def validate_fitted_bounds(bounds: Any, *, partial: bool) -> None:
    """Inspect per-column saved numeric limits without changing their scalar types."""
    if type(bounds) is not dict:
        raise ValueError("Fitted bounds must be a dictionary.")
    for column, bound in bounds.items():
        if not isinstance(column, str) or not column:
            raise ValueError("Fitted bound columns must be nonempty strings.")
        _validate_bound(bound, partial=partial)


def _validate_bound(bound: Any, *, partial: bool) -> None:
    """Allow manual open limits while requiring both learned percentile limits."""
    if type(bound) is not dict:
        raise ValueError("Each fitted bound must be a dictionary.")
    fields = {"lower", "upper"}
    if set(bound) - fields or (not partial and set(bound) != fields):
        raise ValueError("Unexpected fitted bound fields.")
    for value in bound.values():
        if value is None and partial:
            continue
        if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
            raise ValueError("Fitted limits must be real numeric scalars.")


def _filter_y_polars(y: Any, mask_series: Any) -> Any:
    """Apply a Polars keep-mask to ``y`` for every target shape the dispatcher accepts.

    ``X.filter(mask)`` and ``y.gather(mask.arg_true())`` keep the same rows in
    the same order, so the pair stays aligned.
    """
    if y is None:
        return None
    return select_rows_by_position(y, mask_series.arg_true())


def _apply_pandas_mask(X_pd: Any, y: Any, mask: pd.Series) -> tuple[Any, Any]:
    """Apply a Pandas boolean keep-mask to ``X`` and to ``y``.

    ``y`` is selected with ``.iloc`` on the mask's positions, never with
    ``y[mask]``: label-aligned boolean indexing expands or misaligns on a
    duplicated index, and it cannot index a numpy or list target at all.
    """
    keep = np.flatnonzero(mask.to_numpy())
    return X_pd.iloc[keep], select_rows_by_position(y, keep)
