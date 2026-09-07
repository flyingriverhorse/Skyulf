"""Shared mask helpers for outlier nodes.

Both helpers select rows *positionally* and route ``y`` through
``select_rows_by_position``, so every target shape the dispatcher accepts is
filtered rather than silently passed through. A helper that returns ``y``
unchanged for a shape it does not recognise is what desynchronised ``X`` and
``y`` here (OC-166): numpy targets came back with all their rows while ``X``
lost the outliers, and list targets crashed on one engine and no-oped on the
other.
"""

from typing import Any

import numpy as np
import pandas as pd

from .._helpers import select_rows_by_position


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
