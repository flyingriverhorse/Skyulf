"""Shared helpers for the time-series preprocessing nodes.

These nodes are row-order dependent. To stay deterministic across engines we
optionally sort by a user-supplied ``sort_by`` column before computing lags or
rolling windows, and (when ``group_by`` is given) compute within each group.

Because sorting reorders rows, the sort helpers return the row positions they
used: a paired ``y`` must be permuted through the *same* positions or it keeps
its original order while ``X`` takes the new one.
"""

from datetime import datetime
from typing import Any

import pandas as pd
import polars as pl


def parse_datetime_scalar(value: str) -> datetime | None:
    """Parse one date string independently, using UTC for native Polars expressions."""
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if isinstance(parsed, pd.Timestamp):
        return parsed.to_pydatetime()
    return None


# Supported calendar parts extracted by DateFeatures. Keys are the public
# feature names; values are the pandas ``.dt`` accessor used to compute them.
DATE_FEATURE_ACCESSORS: dict[str, str] = {
    "year": "year",
    "month": "month",
    "day": "day",
    "dayofweek": "dayofweek",
    "dayofyear": "dayofyear",
    "quarter": "quarter",
    "weekofyear": "isocalendar.week",
    "hour": "hour",
    "minute": "minute",
    "is_weekend": "is_weekend",
    "is_month_start": "is_month_start",
    "is_month_end": "is_month_end",
}

# Rolling aggregations we expose. Mapped to the pandas ``Rolling`` method name;
# the polars path maps the same keys onto ``pl.Expr.rolling_*`` builders.
ROLLING_AGGREGATIONS: list[str] = ["mean", "sum", "min", "max", "std", "median"]


def filter_existing_columns(columns: Any, available: list[str]) -> list[str]:
    """Return configured columns that actually exist, preserving order.

    Named distinctly from ``skyulf.utils.resolve_columns`` (a different,
    unrelated function with a different signature/auto-detect semantics used
    by the encoders) to avoid a naming collision that risks readers/refactors
    conflating the two.
    """
    if not columns:
        return []
    return [c for c in columns if c in available]


def coerce_lags(lags: Any) -> list[int]:
    """Normalise the ``lags`` config into a sorted list of positive ints."""
    if isinstance(lags, int):
        lags = [lags]
    out = sorted({int(v) for v in (lags or []) if int(v) > 0})
    return out


def coerce_aggregations(aggs: Any) -> list[str]:
    """Keep only recognised rolling aggregation names, preserving order."""
    if isinstance(aggs, str):
        aggs = [aggs]
    return [a for a in (aggs or []) if a in ROLLING_AGGREGATIONS]


def sort_with_positions_pandas(df: pd.DataFrame, sort_by: str | None) -> tuple[pd.DataFrame, Any]:
    """Stable-sort ``df`` by ``sort_by``, returning it plus the row positions used.

    The positions are indices into the *unsorted* frame, so handing the same
    value to ``select_rows_by_position`` permutes a paired ``y`` identically
    . They come from a RangeIndex'd copy of the sort key run through
    pandas' own ``sort_values``, so the order is identical to ``sort_pandas`` by
    construction rather than by a second sort implementation that could drift —
    including its ``na_position="last"`` default and stable tie-breaking.

    ``None`` positions mean the frame was not reordered, i.e. ``y`` needs no
    change either.
    """
    if not sort_by or sort_by not in df.columns:
        return df, None
    positions = df[sort_by].reset_index(drop=True).sort_values(kind="mergesort").index.to_numpy()
    return df.iloc[positions], positions


def sort_pandas(df: pd.DataFrame, sort_by: str | None) -> pd.DataFrame:
    """Stable-sort a pandas frame by ``sort_by`` when present."""
    return sort_with_positions_pandas(df, sort_by)[0]


def sort_with_positions_polars(X: Any, sort_by: str | None) -> tuple[Any, Any]:
    """Polars counterpart of ``sort_with_positions_pandas``.

    ``pl.arg_sort_by`` is the expression form of ``DataFrame.sort`` and accepts
    the same ``nulls_last`` / ``maintain_order`` flags, so ``X.gather(order)``
    equals the ``X.sort(...)`` this replaces; verified on ties, nulls, dates and
    single-row frames.
    """
    if not sort_by or sort_by not in X.columns:
        return X, None
    order = X.select(pl.arg_sort_by(sort_by, nulls_last=True, maintain_order=True)).to_series()
    return X.gather(order), order
