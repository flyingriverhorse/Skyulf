"""Shared helpers for the time-series preprocessing nodes.

These nodes are row-order dependent. To stay deterministic across engines we
optionally sort by a user-supplied ``sort_by`` column before computing lags or
rolling windows, and (when ``group_by`` is given) compute within each group.
"""

from typing import Any

import pandas as pd

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


def sort_pandas(df: pd.DataFrame, sort_by: str | None) -> pd.DataFrame:
    """Stable-sort a pandas frame by ``sort_by`` when present."""
    if sort_by and sort_by in df.columns:
        return df.sort_values(sort_by, kind="mergesort")
    return df
