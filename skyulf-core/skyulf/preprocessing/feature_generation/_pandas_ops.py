"""Pandas-engine op handlers for FeatureGeneration."""

import builtins
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ._common import (
    DEFAULT_EPSILON,
    SEASON_BY_MONTH,
    TIME_OF_DAY_BUCKETS,
    TIME_OF_DAY_DEFAULT,
    _resolve_group_agg_cols,
    _resolve_output_col,
    _resolve_similarity_pair,
    _safe_divide,
    _vectorised_similarity,
)

logger = logging.getLogger(__name__)


def _pandas_arith_terms(op: dict[str, Any], df_out: Any) -> tuple[list[pd.Series], list[float]]:
    valid = [
        c
        for c in op.get("input_columns", []) + op.get("secondary_columns", [])
        if c in df_out.columns
    ]
    fill_val = op.get("fillna") if op.get("fillna") is not None else 0
    series_list = [pd.to_numeric(df_out[c], errors="coerce").fillna(fill_val) for c in valid]
    const_vals = [builtins.float(c) for c in op.get("constants", [])]
    return series_list, const_vals


def _sum_pandas_series(series_list: list[pd.Series], idx: Any) -> pd.Series:
    """Sum a list of pandas series with index ``idx``; empty list ⇒ zero series."""
    res = pd.Series(0.0, index=idx)
    for s in series_list:
        res = res.add(s, fill_value=0)
    return res


def _pandas_add(
    series_list: list[pd.Series], const_vals: list[float], idx: Any, _eps: float
) -> pd.Series:
    res = _sum_pandas_series(series_list, idx)
    for c in const_vals:
        res = res.add(c)
    return res


def _pandas_subtract(
    series_list: list[pd.Series], const_vals: list[float], idx: Any, _eps: float
) -> pd.Series:
    res = series_list[0].copy() if series_list else pd.Series(0.0, index=idx)
    for s in series_list[1:]:
        res = res.subtract(s, fill_value=0)  # type: ignore[call-overload]  # ty: ignore[call-non-callable]
    for c in const_vals:
        res = res.sub(c)
    return res


def _pandas_multiply(
    series_list: list[pd.Series], const_vals: list[float], idx: Any, _eps: float
) -> pd.Series:
    res = pd.Series(1.0, index=idx)
    for s in series_list:
        res = res.multiply(s, fill_value=1)  # type: ignore[call-overload]  # ty: ignore[call-non-callable]
    for c in const_vals:
        res = res.mul(c)  # type: ignore[call-overload]  # ty: ignore[no-matching-overload]
    return res


def _pandas_divide(
    series_list: list[pd.Series], const_vals: list[float], idx: Any, epsilon: float
) -> pd.Series | None:
    if series_list:
        res = series_list[0].copy()
        others = series_list[1:]
    elif const_vals:
        res = pd.Series(const_vals[0], index=idx)
        others = []
        const_vals = const_vals[1:]
    else:
        return None
    for s in others:
        res = _safe_divide(res, s, epsilon)
    for c in const_vals:
        denom = c if abs(c) > epsilon else epsilon
        res = res.div(denom)
    return res


_PANDAS_ARITH_BUILDERS: dict[
    str, Callable[[list[pd.Series], list[float], Any, float], pd.Series | None]
] = {
    "add": _pandas_add,
    "subtract": _pandas_subtract,
    "multiply": _pandas_multiply,
    "divide": lambda s, c, idx, eps: _pandas_divide(s, c, idx, eps),
}


def _pandas_arith(op: dict[str, Any], df_out: Any, epsilon: float) -> pd.Series | None:
    method = op.get("method")
    series_list, const_vals = _pandas_arith_terms(op, df_out)
    if not series_list and not const_vals:
        return None
    builder = _PANDAS_ARITH_BUILDERS.get(method or "")
    return builder(series_list, const_vals, df_out.index, epsilon) if builder else None


def _pandas_ratio(op: dict[str, Any], df_out: Any, epsilon: float) -> pd.Series | None:
    nums = [
        pd.to_numeric(df_out[c], errors="coerce").fillna(0)
        for c in op.get("input_columns", [])
        if c in df_out.columns
    ]
    dens = [
        pd.to_numeric(df_out[c], errors="coerce").fillna(0)
        for c in op.get("secondary_columns", [])
        if c in df_out.columns
    ]
    if not nums or not dens:
        return None
    return _safe_divide(
        _sum_pandas_series(nums, df_out.index), _sum_pandas_series(dens, df_out.index), epsilon
    )


def _pandas_similarity(op: dict[str, Any], df_out: Any, _eps: float) -> pd.Series | None:
    pair = _resolve_similarity_pair(op, list(df_out.columns))
    if pair is None:
        return None
    col_a, col_b = pair
    return _vectorised_similarity(df_out[col_a], df_out[col_b], op.get("method") or "ratio")


def _pandas_season(d: Any) -> Any:
    """Map a datetime series' month to its meteorological season name."""
    return d.dt.month.map(SEASON_BY_MONTH)


def _pandas_time_of_day(d: Any) -> Any:
    """Bucket a datetime series' hour into a time-of-day label."""
    hour = d.dt.hour
    conditions = [(hour >= lo) & (hour <= hi) for lo, hi, _label in TIME_OF_DAY_BUCKETS]
    choices = [label for _lo, _hi, label in TIME_OF_DAY_BUCKETS]
    return pd.Series(
        np.select(conditions, choices, default=TIME_OF_DAY_DEFAULT), index=d.index
    )


_PANDAS_DT_FEATURES: dict[str, Callable[[Any], Any]] = {
    "year": lambda d: d.dt.year,
    "month": lambda d: d.dt.month,
    "day": lambda d: d.dt.day,
    "hour": lambda d: d.dt.hour,
    "minute": lambda d: d.dt.minute,
    "second": lambda d: d.dt.second,
    "quarter": lambda d: d.dt.quarter,
    "weekday": lambda d: d.dt.dayofweek,
    "is_weekend": lambda d: (d.dt.dayofweek >= 5).astype(int),
    "week": lambda d: d.dt.isocalendar().week.astype(int),
    "month_name": lambda d: d.dt.month_name(),
    "day_name": lambda d: d.dt.day_name(),
    "season": _pandas_season,
    "time_of_day": _pandas_time_of_day,
}


def _pandas_datetime_apply(op: dict[str, Any], df_out: Any) -> None:
    """Materialise datetime-extract features onto ``df_out`` in place."""
    valid = [c for c in op.get("input_columns", []) if c in df_out.columns]
    features = op.get("datetime_features", [])
    for col in valid:
        try:
            dt = pd.to_datetime(df_out[col], errors="coerce")
            for feat in features:
                builder = _PANDAS_DT_FEATURES.get(feat)
                if builder is None:
                    continue
                df_out[f"{col}_{feat}"] = builder(dt)
        except Exception as e:
            logger.warning(f"Failed to extract datetime features for column {col}: {e}")


_PANDAS_AGG_METHODS = {"mean", "sum", "count", "min", "max", "std", "median"}


def _pandas_group_agg(op: dict[str, Any], df_out: Any, _eps: float) -> pd.Series | None:
    """Group-by aggregation broadcast back per row via ``groupby().transform()``."""
    resolved = _resolve_group_agg_cols(op, list(df_out.columns))
    if resolved is None:
        return None
    group_col, target_col, method = resolved
    if method not in _PANDAS_AGG_METHODS:
        return None
    target = df_out[target_col]
    if method != "count":
        target = pd.to_numeric(target, errors="coerce")
    return target.groupby(df_out[group_col]).transform(method)


_PANDAS_OP_HANDLERS: dict[str, Callable[[dict[str, Any], Any, float], pd.Series | None]] = {
    "arithmetic": _pandas_arith,
    "ratio": _pandas_ratio,
    "similarity": _pandas_similarity,
    "group_agg": _pandas_group_agg,
}


def _featgen_apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    operations = params.get("operations", [])
    if not operations:
        return X, y
    epsilon = params.get("epsilon", DEFAULT_EPSILON)
    allow_overwrite = params.get("allow_overwrite", False)

    df_out = X.copy()
    for i, op in enumerate(operations):
        op_type = op.get("operation_type", "arithmetic")
        try:
            if op_type == "datetime_extract":
                _pandas_datetime_apply(op, df_out)
                continue
            handler = _PANDAS_OP_HANDLERS.get(op_type)
            if handler is None:
                continue
            result = handler(op, df_out, epsilon)
            if result is None:
                continue
            output_col = _resolve_output_col(op, i, list(df_out.columns), allow_overwrite)
            round_digits = op.get("round_digits")
            if round_digits is not None:
                result = result.round(round_digits)
            df_out[output_col] = result
        except Exception as e:
            logger.warning(f"Failed to apply {op_type} operation (index {i}): {e}")
    return df_out, y
