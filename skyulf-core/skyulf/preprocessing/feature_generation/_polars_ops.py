"""Polars-engine op handlers for FeatureGeneration."""

from collections.abc import Callable
from typing import Any

from ._common import (
    DEFAULT_EPSILON,
    _compute_similarity_score,
    _resolve_group_agg_cols,
    _resolve_output_col,
    _resolve_similarity_pair,
)


def _polars_arith_terms(op: dict[str, Any], existing: list[str]) -> tuple[list[Any], list[float]]:
    import polars as pl

    valid = [
        c for c in op.get("input_columns", []) + op.get("secondary_columns", []) if c in existing
    ]
    fill_val = op.get("fillna") if op.get("fillna") is not None else 0
    col_exprs = [pl.col(c).cast(pl.Float64).fill_null(fill_val) for c in valid]
    const_vals = [float(c) for c in op.get("constants", [])]
    return col_exprs, const_vals


def _polars_add(col_exprs: list[Any], const_vals: list[float], _epsilon: float) -> Any:
    import polars as pl

    return pl.sum_horizontal(col_exprs) + sum(const_vals) if col_exprs else pl.lit(sum(const_vals))


def _polars_subtract(col_exprs: list[Any], const_vals: list[float], _epsilon: float) -> Any:
    import polars as pl

    expr = col_exprs[0] if col_exprs else pl.lit(0.0)
    for e in col_exprs[1:]:
        expr = expr - e
    for c in const_vals:
        expr = expr - c
    return expr


def _polars_multiply(col_exprs: list[Any], const_vals: list[float], _epsilon: float) -> Any:
    import polars as pl

    expr = pl.lit(1.0)
    for e in col_exprs:
        expr = expr * e
    for c in const_vals:
        expr = expr * c
    return expr


def _polars_divide(col_exprs: list[Any], const_vals: list[float], epsilon: float) -> Any | None:
    import polars as pl

    if col_exprs:
        expr = col_exprs[0]
        others = col_exprs[1:]
    elif const_vals:
        expr = pl.lit(const_vals[0])
        others = []
        const_vals = const_vals[1:]
    else:
        return None

    def safe_denom(d: Any) -> Any:
        return pl.when(d.abs() < epsilon).then(epsilon).otherwise(d)

    for e in others:
        expr = expr / safe_denom(e)
    for c in const_vals:
        expr = expr / (c if abs(c) > epsilon else epsilon)
    return expr


_POLARS_ARITH_BUILDERS: dict[str, Callable[[list[Any], list[float], float], Any | None]] = {
    "add": _polars_add,
    "subtract": _polars_subtract,
    "multiply": _polars_multiply,
    "divide": lambda col_exprs, const_vals, eps: _polars_divide(col_exprs, const_vals, eps),
}


def _polars_arith(op: dict[str, Any], existing: list[str], epsilon: float) -> Any | None:
    method = op.get("method")
    col_exprs, const_vals = _polars_arith_terms(op, existing)
    if not col_exprs and not const_vals:
        return None
    builder = _POLARS_ARITH_BUILDERS.get(method or "")
    return builder(col_exprs, const_vals, epsilon) if builder else None


def _polars_ratio(op: dict[str, Any], existing: list[str], epsilon: float) -> Any | None:
    import polars as pl

    nums = [
        pl.col(c).cast(pl.Float64).fill_null(0)
        for c in op.get("input_columns", [])
        if c in existing
    ]
    dens = [
        pl.col(c).cast(pl.Float64).fill_null(0)
        for c in op.get("secondary_columns", [])
        if c in existing
    ]
    if not nums or not dens:
        return None
    num_sum = pl.sum_horizontal(nums)
    den_sum = pl.sum_horizontal(dens)
    return num_sum / pl.when(den_sum.abs() < epsilon).then(epsilon).otherwise(den_sum)


def _polars_similarity(op: dict[str, Any], existing: list[str], _eps: float) -> Any | None:
    import polars as pl

    pair = _resolve_similarity_pair(op, existing)
    if pair is None:
        return None
    col_a, col_b = pair
    method = op.get("method") or "ratio"
    a_empty = pl.col(col_a).is_null() | (pl.col(col_a).cast(pl.String) == "")
    b_empty = pl.col(col_b).is_null() | (pl.col(col_b).cast(pl.String) == "")

    def sim_func(struct_val: Any) -> float:
        return _compute_similarity_score(struct_val.get("a"), struct_val.get("b"), method)

    return (
        pl.when(a_empty & b_empty)
        .then(pl.lit(100.0))
        .when(a_empty | b_empty)
        .then(pl.lit(0.0))
        .otherwise(
            pl.struct([pl.col(col_a).alias("a"), pl.col(col_b).alias("b")]).map_elements(
                sim_func, return_dtype=pl.Float64
            )
        )
    )


_POLARS_DT_FEATURES: dict[str, Callable[[Any], Any]] = {}


def _register_polars_dt() -> None:
    import polars as pl

    _POLARS_DT_FEATURES.update(
        {
            "year": lambda d: d.dt.year(),
            "month": lambda d: d.dt.month(),
            "day": lambda d: d.dt.day(),
            "hour": lambda d: d.dt.hour(),
            "minute": lambda d: d.dt.minute(),
            "second": lambda d: d.dt.second(),
            "quarter": lambda d: d.dt.quarter(),
            # Polars dt.weekday() is ISO 1-indexed; subtract 1 to match
            # pandas dayofweek (Mon=0, Sun=6).
            "weekday": lambda d: d.dt.weekday() - 1,
            # Use raw 1-indexed value: Sat=6, Sun=7 ⇒ ``>= 6`` catches both.
            "is_weekend": lambda d: (d.dt.weekday() >= 6).cast(pl.Int64),
            "week": lambda d: d.dt.week(),
            "month_name": lambda d: d.dt.strftime("%B"),
            "day_name": lambda d: d.dt.strftime("%A"),
        }
    )


def _build_polars_dt_exprs(col: str, base_dt: Any, features: list[str]) -> list[Any]:
    """Return the per-feature Polars expressions for one datetime column."""
    exprs: list[Any] = []
    for feat in features:
        builder = _POLARS_DT_FEATURES.get(feat)
        if builder is not None:
            exprs.append(builder(base_dt).alias(f"{col}_{feat}"))
    return exprs


def _polars_datetime_apply(op: dict[str, Any], X_out: Any) -> Any:
    """Materialise datetime-extract feature columns onto ``X_out`` (Polars)."""
    import polars as pl

    if not _POLARS_DT_FEATURES:
        _register_polars_dt()

    valid = [c for c in op.get("input_columns", []) if c in X_out.columns]
    features = op.get("datetime_features", [])
    dt_exprs: list[Any] = []
    for col in valid:
        base_dt = pl.col(col)
        if X_out.schema[col] == pl.String:
            base_dt = pl.col(col).str.to_datetime(strict=False)
        dt_exprs.extend(_build_polars_dt_exprs(col, base_dt, features))
    if dt_exprs:
        X_out = X_out.with_columns(dt_exprs)
    return X_out


_POLARS_AGG_BUILDERS: dict[str, Callable[[Any], Any]] = {
    "mean": lambda c: c.mean(),
    "sum": lambda c: c.sum(),
    "count": lambda c: c.count(),
    "min": lambda c: c.min(),
    "max": lambda c: c.max(),
    "std": lambda c: c.std(),
    "median": lambda c: c.median(),
}


def _polars_group_agg(op: dict[str, Any], existing: list[str], _epsilon: float) -> Any | None:
    """Group-by aggregation broadcast back per row via Polars window ``over``."""
    import polars as pl

    resolved = _resolve_group_agg_cols(op, existing)
    if resolved is None:
        return None
    group_col, target_col, method = resolved
    builder = _POLARS_AGG_BUILDERS.get(method)
    if builder is None:
        return None
    target = pl.col(target_col)
    if method != "count":
        target = target.cast(pl.Float64)
    return builder(target).over(group_col)


_POLARS_OP_HANDLERS: dict[str, Callable[[dict[str, Any], list[str], float], Any | None]] = {
    "arithmetic": _polars_arith,
    "ratio": _polars_ratio,
    "similarity": _polars_similarity,
    "group_agg": _polars_group_agg,
}


def _featgen_apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    operations = params.get("operations", [])
    if not operations:
        return X, y
    epsilon = params.get("epsilon", DEFAULT_EPSILON)
    allow_overwrite = params.get("allow_overwrite", False)

    X_out = X
    for i, op in enumerate(operations):
        op_type = op.get("operation_type", "arithmetic")
        try:
            if op_type == "datetime_extract":
                X_out = _polars_datetime_apply(op, X_out)
                continue
            handler = _POLARS_OP_HANDLERS.get(op_type)
            if handler is None:
                continue
            expr = handler(op, list(X_out.columns), epsilon)
            if expr is None:
                continue
            output_col = _resolve_output_col(op, i, list(X_out.columns), allow_overwrite)
            round_digits = op.get("round_digits")
            if round_digits is not None:
                expr = expr.round(round_digits)
            X_out = X_out.with_columns(expr.alias(output_col))
        except Exception:
            pass  # nosec B110 - skip a malformed op; other feature-generation ops still apply
    return X_out, y
