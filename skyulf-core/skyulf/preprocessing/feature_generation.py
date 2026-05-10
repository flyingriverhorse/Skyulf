"""Feature-generation nodes (PolynomialFeatures + FeatureGeneration math).

Per-engine apply paths are split into small handlers and dispatched via
:func:`apply_dual_engine`. The per-operation logic (``arithmetic`` /
``ratio`` / ``similarity`` / ``datetime_extract``) lives in module-level
helpers indexed by ``_FEATGEN_OPS_POLARS`` / ``_FEATGEN_OPS_PANDAS``.

Calculator ``fit`` paths are sklearn-bound, so they use :func:`to_pandas`
once at the top instead of going through ``fit_dual_engine``.
"""

import builtins
import math
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from ..registry import NodeRegistry
from ..core.meta.decorators import node_meta
from ..utils import detect_numeric_columns
from .base import BaseApplier, BaseCalculator, apply_method, fit_method
from ._artifacts import FeatureGenerationArtifact, PolynomialFeaturesArtifact
from ._helpers import to_pandas
from .dispatcher import apply_dual_engine
from ..engines import SkyulfDataFrame

# --- Optional Dependencies ---
fuzz: Any = None
try:
    from rapidfuzz import fuzz as _fuzz  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]

    fuzz = _fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

# --- Constants ---
DEFAULT_EPSILON = 1e-9
FEATURE_MATH_ALLOWED_TYPES = {
    "arithmetic",
    "ratio",
    "similarity",
    "datetime_extract",
    "group_agg",
    "polynomial",
}
ALLOWED_DATETIME_FEATURES = {
    "year",
    "quarter",
    "month",
    "month_name",
    "week",
    "day",
    "day_name",
    "weekday",
    "is_weekend",
    "hour",
    "minute",
    "second",
    "season",
    "time_of_day",
}

# --- Helpers ---


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
        return val if not math.isnan(val) else None
    except (TypeError, ValueError):
        return None


def _safe_divide(numerator: pd.Series, denominator: pd.Series, epsilon: float) -> pd.Series:
    adjusted = denominator.copy()
    adjusted = adjusted.replace({0: epsilon, -0.0: epsilon})
    adjusted = adjusted.fillna(epsilon)
    mask = adjusted.abs() < epsilon
    if mask.any():
        adjusted[mask] = epsilon
    return numerator / adjusted


_FUZZ_METHODS: Dict[str, str] = {
    "token_sort_ratio": "token_sort_ratio",
    "token_set_ratio": "token_set_ratio",
}


def _compute_similarity_score(a: Any, b: Any, method: str) -> float:
    text_a = "" if pd.isna(a) else str(a)
    text_b = "" if pd.isna(b) else str(b)
    if not text_a and not text_b:
        return 100.0
    if not text_a or not text_b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        attr = _FUZZ_METHODS.get(method, "ratio")
        return float(getattr(fuzz, attr)(text_a, text_b))
    return SequenceMatcher(None, text_a, text_b).ratio() * 100.0


def _vectorised_similarity(s_a: pd.Series, s_b: pd.Series, method: str) -> pd.Series:
    """Vectorised element-wise similarity between two pandas string Series.

    Avoids df.apply(axis=1) overhead by:
    1. Pre-computing null/empty masks with numpy/pandas ops (vectorised).
    2. Only invoking the per-pair Python similarity for rows that need it.
    """
    a_str = s_a.fillna("").astype(str)
    b_str = s_b.fillna("").astype(str)
    a_empty = a_str.eq("")
    b_empty = b_str.eq("")
    both_empty = a_empty & b_empty
    one_empty = (a_empty | b_empty) & ~both_empty
    needs_compute = ~(a_empty | b_empty)

    result = pd.Series(0.0, index=s_a.index, dtype=float)
    result.loc[both_empty] = 100.0
    result.loc[one_empty] = 0.0

    if needs_compute.any():
        idx = needs_compute[needs_compute].index
        for i in idx:
            result.loc[i] = _compute_similarity_score(a_str.at[i], b_str.at[i], method)

    return result


# -----------------------------------------------------------------------------
# Polynomial Features — shared sklearn helper
# -----------------------------------------------------------------------------


def _polynomial_compute(
    X_subset: pd.DataFrame, valid_cols: List[str], params: Dict[str, Any]
) -> Optional[Tuple[Any, List[str]]]:
    """Run sklearn PolynomialFeatures + name normalisation; ``None`` ⇒ skip."""
    poly = PolynomialFeatures(
        degree=params.get("degree", 2),
        interaction_only=params.get("interaction_only", False),
        include_bias=params.get("include_bias", False),
    )
    poly.fit(X_subset)
    transformed = poly.transform(X_subset)
    if hasattr(transformed, "values"):
        transformed = transformed.values
    feature_names = poly.get_feature_names_out(valid_cols)

    include_input = params.get("include_input_features", False)
    keep = [i for i, p in enumerate(poly.powers_) if not (sum(p) == 1 and not include_input)]
    if not keep:
        return None

    transformed = transformed[:, keep]
    feature_names = feature_names[keep]
    output_prefix = params.get("output_prefix", "poly")
    new_names = [
        f"{output_prefix}_{name.replace(' ', '_').replace('^', '_pow_')}" for name in feature_names
    ]
    return transformed, new_names


def _polynomial_apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    import polars as pl

    valid_cols = [c for c in params.get("columns", []) if c in X.columns]
    if not valid_cols:
        return X, _y

    result = _polynomial_compute(X.select(valid_cols).to_pandas(), valid_cols, params)
    if result is None:
        return X, _y
    transformed, new_names = result
    df_poly = pl.DataFrame(transformed, schema=new_names)
    return pl.concat([X, df_poly], how="horizontal"), _y


def _polynomial_apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    valid_cols = [c for c in params.get("columns", []) if c in X.columns]
    if not valid_cols:
        return X, _y

    result = _polynomial_compute(X[valid_cols], valid_cols, params)
    if result is None:
        return X, _y
    transformed, new_names = result
    df_poly = pd.DataFrame(cast(Any, transformed), columns=cast(Any, new_names), index=X.index)
    return pd.concat(cast(Any, [X, df_poly]), axis=1), _y


class PolynomialFeaturesApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, _polynomial_apply_polars, _polynomial_apply_pandas)


@NodeRegistry.register("PolynomialFeatures", PolynomialFeaturesApplier)
@NodeRegistry.register("PolynomialFeaturesNode", PolynomialFeaturesApplier)
@node_meta(
    id="PolynomialFeatures",
    name="Polynomial Features",
    category="Feature Engineering",
    description="Generate polynomial and interaction features.",
    params={"degree": 2, "interaction_only": False, "include_bias": False},
)
class PolynomialFeaturesCalculator(BaseCalculator):
    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> PolynomialFeaturesArtifact:
        X_pd = to_pandas(X)
        cols = list(config.get("columns", []))
        if not cols and config.get("auto_detect", False):
            cols = detect_numeric_columns(X_pd)
        cols = [c for c in cols if c in X_pd.columns]
        if not cols:
            return cast(PolynomialFeaturesArtifact, {})

        degree = config.get("degree", 2)
        interaction_only = config.get("interaction_only", False)
        include_bias = config.get("include_bias", False)

        poly = PolynomialFeatures(
            degree=degree, interaction_only=interaction_only, include_bias=include_bias
        )
        poly.fit(X_pd[cols])
        return cast(
            PolynomialFeaturesArtifact,
            {
                "type": "polynomial_features",
                "columns": cols,
                "degree": degree,
                "interaction_only": interaction_only,
                "include_bias": include_bias,
                "include_input_features": config.get("include_input_features", False),
                "output_prefix": config.get("output_prefix", "poly"),
                "feature_names": poly.get_feature_names_out(cols).tolist(),
            },
        )


# -----------------------------------------------------------------------------
# Feature Generation — per-op handlers
# -----------------------------------------------------------------------------


def _resolve_output_col(
    op: Dict[str, Any], i: int, existing: List[str], allow_overwrite: bool
) -> str:
    """Pick a non-colliding output column name for op ``i``."""
    output_col = op.get("output_column")
    if not output_col:
        base = f"{op.get('operation_type', 'arithmetic')}_{i}"
        prefix = op.get("output_prefix")
        output_col = f"{prefix}_{base}" if prefix else base
    if output_col in existing and not allow_overwrite:
        j = 1
        while f"{output_col}_{j}" in existing:
            j += 1
        output_col = f"{output_col}_{j}"
    return output_col


# --- Polars op handlers ---


def _polars_arith_terms(op: Dict[str, Any], existing: List[str]) -> Tuple[List[Any], List[float]]:
    import polars as pl

    valid = [
        c for c in op.get("input_columns", []) + op.get("secondary_columns", []) if c in existing
    ]
    fill_val = op.get("fillna") if op.get("fillna") is not None else 0
    col_exprs = [pl.col(c).cast(pl.Float64).fill_null(fill_val) for c in valid]
    const_vals = [float(c) for c in op.get("constants", [])]
    return col_exprs, const_vals


def _polars_add(col_exprs: List[Any], const_vals: List[float], _epsilon: float) -> Any:
    import polars as pl

    return pl.sum_horizontal(col_exprs) + sum(const_vals) if col_exprs else pl.lit(sum(const_vals))


def _polars_subtract(col_exprs: List[Any], const_vals: List[float], _epsilon: float) -> Any:
    import polars as pl

    expr = col_exprs[0] if col_exprs else pl.lit(0.0)
    for e in col_exprs[1:]:
        expr = expr - e
    for c in const_vals:
        expr = expr - c
    return expr


def _polars_multiply(col_exprs: List[Any], const_vals: List[float], _epsilon: float) -> Any:
    import polars as pl

    expr = pl.lit(1.0)
    for e in col_exprs:
        expr = expr * e
    for c in const_vals:
        expr = expr * c
    return expr


_POLARS_ARITH_BUILDERS: Dict[str, Callable[[List[Any], List[float], float], Optional[Any]]] = {
    "add": _polars_add,
    "subtract": _polars_subtract,
    "multiply": _polars_multiply,
    "divide": lambda col_exprs, const_vals, eps: _polars_divide(col_exprs, const_vals, eps),
}


def _polars_arith(op: Dict[str, Any], existing: List[str], epsilon: float) -> Optional[Any]:
    method = op.get("method")
    col_exprs, const_vals = _polars_arith_terms(op, existing)
    if not col_exprs and not const_vals:
        return None
    builder = _POLARS_ARITH_BUILDERS.get(method or "")
    return builder(col_exprs, const_vals, epsilon) if builder else None


def _polars_divide(col_exprs: List[Any], const_vals: List[float], epsilon: float) -> Optional[Any]:
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


def _polars_ratio(op: Dict[str, Any], existing: List[str], epsilon: float) -> Optional[Any]:
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


def _resolve_similarity_pair(op: Dict[str, Any], existing: List[str]) -> Optional[Tuple[str, str]]:
    """Return ``(col_a, col_b)`` for similarity ops, or ``None`` if unresolved."""
    inputs = op.get("input_columns", [])
    secondary = op.get("secondary_columns", [])
    col_a = inputs[0] if inputs else None
    col_b = secondary[0] if secondary else (inputs[1] if len(inputs) > 1 else None)
    if not col_a or not col_b or col_a not in existing or col_b not in existing:
        return None
    return col_a, col_b


def _polars_similarity(op: Dict[str, Any], existing: List[str], _eps: float) -> Optional[Any]:
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


_POLARS_DT_FEATURES: Dict[str, Callable[[Any], Any]] = {}


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


def _build_polars_dt_exprs(col: str, base_dt: Any, features: List[str]) -> List[Any]:
    """Return the per-feature Polars expressions for one datetime column."""
    exprs: List[Any] = []
    for feat in features:
        builder = _POLARS_DT_FEATURES.get(feat)
        if builder is not None:
            exprs.append(builder(base_dt).alias(f"{col}_{feat}"))
    return exprs


def _polars_datetime_apply(op: Dict[str, Any], X_out: Any) -> Any:
    """Materialise datetime-extract feature columns onto ``X_out`` (Polars)."""
    import polars as pl

    if not _POLARS_DT_FEATURES:
        _register_polars_dt()

    valid = [c for c in op.get("input_columns", []) if c in X_out.columns]
    features = op.get("datetime_features", [])
    dt_exprs: List[Any] = []
    for col in valid:
        base_dt = pl.col(col)
        if X_out.schema[col] == pl.String:
            base_dt = pl.col(col).str.to_datetime(strict=False)
        dt_exprs.extend(_build_polars_dt_exprs(col, base_dt, features))
    if dt_exprs:
        X_out = X_out.with_columns(dt_exprs)
    return X_out


_POLARS_AGG_BUILDERS: Dict[str, Callable[[Any], Any]] = {
    "mean": lambda c: c.mean(),
    "sum": lambda c: c.sum(),
    "count": lambda c: c.count(),
    "min": lambda c: c.min(),
    "max": lambda c: c.max(),
    "std": lambda c: c.std(),
    "median": lambda c: c.median(),
}


def _resolve_group_agg_cols(
    op: Dict[str, Any], existing: List[str]
) -> Optional[Tuple[str, str, str]]:
    """Return ``(group_col, target_col, method)`` if op is well-formed, else None."""
    group_cols = [c for c in op.get("input_columns", []) if c in existing]
    target_cols = [c for c in op.get("secondary_columns", []) if c in existing]
    if not group_cols or not target_cols:
        return None
    return group_cols[0], target_cols[0], op.get("method") or "mean"


def _polars_group_agg(op: Dict[str, Any], existing: List[str], _epsilon: float) -> Optional[Any]:
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


_POLARS_OP_HANDLERS: Dict[str, Callable[[Dict[str, Any], List[str], float], Optional[Any]]] = {
    "arithmetic": _polars_arith,
    "ratio": _polars_ratio,
    "similarity": _polars_similarity,
    "group_agg": _polars_group_agg,
}


def _featgen_apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
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
            pass
    return X_out, y


# --- Pandas op handlers ---


def _pandas_arith_terms(op: Dict[str, Any], df_out: Any) -> Tuple[List[pd.Series], List[float]]:
    valid = [
        c
        for c in op.get("input_columns", []) + op.get("secondary_columns", [])
        if c in df_out.columns
    ]
    fill_val = op.get("fillna") if op.get("fillna") is not None else 0
    series_list = [pd.to_numeric(df_out[c], errors="coerce").fillna(fill_val) for c in valid]
    const_vals = [builtins.float(c) for c in op.get("constants", [])]
    return series_list, const_vals


def _pandas_add(
    series_list: List[pd.Series], const_vals: List[float], idx: Any, _eps: float
) -> pd.Series:
    res = _sum_pandas_series(series_list, idx)
    for c in const_vals:
        res = res.add(c)
    return res


def _pandas_subtract(
    series_list: List[pd.Series], const_vals: List[float], idx: Any, _eps: float
) -> pd.Series:
    res = series_list[0].copy() if series_list else pd.Series(0.0, index=idx)
    for s in series_list[1:]:
        res = res.subtract(s, fill_value=0)  # type: ignore[call-overload]
    for c in const_vals:
        res = res.sub(c)
    return res


def _pandas_multiply(
    series_list: List[pd.Series], const_vals: List[float], idx: Any, _eps: float
) -> pd.Series:
    res = pd.Series(1.0, index=idx)
    for s in series_list:
        res = res.multiply(s, fill_value=1)  # type: ignore[call-overload]
    for c in const_vals:
        res = res.mul(c)  # type: ignore[call-overload]
    return res


_PANDAS_ARITH_BUILDERS: Dict[
    str, Callable[[List[pd.Series], List[float], Any, float], Optional[pd.Series]]
] = {
    "add": _pandas_add,
    "subtract": _pandas_subtract,
    "multiply": _pandas_multiply,
    "divide": lambda s, c, idx, eps: _pandas_divide(s, c, idx, eps),
}


def _pandas_arith(op: Dict[str, Any], df_out: Any, epsilon: float) -> Optional[pd.Series]:
    method = op.get("method")
    series_list, const_vals = _pandas_arith_terms(op, df_out)
    if not series_list and not const_vals:
        return None
    builder = _PANDAS_ARITH_BUILDERS.get(method or "")
    return builder(series_list, const_vals, df_out.index, epsilon) if builder else None


def _pandas_divide(
    series_list: List[pd.Series], const_vals: List[float], idx: Any, epsilon: float
) -> Optional[pd.Series]:
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


def _sum_pandas_series(series_list: List[pd.Series], idx: Any) -> pd.Series:
    """Sum a list of pandas series with index ``idx``; empty list ⇒ zero series."""
    res = pd.Series(0.0, index=idx)
    for s in series_list:
        res = res.add(s, fill_value=0)
    return res


def _pandas_ratio(op: Dict[str, Any], df_out: Any, epsilon: float) -> Optional[pd.Series]:
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


def _pandas_similarity(op: Dict[str, Any], df_out: Any, _eps: float) -> Optional[pd.Series]:
    pair = _resolve_similarity_pair(op, list(df_out.columns))
    if pair is None:
        return None
    col_a, col_b = pair
    return _vectorised_similarity(df_out[col_a], df_out[col_b], op.get("method") or "ratio")


_PANDAS_DT_FEATURES: Dict[str, Callable[[Any], Any]] = {
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
}


def _pandas_datetime_apply(op: Dict[str, Any], df_out: Any) -> None:
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
        except Exception:
            pass


_PANDAS_AGG_METHODS = {"mean", "sum", "count", "min", "max", "std", "median"}


def _pandas_group_agg(op: Dict[str, Any], df_out: Any, _eps: float) -> Optional[pd.Series]:
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


_PANDAS_OP_HANDLERS: Dict[str, Callable[[Dict[str, Any], Any, float], Optional[pd.Series]]] = {
    "arithmetic": _pandas_arith,
    "ratio": _pandas_ratio,
    "similarity": _pandas_similarity,
    "group_agg": _pandas_group_agg,
}


def _featgen_apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
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
        except Exception:
            pass
    return df_out, y


class FeatureGenerationApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, _featgen_apply_polars, _featgen_apply_pandas)


@NodeRegistry.register("FeatureGeneration", FeatureGenerationApplier)
@NodeRegistry.register("FeatureMath", FeatureGenerationApplier)
@NodeRegistry.register("FeatureGenerationNode", FeatureGenerationApplier)
@node_meta(
    id="FeatureGenerationNode",
    name="Feature Generation (Math)",
    category="Feature Engineering",
    description="Generate new features using mathematical operations.",
    params={"operations": []},
)
class FeatureGenerationCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> FeatureGenerationArtifact:
        return {
            "type": "feature_generation",
            "operations": config.get("operations", []),
            "epsilon": config.get("epsilon", DEFAULT_EPSILON),
            "allow_overwrite": config.get("allow_overwrite", False),
        }
