"""Type-casting node — routes apply through the dual-engine dispatcher.

The fit step is config-only (no per-engine math), so it stays single-path.
The pandas apply path was previously a single CCN-busting function; it is now
split per dtype family (`float / int / bool / datetime / other`).
"""

from typing import Any

import numpy as np
import pandas as pd

from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from ._artifacts import CastingArtifact
from ._schema import SkyulfSchema
from .base import BaseApplier, BaseCalculator, apply_method, fit_method
from .dispatcher import apply_dual_engine

# Map common aliases to canonical pandas dtype labels.
TYPE_ALIASES = {
    "float": "float64",
    "float32": "float32",
    "float64": "float64",
    "double": "float64",
    "numeric": "float64",
    "int": "int64",
    "int32": "int32",
    "int64": "int64",
    "integer": "int64",
    "string": "string",
    "str": "string",
    "text": "string",
    "category": "category",
    "categorical": "category",
    "bool": "boolean",
    "boolean": "boolean",
    "datetime": "datetime64[ns]",
    "date": "datetime64[ns]",
    "datetime64": "datetime64[ns]",
    "datetime64[ns]": "datetime64[ns]",
}

# Shared string-alias table for boolean coercion (used by both the pandas
# and Polars apply paths so "yes"/"no"-style flags are recognized
# identically regardless of engine).
_BOOL_TRUE_ALIASES = ("true", "yes", "1", "on", "y", "t")
_BOOL_FALSE_ALIASES = ("false", "no", "0", "off", "n", "f")


# -----------------------------------------------------------------------------
# Polars apply
# -----------------------------------------------------------------------------


def _resolve_polars_dtype(dtype_str: str) -> Any:
    """Map a string dtype to a Polars type, or ``None`` if unsupported."""
    import polars as pl

    # Use a small alias table so the function stays low-CCN. Datetime variants
    # ("date", "datetime[...]") are handled by the prefix check below.
    table = {
        "float": pl.Float64,
        "float64": pl.Float64,
        "double": pl.Float64,
        "numeric": pl.Float64,
        "float32": pl.Float32,
        "int": pl.Int64,
        "int64": pl.Int64,
        "integer": pl.Int64,
        "int32": pl.Int32,
        "int16": pl.Int16,
        "int8": pl.Int8,
        "uint": pl.UInt64,
        "uint64": pl.UInt64,
        "uint32": pl.UInt32,
        "uint16": pl.UInt16,
        "uint8": pl.UInt8,
        "string": pl.String,
        "str": pl.String,
        "text": pl.String,
        "bool": pl.Boolean,
        "boolean": pl.Boolean,
        "category": pl.Categorical,
        "categorical": pl.Categorical,
        "date": pl.Datetime,
    }
    if dtype_str in table:
        return table[dtype_str]
    if dtype_str.startswith("datetime"):
        return pl.Datetime
    return None


def _bool_expr_from_string_col_polars(col: str) -> Any:
    """Build a Polars expression casting a String/Utf8 column to Boolean.

    Polars has no direct Utf8->Boolean cast at all (raises
    `InvalidOperationError` even with ``strict=False``), unlike pandas'
    `.astype("boolean")` which happens to fail the same way for strings but
    then falls back to a per-value alias table (`_coerce_boolean_value`).
    Without this, casting a "yes"/"no"-style string column to boolean on the
    Polars engine hard-crashes instead of coercing - a stark engine-parity
    break for a very common use case. Mirrors `_coerce_boolean_value`'s alias
    table; unrecognized values become null (the caller decides whether to
    raise on that, mirroring `coerce_on_error`).
    """
    import polars as pl

    normalized = pl.col(col).str.strip_chars().str.to_lowercase()
    return (
        pl.when(normalized.is_in(list(_BOOL_TRUE_ALIASES)))
        .then(True)  # noqa: FBT003
        .when(normalized.is_in(list(_BOOL_FALSE_ALIASES)))
        .then(False)  # noqa: FBT003
        .otherwise(None)
        .alias(col)
    )


def _casting_apply_polars(X: Any, y: Any, params: dict[str, Any]) -> Any:
    import polars as pl

    type_map = params.get("type_map", {})
    if not type_map:
        return X, y
    coerce_on_error = params.get("coerce_on_error", True)

    exprs = []
    string_bool_cols: list[str] = []
    for col, target_dtype in type_map.items():
        if col not in X.columns:
            continue
        pl_dtype = _resolve_polars_dtype(str(target_dtype).lower())
        if pl_dtype is None:
            # In strict mode (coerce_on_error=False), an unsupported dtype
            # string is a configuration error and must be surfaced loudly -
            # previously this silently skipped the column even in strict
            # mode, while the equivalent pandas path raises. In coerce mode
            # we keep the existing best-effort "leave untouched" behavior.
            if not coerce_on_error:
                raise ValueError(
                    f"Column '{col}': unsupported target dtype '{target_dtype}' for the "
                    "Polars engine."
                )
            continue
        if pl_dtype == pl.Boolean and X.schema[col] in (pl.String, pl.Utf8, pl.Categorical):
            exprs.append(_bool_expr_from_string_col_polars(col))
            string_bool_cols.append(col)
            continue
        exprs.append(pl.col(col).cast(pl_dtype, strict=not coerce_on_error).alias(col))

    if not exprs:
        return X, y

    result = X.with_columns(exprs)

    if not coerce_on_error and string_bool_cols:
        for col in string_bool_cols:
            newly_null = result[col].is_null() & X[col].is_not_null()
            if newly_null.any():
                raise ValueError(
                    f"Cannot cast column '{col}' to boolean: contains value(s) "
                    "not recognized as true/false."
                )

    return result, y


# -----------------------------------------------------------------------------
# Pandas apply — split per dtype family to keep CCN low
# -----------------------------------------------------------------------------


def _coerce_boolean_value(value: Any) -> bool | None:
    """Robustly coerce a single value to bool; ``None`` if undecidable."""
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    s = str(value).strip().lower()
    if s in _BOOL_TRUE_ALIASES:
        return True
    if s in _BOOL_FALSE_ALIASES:
        return False
    return None


def _cast_float(series: pd.Series, target_dtype: Any, coerce_on_error: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce" if coerce_on_error else "raise")
    return numeric.astype(target_dtype)


def _drop_fractional_or_raise(numeric: pd.Series, col: str, coerce_on_error: bool) -> pd.Series:
    """Mask fractional values to NaN (coerce) or raise."""
    valid = numeric.notna()
    # Use a small FIXED absolute tolerance rather than np.isclose's default
    # rtol=1e-5, which scales with magnitude and would let large fractional
    # values (e.g. 100000.001) slip through as "close enough" to integral.
    fractional_mask = valid & (np.abs(numeric - np.round(numeric)) >= 1e-9)
    if not fractional_mask.any():
        return numeric
    if not coerce_on_error:
        raise ValueError(f"Column {col} contains fractional values, cannot cast to integer.")
    numeric.loc[fractional_mask] = np.nan
    return numeric


def _mask_out_of_range_or_raise(
    numeric: pd.Series, col: str, target_dtype: Any, coerce_on_error: bool
) -> pd.Series:
    """Mask values outside the target integer dtype's range to NaN (coerce) or raise.

    Without this, ``pandas.Series.astype`` silently clamps out-of-range floats
    (e.g. 3e9 -> int32) instead of erroring or nulling, diverging from the
    polars apply path which correctly nulls/raises via ``strict`` casts.
    """
    try:
        info = np.iinfo(str(target_dtype))
    except TypeError:
        # Unrecognized/non-integer dtype string: skip range-checking rather
        # than fail the cast outright.
        return numeric
    valid = numeric.notna()
    out_of_range_mask = valid & ((numeric < info.min) | (numeric > info.max))
    if not out_of_range_mask.any():
        return numeric
    if not coerce_on_error:
        raise OverflowError(
            f"Column {col} contains values out of range for {target_dtype} "
            f"(valid range [{info.min}, {info.max}])."
        )
    numeric.loc[out_of_range_mask] = np.nan
    return numeric


# Map each numpy integer dtype string to its pandas nullable (NA-capable)
# counterpart, preserving the requested width/signedness instead of always
# widening to Int64.
_NULLABLE_INT_DTYPES = {
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
    "int": "Int64",
    "uint8": "UInt8",
    "uint16": "UInt16",
    "uint32": "UInt32",
    "uint64": "UInt64",
    "uint": "UInt64",
}


def _cast_int(series: pd.Series, col: str, target_dtype: Any, coerce_on_error: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce" if coerce_on_error else "raise")
    numeric = _drop_fractional_or_raise(numeric, col, coerce_on_error)
    numeric = _mask_out_of_range_or_raise(numeric, col, target_dtype, coerce_on_error)
    if numeric.isna().any():
        # NaNs require a pandas nullable container. Previously this only
        # covered int32/int64/int, so casting a narrower dtype (int8/int16/
        # uint8/etc.) with coerce_on_error=True and any out-of-range/NaN
        # value fell through to `numeric.astype(target_dtype)` below, which
        # raises on plain numpy int dtypes with NaN present — that raise was
        # then silently swallowed by the caller's best-effort except block,
        # leaving the ENTIRE column completely uncast (not even in-range
        # values got converted). Look up the matching nullable dtype for any
        # integer width instead of only the three previously handled.
        nullable_dtype = _NULLABLE_INT_DTYPES.get(str(target_dtype).lower(), "Int64")
        return numeric.astype(nullable_dtype)  # ty: ignore[no-matching-overload]
    return numeric.astype(target_dtype)


def _cast_bool(series: pd.Series, coerce_on_error: bool) -> pd.Series:
    try:
        return series.astype("boolean")
    except (TypeError, ValueError):
        if not coerce_on_error:
            raise
        coerced = [
            (pd.NA if (result := _coerce_boolean_value(val)) is None else result) for val in series
        ]
        return pd.Series(coerced, index=series.index, dtype="boolean")


def _cast_datetime(series: pd.Series, coerce_on_error: bool) -> pd.Series:
    return pd.to_datetime(series, errors="coerce" if coerce_on_error else "raise")


def _cast_one_column(
    series: pd.Series, col: str, target_dtype: Any, coerce_on_error: bool
) -> pd.Series:
    """Dispatch a single column to the right family caster."""
    dtype_str = str(target_dtype).lower()
    if dtype_str.startswith("float"):
        return _cast_float(series, target_dtype, coerce_on_error)
    if dtype_str.startswith("int"):
        return _cast_int(series, col, target_dtype, coerce_on_error)
    if dtype_str.startswith("bool"):
        return _cast_bool(series, coerce_on_error)
    if dtype_str.startswith("datetime"):
        return _cast_datetime(series, coerce_on_error)
    return series.astype(target_dtype)


def _casting_apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> Any:
    type_map = params.get("type_map", {})
    if not type_map:
        return X, y
    coerce_on_error = params.get("coerce_on_error", True)

    df_out = X.copy()
    for col, target_dtype in type_map.items():
        if col not in df_out.columns:
            continue
        try:
            df_out[col] = _cast_one_column(df_out[col], col, target_dtype, coerce_on_error)
        except Exception:
            if not coerce_on_error:
                raise
            # Best-effort: leave the column as-is when coerce_on_error is True.
    return df_out, y


class CastingApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, _casting_apply_polars, _casting_apply_pandas)


@NodeRegistry.register("Casting", CastingApplier)
@node_meta(
    id="Casting",
    name="Type Casting",
    category="Data Operations",
    description="Cast columns to specific data types.",
    params={"type_map": {}, "coerce_on_error": True},
)
class CastingCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        # Casting preserves the column set but rewrites dtype labels.
        column_types = dict(config.get("column_types", {}) or {})
        target_type = config.get("target_type")
        columns = config.get("columns", []) or []
        if target_type and columns:
            for col in columns:
                column_types.setdefault(col, target_type)
        new_schema = input_schema
        for col, dtype in column_types.items():
            resolved = TYPE_ALIASES.get(str(dtype).lower(), str(dtype))
            new_schema = new_schema.with_dtype(col, resolved)
        return new_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> CastingArtifact:  # pylint: disable=arguments-differ
        # Config supports two shapes:
        #   {'columns': ['col1'], 'target_type': 'float'}
        #   {'column_types': {'col1': 'float', 'col2': 'int'}}
        target_type = config.get("target_type")
        columns = config.get("columns", [])
        column_types = config.get("column_types", {})

        final_map: dict[str, Any] = {}
        for col, dtype in column_types.items():
            if col in X.columns:
                final_map[col] = TYPE_ALIASES.get(str(dtype).lower(), dtype)

        if target_type and columns:
            resolved_type = TYPE_ALIASES.get(str(target_type).lower(), target_type)
            for col in columns:
                if col in X.columns:
                    final_map[col] = resolved_type

        return {
            "type": "casting",
            "type_map": final_map,
            "coerce_on_error": config.get("coerce_on_error", True),
        }
