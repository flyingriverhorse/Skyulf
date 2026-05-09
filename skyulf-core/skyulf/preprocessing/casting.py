"""Type-casting node — routes apply through the dual-engine dispatcher.

The fit step is config-only (no per-engine math), so it stays single-path.
The pandas apply path was previously a single CCN-busting function; it is now
split per dtype family (`float / int / bool / datetime / other`).
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseApplier, BaseCalculator, apply_method, fit_method
from .dispatcher import apply_dual_engine
from ._artifacts import CastingArtifact
from ._schema import SkyulfSchema
from ..registry import NodeRegistry
from ..core.meta.decorators import node_meta

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


def _casting_apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Any:
    import polars as pl

    type_map = params.get("type_map", {})
    if not type_map:
        return X, y
    coerce_on_error = params.get("coerce_on_error", True)

    exprs = []
    for col, target_dtype in type_map.items():
        if col not in X.columns:
            continue
        pl_dtype = _resolve_polars_dtype(str(target_dtype).lower())
        if pl_dtype is None:
            continue
        exprs.append(pl.col(col).cast(pl_dtype, strict=not coerce_on_error).alias(col))

    return (X.with_columns(exprs) if exprs else X), y


# -----------------------------------------------------------------------------
# Pandas apply — split per dtype family to keep CCN low
# -----------------------------------------------------------------------------


def _coerce_boolean_value(value: Any) -> Optional[bool]:
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
    if s in ("true", "yes", "1", "on", "y", "t"):
        return True
    if s in ("false", "no", "0", "off", "n", "f"):
        return False
    return None


def _cast_float(series: pd.Series, target_dtype: Any, coerce_on_error: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce" if coerce_on_error else "raise")
    return numeric.astype(target_dtype)


def _drop_fractional_or_raise(numeric: pd.Series, col: str, coerce_on_error: bool) -> pd.Series:
    """Mask fractional values to NaN (coerce) or raise."""
    valid = numeric.notna()
    fractional_mask = valid & ~np.isclose(numeric, np.round(numeric))
    if not fractional_mask.any():
        return numeric
    if not coerce_on_error:
        raise ValueError(f"Column {col} contains fractional values, cannot cast to integer.")
    numeric.loc[fractional_mask] = np.nan
    return numeric


def _cast_int(series: pd.Series, col: str, target_dtype: Any, coerce_on_error: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce" if coerce_on_error else "raise")
    numeric = _drop_fractional_or_raise(numeric, col, coerce_on_error)
    if numeric.isna().any() and target_dtype in ("int32", "int64", "int"):
        # NaNs require pandas' nullable Int64 container.
        return numeric.astype("Int64")
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


def _casting_apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Any:
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
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
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
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
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
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> CastingArtifact:
        # Config supports two shapes:
        #   {'columns': ['col1'], 'target_type': 'float'}
        #   {'column_types': {'col1': 'float', 'col2': 'int'}}
        target_type = config.get("target_type")
        columns = config.get("columns", [])
        column_types = config.get("column_types", {})

        final_map: Dict[str, Any] = {}
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
