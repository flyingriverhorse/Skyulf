from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import BaseApplier, BaseCalculator
from ..registry import NodeRegistry
from ..core.meta.decorators import node_meta
from ..engines import SkyulfDataFrame, get_engine
from ..utils import pack_pipeline_output, unpack_pipeline_input

# Map common aliases to pandas types
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


def _coerce_boolean_value(value: Any) -> Optional[bool]:
    """
    Robustly coerce a value to a boolean.
    Returns None if coercion fails.
    """
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


class CastingApplier(BaseApplier):
    def apply(
        self, df: SkyulfDataFrame, params: Dict[str, Any]
    ) -> Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)
        
        type_map = params.get("type_map", {})
        coerce_on_error = params.get("coerce_on_error", True)

        if not type_map:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            
            exprs = []
            for col, target_dtype in type_map.items():
                if col not in X.columns:
                    continue
                
                dtype_str = str(target_dtype).lower()
                pl_dtype = None
                
                # Map to Polars types
                if dtype_str in ["float", "float64", "double", "numeric"]:
                    pl_dtype = pl.Float64
                elif dtype_str == "float32":
                    pl_dtype = pl.Float32
                elif dtype_str in ["int", "int64", "integer"]:
                    pl_dtype = pl.Int64
                elif dtype_str == "int32":
                    pl_dtype = pl.Int32
                elif dtype_str in ["string", "str", "text"]:
                    pl_dtype = pl.String
                elif dtype_str in ["bool", "boolean"]:
                    pl_dtype = pl.Boolean
                elif dtype_str in ["category", "categorical"]:
                    pl_dtype = pl.Categorical
                elif dtype_str.startswith("datetime") or dtype_str == "date":
                    pl_dtype = pl.Datetime
                
                if pl_dtype:
                    # Handle coercion if needed (strict=False returns null on error)
                    if coerce_on_error:
                        exprs.append(pl.col(col).cast(pl_dtype, strict=False).alias(col))
                    else:
                        exprs.append(pl.col(col).cast(pl_dtype, strict=True).alias(col))
                else:
                    # Fallback or unknown type
                    pass
            
            if exprs:
                X = X.with_columns(exprs)
            
            return pack_pipeline_output(X, y, is_tuple)

        # Pandas Path
        return self._apply_dataframe(X, params, y, is_tuple)

    def _apply_dataframe(  # noqa: C901
        self, df: pd.DataFrame, params: Dict[str, Any], y=None, is_tuple=False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Any]]:
        type_map = params.get("type_map", {})
        coerce_on_error = params.get("coerce_on_error", True)

        df_out = df.copy()

        for col, target_dtype in type_map.items():
            if col not in df_out.columns:
                continue

            try:
                series = df_out[col]

                # Determine family
                dtype_str = str(target_dtype).lower()

                if dtype_str.startswith("float"):
                    # Float Family
                    numeric = pd.to_numeric(
                        series, errors="coerce" if coerce_on_error else "raise"
                    )
                    df_out[col] = numeric.astype(target_dtype)


                elif dtype_str.startswith("int"):
                    # Int Family
                    numeric = pd.to_numeric(
                        series, errors="coerce" if coerce_on_error else "raise"
                    )

                    # Check for fractional values
                    if coerce_on_error:
                        # If coercing, we set fractional to NaN
                        valid_mask = numeric.notna()
                        fractional_mask = valid_mask & ~np.isclose(
                            numeric, np.round(numeric)
                        )
                        if fractional_mask.any():
                            numeric.loc[fractional_mask] = np.nan
                    else:
                        # If not coercing, we raise error on fractional
                        fractional_mask = numeric.notna() & ~np.isclose(
                            numeric, np.round(numeric)
                        )
                        if fractional_mask.any():
                            raise ValueError(
                                f"Column {col} contains fractional values, cannot cast to integer."
                            )

                    # Handle NaNs -> Nullable Int64
                    if numeric.isna().any():
                        # Use nullable Int64 if target is standard int
                        # If target is already nullable (Int64), use it.
                        # If target is numpy int (int64), we must upgrade to Int64 to hold NaNs
                        if target_dtype in ["int32", "int64", "int"]:
                            df_out[col] = numeric.astype("Int64")
                        else:
                            df_out[col] = numeric.astype(target_dtype)
                    else:
                        df_out[col] = numeric.astype(target_dtype)

                elif dtype_str.startswith("bool"):
                    # Boolean Family
                    try:
                        df_out[col] = series.astype("boolean")
                    except (TypeError, ValueError):
                        if not coerce_on_error:
                            raise
                        # Robust coercion
                        coerced_values = [
                            (
                                pd.NA
                                if (result := _coerce_boolean_value(val)) is None
                                else result
                            )
                            for val in series
                        ]
                        df_out[col] = pd.Series(
                            coerced_values, index=series.index, dtype="boolean"
                        )

                elif dtype_str.startswith("datetime"):
                    # Datetime Family
                    errors = "coerce" if coerce_on_error else "raise"
                    df_out[col] = pd.to_datetime(series, errors=errors)  # type: ignore

                else:
                    # String / Category / Other
                    df_out[col] = series.astype(target_dtype)

            except Exception:
                if not coerce_on_error:
                    raise
                # If coercion is on, we might leave it as is or try best effort?
                pass

        return pack_pipeline_output(df_out, y, is_tuple)


@NodeRegistry.register("Casting", CastingApplier)
@node_meta(
    id="Casting",
    name="Type Casting",
    category="Data Operations",
    description="Cast columns to specific data types.",
    params={"type_map": {}, "coerce_on_error": True}
)
class CastingCalculator(BaseCalculator):
    def fit(
        self, df: SkyulfDataFrame, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Config: {'columns': ['col1'], 'target_type': 'float'}
        # OR {'column_types': {'col1': 'float', 'col2': 'int'}}
        
        X, _, _ = unpack_pipeline_input(df)
        # We don't need to convert to pandas just to check columns, 
        # but let's keep it consistent if we need complex logic.
        # Here we just check columns existence.
        
        target_type = config.get("target_type")
        columns = config.get("columns", [])
        column_types = config.get("column_types", {})

        # Normalize to column_types map
        final_map = {}

        # 1. Process explicit map
        for col, dtype in column_types.items():
            if col in X.columns:
                final_map[col] = TYPE_ALIASES.get(str(dtype).lower(), dtype)

        # 2. Process list + single type
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
