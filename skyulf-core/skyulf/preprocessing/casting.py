from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from .base import BaseCalculator, BaseApplier

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
        if value == 1: return True
        if value == 0: return False
        return None
        
    s = str(value).strip().lower()
    if s in ('true', 'yes', '1', 'on', 'y', 't'):
        return True
    if s in ('false', 'no', '0', 'off', 'n', 'f'):
        return False
        
    return None

class CastingCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'columns': ['col1'], 'target_type': 'float'} 
        # OR {'column_types': {'col1': 'float', 'col2': 'int'}}
        
        target_type = config.get('target_type')
        columns = config.get('columns', [])
        column_types = config.get('column_types', {})
        
        # Normalize to column_types map
        final_map = {}
        
        # 1. Process explicit map
        for col, dtype in column_types.items():
            if col in df.columns:
                final_map[col] = TYPE_ALIASES.get(str(dtype).lower(), dtype)
                
        # 2. Process list + single type
        if target_type and columns:
            resolved_type = TYPE_ALIASES.get(str(target_type).lower(), target_type)
            for col in columns:
                if col in df.columns:
                    final_map[col] = resolved_type
                    
        return {
            'type': 'casting',
            'type_map': final_map,
            'coerce_on_error': config.get('coerce_on_error', True)
        }

class CastingApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        type_map = params.get('type_map', {})
        coerce_on_error = params.get('coerce_on_error', True)
        
        if not type_map:
            return df
            
        df_out = df.copy()
        
        for col, target_dtype in type_map.items():
            if col not in df_out.columns:
                continue
                
            try:
                series = df_out[col]
                
                # Determine family
                dtype_str = str(target_dtype).lower()
                
                if dtype_str.startswith('float'):
                    # Float Family
                    numeric = pd.to_numeric(series, errors='coerce' if coerce_on_error else 'raise')
                    df_out[col] = numeric.astype(target_dtype)
                    
                elif dtype_str.startswith('int'):
                    # Int Family
                    numeric = pd.to_numeric(series, errors='coerce' if coerce_on_error else 'raise')
                    
                    # Check for fractional values
                    if coerce_on_error:
                        # If coercing, we set fractional to NaN
                        valid_mask = numeric.notna()
                        fractional_mask = valid_mask & ~np.isclose(numeric, np.round(numeric))
                        if fractional_mask.any():
                            numeric.loc[fractional_mask] = np.nan
                    else:
                        # If not coercing, we raise error on fractional
                        fractional_mask = numeric.notna() & ~np.isclose(numeric, np.round(numeric))
                        if fractional_mask.any():
                            raise ValueError(f"Column {col} contains fractional values, cannot cast to integer.")

                    # Handle NaNs -> Nullable Int64
                    if numeric.isna().any():
                        # Use nullable Int64 if target is standard int
                        # If target is already nullable (Int64), use it.
                        # If target is numpy int (int64), we must upgrade to Int64 to hold NaNs
                        if target_dtype in ['int32', 'int64', 'int']:
                            df_out[col] = numeric.astype('Int64')
                        else:
                            df_out[col] = numeric.astype(target_dtype)
                    else:
                        df_out[col] = numeric.astype(target_dtype)
                        
                elif dtype_str.startswith('bool'):
                    # Boolean Family
                    try:
                        df_out[col] = series.astype('boolean')
                    except (TypeError, ValueError):
                        if not coerce_on_error:
                            raise
                        # Robust coercion
                        coerced_values = [
                            pd.NA if (result := _coerce_boolean_value(val)) is None else result
                            for val in series
                        ]
                        df_out[col] = pd.Series(coerced_values, index=series.index, dtype="boolean")
                        
                elif dtype_str.startswith('datetime'):
                    # Datetime Family
                    errors = 'coerce' if coerce_on_error else 'raise'
                    df_out[col] = pd.to_datetime(series, errors=errors)
                    
                else:
                    # String / Category / Other
                    df_out[col] = series.astype(target_dtype)
                    
            except Exception:
                if not coerce_on_error:
                    raise
                # If coercion is on, we might leave it as is or try best effort?
                pass
                
        return df_out
