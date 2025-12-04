import pandas as pd
import numpy as np
from typing import List, Set

def _is_binary_numeric(series: pd.Series) -> bool:
    """Check if a numeric series contains only 0s and 1s (or close to them)."""
    unique_vals = series.dropna().unique()
    if len(unique_vals) > 2:
        return False
    
    # Check if values are close to 0 or 1
    for val in unique_vals:
        if not (np.isclose(val, 0) or np.isclose(val, 1)):
            return False
    return True

def detect_numeric_columns(frame: pd.DataFrame) -> List[str]:
    """
    Find numeric-like columns that have more than one non-binary value.
    
    Logic ported from V1 (core/shared/utils.py):
    1. Excludes boolean columns.
    2. Tries to convert strings to numbers (e.g. "1.5").
    3. Excludes binary (0/1) columns.
    4. Excludes constant columns (0 or 1 unique value).
    """
    detected: List[str] = []
    seen: Set[str] = set()

    for column in frame.columns:
        if column in seen:
            continue
            
        series = frame[column]
        dtype = series.dtype
        
        # 1. Exclude explicit booleans
        if pd.api.types.is_bool_dtype(dtype):
            continue

        # 2. Try to convert to numeric (handles strings like "1.5")
        numeric_series = pd.to_numeric(series, errors="coerce")
        valid = numeric_series.dropna()
        
        if valid.empty:
            continue

        # 3. Exclude 0/1 columns (Binary)
        if _is_binary_numeric(valid):
            continue

        # 4. Exclude constant columns
        if valid.nunique() < 2:
            continue

        detected.append(column)
        seen.add(column)

    return detected

from typing import Dict, Any, Callable, Optional

def resolve_columns(
    df: pd.DataFrame, 
    config: Dict[str, Any], 
    default_selection_func: Optional[Callable[[pd.DataFrame], List[str]]] = None,
    target_column_key: str = 'target_column'
) -> List[str]:
    """
    Resolves the list of columns to process based on configuration and auto-detection.
    
    Logic:
    1. If 'columns' is explicitly provided in config, use it (filtering for existence in df).
       - Does NOT exclude target column if explicitly requested.
    2. If 'columns' is missing/empty and default_selection_func is provided:
       - Auto-detect columns using the function.
       - Exclude the target column (if specified in config) from this auto-detected list.
    3. Filter to ensure all columns exist in the dataframe.
    """
    cols = config.get('columns')
    
    # Case 1: Explicit columns provided
    if cols:
        # Just filter for existence
        return [c for c in cols if c in df.columns]
        
    # Case 2: Auto-detection
    if default_selection_func:
        cols = default_selection_func(df)
        
        # Exclude target column during auto-detection
        target_col = config.get(target_column_key)
        if target_col and target_col in cols:
            cols = [c for c in cols if c != target_col]
            
        # Filter for existence (though auto-detect usually returns existing cols)
        return [c for c in cols if c in df.columns]
        
    return []
