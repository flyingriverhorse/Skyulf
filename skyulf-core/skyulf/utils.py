import pandas as pd
import numpy as np
from typing import List, Set, Union, Tuple, Optional, Dict, Any, Callable
from .data.dataset import SplitDataset

def get_data_stats(data: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series], SplitDataset]) -> Tuple[int, Set[str]]:
    """
    Calculates row count and column set for various data structures.
    Supports DataFrame, (X, y) tuple, and SplitDataset.
    """
    rows = 0
    cols = set()
    
    if isinstance(data, pd.DataFrame):
        rows = len(data)
        cols = set(data.columns)
    elif isinstance(data, tuple) and len(data) == 2:
        # Handle (X, y) tuple
        # Check if first element is DataFrame/Series
        if hasattr(data[0], 'shape'):
             rows = len(data[0])
             if hasattr(data[0], 'columns'):
                 cols = set(data[0].columns)
    elif isinstance(data, SplitDataset):
        # Sum rows from all splits
        # Train
        r, c = get_data_stats(data.train)
        rows += r
        cols = c # Assume columns are same
        
        # Test
        r, _ = get_data_stats(data.test)
        rows += r
        
        # Validation
        if data.validation is not None:
            r, _ = get_data_stats(data.validation)
            rows += r
            
    return rows, cols

def unpack_pipeline_input(data: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]) -> Tuple[pd.DataFrame, Optional[pd.Series], bool]:
    """
    Unpacks input which might be a DataFrame or a (X, y) tuple.
    Returns: (X, y, is_tuple)
    """
    if isinstance(data, tuple):
        return data[0], data[1], True
    return data, None, False

def pack_pipeline_output(X: pd.DataFrame, y: Optional[pd.Series], was_tuple: bool) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """
    Packs output back into a tuple if the input was a tuple and y is present.
    Otherwise, if y is present, concatenates it back to X.
    """
    if was_tuple and y is not None:
        return (X, y)
    
    if y is not None:
        # Re-attach y to X
        # Ensure indices align (they should if coming from same operation)
        return pd.concat([X, y], axis=1)
        
    return X

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
