from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .data.dataset import SplitDataset
from .engines import SkyulfDataFrame, get_engine


def get_data_stats(
    data: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, Any], SplitDataset],
) -> Tuple[int, Set[str]]:
    """
    Calculates row count and column set for various data structures.
    Supports DataFrame, (X, y) tuple, and SplitDataset.
    """
    rows = 0
    cols = set()

    # Check for DataFrame-like object (Pandas, Polars, Wrapper)
    if hasattr(data, "shape") and hasattr(data, "columns") and not isinstance(data, tuple):
        rows = data.shape[0]
        cols = set(data.columns)
    elif isinstance(data, tuple) and len(data) == 2:
        # Handle (X, y) tuple
        # Check if first element is DataFrame/Series
        if hasattr(data[0], "shape"):
            rows = data[0].shape[0]
            if hasattr(data[0], "columns"):
                cols = set(data[0].columns)
    elif isinstance(data, SplitDataset):
        # Sum rows from all splits
        # Train
        r, c = get_data_stats(data.train)
        rows += r
        cols = c  # Assume columns are same

        # Test
        r, _ = get_data_stats(data.test)
        rows += r

        # Validation
        if data.validation is not None:
            r, _ = get_data_stats(data.validation)
            rows += r

    return rows, cols


def unpack_pipeline_input(
    data: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, Any]],
) -> Tuple[Union[pd.DataFrame, SkyulfDataFrame], Optional[Any], bool]:
    """
    Unpacks input which might be a DataFrame or a (X, y) tuple.
    Returns: (X, y, is_tuple)
    """
    if isinstance(data, tuple):
        return data[0], data[1], True
    return data, None, False


def pack_pipeline_output(
    X: Union[pd.DataFrame, SkyulfDataFrame], 
    y: Optional[Any], 
    was_tuple: bool
) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, Any]]:
    """
    Packs output back into a tuple if the input was a tuple and y is present.
    Otherwise, if y is present, concatenates it back to X.
    """
    if was_tuple and y is not None:
        return (X, y)

    if y is not None:
        # Re-attach y to X
        engine = get_engine(X)
        
        if engine.__name__ == "PolarsEngine":
            # Polars specific concat
            import polars as pl
            if not isinstance(y, (pl.Series, pl.DataFrame)):
                # Try to convert y to Series
                try:
                    y = pl.Series(y)
                except:
                    pass # Let it fail or handle otherwise
            
            if isinstance(X, pl.DataFrame):
                return X.hstack([y] if isinstance(y, pl.Series) else y)
            # Handle Wrapper
            if hasattr(X, "_df"):
                 return X._df.hstack([y] if isinstance(y, pl.Series) else y)

        # Default to Pandas behavior (convert if needed or assume Pandas)
        if hasattr(X, "to_pandas"):
             X_pd = X.to_pandas()
        else:
             X_pd = X
             
        if hasattr(y, "to_pandas"):
             y_pd = y.to_pandas()
        else:
             y_pd = y
             
        # Ensure indices align (they should if coming from same operation)
        return pd.concat([X_pd, y_pd], axis=1)

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


def detect_numeric_columns(frame: Union[pd.DataFrame, SkyulfDataFrame]) -> List[str]:
    """
    Find numeric-like columns that have more than one non-binary value.

    Logic ported from V1 (core/shared/utils.py):
    1. Excludes boolean columns.
    2. Tries to convert strings to numbers (e.g. "1.5").
    3. Excludes binary (0/1) columns.
    4. Excludes constant columns (0 or 1 unique value).
    """
    # Convert to Pandas for analysis if not already
    # TODO: Implement native Polars logic for performance
    if not isinstance(frame, pd.DataFrame):
        if hasattr(frame, "to_pandas"):
            frame = frame.to_pandas()
            
    # ... existing logic ...
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
    df: Union[pd.DataFrame, SkyulfDataFrame],
    config: Dict[str, Any],
    default_selection_func: Optional[Callable[[Union[pd.DataFrame, SkyulfDataFrame]], List[str]]] = None,
    target_column_key: str = "target_column",
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
    cols = config.get("columns")

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
