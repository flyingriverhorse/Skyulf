"""
Text normalization utilities for consistent text casing.

This module provides functions to normalize text case across columns,
supporting lowercase, uppercase, title case, and sentence case transformations.
"""

from typing import Any, List, Literal, Optional, Union

import pandas as pd
from pandas.api import types as pd_types

from ...core.exceptions import DataIngestionError


def _auto_detect_text_columns(df: pd.DataFrame) -> List[str]:
    """Auto-detect text columns in a DataFrame."""
    text_columns = []
    for column in df.columns:
        series = df[column]
        if (
            pd_types.is_string_dtype(series)
            or pd_types.is_object_dtype(series)
            or pd_types.is_categorical_dtype(series)
        ):
            # Check if at least some values are strings
            sample = series.dropna().head(10)
            if len(sample) > 0 and all(isinstance(x, str) for x in sample):
                text_columns.append(column)
    return text_columns


def _sentence_case(value: Any) -> Any:
    """Convert text to sentence case (first letter uppercase, rest lowercase)."""
    if value is pd.NA or value is None:
        return value
    text = str(value)
    if not text:
        return text
    leading_len = len(text) - len(text.lstrip())
    leading = text[:leading_len]
    remainder = text[leading_len:]
    if not remainder:
        return text
    return leading + remainder[0].upper() + remainder[1:].lower()


def normalize_text_case(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    mode: Literal["lower", "upper", "title", "sentence"] = "lower",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Normalize text case across specified columns.
    
    Applies consistent casing transformations to text columns. If no columns
    are specified, automatically detects text columns and applies the transformation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to transform.
    columns : str, list of str, or None, default=None
        Column(s) to normalize. If None, auto-detects text columns.
    mode : {'lower', 'upper', 'title', 'sentence'}, default='lower'
        Case normalization mode:
        - 'lower': Convert to lowercase
        - 'upper': Convert to UPPERCASE
        - 'title': Convert To Title Case
        - 'sentence': Convert to sentence case (first letter uppercase)
    inplace : bool, default=False
        If True, modify the DataFrame in place.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with normalized text columns.
        
    Raises
    ------
    DataIngestionError
        If specified columns don't exist or invalid mode.
        
    Examples
    --------
    >>> df = pd.DataFrame({'name': ['JOHN DOE', 'jane smith', 'Bob JONES']})
    >>> normalize_text_case(df, columns='name', mode='title')
         name
    0  John Doe
    1  Jane Smith
    2  Bob Jones
    
    >>> df = pd.DataFrame({'city': ['NEW YORK', 'los angeles']})
    >>> normalize_text_case(df, mode='lower')
              city
    0     new york
    1  los angeles
    
    >>> df = pd.DataFrame({'desc': ['HELLO WORLD', 'GOOD MORNING']})
    >>> normalize_text_case(df, columns='desc', mode='sentence')
              desc
    0  Hello world
    1  Good morning
    
    Notes
    -----
    - Preserves NA/None values
    - Auto-detection only processes string/object/categorical columns
    - Title case may produce unexpected results for some text patterns
    """
    if df.empty:
        return df if inplace else df.copy()
    
    # Validate mode
    valid_modes = {"lower", "upper", "title", "sentence"}
    if mode not in valid_modes:
        raise DataIngestionError(
            f"Invalid mode '{mode}'. Must be one of {valid_modes}"
        )
    
    # Handle column selection
    if columns is None:
        target_columns = _auto_detect_text_columns(df)
        if not target_columns:
            return df if inplace else df.copy()
    else:
        if isinstance(columns, str):
            columns = [columns]
        
        # Check for missing columns
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise DataIngestionError(
                f"Columns not found in DataFrame: {missing}"
            )
        
        target_columns = columns
    
    # Create working DataFrame
    working_df = df if inplace else df.copy()
    
    # Process each column
    for column in target_columns:
        series = working_df[column]
        
        # Skip non-text columns
        if not (
            pd_types.is_string_dtype(series)
            or pd_types.is_object_dtype(series)
            or pd_types.is_categorical_dtype(series)
        ):
            continue
        
        # Convert to string type
        string_series = series.astype("string")
        
        # Apply transformation
        if mode == "upper":
            normalized = string_series.str.upper()
        elif mode == "title":
            normalized = string_series.str.title()
        elif mode == "sentence":
            normalized = string_series.map(_sentence_case).astype("string")
        else:  # lower
            normalized = string_series.str.lower()
        
        # Update the column preserving dtype
        if pd_types.is_object_dtype(series):
            working_df[column] = normalized.astype(object)
        elif pd_types.is_string_dtype(series):
            working_df[column] = normalized.astype(series.dtype)
        else:
            working_df[column] = normalized.astype("string")
    
    return working_df
