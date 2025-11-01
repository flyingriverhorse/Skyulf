"""
Character cleaning utilities for removing special characters and trimming whitespace.

This module provides functions to clean text by removing unwanted characters
and trimming excessive whitespace.
"""

import re
from typing import Any, List, Literal, Optional, Union

import pandas as pd
from pandas.api import types as pd_types

from ...core.exceptions import DataIngestionError
from .text_normalization import _auto_detect_text_columns


def remove_special_characters(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    mode: Literal[
        "keep_alphanumeric",
        "keep_alphanumeric_space",
        "letters_only",
        "digits_only",
    ] = "keep_alphanumeric",
    replacement: str = "",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Remove special characters from text columns.
    
    Provides preset modes for common character cleaning tasks,
    allowing you to keep only specific character types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean.
    columns : str, list of str, or None, default=None
        Column(s) to clean. If None, auto-detects text columns.
    mode : str, default='keep_alphanumeric'
        Cleaning mode:
        - 'keep_alphanumeric': Keep only letters and numbers (A-Z, a-z, 0-9)
        - 'keep_alphanumeric_space': Keep letters, numbers, and spaces
        - 'letters_only': Keep only letters (A-Z, a-z)
        - 'digits_only': Keep only numbers (0-9)
    replacement : str, default=''
        String to replace removed characters with.
    inplace : bool, default=False
        If True, modify the DataFrame in place.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned text columns.
        
    Examples
    --------
    >>> df = pd.DataFrame({'code': ['ABC-123', 'XYZ#456', 'TEST@789']})
    >>> remove_special_characters(df, columns='code', mode='keep_alphanumeric')
         code
    0  ABC123
    1  XYZ456
    2  TEST789
    
    >>> df = pd.DataFrame({'text': ['Hello, World!', 'Test@#123']})
    >>> remove_special_characters(df, columns='text', mode='keep_alphanumeric_space')
              text
    0  Hello World
    1     Test 123
    
    >>> df = pd.DataFrame({'mixed': ['ABC123', 'XYZ789']})
    >>> remove_special_characters(df, columns='mixed', mode='letters_only')
      mixed
    0   ABC
    1   XYZ
    
    >>> df = pd.DataFrame({'mixed': ['ABC123', 'XYZ789']})
    >>> remove_special_characters(df, columns='mixed', mode='digits_only')
      mixed
    0   123
    1   789
    
    Notes
    -----
    - When replacement=' ' and mode includes spaces, collapses multiple spaces
    - Preserves NA/None values
    - Auto-detection processes string/object/categorical columns only
    """
    if df.empty:
        return df if inplace else df.copy()
    
    # Validate mode
    valid_modes = {
        "keep_alphanumeric",
        "keep_alphanumeric_space",
        "letters_only",
        "digits_only",
    }
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
        
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise DataIngestionError(
                f"Columns not found in DataFrame: {missing}"
            )
        
        target_columns = columns
    
    # Create working DataFrame
    working_df = df if inplace else df.copy()
    
    # Define patterns for each mode
    pattern_map = {
        "keep_alphanumeric": re.compile(r"[^0-9A-Za-z]+"),
        "keep_alphanumeric_space": re.compile(r"[^0-9A-Za-z\s]+"),
        "letters_only": re.compile(r"[^A-Za-z]+"),
        "digits_only": re.compile(r"[^0-9]+"),
    }
    
    pattern = pattern_map[mode]
    collapse_whitespace = mode == "keep_alphanumeric_space" or replacement == " "
    
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
        
        string_series = series.astype("string")
        
        def _clean(entry: Any) -> Any:
            if entry is pd.NA or entry is None:
                return entry
            text = str(entry)
            cleaned = pattern.sub(replacement, text)
            if collapse_whitespace:
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return cleaned
        
        cleaned_series = string_series.map(_clean).astype("string")
        
        # Update column preserving dtype
        if pd_types.is_object_dtype(series):
            working_df[column] = cleaned_series.astype(object)
        elif pd_types.is_string_dtype(series):
            working_df[column] = cleaned_series.astype(series.dtype)
        else:
            working_df[column] = cleaned_series.astype("string")
    
    return working_df


def trim_whitespace(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    mode: Literal["leading", "trailing", "both"] = "both",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Trim leading and/or trailing whitespace from text columns.
    
    Removes whitespace characters (spaces, tabs, newlines) from the
    beginning and/or end of strings.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to process.
    columns : str, list of str, or None, default=None
        Column(s) to trim. If None, auto-detects text columns.
    mode : {'leading', 'trailing', 'both'}, default='both'
        Trimming mode:
        - 'leading': Remove only leading (left) whitespace
        - 'trailing': Remove only trailing (right) whitespace
        - 'both': Remove both leading and trailing whitespace
    inplace : bool, default=False
        If True, modify the DataFrame in place.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with trimmed text columns.
        
    Examples
    --------
    >>> df = pd.DataFrame({'text': ['  hello', 'world  ', '  test  ']})
    >>> trim_whitespace(df, columns='text', mode='both')
        text
    0  hello
    1  world
    2   test
    
    >>> df = pd.DataFrame({'name': ['  John', '  Jane  ', 'Bob  ']})
    >>> trim_whitespace(df, columns='name', mode='leading')
         name
    0    John
    1  Jane  
    2  Bob  
    
    >>> df = pd.DataFrame({'code': ['ABC  ', '  XYZ', '  123  ']})
    >>> trim_whitespace(df, columns='code', mode='trailing')
        code
    0     ABC
    1    XYZ
    2    123
    
    Notes
    -----
    - Preserves NA/None values
    - Auto-detection processes string/object/categorical columns only
    - Whitespace includes spaces, tabs, newlines, and other Unicode whitespace
    """
    if df.empty:
        return df if inplace else df.copy()
    
    # Validate mode
    valid_modes = {"leading", "trailing", "both"}
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
        
        string_series = series.astype("string")
        
        # Apply appropriate trimming
        if mode == "leading":
            transformed = string_series.str.lstrip()
        elif mode == "trailing":
            transformed = string_series.str.rstrip()
        else:  # both
            transformed = string_series.str.strip()
        
        # Update column preserving dtype
        if pd_types.is_object_dtype(series):
            working_df[column] = transformed.astype(object)
        elif pd_types.is_string_dtype(series):
            working_df[column] = transformed.astype(series.dtype)
        else:
            working_df[column] = transformed.astype("string")
    
    return working_df
