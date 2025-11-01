"""
Regex-based text cleaning utilities with preset patterns.

This module provides regex-driven text cleaning with common presets
for date normalization, whitespace collapse, and digit extraction.
"""

import re
from typing import Any, List, Literal, Optional, Union

import pandas as pd
from pandas.api import types as pd_types

from ...core.exceptions import DataIngestionError
from .text_normalization import _auto_detect_text_columns


# Two-digit year pivot (years < 50 are 20xx, >= 50 are 19xx)
TWO_DIGIT_YEAR_PIVOT = 50


def regex_cleanup(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    mode: Literal[
        "normalize_slash_dates",
        "collapse_whitespace",
        "extract_digits",
        "custom",
    ] = "normalize_slash_dates",
    pattern: Optional[str] = None,
    replacement: str = "",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Apply regex-based cleanup transformations to text columns.
    
    Provides preset regex patterns for common text cleaning tasks,
    as well as support for custom regex patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean.
    columns : str, list of str, or None, default=None
        Column(s) to clean. If None, auto-detects text columns.
    mode : str, default='normalize_slash_dates'
        Cleanup mode:
        - 'normalize_slash_dates': Convert M/D/YY or M-D-YYYY to YYYY-MM-DD
        - 'collapse_whitespace': Replace multiple spaces with single space
        - 'extract_digits': Extract only digits from text
        - 'custom': Use custom regex pattern
    pattern : str, optional
        Custom regex pattern (required when mode='custom').
    replacement : str, default=''
        Replacement string for custom pattern.
    inplace : bool, default=False
        If True, modify the DataFrame in place.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned text columns.
        
    Raises
    ------
    DataIngestionError
        If custom mode requires pattern or pattern is invalid.
        
    Examples
    --------
    >>> df = pd.DataFrame({'date': ['12/31/2023', '1-5-24', '06/15/2024']})
    >>> regex_cleanup(df, columns='date', mode='normalize_slash_dates')
            date
    0  2023-12-31
    1  2024-01-05
    2  2024-06-15
    
    >>> df = pd.DataFrame({'text': ['hello    world', 'too   many     spaces']})
    >>> regex_cleanup(df, columns='text', mode='collapse_whitespace')
                   text
    0      hello world
    1  too many spaces
    
    >>> df = pd.DataFrame({'code': ['ABC-123-XYZ', 'ID#456']})
    >>> regex_cleanup(df, columns='code', mode='extract_digits')
      code
    0  123
    1  456
    
    >>> df = pd.DataFrame({'email': ['user@example.com', 'admin@test.org']})
    >>> regex_cleanup(df, columns='email', mode='custom', 
    ...               pattern=r'@.*', replacement='@domain.com')
             email
    0  user@domain.com
    1  admin@domain.com
    
    Notes
    -----
    - Date normalization assumes M/D/Y or M-D-Y format
    - Two-digit years < 50 are interpreted as 20xx, >= 50 as 19xx
    - Custom patterns must be valid regex
    - Preserves NA/None values
    """
    if df.empty:
        return df if inplace else df.copy()
    
    # Validate mode
    valid_modes = {"normalize_slash_dates", "collapse_whitespace", "extract_digits", "custom"}
    if mode not in valid_modes:
        raise DataIngestionError(
            f"Invalid mode '{mode}'. Must be one of {valid_modes}"
        )
    
    # Validate custom mode requirements
    if mode == "custom":
        if not pattern or not isinstance(pattern, str):
            raise DataIngestionError(
                "Custom mode requires 'pattern' parameter"
            )
        try:
            compiled_pattern = re.compile(pattern)
        except re.error as exc:
            raise DataIngestionError(
                f"Invalid regex pattern: {exc}"
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
    
    # Compile patterns for preset modes
    date_pattern = re.compile(r"\b(\d{1,2})[\/-](\d{1,2})[\/-](\d{2,4})\b")
    
    def _normalize_date_text(text: str) -> str:
        """Normalize date format to YYYY-MM-DD."""
        def _replace(match: re.Match[str]) -> str:
            month = int(match.group(1))
            day = int(match.group(2))
            year_token = match.group(3)
            
            # Handle 2-digit years
            if len(year_token) == 2:
                year_value = int(year_token)
                year_value += 2000 if year_value < TWO_DIGIT_YEAR_PIVOT else 1900
            else:
                year_value = int(year_token)
            
            return f"{year_value:04d}-{month:02d}-{day:02d}"
        
        return date_pattern.sub(_replace, text)
    
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
        
        # Define transformation function
        def _transform(entry: Any) -> Any:
            if entry is pd.NA or entry is None:
                return entry
            text = str(entry)
            if not text:
                return text
            
            if mode == "normalize_slash_dates":
                return _normalize_date_text(text)
            elif mode == "collapse_whitespace":
                return re.sub(r"\s+", " ", text).strip()
            elif mode == "extract_digits":
                return re.sub(r"\D+", "", text)
            elif mode == "custom" and compiled_pattern is not None:
                return compiled_pattern.sub(replacement, text)
            return text
        
        # Apply transformation
        transformed = string_series.map(_transform).astype("string")
        
        # Update column preserving dtype
        if pd_types.is_object_dtype(series):
            working_df[column] = transformed.astype(object)
        elif pd_types.is_string_dtype(series):
            working_df[column] = transformed.astype(series.dtype)
        else:
            working_df[column] = transformed.astype("string")
    
    return working_df
