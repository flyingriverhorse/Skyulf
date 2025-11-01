"""
Alias mapping utilities for standardizing terminology and fixing typos.

This module provides functions to map variations of terms to standard values,
useful for cleaning categorical data and ensuring consistency.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pandas.api import types as pd_types

from ...core.exceptions import DataIngestionError
from .text_normalization import _auto_detect_text_columns


def _normalize_key(text: str) -> str:
    """Normalize text for matching (lowercase, strip)."""
    return str(text).strip().lower()


def replace_aliases(
    df: pd.DataFrame,
    column: str,
    aliases: Dict[str, Union[str, List[str]]],
    case_sensitive: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Replace alias values with standardized terms.
    
    Maps variations of terms (typos, abbreviations, alternative spellings)
    to standard canonical values. Useful for cleaning categorical data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to process.
    column : str
        Column containing values to standardize.
    aliases : dict
        Mapping of canonical values to their aliases.
        Format: {'canonical': 'alias'} or {'canonical': ['alias1', 'alias2', ...]}
    case_sensitive : bool, default=False
        If False, matching is case-insensitive.
    inplace : bool, default=False
        If True, modify the DataFrame in place.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized values.
        
    Raises
    ------
    DataIngestionError
        If column doesn't exist or aliases format is invalid.
        
    Examples
    --------
    >>> df = pd.DataFrame({'status': ['active', 'Active', 'ACTIVE', 'inactive']})
    >>> aliases = {'active': ['Active', 'ACTIVE'], 'inactive': ['Inactive', 'INACTIVE']}
    >>> replace_aliases(df, 'status', aliases)
         status
    0    active
    1    active
    2    active
    3  inactive
    
    >>> df = pd.DataFrame({'country': ['US', 'USA', 'United States', 'UK']})
    >>> aliases = {
    ...     'United States': ['US', 'USA', 'U.S.', 'U.S.A.'],
    ...     'United Kingdom': ['UK', 'U.K.', 'Britain']
    ... }
    >>> replace_aliases(df, 'country', aliases)
              country
    0   United States
    1   United States
    2   United States
    3  United Kingdom
    
    >>> df = pd.DataFrame({'dept': ['sales', 'sls', 'Sale', 'marketing', 'mktg']})
    >>> aliases = {
    ...     'Sales': ['sls', 'sale', 'sales dept'],
    ...     'Marketing': ['mktg', 'mkt', 'marketing dept']
    ... }
    >>> replace_aliases(df, 'dept', aliases, case_sensitive=False)
            dept
    0      Sales
    1      Sales
    2      Sales
    3  Marketing
    4  Marketing
    
    Notes
    -----
    - By default, matching is case-insensitive
    - Preserves NA/None values
    - Unmatched values remain unchanged
    - First matching alias wins if multiple mappings exist
    """
    if df.empty:
        return df if inplace else df.copy()
    
    # Validate column exists
    if column not in df.columns:
        raise DataIngestionError(
            f"Column '{column}' not found in DataFrame"
        )
    
    # Validate aliases format
    if not isinstance(aliases, dict):
        raise DataIngestionError(
            "Parameter 'aliases' must be a dictionary"
        )
    
    # Create working DataFrame
    working_df = df if inplace else df.copy()
    
    # Build reverse mapping (alias -> canonical)
    reverse_map: Dict[str, str] = {}
    
    for canonical, alias_values in aliases.items():
        if isinstance(alias_values, str):
            alias_values = [alias_values]
        elif not isinstance(alias_values, list):
            continue
        
        for alias in alias_values:
            if not isinstance(alias, str):
                continue
            
            key = alias if case_sensitive else _normalize_key(alias)
            reverse_map[key] = str(canonical)
    
    # If no valid mappings, return unchanged
    if not reverse_map:
        return working_df
    
    # Get the series
    series = working_df[column]
    
    # Skip if not text-like
    if not (
        pd_types.is_string_dtype(series)
        or pd_types.is_object_dtype(series)
        or pd_types.is_categorical_dtype(series)
    ):
        return working_df
    
    # Define mapping function
    def _map_value(value: Any) -> Any:
        if value is pd.NA or value is None:
            return value
        
        str_value = str(value)
        lookup_key = str_value if case_sensitive else _normalize_key(str_value)
        
        return reverse_map.get(lookup_key, value)
    
    # Apply mapping
    mapped_series = series.map(_map_value)
    
    # Update column preserving dtype
    if pd_types.is_object_dtype(series):
        working_df[column] = mapped_series.astype(object)
    elif pd_types.is_string_dtype(series):
        working_df[column] = mapped_series.astype(series.dtype)
    elif pd_types.is_categorical_dtype(series):
        # Rebuild categories to include new canonical values
        all_values = set(mapped_series.dropna().unique())
        working_df[column] = pd.Categorical(mapped_series, categories=sorted(all_values))
    else:
        working_df[column] = mapped_series
    
    return working_df
