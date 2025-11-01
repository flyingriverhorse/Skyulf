"""
Date standardization utilities for parsing and formatting dates.

This module provides functions to parse dates from various formats
and standardize them to a consistent output format.
"""

from typing import Any, List, Literal, Optional, Union

import pandas as pd
from pandas.api import types as pd_types

from ...core.exceptions import DataIngestionError


def standardize_dates(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    output_format: str = "%Y-%m-%d",
    infer_format: bool = True,
    dayfirst: bool = False,
    errors: Literal["raise", "coerce", "ignore"] = "coerce",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Standardize date columns to a consistent format.
    
    Parses dates from various input formats and converts them to a
    standardized output format. Handles both datetime and string columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to process.
    columns : str or list of str
        Column(s) containing dates to standardize.
    output_format : str, default='%Y-%m-%d'
        Output date format using strftime format codes.
        Common formats:
        - '%Y-%m-%d': ISO date (2024-01-31)
        - '%Y-%m-%d %H:%M:%S': ISO datetime (2024-01-31 14:30:00)
        - '%m/%d/%Y': US format (01/31/2024)
        - '%d/%m/%Y': European format (31/01/2024)
    infer_format : bool, default=True
        If True, attempt to infer the date format automatically.
    dayfirst : bool, default=False
        If True, interpret ambiguous dates with day first (DD/MM/YYYY).
        If False, interpret with month first (MM/DD/YYYY).
    errors : {'raise', 'coerce', 'ignore'}, default='coerce'
        How to handle parsing errors:
        - 'raise': Raise an exception
        - 'coerce': Set invalid dates to NaT (Not a Time)
        - 'ignore': Return original values
    inplace : bool, default=False
        If True, modify the DataFrame in place.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized date columns.
        
    Raises
    ------
    DataIngestionError
        If columns don't exist or output_format is invalid.
        
    Examples
    --------
    >>> df = pd.DataFrame({'date': ['2024-01-31', '2024/02/28', '03-15-2024']})
    >>> standardize_dates(df, columns='date', output_format='%Y-%m-%d')
            date
    0  2024-01-31
    1  2024-02-28
    2  2024-03-15
    
    >>> df = pd.DataFrame({'created': ['01/31/2024', '02/28/2024']})
    >>> standardize_dates(df, columns='created', output_format='%d/%m/%Y')
        created
    0  31/01/2024
    1  28/02/2024
    
    >>> df = pd.DataFrame({'timestamp': ['2024-01-31 14:30', '2024-02-28 16:45']})
    >>> standardize_dates(df, columns='timestamp', 
    ...                   output_format='%Y-%m-%d %H:%M:%S')
              timestamp
    0  2024-01-31 14:30:00
    1  2024-02-28 16:45:00
    
    >>> df = pd.DataFrame({'date': ['31/01/2024', '28/02/2024']})
    >>> standardize_dates(df, columns='date', dayfirst=True)
            date
    0  2024-01-31
    1  2024-02-28
    
    Notes
    -----
    - Works with both datetime and string columns
    - Preserves NA/None values as NaT
    - With errors='coerce', unparseable dates become NaT
    - Removes timezone information from datetime objects
    - Output is always string format for consistency
    """
    if df.empty:
        return df if inplace else df.copy()
    
    # Validate output format
    if not isinstance(output_format, str):
        raise DataIngestionError(
            "Parameter 'output_format' must be a string"
        )
    
    # Test format string validity
    try:
        pd.Timestamp('2024-01-01').strftime(output_format)
    except (ValueError, TypeError) as e:
        raise DataIngestionError(
            f"Invalid output_format '{output_format}': {e}"
        )
    
    # Handle column selection
    if isinstance(columns, str):
        columns = [columns]
    
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise DataIngestionError(
            f"Columns not found in DataFrame: {missing}"
        )
    
    # Create working DataFrame
    working_df = df if inplace else df.copy()
    
    # Process each column
    for column in columns:
        series = working_df[column]
        
        # Convert to datetime
        candidate_datetime: Optional[pd.Series] = None
        
        if pd_types.is_datetime64_any_dtype(series):
            # Already datetime, just convert with error handling
            candidate_datetime = pd.to_datetime(series, errors=errors)
        elif (
            pd_types.is_string_dtype(series)
            or pd_types.is_object_dtype(series)
            or pd_types.is_categorical_dtype(series)
        ):
            # Parse from string
            try:
                candidate_datetime = pd.to_datetime(
                    series,
                    errors=errors,
                    infer_datetime_format=infer_format,
                    dayfirst=dayfirst,
                )
            except Exception as e:
                if errors == "raise":
                    raise DataIngestionError(
                        f"Failed to parse dates in column '{column}': {e}"
                    )
                elif errors == "ignore":
                    continue
                else:  # coerce
                    candidate_datetime = pd.to_datetime(series, errors="coerce")
        else:
            # Not a suitable column type
            if errors == "raise":
                raise DataIngestionError(
                    f"Column '{column}' is not a suitable date type"
                )
            continue
        
        if candidate_datetime is None:
            continue
        
        # Remove timezone information for consistency
        try:
            candidate_datetime = candidate_datetime.dt.tz_localize(None)  # type: ignore
        except (AttributeError, TypeError, ValueError):
            pass
        
        # Format to string
        mask = candidate_datetime.notna()
        if not mask.any():
            # No valid dates found
            if errors == "raise":
                raise DataIngestionError(
                    f"No valid dates found in column '{column}'"
                )
            continue
        
        # Apply formatting
        formatted_values = candidate_datetime.dt.strftime(output_format)
        
        # Create result series preserving NA values
        result_series = pd.Series(index=series.index, dtype="string")
        result_series.loc[mask] = formatted_values.loc[mask].astype("string")
        
        # Update column
        working_df[column] = result_series
    
    return working_df
