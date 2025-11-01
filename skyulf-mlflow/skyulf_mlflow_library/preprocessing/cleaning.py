"""Data cleaning utilities for preprocessing."""

from typing import List, Optional, Union

import pandas as pd
import numpy as np

from skyulf_mlflow_library.core.types import DataFrame
from skyulf_mlflow_library.exceptions import PreprocessingError


def drop_missing_rows(
    df: DataFrame,
    threshold: Optional[float] = None,
    how: str = 'any',
    subset: Optional[List[str]] = None,
    inplace: bool = False
) -> DataFrame:
    """
    Remove rows with missing values.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    threshold : float, optional
        Percentage threshold (0-100). Rows with missing % >= threshold are dropped.
        If None, uses 'how' parameter instead.
    how : {'any', 'all'}, default='any'
        Determines if row is removed when 'threshold' is None:
        - 'any': Drop rows that contain any missing values
        - 'all': Drop rows where all values are missing
    subset : list of str, optional
        Labels along other axis to consider.
    inplace : bool, default=False
        If True, do operation inplace and return None.

    Returns
    -------
    DataFrame or None
        DataFrame with missing rows removed, or None if inplace=True.

    Examples
    --------
    >>> from skyulf_mlflow_library.preprocessing import drop_missing_rows
    >>> df_clean = drop_missing_rows(df, threshold=50)  # Drop rows with >50% missing
    >>> df_clean = drop_missing_rows(df, how='any')  # Drop rows with any missing
    """
    if not inplace:
        df = df.copy()

    try:
        if threshold is not None:
            # Calculate percentage of missing values per row
            missing_pct = df.isna().mean(axis=1) * 100
            mask = missing_pct < threshold
            result = df[mask]
        else:
            # Use pandas dropna with how parameter
            result = df.dropna(axis=0, how=how, subset=subset)

        if inplace:
            df.drop(df.index, inplace=True)
            df.update(result)
            return None
        return result

    except Exception as e:
        raise PreprocessingError(f"Failed to drop missing rows: {str(e)}")


def drop_missing_columns(
    df: DataFrame,
    threshold: Optional[float] = None,
    columns: Optional[List[str]] = None,
    inplace: bool = False
) -> DataFrame:
    """
    Remove columns with missing values.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    threshold : float, optional
        Percentage threshold (0-100). Columns with missing % >= threshold are dropped.
        If None, drops specified columns that have any missing values.
    columns : list of str, optional
        Specific columns to check. If None, checks all columns.
    inplace : bool, default=False
        If True, do operation inplace and return None.

    Returns
    -------
    DataFrame or None
        DataFrame with missing columns removed, or None if inplace=True.

    Examples
    --------
    >>> from skyulf_mlflow_library.preprocessing import drop_missing_columns
    >>> df_clean = drop_missing_columns(df, threshold=30)  # Drop cols with >30% missing
    """
    if not inplace:
        df = df.copy()

    try:
        columns_to_check = columns if columns else df.columns.tolist()
        
        if threshold is not None:
            # Calculate percentage of missing values per column
            missing_pct = df[columns_to_check].isna().mean() * 100
            columns_to_drop = missing_pct[missing_pct >= threshold].index.tolist()
        else:
            # Drop columns that have any missing values
            columns_to_drop = [col for col in columns_to_check if df[col].isna().any()]

        result = df.drop(columns=columns_to_drop)

        if inplace:
            for col in columns_to_drop:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            return None
        return result

    except Exception as e:
        raise PreprocessingError(f"Failed to drop missing columns: {str(e)}")


def remove_duplicates(
    df: DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first',
    inplace: bool = False
) -> DataFrame:
    """
    Remove duplicate rows from DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    subset : list of str, optional
        Only consider certain columns for identifying duplicates.
        By default, use all columns.
    keep : {'first', 'last', False}, default='first'
        Determines which duplicates to keep:
        - 'first': Keep first occurrence
        - 'last': Keep last occurrence
        - False: Drop all duplicates
    inplace : bool, default=False
        If True, do operation inplace and return None.

    Returns
    -------
    DataFrame or None
        DataFrame with duplicates removed, or None if inplace=True.

    Examples
    --------
    >>> from skyulf_mlflow_library.preprocessing import remove_duplicates
    >>> df_clean = remove_duplicates(df, keep='first')
    >>> df_clean = remove_duplicates(df, subset=['col1', 'col2'])
    """
    try:
        return df.drop_duplicates(subset=subset, keep=keep, inplace=inplace)
    except Exception as e:
        raise PreprocessingError(f"Failed to remove duplicates: {str(e)}")


def remove_outliers(
    df: DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'iqr',
    threshold: float = 1.5,
    inplace: bool = False
) -> DataFrame:
    """
    Remove outliers from DataFrame using various methods.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    columns : list of str, optional
        Columns to check for outliers. If None, checks all numeric columns.
    method : {'iqr', 'zscore'}, default='iqr'
        Method to use for outlier detection:
        - 'iqr': Interquartile range method
        - 'zscore': Z-score method
    threshold : float, default=1.5
        Threshold for outlier detection:
        - For 'iqr': IQR multiplier (typically 1.5 or 3.0)
        - For 'zscore': Z-score threshold (typically 3.0)
    inplace : bool, default=False
        If True, do operation inplace and return None.

    Returns
    -------
    DataFrame or None
        DataFrame with outliers removed, or None if inplace=True.

    Examples
    --------
    >>> from skyulf_mlflow_library.preprocessing import remove_outliers
    >>> df_clean = remove_outliers(df, method='iqr', threshold=1.5)
    >>> df_clean = remove_outliers(df, columns=['age', 'salary'], method='zscore', threshold=3.0)
    """
    if not inplace:
        df = df.copy()

    try:
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        mask = pd.Series(True, index=df.index)

        for col in columns:
            if col not in df.columns:
                continue

            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                z_scores = np.abs((df[col] - mean) / std)
                col_mask = z_scores < threshold
            
            else:
                raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")

            mask &= col_mask

        result = df[mask]

        if inplace:
            df.drop(df.index, inplace=True)
            df.update(result)
            return None
        return result

    except Exception as e:
        raise PreprocessingError(f"Failed to remove outliers: {str(e)}")


def fill_missing(
    df: DataFrame,
    strategy: Union[str, dict],
    value: Optional[Union[float, str]] = None,
    inplace: bool = False
) -> DataFrame:
    """
    Fill missing values with specified strategy.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    strategy : str or dict
        Strategy to use:
        - 'constant': Fill with constant value
        - 'mean': Fill with column mean
        - 'median': Fill with column median
        - 'mode': Fill with column mode
        - 'ffill': Forward fill
        - 'bfill': Backward fill
        - dict: Dictionary mapping column names to strategies
    value : float or str, optional
        Value to use when strategy='constant'.
    inplace : bool, default=False
        If True, do operation inplace and return None.

    Returns
    -------
    DataFrame or None
        DataFrame with filled values, or None if inplace=True.

    Examples
    --------
    >>> from skyulf_mlflow_library.preprocessing import fill_missing
    >>> df_filled = fill_missing(df, strategy='mean')
    >>> df_filled = fill_missing(df, strategy='constant', value=0)
    >>> df_filled = fill_missing(df, strategy={'col1': 'mean', 'col2': 'median'})
    """
    if not inplace:
        df = df.copy()

    try:
        if isinstance(strategy, dict):
            # Different strategy per column
            for col, col_strategy in strategy.items():
                if col not in df.columns:
                    continue
                df[col] = _fill_column(df[col], col_strategy, value)
        else:
            # Same strategy for all columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = _fill_column(df[col], strategy, value)

        if inplace:
            return None
        return df

    except Exception as e:
        raise PreprocessingError(f"Failed to fill missing values: {str(e)}")


def _fill_column(series: pd.Series, strategy: str, value: Optional[Union[float, str]] = None) -> pd.Series:
    """Helper function to fill a single column."""
    if strategy == 'constant':
        return series.fillna(value)
    elif strategy == 'mean':
        return series.fillna(series.mean())
    elif strategy == 'median':
        return series.fillna(series.median())
    elif strategy == 'mode':
        mode_val = series.mode()
        if len(mode_val) > 0:
            return series.fillna(mode_val[0])
        return series
    elif strategy == 'ffill':
        return series.fillna(method='ffill')
    elif strategy == 'bfill':
        return series.fillna(method='bfill')
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
