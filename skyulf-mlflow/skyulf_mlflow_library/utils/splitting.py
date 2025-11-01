"""
Data splitting utilities for train/test/validation splits.

This module provides enhanced train-test splitting with stratification,
group-based splitting, and time-series splitting support.
"""

from typing import Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import (
    train_test_split as sklearn_train_test_split,
    StratifiedShuffleSplit,
    GroupShuffleSplit,
)


def train_test_split(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: Optional[Union[pd.Series, str]] = None,
    groups: Optional[Union[pd.Series, str]] = None,
    shuffle: bool = True,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
]:
    """
    Split data into train and test sets with enhanced options.
    
    This is an enhanced wrapper around sklearn's train_test_split with
    additional support for stratification by column name and group-based splitting.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input features to split.
    y : pd.Series, optional
        Target variable to split. If None, only returns X_train, X_test.
    test_size : float, default=0.2
        Proportion of data to include in test split (0.0 to 1.0).
    random_state : int, optional
        Random seed for reproducibility.
    stratify : pd.Series or str, optional
        Column to use for stratified sampling. Can be:
        - pd.Series: Stratification variable
        - str: Column name in X to use for stratification
        - None: No stratification
    groups : pd.Series or str, optional
        Groups for group-based splitting. Samples from same group
        stay together in either train or test set.
    shuffle : bool, default=True
        Whether to shuffle data before splitting.
    
    Returns
    -------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series (if y is provided)
        Training target.
    y_test : pd.Series (if y is provided)
        Test target.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from skyulf_mlflow_library.utils import train_test_split
    >>> 
    >>> # Basic split
    >>> df = pd.DataFrame({
    ...     'feature1': range(100),
    ...     'feature2': range(100, 200),
    ...     'target': [0] * 50 + [1] * 50
    ... })
    >>> X = df[['feature1', 'feature2']]
    >>> y = df['target']
    >>> 
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.2, random_state=42
    ... )
    >>> 
    >>> # Stratified split (preserve class distribution)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.2, stratify=y, random_state=42
    ... )
    >>> 
    >>> # Stratify by column name
    >>> df_with_category = df.copy()
    >>> df_with_category['category'] = ['A'] * 25 + ['B'] * 25 + ['C'] * 25 + ['D'] * 25
    >>> X_train, X_test = train_test_split(
    ...     df_with_category, test_size=0.2, stratify='category', random_state=42
    ... )
    >>> 
    >>> # Group-based split (keep user sessions together)
    >>> df_users = df.copy()
    >>> df_users['user_id'] = [i % 10 for i in range(100)]  # 10 users
    >>> X_train, X_test = train_test_split(
    ...     df_users, test_size=0.2, groups='user_id', random_state=42
    ... )
    """
    
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    
    # Handle stratify as column name
    if isinstance(stratify, str):
        if stratify not in X.columns:
            raise ValueError(f"Stratify column '{stratify}' not found in X")
        stratify = X[stratify]
    
    # Handle groups as column name
    if isinstance(groups, str):
        if groups not in X.columns:
            raise ValueError(f"Groups column '{groups}' not found in X")
        groups = X[groups]
    
    # Group-based splitting
    if groups is not None:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state
        )
        
        train_idx, test_idx = next(splitter.split(X, y, groups))
        
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        
        if y is not None:
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_test = y.iloc[test_idx].reset_index(drop=True)
            return X_train, X_test, y_train, y_test
        else:
            return X_train, X_test
    
    # Stratified splitting (when stratify is provided)
    if stratify is not None:
        if y is None:
            # Stratified split without target
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_state
            )
            
            train_idx, test_idx = next(splitter.split(X, stratify))
            
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_test = X.iloc[test_idx].reset_index(drop=True)
            
            return X_train, X_test
        else:
            # Use sklearn with stratification
            X_train, X_test, y_train, y_test = sklearn_train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
                shuffle=shuffle
            )
            return X_train, X_test, y_train, y_test
    
    # Standard sklearn split
    if y is not None:
        X_train, X_test, y_train, y_test = sklearn_train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle
        )
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test = sklearn_train_test_split(
            X,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle
        )
        return X_train, X_test


def train_val_test_split(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: Optional[Union[pd.Series, str]] = None,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
]:
    """
    Split data into train, validation, and test sets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input features.
    y : pd.Series, optional
        Target variable.
    test_size : float, default=0.2
        Proportion for test set.
    val_size : float, default=0.2
        Proportion for validation set (from remaining data after test split).
    random_state : int, optional
        Random seed.
    stratify : pd.Series or str, optional
        Stratification column.
    
    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame
        Train, validation, and test features.
    y_train, y_val, y_test : pd.Series (if y provided)
        Train, validation, and test targets.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.utils import train_val_test_split
    >>> 
    >>> X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    ...     X, y, test_size=0.2, val_size=0.2, random_state=42, stratify=y
    ... )
    """
    
    # First split: separate test set
    if y is not None:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Second split: separate validation from train
        # Adjust val_size to be proportion of remaining data
        adjusted_val_size = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=adjusted_val_size,
            random_state=random_state,
            stratify=y_temp if stratify is not None else None
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_temp, X_test = train_test_split(
            X,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        adjusted_val_size = val_size / (1 - test_size)
        
        X_train, X_val = train_test_split(
            X_temp,
            test_size=adjusted_val_size,
            random_state=random_state,
            stratify=X_temp[stratify] if isinstance(stratify, str) else stratify
        )
        
        return X_train, X_val, X_test
