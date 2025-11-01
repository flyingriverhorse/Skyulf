"""
Smart binning transformer for discretizing continuous features.

This module provides the SmartBinning transformer with multiple strategies
for converting continuous variables into categorical bins.
"""

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from ...core.base import BaseTransformer
from ...core.exceptions import FeatureEngineeringError


class SmartBinning(BaseTransformer):
    """
    Discretize continuous features into bins using multiple strategies.
    
    Supports four binning strategies: equal_width, equal_frequency (quantile),
    custom bin edges, and KBinsDiscretizer (with kmeans/quantile/uniform).
    
    Parameters
    ----------
    strategy : {'equal_width', 'equal_frequency', 'custom', 'kbins'}, default='equal_width'
        Binning strategy:
        - 'equal_width': Bins of equal width
        - 'equal_frequency': Bins with equal number of samples (quantiles)
        - 'custom': User-defined bin edges
        - 'kbins': Use sklearn's KBinsDiscretizer
    columns : list of str, optional
        Columns to bin. If None, bins all numeric columns.
    n_bins : int, default=5
        Number of bins (for equal_width, equal_frequency, kbins).
    bins : dict, optional
        Custom bin edges for each column. Format: {'column': [edge1, edge2, ...]}.
        Required when strategy='custom'.
    labels : dict or str, optional
        Labels for bins:
        - dict: {'column': ['label1', 'label2', ...]}
        - 'range': Use interval notation (default)
        - 'ordinal': Use integers (0, 1, 2, ...)
    include_lowest : bool, default=True
        Whether the first interval should include the left boundary.
    drop_original : bool, default=False
        If True, drop original columns after binning.
    suffix : str, default='_binned'
        Suffix to add to output column names.
    kbins_strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
        Strategy for KBinsDiscretizer (only used when strategy='kbins').
    kbins_encode : {'ordinal', 'onehot', 'onehot-dense'}, default='ordinal'
        Encoding for KBinsDiscretizer output.
        
    Examples
    --------
    >>> # Equal width binning
    >>> df = pd.DataFrame({'age': [25, 35, 45, 55, 65, 75]})
    >>> binner = SmartBinning(strategy='equal_width', columns=['age'], n_bins=3)
    >>> result = binner.fit_transform(df)
    >>> print(result['age_binned'])
    0    (24.95, 41.667]
    1    (24.95, 41.667]
    2    (41.667, 58.333]
    3    (41.667, 58.333]
    4    (58.333, 75.0]
    5    (58.333, 75.0]
    
    >>> # Equal frequency (quantile) binning
    >>> binner = SmartBinning(strategy='equal_frequency', columns=['age'], n_bins=3)
    >>> result = binner.fit_transform(df)
    
    >>> # Custom bin edges
    >>> binner = SmartBinning(
    ...     strategy='custom',
    ...     columns=['age'],
    ...     bins={'age': [0, 30, 50, 70, 100]},
    ...     labels={'age': ['Young', 'Middle', 'Senior', 'Elder']}
    ... )
    >>> result = binner.fit_transform(df)
    >>> print(result['age_binned'])
    0     Young
    1    Middle
    2    Middle
    3    Senior
    4    Senior
    5     Elder
    
    >>> # KBinsDiscretizer with kmeans
    >>> binner = SmartBinning(
    ...     strategy='kbins',
    ...     columns=['age'],
    ...     n_bins=3,
    ...     kbins_strategy='kmeans'
    ... )
    >>> result = binner.fit_transform(df)
    
    Notes
    -----
    - Equal width: Divides range into equal-sized intervals
    - Equal frequency: Each bin has approximately the same number of samples
    - Custom: Full control over bin boundaries and labels
    - KBins: Advanced binning with kmeans clustering support
    - Missing values are preserved as NaN in output
    - Use drop_original=True to replace original columns
    """
    
    def __init__(
        self,
        strategy: Literal['equal_width', 'equal_frequency', 'custom', 'kbins'] = 'equal_width',
        columns: Optional[List[str]] = None,
        n_bins: int = 5,
        bins: Optional[Dict[str, List[float]]] = None,
        labels: Optional[Union[Dict[str, List[str]], Literal['range', 'ordinal']]] = 'range',
        include_lowest: bool = True,
        drop_original: bool = False,
        suffix: str = '_binned',
        kbins_strategy: Literal['uniform', 'quantile', 'kmeans'] = 'quantile',
        kbins_encode: Literal['ordinal', 'onehot', 'onehot-dense'] = 'ordinal',
    ):
        super().__init__()
        self.strategy = strategy
        self.columns = columns
        self.n_bins = n_bins
        self.bins = bins or {}
        self.labels = labels
        self.include_lowest = include_lowest
        self.drop_original = drop_original
        self.suffix = suffix
        self.kbins_strategy = kbins_strategy
        self.kbins_encode = kbins_encode
        
        self._bin_edges: Dict[str, np.ndarray] = {}
        self._kbins_discretizers: Dict[str, KBinsDiscretizer] = {}
    
    def _check_dataframe(self, X: pd.DataFrame) -> None:
        """Validate that input is a pandas DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError(
                f"Expected pandas DataFrame, got {type(X).__name__}"
            )
    
    def _validate_columns(self, X: pd.DataFrame) -> None:
        """Validate that specified columns exist in the DataFrame."""
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise FeatureEngineeringError(
                f"Columns not found in DataFrame: {missing}"
            )
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'SmartBinning':
        """
        Fit the binning transformer.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        y : pd.Series, optional
            Target variable (not used).
            
        Returns
        -------
        self
            Fitted transformer.
        """
        self._check_dataframe(X)
        
        # Determine columns to bin
        if self.columns is None:
            # Bin all numeric columns
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not self.columns:
            raise FeatureEngineeringError("No columns specified for binning")
        
        # Validate columns exist
        missing = [col for col in self.columns if col not in X.columns]
        if missing:
            raise FeatureEngineeringError(f"Columns not found: {missing}")
        
        # Validate strategy
        valid_strategies = {'equal_width', 'equal_frequency', 'custom', 'kbins'}
        if self.strategy not in valid_strategies:
            raise FeatureEngineeringError(
                f"Invalid strategy '{self.strategy}'. Must be one of {valid_strategies}"
            )
        
        # Validate n_bins
        if self.n_bins < 2:
            raise FeatureEngineeringError("n_bins must be at least 2")
        
        # Fit based on strategy
        if self.strategy == 'custom':
            self._fit_custom(X)
        elif self.strategy == 'kbins':
            self._fit_kbins(X)
        else:
            self._fit_pandas(X)
        
        # Import TransformerState
        from ...core.types import TransformerState
        self.state = TransformerState.FITTED
        return self
    
    def _fit_custom(self, X: pd.DataFrame) -> None:
        """Validate custom bins."""
        if not self.bins:
            raise FeatureEngineeringError(
                "strategy='custom' requires bins parameter"
            )
        
        for col in self.columns:
            if col not in self.bins:
                raise FeatureEngineeringError(
                    f"Custom bins not provided for column '{col}'"
                )
            
            bin_edges = self.bins[col]
            if len(bin_edges) < 2:
                raise FeatureEngineeringError(
                    f"Column '{col}' needs at least 2 bin edges"
                )
            
            # Store as numpy array
            self._bin_edges[col] = np.array(sorted(bin_edges))
    
    def _fit_kbins(self, X: pd.DataFrame) -> None:
        """Fit KBinsDiscretizer for each column."""
        for col in self.columns:
            values = X[col].dropna().values.reshape(-1, 1)
            
            if len(values) == 0:
                raise FeatureEngineeringError(
                    f"Column '{col}' has no valid values for binning"
                )
            
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode=self.kbins_encode,
                strategy=self.kbins_strategy,
            )
            discretizer.fit(values)
            self._kbins_discretizers[col] = discretizer
            self._bin_edges[col] = discretizer.bin_edges_[0]
    
    def _fit_pandas(self, X: pd.DataFrame) -> None:
        """Compute bin edges for equal_width or equal_frequency."""
        for col in self.columns:
            values = X[col].dropna()
            
            if len(values) == 0:
                raise FeatureEngineeringError(
                    f"Column '{col}' has no valid values for binning"
                )
            
            if self.strategy == 'equal_width':
                # Equal width bins
                min_val = values.min()
                max_val = values.max()
                self._bin_edges[col] = np.linspace(min_val, max_val, self.n_bins + 1)
                
            elif self.strategy == 'equal_frequency':
                # Equal frequency (quantile) bins
                quantiles = np.linspace(0, 1, self.n_bins + 1)
                self._bin_edges[col] = values.quantile(quantiles).values
                # Remove duplicates
                self._bin_edges[col] = np.unique(self._bin_edges[col])
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by binning specified columns.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.
            
        Returns
        -------
        pd.DataFrame
            Transformed data with binned columns.
        """
        self._check_is_fitted()
        self._check_dataframe(X)
        
        X_transformed = X.copy()
        
        for col in self.columns:
            if col not in X_transformed.columns:
                continue
            
            output_col = f"{col}{self.suffix}"
            
            if self.strategy == 'kbins' and self.kbins_encode in ['onehot', 'onehot-dense']:
                # KBins with one-hot encoding
                discretizer = self._kbins_discretizers[col]
                values = X_transformed[col].values.reshape(-1, 1)
                
                # Handle missing values
                mask = pd.isna(X_transformed[col])
                if mask.any():
                    # Create temporary filled array
                    filled_values = np.where(mask, 0, values.ravel()).reshape(-1, 1)
                    transformed = discretizer.transform(filled_values)
                    # Create columns for one-hot
                    n_features = transformed.shape[1]
                    for i in range(n_features):
                        col_name = f"{output_col}_{i}"
                        X_transformed[col_name] = transformed[:, i]
                        X_transformed.loc[mask, col_name] = np.nan
                else:
                    transformed = discretizer.transform(values)
                    n_features = transformed.shape[1]
                    for i in range(n_features):
                        X_transformed[f"{output_col}_{i}"] = transformed[:, i]
                
            elif self.strategy == 'kbins':
                # KBins with ordinal encoding
                discretizer = self._kbins_discretizers[col]
                values = X_transformed[col].values.reshape(-1, 1)
                
                # Handle missing values
                mask = pd.isna(X_transformed[col])
                if mask.any():
                    filled_values = np.where(mask, 0, values.ravel()).reshape(-1, 1)
                    binned = discretizer.transform(filled_values).ravel()
                    binned = binned.astype(float)
                    binned[mask] = np.nan
                else:
                    binned = discretizer.transform(values).ravel()
                
                X_transformed[output_col] = binned
                
            else:
                # Pandas cut for equal_width, equal_frequency, custom
                bin_edges = self._bin_edges[col]
                
                # Get labels
                bin_labels = self._get_labels(col, len(bin_edges) - 1)
                
                # Apply binning
                binned = pd.cut(
                    X_transformed[col],
                    bins=bin_edges,
                    labels=bin_labels,
                    include_lowest=self.include_lowest,
                    duplicates='drop',
                )
                
                X_transformed[output_col] = binned
            
            # Drop original column if requested
            if self.drop_original:
                X_transformed = X_transformed.drop(columns=[col])
        
        return X_transformed
    
    def _get_labels(self, column: str, n_bins: int) -> Optional[List]:
        """Get labels for bins."""
        if self.labels == 'range':
            # Use default interval notation
            return None
        elif self.labels == 'ordinal':
            # Use integers
            return list(range(n_bins))
        elif isinstance(self.labels, dict):
            if column in self.labels:
                labels = self.labels[column]
                if len(labels) != n_bins:
                    raise FeatureEngineeringError(
                        f"Column '{column}': expected {n_bins} labels, got {len(labels)}"
                    )
                return labels
            return None
        return None
    
    def get_bin_edges(self, column: str) -> Optional[np.ndarray]:
        """
        Get bin edges for a column.
        
        Parameters
        ----------
        column : str
            Column name.
            
        Returns
        -------
        np.ndarray or None
            Bin edges if available.
        """
        self._check_is_fitted()
        return self._bin_edges.get(column)
    
    def get_feature_names_out(self) -> List[str]:
        """
        Get names of output features.
        
        Returns
        -------
        list of str
            Names of binned features.
        """
        self._check_is_fitted()
        
        if self.strategy == 'kbins' and self.kbins_encode in ['onehot', 'onehot-dense']:
            # Multiple columns per input column
            features = []
            for col in self.columns:
                if col in self._kbins_discretizers:
                    n_bins = self._kbins_discretizers[col].n_bins
                    for i in range(n_bins):
                        features.append(f"{col}{self.suffix}_{i}")
            return features
        else:
            # One column per input column
            return [f"{col}{self.suffix}" for col in self.columns]
