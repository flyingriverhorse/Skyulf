"""
Target Encoder for supervised categorical encoding.

This module provides target encoding (also known as mean encoding or 
likelihood encoding) which encodes categories based on the target variable.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ...core.base import BaseEncoder
from ...core.exceptions import FeatureEngineeringError
from ...core.types import TransformerState


class TargetEncoder(BaseEncoder):
    """
    Encode categorical features using target statistics.
    
    Target encoding replaces categories with statistics (typically mean) of the
    target variable for that category. Useful for high-cardinality features.
    
    **Warning**: Target encoding can lead to overfitting. Use with cross-validation
    or add smoothing to mitigate this.
    
    Parameters
    ----------
    columns : list of str, optional
        Column names to encode. If None, encodes all object/category columns.
    smoothing : float, default=1.0
        Smoothing parameter to avoid overfitting. Higher values mean more smoothing
        (closer to global mean). Formula: (n * cat_mean + smoothing * global_mean) / (n + smoothing)
    min_samples_leaf : int, default=1
        Minimum samples to calculate category encoding. Categories with fewer samples
        use the global mean.
    handle_unknown : {'global_mean', 'value'}, default='global_mean'
        How to handle unknown categories:
        - 'global_mean': Use global mean of target
        - 'value': Use specified unknown_value
    unknown_value : float, default=0.0
        Value to use for unknown categories when handle_unknown='value'.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from skyulf_mlflow_library.features.encoding import TargetEncoder
    >>> 
    >>> # Classification example
    >>> df = pd.DataFrame({
    ...     'city': ['NYC', 'LA', 'NYC', 'LA', 'SF', 'SF', 'NYC'],
    ...     'target': [1, 0, 1, 0, 1, 1, 0]
    ... })
    >>> 
    >>> encoder = TargetEncoder(columns=['city'], smoothing=1.0)
    >>> result = encoder.fit_transform(df, df['target'])
    >>> print(result)
    >>> # NYC: (2*0.666 + 1*0.571) / 3 = 0.635
    >>> # LA:  (2*0.0 + 1*0.571) / 3 = 0.190
    >>> # SF:  (2*1.0 + 1*0.571) / 3 = 0.857
    
    >>> # Regression example
    >>> df_reg = pd.DataFrame({
    ...     'category': ['A', 'B', 'A', 'B', 'A', 'C'],
    ...     'value': [100, 200, 150, 180, 120, 300]
    ... })
    >>> 
    >>> encoder = TargetEncoder(columns=['category'])
    >>> result = encoder.fit_transform(df_reg, df_reg['value'])
    >>> print(result)
    """
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
        handle_unknown: str = 'global_mean',
        unknown_value: float = 0.0,
        **kwargs
    ):
        """
        Initialize the target encoder.
        
        Parameters
        ----------
        columns : list of str, optional
            Columns to encode.
        smoothing : float, default=1.0
            Smoothing factor to reduce overfitting.
        min_samples_leaf : int, default=1
            Minimum samples for category encoding.
        handle_unknown : str, default='global_mean'
            Strategy for unknown categories.
        unknown_value : float, default=0.0
            Value for unknown categories.
        **kwargs : dict
            Additional parameters.
        """
        super().__init__(columns=columns, **kwargs)
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        
        self._encodings: Dict[str, Dict] = {}
        self._global_mean: float = 0.0
        self._feature_names: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        """
        Fit the target encoder to the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit.
        y : pd.Series
            Target variable.
        
        Returns
        -------
        self
            Fitted encoder.
        """
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("X must be a pandas DataFrame")
        
        if y is None:
            raise FeatureEngineeringError(
                "Target encoder requires target variable y"
            )
        
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        if len(X) != len(y):
            raise FeatureEngineeringError(
                f"X and y must have same length. Got {len(X)} and {len(y)}"
            )
        
        # Determine columns to encode
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not self.columns:
            raise FeatureEngineeringError("No columns to encode")
        
        self._validate_columns(X)
        self._feature_names = self.columns.copy()
        
        # Calculate global mean
        self._global_mean = y.mean()
        
        # Calculate encodings for each column
        for col in self.columns:
            # Group by category and calculate statistics
            stats = pd.DataFrame({
                'target': y,
                'category': X[col]
            }).groupby('category')['target'].agg(['mean', 'count'])
            
            # Apply smoothing
            encodings = {}
            for category in stats.index:
                cat_mean = stats.loc[category, 'mean']
                cat_count = stats.loc[category, 'count']
                
                if cat_count < self.min_samples_leaf:
                    # Use global mean for rare categories
                    encodings[category] = self._global_mean
                else:
                    # Apply smoothing formula
                    smooth_mean = (
                        cat_count * cat_mean + self.smoothing * self._global_mean
                    ) / (cat_count + self.smoothing)
                    encodings[category] = smooth_mean
            
            self._encodings[col] = encodings
        
        self.state = TransformerState.FITTED
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted target encoder.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.
        
        Returns
        -------
        pd.DataFrame
            Transformed data with target encoded columns.
        """
        self._check_is_fitted()
        
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("X must be a pandas DataFrame")
        
        self._validate_columns(X)
        
        X_transformed = X.copy()
        
        for col in self.columns:
            encodings = self._encodings[col]
            
            # Map categories to encodings
            if self.handle_unknown == 'global_mean':
                default_value = self._global_mean
            else:  # 'value'
                default_value = self.unknown_value
            
            X_transformed[col] = X[col].map(encodings).fillna(default_value)
        
        return X_transformed
    
    def get_feature_names_out(self) -> List[str]:
        """
        Get output feature names.
        
        Returns
        -------
        list of str
            Output feature names (same as input).
        """
        self._check_is_fitted()
        return self._feature_names
    
    def get_encodings(self, column: Optional[str] = None) -> Dict:
        """
        Get the learned encodings for categories.
        
        Parameters
        ----------
        column : str, optional
            Column name to get encodings for. If None, returns all.
        
        Returns
        -------
        dict
            Dictionary of encodings for each category.
        """
        self._check_is_fitted()
        
        if column is not None:
            if column not in self._encodings:
                raise FeatureEngineeringError(
                    f"Column '{column}' not found in encodings"
                )
            return self._encodings[column]
        
        return self._encodings
