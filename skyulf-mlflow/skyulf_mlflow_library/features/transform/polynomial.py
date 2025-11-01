"""
Polynomial and interaction features for feature engineering.

This module provides transformers for creating polynomial features,
interaction terms, and other mathematical transformations.
"""

from typing import List, Optional

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures as SklearnPolyFeatures

from ...core.base import BaseTransformer
from ...core.exceptions import FeatureEngineeringError
from ...core.types import TransformerState


class PolynomialFeatures(BaseTransformer):
    """
    Generate polynomial and interaction features.
    
    Creates new features by raising existing features to powers and
    computing interactions between features.
    
    Parameters
    ----------
    columns : list of str, optional
        Columns to create polynomial features from. If None, uses all numeric columns.
    degree : int, default=2
        Maximum degree of polynomial features.
    interaction_only : bool, default=False
        If True, only interaction features are produced (no powers).
    include_bias : bool, default=False
        If True, include a bias column (all ones).
    
    Examples
    --------
    >>> import pandas as pd
    >>> from skyulf_mlflow_library.features.transform import PolynomialFeatures
    >>> 
    >>> # Create polynomial features
    >>> df = pd.DataFrame({
    ...     'x1': [1, 2, 3],
    ...     'x2': [4, 5, 6]
    ... })
    >>> 
    >>> poly = PolynomialFeatures(degree=2)
    >>> result = poly.fit_transform(df)
    >>> print(result.columns)
    >>> # ['x1', 'x2', 'x1^2', 'x1 x2', 'x2^2']
    
    >>> # Interaction features only
    >>> poly_inter = PolynomialFeatures(degree=2, interaction_only=True)
    >>> result = poly_inter.fit_transform(df)
    >>> print(result.columns)
    >>> # ['x1', 'x2', 'x1 x2']
    """
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        **kwargs
    ):
        """
        Initialize polynomial features transformer.
        
        Parameters
        ----------
        columns : list of str, optional
            Columns to transform.
        degree : int, default=2
            Polynomial degree.
        interaction_only : bool, default=False
            Only create interactions.
        include_bias : bool, default=False
            Include bias term.
        **kwargs : dict
            Additional parameters.
        """
        super().__init__(columns=columns, **kwargs)
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        
        self._poly = SklearnPolyFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        self._feature_names: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PolynomialFeatures":
        """
        Fit the polynomial features transformer.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        y : pd.Series, optional
            Target (not used).
        
        Returns
        -------
        self
            Fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("X must be a pandas DataFrame")
        
        # Determine columns
        if self.columns is None:
            self.columns = X.select_dtypes(include=['number']).columns.tolist()
        
        if not self.columns:
            raise FeatureEngineeringError("No numeric columns found")
        
        self._validate_columns(X)
        
        # Fit polynomial transformer
        self._poly.fit(X[self.columns])
        
        # Get feature names
        self._feature_names = self._poly.get_feature_names_out(self.columns).tolist()
        
        self.state = TransformerState.FITTED
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to polynomial features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        
        Returns
        -------
        pd.DataFrame
            Transformed data with polynomial features.
        """
        self._check_is_fitted()
        
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("X must be a pandas DataFrame")
        
        self._validate_columns(X)
        
        # Keep non-transformed columns
        other_cols = [col for col in X.columns if col not in self.columns]
        X_other = X[other_cols].copy() if other_cols else pd.DataFrame(index=X.index)
        
        # Transform selected columns
        X_poly = self._poly.transform(X[self.columns])
        X_poly_df = pd.DataFrame(
            X_poly,
            columns=self._feature_names,
            index=X.index
        )
        
        # Combine
        result = pd.concat([X_other, X_poly_df], axis=1)
        
        return result
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names."""
        self._check_is_fitted()
        return self._feature_names


class InteractionFeatures(BaseTransformer):
    """
    Create interaction features between specified column pairs.
    
    More flexible than PolynomialFeatures for creating specific interactions.
    
    Parameters
    ----------
    interactions : list of tuple
        List of (col1, col2) tuples specifying which columns to interact.
        Example: [('age', 'income'), ('age', 'credit_score')]
    operation : {'multiply', 'add', 'subtract', 'divide'}, default='multiply'
        Operation to use for creating interactions.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from skyulf_mlflow_library.features.transform import InteractionFeatures
    >>> 
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35],
    ...     'income': [50000, 60000, 70000],
    ...     'credit_score': [700, 750, 800]
    ... })
    >>> 
    >>> inter = InteractionFeatures(
    ...     interactions=[('age', 'income'), ('age', 'credit_score')]
    ... )
    >>> result = inter.fit_transform(df)
    >>> print(result.columns)
    >>> # [..., 'age_x_income', 'age_x_credit_score']
    """
    
    def __init__(
        self,
        interactions: List[tuple],
        operation: str = 'multiply',
        **kwargs
    ):
        """
        Initialize interaction features transformer.
        
        Parameters
        ----------
        interactions : list of tuple
            Column pairs to interact.
        operation : str, default='multiply'
            Interaction operation.
        **kwargs : dict
            Additional parameters.
        """
        super().__init__(**kwargs)
        self.interactions = interactions
        self.operation = operation
        
        if operation not in {'multiply', 'add', 'subtract', 'divide'}:
            raise FeatureEngineeringError(
                f"Invalid operation: {operation}. "
                "Must be 'multiply', 'add', 'subtract', or 'divide'"
            )
        
        self._feature_names: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "InteractionFeatures":
        """
        Fit the interaction features transformer.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        y : pd.Series, optional
            Target (not used).
        
        Returns
        -------
        self
            Fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("X must be a pandas DataFrame")
        
        # Validate interactions
        for col1, col2 in self.interactions:
            if col1 not in X.columns:
                raise FeatureEngineeringError(f"Column '{col1}' not found")
            if col2 not in X.columns:
                raise FeatureEngineeringError(f"Column '{col2}' not found")
        
        # Generate feature names
        op_symbols = {
            'multiply': 'x',
            'add': '+',
            'subtract': '-',
            'divide': '/'
        }
        symbol = op_symbols[self.operation]
        
        self._feature_names = [
            f"{col1}_{symbol}_{col2}" for col1, col2 in self.interactions
        ]
        
        self.state = TransformerState.FITTED
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by creating interaction features.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        
        Returns
        -------
        pd.DataFrame
            Data with interaction features added.
        """
        self._check_is_fitted()
        
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("X must be a pandas DataFrame")
        
        X_transformed = X.copy()
        
        for (col1, col2), feat_name in zip(self.interactions, self._feature_names):
            if self.operation == 'multiply':
                X_transformed[feat_name] = X[col1] * X[col2]
            elif self.operation == 'add':
                X_transformed[feat_name] = X[col1] + X[col2]
            elif self.operation == 'subtract':
                X_transformed[feat_name] = X[col1] - X[col2]
            elif self.operation == 'divide':
                X_transformed[feat_name] = X[col1] / (X[col2] + 1e-9)
        
        return X_transformed
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names."""
        self._check_is_fitted()
        return self._feature_names
