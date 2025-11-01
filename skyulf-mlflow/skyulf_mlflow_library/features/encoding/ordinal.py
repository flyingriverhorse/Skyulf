"""
Ordinal Encoder for preserving order in categorical features.

This module provides ordinal encoding that maintains the natural order
of categorical variables (e.g., low < medium < high).
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder as SklearnOrdinalEncoder

from ...core.base import BaseEncoder
from ...core.exceptions import FeatureEngineeringError
from ...core.types import TransformerState


class OrdinalEncoder(BaseEncoder):
    """
    Encode categorical features with ordinal (ordered) values.
    
    Useful for categorical features with natural ordering like:
    - Ratings: ['poor', 'fair', 'good', 'excellent']
    - Education: ['high school', 'bachelor', 'master', 'phd']
    - Sizes: ['small', 'medium', 'large']
    
    Parameters
    ----------
    columns : list of str, optional
        Column names to encode. If None, encodes all object/category columns.
    categories : dict or list, optional
        Categories for each column in order. Can be:
        - dict: {column: [ordered_categories]}
        - list: ordered categories for single column
        - 'auto': Determine categories automatically
    handle_unknown : {'error', 'use_encoded_value'}, default='error'
        How to handle unknown categories during transform:
        - 'error': Raise an error
        - 'use_encoded_value': Use the value given by unknown_value parameter
    unknown_value : int, default=-1
        Value to use for unknown categories when handle_unknown='use_encoded_value'.
    dtype : dtype, default=np.float64
        Desired dtype of output.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from skyulf_mlflow_library.features.encoding import OrdinalEncoder
    >>> 
    >>> # Example with explicit categories
    >>> df = pd.DataFrame({
    ...     'size': ['small', 'large', 'medium', 'small'],
    ...     'rating': ['good', 'excellent', 'poor', 'fair']
    ... })
    >>> 
    >>> encoder = OrdinalEncoder(
    ...     columns=['size', 'rating'],
    ...     categories={
    ...         'size': ['small', 'medium', 'large'],
    ...         'rating': ['poor', 'fair', 'good', 'excellent']
    ...     }
    ... )
    >>> result = encoder.fit_transform(df)
    >>> print(result)
       size  rating
    0   0.0     2.0
    1   2.0     3.0
    2   1.0     0.0
    3   0.0     1.0
    
    >>> # Auto-detect categories
    >>> encoder_auto = OrdinalEncoder(columns=['size'])
    >>> result = encoder_auto.fit_transform(df)
    """
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        categories: Optional[Union[Dict[str, List], List, str]] = 'auto',
        handle_unknown: str = 'error',
        unknown_value: int = -1,
        dtype: Any = None,
        **kwargs
    ):
        """
        Initialize the ordinal encoder.
        
        Parameters
        ----------
        columns : list of str, optional
            Columns to encode.
        categories : dict, list, or 'auto', default='auto'
            Category ordering for each column.
        handle_unknown : str, default='error'
            Strategy for handling unknown categories.
        unknown_value : int, default=-1
            Value for unknown categories.
        dtype : dtype, optional
            Output data type.
        **kwargs : dict
            Additional parameters.
        """
        super().__init__(columns=columns, **kwargs)
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.dtype = dtype
        self._encoders: Dict[str, SklearnOrdinalEncoder] = {}
        self._feature_names: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OrdinalEncoder":
        """
        Fit the ordinal encoder to the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit.
        y : pd.Series, optional
            Target variable (not used).
        
        Returns
        -------
        self
            Fitted encoder.
        """
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("X must be a pandas DataFrame")
        
        # Determine columns to encode
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not self.columns:
            raise FeatureEngineeringError("No columns to encode")
        
        self._validate_columns(X)
        self._feature_names = self.columns.copy()
        
        # Fit encoder for each column
        for col in self.columns:
            # Prepare categories for this column
            if self.categories == 'auto':
                cat_list = [X[col].dropna().unique().tolist()]
            elif isinstance(self.categories, dict):
                if col not in self.categories:
                    raise FeatureEngineeringError(
                        f"Categories not specified for column '{col}'"
                    )
                cat_list = [self.categories[col]]
            elif isinstance(self.categories, list):
                cat_list = [self.categories]
            else:
                raise FeatureEngineeringError(
                    f"Invalid categories type: {type(self.categories)}"
                )
            
            # Create and fit sklearn encoder
            encoder_kwargs = {
                'categories': cat_list,
                'handle_unknown': self.handle_unknown,
                'dtype': self.dtype
            }
            # Only pass unknown_value if handle_unknown='use_encoded_value'
            if self.handle_unknown == 'use_encoded_value':
                encoder_kwargs['unknown_value'] = self.unknown_value
            
            encoder = SklearnOrdinalEncoder(**encoder_kwargs)
            
            encoder.fit(X[[col]])
            self._encoders[col] = encoder
        
        self.state = TransformerState.FITTED
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted ordinal encoder.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.
        
        Returns
        -------
        pd.DataFrame
            Transformed data with ordinal encoded columns.
        """
        self._check_is_fitted()
        
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("X must be a pandas DataFrame")
        
        self._validate_columns(X)
        
        X_transformed = X.copy()
        
        for col in self.columns:
            encoded = self._encoders[col].transform(X[[col]])
            X_transformed[col] = encoded.ravel()
        
        return X_transformed
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert ordinal encoded data back to original categories.
        
        Parameters
        ----------
        X : pd.DataFrame
            Encoded data to inverse transform.
        
        Returns
        -------
        pd.DataFrame
            Data with original categorical values.
        """
        self._check_is_fitted()
        
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("X must be a pandas DataFrame")
        
        X_inverse = X.copy()
        
        for col in self.columns:
            if col in X_inverse.columns:
                decoded = self._encoders[col].inverse_transform(
                    X_inverse[[col]]
                )
                X_inverse[col] = decoded.ravel()
        
        return X_inverse
    
    def get_feature_names_out(self) -> List[str]:
        """
        Get output feature names.
        
        Returns
        -------
        list of str
            Output feature names (same as input for ordinal encoding).
        """
        self._check_is_fitted()
        return self._feature_names
