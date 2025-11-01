"""
Hash Encoder for high-cardinality categorical features.

This module provides hash encoding which uses hashing to convert categories
into a fixed number of features, useful for very high cardinality data.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

from ...core.base import BaseEncoder
from ...core.exceptions import FeatureEngineeringError
from ...core.types import TransformerState


class HashEncoder(BaseEncoder):
    """
    Encode categorical features using feature hashing (hashing trick).
    
    Hash encoding converts categorical values into a fixed number of features
    using a hash function. This is useful for:
    - Very high cardinality features (millions of categories)
    - Online learning where categories are unknown upfront
    - Memory-efficient encoding
    
    **Note**: Hash collisions may occur (multiple categories map to same hash),
    which is acceptable for high-dimensional feature spaces.
    
    Parameters
    ----------
    columns : list of str, optional
        Column names to encode. If None, encodes all object/category columns.
    n_components : int, default=8
        Number of hash features to create for each input column.
        Higher values reduce collision probability but increase dimensionality.
    dtype : dtype, default=np.float32
        Data type of output features.
    alternate_sign : bool, default=True
        If True, uses signed hash to reduce bias from collisions.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from skyulf_mlflow_library.features.encoding import HashEncoder
    >>> 
    >>> # High cardinality example
    >>> df = pd.DataFrame({
    ...     'user_id': ['user_001', 'user_002', 'user_003', 'user_001'],
    ...     'item_id': ['item_A', 'item_B', 'item_A', 'item_C']
    ... })
    >>> 
    >>> encoder = HashEncoder(columns=['user_id', 'item_id'], n_components=4)
    >>> result = encoder.fit_transform(df)
    >>> print(result.shape)  # (4, 8) - 4 samples, 4 features per column
    >>> print(result.columns)
    >>> # ['user_id_0', 'user_id_1', 'user_id_2', 'user_id_3',
    >>> #  'item_id_0', 'item_id_1', 'item_id_2', 'item_id_3']
    
    >>> # Compare with one-hot encoding size
    >>> # One-hot would create 6 columns (3 users + 3 items)
    >>> # Hash encoding creates 8 columns regardless of cardinality
    """
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        n_components: int = 8,
        dtype: type = np.float32,
        alternate_sign: bool = True,
        **kwargs
    ):
        """
        Initialize the hash encoder.
        
        Parameters
        ----------
        columns : list of str, optional
            Columns to encode.
        n_components : int, default=8
            Number of hash features per column.
        dtype : dtype, default=np.float32
            Output data type.
        alternate_sign : bool, default=True
            Use signed hash values.
        **kwargs : dict
            Additional parameters.
        """
        super().__init__(columns=columns, **kwargs)
        self.n_components = n_components
        self.dtype = dtype
        self.alternate_sign = alternate_sign
        
        self._hashers: dict = {}
        self._feature_names_out: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "HashEncoder":
        """
        Fit the hash encoder to the data.
        
        Hash encoding is stateless, so this just sets up the hashers.
        
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
        
        # Create a hasher for each column
        self._feature_names_out = []
        for col in self.columns:
            hasher = FeatureHasher(
                n_features=self.n_components,
                input_type='string',
                dtype=self.dtype,
                alternate_sign=self.alternate_sign
            )
            self._hashers[col] = hasher
            
            # Generate feature names
            for i in range(self.n_components):
                self._feature_names_out.append(f"{col}_{i}")
        
        self.state = TransformerState.FITTED
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using hash encoding.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.
        
        Returns
        -------
        pd.DataFrame
            Transformed data with hash encoded features.
        """
        self._check_is_fitted()
        
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("X must be a pandas DataFrame")
        
        self._validate_columns(X)
        
        # Keep non-encoded columns
        non_encoded_cols = [col for col in X.columns if col not in self.columns]
        X_transformed = X[non_encoded_cols].copy() if non_encoded_cols else pd.DataFrame(index=X.index)
        
        # Hash encode each column
        for col in self.columns:
            # Convert to list of strings (FeatureHasher expects iterable of strings)
            col_data = X[col].astype(str).values
            col_data_formatted = [[val] for val in col_data]
            
            # Apply hashing
            hashed = self._hashers[col].transform(col_data_formatted)
            hashed_dense = hashed.toarray()
            
            # Create column names
            hash_cols = [f"{col}_{i}" for i in range(self.n_components)]
            hash_df = pd.DataFrame(
                hashed_dense,
                columns=hash_cols,
                index=X.index
            )
            
            # Add to result
            X_transformed = pd.concat([X_transformed, hash_df], axis=1)
        
        return X_transformed
    
    def get_feature_names_out(self) -> List[str]:
        """
        Get output feature names.
        
        Returns
        -------
        list of str
            Output feature names (hash features).
        """
        self._check_is_fitted()
        return self._feature_names_out
    
    def get_params(self) -> dict:
        """
        Get parameters for this encoder.
        
        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        return {
            'columns': self.columns,
            'n_components': self.n_components,
            'dtype': self.dtype,
            'alternate_sign': self.alternate_sign,
            **self.config
        }
