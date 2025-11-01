"""Label encoding transformer for categorical features."""

from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder

from skyulf_mlflow_library.core.base import BaseEncoder
from skyulf_mlflow_library.core.types import DataFrame, TransformerState
from skyulf_mlflow_library.exceptions import EncodingError


class LabelEncoder(BaseEncoder):
    """
    Encode categorical features as integers.

    Each unique category value is mapped to an integer value from 0 to n_categories-1.
    This is useful for ordinal-like categorical variables or as input for tree-based models.

    Parameters
    ----------
    columns : list of str, optional
        List of column names to encode. If None, all object/category columns are encoded.
    handle_unknown : {'error', 'use_encoded_value'}, default='use_encoded_value'
        How to handle unknown categories during transform.
        - 'error': Raise an error
        - 'use_encoded_value': Use unknown_value for unseen categories
    unknown_value : int, default=-1
        Value to use for unknown categories when handle_unknown='use_encoded_value'.
    drop_original : bool, default=False
        Whether to drop original columns after encoding.

    Attributes
    ----------
    classes_ : dict
        Dictionary mapping column names to arrays of unique categories.
    encoders_ : dict
        Dictionary mapping column names to fitted LabelEncoder instances.

    Examples
    --------
    >>> from skyulf_mlflow_library.features.encoding import LabelEncoder
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({
    ...     'color': ['red', 'blue', 'red', 'green', 'blue'],
    ...     'size': ['S', 'M', 'L', 'M', 'S']
    ... })
    >>>
    >>> encoder = LabelEncoder(columns=['color', 'size'])
    >>> df_encoded = encoder.fit_transform(df)
    >>> print(df_encoded)
       color  size  color_encoded  size_encoded
    0    red     S              2             0
    1   blue     M              0             1
    2    red     L              2             2
    3  green     S              1             0
    4   blue     M              0             1
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        handle_unknown: str = "use_encoded_value",
        unknown_value: int = -1,
        drop_original: bool = False,
        **kwargs
    ):
        super().__init__(columns=columns, handle_unknown=handle_unknown, **kwargs)
        self.unknown_value = unknown_value
        self.drop_original = drop_original
        self.classes_: Optional[Dict[str, np.ndarray]] = None
        self.encoders_: Optional[Dict[str, SklearnLabelEncoder]] = None

    def fit(self, X: DataFrame, y: Optional[pd.Series] = None) -> "LabelEncoder":
        """
        Fit the label encoder to the data.

        Parameters
        ----------
        X : DataFrame
            Input data with categorical columns to encode.
        y : Series, optional
            Target variable (not used, present for API consistency).

        Returns
        -------
        self : LabelEncoder
            Fitted encoder.
        """
        X = X.copy()
        self._validate_columns(X)

        # Determine columns to encode
        if self.columns is None:
            # Auto-detect categorical columns
            self._original_columns = [
                col for col in X.columns
                if X[col].dtype in ['object', 'category']
            ]
        else:
            self._original_columns = self.columns.copy()

        if not self._original_columns:
            raise EncodingError("No columns to encode. Specify columns or ensure data has categorical columns.")

        # Initialize encoders and classes
        self.encoders_ = {}
        self.classes_ = {}

        # Fit encoder for each column
        for col in self._original_columns:
            encoder = SklearnLabelEncoder()
            
            # Handle missing values by treating them as a category
            series = X[col].copy()
            has_missing = series.isna().any()
            
            if has_missing:
                # Replace NaN with a special marker
                series = series.fillna('__MISSING__')
            
            try:
                encoder.fit(series)
                self.encoders_[col] = encoder
                self.classes_[col] = encoder.classes_
            except Exception as e:
                raise EncodingError(f"Failed to fit label encoder for column '{col}': {str(e)}")

        # Update state and metadata
        self.state = TransformerState.FITTED
        self.metadata = {
            "n_features_in": len(self._original_columns),
            "feature_names_in": self._original_columns,
            "classes": {col: classes.tolist() for col, classes in self.classes_.items()},
            "n_categories": {col: len(classes) for col, classes in self.classes_.items()},
        }

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Transform data using the fitted label encoder.

        Parameters
        ----------
        X : DataFrame
            Input data to transform.

        Returns
        -------
        DataFrame
            Transformed data with label encoded columns.
        """
        self._check_is_fitted()
        X = X.copy()

        # Create encoded columns
        for col in self._original_columns:
            encoder = self.encoders_[col]
            encoded_col_name = f"{col}_encoded"
            
            try:
                series = X[col].copy()
                has_missing = series.isna().any()
                
                if has_missing:
                    series = series.fillna('__MISSING__')
                
                # Transform with unknown handling
                if self.handle_unknown == 'use_encoded_value':
                    # Map known categories
                    encoded = pd.Series(self.unknown_value, index=X.index)
                    mask = series.isin(encoder.classes_)
                    if mask.any():
                        encoded[mask] = encoder.transform(series[mask])
                else:
                    # Will raise error if unknown categories found
                    encoded = encoder.transform(series)
                
                X[encoded_col_name] = encoded.astype('int64')
                
            except Exception as e:
                raise EncodingError(f"Failed to transform column '{col}': {str(e)}")

        # Drop original columns if requested
        if self.drop_original:
            X = X.drop(columns=self._original_columns)

        return X

    def inverse_transform(self, X: DataFrame) -> DataFrame:
        """
        Convert encoded values back to original categories.

        Parameters
        ----------
        X : DataFrame
            Encoded data.

        Returns
        -------
        DataFrame
            Data with original categorical values.
        """
        self._check_is_fitted()
        X = X.copy()

        for col in self._original_columns:
            encoder = self.encoders_[col]
            encoded_col_name = f"{col}_encoded"
            
            if encoded_col_name not in X.columns:
                continue
            
            try:
                # Get encoded values
                encoded_values = X[encoded_col_name].values
                
                # Handle unknown value
                mask = encoded_values != self.unknown_value
                decoded = pd.Series(index=X.index, dtype='object')
                
                if mask.any():
                    # Clip to valid range
                    valid_encoded = np.clip(
                        encoded_values[mask],
                        0,
                        len(encoder.classes_) - 1
                    )
                    decoded[mask] = encoder.inverse_transform(valid_encoded.astype(int))
                
                # Replace __MISSING__ marker with NaN
                decoded = decoded.replace('__MISSING__', np.nan)
                
                X[col] = decoded
                X = X.drop(columns=[encoded_col_name])
                
            except Exception as e:
                raise EncodingError(f"Failed to inverse transform column '{col}': {str(e)}")

        return X

    def get_classes(self, column: Optional[str] = None) -> Dict[str, List]:
        """
        Get the unique categories for each column.

        Parameters
        ----------
        column : str, optional
            Specific column name. If None, returns classes for all columns.

        Returns
        -------
        dict or list
            Dictionary mapping column names to lists of categories,
            or list of categories if column is specified.
        """
        self._check_is_fitted()
        
        if column is not None:
            if column not in self.classes_:
                raise ValueError(f"Column '{column}' not found in fitted encoder.")
            return self.classes_[column].tolist()
        
        return {col: classes.tolist() for col, classes in self.classes_.items()}
