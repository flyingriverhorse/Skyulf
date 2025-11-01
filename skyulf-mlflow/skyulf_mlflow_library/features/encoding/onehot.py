"""One-hot encoding transformer for categorical features."""

from typing import Dict, List, Optional

import pandas as pd
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

from skyulf_mlflow_library.core.base import BaseEncoder
from skyulf_mlflow_library.core.types import DataFrame, TransformerState
from skyulf_mlflow_library.exceptions import EncodingError


class OneHotEncoder(BaseEncoder):
    """
    One-hot encode categorical features.

    Creates binary columns for each category in the specified columns.
    Compatible with scikit-learn's OneHotEncoder with additional features.

    Parameters
    ----------
    columns : list of str, optional
        List of column names to encode. If None, all object/category columns are encoded.
    drop : {'first', 'if_binary', None}, default=None
        Specifies a methodology to use to drop one of the categories per feature.
    handle_unknown : {'error', 'ignore'}, default='ignore'
        How to handle unknown categories during transform.
    sparse : bool, default=False
        Whether to return sparse matrix or dense array.
    max_categories : int, optional
        Maximum number of categories to encode. Categories beyond this are grouped.
    min_frequency : int or float, optional
        Minimum frequency for a category to be encoded separately.

    Examples
    --------
    >>> from skyulf_mlflow_library.features.encoding import OneHotEncoder
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({
    ...     'color': ['red', 'blue', 'red', 'green'],
    ...     'size': ['S', 'M', 'L', 'M']
    ... })
    >>>
    >>> encoder = OneHotEncoder(columns=['color'])
    >>> df_encoded = encoder.fit_transform(df)
    >>> print(df_encoded.columns.tolist())
    ['size', 'color_blue', 'color_green', 'color_red']
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        drop: Optional[str] = None,
        handle_unknown: str = "ignore",
        sparse: bool = False,
        max_categories: Optional[int] = None,
        min_frequency: Optional[float] = None,
        **kwargs
    ):
        super().__init__(columns=columns, handle_unknown=handle_unknown, **kwargs)
        self.drop = drop
        self.sparse = sparse
        self.max_categories = max_categories
        self.min_frequency = min_frequency
        self._encoder: Optional[SklearnOneHotEncoder] = None
        self._feature_names_out: Optional[List[str]] = None
        self._original_columns: Optional[List[str]] = None

    def fit(self, X: DataFrame, y: Optional[pd.Series] = None) -> "OneHotEncoder":
        """
        Fit the one-hot encoder to the data.

        Parameters
        ----------
        X : DataFrame
            Input data with categorical columns to encode.
        y : Series, optional
            Target variable (not used, present for API consistency).

        Returns
        -------
        self : OneHotEncoder
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

        # Initialize sklearn encoder
        self._encoder = SklearnOneHotEncoder(
            drop=self.drop,
            handle_unknown=self.handle_unknown,
            sparse_output=self.sparse,
            max_categories=self.max_categories,
            min_frequency=self.min_frequency,
        )

        # Fit the encoder
        try:
            self._encoder.fit(X[self._original_columns])
            self._feature_names_out = self._encoder.get_feature_names_out(self._original_columns).tolist()
        except Exception as e:
            raise EncodingError(f"Failed to fit one-hot encoder: {str(e)}")

        # Update state and metadata
        self.state = TransformerState.FITTED
        self.metadata = {
            "n_features_in": len(self._original_columns),
            "n_features_out": len(self._feature_names_out),
            "feature_names_in": self._original_columns,
            "feature_names_out": self._feature_names_out,
            "categories": [cat.tolist() for cat in self._encoder.categories_],
        }

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Transform data using the fitted one-hot encoder.

        Parameters
        ----------
        X : DataFrame
            Input data to transform.

        Returns
        -------
        DataFrame
            Transformed data with one-hot encoded columns.
        """
        self._check_is_fitted()
        X = X.copy()

        try:
            # Transform using sklearn encoder
            encoded = self._encoder.transform(X[self._original_columns])
            
            # Convert to dense if sparse
            if self.sparse:
                encoded = encoded.toarray()

            # Create DataFrame with encoded features
            encoded_df = pd.DataFrame(
                encoded,
                columns=self._feature_names_out,
                index=X.index
            )

            # Drop original columns and add encoded columns
            X = X.drop(columns=self._original_columns)
            X = pd.concat([X, encoded_df], axis=1)

        except Exception as e:
            raise EncodingError(f"Failed to transform data: {str(e)}")

        return X

    def get_feature_names_out(self) -> List[str]:
        """
        Get output feature names after transformation.

        Returns
        -------
        list of str
            Names of the output features.
        """
        self._check_is_fitted()
        return self._feature_names_out.copy()

    def inverse_transform(self, X: DataFrame) -> DataFrame:
        """
        Convert one-hot encoded columns back to original categorical values.

        Parameters
        ----------
        X : DataFrame
            One-hot encoded data.

        Returns
        -------
        DataFrame
            Data with original categorical columns.
        """
        self._check_is_fitted()
        X = X.copy()

        try:
            # Extract encoded columns
            encoded_data = X[self._feature_names_out].values

            # Inverse transform
            original_data = self._encoder.inverse_transform(encoded_data)

            # Create DataFrame with original columns
            original_df = pd.DataFrame(
                original_data,
                columns=self._original_columns,
                index=X.index
            )

            # Drop encoded columns and add original columns
            X = X.drop(columns=self._feature_names_out)
            for col in self._original_columns:
                X[col] = original_df[col]

        except Exception as e:
            raise EncodingError(f"Failed to inverse transform data: {str(e)}")

        return X
