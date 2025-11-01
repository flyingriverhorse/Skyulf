"""Base classes and interfaces for Skyulf-MLFlow library."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from skyulf_mlflow_library.core.types import (
    DataFrame,
    TransformerConfig,
    TransformerMetadata,
    TransformerState,
)
from skyulf_mlflow_library.exceptions import TransformerNotFittedError


class BaseTransformer(ABC):
    """
    Base class for all transformers in Skyulf-MLFlow.

    This class follows scikit-learn's transformer interface with fit/transform methods.
    All custom transformers should inherit from this class.

    Attributes:
        columns: List of column names to transform. If None, transforms all applicable columns.
        state: Current state of the transformer (not_fitted, fitted, partial_fitted).
        metadata: Metadata about the transformer and fitted parameters.
    """

    def __init__(self, columns: Optional[List[str]] = None, **kwargs):
        """
        Initialize the transformer.

        Args:
            columns: List of column names to transform. If None, all applicable columns.
            **kwargs: Additional configuration parameters.
        """
        self.columns = columns
        self.config = kwargs
        self.state = TransformerState.NOT_FITTED
        self.metadata: TransformerMetadata = {}

    @abstractmethod
    def fit(self, X: DataFrame, y: Optional[pd.Series] = None) -> "BaseTransformer":
        """
        Fit the transformer to the data.

        Args:
            X: Input DataFrame to fit.
            y: Target variable (optional, for supervised transformers).

        Returns:
            self: The fitted transformer.
        """
        pass

    @abstractmethod
    def transform(self, X: DataFrame) -> DataFrame:
        """
        Transform the data using the fitted transformer.

        Args:
            X: Input DataFrame to transform.

        Returns:
            DataFrame: Transformed DataFrame.

        Raises:
            TransformerNotFittedError: If transformer is not fitted.
        """
        pass

    def fit_transform(self, X: DataFrame, y: Optional[pd.Series] = None) -> DataFrame:
        """
        Fit the transformer and transform the data in one step.

        Args:
            X: Input DataFrame to fit and transform.
            y: Target variable (optional, for supervised transformers).

        Returns:
            DataFrame: Transformed DataFrame.
        """
        return self.fit(X, y).transform(X)

    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters of the transformer.

        Returns:
            Dict: Transformer parameters.
        """
        params = {"columns": self.columns}
        params.update(self.config)
        return params

    def set_params(self, **params) -> "BaseTransformer":
        """
        Set parameters of the transformer.

        Args:
            **params: Parameters to set.

        Returns:
            self: The transformer with updated parameters.
        """
        for key, value in params.items():
            if key == "columns":
                self.columns = value
            else:
                self.config[key] = value
        return self

    def get_metadata(self) -> TransformerMetadata:
        """
        Get metadata about the transformer.

        Returns:
            Dict: Transformer metadata.
        """
        return self.metadata.copy()

    def _check_is_fitted(self):
        """
        Check if the transformer is fitted.

        Raises:
            TransformerNotFittedError: If transformer is not fitted.
        """
        if self.state == TransformerState.NOT_FITTED:
            raise TransformerNotFittedError(
                f"{self.__class__.__name__} is not fitted yet. Call 'fit' first."
            )

    def _validate_columns(self, X: DataFrame):
        """
        Validate that specified columns exist in the DataFrame.

        Args:
            X: Input DataFrame to validate.

        Raises:
            ColumnNotFoundError: If specified columns are not in DataFrame.
        """
        if self.columns is not None:
            from skyulf_mlflow_library.exceptions import ColumnNotFoundError

            missing_cols = set(self.columns) - set(X.columns)
            if missing_cols:
                raise ColumnNotFoundError(
                    column=list(missing_cols)[0], available_columns=list(X.columns)
                )

    def __repr__(self) -> str:
        """String representation of the transformer."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params_str})"


class BaseEncoder(BaseTransformer):
    """Base class for all encoding transformers."""

    def __init__(self, columns: Optional[List[str]] = None, handle_unknown: str = "error", **kwargs):
        """
        Initialize the encoder.

        Args:
            columns: List of column names to encode.
            handle_unknown: How to handle unknown categories ('error', 'ignore', 'use_encoded_value').
            **kwargs: Additional configuration parameters.
        """
        super().__init__(columns=columns, handle_unknown=handle_unknown, **kwargs)
        self.handle_unknown = handle_unknown


class BaseScaler(BaseTransformer):
    """Base class for all scaling transformers."""

    def __init__(self, columns: Optional[List[str]] = None, copy: bool = True, **kwargs):
        """
        Initialize the scaler.

        Args:
            columns: List of column names to scale.
            copy: Whether to copy data before scaling.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(columns=columns, copy=copy, **kwargs)
        self.copy = copy


class BaseImputer(BaseTransformer):
    """Base class for all imputation transformers."""

    def __init__(
        self, columns: Optional[List[str]] = None, strategy: str = "mean", fill_value: Any = None, **kwargs
    ):
        """
        Initialize the imputer.

        Args:
            columns: List of column names to impute.
            strategy: Imputation strategy.
            fill_value: Value to use when strategy is 'constant'.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(columns=columns, strategy=strategy, fill_value=fill_value, **kwargs)
        self.strategy = strategy
        self.fill_value = fill_value


class BaseFilter(ABC):
    """
    Base class for filter operations (operations that don't require fitting).

    Filters are stateless operations that can be applied directly to data
    without a fit step.
    """

    @abstractmethod
    def apply(self, X: DataFrame) -> DataFrame:
        """
        Apply the filter to the data.

        Args:
            X: Input DataFrame to filter.

        Returns:
            DataFrame: Filtered DataFrame.
        """
        pass

    def __call__(self, X: DataFrame) -> DataFrame:
        """Allow filter to be called as a function."""
        return self.apply(X)
