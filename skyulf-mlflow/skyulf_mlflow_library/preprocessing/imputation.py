"""Imputation transformers for handling missing values."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer as SklearnSimpleImputer

from ..core.base import BaseImputer
from ..core.types import DataFrame, TransformerState
from skyulf_mlflow_library.exceptions import ImputationError


class SimpleImputer(BaseImputer):
    """Simple imputation transformer for completing missing values."""

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        strategy: str = "mean",
        fill_value: Optional[Union[str, float]] = None,
        copy: bool = True,
        **kwargs,
    ):
        super().__init__(columns=columns, strategy=strategy, fill_value=fill_value, **kwargs)
        self.copy = copy
        self._imputer: Optional[SklearnSimpleImputer] = None
        self._statistics: Optional[Dict[str, float]] = None

    def fit(self, X: DataFrame, y: Optional[pd.Series] = None) -> "SimpleImputer":
        X = X.copy()
        self._validate_columns(X)

        if self.columns is None:
            # For most_frequent, include all columns; for mean/median, only numeric
            if self.strategy == 'most_frequent':
                self._original_columns = X.columns.tolist()
            else:
                self._original_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self._original_columns = self.columns.copy()

        if not self._original_columns:
            raise ImputationError("No columns to impute. Specify columns or ensure data has numeric columns.")

        self._imputer = SklearnSimpleImputer(
            strategy=self.strategy,
            fill_value=self.fill_value,
            copy=self.copy,
        )

        try:
            self._imputer.fit(X[self._original_columns])
            if hasattr(self._imputer, "statistics_"):
                self._statistics = dict(zip(self._original_columns, self._imputer.statistics_))
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ImputationError(f"Failed to fit imputer: {exc}") from exc

        self.state = TransformerState.FITTED
        self.metadata = {
            "n_features": len(self._original_columns),
            "feature_names": self._original_columns,
            "strategy": self.strategy,
            "statistics": self._statistics,
        }

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        self._check_is_fitted()
        X = X.copy() if self.copy else X

        try:
            X[self._original_columns] = self._imputer.transform(X[self._original_columns])
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ImputationError(f"Failed to transform data: {exc}") from exc

        return X


try:
    from sklearn.impute import KNNImputer as SklearnKNNImputer

    class KNNImputer(BaseImputer):
        """K-Nearest Neighbors imputation for completing missing values."""

        def __init__(
            self,
            columns: Optional[List[str]] = None,
            n_neighbors: int = 5,
            weights: str = "uniform",
            copy: bool = True,
            **kwargs,
        ):
            super().__init__(columns=columns, **kwargs)
            self.n_neighbors = n_neighbors
            self.weights = weights
            self.copy = copy
            self._imputer: Optional[SklearnKNNImputer] = None

        def fit(self, X: DataFrame, y: Optional[pd.Series] = None) -> "KNNImputer":
            X = X.copy()
            self._validate_columns(X)

            if self.columns is None:
                self._original_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            else:
                self._original_columns = self.columns.copy()

            if not self._original_columns:
                raise ImputationError("No columns to impute.")

            self._imputer = SklearnKNNImputer(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                copy=self.copy,
            )

            try:
                self._imputer.fit(X[self._original_columns])
            except Exception as exc:  # pragma: no cover - defensive guard
                raise ImputationError(f"Failed to fit KNN imputer: {exc}") from exc

            self.state = TransformerState.FITTED
            self.metadata = {
                "n_features": len(self._original_columns),
                "feature_names": self._original_columns,
                "n_neighbors": self.n_neighbors,
                "weights": self.weights,
            }

            return self

        def transform(self, X: DataFrame) -> DataFrame:
            self._check_is_fitted()
            X = X.copy() if self.copy else X

            try:
                X[self._original_columns] = self._imputer.transform(X[self._original_columns])
            except Exception as exc:  # pragma: no cover - defensive guard
                raise ImputationError(f"Failed to transform data: {exc}") from exc

            return X

except ImportError:  # pragma: no cover - optional dependency guard
    pass
