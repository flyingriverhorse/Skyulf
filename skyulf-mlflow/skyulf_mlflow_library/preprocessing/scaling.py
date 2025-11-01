"""Scaling transformers for Skyulf MLflow."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MaxAbsScaler as SklearnMaxAbsScaler,
    MinMaxScaler as SklearnMinMaxScaler,
    RobustScaler as SklearnRobustScaler,
    StandardScaler as SklearnStandardScaler,
)

from ..core.base import BaseScaler
from ..core.types import DataFrame, TransformerState
from skyulf_mlflow_library.exceptions import ScalingError


class StandardScaler(BaseScaler):
    """Standardize features by removing the mean and scaling to unit variance."""

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        with_mean: bool = True,
        with_std: bool = True,
        copy: bool = True,
        **kwargs,
    ):
        super().__init__(columns=columns, copy=copy, **kwargs)
        self.with_mean = with_mean
        self.with_std = with_std
        self._scaler: Optional[SklearnStandardScaler] = None

    def fit(self, X: DataFrame, y: Optional[pd.Series] = None) -> "StandardScaler":
        X = X.copy()
        self._validate_columns(X)

        if self.columns is None:
            self._original_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self._original_columns = self.columns.copy()

        if not self._original_columns:
            raise ScalingError("No numeric columns to scale.")

        self._scaler = SklearnStandardScaler(
            with_mean=self.with_mean,
            with_std=self.with_std,
            copy=self.copy,
        )

        try:
            self._scaler.fit(X[self._original_columns])
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ScalingError(f"Failed to fit standard scaler: {exc}") from exc

        self.state = TransformerState.FITTED
        self.metadata = {
            "n_features": len(self._original_columns),
            "feature_names": self._original_columns,
            "mean": self._scaler.mean_.tolist() if self.with_mean else None,
            "scale": self._scaler.scale_.tolist() if self.with_std else None,
            "var": self._scaler.var_.tolist() if self.with_std else None,
        }

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        self._check_is_fitted()
        X = X.copy() if self.copy else X

        try:
            X[self._original_columns] = self._scaler.transform(X[self._original_columns])
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ScalingError(f"Failed to transform data: {exc}") from exc

        return X

    def inverse_transform(self, X: DataFrame) -> DataFrame:
        self._check_is_fitted()
        X = X.copy()

        try:
            X[self._original_columns] = self._scaler.inverse_transform(X[self._original_columns])
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ScalingError(f"Failed to inverse transform data: {exc}") from exc

        return X


class MinMaxScaler(BaseScaler):
    """Transform features by scaling each feature to a given range."""

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        feature_range: Tuple[float, float] = (0.0, 1.0),
        clip: bool = False,
        copy: bool = True,
        **kwargs,
    ):
        super().__init__(columns=columns, copy=copy, **kwargs)
        self.feature_range = feature_range
        self.clip = clip
        self._scaler: Optional[SklearnMinMaxScaler] = None

    def fit(self, X: DataFrame, y: Optional[pd.Series] = None) -> "MinMaxScaler":
        X = X.copy()
        self._validate_columns(X)

        if self.columns is None:
            self._original_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self._original_columns = self.columns.copy()

        if not self._original_columns:
            raise ScalingError("No numeric columns to scale.")

        self._scaler = SklearnMinMaxScaler(
            feature_range=self.feature_range,
            clip=self.clip,
            copy=self.copy,
        )

        try:
            self._scaler.fit(X[self._original_columns])
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ScalingError(f"Failed to fit MinMax scaler: {exc}") from exc

        self.state = TransformerState.FITTED
        self.metadata = {
            "n_features": len(self._original_columns),
            "feature_names": self._original_columns,
            "min": self._scaler.min_.tolist(),
            "scale": self._scaler.scale_.tolist(),
            "data_min": self._scaler.data_min_.tolist(),
            "data_max": self._scaler.data_max_.tolist(),
            "feature_range": self.feature_range,
        }

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        self._check_is_fitted()
        X = X.copy() if self.copy else X

        try:
            X[self._original_columns] = self._scaler.transform(X[self._original_columns])
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ScalingError(f"Failed to transform data: {exc}") from exc

        return X

    def inverse_transform(self, X: DataFrame) -> DataFrame:
        self._check_is_fitted()
        X = X.copy()

        try:
            X[self._original_columns] = self._scaler.inverse_transform(X[self._original_columns])
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ScalingError(f"Failed to inverse transform data: {exc}") from exc

        return X


class RobustScaler(BaseScaler):
    """Scale features using statistics that are robust to outliers."""

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        copy: bool = True,
        **kwargs,
    ):
        super().__init__(columns=columns, copy=copy, **kwargs)
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self._scaler: Optional[SklearnRobustScaler] = None

    def fit(self, X: DataFrame, y: Optional[pd.Series] = None) -> "RobustScaler":
        X = X.copy()
        self._validate_columns(X)

        if self.columns is None:
            self._original_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self._original_columns = self.columns.copy()

        if not self._original_columns:
            raise ScalingError("No numeric columns to scale.")

        self._scaler = SklearnRobustScaler(
            with_centering=self.with_centering,
            with_scaling=self.with_scaling,
            quantile_range=self.quantile_range,
            copy=self.copy,
        )

        try:
            self._scaler.fit(X[self._original_columns])
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ScalingError(f"Failed to fit Robust scaler: {exc}") from exc

        self.state = TransformerState.FITTED
        self.metadata = {
            "n_features": len(self._original_columns),
            "feature_names": self._original_columns,
            "center": self._scaler.center_.tolist() if self.with_centering else None,
            "scale": self._scaler.scale_.tolist() if self.with_scaling else None,
            "quantile_range": list(self.quantile_range),
        }

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        self._check_is_fitted()
        X = X.copy() if self.copy else X

        try:
            X[self._original_columns] = self._scaler.transform(X[self._original_columns])
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ScalingError(f"Failed to transform data: {exc}") from exc

        return X

    def inverse_transform(self, X: DataFrame) -> DataFrame:
        self._check_is_fitted()
        X = X.copy()

        try:
            X[self._original_columns] = self._scaler.inverse_transform(X[self._original_columns])
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ScalingError(f"Failed to inverse transform data: {exc}") from exc

        return X


class MaxAbsScaler(BaseScaler):
    """Scale each feature by its maximum absolute value."""

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        copy: bool = True,
        **kwargs,
    ):
        super().__init__(columns=columns, copy=copy, **kwargs)
        self._scaler: Optional[SklearnMaxAbsScaler] = None

    def fit(self, X: DataFrame, y: Optional[pd.Series] = None) -> "MaxAbsScaler":
        X = X.copy()
        self._validate_columns(X)

        if self.columns is None:
            self._original_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self._original_columns = self.columns.copy()

        if not self._original_columns:
            raise ScalingError("No numeric columns to scale.")

        self._scaler = SklearnMaxAbsScaler(copy=self.copy)

        try:
            self._scaler.fit(X[self._original_columns])
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ScalingError(f"Failed to fit MaxAbs scaler: {exc}") from exc

        self.state = TransformerState.FITTED
        self.metadata = {
            "n_features": len(self._original_columns),
            "feature_names": self._original_columns,
            "max_abs": self._scaler.max_abs_.tolist(),
            "scale": self._scaler.scale_.tolist(),
        }

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        self._check_is_fitted()
        X = X.copy() if self.copy else X

        try:
            X[self._original_columns] = self._scaler.transform(X[self._original_columns])
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ScalingError(f"Failed to transform data: {exc}") from exc

        return X

    def inverse_transform(self, X: DataFrame) -> DataFrame:
        self._check_is_fitted()
        X = X.copy()

        try:
            X[self._original_columns] = self._scaler.inverse_transform(X[self._original_columns])
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ScalingError(f"Failed to inverse transform data: {exc}") from exc

        return X
