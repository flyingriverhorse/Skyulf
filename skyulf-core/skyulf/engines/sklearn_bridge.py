"""Convert Skyulf engine-agnostic frames into NumPy arrays for scikit-learn."""

from typing import Any

import numpy as np
import pandas as pd

from .pandas_engine import SkyulfPandasWrapper
from .registry import get_engine


class SklearnBridge:
    """Bridge between Skyulf DataFrames (Pandas/Polars) and Scikit-Learn (Numpy)."""

    @staticmethod
    def to_sklearn(X: Any) -> tuple[np.ndarray, Any]:
        """Convert input to Numpy array for Scikit-Learn.

        Args:
            X: Input data (Pandas, Polars, Wrapper, or (X, y) tuple).

        Returns:
            Tuple (X_numpy, y_numpy_or_None)
        """
        y = None

        # Handle tuple (X, y)
        if isinstance(X, tuple):
            X_data, y_data = X
            y = SklearnBridge._convert_single(y_data)

            # Flatten y if it is (N, 1) - common requirement for Sklearn targets
            if y is not None and isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()

            X = X_data

        X_numpy = SklearnBridge._convert_single(X)
        if X_numpy is None:
            raise ValueError("Input X could not be converted to a numpy array (got None).")
        return X_numpy, y

    @staticmethod
    def _convert_single(data: Any) -> np.ndarray | None:
        if data is None:
            return None

        # If it's already numpy, return it
        if isinstance(data, np.ndarray):
            return data

        # Use engine to convert
        engine = get_engine(data)
        values = engine.to_numpy(data)
        if values.dtype != object:
            return values

        native = data.to_native() if isinstance(data, SkyulfPandasWrapper) else data
        if isinstance(native, pd.DataFrame | pd.Series):
            return SklearnBridge._normalize_nullable_numeric(native, values)
        return values

    @staticmethod
    def _normalize_nullable_numeric(
        data: pd.DataFrame | pd.Series, values: np.ndarray
    ) -> np.ndarray:
        """Replace nullable numeric sentinels without rounding observed integer values."""
        dtypes = data.dtypes if isinstance(data, pd.DataFrame) else [data.dtype]
        # Mixed nullable numeric columns can retain pd.NA in an object array.
        # Replace only missing sentinels; floating casts would round integer categories.
        if (
            all(dtype.kind in "biuf" for dtype in dtypes)
            and any(isinstance(dtype, pd.api.extensions.ExtensionDtype) for dtype in dtypes)
            and pd.isna(values).any()
        ):
            return data.to_numpy(na_value=np.nan)
        return values
