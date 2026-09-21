"""Convert Skyulf engine-agnostic frames into NumPy arrays for scikit-learn."""

from datetime import date, time, timedelta
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from .pandas_engine import SkyulfPandasWrapper
from .polars_engine import SkyulfPolarsWrapper
from .registry import get_engine


class SklearnBridge:
    """Bridge between Skyulf DataFrames (Pandas/Polars) and Scikit-Learn (Numpy)."""

    @staticmethod
    def to_sklearn(X: Any, *, validate_features: bool = False) -> tuple[np.ndarray, Any]:
        """Convert input to Numpy array for Scikit-Learn.

        Args:
            X: Input data (Pandas, Polars, Wrapper, or (X, y) tuple).
            validate_features: Reject raw temporal model features. Leave disabled
                for preprocessing or fold partitioning before feature extraction.

        Returns:
            Tuple (X_numpy, y_numpy_or_None)

        Raises:
            TypeError: If either input is not a supported frame, array or sequence.
            ValueError: If features are None or contain unconverted temporal values.
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

        if validate_features:
            SklearnBridge.validate_features(X)
        X_numpy = SklearnBridge._convert_single(X)
        if X_numpy is None:
            raise ValueError("Input X could not be converted to a numpy array (got None).")
        return X_numpy, y

    @staticmethod
    def validate_features(data: Any) -> None:
        """Reject raw temporal features before engine conversion can erase their units."""
        native = (
            data.to_native()
            if isinstance(data, SkyulfPandasWrapper | SkyulfPolarsWrapper)
            else data
        )
        columns = []
        if isinstance(native, pl.DataFrame | pl.Series):
            columns = SklearnBridge._polars_temporal_columns(native)
        elif isinstance(native, pd.DataFrame | pd.Series):
            columns = SklearnBridge._pandas_temporal_columns(native)
        elif isinstance(native, np.ndarray | list | tuple) and SklearnBridge._contains_temporal(
            np.asarray(native)
        ):
            columns = ["array input"]
        if columns:
            raise ValueError(
                f"Raw temporal features are not supported: {', '.join(columns)}. "
                "Use DateFeatures with drop_original=True, remove these columns, or "
                "explicitly convert them to numeric features before model fitting or prediction."
            )

    @staticmethod
    def _polars_temporal_columns(native: pl.DataFrame | pl.Series) -> list[str]:
        """Find temporal Polars columns before conversion erases their units."""
        schema = native.schema if isinstance(native, pl.DataFrame) else {native.name: native.dtype}
        columns = []
        for name, dtype in schema.items():
            series = native[name] if isinstance(native, pl.DataFrame) else native
            if dtype.is_temporal() or (
                dtype == pl.Object and SklearnBridge._contains_temporal(series.to_numpy())
            ):
                columns.append(str(name))
        return columns

    @staticmethod
    def _pandas_temporal_columns(native: pd.DataFrame | pd.Series) -> list[str]:
        """Find typed and object-backed temporal pandas columns."""
        items = native.items() if isinstance(native, pd.DataFrame) else [(native.name, native)]
        columns = []
        for name, series in items:
            if (
                series.dtype.kind in "Mm"
                or isinstance(series.dtype, pd.PeriodDtype)
                or series.dtype.kind == "O"
                and SklearnBridge._contains_temporal(series.to_numpy())
            ):
                columns.append(str(name))
        return columns

    @staticmethod
    def _contains_temporal(values: np.ndarray) -> bool:
        """Recognize typed arrays and object-backed dates without parsing strings."""
        if values.dtype.kind in "Mm":
            return True
        temporal_types = (date, time, timedelta, np.datetime64, np.timedelta64, pd.Period)
        return values.dtype.kind == "O" and any(
            isinstance(value, temporal_types) for value in values.flat
        )

    @staticmethod
    def _convert_single(data: Any) -> np.ndarray | None:
        """Convert supported array-like containers without accepting arbitrary scalars."""
        if data is None:
            return None

        # If it's already numpy, return it
        if isinstance(data, np.ndarray):
            return data

        native = (
            data.to_native()
            if isinstance(data, SkyulfPandasWrapper | SkyulfPolarsWrapper)
            else data
        )
        if not isinstance(
            native, pd.DataFrame | pd.Series | pl.DataFrame | pl.Series | list | tuple
        ):
            raise TypeError(
                f"Unsupported input container {type(data).__name__}. "
                "Expected a pandas or Polars DataFrame/Series (or Skyulf wrapper), "
                "a NumPy array, or a Python list/tuple. Select a split before conversion."
            )

        # Use engine to convert
        engine = get_engine(data)
        values = engine.to_numpy(data)
        if values.dtype != object:
            return values

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
