"""Tests for skyulf.engines.sklearn_bridge (SklearnBridge dataframe/array conversion)."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines.sklearn_bridge import SklearnBridge


def test_to_sklearn_converts_plain_dataframe():
    """A bare pandas DataFrame (no y) should convert to a numpy array with y=None."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    X_np, y = SklearnBridge.to_sklearn(df)
    assert isinstance(X_np, np.ndarray)
    np.testing.assert_array_equal(X_np, df.to_numpy())
    assert y is None


def test_to_sklearn_converts_tuple_with_series_target():
    """An (X, y) tuple should convert both members, keeping shapes intact."""
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    X_np, y_np = SklearnBridge.to_sklearn((X, y))
    np.testing.assert_array_equal(X_np, X.to_numpy())
    np.testing.assert_array_equal(y_np, y.to_numpy())


def test_to_sklearn_flattens_2d_single_column_target():
    """A (N, 1)-shaped y array should be raveled to 1-D for sklearn compatibility."""
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.DataFrame({"target": [0, 1, 0]})  # becomes (3, 1) numpy array
    X_np, y_np = SklearnBridge.to_sklearn((X, y))
    assert y_np.ndim == 1
    np.testing.assert_array_equal(y_np, np.array([0, 1, 0]))


def test_to_sklearn_polars_dataframe_converts_via_engine():
    """A polars DataFrame should be converted to numpy via the polars engine."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    X_np, y = SklearnBridge.to_sklearn(df)
    np.testing.assert_array_equal(X_np, df.to_numpy())
    assert y is None


def test_to_sklearn_raises_when_x_converts_to_none():
    """If X converts to None (e.g. X itself is None), a ValueError should be raised."""
    with pytest.raises(ValueError, match="could not be converted"):
        SklearnBridge.to_sklearn(None)


def test_to_sklearn_passes_through_existing_numpy_array():
    """An already-numpy X should be returned unchanged (no re-conversion)."""
    X = np.array([[1, 2], [3, 4]])
    X_np, y = SklearnBridge.to_sklearn(X)
    assert X_np is X
    assert y is None


def test_to_sklearn_tuple_with_none_target():
    """A tuple with a None target should return y=None without raising."""
    X = pd.DataFrame({"a": [1, 2]})
    X_np, y_np = SklearnBridge.to_sklearn((X, None))
    np.testing.assert_array_equal(X_np, X.to_numpy())
    assert y_np is None


def test_convert_single_returns_none_for_none_input():
    """_convert_single should short-circuit to None for None input."""
    assert SklearnBridge._convert_single(None) is None
