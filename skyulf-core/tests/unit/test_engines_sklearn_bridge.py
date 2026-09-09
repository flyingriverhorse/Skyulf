"""Tests for skyulf.engines.sklearn_bridge (SklearnBridge dataframe/array conversion)."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines.registry import EngineRegistry
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


@pytest.mark.parametrize("wrapped", [False, True])
def test_to_sklearn_normalizes_nullable_numeric_frame(wrapped: bool) -> None:
    """Nullable numeric features must reach sklearn as numbers and NaN, never pd.NA."""
    frame = pd.DataFrame(
        {
            "integer": pd.Series([1, None, 3], dtype="Int64"),
            "real": pd.Series([2, None, 4], dtype="Float64"),
            "flag": pd.Series([True, None, False], dtype="boolean"),
        }
    )
    original = frame.copy(deep=True)
    data = EngineRegistry.wrap(frame) if wrapped else frame

    values, target = SklearnBridge.to_sklearn(data)

    assert values[[0, 2]].tolist() == [[1, 2, True], [3, 4, False]]
    assert all(np.isnan(value) for value in values[1])
    assert target is None
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("as_frame", [False, True])
def test_to_sklearn_normalizes_nullable_boolean_target(as_frame: bool) -> None:
    """Nullable boolean targets preserve missing labels as NaN across target shapes."""
    target = pd.Series([True, None, False], dtype="boolean")
    values, labels = SklearnBridge.to_sklearn(
        (np.array([[1], [2], [3]]), target.to_frame() if as_frame else target)
    )

    assert values.shape == (3, 1)
    assert labels.shape == (3,)
    assert labels[[0, 2]].tolist() == [True, False]
    assert np.isnan(labels[1])


@pytest.mark.parametrize("dtype", ["category", "string", "object"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_to_sklearn_preserves_nonnumeric_columns(dtype: str, wrapped: bool) -> None:
    """The numeric bridge fix must leave categorical values and missing sentinels intact."""
    frame = pd.DataFrame(
        {
            "number": pd.Series([1, None, 3], dtype="Int64"),
            "label": pd.Series(["north", "south", "north"], dtype=dtype),
        }
    )
    data = EngineRegistry.wrap(frame) if wrapped else frame

    values, _ = SklearnBridge.to_sklearn(data)

    assert values.dtype == object
    assert values[:, 1].tolist() == ["north", "south", "north"]
    assert values[1, 0] is pd.NA
    assert values[[0, 2], 0].tolist() == [1, 3]


@pytest.mark.parametrize("as_frame", [False, True])
def test_to_sklearn_preserves_nullable_integer_label_precision(as_frame: bool) -> None:
    """Integer class labels above float precision must remain distinct after conversion."""
    target = pd.Series([2**53, 2**53 + 1], dtype="Int64")

    _, labels = SklearnBridge.to_sklearn(
        (np.array([[1], [2]]), target.to_frame() if as_frame else target)
    )

    assert labels.dtype.kind == "i"
    assert labels.tolist() == [2**53, 2**53 + 1]


def test_to_sklearn_preserves_nonmissing_nullable_integer_object_array() -> None:
    """Mixed nullable integer columns without missing data must keep exact large values."""
    frame = pd.DataFrame(
        {
            "first": pd.Series([2**53, 2**53 + 1], dtype="Int64"),
            "second": pd.Series([1, 2], dtype="Int32"),
        }
    )

    _, labels = SklearnBridge.to_sklearn((np.array([[1], [2]]), frame))

    assert labels.tolist() == [[2**53, 1], [2**53 + 1, 2]]
