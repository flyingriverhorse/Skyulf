"""Tests for skyulf.engines.pandas_engine (SkyulfPandasWrapper + PandasEngine)."""

import numpy as np
import pandas as pd
import pytest

from skyulf.engines.pandas_engine import PandasEngine, SkyulfPandasWrapper
from skyulf.engines.registry import EngineName, EngineRegistry


@pytest.fixture
def df():
    """A small deterministic pandas DataFrame for wrapper tests."""
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


# ---------------------------------------------------------------------------
# SkyulfPandasWrapper
# ---------------------------------------------------------------------------


def test_wrapper_columns_returns_list(df):
    """columns property should return a plain list, not a pandas Index."""
    wrapper = SkyulfPandasWrapper(df)
    assert wrapper.columns == ["a", "b"]
    assert isinstance(wrapper.columns, list)


def test_wrapper_shape_matches_dataframe(df):
    """shape property should mirror the underlying DataFrame's shape."""
    wrapper = SkyulfPandasWrapper(df)
    assert wrapper.shape == (3, 2)


def test_wrapper_select_returns_subset(df):
    """select() should return a new wrapper containing only requested columns."""
    wrapper = SkyulfPandasWrapper(df)
    selected = wrapper.select(["a"])
    assert isinstance(selected, SkyulfPandasWrapper)
    assert selected.columns == ["a"]


def test_wrapper_drop_removes_column(df):
    """drop() should return a new wrapper without the dropped column."""
    wrapper = SkyulfPandasWrapper(df)
    dropped = wrapper.drop(["a"])
    assert dropped.columns == ["b"]


def test_wrapper_select_with_bare_string_returns_dataframe_not_series(df):
    """select() with a plain string column name must return a wrapper around a
    DataFrame (matching Polars' select() semantics), not a bare Series
    (regression guard for r5 select() str-vs-list engine parity bug)."""
    wrapper = SkyulfPandasWrapper(df)
    selected = wrapper.select("a")
    assert isinstance(selected, SkyulfPandasWrapper)
    assert isinstance(selected.to_pandas(), pd.DataFrame)
    assert selected.columns == ["a"]


def test_wrapper_with_column_adds_new_column(df):
    """with_column() should add a new column via assign, without mutating original."""
    wrapper = SkyulfPandasWrapper(df)
    result = wrapper.with_column("c", [7, 8, 9])
    assert "c" in result.columns
    assert "c" not in df.columns


def test_wrapper_with_column_ignores_mismatched_series_index(df):
    """with_column() must assign Series values positionally, matching Polars,
    instead of pandas' default index-alignment which would silently produce
    NaNs for a Series whose index doesn't match the target frame's
    (regression guard for r5 pandas with_column index-misalignment bug)."""
    wrapper = SkyulfPandasWrapper(df)
    mismatched = pd.Series([7, 8, 9], index=[100, 101, 102])
    result = wrapper.with_column("c", mismatched)
    assert list(result.to_pandas()["c"]) == [7, 8, 9]
    assert not result.to_pandas()["c"].isna().any()


def test_wrapper_with_column_raises_on_length_mismatch(df):
    """with_column() should raise a clear error rather than silently
    NaN-filling when a mismatched-index Series has the wrong length."""
    wrapper = SkyulfPandasWrapper(df)
    too_short = pd.Series([7, 8], index=[100, 101])
    with pytest.raises(ValueError, match="Length mismatch"):
        wrapper.with_column("c", too_short)


def test_wrapper_to_pandas_returns_same_object(df):
    """to_pandas() should return the underlying DataFrame unchanged."""
    wrapper = SkyulfPandasWrapper(df)
    assert wrapper.to_pandas() is df


def test_wrapper_to_arrow_roundtrips(df):
    """to_arrow() should produce a pyarrow Table with matching data."""
    wrapper = SkyulfPandasWrapper(df)
    table = wrapper.to_arrow()
    assert table.num_rows == 3
    assert table.column_names == ["a", "b"]


def test_wrapper_copy_is_independent(df):
    """copy() should return a wrapper whose underlying frame is a distinct copy."""
    wrapper = SkyulfPandasWrapper(df)
    copied = wrapper.copy()
    copied.to_pandas().loc[0, "a"] = 999
    assert df.loc[0, "a"] == 1


def test_wrapper_getitem_selects_column(df):
    """__getitem__ should delegate to the underlying DataFrame's column access."""
    wrapper = SkyulfPandasWrapper(df)
    pd.testing.assert_series_equal(wrapper["a"], df["a"])


def test_wrapper_setitem_mutates_underlying_frame(df):
    """__setitem__ should assign into the wrapped DataFrame directly."""
    wrapper = SkyulfPandasWrapper(df)
    wrapper["c"] = [10, 11, 12]
    assert list(df["c"]) == [10, 11, 12]


def test_wrapper_setitem_ignores_mismatched_series_index(df):
    """__setitem__ must also assign positionally for a mismatched-index
    Series, mirroring with_column()'s fix (regression guard for r5)."""
    wrapper = SkyulfPandasWrapper(df)
    mismatched = pd.Series([10, 11, 12], index=[7, 8, 9])
    wrapper["c"] = mismatched
    assert list(df["c"]) == [10, 11, 12]
    assert not df["c"].isna().any()


def test_wrapper_len_returns_row_count(df):
    """__len__ should return the number of rows."""
    wrapper = SkyulfPandasWrapper(df)
    assert len(wrapper) == 3


def test_wrapper_getattr_delegates_to_dataframe(df):
    """__getattr__ should expose pandas-only attributes like `.loc`."""
    wrapper = SkyulfPandasWrapper(df)
    assert wrapper.loc[0, "a"] == 1


# ---------------------------------------------------------------------------
# PandasEngine
# ---------------------------------------------------------------------------


def test_pandas_engine_registered_in_registry():
    """PandasEngine should self-register under the 'pandas' key at import time."""
    assert EngineRegistry.get("pandas") is PandasEngine


def test_pandas_engine_is_compatible_true_for_dataframe(df):
    """is_compatible should be True for a real pandas.DataFrame."""
    assert PandasEngine.is_compatible(df) is True


def test_pandas_engine_is_compatible_false_for_list():
    """is_compatible should be False for non-DataFrame data."""
    assert PandasEngine.is_compatible([1, 2, 3]) is False


def test_pandas_engine_from_pandas_is_identity(df):
    """from_pandas is a no-op identity conversion for the pandas engine."""
    assert PandasEngine.from_pandas(df) is df


def test_pandas_engine_to_numpy_from_dataframe(df):
    """to_numpy should convert a DataFrame via its native to_numpy()."""
    result = PandasEngine.to_numpy(df)
    np.testing.assert_array_equal(result, df.to_numpy())


def test_pandas_engine_to_numpy_from_wrapper(df):
    """to_numpy should unwrap a SkyulfPandasWrapper before converting."""
    wrapper = SkyulfPandasWrapper(df)
    result = PandasEngine.to_numpy(wrapper)
    np.testing.assert_array_equal(result, df.to_numpy())


class _NoToNumpyDelegationWrapper(SkyulfPandasWrapper):
    """A wrapper variant whose __getattr__ hides `to_numpy` so that
    `hasattr(wrapper, "to_numpy")` is False, forcing PandasEngine.to_numpy's
    isinstance(SkyulfPandasWrapper) fallback branch to run."""

    def __getattr__(self, name):
        if name == "to_numpy":
            raise AttributeError(name)
        return super().__getattr__(name)


def test_pandas_engine_to_numpy_wrapper_without_delegated_to_numpy(df):
    """to_numpy must fall back to `.to_pandas().to_numpy()` for a wrapper whose
    __getattr__ does not expose `to_numpy` directly (the isinstance branch)."""
    wrapper = _NoToNumpyDelegationWrapper(df)
    assert not hasattr(wrapper, "to_numpy")
    result = PandasEngine.to_numpy(wrapper)
    np.testing.assert_array_equal(result, df.to_numpy())


def test_pandas_engine_to_numpy_from_plain_list():
    """to_numpy should fall back to np.array for objects without to_numpy."""
    result = PandasEngine.to_numpy([1, 2, 3])
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_pandas_engine_wrap_creates_wrapper(df):
    """wrap() should produce a SkyulfPandasWrapper around a raw DataFrame."""
    wrapped = PandasEngine.wrap(df)
    assert isinstance(wrapped, SkyulfPandasWrapper)
    assert wrapped.to_pandas() is df


def test_pandas_engine_wrap_is_idempotent_for_wrapper(df):
    """wrap() should return the same wrapper instance if already wrapped."""
    wrapper = SkyulfPandasWrapper(df)
    assert PandasEngine.wrap(wrapper) is wrapper


def test_pandas_engine_create_dataframe_from_dict():
    """create_dataframe should build a pandas.DataFrame from a dict of columns."""
    result = PandasEngine.create_dataframe({"x": [1, 2], "y": [3, 4]})
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["x", "y"]


def test_pandas_engine_name_is_pandas_enum():
    """The engine's `name` attribute should be the PANDAS enum member."""
    assert PandasEngine.name == EngineName.PANDAS
