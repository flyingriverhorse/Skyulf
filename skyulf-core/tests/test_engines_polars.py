"""Tests for skyulf.engines.polars_engine (SkyulfPolarsWrapper + PolarsEngine)."""

import numpy as np
import polars as pl
import pytest

from skyulf.engines.polars_engine import PolarsEngine, SkyulfPolarsWrapper
from skyulf.engines.registry import EngineName, EngineRegistry


@pytest.fixture
def pl_df():
    """A small deterministic polars DataFrame for wrapper tests."""
    return pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


# ---------------------------------------------------------------------------
# SkyulfPolarsWrapper
# ---------------------------------------------------------------------------


def test_wrapper_columns_returns_polars_columns(pl_df):
    """columns property should mirror the underlying polars DataFrame's columns."""
    wrapper = SkyulfPolarsWrapper(pl_df)
    assert wrapper.columns == ["a", "b"]


def test_wrapper_shape_matches_dataframe(pl_df):
    """shape property should mirror the underlying DataFrame's shape."""
    wrapper = SkyulfPolarsWrapper(pl_df)
    assert wrapper.shape == (3, 2)


def test_wrapper_select_returns_subset(pl_df):
    """select() should return a new wrapper containing only requested columns."""
    wrapper = SkyulfPolarsWrapper(pl_df)
    selected = wrapper.select(["a"])
    assert isinstance(selected, SkyulfPolarsWrapper)
    assert selected.columns == ["a"]


def test_wrapper_drop_removes_column(pl_df):
    """drop() should return a new wrapper without the dropped column."""
    wrapper = SkyulfPolarsWrapper(pl_df)
    dropped = wrapper.drop(["a"])
    assert dropped.columns == ["b"]


def test_wrapper_with_column_adds_new_column(pl_df):
    """with_column() should add a new column via with_columns, without mutating original."""
    wrapper = SkyulfPolarsWrapper(pl_df)
    result = wrapper.with_column("c", [7, 8, 9])
    assert "c" in result.columns
    assert "c" not in pl_df.columns


def test_wrapper_to_pandas_converts(pl_df):
    """to_pandas() should produce a pandas.DataFrame with matching values."""
    wrapper = SkyulfPolarsWrapper(pl_df)
    pdf = wrapper.to_pandas()
    assert list(pdf.columns) == ["a", "b"]
    assert pdf["a"].tolist() == [1, 2, 3]


def test_wrapper_to_arrow_roundtrips(pl_df):
    """to_arrow() should produce an Arrow table with matching row count."""
    wrapper = SkyulfPolarsWrapper(pl_df)
    table = wrapper.to_arrow()
    assert table.num_rows == 3


def test_wrapper_copy_is_independent(pl_df):
    """copy() should clone the underlying frame so mutation does not leak."""
    wrapper = SkyulfPolarsWrapper(pl_df)
    copied = wrapper.copy()
    assert isinstance(copied, SkyulfPolarsWrapper)
    assert copied.to_pandas()["a"].tolist() == pl_df["a"].to_list()


def test_wrapper_getitem_selects_column(pl_df):
    """__getitem__ should delegate to the underlying polars DataFrame."""
    wrapper = SkyulfPolarsWrapper(pl_df)
    result = wrapper["a"]
    assert result.to_list() == [1, 2, 3]


def test_wrapper_len_returns_height(pl_df):
    """__len__ should return the polars DataFrame's height (row count)."""
    wrapper = SkyulfPolarsWrapper(pl_df)
    assert len(wrapper) == 3


def test_wrapper_getattr_delegates_to_dataframe(pl_df):
    """__getattr__ should expose polars-only attributes/methods like `.height`."""
    wrapper = SkyulfPolarsWrapper(pl_df)
    assert wrapper.height == 3


# ---------------------------------------------------------------------------
# PolarsEngine
# ---------------------------------------------------------------------------


def test_polars_engine_registered_in_registry():
    """PolarsEngine should self-register under the 'polars' key at import time."""
    assert EngineRegistry.get("polars") is PolarsEngine


def test_polars_engine_is_compatible_true_for_dataframe(pl_df):
    """is_compatible should be True for a real polars.DataFrame."""
    assert PolarsEngine.is_compatible(pl_df) is True


def test_polars_engine_is_compatible_false_for_list():
    """is_compatible should be False for non-DataFrame data."""
    assert PolarsEngine.is_compatible([1, 2, 3]) is False


def test_polars_engine_from_pandas_converts(pl_df):
    """from_pandas should convert a pandas.DataFrame into a polars.DataFrame."""
    pdf = pl_df.to_pandas()
    converted = PolarsEngine.from_pandas(pdf)
    assert isinstance(converted, pl.DataFrame)
    assert converted.columns == ["a", "b"]


def test_polars_engine_to_numpy_from_wrapper(pl_df):
    """to_numpy should unwrap a SkyulfPolarsWrapper before converting."""
    wrapper = SkyulfPolarsWrapper(pl_df)
    result = PolarsEngine.to_numpy(wrapper)
    np.testing.assert_array_equal(result, pl_df.to_numpy())


def test_polars_engine_to_numpy_from_dataframe(pl_df):
    """to_numpy should convert a raw polars.DataFrame via its native to_numpy()."""
    result = PolarsEngine.to_numpy(pl_df)
    np.testing.assert_array_equal(result, pl_df.to_numpy())


def test_polars_engine_to_numpy_from_plain_list():
    """to_numpy should fall back to np.array for objects without to_numpy."""
    result = PolarsEngine.to_numpy([1, 2, 3])
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_polars_engine_wrap_creates_wrapper(pl_df):
    """wrap() should produce a SkyulfPolarsWrapper around a raw polars DataFrame."""
    wrapped = PolarsEngine.wrap(pl_df)
    assert isinstance(wrapped, SkyulfPolarsWrapper)


def test_polars_engine_wrap_is_idempotent_for_wrapper(pl_df):
    """wrap() should return the same wrapper instance if already wrapped."""
    wrapper = SkyulfPolarsWrapper(pl_df)
    assert PolarsEngine.wrap(wrapper) is wrapper


def test_polars_engine_create_dataframe_from_dict():
    """create_dataframe should build a polars.DataFrame from a dict of columns."""
    result = PolarsEngine.create_dataframe({"x": [1, 2], "y": [3, 4]})
    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["x", "y"]


def test_polars_engine_name_is_polars_enum():
    """The engine's `name` attribute should be the POLARS enum member."""
    assert PolarsEngine.name == EngineName.POLARS
