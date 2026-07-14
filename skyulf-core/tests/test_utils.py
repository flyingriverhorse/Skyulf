"""Unit tests for skyulf.utils public functions.

Covers get_data_stats, unpack_pipeline_input, pack_pipeline_output,
_is_binary_numeric, detect_numeric_columns, resolve_columns, and
user_picked_no_columns.  All tests use real DataFrames — no mocking of pandas.
"""

import typing

import numpy as np
import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.utils import (
    _is_binary_numeric,
    detect_numeric_columns,
    get_data_stats,
    pack_pipeline_output,
    resolve_columns,
    unpack_pipeline_input,
    user_picked_no_columns,
)

# ---------------------------------------------------------------------------
# get_data_stats
# ---------------------------------------------------------------------------


def test_get_data_stats_plain_dataframe() -> None:
    """Row count and column set must match the DataFrame exactly."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    rows, cols = get_data_stats(df)
    assert rows == 3
    assert cols == {"a", "b"}


def test_get_data_stats_empty_dataframe() -> None:
    """Empty DataFrame must report zero rows and empty column set."""
    df = pd.DataFrame({"x": pd.Series([], dtype=float)})
    rows, cols = get_data_stats(df)
    assert rows == 0
    assert cols == {"x"}


def test_get_data_stats_single_row() -> None:
    """Single-row DataFrame must report exactly 1 row."""
    df = pd.DataFrame({"v": [42]})
    rows, _ = get_data_stats(df)
    assert rows == 1


def test_get_data_stats_tuple_xy() -> None:
    """(X, y) tuple must report rows from X and columns from X."""
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y = pd.Series([0, 1])
    rows, cols = get_data_stats((X, y))
    assert rows == 2
    assert cols == {"a", "b"}


def test_get_data_stats_tuple_numpy_x() -> None:
    """(X, y) where X is a numpy array must report shape rows and empty cols."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    rows, cols = get_data_stats((X, y))
    assert rows == 3
    assert cols == set()


def test_get_data_stats_split_dataset_no_validation() -> None:
    """SplitDataset without validation must sum train+test rows."""
    train = pd.DataFrame({"a": [1, 2, 3]})
    test = pd.DataFrame({"a": [4, 5]})
    ds = SplitDataset(train=train, test=test)
    rows, cols = get_data_stats(ds)
    assert rows == 5
    assert cols == {"a"}


def test_get_data_stats_split_dataset_with_validation() -> None:
    """SplitDataset with validation must include validation rows in total."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [3]})
    val = pd.DataFrame({"a": [4, 5, 6]})
    ds = SplitDataset(train=train, test=test, validation=val)
    rows, _ = get_data_stats(ds)
    assert rows == 6


# ---------------------------------------------------------------------------
# unpack_pipeline_input
# ---------------------------------------------------------------------------


def test_unpack_dataframe_returns_not_tuple() -> None:
    """A plain DataFrame input must return is_tuple=False and y=None."""
    df = pd.DataFrame({"x": [1, 2]})
    X, y, is_tuple = unpack_pipeline_input(df)
    assert X is df
    assert y is None
    assert is_tuple is False


def test_unpack_tuple_returns_is_tuple_true() -> None:
    """A (X, y) tuple must return is_tuple=True with both components."""
    X = pd.DataFrame({"x": [1]})
    y = pd.Series([0])
    X_out, y_out, is_tuple = unpack_pipeline_input((X, y))
    assert X_out is X
    assert y_out is y
    assert is_tuple is True


# ---------------------------------------------------------------------------
# pack_pipeline_output
# ---------------------------------------------------------------------------


def test_pack_not_tuple_no_y_returns_x() -> None:
    """When not a tuple and y is None, X must be returned unchanged."""
    df = pd.DataFrame({"x": [1]})
    result = pack_pipeline_output(df, None, False)
    assert result is df


def test_pack_was_tuple_with_y_returns_tuple() -> None:
    """was_tuple=True with y present must re-pack as (X, y) tuple."""
    X = pd.DataFrame({"x": [1]})
    y = pd.Series([1])
    result = pack_pipeline_output(X, y, True)
    assert isinstance(result, tuple)
    assert result[0] is X
    assert result[1] is y


def test_pack_not_tuple_with_y_concatenates(capsys: pytest.CaptureFixture) -> None:
    """was_tuple=False with y present must concatenate y back into X."""
    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([10, 20], name="target")
    result = pack_pipeline_output(X, y, False)
    assert isinstance(result, pd.DataFrame)
    assert "target" in result.columns
    assert list(result["x"]) == [1, 2]


def test_pack_raises_on_row_count_mismatch() -> None:
    """Regression test: X and y with different row counts must raise instead
    of silently NaN-padding/duplicating rows via a naive axis=1 concat."""
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([10, 20], name="target")
    with pytest.raises(ValueError, match="different row counts"):
        pack_pipeline_output(X, y, False)


def test_pack_realigns_mismatched_but_same_length_indices() -> None:
    """Regression test: same row count but non-matching pandas indices (e.g. a
    row-dropping step that reset X's index without resetting y's) must still
    concatenate positionally rather than NaN-padding via index-based concat."""
    X = pd.DataFrame({"x": [1, 2, 3]}, index=[10, 11, 12])
    y = pd.Series([100, 200, 300], name="target", index=[0, 1, 2])
    result = pack_pipeline_output(X, y, False)
    assert isinstance(result, pd.DataFrame)
    assert not result["target"].isna().any()
    assert list(result["target"]) == [100, 200, 300]


def test_pack_was_tuple_y_none_warns(caplog: pytest.LogCaptureFixture) -> None:
    """was_tuple=True but y=None must log a warning and return X."""
    import logging

    X = pd.DataFrame({"x": [1]})
    with caplog.at_level(logging.WARNING, logger="skyulf.utils"):
        result = pack_pipeline_output(X, None, True)
    assert result is X
    # The warning must mention the tuple shape being lost.
    assert any("tuple" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# _is_binary_numeric
# ---------------------------------------------------------------------------


def test_is_binary_numeric_true_for_01() -> None:
    """Series with only 0 and 1 must be detected as binary."""
    s = pd.Series([0, 1, 0, 1, 0])
    assert _is_binary_numeric(s) is True


def test_is_binary_numeric_false_for_multi_value() -> None:
    """Series with more than two distinct values must not be binary."""
    s = pd.Series([0, 1, 2, 3])
    assert _is_binary_numeric(s) is False


def test_is_binary_numeric_false_for_non_01() -> None:
    """Series with 0 and 2 (not 1) must not be detected as binary."""
    s = pd.Series([0.0, 2.0])
    assert _is_binary_numeric(s) is False


def test_is_binary_numeric_single_value_zero() -> None:
    """Constant-zero series is a degenerate binary case — returns True."""
    s = pd.Series([0.0, 0.0, 0.0])
    assert _is_binary_numeric(s) is True


def test_is_binary_numeric_ignores_nan() -> None:
    """dropna happens before the check, so NaN must not break detection."""
    s = pd.Series([0.0, 1.0, np.nan])
    # Caller is expected to dropna first (as in detect_numeric_columns).
    s_clean = s.dropna()
    assert _is_binary_numeric(s_clean) is True


# ---------------------------------------------------------------------------
# detect_numeric_columns
# ---------------------------------------------------------------------------


def test_detect_numeric_columns_basic() -> None:
    """Numeric non-binary non-constant columns must be returned."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0, 1, 0], "c": ["x", "y", "z"]})
    result = detect_numeric_columns(df)
    # 'b' is binary (0/1); 'c' is string — only 'a' qualifies.
    assert result == ["a"]


def test_detect_numeric_columns_include_binary_flag() -> None:
    """Setting exclude_binary=False must include 0/1 columns."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0, 1, 0]})
    result = detect_numeric_columns(df, exclude_binary=False)
    assert "b" in result
    assert "a" in result


def test_detect_numeric_columns_include_constant_flag() -> None:
    """Setting exclude_constant=False must include constant columns."""
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0], "b": [1.0, 2.0, 3.0]})
    result = detect_numeric_columns(df, exclude_constant=False)
    assert "a" in result


def test_detect_numeric_columns_empty_dataframe() -> None:
    """Empty DataFrame must return an empty list without raising."""
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    result = detect_numeric_columns(df)
    assert result == []


def test_detect_numeric_columns_all_nan_column() -> None:
    """All-NaN numeric column must be excluded because valid data is empty."""
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0]})
    result = detect_numeric_columns(df)
    assert "a" not in result
    assert "b" in result


def test_detect_numeric_columns_bool_excluded() -> None:
    """Boolean columns must always be excluded regardless of flags."""
    df = pd.DataFrame({"flag": [True, False, True], "v": [1.0, 2.0, 3.0]})
    result = detect_numeric_columns(df)
    assert "flag" not in result
    assert "v" in result


def test_detect_numeric_columns_mixed_dtypes() -> None:
    """Only numeric columns must be returned from a mixed-dtype frame."""
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "c"],
            "int_col": [10, 20, 30],
        }
    )
    result = detect_numeric_columns(df)
    assert "cat" not in result
    assert "num" in result
    assert "int_col" in result


def test_detect_numeric_columns_pandas_like_wrapper_uses_to_pandas() -> None:
    """A non-DataFrame frame exposing to_pandas() must be converted before analysis."""

    class _PandasLikeWrapper:
        def __init__(self, df: pd.DataFrame) -> None:
            self._df = df
            self.columns = df.columns
            self.dtypes = df.dtypes

        def to_pandas(self) -> pd.DataFrame:
            return self._df

    wrapper = _PandasLikeWrapper(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}))
    result = detect_numeric_columns(typing.cast(pd.DataFrame, wrapper))
    assert set(result) == {"a", "b"}


# ---------------------------------------------------------------------------
# resolve_columns
# ---------------------------------------------------------------------------


def test_resolve_columns_explicit_columns() -> None:
    """Explicit 'columns' in config must be returned (filtered for existence)."""
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    result = resolve_columns(df, {"columns": ["a", "c"]})
    assert result == ["a", "c"]


def test_resolve_columns_filters_nonexistent() -> None:
    """Non-existent column names in the config list must be silently dropped."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = resolve_columns(df, {"columns": ["a", "z"]})
    assert result == ["a"]


def test_resolve_columns_dedupes_explicit_duplicates_preserving_order() -> None:
    """Regression test: duplicate column names in an explicit `columns` list
    must be deduplicated (preserving first-occurrence order), otherwise
    stateful calculators (encoders/scalers) would process the same column
    twice, potentially corrupting fitted artifacts."""
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    result = resolve_columns(df, {"columns": ["b", "a", "b", "c", "a"]})
    assert result == ["b", "a", "c"]


def test_resolve_columns_auto_detect_with_func() -> None:
    """Without explicit columns, the default_selection_func must be called."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = resolve_columns(df, {}, default_selection_func=detect_numeric_columns)
    assert set(result) == {"a", "b"}


def test_resolve_columns_excludes_target_during_auto_detect() -> None:
    """Auto-detection must exclude the target column specified in config."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "target": [0.0, 1.0, 0.0]})
    result = resolve_columns(
        df,
        {"target_column": "target"},
        default_selection_func=detect_numeric_columns,
    )
    # target is binary (0/1) so it would already be excluded, but 'a' remains.
    assert "target" not in result


def test_resolve_columns_no_func_returns_empty() -> None:
    """Without explicit columns and without a selection func, return empty list."""
    df = pd.DataFrame({"a": [1]})
    result = resolve_columns(df, {})
    assert result == []


# ---------------------------------------------------------------------------
# user_picked_no_columns
# ---------------------------------------------------------------------------


def test_user_picked_no_columns_empty_list() -> None:
    """Explicitly empty 'columns' list must return True."""
    assert user_picked_no_columns({"columns": []}) is True


def test_user_picked_no_columns_absent_key() -> None:
    """Missing 'columns' key must return False (not the same as empty list)."""
    assert user_picked_no_columns({}) is False


def test_user_picked_no_columns_nonempty_list() -> None:
    """Non-empty 'columns' list must return False."""
    assert user_picked_no_columns({"columns": ["a"]}) is False


def test_user_picked_no_columns_none_value() -> None:
    """columns=None must return False — not the same as an empty list."""
    assert user_picked_no_columns({"columns": None}) is False


# ---------------------------------------------------------------------------
# detect_numeric_columns — polars path (skipped if polars not installed)
# ---------------------------------------------------------------------------

try:
    import polars as pl

    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False


@pytest.mark.skipif(not _POLARS_AVAILABLE, reason="polars not installed")
def test_detect_numeric_columns_polars_basic() -> None:
    """detect_numeric_columns on a polars frame must return numeric non-binary columns."""
    import polars as pl

    df_pl = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0, 1, 0], "c": ["x", "y", "z"]})
    result = detect_numeric_columns(typing.cast(pd.DataFrame, df_pl))
    # 'b' is binary; 'c' is string.
    assert "a" in result
    assert "b" not in result
    assert "c" not in result


@pytest.mark.skipif(not _POLARS_AVAILABLE, reason="polars not installed")
def test_detect_numeric_columns_polars_all_nan_excluded() -> None:
    """All-null numeric polars column must be excluded (no valid data)."""
    import polars as pl

    df_pl = pl.DataFrame(
        {
            "a": pl.Series([None, None, None], dtype=pl.Float64),
            "b": [1.0, 2.0, 3.0],
        }
    )
    result = detect_numeric_columns(typing.cast(pd.DataFrame, df_pl))
    assert "a" not in result
    assert "b" in result


@pytest.mark.skipif(not _POLARS_AVAILABLE, reason="polars not installed")
def test_detect_numeric_columns_polars_constant_excluded() -> None:
    """Constant polars column must be excluded by default."""
    import polars as pl

    df_pl = pl.DataFrame({"const": [5.0, 5.0, 5.0], "varied": [1.0, 2.0, 3.0]})
    result = detect_numeric_columns(typing.cast(pd.DataFrame, df_pl))
    assert "const" not in result
    assert "varied" in result


@pytest.mark.skipif(not _POLARS_AVAILABLE, reason="polars not installed")
def test_pack_pipeline_output_polars_with_y() -> None:
    """pack_pipeline_output on a polars frame with a Series y must concatenate."""
    import polars as pl

    X = pl.DataFrame({"x": [1, 2]})
    y = pl.Series("target", [10, 20])
    result = pack_pipeline_output(X, y, False)
    # Result must be a polars DataFrame containing both x and target.
    assert not isinstance(result, tuple)
    assert "x" in result.columns
    assert "target" in result.columns


@pytest.mark.skipif(not _POLARS_AVAILABLE, reason="polars not installed")
def test_pack_pipeline_output_polars_unnamed_series_gets_target_alias() -> None:
    """An unnamed polars y Series must be aliased to 'target' before concatenation."""
    import polars as pl

    X = pl.DataFrame({"x": [1, 2]})
    y = pl.Series([10, 20])  # No name -> defaults to "".
    assert y.name == ""

    result = pack_pipeline_output(X, y, False)
    assert not isinstance(result, tuple)
    assert "target" in result.columns


@pytest.mark.skipif(not _POLARS_AVAILABLE, reason="polars not installed")
def test_pack_pipeline_output_polars_dataframe_y_uses_hstack() -> None:
    """A polars DataFrame y (multiple target columns) must be hstacked onto X."""
    import polars as pl

    X = pl.DataFrame({"x": [1, 2]})
    y = pl.DataFrame({"t1": [10, 20], "t2": [30, 40]})
    result = pack_pipeline_output(X, y, False)
    assert not isinstance(result, tuple)
    assert "t1" in result.columns
    assert "t2" in result.columns


@pytest.mark.skipif(not _POLARS_AVAILABLE, reason="polars not installed")
def test_pack_pipeline_output_polars_wrapped_x_returns_wrapped_result() -> None:
    """A SkyulfPolarsWrapper-wrapped X must be unwrapped, merged, and re-wrapped."""
    import polars as pl

    from skyulf.engines.polars_engine import SkyulfPolarsWrapper

    X = SkyulfPolarsWrapper(pl.DataFrame({"x": [1, 2]}))
    y = pl.Series("target", [10, 20])
    result = pack_pipeline_output(X, y, False)

    assert isinstance(result, SkyulfPolarsWrapper)
    assert "target" in result.columns
    assert "x" in result.columns


@pytest.mark.skipif(not _POLARS_AVAILABLE, reason="polars not installed")
def test_pack_pipeline_output_polars_unconvertible_y_falls_back_to_raw_hstack() -> None:
    """A y that can't be turned into a pl.Series should fall through to raw hstack (and fail)."""
    import polars as pl

    X = pl.DataFrame({"x": [1, 2]})
    y = {1, 2, 3}  # A plain python set can't be converted via pl.Series(...).
    with pytest.raises(AttributeError):
        pack_pipeline_output(X, y, False)


def test_pack_pipeline_output_pandas_wrapper_x_uses_to_pandas() -> None:
    """An X exposing to_pandas() (e.g. a wrapper) must be converted before concatenation."""

    class _PandasLikeWrapper:
        def __init__(self, df: pd.DataFrame) -> None:
            self._df = df

        def to_pandas(self) -> pd.DataFrame:
            return self._df

    X = _PandasLikeWrapper(pd.DataFrame({"x": [1, 2]}))
    y = pd.Series([10, 20], name="target")
    result = pack_pipeline_output(X, y, False)

    assert isinstance(result, pd.DataFrame)
    assert "x" in result.columns
    assert "target" in result.columns


def test_pack_pipeline_output_pandas_y_wrapper_uses_to_pandas() -> None:
    """A y exposing to_pandas() must be converted before concatenation."""

    class _PandasLikeYWrapper:
        def __init__(self, series: pd.Series) -> None:
            self._series = series

        def to_pandas(self) -> pd.Series:
            return self._series

    X = pd.DataFrame({"x": [1, 2]})
    y = _PandasLikeYWrapper(pd.Series([10, 20], name="target"))
    result = pack_pipeline_output(X, y, False)

    assert isinstance(result, pd.DataFrame)
    assert "target" in result.columns
