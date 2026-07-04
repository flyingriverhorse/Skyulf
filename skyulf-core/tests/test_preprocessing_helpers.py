"""Unit tests for the shared preprocessing helpers in `_helpers.py`.

Covers every public function: resolve_valid_columns, safe_scale, to_pandas,
is_polars, auto_detect_text_columns, auto_detect_numeric_columns, and
auto_detect_datetime_columns, for both pandas and polars frames.
"""

import typing
from datetime import date, datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing._helpers import (
    auto_detect_datetime_columns,
    auto_detect_numeric_columns,
    auto_detect_text_columns,
    is_polars,
    resolve_valid_columns,
    safe_scale,
    to_pandas,
)

# ---------------------------------------------------------------------------
# resolve_valid_columns
# ---------------------------------------------------------------------------


def test_resolve_valid_columns_filters_missing() -> None:
    """Requested columns not present on the frame must be dropped."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = resolve_valid_columns(df, ["a", "c", "b"])
    assert result == ["a", "b"]


def test_resolve_valid_columns_preserves_requested_order() -> None:
    """Order of the requested list must be preserved, not the frame's column order."""
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    result = resolve_valid_columns(df, ["c", "a"])
    assert result == ["c", "a"]


def test_resolve_valid_columns_empty_requested_returns_empty() -> None:
    """An empty requested list yields an empty result."""
    df = pd.DataFrame({"a": [1]})
    assert resolve_valid_columns(df, []) == []


def test_resolve_valid_columns_none_present_returns_empty() -> None:
    """If none of the requested columns exist, the result is empty."""
    df = pd.DataFrame({"a": [1]})
    assert resolve_valid_columns(df, ["x", "y"]) == []


def test_resolve_valid_columns_works_with_polars_frame() -> None:
    """resolve_valid_columns must also work against a Polars DataFrame."""
    df = pl.DataFrame({"a": [1], "b": [2]})
    result = resolve_valid_columns(df, ["b", "z"])
    assert result == ["b"]


# ---------------------------------------------------------------------------
# safe_scale
# ---------------------------------------------------------------------------


def test_safe_scale_replaces_zeros_with_one() -> None:
    """Zero entries in the scale array must become 1.0."""
    arr = np.array([0.0, 2.0, 0.0, 5.0])
    result = safe_scale(arr)
    assert list(result) == [1.0, 2.0, 1.0, 5.0]


def test_safe_scale_mutates_input_array() -> None:
    """safe_scale mutates and returns the same array object (no copy)."""
    arr = np.array([0.0, 3.0])
    result = safe_scale(arr)
    assert result is arr
    assert arr[0] == 1.0


def test_safe_scale_no_zeros_is_noop() -> None:
    """An array with no zeros must be returned unchanged."""
    arr = np.array([2.0, 4.0, 8.0])
    result = safe_scale(arr)
    assert list(result) == [2.0, 4.0, 8.0]


# ---------------------------------------------------------------------------
# to_pandas
# ---------------------------------------------------------------------------


def test_to_pandas_passes_through_pandas_frame() -> None:
    """A pandas DataFrame must be returned unchanged (same object)."""
    df = pd.DataFrame({"a": [1, 2]})
    result = to_pandas(df)
    assert result is df


def test_to_pandas_converts_polars_frame() -> None:
    """A Polars DataFrame must be converted to a pandas DataFrame with equal values."""
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    result = to_pandas(df)
    assert isinstance(result, pd.DataFrame)
    assert result["a"].tolist() == [1, 2]
    assert result["b"].tolist() == ["x", "y"]


# ---------------------------------------------------------------------------
# is_polars
# ---------------------------------------------------------------------------


def test_is_polars_true_for_polars_frame() -> None:
    """is_polars must be True for a Polars DataFrame."""
    df = pl.DataFrame({"a": [1]})
    assert is_polars(df) is True


def test_is_polars_false_for_pandas_frame() -> None:
    """is_polars must be False for a pandas DataFrame."""
    df = pd.DataFrame({"a": [1]})
    assert is_polars(df) is False


# ---------------------------------------------------------------------------
# auto_detect_text_columns
# ---------------------------------------------------------------------------


def test_auto_detect_text_columns_pandas_object_dtype() -> None:
    """Object-dtype pandas columns must be detected as text."""
    df = pd.DataFrame({"name": ["a", "b"], "amount": [1.0, 2.0]})
    assert auto_detect_text_columns(df) == ["name"]


def test_auto_detect_text_columns_pandas_category_dtype() -> None:
    """Category-dtype pandas columns must be detected as text."""
    df = pd.DataFrame({"cat": pd.Categorical(["x", "y"]), "num": [1, 2]})
    assert auto_detect_text_columns(df) == ["cat"]


def test_auto_detect_text_columns_pandas_none_present() -> None:
    """A purely numeric pandas frame yields no text columns."""
    df = pd.DataFrame({"num": [1, 2, 3]})
    assert auto_detect_text_columns(df) == []


def test_auto_detect_text_columns_polars_utf8() -> None:
    """Polars Utf8 columns must be detected as text."""
    df = pl.DataFrame({"name": ["a", "b"], "amount": [1.0, 2.0]})
    assert auto_detect_text_columns(typing.cast(pd.DataFrame, df)) == ["name"]


# ---------------------------------------------------------------------------
# auto_detect_numeric_columns
# ---------------------------------------------------------------------------


def test_auto_detect_numeric_columns_pandas() -> None:
    """Numeric pandas columns (int/float) must be detected; object columns excluded."""
    df = pd.DataFrame({"num": [1, 2], "flt": [1.5, 2.5], "txt": ["a", "b"]})
    result = auto_detect_numeric_columns(df)
    assert set(result) == {"num", "flt"}


def test_auto_detect_numeric_columns_polars() -> None:
    """Numeric Polars columns must be detected across int/float dtypes."""
    df = pl.DataFrame({"num": [1, 2], "flt": [1.5, 2.5], "txt": ["a", "b"]})
    result = auto_detect_numeric_columns(typing.cast(pd.DataFrame, df))
    assert set(result) == {"num", "flt"}


def test_auto_detect_numeric_columns_pandas_none_present() -> None:
    """A purely text pandas frame yields no numeric columns."""
    df = pd.DataFrame({"txt": ["a", "b"]})
    assert auto_detect_numeric_columns(df) == []


# ---------------------------------------------------------------------------
# auto_detect_datetime_columns
# ---------------------------------------------------------------------------


def test_auto_detect_datetime_columns_pandas() -> None:
    """Datetime64 pandas columns must be detected; other dtypes excluded."""
    df = pd.DataFrame(
        {
            "when": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "num": [1, 2],
        }
    )
    assert auto_detect_datetime_columns(df) == ["when"]


def test_auto_detect_datetime_columns_pandas_none_present() -> None:
    """A frame with no datetime columns yields an empty list."""
    df = pd.DataFrame({"num": [1, 2]})
    assert auto_detect_datetime_columns(df) == []


def test_auto_detect_datetime_columns_polars_date_and_datetime() -> None:
    """Polars Date and Datetime columns must both be detected."""
    df = pl.DataFrame(
        {
            "d": [date(2020, 1, 1), date(2020, 1, 2)],
            "dt": [datetime(2020, 1, 1, 1), datetime(2020, 1, 2, 2)],
            "num": [1, 2],
        }
    )
    result = auto_detect_datetime_columns(typing.cast(pd.DataFrame, df))
    assert set(result) == {"d", "dt"}
