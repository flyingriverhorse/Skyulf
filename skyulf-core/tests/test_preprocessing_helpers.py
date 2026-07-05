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
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing._helpers import (
    auto_detect_datetime_columns,
    auto_detect_numeric_columns,
    auto_detect_text_columns,
    is_polars,
    resolve_valid_columns,
    safe_scale,
    to_pandas,
)

_resolve_valid_columns_cases = TestCaseLoader("preprocessing/helpers_resolve_valid_columns").load()
_safe_scale_cases = TestCaseLoader("preprocessing/helpers_safe_scale").load()
_is_polars_cases = TestCaseLoader("preprocessing/helpers_is_polars").load()
_auto_detect_text_columns_cases = TestCaseLoader(
    "preprocessing/helpers_auto_detect_text_columns"
).load()
_auto_detect_numeric_columns_cases = TestCaseLoader(
    "preprocessing/helpers_auto_detect_numeric_columns"
).load()
_auto_detect_datetime_columns_cases = TestCaseLoader(
    "preprocessing/helpers_auto_detect_datetime_columns"
).load()


def _build_frame(frame_type: str, columns: dict) -> pd.DataFrame | pl.DataFrame:
    """Build a pandas or Polars DataFrame from a column-name -> values mapping."""
    if frame_type == "polars":
        return pl.DataFrame(columns)
    return pd.DataFrame(columns)


def _build_datetime_frame(
    frame_type: str, datetime_columns: dict, other_columns: dict
) -> pd.DataFrame | pl.DataFrame:
    """Build a frame with ISO-string datetime columns parsed per engine."""
    if frame_type == "polars":
        data = dict(other_columns)
        for col, vals in datetime_columns.items():
            data[col] = [
                date.fromisoformat(v) if len(v) == 10 else datetime.fromisoformat(v) for v in vals
            ]
        return pl.DataFrame(data)
    df = pd.DataFrame(other_columns)
    for col, vals in datetime_columns.items():
        df[col] = pd.to_datetime(vals)
    return df


# ---------------------------------------------------------------------------
# resolve_valid_columns
# ---------------------------------------------------------------------------


class TestResolveValidColumns:
    """resolve_valid_columns — scenarios loaded from
    ``tests/test_cases/preprocessing/helpers_resolve_valid_columns.json``.
    """

    @pytest.mark.parametrize(*_resolve_valid_columns_cases)
    def test_resolve_valid_columns(
        self, frame_type: str, frame_columns: dict, requested: list, expected: list
    ) -> None:
        df = _build_frame(frame_type, frame_columns)
        result = resolve_valid_columns(df, requested)
        assert result == expected


# ---------------------------------------------------------------------------
# safe_scale
# ---------------------------------------------------------------------------


class TestSafeScale:
    """safe_scale — scenarios loaded from
    ``tests/test_cases/preprocessing/helpers_safe_scale.json``.
    """

    @pytest.mark.parametrize(*_safe_scale_cases)
    def test_safe_scale(self, input_values: list, expected: list) -> None:
        arr = np.array(input_values)
        result = safe_scale(arr)
        assert list(result) == expected
        # safe_scale must mutate and return the same array object (no copy).
        assert result is arr


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


class TestIsPolars:
    """is_polars — scenarios loaded from
    ``tests/test_cases/preprocessing/helpers_is_polars.json``.
    """

    @pytest.mark.parametrize(*_is_polars_cases)
    def test_is_polars(self, frame_type: str, expected: bool) -> None:
        df = _build_frame(frame_type, {"a": [1]})
        assert is_polars(df) is expected


# ---------------------------------------------------------------------------
# auto_detect_text_columns
# ---------------------------------------------------------------------------


class TestAutoDetectTextColumns:
    """auto_detect_text_columns — scenarios loaded from
    ``tests/test_cases/preprocessing/helpers_auto_detect_text_columns.json``.
    """

    @pytest.mark.parametrize(*_auto_detect_text_columns_cases)
    def test_auto_detect_text_columns(
        self, frame_type: str, columns: dict, category_columns: list, expected: list
    ) -> None:
        if frame_type == "polars":
            df = pl.DataFrame(columns)
            result = auto_detect_text_columns(typing.cast(pd.DataFrame, df))
        else:
            df = pd.DataFrame(columns)
            for col in category_columns:
                df[col] = pd.Categorical(df[col])
            result = auto_detect_text_columns(df)
        assert result == expected


# ---------------------------------------------------------------------------
# auto_detect_numeric_columns
# ---------------------------------------------------------------------------


class TestAutoDetectNumericColumns:
    """auto_detect_numeric_columns — scenarios loaded from
    ``tests/test_cases/preprocessing/helpers_auto_detect_numeric_columns.json``.
    """

    @pytest.mark.parametrize(*_auto_detect_numeric_columns_cases)
    def test_auto_detect_numeric_columns(
        self, frame_type: str, columns: dict, expected: list
    ) -> None:
        df = _build_frame(frame_type, columns)
        result = auto_detect_numeric_columns(typing.cast(pd.DataFrame, df))
        assert set(result) == set(expected)


# ---------------------------------------------------------------------------
# auto_detect_datetime_columns
# ---------------------------------------------------------------------------


class TestAutoDetectDatetimeColumns:
    """auto_detect_datetime_columns — scenarios loaded from
    ``tests/test_cases/preprocessing/helpers_auto_detect_datetime_columns.json``.
    """

    @pytest.mark.parametrize(*_auto_detect_datetime_columns_cases)
    def test_auto_detect_datetime_columns(
        self, frame_type: str, datetime_columns: dict, other_columns: dict, expected: list
    ) -> None:
        df = _build_datetime_frame(frame_type, datetime_columns, other_columns)
        result = auto_detect_datetime_columns(typing.cast(pd.DataFrame, df))
        assert set(result) == set(expected)
