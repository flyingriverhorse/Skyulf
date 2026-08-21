"""Tests for skyulf.preprocessing.time_series._common helper functions."""

import pandas as pd
import pytest
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.time_series._common import (
    coerce_aggregations,
    coerce_lags,
    filter_existing_columns,
    sort_pandas,
)

_resolve_columns_cases = TestCaseLoader(
    "preprocessing/time_series_common", group="resolve_columns"
).load()
_coerce_lags_cases = TestCaseLoader("preprocessing/time_series_common", group="coerce_lags").load()
_coerce_aggregations_cases = TestCaseLoader(
    "preprocessing/time_series_common", group="coerce_aggregations"
).load()
_sort_pandas_cases = TestCaseLoader("preprocessing/time_series_common", group="sort_pandas").load()


@pytest.mark.parametrize(*_resolve_columns_cases)
def test_resolve_columns(
    columns: list[str] | None, available: list[str], expected: list[str]
) -> None:
    """``resolve_columns`` keeps only available columns, preserving order.

    Loaded from ``tests/test_cases/preprocessing/time_series_common.json`` (group ``resolve_columns``).
    """
    assert filter_existing_columns(columns, available) == expected


@pytest.mark.parametrize(*_coerce_lags_cases)
def test_coerce_lags(lags_input: int | list[int] | None, expected: list[int]) -> None:
    """``coerce_lags`` normalises int/list/None input into a sorted, deduped, positive list.

    Loaded from ``tests/test_cases/preprocessing/time_series_common.json`` (group ``coerce_lags``).
    """
    assert coerce_lags(lags_input) == expected


@pytest.mark.parametrize(*_coerce_aggregations_cases)
def test_coerce_aggregations(
    aggregations_input: str | list[str] | None, expected: list[str]
) -> None:
    """``coerce_aggregations`` normalises string/list/None input, filtering unknown names.

    Loaded from ``tests/test_cases/preprocessing/time_series_common.json`` (group ``coerce_aggregations``).
    """
    assert coerce_aggregations(aggregations_input) == expected


@pytest.mark.parametrize(*_sort_pandas_cases)
def test_sort_pandas(df_data: dict, sort_by: str | None, expected_cols: dict) -> None:
    """``sort_pandas`` stable-sorts by ``sort_by`` when present, and is a no-op otherwise.

    Loaded from ``tests/test_cases/preprocessing/time_series_common.json`` (group ``sort_pandas``).
    """
    df = pd.DataFrame(df_data)
    result = sort_pandas(df, sort_by)
    for col, expected in expected_cols.items():
        assert list(result[col]) == expected


# ---------------------------------------------------------------------------
# Real-shaped dataset: customers.csv (string-typed ISO date column)
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Verify sort_pandas behavior on customers.csv's signup_date column — a
    real-shaped string-typed ISO date column that must sort correctly using
    lexicographic order (ISO dates sort the same as alphabetic order).
    """

    def test_sort_pandas_by_signup_date_produces_ascending_order(self) -> None:
        """sort_pandas on signup_date must return all rows in ISO-date ascending
        order without dropping any rows or raising."""
        df = load_sample_dataset("customers")
        sorted_df = sort_pandas(df, "signup_date")
        assert len(sorted_df) == len(df)
        dates = sorted_df["signup_date"].tolist()
        assert dates == sorted(dates)
