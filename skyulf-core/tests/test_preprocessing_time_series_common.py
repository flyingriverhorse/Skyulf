"""Tests for skyulf.preprocessing.time_series._common helper functions."""

import pandas as pd
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.preprocessing.time_series._common import (
    coerce_aggregations,
    coerce_lags,
    resolve_columns,
    sort_pandas,
)


def test_resolve_columns_filters_to_available():
    """Only columns present in `available` should be kept, preserving order."""
    result = resolve_columns(["a", "z", "b"], ["a", "b", "c"])
    assert result == ["a", "b"]


def test_resolve_columns_empty_input_returns_empty_list():
    """A falsy `columns` argument should short-circuit to an empty list."""
    assert resolve_columns(None, ["a", "b"]) == []
    assert resolve_columns([], ["a", "b"]) == []


def test_coerce_lags_accepts_single_int():
    """A bare int should be normalised into a single-element sorted list."""
    assert coerce_lags(3) == [3]


def test_coerce_lags_dedups_and_sorts():
    """Duplicate and unordered lag values should be deduplicated and sorted."""
    assert coerce_lags([3, 1, 3, 2]) == [1, 2, 3]


def test_coerce_lags_drops_non_positive_values():
    """Zero and negative lag values should be dropped."""
    assert coerce_lags([0, -1, 2]) == [2]


def test_coerce_lags_empty_or_none_returns_empty_list():
    """None/empty input should return an empty list, not raise."""
    assert coerce_lags(None) == []
    assert coerce_lags([]) == []


def test_coerce_aggregations_accepts_single_string():
    """A bare string should be normalised into a single-element list."""
    assert coerce_aggregations("mean") == ["mean"]


def test_coerce_aggregations_filters_unknown_names():
    """Only recognised rolling aggregation names should be kept, order preserved."""
    result = coerce_aggregations(["mean", "bogus", "sum"])
    assert result == ["mean", "sum"]


def test_coerce_aggregations_empty_or_none_returns_empty_list():
    """None/empty input should return an empty list, not raise."""
    assert coerce_aggregations(None) == []
    assert coerce_aggregations([]) == []


def test_sort_pandas_sorts_by_given_column():
    """sort_by column present in the frame should trigger a stable mergesort."""
    df = pd.DataFrame({"t": [3, 1, 2], "v": ["c", "a", "b"]})
    result = sort_pandas(df, "t")
    assert list(result["t"]) == [1, 2, 3]
    assert list(result["v"]) == ["a", "b", "c"]


def test_sort_pandas_passthrough_when_column_missing():
    """When sort_by is absent from the frame, the original frame should be returned unsorted."""
    df = pd.DataFrame({"t": [3, 1, 2]})
    result = sort_pandas(df, "nonexistent")
    assert list(result["t"]) == [3, 1, 2]


def test_sort_pandas_passthrough_when_sort_by_none():
    """A None sort_by should be a no-op passthrough."""
    df = pd.DataFrame({"t": [3, 1, 2]})
    result = sort_pandas(df, None)
    assert list(result["t"]) == [3, 1, 2]


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
