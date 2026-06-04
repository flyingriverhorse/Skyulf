"""Tests for the lightweight data-validation expectations (`profiling.expect`)."""

import pandas as pd
import pytest

from skyulf.profiling.expect import (
    ExpectationError,
    expect_columns_exist,
    expect_no_nulls,
    expect_unique,
    expect_value_range,
)


def _df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0], "c": ["x", "y", "z"]})


def test_columns_exist_passes_and_fails():
    expect_columns_exist(_df(), ["a", "b"])
    with pytest.raises(ExpectationError, match="missing"):
        expect_columns_exist(_df(), ["a", "missing"])


def test_no_nulls_passes_on_clean_frame():
    expect_no_nulls(_df())


def test_no_nulls_reports_offending_columns():
    df = _df()
    df.loc[0, "a"] = None
    with pytest.raises(ExpectationError, match="Null values found"):
        expect_no_nulls(df)
    # Restricting to a clean column passes.
    expect_no_nulls(df, ["b", "c"])


def test_value_range_inclusive_bounds():
    expect_value_range(_df(), "a", minimum=1, maximum=3)
    with pytest.raises(ExpectationError):
        expect_value_range(_df(), "a", minimum=2)
    with pytest.raises(ExpectationError):
        expect_value_range(_df(), "a", maximum=2)


def test_value_range_strict_bounds():
    with pytest.raises(ExpectationError):
        expect_value_range(_df(), "a", minimum=1, inclusive=False)
    expect_value_range(_df(), "a", minimum=0, inclusive=False)


def test_value_range_ignores_nulls():
    df = _df()
    df.loc[0, "b"] = None
    expect_value_range(df, "b", minimum=10, maximum=30)


def test_unique_detects_duplicates():
    df = pd.DataFrame({"id": [1, 1, 2]})
    with pytest.raises(ExpectationError, match="duplicate"):
        expect_unique(df, ["id"])
    expect_unique(pd.DataFrame({"id": [1, 2, 3]}), ["id"])


def test_polars_frame_is_supported():
    pl = pytest.importorskip("polars")
    pdf = pl.DataFrame({"a": [1, 2, 3]})
    expect_no_nulls(pdf)
    with pytest.raises(ExpectationError):
        expect_value_range(pdf, "a", maximum=2)
