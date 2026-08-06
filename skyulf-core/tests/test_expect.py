"""Tests for the lightweight data-validation expectations (`profiling.expect`)."""

import pandas as pd
import pytest

from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.profiling.expect import (
    ExpectationError,
    expect_columns_exist,
    expect_no_nulls,
    expect_unique,
    expect_value_range,
)


def _df():
    """Build a simple Pandas frame for expectation tests."""
    return pd.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0], "c": ["x", "y", "z"]})


def _polars_variants(data: dict[str, object]) -> list[object]:
    """Return raw and wrapped Polars frames with equivalent contents."""
    pl = pytest.importorskip("polars")
    raw = pl.DataFrame(data)
    return [raw, SkyulfPolarsWrapper(raw)]


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


def test_polars_expectations_match_pandas_null_nan_and_range_messages() -> None:
    """Raw and wrapped Polars frames preserve Pandas expectation semantics."""
    pandas_frame = pd.DataFrame({"value": [1.0, float("nan"), None, 3.0]})
    with pytest.raises(ExpectationError) as pandas_null_error:
        expect_no_nulls(pandas_frame)
    with pytest.raises(ExpectationError) as pandas_range_error:
        expect_value_range(pandas_frame, "value", maximum=2)

    for frame in _polars_variants({"value": [1.0, float("nan"), None, 3.0]}):
        with pytest.raises(ExpectationError) as polars_null_error:
            expect_no_nulls(frame)
        with pytest.raises(ExpectationError) as polars_range_error:
            expect_value_range(frame, "value", maximum=2)
        assert str(polars_null_error.value) == str(pandas_null_error.value)
        assert str(polars_range_error.value) == str(pandas_range_error.value)


def test_polars_expect_unique_matches_pandas_for_raw_and_wrapped_frames() -> None:
    """Duplicate-row counts and messages match the Pandas path."""
    pandas_frame = pd.DataFrame({"left": [1, 1, 2], "right": ["a", "a", "b"]})
    with pytest.raises(ExpectationError) as pandas_error:
        expect_unique(pandas_frame, ["left", "right"])

    for frame in _polars_variants({"left": [1, 1, 2], "right": ["a", "a", "b"]}):
        with pytest.raises(ExpectationError) as polars_error:
            expect_unique(frame, ["left", "right"])
        assert str(polars_error.value) == str(pandas_error.value)


def test_polars_columns_and_strict_bounds_match_pandas_messages() -> None:
    """Missing-column and exclusive-bound failures stay byte-for-byte compatible."""
    pandas_frame = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    with pytest.raises(ExpectationError) as pandas_columns_error:
        expect_columns_exist(pandas_frame, ["missing"])
    with pytest.raises(ExpectationError) as pandas_bound_error:
        expect_value_range(pandas_frame, "value", minimum=1, inclusive=False)

    for frame in _polars_variants({"value": [1.0, 2.0, 3.0]}):
        with pytest.raises(ExpectationError) as polars_columns_error:
            expect_columns_exist(frame, ["missing"])
        with pytest.raises(ExpectationError) as polars_bound_error:
            expect_value_range(frame, "value", minimum=1, inclusive=False)
        assert str(polars_columns_error.value) == str(pandas_columns_error.value)
        assert str(polars_bound_error.value) == str(pandas_bound_error.value)


def test_polars_expectations_do_not_convert_to_pandas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native Polars expectation paths must not route through to_pandas()."""
    import skyulf.profiling.expect as expectation_module

    def fail_to_pandas(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unexpected pandas conversion")

    monkeypatch.setattr(expectation_module, "_as_pandas", fail_to_pandas)
    for frame in _polars_variants({"value": [1.0, 2.0, 3.0]}):
        expect_columns_exist(frame, ["value"])
        expect_no_nulls(frame)
        expect_value_range(frame, "value", minimum=1, maximum=3)
        expect_unique(frame, ["value"])
