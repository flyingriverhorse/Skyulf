"""Empty datasets must not silently pass row-level data-quality checks."""

from collections.abc import Callable
from typing import Any

import pandas as pd
import polars as pl
import pytest

from skyulf import (
    ExpectationError,
    expect_columns_exist,
    expect_no_nulls,
    expect_unique,
    expect_value_range,
)
from skyulf.engines.pandas_engine import SkyulfPandasWrapper
from skyulf.engines.polars_engine import SkyulfPolarsWrapper


@pytest.fixture(params=["pandas", "pandas_wrapper", "polars", "polars_wrapper"])
def make_frame(request: pytest.FixtureRequest) -> Callable[[list[float | None]], Any]:
    """Exercise native inputs and the supported engine wrappers through public APIs."""

    def build(values: list[float | None]) -> Any:
        """Keep the numeric schema present even when there are no rows."""
        if request.param.startswith("pandas"):
            frame = pd.DataFrame({"value": pd.Series(values, dtype="float64")})
            return SkyulfPandasWrapper(frame) if request.param.endswith("wrapper") else frame
        frame = pl.DataFrame({"value": values}, schema={"value": pl.Float64})
        return SkyulfPolarsWrapper(frame) if request.param.endswith("wrapper") else frame

    return build


@pytest.mark.parametrize("check", [expect_no_nulls, expect_unique, expect_value_range])
def test_empty_rows_fail_by_default(make_frame: Callable, check: Callable) -> None:
    """A failed ingestion must not look like a successful data-quality check."""
    columns = "value" if check is expect_value_range else ["value"]
    with pytest.raises(ExpectationError, match="at least one row"):
        check(make_frame([]), columns)


@pytest.mark.parametrize("check", [expect_no_nulls, expect_unique, expect_value_range])
def test_empty_rows_can_be_allowed_explicitly(make_frame: Callable, check: Callable) -> None:
    """Callers can deliberately preserve checks that accept an empty partition."""
    columns = "value" if check is expect_value_range else ["value"]
    kwargs = {"minimum": 0, "maximum": 10} if check is expect_value_range else {}
    assert check(make_frame([]), columns, allow_empty=True, **kwargs) is None


@pytest.mark.parametrize("allow_empty", [False, True])
@pytest.mark.parametrize("check", [expect_no_nulls, expect_unique, expect_value_range])
def test_empty_frames_still_validate_requested_columns(
    make_frame: Callable, check: Callable, allow_empty: bool
) -> None:
    """The empty-row option must not hide a broken column contract."""
    columns = "missing" if check is expect_value_range else ["missing"]
    with pytest.raises(ExpectationError, match="Expected columns are missing"):
        check(make_frame([]), columns, allow_empty=allow_empty)


@pytest.mark.parametrize(
    ("check", "values", "columns", "kwargs", "error"),
    [
        (expect_no_nulls, [None], ["value"], {}, "Null values found"),
        (expect_unique, [1, 1], ["value"], {}, "duplicate rows"),
        (expect_value_range, [11], "value", {"maximum": 10}, "not <= 10"),
    ],
)
def test_allow_empty_does_not_disable_checks_on_existing_rows(
    make_frame: Callable,
    check: Callable,
    values: list[float | None],
    columns: str | list[str],
    kwargs: dict,
    error: str,
) -> None:
    """Opting into empty frames must retain the actual null, range and duplicate checks."""
    with pytest.raises(ExpectationError, match=error):
        check(make_frame(values), columns, allow_empty=True, **kwargs)


def test_null_only_range_is_not_an_empty_frame(make_frame: Callable) -> None:
    """Range checks must keep ignoring null values when the input has rows."""
    assert expect_value_range(make_frame([None]), "value", minimum=0, maximum=10) is None


def test_columns_exist_remains_a_schema_only_check(make_frame: Callable) -> None:
    """A schema check can validate a typed frame before any rows are available."""
    assert expect_columns_exist(make_frame([]), ["value"]) is None


@pytest.mark.parametrize("values", [[], [None]])
@pytest.mark.parametrize("wrapped", [False, True])
def test_range_with_no_observed_values_accepts_inferred_null_dtype(
    values: list[None], wrapped: bool
) -> None:
    """An empty or all-null Polars column must not invoke unsupported Null comparisons."""
    raw = pl.DataFrame({"value": values})
    frame = SkyulfPolarsWrapper(raw) if wrapped else raw
    assert expect_value_range(frame, "value", minimum=0, maximum=10, allow_empty=not values) is None


def test_no_nulls_checks_row_count_without_an_explicit_subset(make_frame: Callable) -> None:
    """The default all-columns path must enforce the same empty-row policy."""
    with pytest.raises(ExpectationError, match="at least one row"):
        expect_no_nulls(make_frame([]))


def test_no_nulls_does_not_confuse_no_columns_with_no_rows() -> None:
    """A zero-width Pandas frame can still contain rows and must use its row count."""
    assert expect_no_nulls(pd.DataFrame(index=[0, 1])) is None


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("allow_empty", [False, True])
def test_empty_polars_checks_stay_native(
    monkeypatch: pytest.MonkeyPatch, wrapped: bool, allow_empty: bool
) -> None:
    """An empty-data guard must not introduce an expensive Polars conversion."""
    import skyulf.profiling.expect as expectation_module

    def fail_to_pandas(*args: object, **kwargs: object) -> None:
        """Expose any regression that converts native numeric checks to Pandas."""
        raise AssertionError("unexpected pandas conversion")

    monkeypatch.setattr(expectation_module, "_as_pandas", fail_to_pandas)
    raw = pl.DataFrame(schema={"value": pl.Float64})
    frame = SkyulfPolarsWrapper(raw) if wrapped else raw
    checks: list[Callable[..., None]] = [expect_no_nulls, expect_unique, expect_value_range]
    for check in checks:
        columns = "value" if check is expect_value_range else ["value"]
        if allow_empty:
            assert check(frame, columns, allow_empty=True) is None
        else:
            with pytest.raises(ExpectationError, match="at least one row"):
                check(frame, columns)
