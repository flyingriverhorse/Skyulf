"""Winsorization must preserve exact integer features or require explicit casting."""

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal as assert_polars_frame_equal

from skyulf.core.schema import SkyulfSchema
from skyulf.engines import SkyulfPolarsWrapper
from skyulf.preprocessing.outliers.winsorize import WinsorizeApplier, WinsorizeCalculator


def _assert_equal(actual, expected):
    """Compare native values and schema without a numerical tolerance."""
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    else:
        assert_polars_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("dtype", ["int64", "Int64", "uint64", "UInt64"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_integral_bounds_preserve_large_integer_values_and_empty_schema(engine, dtype):
    """Clipping one row must never round an untouched integer or lose nullable storage."""
    values = [0, 2**53 + 1, 2**53 + 3]
    if dtype[0].isupper():
        values.append(None)
    frame = pd.DataFrame({"x": pd.Series(values, dtype=dtype)})
    frame.index = [i // 2 for i in range(len(frame))]
    expected = frame.copy()
    expected.iloc[0, 0] = 1
    if engine == "polars":
        frame, expected = pl.from_pandas(frame), pl.from_pandas(expected)
    state = {"bounds": {"x": {"lower": 1.0, "upper": 2**53 + 3}}}
    before = pickle.dumps(state)
    applier = WinsorizeApplier()
    for size in (len(frame), 1, 0):
        request = frame.head(size)
        _assert_equal(applier.apply(request, state), expected.head(size))
    assert pickle.dumps(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("bound", [0.5, np.inf, -np.inf, np.nan, 300])
@pytest.mark.parametrize("empty", [False, True])
def test_integer_bounds_reject_inexact_or_unrepresentable_limits(engine, bound, empty):
    """Unsupported limits must reject before row matching, including empty requests."""
    frame = pd.DataFrame({"x": pd.Series([2, 3], dtype="Int8")})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    if empty:
        frame = frame.head(0)
    state = {"bounds": {"x": {"lower": bound, "upper": bound}}}
    with pytest.raises(ValueError, match="Casting"):
        WinsorizeApplier().apply(frame, state)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_integer_fit_keeps_exact_endpoint_quantiles(engine):
    """Native min/max endpoints must not pass large training integers through float64."""
    frame = pd.DataFrame({"x": pd.Series([2**53 + 1, 2**53 + 3], dtype="Int64")})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    config = {"columns": ["x"], "lower_percentile": 0, "upper_percentile": 100}
    calculator = WinsorizeCalculator()
    state = calculator.fit(frame, config)
    assert int(state["bounds"]["x"]["lower"]) == 2**53 + 1
    assert int(state["bounds"]["x"]["upper"]) == 2**53 + 3
    _assert_equal(WinsorizeApplier().apply(frame, state), frame)
    assert calculator.infer_output_schema(SkyulfSchema.from_dataframe(frame), config) == (
        SkyulfSchema.from_dataframe(frame)
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("values", [[0, 2, 4, 6], [2**53 + 1, 2**53 + 3]])
def test_integer_fit_rejects_fractional_or_unsafe_interpolated_quantiles(engine, values):
    """Fit must explain the explicit cast before unsafe percentile bounds are saved."""
    frame = pd.DataFrame({"x": values})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    with pytest.raises(ValueError, match="Casting"):
        WinsorizeCalculator().fit(
            frame, {"columns": ["x"], "lower_percentile": 25, "upper_percentile": 75}
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_explicit_float_input_keeps_fractional_winsorization(engine):
    """An explicit float feature must retain ordinary fractional quantiles and null values."""
    frame = pd.DataFrame({"x": [0.0, 2.0, 4.0, 6.0, np.nan]})
    expected = pd.DataFrame({"x": [1.5, 2.0, 4.0, 4.5, np.nan]})
    if engine == "polars":
        frame, expected = pl.from_pandas(frame), pl.from_pandas(expected)
    state = WinsorizeCalculator().fit(
        frame, {"columns": ["x"], "lower_percentile": 25, "upper_percentile": 75}
    )
    _assert_equal(WinsorizeApplier().apply(frame, state), expected)


@pytest.mark.parametrize("dtype,value", [(pl.Int64, 2**53 + 1), (pl.UInt64, 2**64 - 1)])
@pytest.mark.parametrize("wrapped", [False, True])
def test_polars_nullable_integer_fit_preserves_original_values(dtype, value, wrapped):
    """Selected nullable integers must not lose precision during pandas conversion for fitting."""
    frame = pl.DataFrame({"x": pl.Series([value, None], dtype=dtype)})
    request = SkyulfPolarsWrapper(frame) if wrapped else frame
    calculator = WinsorizeCalculator()
    state = calculator.fit(
        request, {"columns": ["x"], "lower_percentile": 0, "upper_percentile": 100}
    )
    assert int(state["bounds"]["x"]["lower"]) == value
    assert int(state["bounds"]["x"]["upper"]) == value
    with pytest.raises(ValueError, match="Casting"):
        calculator.fit(request, {"columns": ["x"], "lower_percentile": 25, "upper_percentile": 75})


@pytest.mark.parametrize("dtype", ["object", "string"])
def test_explicit_noninteger_columns_keep_native_fit_conversion(dtype):
    """Numeric parsing for an explicitly selected object/string column must retain its old rules."""
    frame = pd.DataFrame({"x": pd.Series(["0", "2", "4", "6"], dtype=dtype)})
    state = WinsorizeCalculator().fit(
        frame, {"columns": ["x"], "lower_percentile": 25, "upper_percentile": 75}
    )
    assert state["bounds"]["x"] == {"lower": 1.5, "upper": 4.5}


def test_arrow_uint64_clipping_retains_unsigned_bound_scalars():
    """Arrow must not box a valid UInt64 bound into its signed integer scalar path."""
    frame = pd.DataFrame({"x": pd.Series([0, 2**64 - 1, None], dtype="uint64[pyarrow]")})
    expected = pd.DataFrame({"x": pd.Series([1, 2**64 - 1, None], dtype="uint64[pyarrow]")})
    state = {"bounds": {"x": {"lower": 1, "upper": 2**64 - 1}}}
    applier = WinsorizeApplier()
    for size in (len(frame), 1, 0):
        _assert_equal(applier.apply(frame.head(size), state), expected.head(size))
