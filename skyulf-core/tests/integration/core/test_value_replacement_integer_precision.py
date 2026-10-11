"""Integer replacement rules require exact values and stable native null containers."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.cleaning.value_replacement import (
    ValueReplacementApplier,
    ValueReplacementCalculator,
)


def _frame(engine, values=(1, 2**53 + 1), dtype="int64"):
    """Keep large integers exact before invoking either native replacement implementation."""
    frame = pd.DataFrame({"x": pd.array(values, dtype=dtype), "target": range(len(values))})
    frame.index = [8] * len(frame)
    return pl.from_pandas(frame) if engine == "polars" else frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "rules",
    [
        {"mapping": {"1": 0.5}},
        {"mapping": {"x": {"1": 0.5}}},
        {"to_replace": "1", "value": 0.5},
        {"to_replace": ["1"], "value": 0.5},
        {"to_replace": ["1"], "value": [0.5]},
        {"to_replace": {"1": 0.5}},
        {"replacements": [{"old": "1", "new": 0.5}]},
    ],
)
def test_fractional_integer_rules_require_explicit_casting(engine, rules):
    """Invalid numeric rules must fail before matching rows, including empty and unmatched input."""
    frame = _frame(engine)
    state = ValueReplacementCalculator().fit(frame, {"columns": ["x"], **rules})
    for chunk in (frame, frame.tail(1), frame.head(0)):
        with pytest.raises(ValueError, match="Casting"):
            ValueReplacementApplier().apply(chunk, state)
    assert frame["x"].to_list() == [1, 2**53 + 1]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype,value", [("int8", 128), ("uint64", -1), ("int64", np.inf)])
def test_integer_replacement_rejects_out_of_range_and_infinite_values(engine, dtype, value):
    """Neither native integer wrapping nor implicit Float64 promotion may corrupt other rows."""
    frame = _frame(engine, (1, 2), dtype)
    with pytest.raises(ValueError, match="Casting"):
        ValueReplacementApplier().apply(frame, {"columns": ["x"], "mapping": {1: value}})
    assert frame["x"].to_list() == [1, 2]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "dtype,values,replacement,expected",
    [
        ("int64", [1, 2**53 + 1], 2.0, [2, 2**53 + 1]),
        ("uint64", [1, 2**64 - 1], 2, [2, 2**64 - 1]),
        ("int8", [1, 2], None, [None, 2]),
        ("int64", [1, 2**53 + 1], np.nan, [None, 2**53 + 1]),
        ("UInt64", [1, 2**64 - 1], pd.NA, [None, 2**64 - 1]),
    ],
)
def test_integer_replacements_preserve_exact_values_and_empty_schema(
    engine, dtype, values, replacement, expected
):
    """Integral floats and missing values must preserve width, sign, precision and row identity."""
    frame = _frame(engine, values, dtype)
    state = ValueReplacementCalculator().fit(frame, {"columns": ["x"], "mapping": {1: replacement}})
    output = ValueReplacementApplier().apply(frame, state)
    for actual, wanted in zip(output["x"].to_list(), expected, strict=True):
        assert pd.isna(actual) if wanted is None else actual == wanted
    for chunk in (frame.tail(1), frame.head(0)):
        observed = ValueReplacementApplier().apply(chunk, state)
        wanted = output.tail(1) if len(chunk) else output.head(0)
        if isinstance(frame, pl.DataFrame):
            assert observed.equals(wanted)
            assert observed.schema == wanted.schema
            assert output.schema["x"] == frame.schema["x"]
        else:
            pd.testing.assert_frame_equal(observed, wanted)
            assert output["x"].dtype.itemsize == frame["x"].dtype.itemsize
            assert output["x"].dtype.kind == frame["x"].dtype.kind
    assert frame["x"].to_list() == values
    assert output["target"].to_list() == [0, 1]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("key", [True, 1.5, "oops"])
@pytest.mark.parametrize("mapping", [False, True])
def test_inapplicable_integer_keys_do_not_activate_numeric_rules(engine, key, mapping):
    """Keys that cannot identify an integer observation cannot require a fractional cast."""
    frame = _frame(engine)
    rules = {"mapping": {key: 0.5}} if mapping else {"to_replace": [key], "value": [0.5]}
    output = ValueReplacementApplier().apply(frame, {"columns": ["x"], **rules})
    assert output.equals(frame)


def test_explicit_float_cast_allows_fractional_replacements():
    """A deliberate floating input remains the supported numeric feature path for fractions."""
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    output = ValueReplacementApplier().apply(frame, {"columns": ["x"], "mapping": {1: 0.5}})
    pd.testing.assert_frame_equal(output, pd.DataFrame({"x": [0.5, 2.0]}))


def test_pandas_object_replacements_keep_native_fractional_values():
    """The integer guard must not constrain object columns that already retain exact mixed values."""
    frame = pd.DataFrame({"x": pd.Series([1, 2**53 + 1], dtype=object)})
    output = ValueReplacementApplier().apply(frame, {"columns": ["x"], "mapping": {1: 0.5}})
    pd.testing.assert_series_equal(output["x"], pd.Series([0.5, 2**53 + 1], dtype=object, name="x"))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("replacement", [None, np.nan])
def test_scalar_integer_missing_replacement_keeps_exact_nullable_values(engine, replacement):
    """Scalar replacement must use an integer null instead of pandas's scalar None object path."""
    frame = _frame(engine, [1, None, 2**53 + 1], "Int64")
    output = ValueReplacementApplier().apply(
        frame, {"columns": ["x"], "to_replace": 1, "value": replacement}
    )
    values = output["x"].to_list()
    assert pd.isna(values[0]) and pd.isna(values[1])
    assert values[2] == 2**53 + 1
    assert str(output["x"].dtype) == "Int64"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("mapping", [False, True])
@pytest.mark.parametrize(
    "dtype,values,key,expected",
    [
        ("int64", [2**53, 2**53 + 1, 2**53 + 2], float(2**53), [0, 2**53 + 1, 2**53 + 2]),
        ("Int64", [2**53, 2**53 + 1, 2**53 + 2], float(2**53), [0, 2**53 + 1, 2**53 + 2]),
        ("uint64", [2**64 - 2, 2**64 - 1], float(2**64), [2**64 - 2, 2**64 - 1]),
        ("UInt64", [2**64 - 2, 2**64 - 1], float(2**64), [2**64 - 2, 2**64 - 1]),
    ],
)
def test_integer_rules_match_exact_keys_without_float_comparisons(
    engine, mapping, dtype, values, key, expected
):
    """Float-represented keys must not match an adjacent integer or wrap past the dtype limit."""
    frame = _frame(engine, values, dtype)
    rules = {"mapping": {key: 0}} if mapping else {"to_replace": [key], "value": [0]}
    output = ValueReplacementApplier().apply(frame, {"columns": ["x"], **rules})
    assert output["x"].to_list() == expected
    assert frame["x"].to_list() == values


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("nonnumeric", ["text", False])
def test_mixed_replacement_values_cannot_bypass_integer_fraction_guard(engine, nonnumeric):
    """An unrelated categorical rule must not enable silent numeric widening of integer rows."""
    frame = _frame(engine)
    with pytest.raises(ValueError, match="Casting"):
        ValueReplacementApplier().apply(
            frame, {"columns": ["x"], "mapping": {1: 0.5, 2: nonnumeric}}
        )
    assert frame["x"].to_list() == [1, 2**53 + 1]
