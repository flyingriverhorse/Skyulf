"""Invalid-value numeric rules and infinity cleanup must preserve exact integer observations."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from skyulf.engines import SkyulfPolarsWrapper
from skyulf.engines.pandas_engine import SkyulfPandasWrapper
from skyulf.preprocessing.cleaning.invalid_value import (
    InvalidValueReplacementApplier,
    InvalidValueReplacementCalculator,
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "dtype,replacement", [("int64", 0.5), ("uint64", -1), ("int8", 128), ("int64", np.inf)]
)
def test_integer_rules_require_representable_numeric_replacements(engine, dtype, replacement):
    """Configured numeric rules must reject unsafe casts even when the request is unmatched or empty."""
    frame = pd.DataFrame({"value": pd.Series([0, 2], dtype=dtype)})
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    state = InvalidValueReplacementCalculator().fit(
        frame, {"columns": ["value"], "rule": "zero", "replacement": replacement}
    )
    for chunk in (frame, frame.tail(1), frame.head(0)):
        with pytest.raises(ValueError, match="Casting"):
            InvalidValueReplacementApplier().apply(chunk, state)
    assert frame["value"].to_list() == [0, 2]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype,large", [("int64", 2**53 + 1), ("uint64", 2**64 - 1)])
def test_integral_float_replacements_preserve_exact_integer_outputs(engine, dtype, large):
    """An integral replacement must not cause native expression promotion to round another row."""
    frame = pd.DataFrame({"value": pd.Series([0, large], dtype=dtype)})
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    state = InvalidValueReplacementCalculator().fit(frame, {"rule": "zero", "replacement": 1.0})
    for chunk in (frame, frame.tail(1), frame.head(0)):
        result = InvalidValueReplacementApplier().apply(chunk, state)
        assert result["value"].dtype == chunk["value"].dtype
        assert result["value"].to_list() == (
            [1, large] if len(chunk) == 2 else chunk["value"].to_list()
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "kind,limit,expected",
    [
        ("max_value", float(2**53), [2**53, 0, 0]),
        ("min_value", float(2**53 + 2), [0, 0, 2**53 + 2]),
    ],
)
def test_integer_range_uses_exact_integral_threshold_comparisons(engine, kind, limit, expected):
    """Floating representations of integral bounds must not merge adjacent large integer rows."""
    values = [2**53, 2**53 + 1, 2**53 + 2]
    frame = pd.DataFrame({"value": values})
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    state = InvalidValueReplacementCalculator().fit(
        frame, {"rule": "custom_range", kind: limit, "replacement": 0}
    )
    result = InvalidValueReplacementApplier().apply(frame, state)
    assert result["value"].to_list() == expected
    assert frame["value"].to_list() == values


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("config", [{}, {"replace_inf": True}, {"rule": "custom_range"}])
def test_integer_noop_rules_do_not_validate_unused_fractional_replacements(engine, config):
    """An impossible infinity match or absent range must retain the original no-op contract."""
    frame = pd.DataFrame({"value": pd.Series([0, 2**64 - 1], dtype="UInt64")})
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    state = InvalidValueReplacementCalculator().fit(frame, {"replacement": 0.5, **config})
    for chunk in (frame, frame.head(0)):
        result = InvalidValueReplacementApplier().apply(chunk, state)
        assert result.equals(chunk)
        assert result["value"].dtype == chunk["value"].dtype


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("limit", [float(2**64), np.inf])
def test_integer_comparison_bounds_need_not_fit_column_dtype(engine, limit):
    """An upper bound beyond UInt64 is still a valid comparison that must preserve all rows."""
    frame = pd.DataFrame({"value": pd.Series([0, 2**64 - 1], dtype="UInt64")})
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    state = InvalidValueReplacementCalculator().fit(
        frame, {"rule": "custom_range", "max_value": limit, "replacement": 0}
    )
    result = InvalidValueReplacementApplier().apply(frame, state)
    assert result.equals(frame)
    assert result["value"].dtype == frame["value"].dtype


@pytest.mark.parametrize("dtype", ["Int64", "UInt64", "int64[pyarrow]", "uint64[pyarrow]"])
def test_integral_numeric_rules_preserve_pandas_extension_types(dtype):
    """Normalization must retain nullable and Arrow masks, widths and exact large observations."""
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"value": pd.Series([0, 2**53 + 1, None], dtype=dtype)})
    expected = frame.copy()
    expected.loc[0, "value"] = 1
    state = InvalidValueReplacementCalculator().fit(frame, {"rule": "zero", "replacement": 1.0})
    for chunk in (frame, frame.tail(2), frame.head(0)):
        result = InvalidValueReplacementApplier().apply(chunk, state)
        pd.testing.assert_frame_equal(result, expected.loc[chunk.index])
    assert int(frame["value"].to_list()[1]) == 2**53 + 1


@pytest.mark.parametrize(
    "dtype", ["int8[pyarrow]", "uint8[pyarrow]", "int64[pyarrow]", "uint64[pyarrow]"]
)
@pytest.mark.parametrize("kind", ["min_value", "max_value"])
@pytest.mark.parametrize("limit", [float(2**63), float(2**64), float(-(2**64))])
def test_arrow_integer_range_masks_keep_exact_out_of_range_comparisons(dtype, kind, limit):
    """Arrow scalar boxing must not reject bounds or round integers at the signed/unsigned limits."""
    pytest.importorskip("pyarrow")
    bounds = np.iinfo(np.dtype(dtype.split("[")[0]))
    values = [int(bounds.min), 0, int(bounds.max), None]
    frame = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})
    expected = pd.DataFrame(
        {
            "value": pd.Series(
                [
                    7
                    if value is not None
                    and (value < int(limit) if kind == "min_value" else value > int(limit))
                    else value
                    for value in values
                ],
                dtype=dtype,
            )
        }
    )
    state = InvalidValueReplacementCalculator().fit(
        frame, {"rule": "custom_range", kind: limit, "replacement": 7}
    )
    for chunk in (frame, frame.tail(2), frame.head(0)):
        result = InvalidValueReplacementApplier().apply(chunk, state)
        pd.testing.assert_frame_equal(result, expected.loc[chunk.index])
    assert frame["value"].iloc[2] == int(bounds.max)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("replacement", [True, "missing"])
def test_integer_rules_keep_nonnumeric_replacement_engine_semantics(engine, replacement):
    """Boolean and string replacements keep each engine's established type behavior."""
    frame = pd.DataFrame({"value": [0, 2**53 + 1]})
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    state = InvalidValueReplacementCalculator().fit(
        frame, {"rule": "zero", "replacement": replacement}
    )
    result = InvalidValueReplacementApplier().apply(frame, state)
    if engine == "pandas":
        assert result["value"].dtype == object
        assert result["value"].to_list() == [replacement, 2**53 + 1]
    elif replacement is True:
        assert result["value"].dtype == pl.Int64
        assert result["value"].to_list() == [1, 2**53 + 1]
    else:
        assert result["value"].dtype == pl.String
        assert result["value"].to_list() == ["missing", str(2**53 + 1)]


def _integer_frame(engine: str) -> Any:
    """Keep signed and unsigned observations beyond exact Float64 representation."""
    frame = pd.DataFrame(
        {
            "signed": pd.Series(
                [-9007199254740993, 9007199254740993, None, 9223372036854775807],
                dtype="Int64",
            ),
            "unsigned": pd.Series(
                [9007199254740993, 18446744073709551615, None, 0], dtype="UInt64"
            ),
        }
    )
    if engine == "pandas":
        return frame
    if engine == "wrapped_pandas":
        return SkyulfPandasWrapper(frame)
    native = pl.from_pandas(frame)
    return SkyulfPolarsWrapper(native) if engine == "wrapped_polars" else native


def _native(frame: Any) -> Any:
    """Expose frame values for engine-aware equality without altering numeric dtypes."""
    if isinstance(frame, (SkyulfPandasWrapper, SkyulfPolarsWrapper)):
        return frame.to_native()
    return frame


@pytest.mark.parametrize("engine", ["pandas", "wrapped_pandas", "polars", "wrapped_polars"])
@pytest.mark.parametrize(
    "flags",
    [
        {"replace_inf": True},
        {"replace_neg_inf": True},
        {"replace_inf": True, "replace_neg_inf": True},
    ],
)
def test_infinity_cleanup_preserves_exact_integer_values_and_dtypes(
    engine: str, flags: dict[str, bool]
) -> None:
    """An impossible infinity match must not round large integers or widen their dtype."""
    frame = _integer_frame(engine)
    target = np.array([0, 1, 0, 1])
    original = _native(_integer_frame(engine))
    artifact = InvalidValueReplacementCalculator().fit((frame, target), flags)

    result, result_target = InvalidValueReplacementApplier().apply((frame, target), artifact)

    assert int(_native(result)["signed"][0]) == -9007199254740993
    assert int(_native(result)["unsigned"][1]) == 18446744073709551615
    if isinstance(original, pd.DataFrame):
        pd.testing.assert_frame_equal(_native(result), original)
        pd.testing.assert_frame_equal(_native(frame), original)
    else:
        assert_frame_equal(_native(result), original)
        assert_frame_equal(_native(frame), original)
    assert result_target is target


@pytest.mark.parametrize("wrapped", [False, True])
def test_float_training_artifact_preserves_integer_inference_values(wrapped: bool) -> None:
    """Replay must inspect current dtypes rather than assume training columns stay floating."""
    training = pd.DataFrame({"value": [float("inf"), 1.0]})
    artifact = InvalidValueReplacementCalculator().fit(
        training, {"replace_inf": True, "replace_neg_inf": True}
    )
    inference = pl.DataFrame({"value": [9007199254740993, None]}, schema={"value": pl.Int64})
    frame = SkyulfPolarsWrapper(inference) if wrapped else inference

    result = InvalidValueReplacementApplier().apply(frame, artifact)

    assert_frame_equal(_native(result), inference)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({"replace_inf": True}, [np.nan, -np.inf, 2.0, np.nan]),
        ({"replace_neg_inf": True}, [np.inf, np.nan, 2.0, np.nan]),
        ({"replace_inf": True, "replace_neg_inf": True}, [np.nan, np.nan, 2.0, np.nan]),
    ],
)
def test_integer_preservation_keeps_float_infinity_replacement(
    flags: dict[str, bool], expected: list[float]
) -> None:
    """Skipping impossible integer matches must still replace selected float infinities."""
    frame = pl.DataFrame({"value": [np.inf, -np.inf, 2.0, np.nan]})
    artifact = InvalidValueReplacementCalculator().fit(frame, flags)

    result = InvalidValueReplacementApplier().apply(frame, artifact)

    np.testing.assert_allclose(result["value"].to_numpy(), expected, equal_nan=True)


def test_integer_rule_still_runs_alongside_infinity_flags() -> None:
    """Integer infinity checks are irrelevant but configured numeric rules remain effective."""
    frame = pl.DataFrame({"value": [-1, 9007199254740993, None]}, schema={"value": pl.Int64})
    artifact = InvalidValueReplacementCalculator().fit(
        frame, {"replace_inf": True, "rule": "negative", "value": 0}
    )

    result = InvalidValueReplacementApplier().apply(frame, artifact)

    assert_frame_equal(
        result,
        pl.DataFrame({"value": [0, 9007199254740993, None]}, schema={"value": pl.Int64}),
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int64", "uint64", "Int64", "UInt64"])
@pytest.mark.parametrize("replacement", [None, np.nan, pytest.param(pd.NA, id="pandas_missing")])
def test_null_rules_preserve_exact_integer_schema_across_requests(engine, dtype, replacement):
    """Replacing an integer by missing must not round untouched values or vary chunk types."""
    limit = 127 if dtype.endswith("8") else 9007199254740993
    original = pd.DataFrame({"value": pd.Series([0, limit], dtype=dtype), "target": [7, 9]})
    frame = original if engine == "pandas" else pl.from_pandas(original)
    state = InvalidValueReplacementCalculator().fit(
        frame, {"columns": ["value"], "rule": "zero", "replacement": replacement}
    )
    applier = InvalidValueReplacementApplier()
    full = applier.apply(frame, state)
    nullable = dtype.replace("uint", "UInt").replace("int", "Int")
    expected = original.copy()
    expected["value"] = pd.Series([None, limit], dtype=nullable)
    if isinstance(frame, pd.DataFrame):
        pd.testing.assert_frame_equal(full, expected)
        for chunk in (frame.iloc[:1], frame.iloc[1:], frame.head(0)):
            pd.testing.assert_frame_equal(applier.apply(chunk, state), expected.loc[chunk.index])
        pd.testing.assert_frame_equal(frame, original)
    else:
        assert_frame_equal(full, pl.from_pandas(expected))
        for start, count in ((0, 1), (1, 1), (0, 0)):
            assert_frame_equal(
                applier.apply(frame.slice(start, count), state), full.slice(start, count)
            )
        assert_frame_equal(frame, pl.from_pandas(original))
    assert int(full["value"].iloc[1] if engine == "pandas" else full["value"][1]) == limit
