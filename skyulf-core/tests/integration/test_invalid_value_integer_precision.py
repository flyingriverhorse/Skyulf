"""OC-297: infinity-only cleanup must preserve exact integer observations."""

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
