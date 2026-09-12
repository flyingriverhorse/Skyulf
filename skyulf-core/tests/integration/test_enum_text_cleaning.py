"""Enum text columns must participate in automatic cleaning on both engines."""

from typing import Any

import pandas as pd
import polars as pl
import pytest

from skyulf.engines.registry import EngineRegistry
from skyulf.preprocessing._helpers import auto_detect_text_columns
from skyulf.preprocessing.cleaning.alias import AliasReplacementApplier, AliasReplacementCalculator
from skyulf.preprocessing.cleaning.text import TextCleaningApplier, TextCleaningCalculator


@pytest.fixture(params=["pandas", "polars", "pandas_wrapped", "polars_wrapped"])
def make_frame(request: pytest.FixtureRequest) -> Any:
    """Use equivalent categorical data in native and wrapped frames."""

    def build(values: list[str | None]) -> Any:
        """Keep the category dictionary separate from observed values."""
        categories = ["unused", *dict.fromkeys(v for v in values if v is not None)]
        data = {
            "status": values,
            "plain": values,
            "qty": list(range(len(values))),
            "target": ["KEEP"] * len(values),
        }
        if request.param.startswith("polars"):
            frame = pl.DataFrame(data).with_columns(pl.col("status").cast(pl.Enum(categories)))
        else:
            frame = pd.DataFrame(data)
            frame["status"] = pd.Categorical(values, categories=categories)
        return EngineRegistry.wrap(frame) if request.param.endswith("wrapped") else frame

    return build


def _pandas(frame: Any) -> pd.DataFrame:
    """Normalize assertion input without changing the tested frame."""
    if hasattr(frame, "to_native"):
        frame = frame.to_native()
    return frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame


@pytest.mark.parametrize("values", [["A", None, "B"], [None, None], []])
def test_enum_text_detection_includes_null_and_empty_columns(
    make_frame: Any,
    values: list[str | None],
) -> None:
    """Enum text detection must depend on dtype, including unobserved categories."""
    frame = make_frame(values)
    detected = auto_detect_text_columns(frame)

    assert "status" in detected
    assert "qty" not in detected


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "calculator,applier,config,expected,heldout_expected",
    [
        (
            TextCleaningCalculator,
            TextCleaningApplier,
            {"operations": [{"op": "trim"}, {"op": "case", "mode": "lower"}]},
            ["yes", "no!", None, "maybe"],
            ["no!", "yes", None],
        ),
        (
            AliasReplacementCalculator,
            AliasReplacementApplier,
            {"alias_type": "boolean"},
            ["Yes", "No", None, "Maybe"],
            ["No", "Yes", None],
        ),
    ],
)
def test_enum_cleaning_fit_and_replay(
    make_frame: Any,
    explicit: bool,
    calculator: Any,
    applier: Any,
    config: dict[str, Any],
    expected: list[str | None],
    heldout_expected: list[str | None],
) -> None:
    """Automatic Enum cleaning must match explicit selection and preserve nulls and other columns."""
    frame = make_frame([" YES ", "No!", None, "Maybe"])
    original = _pandas(frame).copy(deep=True)
    selection = {"columns": ["status"]} if explicit else {"target_column": "target"}
    params = calculator().fit(frame, {**config, **selection})
    out = _pandas(applier().apply(frame, params))

    assert params["columns"] == (["status"] if explicit else ["status", "plain"])
    pd.testing.assert_series_equal(
        out["status"].astype("string"), pd.Series(expected, name="status", dtype="string")
    )
    untouched = ["qty", "target", "plain"] if explicit else ["qty", "target"]
    pd.testing.assert_frame_equal(out[untouched], original[untouched])
    pd.testing.assert_frame_equal(_pandas(frame), original)

    heldout = make_frame([" No! ", "YES", None])
    replay = _pandas(applier().apply(heldout, params))
    pd.testing.assert_series_equal(
        replay["status"].astype("string"),
        pd.Series(heldout_expected, name="status", dtype="string"),
    )


@pytest.mark.parametrize(
    "calculator,applier",
    [
        (TextCleaningCalculator, TextCleaningApplier),
        (AliasReplacementCalculator, AliasReplacementApplier),
    ],
)
def test_enum_cleaning_explicit_empty_selection_is_noop(
    make_frame: Any,
    calculator: Any,
    applier: Any,
) -> None:
    """Adding Enum auto-selection must not override an explicit opt-out."""
    frame = make_frame([" YES ", None])
    params = calculator().fit(frame, {"columns": []})

    assert params == {}
    pd.testing.assert_frame_equal(_pandas(applier().apply(frame, params)), _pandas(frame))
