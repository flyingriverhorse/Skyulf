"""Numeric imputation and text cleaning must honor semantic column types."""

from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines.registry import EngineRegistry
from skyulf.preprocessing._helpers import auto_detect_text_columns
from skyulf.preprocessing.cleaning.alias import AliasReplacementApplier, AliasReplacementCalculator
from skyulf.preprocessing.cleaning.text import TextCleaningApplier, TextCleaningCalculator
from skyulf.preprocessing.imputation.simple import SimpleImputerApplier, SimpleImputerCalculator
from skyulf.utils import detect_numeric_columns


@pytest.fixture(params=["pandas", "polars", "pandas_wrapped", "polars_wrapped"])
def make_frame(request: pytest.FixtureRequest) -> Any:
    """Keep equivalent dtypes while testing both native and public wrapper inputs."""

    def build(data: dict[str, Any]) -> Any:
        """Convert pandas fixtures without converting Decimal values to strings."""
        frame: Any = pd.DataFrame(data)
        if request.param.startswith("polars"):
            frame = pl.from_pandas(frame)
            if "enum" in frame.columns:
                frame = frame.with_columns(pl.col("enum").cast(pl.Enum([" YES ", " NO "])))
        return EngineRegistry.wrap(frame) if request.param.endswith("wrapped") else frame

    return build


def _pandas(frame: Any) -> pd.DataFrame:
    """Normalize assertions without changing the public input under test."""
    if hasattr(frame, "to_native"):
        frame = frame.to_native()
    return frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize(
    "text",
    [
        pd.Series(["a", np.nan, "b"], dtype=object),
        pd.Series(["1", np.nan, "3"], dtype=object),
        pd.Series(pd.Categorical(["a", None, "b"])),
    ],
    ids=["strings", "numeric-strings", "categorical"],
)
def test_numeric_imputation_rejects_explicit_text_before_fitting(
    make_frame: Any, strategy: str, text: pd.Series
) -> None:
    """Explicit mixed selections must fail clearly instead of silently losing requested columns."""
    frame = make_frame({"num": [1.0, np.nan, 3.0], "text": text})
    original = _pandas(frame).copy(deep=True)

    with pytest.raises(ValueError, match=rf"{strategy}.*numeric.*text"):
        SimpleImputerCalculator().fit(frame, {"columns": ["num", "text"], "strategy": strategy})

    pd.testing.assert_frame_equal(_pandas(frame), original)


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("explicit", [False, True], ids=["auto", "explicit"])
def test_numeric_imputation_valid_selections_replay_fitted_statistics(
    make_frame: Any, strategy: str, explicit: bool
) -> None:
    """Validation must retain Decimal inputs and explicitly selected binary and constant numbers."""
    frame = make_frame(
        {
            "num": [1.0, np.nan, 3.0],
            "binary": [0.0, np.nan, 1.0],
            "constant": [2.0, np.nan, 2.0],
            "price": [Decimal("1.25"), None, Decimal("3.75")],
            "text": ["KEEP", None, "TEXT"],
            "target": [10.0, 20.0, 30.0],
        }
    )
    original = _pandas(frame).copy(deep=True)
    columns = ["num", "binary", "constant", "price"] if explicit else ["num", "price"]
    config: dict[str, Any] = {"strategy": strategy, "target_column": "target"}
    if explicit:
        config["columns"] = columns
    artifact = SimpleImputerCalculator().fit(frame, config)
    filled = _pandas(SimpleImputerApplier().apply(frame, artifact))

    assert artifact["columns"] == columns
    expected_fills = {"num": 2.0, "price": 2.5}
    if explicit:
        expected_fills.update(binary=0.5, constant=2.0)
    assert artifact["fill_values"] == expected_fills
    assert artifact["missing_counts"] == dict.fromkeys(columns, 1)
    for col, fill in expected_fills.items():
        assert filled[col].iloc[1] == fill
    untouched = [col for col in original if col not in columns]
    pd.testing.assert_frame_equal(filled[untouched], original[untouched])

    heldout = make_frame(
        {
            "num": [100.0, np.nan],
            "binary": [1.0, np.nan],
            "constant": [99.0, np.nan],
            "price": [Decimal("100.25"), None],
            "text": ["UNSEEN", "VALUES"],
        }
    )
    heldout_original = _pandas(heldout).copy(deep=True)
    replay = SimpleImputerApplier().apply(heldout, artifact)
    replay_frame = _pandas(replay)
    for col, fill in expected_fills.items():
        assert replay_frame[col].iloc[1] == fill
    assert "price" in detect_numeric_columns(replay, False, False)
    assert replay_frame["price"].iloc[0] == 100.25
    pd.testing.assert_frame_equal(_pandas(heldout), heldout_original)
    pd.testing.assert_frame_equal(_pandas(frame), original)


@pytest.mark.parametrize("strategy,fill", [("most_frequent", "a"), ("constant", "replacement")])
def test_nonnumeric_imputation_strategies_keep_explicit_text_support(
    make_frame: Any, strategy: str, fill: str
) -> None:
    """Restricting numeric strategies must not reject supported text imputation or alter replay."""
    frame = make_frame({"text": pd.Series(["a", np.nan, "a"], dtype=object)})
    artifact = SimpleImputerCalculator().fit(
        frame, {"columns": ["text"], "strategy": strategy, "fill_value": fill}
    )
    heldout = make_frame({"text": pd.Series(["unseen", np.nan], dtype=object)})

    assert artifact["fill_values"] == {"text": fill}
    assert _pandas(SimpleImputerApplier().apply(heldout, artifact))["text"].tolist() == [
        "unseen",
        fill,
    ]


@pytest.mark.parametrize(
    "calculator,applier,config,expected",
    [
        (
            TextCleaningCalculator,
            TextCleaningApplier,
            {"operations": [{"op": "trim"}, {"op": "case", "mode": "lower"}]},
            ["yes", None, "no"],
        ),
        (
            AliasReplacementCalculator,
            AliasReplacementApplier,
            {"alias_type": "boolean"},
            ["Yes", None, "No"],
        ),
    ],
)
def test_automatic_text_cleaning_preserves_decimal_columns(
    make_frame: Any, calculator: Any, applier: Any, config: dict[str, Any], expected: list[Any]
) -> None:
    """Automatic cleaning must preserve numeric Decimal values through fitting and held-out replay."""
    frame = make_frame(
        {
            "price": [Decimal("1.25"), None, Decimal("3.75")],
            "plain": pd.Series([" YES ", None, " NO "], dtype=object),
            "string": pd.Series([" YES ", None, " NO "], dtype="string"),
            "category": pd.Categorical([" YES ", None, " NO "]),
            "enum": pd.Categorical([" YES ", None, " NO "]),
            "target": [" KEEP ", " TARGET ", " VALUES "],
        }
    )
    original = _pandas(frame).copy(deep=True)
    artifact = calculator().fit(frame, {**config, "target_column": "target"})
    cleaned = applier().apply(frame, artifact)
    output = _pandas(cleaned)

    assert cleaned["price"].dtype == frame["price"].dtype
    pd.testing.assert_frame_equal(output[["price", "target"]], original[["price", "target"]])
    assert isinstance(output["price"].iloc[0], Decimal)
    assert auto_detect_text_columns(frame) == ["plain", "string", "category", "enum", "target"]
    assert artifact["columns"] == ["plain", "string", "category", "enum"]
    for col in artifact["columns"]:
        pd.testing.assert_series_equal(
            output[col].astype("string"), pd.Series(expected, name=col, dtype="string")
        )
    heldout = make_frame({"price": [Decimal("9.75"), None], "plain": [" NO ", " YES "]})
    heldout_original = _pandas(heldout).copy(deep=True)
    cleaned_heldout = applier().apply(heldout, artifact)
    replay = _pandas(cleaned_heldout)
    assert cleaned_heldout["price"].dtype == heldout["price"].dtype
    assert replay["plain"].tolist() == [expected[2], expected[0]]
    pd.testing.assert_series_equal(replay["price"], heldout_original["price"])
    pd.testing.assert_frame_equal(_pandas(heldout), heldout_original)
    pd.testing.assert_frame_equal(_pandas(frame), original)
