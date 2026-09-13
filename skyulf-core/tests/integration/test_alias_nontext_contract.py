"""Alias standardization preserves non-text data in explicitly selected columns."""

from datetime import date

import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.cleaning.alias import AliasReplacementApplier, AliasReplacementCalculator


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("mode", ["boolean", "country", "custom"])
def test_alias_keeps_selected_nontext_values_and_types(engine: str, mode: str) -> None:
    """Selecting numeric or boolean columns must not stringify or reinterpret their values."""
    data = {
        "unmatched": [2, 3, 4],
        "numeric_flags": [1, 0, 2],
        "flags": [True, False, True],
        "real": [2.5, None, 4.5],
        "day": [date(2026, 1, 1), None, date(2026, 1, 3)],
        "text": [" YES ", "other", None],
    }
    frame = pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)
    original = frame.copy(deep=True) if isinstance(frame, pd.DataFrame) else frame.clone()
    artifact = AliasReplacementCalculator().fit(
        frame,
        {"columns": list(data), "alias_type": mode, "custom_map": {"yes": "Yes"}},
    )

    output = AliasReplacementApplier().apply(frame, artifact)

    expected = " YES " if mode == "country" else "Yes"
    if isinstance(original, pd.DataFrame):
        assert isinstance(frame, pd.DataFrame)
        pd.testing.assert_frame_equal(output.drop(columns="text"), original.drop(columns="text"))
        pd.testing.assert_frame_equal(frame, original)
    else:
        assert isinstance(frame, pl.DataFrame)
        assert output.drop("text").equals(original.drop("text"))
        assert frame.equals(original)
    assert output["text"].to_list() == [expected, "other", None]


def test_alias_only_normalizes_strings_in_mixed_pandas_column() -> None:
    """A mixed object column must distinguish numeric 1 from the text alias '1'."""
    frame = pd.DataFrame({"flag": [1, "1", True, " true ", 2, None]})
    artifact = AliasReplacementCalculator().fit(frame, {"columns": ["flag"]})

    output = AliasReplacementApplier().apply(frame, artifact)

    assert output["flag"].tolist() == [1, "Yes", True, "Yes", 2, None]
    assert type(output["flag"].iloc[0]) is int
    assert type(output["flag"].iloc[2]) is bool


def test_alias_preserves_values_in_polars_object_column() -> None:
    """Polars object values follow the same text-only alias contract as pandas objects."""
    frame = pl.DataFrame({"flag": pl.Series([1, "1", True, " true ", 2, None], dtype=pl.Object)})
    artifact = AliasReplacementCalculator().fit(frame, {"columns": ["flag"]})

    output = AliasReplacementApplier().apply(frame, artifact)

    assert output["flag"].to_list() == [1, "Yes", True, "Yes", 2, None]
    assert output.schema["flag"] == pl.Object


def test_alias_preserves_exact_nontext_pandas_objects_with_nulls() -> None:
    """Object mapping must not infer float dtype and round integers above float precision."""
    frame = pd.DataFrame({"value": pd.Series([2**53 + 1, None], dtype=object)})
    artifact = AliasReplacementCalculator().fit(frame, {"columns": ["value"]})

    output = AliasReplacementApplier().apply(frame, artifact)

    pd.testing.assert_frame_equal(output, frame)
    assert output["value"].iloc[0] == 2**53 + 1
    assert output["value"].iloc[1] is None


@pytest.mark.parametrize("dtype", ["string", "category"])
def test_alias_normalizes_pandas_extension_text_and_retains_missing(dtype: str) -> None:
    """Preserving object values must not stop nullable or categorical text aliases from mapping."""
    frame = pd.DataFrame({"value": pd.Series([" y ", "No", None], dtype=dtype)})
    artifact = AliasReplacementCalculator().fit(frame, {})

    output = AliasReplacementApplier().apply(frame, artifact)

    assert output["value"].iloc[:2].tolist() == ["Yes", "No"]
    assert pd.isna(output["value"].iloc[2])
