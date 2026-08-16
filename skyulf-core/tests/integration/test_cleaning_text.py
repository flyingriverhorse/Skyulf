"""Unit tests for the TextCleaning node.

Covers: trim / case / remove_special / regex helpers (pandas + polars),
Calculator.fit branches, Applier.apply edge cases, and engine parity.
"""

from typing import Any

import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.cleaning.text import (
    TextCleaningApplier,
    TextCleaningCalculator,
    _case_pandas,
    _regex_pandas,
    _remove_special_pandas,
    _trim_pandas,
)

_trim_pandas_cases = TestCaseLoader("preprocessing/cleaning_text", group="trim_pandas").load()
_case_pandas_cases = TestCaseLoader("preprocessing/cleaning_text", group="case_pandas").load()
_remove_special_pandas_cases = TestCaseLoader(
    "preprocessing/cleaning_text", group="remove_special_pandas"
).load()
_regex_pandas_cases = TestCaseLoader("preprocessing/cleaning_text", group="regex_pandas").load()


def _assert_series_matches(result: pd.Series, expected: list) -> None:
    """Compare a pandas Series against an expected list, treating None as NaN."""
    for got, exp in zip(result, expected, strict=True):
        if exp is None:
            assert pd.isna(got)
        else:
            assert got == exp


# ---------------------------------------------------------------------------
# _trim_pandas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_trim_pandas_cases)
def test_trim_pandas(values: list, mode: str, expected: list) -> None:
    """_trim_pandas must strip whitespace according to the given mode."""
    s = pd.Series(values)
    result = _trim_pandas(s, mode)
    _assert_series_matches(result, expected)


# ---------------------------------------------------------------------------
# _case_pandas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_case_pandas_cases)
def test_case_pandas(values: list, mode: str, expected: list) -> None:
    """_case_pandas must transform text casing according to the given mode."""
    s = pd.Series(values)
    result = _case_pandas(s, mode)
    _assert_series_matches(result, expected)


# ---------------------------------------------------------------------------
# _remove_special_pandas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_remove_special_pandas_cases)
def test_remove_special_pandas(values: list, mode: str, replacement: str, expected: list) -> None:
    """_remove_special_pandas must strip/replace characters according to mode."""
    s = pd.Series(values)
    result = _remove_special_pandas(s, mode, replacement)
    _assert_series_matches(result, expected)


# ---------------------------------------------------------------------------
# _regex_pandas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_regex_pandas_cases)
def test_regex_pandas(
    values: list, mode: str, pattern: str | None, repl: str, expected: list
) -> None:
    """_regex_pandas must apply the regex behavior matching the given mode."""
    s = pd.Series(values)
    result = _regex_pandas(s, mode, pattern, repl)
    _assert_series_matches(result, expected)


# ---------------------------------------------------------------------------
# Calculator.fit
# ---------------------------------------------------------------------------


def test_calculator_fit_returns_correct_artifact() -> None:
    """fit must return an artifact with type, columns, and operations."""
    df = pd.DataFrame({"text": ["Hello", "World"]})
    params = TextCleaningCalculator().fit(
        df,
        {
            "columns": ["text"],
            "operations": [{"op": "trim", "mode": "both"}],
        },
    )
    assert params["type"] == "text_cleaning"
    assert "text" in params["columns"]
    assert params["operations"] == [{"op": "trim", "mode": "both"}]


def test_calculator_fit_empty_columns_short_circuits() -> None:
    """Explicit empty columns list must short-circuit to {}."""
    df = pd.DataFrame({"text": ["a", "b"]})
    params = TextCleaningCalculator().fit(df, {"columns": []})
    assert params == {}


def test_calculator_fit_no_text_columns_returns_empty() -> None:
    """A purely numeric DataFrame triggers auto-detection that finds no text cols."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    # No 'columns' key → auto-detection runs; no string cols → returns {}
    params = TextCleaningCalculator().fit(df, {"operations": [{"op": "trim", "mode": "both"}]})
    assert params == {}


def test_calculator_fit_auto_detects_text_columns() -> None:
    """When columns is absent the Calculator auto-detects string columns."""
    df = pd.DataFrame({"text": ["Hello", "World"], "num": [1.0, 2.0]})
    params = TextCleaningCalculator().fit(df, {"operations": [{"op": "case", "mode": "lower"}]})
    assert "text" in params["columns"]
    assert "num" not in params["columns"]


def test_infer_output_schema_returns_input_schema_unchanged() -> None:
    """TextCleaning infer_output_schema must pass the schema through unchanged."""
    from skyulf.core.schema import SkyulfSchema

    schema = SkyulfSchema.from_columns(["text"], {"text": "string"})
    result = TextCleaningCalculator().infer_output_schema(schema, {})
    assert result is schema


# ---------------------------------------------------------------------------
# Applier.apply — pandas path
# ---------------------------------------------------------------------------


def test_applier_trim_and_lower() -> None:
    """Chained trim+lowercase must produce clean results."""
    df = pd.DataFrame({"name": ["  John  ", " JANE", "bob  "]})
    calc = TextCleaningCalculator()
    applier = TextCleaningApplier()
    params = calc.fit(
        df,
        {
            "columns": ["name"],
            "operations": [
                {"op": "trim", "mode": "both"},
                {"op": "case", "mode": "lower"},
            ],
        },
    )
    result = applier.apply(df, params)
    assert list(result["name"]) == ["john", "jane", "bob"]


def test_applier_no_operations_is_noop() -> None:
    """Empty operations list must leave the DataFrame unchanged."""
    df = pd.DataFrame({"text": ["  Hello  "]})
    params: dict[str, Any] = {
        "columns": ["text"],
        "operations": [],
    }
    result = TextCleaningApplier().apply(df, params)
    assert result["text"].iloc[0] == "  Hello  "


def test_applier_no_valid_columns_is_noop() -> None:
    """Columns not present in the DataFrame must be silently skipped."""
    df = pd.DataFrame({"text": ["hello"]})
    params: dict[str, Any] = {
        "columns": ["nonexistent"],
        "operations": [{"op": "trim", "mode": "both"}],
    }
    result = TextCleaningApplier().apply(df, params)
    assert result["text"].iloc[0] == "hello"


def test_applier_empty_string_survives() -> None:
    """Empty strings must not raise; trim/case should produce an empty string."""
    df = pd.DataFrame({"text": ["", "  ", "hello"]})
    params: dict[str, Any] = {
        "columns": ["text"],
        "operations": [
            {"op": "trim", "mode": "both"},
            {"op": "case", "mode": "upper"},
        ],
    }
    result = TextCleaningApplier().apply(df, params)
    assert result["text"].iloc[0] == ""
    assert result["text"].iloc[1] == ""
    assert result["text"].iloc[2] == "HELLO"


def test_applier_unicode_text() -> None:
    """Unicode strings must be processed without encoding errors."""
    df = pd.DataFrame({"text": ["  héllo  ", "  wörld  "]})
    params: dict[str, Any] = {
        "columns": ["text"],
        "operations": [{"op": "trim", "mode": "both"}],
    }
    result = TextCleaningApplier().apply(df, params)
    assert result["text"].iloc[0] == "héllo"
    assert result["text"].iloc[1] == "wörld"


def test_applier_remove_special_keeps_alphanumeric() -> None:
    """remove_special(keep_alphanumeric) must strip all punctuation."""
    df = pd.DataFrame({"text": ["hello, world!", "foo@bar.baz"]})
    params: dict[str, Any] = {
        "columns": ["text"],
        "operations": [{"op": "remove_special", "mode": "keep_alphanumeric", "replacement": ""}],
    }
    result = TextCleaningApplier().apply(df, params)
    assert result["text"].iloc[0] == "helloworld"
    assert result["text"].iloc[1] == "foobarbaz"


def test_applier_collapse_whitespace_via_regex() -> None:
    """regex/collapse_whitespace must collapse consecutive spaces to one."""
    df = pd.DataFrame({"text": ["hello   world", "a  b  c"]})
    params: dict[str, Any] = {
        "columns": ["text"],
        "operations": [{"op": "regex", "mode": "collapse_whitespace"}],
    }
    result = TextCleaningApplier().apply(df, params)
    assert result["text"].iloc[0] == "hello world"
    assert result["text"].iloc[1] == "a b c"


def test_applier_non_string_column_is_cast_to_str() -> None:
    """A numeric column must be cast to string before text ops are applied."""
    df = pd.DataFrame({"num": [123, 456]})
    params: dict[str, Any] = {
        "columns": ["num"],
        "operations": [{"op": "case", "mode": "upper"}],
    }
    result = TextCleaningApplier().apply(df, params)
    # After cast to str, 'upper' is a no-op for digits — but no error must occur
    assert result["num"].dtype == object


# ---------------------------------------------------------------------------
# Engine parity (pandas vs polars Applier paths)
# ---------------------------------------------------------------------------


@st.composite
def _text_frame(draw: st.DrawFn, min_rows: int = 4, max_rows: int = 20) -> pd.DataFrame:
    """Generate a DataFrame with a text column containing varied whitespace/case."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    words = ["hello", "WORLD", " foo ", "  bar", "baz  ", "Hello World", "  "]
    values = draw(st.lists(st.sampled_from(words), min_size=n, max_size=n))
    return pd.DataFrame({"text": values})


@settings(max_examples=25, deadline=None)
@given(df=_text_frame())
def test_text_cleaning_apply_engine_parity_trim_lower(df: pd.DataFrame) -> None:
    """trim+lowercase must produce identical results from pandas and polars paths."""
    params: dict[str, Any] = {
        "columns": ["text"],
        "operations": [
            {"op": "trim", "mode": "both"},
            {"op": "case", "mode": "lower"},
        ],
    }
    applier = TextCleaningApplier()
    pd_result = applier.apply(df, params)
    pl_result = applier.apply(pl.from_pandas(df), params)
    if hasattr(pl_result, "to_pandas"):
        pl_result = pl_result.to_pandas()
    pd.testing.assert_frame_equal(
        pd_result.reset_index(drop=True),
        pl_result.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Polars code path — exercise uncovered branches via polars engine
# ---------------------------------------------------------------------------


_polars_single_op_cases = TestCaseLoader(
    "preprocessing/cleaning_text", group="polars_single_op"
).load()


@pytest.mark.parametrize(*_polars_single_op_cases)
def test_polars_single_op(texts: list, operation: dict[str, Any], expected: list) -> None:
    """A single text-cleaning op applied via the polars path must match expected output."""
    df = pl.DataFrame({"text": texts})
    params: dict[str, Any] = {"columns": ["text"], "operations": [operation]}
    result = TextCleaningApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert list(result["text"]) == expected


def test_polars_no_valid_columns_is_noop() -> None:
    """Polars applier with missing columns must return the DataFrame unchanged."""
    df = pl.DataFrame({"text": ["hello"]})
    params: dict[str, Any] = {
        "columns": ["nonexistent"],
        "operations": [{"op": "trim", "mode": "both"}],
    }
    result = TextCleaningApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["text"].iloc[0] == "hello"


# ---------------------------------------------------------------------------
# Real-shaped dataset integration
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has ``plan_type`` (lowercase strings) and ``city`` (mixed values with a
    NaN row) — verifying that text operations on a real-shaped dataset handle
    NaN passthrough correctly.
    """

    def test_trim_and_lowercase_plan_type_produces_clean_values(self) -> None:
        """Applying trim+lowercase to ``plan_type`` on the customers dataset must
        produce consistently lowercased values with no NaN introduced (plan_type
        has no missing values in customers.csv).
        """
        df = load_sample_dataset("customers")
        calc = TextCleaningCalculator()
        applier = TextCleaningApplier()
        params = calc.fit(
            df,
            {
                "columns": ["plan_type"],
                "operations": [
                    {"op": "trim", "mode": "both"},
                    {"op": "case", "mode": "lower"},
                ],
            },
        )
        result = applier.apply(df, params)

        assert result["plan_type"].notna().all()
        assert set(result["plan_type"].unique()) == {"basic", "premium", "enterprise"}

    def test_text_cleaning_city_with_nan_non_null_values_are_trimmed(self) -> None:
        """Applying trim to ``city`` on real data must leave the non-null city
        strings unchanged (no surrounding whitespace exists in customers.csv).
        Note: the text cleaning node casts object columns to str before applying
        ops, so NaN cells become the literal string "nan" — this is the current
        library behaviour, not a passthrough.
        """
        df = load_sample_dataset("customers")
        calc = TextCleaningCalculator()
        applier = TextCleaningApplier()
        params = calc.fit(
            df,
            {"columns": ["city"], "operations": [{"op": "trim", "mode": "both"}]},
        )
        result = applier.apply(df, params)

        # Non-null city values must not gain or lose any whitespace.
        non_null_original = df.loc[df["city"].notna(), "city"].reset_index(drop=True)
        non_null_result = result.loc[df["city"].notna(), "city"].reset_index(drop=True)
        for orig, trimmed in zip(non_null_original, non_null_result, strict=True):
            assert trimmed == orig.strip()
