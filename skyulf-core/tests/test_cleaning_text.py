"""Unit tests for the TextCleaning node.

Covers: trim / case / remove_special / regex helpers (pandas + polars),
Calculator.fit branches, Applier.apply edge cases, and engine parity.
"""

from typing import Any, Dict

import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from skyulf.preprocessing.cleaning.text import (
    TextCleaningApplier,
    TextCleaningCalculator,
    _case_pandas,
    _regex_pandas,
    _remove_special_pandas,
    _trim_pandas,
)

# ---------------------------------------------------------------------------
# _trim_pandas
# ---------------------------------------------------------------------------


def test_trim_both_strips_leading_and_trailing() -> None:
    """'both' mode must remove whitespace from both ends."""
    s = pd.Series(["  hello  ", " world"])
    result = _trim_pandas(s, "both")
    assert list(result) == ["hello", "world"]


def test_trim_leading_only() -> None:
    """'leading' mode must strip only the left side."""
    s = pd.Series(["  hi  "])
    result = _trim_pandas(s, "leading")
    assert result.iloc[0] == "hi  "


def test_trim_trailing_only() -> None:
    """'trailing' mode must strip only the right side."""
    s = pd.Series(["  hi  "])
    result = _trim_pandas(s, "trailing")
    assert result.iloc[0] == "  hi"


def test_trim_default_falls_through_to_strip() -> None:
    """Any unrecognised trim mode must fall back to full strip."""
    s = pd.Series(["  x  "])
    result = _trim_pandas(s, "unknown_mode")
    assert result.iloc[0] == "x"


# ---------------------------------------------------------------------------
# _case_pandas
# ---------------------------------------------------------------------------


def test_case_upper() -> None:
    """'upper' must convert all letters to uppercase."""
    s = pd.Series(["hello", "World"])
    result = _case_pandas(s, "upper")
    assert list(result) == ["HELLO", "WORLD"]


def test_case_title() -> None:
    """'title' must capitalise each word."""
    s = pd.Series(["hello world"])
    result = _case_pandas(s, "title")
    assert result.iloc[0] == "Hello World"


def test_case_sentence_capitalises_first_letter() -> None:
    """'sentence' must capitalise only the first letter of the string."""
    s = pd.Series(["hELLO WORLD"])
    result = _case_pandas(s, "sentence")
    assert result.iloc[0] == "Hello world"


def test_case_lower() -> None:
    """Default case mode must convert to lowercase."""
    s = pd.Series(["HELLO", "WoRLd"])
    result = _case_pandas(s, "lower")
    assert list(result) == ["hello", "world"]


def test_case_default_falls_through_to_lower() -> None:
    """Any unrecognised mode defaults to lowercase."""
    s = pd.Series(["ABC"])
    result = _case_pandas(s, "banana")
    assert result.iloc[0] == "abc"


# ---------------------------------------------------------------------------
# _remove_special_pandas
# ---------------------------------------------------------------------------


def test_remove_special_keep_alphanumeric_removes_symbols() -> None:
    """keep_alphanumeric must strip everything except letters and digits."""
    s = pd.Series(["hello, world!", "abc123$%^"])
    result = _remove_special_pandas(s, "keep_alphanumeric", "")
    assert result.iloc[0] == "helloworld"
    assert result.iloc[1] == "abc123"


def test_remove_special_keep_alphanumeric_space_preserves_spaces() -> None:
    """keep_alphanumeric_space must keep spaces but remove other symbols."""
    s = pd.Series(["hello, world!"])
    result = _remove_special_pandas(s, "keep_alphanumeric_space", "")
    assert result.iloc[0] == "hello world"


def test_remove_special_letters_only() -> None:
    """letters_only must strip digits and symbols, keep letters."""
    s = pd.Series(["abc123!"])
    result = _remove_special_pandas(s, "letters_only", "")
    assert result.iloc[0] == "abc"


def test_remove_special_digits_only() -> None:
    """digits_only must strip letters and symbols, keep digits."""
    s = pd.Series(["abc123!"])
    result = _remove_special_pandas(s, "digits_only", "")
    assert result.iloc[0] == "123"


def test_remove_special_unknown_mode_is_noop() -> None:
    """An unrecognised mode must leave the column unchanged."""
    s = pd.Series(["hello!"])
    result = _remove_special_pandas(s, "does_not_exist", "")
    assert result.iloc[0] == "hello!"


def test_remove_special_with_replacement_char() -> None:
    """Replacement parameter must be used as the substitute character."""
    s = pd.Series(["a$b"])
    result = _remove_special_pandas(s, "keep_alphanumeric", "-")
    assert result.iloc[0] == "a-b"


# ---------------------------------------------------------------------------
# _regex_pandas
# ---------------------------------------------------------------------------


def test_regex_collapse_whitespace() -> None:
    """collapse_whitespace must normalise multiple spaces to a single space."""
    s = pd.Series(["hello   world", "  extra  spaces  "])
    result = _regex_pandas(s, "collapse_whitespace", None, "")
    assert result.iloc[0] == "hello world"
    assert result.iloc[1] == "extra spaces"


def test_regex_extract_digits() -> None:
    """extract_digits must return the first digit sequence in the string."""
    s = pd.Series(["abc123xyz", "no digits", "42"])
    result = _regex_pandas(s, "extract_digits", None, "")
    assert result.iloc[0] == "123"
    assert result.iloc[2] == "42"


def test_regex_normalize_slash_dates_is_noop_placeholder() -> None:
    """normalize_slash_dates is a placeholder; input must be returned unchanged."""
    s = pd.Series(["12/01/2024"])
    result = _regex_pandas(s, "normalize_slash_dates", None, "")
    assert result.iloc[0] == "12/01/2024"


def test_regex_custom_with_pattern() -> None:
    """custom mode must apply the provided regex pattern."""
    s = pd.Series(["foo123bar", "baz456qux"])
    result = _regex_pandas(s, "custom", r"\d+", "NUM")
    assert result.iloc[0] == "fooNUMbar"
    assert result.iloc[1] == "bazNUMqux"


def test_regex_custom_without_pattern_is_noop() -> None:
    """custom mode with no pattern must leave the column unchanged."""
    s = pd.Series(["hello"])
    result = _regex_pandas(s, "custom", None, "")
    assert result.iloc[0] == "hello"


def test_regex_unknown_mode_is_noop() -> None:
    """An unrecognised regex mode must return the series unchanged."""
    s = pd.Series(["test"])
    result = _regex_pandas(s, "unknown_mode", None, "")
    assert result.iloc[0] == "test"


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
    params: Dict[str, Any] = {
        "columns": ["text"],
        "operations": [],
    }
    result = TextCleaningApplier().apply(df, params)
    assert result["text"].iloc[0] == "  Hello  "


def test_applier_no_valid_columns_is_noop() -> None:
    """Columns not present in the DataFrame must be silently skipped."""
    df = pd.DataFrame({"text": ["hello"]})
    params: Dict[str, Any] = {
        "columns": ["nonexistent"],
        "operations": [{"op": "trim", "mode": "both"}],
    }
    result = TextCleaningApplier().apply(df, params)
    assert result["text"].iloc[0] == "hello"


def test_applier_empty_string_survives() -> None:
    """Empty strings must not raise; trim/case should produce an empty string."""
    df = pd.DataFrame({"text": ["", "  ", "hello"]})
    params: Dict[str, Any] = {
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
    params: Dict[str, Any] = {
        "columns": ["text"],
        "operations": [{"op": "trim", "mode": "both"}],
    }
    result = TextCleaningApplier().apply(df, params)
    assert result["text"].iloc[0] == "héllo"
    assert result["text"].iloc[1] == "wörld"


def test_applier_remove_special_keeps_alphanumeric() -> None:
    """remove_special(keep_alphanumeric) must strip all punctuation."""
    df = pd.DataFrame({"text": ["hello, world!", "foo@bar.baz"]})
    params: Dict[str, Any] = {
        "columns": ["text"],
        "operations": [{"op": "remove_special", "mode": "keep_alphanumeric", "replacement": ""}],
    }
    result = TextCleaningApplier().apply(df, params)
    assert result["text"].iloc[0] == "helloworld"
    assert result["text"].iloc[1] == "foobarbaz"


def test_applier_collapse_whitespace_via_regex() -> None:
    """regex/collapse_whitespace must collapse consecutive spaces to one."""
    df = pd.DataFrame({"text": ["hello   world", "a  b  c"]})
    params: Dict[str, Any] = {
        "columns": ["text"],
        "operations": [{"op": "regex", "mode": "collapse_whitespace"}],
    }
    result = TextCleaningApplier().apply(df, params)
    assert result["text"].iloc[0] == "hello world"
    assert result["text"].iloc[1] == "a b c"


def test_applier_non_string_column_is_cast_to_str() -> None:
    """A numeric column must be cast to string before text ops are applied."""
    df = pd.DataFrame({"num": [123, 456]})
    params: Dict[str, Any] = {
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
    params: Dict[str, Any] = {
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


def _pl_apply(ops: list) -> pd.DataFrame:
    """Helper: apply ops to a polars DF and return result as pandas."""
    df = pl.DataFrame({"text": ["  Hello  ", "WORLD", " foo bar "]})
    params: Dict[str, Any] = {"columns": ["text"], "operations": ops}
    result = TextCleaningApplier().apply(df, params)
    return result.to_pandas() if hasattr(result, "to_pandas") else result


def test_polars_trim_leading_strips_left_only() -> None:
    """Polars 'leading' trim must remove only leading whitespace."""
    result = _pl_apply([{"op": "trim", "mode": "leading"}])
    # "  Hello  " → "Hello  "
    assert result["text"].iloc[0] == "Hello  "


def test_polars_trim_trailing_strips_right_only() -> None:
    """Polars 'trailing' trim must remove only trailing whitespace."""
    result = _pl_apply([{"op": "trim", "mode": "trailing"}])
    # "  Hello  " → "  Hello"
    assert result["text"].iloc[0] == "  Hello"


def test_polars_case_upper() -> None:
    """Polars 'upper' case must convert all letters to uppercase."""
    result = _pl_apply([{"op": "case", "mode": "upper"}])
    assert result["text"].iloc[0] == "  HELLO  "


def test_polars_case_title() -> None:
    """Polars 'title' case must capitalise each word."""
    df = pl.DataFrame({"text": ["hello world", "foo bar"]})
    params: Dict[str, Any] = {"columns": ["text"], "operations": [{"op": "case", "mode": "title"}]}
    result = TextCleaningApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["text"].iloc[0] == "Hello World"


def test_polars_case_sentence_capitalises_first_letter() -> None:
    """Polars 'sentence' case must capitalise only the first character."""
    df = pl.DataFrame({"text": ["hELLO WORLD"]})
    params: Dict[str, Any] = {
        "columns": ["text"],
        "operations": [{"op": "case", "mode": "sentence"}],
    }
    result = TextCleaningApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    # First char → upper, rest → lower
    assert result["text"].iloc[0] == "Hello world"


def test_polars_remove_special_keep_alphanumeric() -> None:
    """Polars remove_special must strip punctuation from text via polars regex."""
    df = pl.DataFrame({"text": ["hello, world!", "abc123$%"]})
    params: Dict[str, Any] = {
        "columns": ["text"],
        "operations": [{"op": "remove_special", "mode": "keep_alphanumeric", "replacement": ""}],
    }
    result = TextCleaningApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["text"].iloc[0] == "helloworld"
    assert result["text"].iloc[1] == "abc123"


def test_polars_regex_collapse_whitespace() -> None:
    """Polars collapse_whitespace regex must normalise spaces."""
    df = pl.DataFrame({"text": ["hello   world"]})
    params: Dict[str, Any] = {
        "columns": ["text"],
        "operations": [{"op": "regex", "mode": "collapse_whitespace"}],
    }
    result = TextCleaningApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["text"].iloc[0] == "hello world"


def test_polars_regex_extract_digits() -> None:
    """Polars extract_digits regex must extract the first digit sequence."""
    df = pl.DataFrame({"text": ["abc123xyz"]})
    params: Dict[str, Any] = {
        "columns": ["text"],
        "operations": [{"op": "regex", "mode": "extract_digits"}],
    }
    result = TextCleaningApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["text"].iloc[0] == "123"


def test_polars_regex_custom_pattern() -> None:
    """Polars custom regex must substitute matched groups with replacement."""
    df = pl.DataFrame({"text": ["foo123bar"]})
    params: Dict[str, Any] = {
        "columns": ["text"],
        "operations": [{"op": "regex", "mode": "custom", "pattern": r"\d+", "repl": "NUM"}],
    }
    result = TextCleaningApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["text"].iloc[0] == "fooNUMbar"


def test_polars_regex_normalize_slash_dates_is_noop_placeholder() -> None:
    """Polars normalize_slash_dates is a placeholder; input must be returned unchanged."""
    df = pl.DataFrame({"text": ["12/01/2024"]})
    params: Dict[str, Any] = {
        "columns": ["text"],
        "operations": [{"op": "regex", "mode": "normalize_slash_dates"}],
    }
    result = TextCleaningApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["text"].iloc[0] == "12/01/2024"


def test_polars_regex_custom_without_pattern_is_noop() -> None:
    """Polars custom mode with no pattern must leave the column unchanged."""
    df = pl.DataFrame({"text": ["hello"]})
    params: Dict[str, Any] = {
        "columns": ["text"],
        "operations": [{"op": "regex", "mode": "custom"}],
    }
    result = TextCleaningApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["text"].iloc[0] == "hello"


def test_polars_no_valid_columns_is_noop() -> None:
    """Polars applier with missing columns must return the DataFrame unchanged."""
    df = pl.DataFrame({"text": ["hello"]})
    params: Dict[str, Any] = {
        "columns": ["nonexistent"],
        "operations": [{"op": "trim", "mode": "both"}],
    }
    result = TextCleaningApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["text"].iloc[0] == "hello"
