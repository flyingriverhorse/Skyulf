"""Tests for shared encoder helpers (skyulf.preprocessing.encoding._common)."""

import logging

import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.encoding._common import (
    _exclude_target_column,
    _parse_categories_order,
    detect_categorical_columns,
)


def test_parse_categories_order_empty_raw_returns_auto() -> None:
    """A falsy `raw` (None or "") short-circuits to "auto"."""
    assert _parse_categories_order(None, 2) == "auto"
    assert _parse_categories_order("", 2) == "auto"


def test_parse_categories_order_string_input_matching_n_cols() -> None:
    """A newline-separated string with the right number of lines parses per-column."""
    raw = "a,b,c\nx,y"
    result = _parse_categories_order(raw, 2)
    assert result == [["a", "b", "c"], ["x", "y"]]


def test_parse_categories_order_list_input_matching_n_cols() -> None:
    """A list of comma-separated strings parses the same way as the string form."""
    raw = ["a,b", "x,y,z"]
    result = _parse_categories_order(raw, 2)
    assert result == [["a", "b"], ["x", "y", "z"]]


def test_parse_categories_order_non_str_non_list_returns_auto() -> None:
    """An unsupported type (e.g. int) falls back to "auto"."""
    assert _parse_categories_order(123, 2) == "auto"


def test_parse_categories_order_mismatched_line_count_returns_auto() -> None:
    """When the parsed line count does not match n_cols, fall back to "auto"."""
    raw = "a,b\nc,d\ne,f"
    assert _parse_categories_order(raw, 2) == "auto"


def test_parse_categories_order_blank_lines_are_ignored() -> None:
    """Blank lines in the raw string are stripped before counting."""
    raw = "a,b\n\nx,y\n"
    result = _parse_categories_order(raw, 2)
    assert result == [["a", "b"], ["x", "y"]]


def test_exclude_target_column_removes_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A destroying encoder with the target column present removes it and logs a warning."""
    columns = ["city", "target"]
    config = {"target_column": "target"}
    with caplog.at_level(logging.WARNING):
        result = _exclude_target_column(columns, config, "OneHotEncoder", y=None)

    assert result == ["city"]
    assert any("Excluding target column" in rec.message for rec in caplog.records)


def test_exclude_target_column_uses_y_name_when_config_missing() -> None:
    """When config lacks target_column, the target name is inferred from y.name."""
    columns = ["city", "target"]
    y = pd.Series([1, 0], name="target")
    result = _exclude_target_column(columns, {}, "DummyEncoder", y=y)
    assert result == ["city"]


def test_exclude_target_column_noop_for_safe_encoders() -> None:
    """LabelEncoder/OrdinalEncoder are target-safe; columns pass through unchanged."""
    columns = ["city", "target"]
    config = {"target_column": "target"}
    result = _exclude_target_column(columns, config, "LabelEncoder", y=None)
    assert result == columns


def test_exclude_target_column_noop_when_target_not_in_columns() -> None:
    """If the target column isn't in the list, nothing changes and no warning fires."""
    columns = ["city"]
    config = {"target_column": "target"}
    result = _exclude_target_column(columns, config, "OneHotEncoder", y=None)
    assert result == ["city"]


def test_detect_categorical_columns_pandas() -> None:
    """Pandas engine path selects object/category dtype columns."""
    df = pd.DataFrame({"city": ["a", "b"], "amount": [1, 2]})
    assert detect_categorical_columns(df) == ["city"]


def test_detect_categorical_columns_polars() -> None:
    """Polars engine path selects Utf8/Categorical/Object dtype columns."""
    df = pl.DataFrame({"city": ["a", "b"], "amount": [1, 2]})
    assert detect_categorical_columns(df) == ["city"]
