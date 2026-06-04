"""Tests for the schema-contract primitive (``SchemaMismatchError``).

Covers column presence, extra/unexpected columns, dtype mismatches, and
column-order checks via ``SkyulfSchema.assert_compatible`` / ``validate_schema``.
"""

import pandas as pd
import pytest

from skyulf.preprocessing import SchemaMismatchError, SkyulfSchema, validate_schema


def _schema(cols, dtypes=None):
    return SkyulfSchema.from_columns(cols, dtypes)


def test_compatible_passes_on_identical_columns():
    expected = _schema(["a", "b", "c"])
    actual = _schema(["a", "b", "c"])
    expected.assert_compatible(actual)  # no raise


def test_missing_column_raises_with_details():
    expected = _schema(["a", "b", "c"])
    actual = _schema(["a", "b"])
    with pytest.raises(SchemaMismatchError) as exc:
        expected.assert_compatible(actual)
    assert exc.value.missing == ["c"]
    assert exc.value.unexpected == []
    assert "missing columns" in str(exc.value)


def test_unexpected_column_raises_with_details():
    expected = _schema(["a", "b"])
    actual = _schema(["a", "b", "z"])
    with pytest.raises(SchemaMismatchError) as exc:
        expected.assert_compatible(actual)
    assert exc.value.unexpected == ["z"]
    assert exc.value.missing == []


def test_extra_columns_allowed_when_only_presence_relevant():
    # Default contract requires expected columns present; extras still report
    # as unexpected, so this should raise. Presence-only callers that tolerate
    # extras can catch and inspect `.missing`.
    expected = _schema(["a"])
    actual = _schema(["a", "b"])
    with pytest.raises(SchemaMismatchError) as exc:
        expected.assert_compatible(actual)
    assert exc.value.missing == []
    assert exc.value.unexpected == ["b"]


def test_dtype_mismatch_only_checked_when_requested():
    expected = _schema(["a"], {"a": "int64"})
    actual = _schema(["a"], {"a": "float64"})
    # Default: dtype ignored -> passes.
    expected.assert_compatible(actual)
    # Opt-in: dtype checked -> raises.
    with pytest.raises(SchemaMismatchError) as exc:
        expected.assert_compatible(actual, check_dtypes=True)
    assert exc.value.dtype_mismatches == {"a": ("int64", "float64")}


def test_order_mismatch_only_checked_when_requested():
    expected = _schema(["a", "b", "c"])
    actual = _schema(["a", "c", "b"])
    # Default: order ignored -> passes.
    expected.assert_compatible(actual)
    # Opt-in: order checked -> raises.
    with pytest.raises(SchemaMismatchError) as exc:
        expected.assert_compatible(actual, check_order=True)
    assert exc.value.order_mismatch is True


def test_validate_schema_builds_from_dataframe():
    expected = _schema(["a", "b"])
    df = pd.DataFrame({"a": [1], "c": [2]})
    with pytest.raises(SchemaMismatchError) as exc:
        validate_schema(expected, df)
    assert exc.value.missing == ["b"]
    assert exc.value.unexpected == ["c"]


def test_validate_schema_passes_on_match():
    expected = _schema(["a", "b"])
    df = pd.DataFrame({"a": [1], "b": [2]})
    validate_schema(expected, df)  # no raise


def test_error_message_mentions_where_label():
    expected = _schema(["a"])
    actual = _schema([])
    with pytest.raises(SchemaMismatchError, match="on training input"):
        expected.assert_compatible(actual, where="training input")
