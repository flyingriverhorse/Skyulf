"""Tests for skyulf.core.schema (SkyulfSchema dataclass + validate_schema helper)."""

import pandas as pd
import polars as pl
import pytest

from skyulf.core.schema import SchemaMismatchError, SkyulfSchema, validate_schema


def test_from_columns_builds_schema_with_dtypes():
    """from_columns should build a schema carrying the provided dtypes dict."""
    schema = SkyulfSchema.from_columns(["a", "b"], {"a": "int64"})
    assert schema.columns == ("a", "b")
    assert schema.dtypes == {"a": "int64"}


def test_from_dataframe_extracts_pandas_dtypes():
    """from_dataframe on a pandas frame should extract string dtype labels."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    schema = SkyulfSchema.from_dataframe(df)
    assert schema.columns == ("a", "b")
    assert "int" in schema.dtypes["a"]


def test_from_dataframe_extracts_polars_dtypes():
    """from_dataframe on a polars frame (no pandas .dtypes) should use the polars schema."""
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    schema = SkyulfSchema.from_dataframe(df)
    assert schema.columns == ("a", "b")
    assert "a" in schema.dtypes


def test_from_dataframe_handles_object_without_columns():
    """An object with no `columns` attribute should degrade to an empty schema."""
    schema = SkyulfSchema.from_dataframe(object())
    assert schema.columns == ()


def test_drop_removes_columns_and_dtypes():
    """drop() should remove both the column name and its dtype entry."""
    schema = SkyulfSchema.from_columns(["a", "b"], {"a": "int64", "b": "string"})
    result = schema.drop(["a"])
    assert result.columns == ("b",)
    assert result.dtypes == {"b": "string"}


def test_add_appends_new_column_with_dtype():
    """add() should append a new column name with the given dtype."""
    schema = SkyulfSchema.from_columns(["a"])
    result = schema.add("b", "float64")
    assert result.columns == ("a", "b")
    assert result.dtypes["b"] == "float64"


def test_add_is_noop_when_column_already_present():
    """add() should return the identical schema unchanged if the column already exists."""
    schema = SkyulfSchema.from_columns(["a", "b"])
    result = schema.add("a")
    assert result is schema


def test_rename_updates_columns_and_dtypes():
    """rename() should update both the column list and the dtypes mapping keys."""
    schema = SkyulfSchema.from_columns(["a", "b"], {"a": "int64"})
    result = schema.rename({"a": "renamed"})
    assert result.columns == ("renamed", "b")
    assert result.dtypes == {"renamed": "int64"}


def test_rename_raises_on_collision_with_existing_column():
    """Renaming a column to a name that already exists elsewhere in the schema
    must raise instead of silently producing duplicate column names."""
    schema = SkyulfSchema.from_columns(["a", "b"], {"a": "int64", "b": "string"})
    with pytest.raises(ValueError, match="duplicate column"):
        schema.rename({"a": "b"})


def test_rename_raises_on_collision_between_two_renamed_columns():
    """Renaming two different columns to the same target name must raise."""
    schema = SkyulfSchema.from_columns(["a", "b", "c"])
    with pytest.raises(ValueError, match="duplicate column"):
        schema.rename({"a": "x", "b": "x"})


def test_with_dtype_updates_existing_column():
    """with_dtype() should update the dtype label for an existing column."""
    schema = SkyulfSchema.from_columns(["a"], {"a": "int64"})
    result = schema.with_dtype("a", "float64")
    assert result.dtypes["a"] == "float64"


def test_with_dtype_noop_for_unknown_column():
    """with_dtype() on a column not present should return the schema unchanged."""
    schema = SkyulfSchema.from_columns(["a"])
    result = schema.with_dtype("nonexistent", "float64")
    assert result is schema


def test_has_returns_true_for_present_column():
    """has() should return True for a column present in the schema."""
    schema = SkyulfSchema.from_columns(["a", "b"])
    assert schema.has("a") is True


def test_has_returns_false_for_absent_column():
    """has() should return False for a column not present in the schema."""
    schema = SkyulfSchema.from_columns(["a", "b"])
    assert schema.has("z") is False


def test_column_list_returns_plain_list():
    """column_list() should return a plain list (not a tuple)."""
    schema = SkyulfSchema.from_columns(["a", "b"])
    assert schema.column_list() == ["a", "b"]
    assert isinstance(schema.column_list(), list)


def test_contains_and_len_dunder_methods():
    """__contains__ and __len__ should behave like the underlying columns tuple."""
    schema = SkyulfSchema.from_columns(["a", "b", "c"])
    assert "b" in schema
    assert "z" not in schema
    assert len(schema) == 3


def test_assert_compatible_passes_for_identical_schema():
    """assert_compatible should not raise when actual matches expected exactly."""
    expected = SkyulfSchema.from_columns(["a", "b"])
    actual = SkyulfSchema.from_columns(["a", "b"])
    expected.assert_compatible(actual)  # should not raise


def test_assert_compatible_raises_on_missing_column():
    """A missing expected column should raise SchemaMismatchError listing it."""
    expected = SkyulfSchema.from_columns(["a", "b"])
    actual = SkyulfSchema.from_columns(["a"])
    with pytest.raises(SchemaMismatchError) as exc_info:
        expected.assert_compatible(actual)
    assert exc_info.value.missing == ["b"]


def test_assert_compatible_raises_on_unexpected_column():
    """An unexpected extra column in actual should be reported."""
    expected = SkyulfSchema.from_columns(["a"])
    actual = SkyulfSchema.from_columns(["a", "extra"])
    with pytest.raises(SchemaMismatchError) as exc_info:
        expected.assert_compatible(actual)
    assert exc_info.value.unexpected == ["extra"]


def test_assert_compatible_dtype_mismatch_when_checked():
    """check_dtypes=True should surface dtype mismatches for shared columns."""
    expected = SkyulfSchema.from_columns(["a"], {"a": "int64"})
    actual = SkyulfSchema.from_columns(["a"], {"a": "float64"})
    with pytest.raises(SchemaMismatchError) as exc_info:
        expected.assert_compatible(actual, check_dtypes=True)
    assert exc_info.value.dtype_mismatches == {"a": ("int64", "float64")}


def test_assert_compatible_ignores_dtype_mismatch_by_default():
    """Without check_dtypes, differing dtype labels on shared columns should be ignored."""
    expected = SkyulfSchema.from_columns(["a"], {"a": "int64"})
    actual = SkyulfSchema.from_columns(["a"], {"a": "float64"})
    expected.assert_compatible(actual)  # should not raise


def test_assert_compatible_order_mismatch_when_checked():
    """check_order=True should flag a different relative column order."""
    expected = SkyulfSchema.from_columns(["a", "b"])
    actual = SkyulfSchema.from_columns(["b", "a"])
    with pytest.raises(SchemaMismatchError) as exc_info:
        expected.assert_compatible(actual, check_order=True)
    assert exc_info.value.order_mismatch is True


def test_assert_compatible_order_check_skipped_when_columns_missing():
    """Order checking should be skipped (not double-raise) if columns are missing."""
    expected = SkyulfSchema.from_columns(["a", "b"])
    actual = SkyulfSchema.from_columns(["b"])
    with pytest.raises(SchemaMismatchError) as exc_info:
        expected.assert_compatible(actual, check_order=True)
    assert exc_info.value.order_mismatch is False


def test_validate_schema_accepts_raw_dataframe():
    """validate_schema should build a SkyulfSchema from a raw DataFrame and validate it."""
    df = pd.DataFrame({"a": [1, 2]})
    expected = SkyulfSchema.from_columns(["a"])
    validate_schema(expected, df)  # should not raise


def test_validate_schema_accepts_schema_instance_directly():
    """validate_schema should accept an already-built SkyulfSchema as `actual`."""
    expected = SkyulfSchema.from_columns(["a"])
    actual = SkyulfSchema.from_columns(["a"])
    validate_schema(expected, actual)  # should not raise


def test_validate_schema_raises_schema_mismatch_error():
    """validate_schema should raise SchemaMismatchError when columns differ."""
    df = pd.DataFrame({"z": [1, 2]})
    expected = SkyulfSchema.from_columns(["a"])
    with pytest.raises(SchemaMismatchError):
        validate_schema(expected, df)


def test_extract_pandas_dtypes_handles_broken_dtypes_object():
    """_extract_pandas_dtypes should gracefully handle an object whose dtypes.items() raises."""
    from skyulf.core.schema import _extract_pandas_dtypes

    class _BrokenDtypes:
        def items(self):
            raise RuntimeError("broken")

    class _BrokenFrame:
        dtypes = _BrokenDtypes()
        columns = ["a"]

    assert _extract_pandas_dtypes(_BrokenFrame()) == {}


def test_extract_polars_dtypes_handles_broken_schema_object():
    """_extract_polars_dtypes should gracefully handle a schema whose items() raises."""
    from skyulf.core.schema import _extract_polars_dtypes

    class _BrokenSchema:
        def items(self):
            raise RuntimeError("broken")

    class _BrokenFrame:
        schema = _BrokenSchema()

    assert _extract_polars_dtypes(_BrokenFrame()) == {}


def test_extract_polars_dtypes_none_schema_returns_empty():
    """_extract_polars_dtypes should return {} when the frame has no `schema` attribute value."""
    from skyulf.core.schema import _extract_polars_dtypes

    class _NoSchemaFrame:
        schema = None

    assert _extract_polars_dtypes(_NoSchemaFrame()) == {}


def test_extract_polars_dtypes_direct_call_on_real_polars_frame():
    """_extract_polars_dtypes should correctly extract dtype labels from a real polars schema."""
    from skyulf.core.schema import _extract_polars_dtypes

    df = pl.DataFrame({"a": [1, 2], "b": [1.0, 2.0]})
    dtypes = _extract_polars_dtypes(df)
    assert set(dtypes.keys()) == {"a", "b"}
