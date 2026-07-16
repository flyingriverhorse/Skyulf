"""Additional coverage tests for backend/data_ingestion/serialization.py.

Focused on the DataFrame-conversion helpers (Polars handling, pandas
truncation/cleaning, records/columns conversion) and the DataTypeConverter
column-type inference/conversion helpers, since these were split out during
an extract-method refactor and lost direct test coverage.
"""

import pandas as pd
import polars as pl
import pytest

from backend.data_ingestion.serialization import AsyncJSONSafeSerializer, DataTypeConverter


class TestHandlePolarsDataFrame:
    """Tests for AsyncJSONSafeSerializer._handle_polars_dataframe."""

    async def test_non_polars_object_is_not_handled(self):
        result = await AsyncJSONSafeSerializer._handle_polars_dataframe(
            pd.DataFrame({"a": [1]}), True, None
        )
        assert result is AsyncJSONSafeSerializer._NOT_HANDLED

    async def test_records_format(self):
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = await AsyncJSONSafeSerializer._handle_polars_dataframe(df, True, None)
        assert result == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    async def test_columns_format(self):
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = await AsyncJSONSafeSerializer._handle_polars_dataframe(df, False, None)
        assert result == {"a": [1, 2], "b": ["x", "y"]}

    async def test_max_rows_truncates(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4]})
        result = await AsyncJSONSafeSerializer._handle_polars_dataframe(df, True, 2)
        assert result == [{"a": 1}, {"a": 2}]


class TestPrepareDataFrameForSerialization:
    """Tests for AsyncJSONSafeSerializer._prepare_dataframe_for_serialization."""

    async def test_truncates_to_max_rows(self):
        df = pd.DataFrame({"a": range(10)})
        cleaned = await AsyncJSONSafeSerializer._prepare_dataframe_for_serialization(df, 3)
        assert len(cleaned) == 3

    async def test_no_truncation_when_within_max_rows(self):
        df = pd.DataFrame({"a": range(3)})
        cleaned = await AsyncJSONSafeSerializer._prepare_dataframe_for_serialization(df, 10)
        assert len(cleaned) == 3

    async def test_fills_nan_with_none(self):
        # Numeric columns keep NaN as a float sentinel here (pandas coerces None
        # back to NaN for numeric dtypes); the None-cleanup for JSON happens
        # downstream in clean_for_json/_handle_float. Object-dtype columns,
        # however, do retain a real None.
        df = pd.DataFrame({"a": [1.0, float("nan")], "b": pd.Series(["x", None], dtype="object")})
        cleaned = await AsyncJSONSafeSerializer._prepare_dataframe_for_serialization(df, None)
        assert pd.isna(cleaned["a"].iloc[1])
        assert cleaned["b"].iloc[1] is None

    async def test_yields_for_large_dataframe(self, monkeypatch):
        """Cover the branch where row count exceeds the yield threshold."""
        from backend.config import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "SERIALIZATION_YIELD_THRESHOLD_ROWS", 1)
        df = pd.DataFrame({"a": [1, 2, 3]})
        cleaned = await AsyncJSONSafeSerializer._prepare_dataframe_for_serialization(df, None)
        assert len(cleaned) == 3


class TestDataFrameToRecordsAndColumns:
    """Tests for _dataframe_to_records and _dataframe_to_columns."""

    async def test_dataframe_to_records(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        records = await AsyncJSONSafeSerializer._dataframe_to_records(df)
        assert records == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    async def test_dataframe_to_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        columns = await AsyncJSONSafeSerializer._dataframe_to_columns(df)
        assert columns == {"a": [1, 2], "b": ["x", "y"]}

    async def test_dataframe_to_columns_stringifies_column_names(self):
        df = pd.DataFrame({1: [1, 2]})
        columns = await AsyncJSONSafeSerializer._dataframe_to_columns(df)
        assert columns == {"1": [1, 2]}


class TestSafeDictFromDataFrame:
    """Tests for AsyncJSONSafeSerializer.safe_dict_from_dataframe."""

    async def test_polars_dataframe_dispatches_to_polars_handler(self):
        df = pl.DataFrame({"a": [1, 2]})
        result = await AsyncJSONSafeSerializer.safe_dict_from_dataframe(df, records_format=True)
        assert result == [{"a": 1}, {"a": 2}]

    async def test_empty_pandas_dataframe_records_format(self):
        result = await AsyncJSONSafeSerializer.safe_dict_from_dataframe(
            pd.DataFrame(), records_format=True
        )
        assert result == []

    async def test_empty_pandas_dataframe_columns_format(self):
        result = await AsyncJSONSafeSerializer.safe_dict_from_dataframe(
            pd.DataFrame(), records_format=False
        )
        assert result == {}

    async def test_pandas_dataframe_records_format(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = await AsyncJSONSafeSerializer.safe_dict_from_dataframe(df, records_format=True)
        assert result == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    async def test_pandas_dataframe_columns_format(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = await AsyncJSONSafeSerializer.safe_dict_from_dataframe(df, records_format=False)
        assert result == {"a": [1, 2], "b": ["x", "y"]}

    async def test_pandas_dataframe_max_rows(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        result = await AsyncJSONSafeSerializer.safe_dict_from_dataframe(
            df, records_format=True, max_rows=2
        )
        assert result == [{"a": 1}, {"a": 2}]


class TestInferObjectColumnType:
    """Tests for DataTypeConverter._infer_object_column_type."""

    def test_datetime_column(self):
        import datetime as dt

        series = pd.Series([dt.date(2024, 1, 1), dt.datetime(2024, 1, 2)], dtype="object")
        assert DataTypeConverter._infer_object_column_type(series) == "datetime"

    def test_numeric_string_column(self):
        series = pd.Series(["1", "2.5", "-3"], dtype="object")
        assert DataTypeConverter._infer_object_column_type(series) == "numeric_string"

    def test_boolean_string_column(self):
        series = pd.Series(["true", "False", "yes", "0"], dtype="object")
        assert DataTypeConverter._infer_object_column_type(series) == "boolean_string"

    def test_text_column(self):
        series = pd.Series(["hello", "world"], dtype="object")
        assert DataTypeConverter._infer_object_column_type(series) == "text"


class TestInferNonObjectColumnType:
    """Tests for DataTypeConverter._infer_non_object_column_type."""

    def test_integer_column(self):
        series = pd.Series([1, 2, 3], dtype="int64")
        assert DataTypeConverter._infer_non_object_column_type(series) == "integer"

    def test_float_column(self):
        series = pd.Series([1.0, 2.5], dtype="float64")
        assert DataTypeConverter._infer_non_object_column_type(series) == "float"

    def test_datetime_column(self):
        series = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))
        assert DataTypeConverter._infer_non_object_column_type(series) == "datetime"

    def test_boolean_column(self):
        # NOTE: pandas' is_numeric_dtype() returns True for bool dtype, so the
        # numeric branch is checked (and matches) before the boolean branch,
        # meaning bool columns are classified as "float" on this pandas
        # version. This reflects the existing (pre-existing, unrelated to
        # this test addition) behavior rather than the ideal classification.
        series = pd.Series([True, False], dtype="bool")
        assert DataTypeConverter._infer_non_object_column_type(series) == "float"

    def test_fallback_returns_dtype_str(self):
        series = pd.Series(["a", "b"], dtype="category")
        assert DataTypeConverter._infer_non_object_column_type(series) == str(series.dtype)


class TestInferColumnTypes:
    """Sanity check that infer_column_types dispatches to both helpers correctly."""

    def test_mixed_columns(self):
        df = pd.DataFrame({"nums": [1, 2, 3], "text": ["a", "b", "c"]})
        result = DataTypeConverter.infer_column_types(df)
        assert result == {"nums": "integer", "text": "text"}


class TestConvertColumn:
    """Tests for DataTypeConverter._convert_column."""

    def test_convert_to_integer(self):
        series = pd.Series(["1", "2", "3"])
        converted = DataTypeConverter._convert_column(series, "integer")
        assert converted.tolist() == [1, 2, 3]

    def test_convert_to_float(self):
        series = pd.Series(["1.5", "2.5"])
        converted = DataTypeConverter._convert_column(series, "float")
        assert converted.tolist() == pytest.approx([1.5, 2.5])

    def test_convert_to_datetime(self):
        series = pd.Series(["2024-01-01", "2024-01-02"])
        converted = DataTypeConverter._convert_column(series, "datetime")
        assert pd.api.types.is_datetime64_any_dtype(converted)

    def test_convert_to_boolean(self):
        series = pd.Series([True, False])
        converted = DataTypeConverter._convert_column(series, "boolean")
        assert converted.tolist() == [True, False]

    def test_convert_to_text(self):
        series = pd.Series([1, 2])
        converted = DataTypeConverter._convert_column(series, "text")
        assert converted.tolist() == ["1", "2"]

    def test_convert_unrecognized_type_returns_unchanged(self):
        series = pd.Series([1, 2, 3])
        converted = DataTypeConverter._convert_column(series, "unknown_type")
        assert converted is series
