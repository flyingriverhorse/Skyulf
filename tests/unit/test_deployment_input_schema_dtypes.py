"""The deployment input schema must report real column types, not a literal "unknown".

The Inference page renders one chip per input column with its type. Every chip
read "unknown" because the schema builder hardcoded that string, which told users
nothing about what a column expects. Types are captured at training time (so
engineered columns such as ``*_was_missing`` are covered too) and mapped to a
short display label here.
"""

import pandas as pd
import pytest

from backend.ml_pipeline.deployment.service import DeploymentService


class TestPrettyDtype:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("int64", "integer"),
            ("Int32", "integer"),
            ("float64", "float"),
            ("Float32", "float"),
            ("bool", "boolean"),
            ("boolean", "boolean"),
            ("object", "text"),
            ("string", "text"),
            ("category", "category"),
            ("datetime64[ns]", "datetime"),
            ("timedelta64[ns]", "duration"),
            # Polars dtype names (captured as ``str(dtype)`` from the schema)
            ("Date", "date"),
            ("Datetime('us')", "datetime"),
            ("Duration('us')", "duration"),
            ("Categorical", "category"),
            ("Enum(categories: ['a', 'b'])", "category"),
            ("String", "text"),
            ("UInt8", "integer"),
        ],
    )
    def test_maps_known_dtypes_to_display_labels(self, raw, expected):
        assert DeploymentService._pretty_dtype(raw) == expected

    def test_falls_back_to_unknown_for_unrecognised_dtype(self):
        assert DeploymentService._pretty_dtype("complex128") == "unknown"

    def test_falls_back_to_unknown_for_missing_dtype(self):
        assert DeploymentService._pretty_dtype(None) == "unknown"


class TestInputSchemaEntries:
    def test_uses_recorded_dtype_per_column(self):
        entries = DeploymentService._input_schema_entries(
            ["Id", "SepalLengthCm", "SepalLengthCm_was_missing"],
            {"Id": "int64", "SepalLengthCm": "float64", "SepalLengthCm_was_missing": "bool"},
        )
        assert entries == [
            {"name": "Id", "type": "integer"},
            {"name": "SepalLengthCm", "type": "float"},
            {"name": "SepalLengthCm_was_missing", "type": "boolean"},
        ]

    def test_reports_unknown_when_artifact_predates_dtype_capture(self):
        entries = DeploymentService._input_schema_entries(["Id"], None)
        assert entries == [{"name": "Id", "type": "unknown"}]

    def test_reports_unknown_for_a_column_missing_from_the_dtype_map(self):
        entries = DeploymentService._input_schema_entries(["Id", "New"], {"Id": "int64"})
        assert entries == [
            {"name": "Id", "type": "integer"},
            {"name": "New", "type": "unknown"},
        ]

    def test_preserves_feature_column_order(self):
        names = ["c", "a", "b"]
        entries = DeploymentService._input_schema_entries(names, {})
        assert [e["name"] for e in entries] == names


class TestFrameDtypeCapture:
    def test_captures_pandas_dtypes_as_strings(self):
        df = pd.DataFrame(
            {"Id": [1, 2], "Sepal": [1.0, 2.0], "flag": [True, False], "name": ["a", "b"]}
        )

        captured = DeploymentService._dtypes_for_columns(df, ["Id", "Sepal", "flag", "name"])

        assert DeploymentService._pretty_dtype(captured["Id"]) == "integer"
        assert DeploymentService._pretty_dtype(captured["Sepal"]) == "float"
        assert DeploymentService._pretty_dtype(captured["flag"]) == "boolean"
        assert DeploymentService._pretty_dtype(captured["name"]) == "text"

    def test_skips_columns_absent_from_the_frame(self):
        df = pd.DataFrame({"Id": [1, 2]})
        assert DeploymentService._dtypes_for_columns(df, ["Id", "Ghost"]) == {"Id": "int64"}

    def test_captures_polars_dtypes_and_maps_them_to_labels(self):
        import datetime

        import polars as pl

        df = pl.DataFrame(
            {
                "Id": [1, 2],
                "Sepal": [1.0, 2.0],
                "flag": [True, False],
                "name": ["a", "b"],
                "day": [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)],
            }
        )

        captured = DeploymentService._dtypes_for_columns(df, ["Id", "Sepal", "flag", "name", "day"])

        assert DeploymentService._pretty_dtype(captured["Id"]) == "integer"
        assert DeploymentService._pretty_dtype(captured["Sepal"]) == "float"
        assert DeploymentService._pretty_dtype(captured["flag"]) == "boolean"
        assert DeploymentService._pretty_dtype(captured["name"]) == "text"
        assert DeploymentService._pretty_dtype(captured["day"]) == "date"
