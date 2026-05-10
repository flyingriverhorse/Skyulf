"""
Comprehensive tests for FeatureGenerationApplier — covers all operation types
on both the pandas and polars engine paths.

Run with:
    pytest skyulf-core/tests/test_feature_generation_full.py -v
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from skyulf.preprocessing.feature_generation import (
    FeatureGenerationApplier,
    FeatureGenerationCalculator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_APPLIER = FeatureGenerationApplier()
_CALC = FeatureGenerationCalculator()

DATE_COL = "ts"
DATES = ["2024-01-15", "2024-07-04", "2023-12-31", "2022-03-07"]


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(DATES),
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "name_a": ["hello world", "foo bar", "test", ""],
            "name_b": ["hello", "foo", "test data", "xyz"],
        }
    )


def _run(ops: list, df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = _make_df()
    params = _CALC.fit(df, {"operations": ops})
    return _APPLIER.apply(df, params)


def _run_polars(ops: list, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run the same ops through the polars engine and return a pandas result."""
    import polars as pl

    if df is None:
        df = _make_df()
    df_pl = pl.from_pandas(df)
    params = _CALC.fit(df_pl, {"operations": ops})
    out = _APPLIER.apply(df_pl, params)
    if hasattr(out, "to_pandas"):
        return out.to_pandas()
    return out


# ---------------------------------------------------------------------------
# 1. Arithmetic operations — pandas
# ---------------------------------------------------------------------------


class TestArithmeticPandas:
    def test_add(self) -> None:
        out = _run([{"operation_type": "arithmetic", "method": "add", "input_columns": ["a", "b"]}])
        assert "arithmetic_0" in out.columns
        pd.testing.assert_series_equal(
            out["arithmetic_0"].reset_index(drop=True),
            (out["a"] + out["b"]).reset_index(drop=True),
            check_names=False,
        )

    def test_subtract(self) -> None:
        out = _run(
            [{"operation_type": "arithmetic", "method": "subtract", "input_columns": ["a", "b"]}]
        )
        pd.testing.assert_series_equal(
            out["arithmetic_0"].reset_index(drop=True),
            (out["a"] - out["b"]).reset_index(drop=True),
            check_names=False,
        )

    def test_multiply(self) -> None:
        out = _run(
            [{"operation_type": "arithmetic", "method": "multiply", "input_columns": ["a", "b"]}]
        )
        pd.testing.assert_series_equal(
            out["arithmetic_0"].reset_index(drop=True),
            (out["a"] * out["b"]).reset_index(drop=True),
            check_names=False,
        )

    def test_divide(self) -> None:
        out = _run(
            [{"operation_type": "arithmetic", "method": "divide", "input_columns": ["a", "b"]}]
        )
        expected = out["a"] / out["b"]
        pd.testing.assert_series_equal(
            out["arithmetic_0"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_divide_by_zero_safe(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 0.0]})
        out = _run(
            [{"operation_type": "arithmetic", "method": "divide", "input_columns": ["a", "b"]}],
            df=df,
        )
        # Should not raise and should not be NaN/inf
        assert out["arithmetic_0"].notna().all()

    def test_add_with_constant(self) -> None:
        out = _run(
            [
                {
                    "operation_type": "arithmetic",
                    "method": "add",
                    "input_columns": ["a"],
                    "constants": [100],
                }
            ]
        )
        assert (out["arithmetic_0"] == out["a"] + 100).all()

    def test_custom_output_name(self) -> None:
        out = _run(
            [
                {
                    "operation_type": "arithmetic",
                    "method": "add",
                    "input_columns": ["a", "b"],
                    "output_column": "sum_ab",
                }
            ]
        )
        assert "sum_ab" in out.columns


# ---------------------------------------------------------------------------
# 2. Ratio — pandas
# ---------------------------------------------------------------------------


class TestRatioPandas:
    def test_ratio_basic(self) -> None:
        out = _run(
            [
                {
                    "operation_type": "ratio",
                    "input_columns": ["a"],
                    "secondary_columns": ["b"],
                }
            ]
        )
        assert "ratio_0" in out.columns
        expected = out["a"] / out["b"]
        pd.testing.assert_series_equal(
            out["ratio_0"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# 3. Similarity — pandas
# ---------------------------------------------------------------------------


class TestSimilarityPandas:
    def test_similarity_ratio(self) -> None:
        out = _run(
            [
                {
                    "operation_type": "similarity",
                    "method": "ratio",
                    "input_columns": ["name_a", "name_b"],
                }
            ]
        )
        assert "similarity_0" in out.columns
        scores = out["similarity_0"]
        # All scores must be in [0, 100]
        assert (scores >= 0).all() and (scores <= 100).all()

    def test_similarity_both_empty(self) -> None:
        df = pd.DataFrame({"x": ["", ""], "y": ["", ""]})
        out = _run(
            [{"operation_type": "similarity", "method": "ratio", "input_columns": ["x", "y"]}],
            df=df,
        )
        assert (out["similarity_0"] == 100.0).all()

    def test_similarity_one_empty(self) -> None:
        df = pd.DataFrame({"x": ["hello", ""], "y": ["", "world"]})
        out = _run(
            [{"operation_type": "similarity", "method": "ratio", "input_columns": ["x", "y"]}],
            df=df,
        )
        assert (out["similarity_0"] == 0.0).all()

    def test_similarity_identical(self) -> None:
        df = pd.DataFrame({"x": ["abc", "xyz"], "y": ["abc", "xyz"]})
        out = _run(
            [{"operation_type": "similarity", "method": "ratio", "input_columns": ["x", "y"]}],
            df=df,
        )
        assert (out["similarity_0"] == 100.0).all()


# ---------------------------------------------------------------------------
# 4. Datetime extraction — pandas (ALL features)
# ---------------------------------------------------------------------------


class TestDatetimeExtractPandas:
    @pytest.fixture(autouse=True)
    def _result(self) -> None:
        self.out = _run(
            [
                {
                    "operation_type": "datetime_extract",
                    "input_columns": [DATE_COL],
                    "datetime_features": [
                        "year",
                        "quarter",
                        "month",
                        "month_name",
                        "week",
                        "day",
                        "day_name",
                        "weekday",
                        "is_weekend",
                        "hour",
                        "minute",
                        "second",
                    ],
                }
            ]
        )

    def test_year(self) -> None:
        assert list(self.out[f"{DATE_COL}_year"]) == [2024, 2024, 2023, 2022]

    def test_quarter(self) -> None:
        # Jan → Q1, Jul → Q3, Dec → Q4, Mar → Q1
        assert list(self.out[f"{DATE_COL}_quarter"]) == [1, 3, 4, 1]

    def test_month_numeric(self) -> None:
        assert list(self.out[f"{DATE_COL}_month"]) == [1, 7, 12, 3]

    def test_month_name_strings(self) -> None:
        assert list(self.out[f"{DATE_COL}_month_name"]) == [
            "January",
            "July",
            "December",
            "March",
        ]

    def test_month_name_is_distinct_from_month(self) -> None:
        # month is int; month_name is str — they must not be equal
        assert f"{DATE_COL}_month" in self.out.columns
        assert f"{DATE_COL}_month_name" in self.out.columns
        assert self.out[f"{DATE_COL}_month"].dtype != object  # numeric
        assert self.out[f"{DATE_COL}_month_name"].dtype == object  # string

    def test_week_iso(self) -> None:
        # ISO week for 2024-01-15 = 3, 2024-07-04 = 27, 2023-12-31 = 52, 2022-03-07 = 10
        assert list(self.out[f"{DATE_COL}_week"]) == [3, 27, 52, 10]

    def test_day_of_month(self) -> None:
        assert list(self.out[f"{DATE_COL}_day"]) == [15, 4, 31, 7]

    def test_day_name_strings(self) -> None:
        assert list(self.out[f"{DATE_COL}_day_name"]) == ["Monday", "Thursday", "Sunday", "Monday"]

    def test_day_name_is_distinct_from_day(self) -> None:
        assert f"{DATE_COL}_day" in self.out.columns
        assert f"{DATE_COL}_day_name" in self.out.columns
        assert self.out[f"{DATE_COL}_day"].dtype != object
        assert self.out[f"{DATE_COL}_day_name"].dtype == object

    def test_weekday_zero_indexed(self) -> None:
        # 2024-01-15 = Monday = 0; 2024-07-04 = Thursday = 3
        # 2023-12-31 = Sunday = 6; 2022-03-07 = Monday = 0
        assert list(self.out[f"{DATE_COL}_weekday"]) == [0, 3, 6, 0]

    def test_is_weekend(self) -> None:
        # Mon=0, Thu=0, Sun=1, Mon=0
        assert list(self.out[f"{DATE_COL}_is_weekend"]) == [0, 0, 1, 0]

    def test_hour_is_zero_for_date_only(self) -> None:
        assert (self.out[f"{DATE_COL}_hour"] == 0).all()

    def test_no_columns_silently_missing(self) -> None:
        expected_feats = [
            "year",
            "quarter",
            "month",
            "month_name",
            "week",
            "day",
            "day_name",
            "weekday",
            "is_weekend",
            "hour",
            "minute",
            "second",
        ]
        for feat in expected_feats:
            col = f"{DATE_COL}_{feat}"
            assert col in self.out.columns, f"Column '{col}' missing from output"


# ---------------------------------------------------------------------------
# 5. Datetime extraction — polars (all features, engine parity check)
# ---------------------------------------------------------------------------


class TestDatetimeExtractPolars:
    @pytest.fixture(autouse=True)
    def _result(self) -> None:
        try:
            import polars as pl  # noqa: F401
        except ImportError:
            pytest.skip("polars not installed")  # ty: ignore[too-many-positional-arguments]
        self.out_pd = _run(
            [
                {
                    "operation_type": "datetime_extract",
                    "input_columns": [DATE_COL],
                    "datetime_features": [
                        "year",
                        "quarter",
                        "month",
                        "month_name",
                        "week",
                        "day",
                        "day_name",
                        "weekday",
                        "is_weekend",
                        "hour",
                    ],
                }
            ]
        )
        self.out_pl = _run_polars(
            [
                {
                    "operation_type": "datetime_extract",
                    "input_columns": [DATE_COL],
                    "datetime_features": [
                        "year",
                        "quarter",
                        "month",
                        "month_name",
                        "week",
                        "day",
                        "day_name",
                        "weekday",
                        "is_weekend",
                        "hour",
                    ],
                }
            ]
        )

    def test_year_parity(self) -> None:
        pd.testing.assert_series_equal(
            self.out_pd[f"{DATE_COL}_year"].reset_index(drop=True),
            self.out_pl[f"{DATE_COL}_year"].reset_index(drop=True),
            check_dtype=False,
        )

    def test_month_parity(self) -> None:
        pd.testing.assert_series_equal(
            self.out_pd[f"{DATE_COL}_month"].reset_index(drop=True),
            self.out_pl[f"{DATE_COL}_month"].reset_index(drop=True),
            check_dtype=False,
        )

    def test_month_name_parity(self) -> None:
        assert list(self.out_pd[f"{DATE_COL}_month_name"]) == list(
            self.out_pl[f"{DATE_COL}_month_name"]
        )

    def test_week_parity(self) -> None:
        pd.testing.assert_series_equal(
            self.out_pd[f"{DATE_COL}_week"].reset_index(drop=True),
            self.out_pl[f"{DATE_COL}_week"].reset_index(drop=True),
            check_dtype=False,
        )

    def test_day_parity(self) -> None:
        pd.testing.assert_series_equal(
            self.out_pd[f"{DATE_COL}_day"].reset_index(drop=True),
            self.out_pl[f"{DATE_COL}_day"].reset_index(drop=True),
            check_dtype=False,
        )

    def test_day_name_parity(self) -> None:
        assert list(self.out_pd[f"{DATE_COL}_day_name"]) == list(
            self.out_pl[f"{DATE_COL}_day_name"]
        )

    def test_weekday_parity(self) -> None:
        pd.testing.assert_series_equal(
            self.out_pd[f"{DATE_COL}_weekday"].reset_index(drop=True),
            self.out_pl[f"{DATE_COL}_weekday"].reset_index(drop=True),
            check_dtype=False,
        )

    def test_is_weekend_parity(self) -> None:
        pd.testing.assert_series_equal(
            self.out_pd[f"{DATE_COL}_is_weekend"].reset_index(drop=True),
            self.out_pl[f"{DATE_COL}_is_weekend"].reset_index(drop=True),
            check_dtype=False,
        )

    def test_polars_no_columns_missing(self) -> None:
        feats = [
            "year",
            "quarter",
            "month",
            "month_name",
            "week",
            "day",
            "day_name",
            "weekday",
            "is_weekend",
            "hour",
        ]
        for feat in feats:
            col = f"{DATE_COL}_{feat}"
            assert col in self.out_pl.columns, f"Polars: column '{col}' missing"


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_operations_returns_unchanged(self) -> None:
        df = _make_df()
        out = _run([], df=df)
        pd.testing.assert_frame_equal(out, df)

    def test_missing_column_skipped_gracefully(self) -> None:
        """Operations on non-existent columns should not raise."""
        out = _run(
            [
                {
                    "operation_type": "arithmetic",
                    "method": "add",
                    "input_columns": ["nonexistent_col"],
                }
            ]
        )
        # Original columns intact; bad op skipped
        assert "a" in out.columns

    def test_round_digits(self) -> None:
        out = _run(
            [
                {
                    "operation_type": "ratio",
                    "input_columns": ["a"],
                    "secondary_columns": ["b"],
                    "round_digits": 2,
                }
            ]
        )
        # All values should have at most 2 decimal places
        for v in out["ratio_0"]:
            assert v == round(v, 2)

    def test_no_overwrite_without_flag(self) -> None:
        df = _make_df()
        # First op creates 'my_col'; second op tries the same name → should get 'my_col_1'
        out = _run(
            [
                {
                    "operation_type": "arithmetic",
                    "method": "add",
                    "input_columns": ["a", "b"],
                    "output_column": "my_col",
                },
                {
                    "operation_type": "arithmetic",
                    "method": "subtract",
                    "input_columns": ["a", "b"],
                    "output_column": "my_col",
                },
            ],
            df=df,
        )
        assert "my_col" in out.columns
        assert "my_col_1" in out.columns

    def test_tuple_input_passthrough(self) -> None:
        """(X, y) tuple inputs should be preserved."""
        X = _make_df()
        y = pd.Series([0, 1, 0, 1])
        params = _CALC.fit((X, y), {"operations": []})
        result = _APPLIER.apply((X, y), params)
        assert isinstance(result, tuple)
        X_out, y_out = result
        assert list(y_out) == [0, 1, 0, 1]

    def test_datetime_string_column_polars(self) -> None:
        """Polars path should cast string columns to datetime."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("polars not installed")  # ty: ignore[too-many-positional-arguments]
        df_pl = pl.DataFrame({"dt_str": ["2024-01-15", "2024-07-04", "2023-12-31", "2022-03-07"]})
        params = _CALC.fit(
            df_pl,
            {
                "operations": [
                    {
                        "operation_type": "datetime_extract",
                        "input_columns": ["dt_str"],
                        "datetime_features": ["year", "month"],
                    }
                ]
            },
        )
        out = _APPLIER.apply(df_pl, params)
        assert "dt_str_year" in out.columns
        assert "dt_str_month" in out.columns

    def test_nan_similarity_treated_as_empty(self) -> None:
        df = pd.DataFrame({"x": [None, "hello"], "y": [None, "hello"]})
        out = _run(
            [{"operation_type": "similarity", "method": "ratio", "input_columns": ["x", "y"]}],
            df=df,
        )
        # None+None → both empty → 100
        assert out["similarity_0"].iloc[0] == 100.0
        # "hello"+"hello" → identical → 100
        assert out["similarity_0"].iloc[1] == 100.0

    def test_datetime_nan_row_doesnt_crash(self) -> None:
        df = pd.DataFrame(
            {
                "dt": pd.to_datetime(
                    ["2024-01-15", None, "2023-12-31", "2022-03-07"], errors="coerce"
                )
            }
        )
        out = _run(
            [
                {
                    "operation_type": "datetime_extract",
                    "input_columns": ["dt"],
                    "datetime_features": ["year", "month", "month_name"],
                }
            ],
            df=df,
        )
        # Non-null rows must be correct; NaT row can be NaN
        assert out["dt_year"].iloc[0] == 2024
        assert out["dt_year"].iloc[2] == 2023
        assert math.isnan(out["dt_year"].iloc[1])
