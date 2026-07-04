"""Tests for skyulf.profiling._analyzer.decomposition.DecompositionMixin."""

import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer


def _decomposition_df() -> pl.DataFrame:
    """Small dataset used across decomposition split tests."""
    return pl.DataFrame(
        {
            "region": ["north", "north", "south", "south", "east", None],
            "sales": [100, 200, 50, 75, 300, 10],
        }
    )


def test_get_decomposition_split_global_sum() -> None:
    """No split column: measure_agg='sum' should return a single 'Total' row."""
    analyzer = EDAAnalyzer(_decomposition_df())
    result = analyzer.get_decomposition_split(
        measure_col="sales", measure_agg="sum", split_col=None, filters=[]
    )
    assert result == [{"name": "Total", "value": 735, "ratio": 1.0}]


def test_get_decomposition_split_global_count_without_measure() -> None:
    """No measure_col and no split_col: falls back to row-count."""
    analyzer = EDAAnalyzer(_decomposition_df())
    result = analyzer.get_decomposition_split(
        measure_col=None, measure_agg="sum", split_col=None, filters=[]
    )
    assert result == [{"name": "Total", "value": 6, "ratio": 1.0}]


def test_get_decomposition_split_group_by_sum_and_ratio() -> None:
    """Split by region with sum aggregation; ratios should sum to ~1.0."""
    analyzer = EDAAnalyzer(_decomposition_df())
    result = analyzer.get_decomposition_split(
        measure_col="sales", measure_agg="sum", split_col="region", filters=[]
    )
    names = {row["name"] for row in result}
    # Nulls in the split column surface as the "Unknown" bucket.
    assert "Unknown" in names
    assert "north" in names
    total_ratio = sum(row["ratio"] for row in result)
    assert abs(total_ratio - 1.0) < 1e-9


def test_get_decomposition_split_group_by_mean() -> None:
    """measure_agg='mean' should average sales within each region."""
    analyzer = EDAAnalyzer(_decomposition_df())
    result = analyzer.get_decomposition_split(
        measure_col="sales", measure_agg="mean", split_col="region", filters=[]
    )
    by_name = {row["name"]: row["value"] for row in result}
    assert by_name["north"] == 150.0


def test_get_decomposition_split_group_by_min_max() -> None:
    """measure_agg='min'/'max' should be honored per-group."""
    analyzer = EDAAnalyzer(_decomposition_df())
    min_result = analyzer.get_decomposition_split(
        measure_col="sales", measure_agg="min", split_col="region", filters=[]
    )
    max_result = analyzer.get_decomposition_split(
        measure_col="sales", measure_agg="max", split_col="region", filters=[]
    )
    min_by_name = {row["name"]: row["value"] for row in min_result}
    max_by_name = {row["name"]: row["value"] for row in max_result}
    assert min_by_name["north"] == 100
    assert max_by_name["north"] == 200


def test_get_decomposition_split_applies_numeric_filter_as_string() -> None:
    """Numeric filter values arriving as strings (from the frontend) should be coerced."""
    analyzer = EDAAnalyzer(_decomposition_df())
    result = analyzer.get_decomposition_split(
        measure_col="sales",
        measure_agg="sum",
        split_col=None,
        filters=[{"column": "sales", "operator": ">", "value": "100"}],
    )
    assert result == [{"name": "Total", "value": 500, "ratio": 1.0}]


def test_get_decomposition_split_applies_float_filter_value_from_string() -> None:
    """Numeric string filter values should be coerced to ``float`` for Float columns."""
    df = pl.DataFrame(
        {
            "region": ["north", "north", "south", "south"],
            "revenue": [100.5, 200.25, 50.0, 75.75],
        }
    )
    analyzer = EDAAnalyzer(df)
    result = analyzer.get_decomposition_split(
        measure_col="revenue",
        measure_agg="sum",
        split_col=None,
        filters=[{"column": "revenue", "operator": ">", "value": "100.0"}],
    )
    assert result == [{"name": "Total", "value": pytest.approx(300.75), "ratio": 1.0}]


def test_get_decomposition_split_unknown_filter_matches_nulls() -> None:
    """The FE sentinel value 'Unknown' should filter on null/not-null for numeric columns."""
    analyzer = EDAAnalyzer(_decomposition_df())
    result = analyzer.get_decomposition_split(
        measure_col=None,
        measure_agg="sum",
        split_col=None,
        filters=[{"column": "sales", "operator": "==", "value": "Unknown"}],
    )
    # No nulls in the numeric "sales" column, so this filters everything out.
    assert result == [{"name": "Total", "value": 0, "ratio": 1.0}]


def test_get_decomposition_split_missing_column_returns_empty() -> None:
    """An unknown split/measure column should yield an empty list rather than raising."""
    analyzer = EDAAnalyzer(_decomposition_df())
    assert (
        analyzer.get_decomposition_split(
            measure_col="sales", measure_agg="sum", split_col="does_not_exist", filters=[]
        )
        == []
    )
    assert (
        analyzer.get_decomposition_split(
            measure_col="does_not_exist", measure_agg="sum", split_col=None, filters=[]
        )
        == []
    )


def test_get_decomposition_split_filters_are_ignored_for_unknown_columns() -> None:
    """Filters referencing a non-existent column should simply be skipped."""
    analyzer = EDAAnalyzer(_decomposition_df())
    result = analyzer.get_decomposition_split(
        measure_col="sales",
        measure_agg="sum",
        split_col=None,
        filters=[{"column": "does_not_exist", "operator": "==", "value": 1}],
    )
    assert result == [{"name": "Total", "value": 735, "ratio": 1.0}]


def test_get_decomposition_split_unknown_filter_not_equal_matches_non_nulls() -> None:
    """The 'Unknown' sentinel with '!=' should filter to non-null rows for numeric columns."""
    analyzer = EDAAnalyzer(_decomposition_df())
    result = analyzer.get_decomposition_split(
        measure_col=None,
        measure_agg="sum",
        split_col=None,
        filters=[{"column": "sales", "operator": "!=", "value": "Unknown"}],
    )
    # No nulls in "sales" so every row survives the not-null filter.
    assert result == [{"name": "Total", "value": 6, "ratio": 1.0}]


def test_get_decomposition_split_numeric_filter_value_not_castable_falls_back_to_string() -> None:
    """A non-numeric string value for a numeric column should fall back to a string cast."""
    analyzer = EDAAnalyzer(_decomposition_df())
    result = analyzer.get_decomposition_split(
        measure_col=None,
        measure_agg="sum",
        split_col=None,
        filters=[{"column": "sales", "operator": "==", "value": "not-a-number"}],
    )
    # Casting "sales" to Utf8 means no row will ever equal "not-a-number".
    assert result == [{"name": "Total", "value": 0, "ratio": 1.0}]


def test_get_decomposition_split_equality_and_inequality_filters() -> None:
    """The '==' and '!=' operators should filter rows exactly as expected."""
    analyzer = EDAAnalyzer(_decomposition_df())
    eq_result = analyzer.get_decomposition_split(
        measure_col="sales",
        measure_agg="sum",
        split_col=None,
        filters=[{"column": "region", "operator": "==", "value": "north"}],
    )
    assert eq_result == [{"name": "Total", "value": 300, "ratio": 1.0}]

    ne_result = analyzer.get_decomposition_split(
        measure_col="sales",
        measure_agg="sum",
        split_col=None,
        filters=[{"column": "region", "operator": "!=", "value": "north"}],
    )
    # The null-region row compares as null (falsy) against "!= north", so it's excluded too.
    assert ne_result == [{"name": "Total", "value": 425, "ratio": 1.0}]


def test_get_decomposition_split_less_than_and_comparison_filters() -> None:
    """The '<', '>=', '<=' comparison operators should each filter correctly."""
    analyzer = EDAAnalyzer(_decomposition_df())

    lt_result = analyzer.get_decomposition_split(
        measure_col="sales",
        measure_agg="sum",
        split_col=None,
        filters=[{"column": "sales", "operator": "<", "value": 100}],
    )
    assert lt_result == [{"name": "Total", "value": 135, "ratio": 1.0}]

    ge_result = analyzer.get_decomposition_split(
        measure_col="sales",
        measure_agg="sum",
        split_col=None,
        filters=[{"column": "sales", "operator": ">=", "value": 100}],
    )
    assert ge_result == [{"name": "Total", "value": 600, "ratio": 1.0}]

    le_result = analyzer.get_decomposition_split(
        measure_col="sales",
        measure_agg="sum",
        split_col=None,
        filters=[{"column": "sales", "operator": "<=", "value": 100}],
    )
    assert le_result == [{"name": "Total", "value": 235, "ratio": 1.0}]


def test_get_decomposition_split_in_operator_filter() -> None:
    """The 'in' operator should keep rows whose value is in the provided list."""
    analyzer = EDAAnalyzer(_decomposition_df())
    result = analyzer.get_decomposition_split(
        measure_col="sales",
        measure_agg="sum",
        split_col=None,
        filters=[{"column": "region", "operator": "in", "value": ["north", "south"]}],
    )
    assert result == [{"name": "Total", "value": 425, "ratio": 1.0}]


def test_get_decomposition_split_global_mean_min_max_and_unknown_agg() -> None:
    """Global (non-split) aggregation should honor mean/min/max and default on unknown agg."""
    analyzer = EDAAnalyzer(_decomposition_df())

    mean_result = analyzer.get_decomposition_split(
        measure_col="sales", measure_agg="mean", split_col=None, filters=[]
    )
    assert mean_result == [{"name": "Total", "value": pytest.approx(735 / 6), "ratio": 1.0}]

    min_result = analyzer.get_decomposition_split(
        measure_col="sales", measure_agg="min", split_col=None, filters=[]
    )
    assert min_result == [{"name": "Total", "value": 10, "ratio": 1.0}]

    max_result = analyzer.get_decomposition_split(
        measure_col="sales", measure_agg="max", split_col=None, filters=[]
    )
    assert max_result == [{"name": "Total", "value": 300, "ratio": 1.0}]

    unknown_agg_result = analyzer.get_decomposition_split(
        measure_col="sales", measure_agg="median", split_col=None, filters=[]
    )
    # Unknown agg name falls back to row count.
    assert unknown_agg_result == [{"name": "Total", "value": 6, "ratio": 1.0}]


def test_get_decomposition_split_global_aggregate_none_becomes_zero() -> None:
    """A global aggregate over an all-null column should coerce ``None`` to 0."""
    df = pl.DataFrame({"region": ["a", "b"], "sales": pl.Series([None, None], dtype=pl.Int64)})
    analyzer = EDAAnalyzer(df)
    result = analyzer.get_decomposition_split(
        measure_col="sales", measure_agg="mean", split_col=None, filters=[]
    )
    assert result == [{"name": "Total", "value": 0, "ratio": 1.0}]


def test_get_decomposition_split_group_by_no_measure_uses_row_count() -> None:
    """Splitting by a column without a measure_col should count rows per group."""
    analyzer = EDAAnalyzer(_decomposition_df())
    result = analyzer.get_decomposition_split(
        measure_col=None, measure_agg="sum", split_col="region", filters=[]
    )
    by_name = {row["name"]: row["value"] for row in result}
    assert by_name["north"] == 2
    assert by_name["south"] == 2
    assert by_name["Unknown"] == 1


def test_get_decomposition_split_group_by_missing_measure_col_returns_empty() -> None:
    """A split_col that exists but a measure_col that doesn't should return an empty list."""
    analyzer = EDAAnalyzer(_decomposition_df())
    result = analyzer.get_decomposition_split(
        measure_col="does_not_exist", measure_agg="sum", split_col="region", filters=[]
    )
    assert result == []


def test_get_decomposition_split_group_by_unknown_agg_falls_back_to_count() -> None:
    """An unrecognized measure_agg with a split_col should fall back to a row-count aggregate."""
    analyzer = EDAAnalyzer(_decomposition_df())
    result = analyzer.get_decomposition_split(
        measure_col="sales", measure_agg="median", split_col="region", filters=[]
    )
    by_name = {row["name"]: row["value"] for row in result}
    assert by_name["north"] == 2
    assert by_name["south"] == 2


def test_get_decomposition_split_group_by_zero_total_gives_zero_ratios() -> None:
    """When the aggregated total is exactly 0, ratios should default to 0.0 (no div-by-zero)."""
    df = pl.DataFrame({"region": ["a", "a", "b"], "sales": [0, 0, 0]})
    analyzer = EDAAnalyzer(df)
    result = analyzer.get_decomposition_split(
        measure_col="sales", measure_agg="sum", split_col="region", filters=[]
    )
    assert all(row["ratio"] == 0.0 for row in result)
