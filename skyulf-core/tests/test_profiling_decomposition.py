"""Tests for skyulf.profiling._analyzer.decomposition.DecompositionMixin."""

import polars as pl

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
