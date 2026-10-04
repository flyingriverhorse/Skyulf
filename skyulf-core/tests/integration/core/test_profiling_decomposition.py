"""Tests for skyulf.profiling._analyzer.decomposition.DecompositionMixin."""

from typing import Any

import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.profiling.analyzer import EDAAnalyzer

_decomposition_split_cases = TestCaseLoader("profiling/decomposition_split_scenarios").load()


def _decomposition_df() -> pl.DataFrame:
    """Small dataset used across decomposition split tests."""
    return pl.DataFrame(
        {
            "region": ["north", "north", "south", "south", "east", None],
            "sales": [100, 200, 50, 75, 300, 10],
        }
    )


class TestGlobalAggregateScenarios:
    """Global (no ``split_col``) aggregate scenarios over the shared ``sales`` fixture —
    scenarios loaded from ``tests/test_cases/profiling/decomposition_split_scenarios.json``.
    """

    @pytest.mark.parametrize(*_decomposition_split_cases)
    def test_get_decomposition_split_global_scenario(
        self,
        measure_col: str | None,
        measure_agg: str,
        split_col: str | None,
        filters: list[dict[str, Any]],
        expected_value: float,
    ) -> None:
        analyzer = EDAAnalyzer(_decomposition_df())
        result = analyzer.get_decomposition_split(
            measure_col=measure_col, measure_agg=measure_agg, split_col=split_col, filters=filters
        )
        assert result == [{"name": "Total", "value": expected_value, "ratio": 1.0}]


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


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing ``income`` values and a ``city`` column with an empty
    slot — closer to production data than the small synthetic frames used
    elsewhere in this file.
    """

    def test_income_sum_split_by_city_handles_missing_values(self) -> None:
        df = pl.from_pandas(load_sample_dataset("customers"))
        analyzer = EDAAnalyzer(df)
        result = analyzer.get_decomposition_split(
            measure_col="income", measure_agg="sum", split_col="city", filters=[]
        )
        by_name = {row["name"]: row["value"] for row in result}

        # The blank city cell (row 6) surfaces as "Unknown", not a silently dropped row.
        assert "Unknown" in by_name
        # Sum of income for New York rows (52000 + 94000 + 83000 + None) ignoring nulls.
        assert by_name["New York"] == pytest.approx(52000 + 94000 + 83000)
        total_ratio = sum(row["ratio"] for row in result)
        assert abs(total_ratio - 1.0) < 1e-9
