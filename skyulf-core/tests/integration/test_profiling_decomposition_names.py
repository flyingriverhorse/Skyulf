"""Decomposition output fields must not reserve names in the user's dataset."""

import json

import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer


@pytest.mark.parametrize("split_col", ["value", "ratio", "name", "filter_value", "group"])
@pytest.mark.parametrize(
    "measure_agg, expected",
    [
        ("sum", {"a": 4, "b": 8, None: 4}),
        ("mean", {"a": 2, "b": 8, None: 4}),
        ("min", {"a": 1, "b": 8, None: 4}),
        ("max", {"a": 3, "b": 8, None: 4}),
        ("count", {"a": 2, "b": 1, None: 1}),
    ],
)
def test_output_field_names_preserve_group_identity_and_drill_down(
    split_col: str, measure_agg: str, expected: dict[str | None, int]
) -> None:
    """Names, aggregates and filters must still refer to the same observations."""
    source = pl.DataFrame({split_col: ["a", "a", "b", None], "amount": [1, 3, 8, 4]})
    analyzer = EDAAnalyzer(source)
    measure_col = None if measure_agg == "count" else "amount"
    rows = json.loads(
        json.dumps(analyzer.get_decomposition_split(measure_col, measure_agg, split_col, []))
    )
    assert {row["filter_value"]: row["value"] for row in rows} == expected
    assert [row["value"] for row in rows] == sorted(expected.values(), reverse=True)
    for row in rows:
        assert row["name"] == (row["filter_value"] or "Unknown")
        assert row["ratio"] == pytest.approx(row["value"] / sum(expected.values()))
        selected = analyzer.get_decomposition_split(
            measure_col,
            measure_agg,
            None,
            [{"column": split_col, "operator": "==", "value": row["filter_value"]}],
        )
        assert selected == [{"name": "Total", "value": row["value"], "ratio": 1.0}]
    assert source.equals(analyzer.df)


@pytest.mark.parametrize("column", ["value", "ratio", "name"])
def test_numeric_group_column_can_also_be_the_measure(column: str) -> None:
    """Formatting group labels must not turn a numeric measure into strings."""
    analyzer = EDAAnalyzer(pl.DataFrame({column: [1, 1, 3]}))
    rows = analyzer.get_decomposition_split(column, "sum", column, [])
    assert {row["filter_value"]: row["value"] for row in rows} == {"1": 2, "3": 3}
    assert {row["filter_value"]: row["ratio"] for row in rows} == pytest.approx(
        {"1": 0.4, "3": 0.6}
    )


@pytest.mark.parametrize("split_col, measure_col", [("value", "name"), ("ratio", "value")])
def test_measure_names_and_zero_totals_do_not_replace_group_names(
    split_col: str, measure_col: str
) -> None:
    """Aggregating a reserved measure name must preserve labels even with zero totals."""
    analyzer = EDAAnalyzer(pl.DataFrame({split_col: ["a", "b"], measure_col: [0, 0]}))
    rows = analyzer.get_decomposition_split(measure_col, "sum", split_col, [])
    assert {row["filter_value"] for row in rows} == {"a", "b"}
    assert all(row["ratio"] == 0 and row["value"] == 0 for row in rows)
