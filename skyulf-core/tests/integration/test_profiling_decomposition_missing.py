"""Decomposition drill-down must preserve missing groups across JSON round trips."""

import json
from datetime import date

import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer


def test_missing_and_literal_unknown_groups_drill_down_independently() -> None:
    """A display label must neither merge missing rows nor select the literal category."""
    analyzer = EDAAnalyzer(
        pl.DataFrame({"group": ["a", None, "Unknown", None], "value": [1, 2, 3, 4]})
    )
    rows = json.loads(json.dumps(analyzer.get_decomposition_split("value", "sum", "group", [])))
    assert len(rows) == 3
    unknown_rows = [row for row in rows if row["name"] == "Unknown"]
    assert {row["filter_value"]: row["value"] for row in unknown_rows} == {
        None: 6,
        "Unknown": 3,
    }
    for row in rows:
        result = analyzer.get_decomposition_split(
            "value",
            "sum",
            None,
            [{"column": "group", "operator": "==", "value": row["filter_value"]}],
        )
        assert result == [{"name": "Total", "value": row["value"], "ratio": 1.0}]
    assert sum(row["ratio"] for row in rows) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "group",
    [
        pl.Series("group", ["a", None]),
        pl.Series("group", ["a", None], dtype=pl.Categorical),
        pl.Series("group", ["a", None], dtype=pl.Enum(["a"])),
        pl.Series("group", [True, None]),
        pl.Series("group", [1, None]),
        pl.Series("group", [1.5, None]),
        pl.Series("group", [date(2026, 1, 1), None]),
        pl.Series("group", [None, None]),
    ],
    ids=["string", "categorical", "enum", "boolean", "integer", "float", "date", "null"],
)
def test_missing_group_round_trips_for_each_dtype(group: pl.Series) -> None:
    """Missing-value filtering must work independently of the split column dtype."""
    analyzer = EDAAnalyzer(pl.DataFrame({"group": group, "value": [1, 2]}))
    rows = analyzer.get_decomposition_split("value", "sum", "group", [])
    missing = next(row for row in rows if row["name"] == "Unknown")
    result = analyzer.get_decomposition_split(
        "value",
        "sum",
        None,
        [
            {
                "column": "group",
                "operator": "==",
                "value": missing.get("filter_value", missing["name"]),
            }
        ],
    )
    assert result == [{"name": "Total", "value": missing["value"], "ratio": 1.0}]


@pytest.mark.parametrize("operator, expected", [("==", 6), ("!=", 4)])
def test_missing_numeric_filter_includes_normalized_nan(operator: str, expected: int) -> None:
    """NaN shares the analyzer's missing bucket while numeric values stay separate."""
    analyzer = EDAAnalyzer(
        pl.DataFrame({"group": [1.5, None, 2.0, float("nan")], "value": [1, 2, 3, 4]})
    )
    result = analyzer.get_decomposition_split(
        "value",
        "sum",
        None,
        [{"column": "group", "operator": operator, "value": None}],
    )
    assert result == [{"name": "Total", "value": expected, "ratio": 1.0}]


def test_numeric_and_legacy_unknown_filters_keep_their_meaning() -> None:
    """Adding a typed missing bucket must retain earlier numeric drill-down requests."""
    analyzer = EDAAnalyzer(
        pl.DataFrame({"group": [1.5, None, 2.0, float("nan")], "value": [1, 2, 3, 4]})
    )
    rows = analyzer.get_decomposition_split("value", "sum", "group", [])
    assert {row["name"]: row["value"] for row in rows} == {"1.5": 1, "2.0": 3, "Unknown": 6}
    for value, expected in [("1.5", 1), ("Unknown", 6)]:
        result = analyzer.get_decomposition_split(
            "value",
            "sum",
            None,
            [{"column": "group", "operator": "==", "value": value}],
        )
        assert result == [{"name": "Total", "value": expected, "ratio": 1.0}]
