"""Serialized temporal buckets must select their original rows on drill-down."""

import json
from datetime import UTC, date, datetime, time

import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer


@pytest.mark.parametrize(
    "group",
    [
        pl.Series("group", [date(2026, 1, 1), date(2026, 1, 2), None]),
        pl.Series("group", [datetime(2026, 1, 1, 9, 30), datetime(2026, 1, 1, 16, 45), None]),
        pl.Series("group", [time(9, 30), time(16, 45), None]),
        pl.Series(
            "group",
            [
                datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
                datetime(2026, 1, 1, 16, 45, tzinfo=UTC),
                None,
            ],
        ).dt.convert_time_zone("Europe/Vilnius"),
        pl.Series(
            "group", [1767225600000000001, 1767225600000000002, None], dtype=pl.Datetime("ns")
        ),
        pl.Series("group", [34200000000001, 34200000000002, None], dtype=pl.Time),
        pl.Series(
            "group",
            [
                datetime(2026, 10, 25, 0, 30, tzinfo=UTC),
                datetime(2026, 10, 25, 1, 30, tzinfo=UTC),
                None,
            ],
        ).dt.convert_time_zone("Europe/Vilnius"),
    ],
    ids=["date", "datetime", "time", "timezone", "nanoseconds", "time-nanoseconds", "dst-fold"],
)
def test_temporal_decomposition_buckets_round_trip_json(group):
    """Serialized labels must retain native precision, offsets, and missing-bucket identity."""
    analyzer = EDAAnalyzer(pl.DataFrame({"group": group, "amount": [2, 5, 7]}))
    rows = json.loads(json.dumps(analyzer.get_decomposition_split("amount", "sum", "group", [])))
    assert len(rows) == 3

    for row in rows:
        result = analyzer.get_decomposition_split(
            "amount",
            "sum",
            None,
            [{"column": "group", "operator": "==", "value": row["filter_value"]}],
        )
        assert result == [{"name": "Total", "value": row["value"], "ratio": 1.0}]


@pytest.mark.parametrize("serialized", [False, True])
def test_temporal_membership_filter_preserves_native_and_json_inputs(serialized):
    """Typed temporal filtering must retain existing list membership and empty selections."""
    dates = [date(2026, 1, 1), date(2026, 1, 2)]
    analyzer = EDAAnalyzer(pl.DataFrame({"group": dates, "amount": [2, 5]}))
    selected = [dates[0].isoformat()] if serialized else [dates[0]]
    for values, expected in [(selected, 2), ([], 0)]:
        result = analyzer.get_decomposition_split(
            "amount",
            "sum",
            None,
            [{"column": "group", "operator": "in", "value": values}],
        )
        assert result == [{"name": "Total", "value": expected, "ratio": 1.0}]


@pytest.mark.parametrize(
    "values",
    [
        [date(2026, 1, 1), date(2026, 1, 2)],
        [datetime(2026, 1, 1, 9, 30), datetime(2026, 1, 1, 16, 45)],
        [time(9, 30), time(16, 45)],
    ],
)
def test_native_temporal_equality_filter_keeps_direct_library_calls(values):
    """JSON bucket support must preserve Python date/time filters from direct Core callers."""
    analyzer = EDAAnalyzer(pl.DataFrame({"group": values, "amount": [2, 5]}))

    result = analyzer.get_decomposition_split(
        "amount",
        "sum",
        None,
        [{"column": "group", "operator": "==", "value": values[1]}],
    )

    assert result == [{"name": "Total", "value": 5, "ratio": 1.0}]
