"""Regressions for lossless automatic date detection and nullable date bounds."""

import json
import logging
from datetime import UTC, date, datetime

import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_date_formats_do_not_create_missing_values_or_drop_recommendation(reverse):
    """A format change beyond the sample must retain data throughout the public profile."""
    values = ["2024-01-15"] * 50 + ["02/20/2024"] * 950
    if reverse:
        values.reverse()
    frame = pl.DataFrame({"order_date": values})

    analyzer = EDAAnalyzer(frame)
    profile = analyzer.analyze()

    assert profile.columns["order_date"].missing_count == 0
    assert profile.columns["order_date"].missing_percentage == 0
    assert profile.columns["order_date"].dtype != "DateTime"
    assert profile.columns["order_date"].date_stats is None
    assert profile.timeseries is None
    assert profile.missing_cells_percentage == 0
    assert not any(rec.action == "Drop" for rec in profile.recommendations)
    assert analyzer.df["order_date"].to_list() == values
    assert profile.sample_data is not None
    assert [row["order_date"] for row in profile.sample_data] == values
    assert frame["order_date"].to_list() == values


@pytest.mark.parametrize("invalid_first", [False, True])
def test_unparseable_date_text_retains_only_original_missing_values(invalid_first):
    """Invalid text must stay observable even when it follows a valid date sample."""
    values = ["2024-01-15"] * 50 + ["not a date", "2024-02-30"] + [None] * 4
    if invalid_first:
        values.reverse()
    frame = pl.DataFrame({"event_date": values})

    analyzer = EDAAnalyzer(frame)
    profile = analyzer.analyze()

    assert profile.columns["event_date"].missing_count == 4
    assert profile.columns["event_date"].missing_percentage == pytest.approx(100 * 4 / 56)
    assert analyzer.df["event_date"].to_list() == values
    assert profile.sample_data is not None
    assert [row["event_date"] for row in profile.sample_data] == values


@pytest.mark.parametrize(
    "method",
    [(None, "datetime_generic"), (None, "date_generic"), ("%Y-%m-%d", "datetime_format")],
)
def test_selected_date_parser_retains_full_column_on_partial_failure(method, caplog):
    """Every selected parser must leave the column intact when a later value fails."""
    values = ["2024-01-15"] * 50 + ["02/20/2024", None]
    analyzer = EDAAnalyzer(pl.DataFrame({"source": values}))

    with caplog.at_level(logging.WARNING, logger="skyulf.profiling._analyzer.dates"):
        analyzer._apply_date_cast("source", method, "selected date format")

    assert analyzer.df["source"].null_count() == 1
    assert analyzer.df["source"].to_list() == values
    assert any("source" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    "dtype",
    [pl.Date, pl.Datetime("ms"), pl.Datetime("ns"), pl.Datetime("us", "Europe/Vilnius")],
)
def test_all_null_temporal_bounds_are_actual_nulls_in_public_serialization(dtype):
    """Missing extrema must stay nullable in stored dictionaries and JSON responses."""
    frame = pl.DataFrame({"event_date": pl.Series([None] * 3, dtype=dtype)})

    analyzer = EDAAnalyzer(frame)
    profile = analyzer.analyze()
    column = profile.columns["event_date"]

    assert column.dtype == "DateTime"
    assert column.missing_count == 3
    assert column.date_stats is not None
    assert column.date_stats.min_date is None
    assert column.date_stats.max_date is None
    assert column.date_stats.duration_days is None
    assert analyzer.df["event_date"].dtype == dtype
    expected = {"min_date": None, "max_date": None, "duration_days": None}
    assert profile.model_dump(mode="json")["columns"]["event_date"]["date_stats"] == expected
    assert json.loads(profile.model_dump_json())["columns"]["event_date"]["date_stats"] == expected


@pytest.mark.parametrize(
    ("values", "dtype", "expected_min", "expected_max"),
    [
        ([date(2024, 1, 15), None, date(2024, 1, 17)], pl.Date, "2024-01-15", "2024-01-17"),
        (
            [datetime(2024, 1, 15, 10), None, datetime(2024, 1, 17, 10)],
            pl.Datetime("ns"),
            "2024-01-15 10:00:00",
            "2024-01-17 10:00:00",
        ),
        (
            [
                datetime(2024, 1, 15, 10, tzinfo=UTC),
                None,
                datetime(2024, 1, 17, 10, tzinfo=UTC),
            ],
            pl.Datetime("us", "Europe/Vilnius"),
            "2024-01-15 12:00:00+02:00",
            "2024-01-17 12:00:00+02:00",
        ),
    ],
)
def test_native_temporal_columns_retain_dtype_timezone_and_present_bounds(
    values, dtype, expected_min, expected_max
):
    """Lossless detection must preserve existing native date and timezone semantics."""
    frame = pl.DataFrame({"event_date": pl.Series(values, dtype=dtype)})

    analyzer = EDAAnalyzer(frame)
    profile = analyzer.analyze()
    column = profile.columns["event_date"]

    assert analyzer.df["event_date"].equals(frame["event_date"])
    assert analyzer.df["event_date"].dtype == dtype
    assert column.missing_count == 1
    assert column.date_stats is not None
    assert column.date_stats.min_date == expected_min
    assert column.date_stats.max_date == expected_max
    assert column.date_stats.duration_days == 2


def test_uniform_date_strings_still_cast_with_original_nulls():
    """Real missing cells must not block successful automatic detection of valid dates."""
    values = ["2024-01-15"] * 50 + [None, "2024-02-20"]
    frame = pl.DataFrame({"event_date": values})

    analyzer = EDAAnalyzer(frame)
    profile = analyzer.analyze()
    column = profile.columns["event_date"]

    assert column.dtype == "DateTime"
    assert column.missing_count == 1
    assert column.date_stats is not None
    assert column.date_stats.min_date == "2024-01-15 00:00:00"
    assert column.date_stats.max_date == "2024-02-20 00:00:00"
    assert column.date_stats.duration_days == 36
    assert frame["event_date"].to_list() == values
