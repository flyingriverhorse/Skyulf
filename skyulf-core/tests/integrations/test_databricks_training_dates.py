"""Explicit date parsing keeps training instants independent of machine settings."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from skyulf.integrations.databricks.training_dates import TrainingDateSpec, parse_training_date


@pytest.mark.parametrize("value", ["01/02/2026", datetime(2026, 1, 2), date(2026, 1, 2)])
def test_unconfigured_local_dates_are_rejected(value):
    """Strings and naive dates must never silently acquire UTC semantics."""
    with pytest.raises(ValueError):
        parse_training_date(value, TrainingDateSpec())


@pytest.mark.parametrize("value", ["2026-03-29 03:30", "2026-10-25 03:30"])
def test_dst_gaps_and_overlaps_are_rejected(value):
    """A timezone cannot choose an instant for nonexistent or repeated local time."""
    spec = TrainingDateSpec(format="%Y-%m-%d %H:%M", timezone="Europe/Vilnius")
    with pytest.raises(ValueError, match="ambiguous|nonexistent"):
        parse_training_date(value, spec)


def test_offsets_and_distinct_source_zones_preserve_instants():
    """Event and result calendars may represent the same instant differently."""
    expected = datetime(2026, 6, 1, tzinfo=UTC)
    offset = TrainingDateSpec(format="%Y-%m-%dT%H:%M:%S%z")
    local = TrainingDateSpec(format="%d/%m/%Y %H:%M", timezone="Europe/Vilnius")
    assert parse_training_date("2026-06-01T03:00:00+03:00", offset) == expected
    assert parse_training_date("2026-06-01T00:00:00Z", offset) == expected
    assert parse_training_date("01/06/2026 03:00", local) == expected
    assert parse_training_date(expected, local) == expected


def test_date_only_requires_explicit_midnight_and_timezone():
    """Date-only values receive calendar semantics only by an explicit policy."""
    spec = TrainingDateSpec(format="%d/%m/%Y", timezone="Asia/Tokyo", date_only="midnight")
    assert parse_training_date("02/01/2026", spec) == datetime(2026, 1, 1, 15, tzinfo=UTC)
    assert parse_training_date(date(2026, 1, 2), spec) == datetime(2026, 1, 1, 15, tzinfo=UTC)
    with pytest.raises(ValueError, match="date_only"):
        TrainingDateSpec(format="%Y-%m-%d", timezone="UTC")


@pytest.mark.parametrize("format", ["%d/%m", "%y-%m-%d", "%Y-%b-%d", "%Y-%j", "%Y-%m-%d %I:%M %p"])
def test_incomplete_or_locale_dependent_formats_are_rejected(format):
    """A narrow format vocabulary avoids inference and locale-dependent replays."""
    with pytest.raises(ValueError, match="format"):
        TrainingDateSpec(format=format, timezone="UTC", date_only="midnight")


@pytest.mark.parametrize("value", ["2026-02-30 01:00", "2026-2-01 01:00", "2026-02-01", ""])
def test_malformed_values_do_not_coerce_to_missing(value):
    """A supplied value must match the entire declared format and a real calendar date."""
    with pytest.raises(ValueError):
        parse_training_date(value, TrainingDateSpec(format="%Y-%m-%d %H:%M", timezone="UTC"))


def test_null_result_is_allowed_but_null_event_is_rejected():
    """Unknown label availability is distinct from a missing observation instant."""
    for value in (None, pd.NaT, pd.NA, float("nan")):
        assert parse_training_date(value, TrainingDateSpec(), allow_null=True) is None
        with pytest.raises(ValueError, match="null"):
            parse_training_date(value, TrainingDateSpec())


def test_transport_preserves_microseconds_and_rejects_naive_instants():
    """Integer transport must never round a boundary or consult the process timezone."""
    from skyulf.integrations.databricks.training_dates import (
        instant_from_microseconds,
        instant_microseconds,
    )

    instant = datetime(1960, 1, 2, 3, 4, 5, 123456, tzinfo=UTC)
    assert instant_from_microseconds(instant_microseconds(instant)) == instant
    with pytest.raises(ValueError, match="aware"):
        instant_microseconds(instant.replace(tzinfo=None))
