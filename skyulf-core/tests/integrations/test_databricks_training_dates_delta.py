"""Real Delta source normalization before filtering and bounded driver transport."""

import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from skyulf.integrations.databricks.local_retraining import (
    LocalTrainingSpec,
    read_training_snapshot,
    split_labeled_snapshot,
)
from skyulf.integrations.databricks.training_dates import TrainingDateSpec


def _spec(table, **changes):
    """Use exact boundaries so timezone drift changes observable row membership."""
    return LocalTrainingSpec(
        table=table,
        version=0,
        start=datetime(2026, 1, 1, tzinfo=UTC),
        holdout_start=datetime(2026, 2, 1, tzinfo=UTC),
        cutoff=datetime(2026, 3, 1, tzinfo=UTC),
        event_column="event_at",
        result_available_at_column="result_at",
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=10,
        max_bytes=10000,
        **changes,
    )


@pytest.fixture
def source_table(delta_spark):
    """Each test owns a real versioned Delta table and removes its catalog entry."""
    table = f"spark_catalog.default.training_dates_{uuid4().hex}"
    yield table
    delta_spark.sql(f"DROP TABLE IF EXISTS {table}")


def test_native_timestamps_keep_instants_in_non_utc_session_and_process(delta_spark, source_table):
    """Spark and Python timezone differences must not change exact window membership."""
    previous = delta_spark.conf.get("spark.sql.session.timeZone")
    previous_tz = os.environ.get("TZ")
    delta_spark.conf.set("spark.sql.session.timeZone", "Asia/Tokyo")
    os.environ["TZ"] = "America/New_York"
    if hasattr(time, "tzset"):
        time.tzset()
    try:
        events = [
            datetime(2025, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 31, 23, 59, 59, 999999, tzinfo=UTC),
            datetime(2026, 2, 1, tzinfo=UTC),
            datetime(2026, 2, 28, 23, 59, 59, 999999, tzinfo=UTC),
            datetime(2026, 3, 1, tzinfo=UTC),
        ]
        delta_spark.createDataFrame(
            [(i, event, event, float(i), float(i * 2)) for i, event in enumerate(events)],
            "id long, event_at timestamp, result_at timestamp, x double, target double",
        ).write.format("delta").saveAsTable(source_table)
        spec = _spec(source_table)
        frame = read_training_snapshot(delta_spark, spec)
        train, holdout, missing = split_labeled_snapshot(frame, spec)
        assert frame["id"].tolist() == [1, 2, 3, 4]
        assert frame.iloc[1]["event_at"].to_pydatetime() == events[2]
        assert train["x"].tolist() == [1.0, 2.0]
        assert holdout["x"].tolist() == [3.0, 4.0]
        assert missing == 0
        with pytest.raises(ValueError, match="max_rows"):
            read_training_snapshot(delta_spark, replace(spec, max_rows=3))
        with pytest.raises(ValueError, match="max_bytes"):
            read_training_snapshot(delta_spark, replace(spec, max_bytes=1))
        with pytest.raises(ValueError, match="default parsing"):
            read_training_snapshot(
                delta_spark, replace(spec, event_time_parsing=TrainingDateSpec(timezone="UTC"))
            )
    finally:
        delta_spark.conf.set("spark.sql.session.timeZone", previous)
        if previous_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous_tz
        if hasattr(time, "tzset"):
            time.tzset()


def test_string_sources_normalize_before_window_and_label_filter(delta_spark, source_table):
    """Day-first strings, independent zones and unknown results share one UTC contract."""
    rows = [
        (1, "01/01/2026 03:00", "2026-01-01T00:00:00Z", 1.0, 2.0),
        (2, "15/01/2026 03:00", "2026-01-15T03:00:00+03:00", 2.0, 4.0),
        (3, "01/02/2026 03:00", "2026-02-01T00:00:00Z", 3.0, 6.0),
        (4, "15/02/2026 03:00", "2026-03-01T03:00:00+03:00", 4.0, 8.0),
        (5, "16/02/2026 03:00", None, 5.0, None),
        (6, "01/03/2026 03:00", "2026-03-01T00:00:00Z", 6.0, 12.0),
    ]
    delta_spark.createDataFrame(
        rows, "id long, event_at string, result_at string, x double, target double"
    ).write.format("delta").saveAsTable(source_table)
    spec = _spec(
        source_table,
        event_time_parsing=TrainingDateSpec(format="%d/%m/%Y %H:%M", timezone="Europe/Istanbul"),
        result_time_parsing=TrainingDateSpec(format="%Y-%m-%dT%H:%M:%S%z"),
    )
    frame = read_training_snapshot(delta_spark, spec)
    train, holdout, missing = split_labeled_snapshot(frame, spec)
    assert frame["id"].tolist() == [1, 2, 3, 4, 5]
    assert train["x"].tolist() == [1.0, 2.0]
    assert holdout["x"].tolist() == [3.0, 4.0]
    assert missing == 1


@pytest.mark.parametrize(
    "event,result",
    [
        (None, None),
        ("1999-99-99 00:00", None),
        ("1999-01-01 00:00", "bad"),
        ("2026-03-29 03:30", None),
        ("2026-10-25 03:30", None),
    ],
)
def test_invalid_source_dates_cannot_disappear_outside_window(
    delta_spark, source_table, event, result
):
    """Validation must visit malformed, null and DST-invalid rows before event filtering."""
    delta_spark.createDataFrame(
        [(1, event, result, 1.0, 2.0)],
        "id long, event_at string, result_at string, x double, target double",
    ).write.format("delta").saveAsTable(source_table)
    parser = TrainingDateSpec(format="%Y-%m-%d %H:%M", timezone="Europe/Vilnius")
    with pytest.raises(ValueError, match="Invalid training dates"):
        read_training_snapshot(
            delta_spark, _spec(source_table, event_time_parsing=parser, result_time_parsing=parser)
        )


@pytest.mark.parametrize(
    "dtype,expression,parser",
    [
        (
            "date",
            "DATE '2026-01-02'",
            TrainingDateSpec(timezone="Asia/Tokyo", date_only="midnight"),
        ),
        (
            "timestamp_ntz",
            "TIMESTAMP_NTZ '2026-01-02 00:00:00'",
            TrainingDateSpec(timezone="Asia/Tokyo"),
        ),
    ],
)
def test_native_local_dates_require_calendar_rules(
    delta_spark, source_table, dtype, expression, parser
):
    """Native dates and timestamp-without-timezone need explicit local interpretation."""
    delta_spark.sql(
        f"SELECT 1L id, {expression} event_at, {expression} result_at, 1.0D x, 2.0D target"
    ).write.format("delta").saveAsTable(source_table)
    with pytest.raises(ValueError, match="require"):
        read_training_snapshot(delta_spark, _spec(source_table))
    frame = read_training_snapshot(
        delta_spark, _spec(source_table, event_time_parsing=parser, result_time_parsing=parser)
    )
    assert frame.iloc[0]["event_at"].to_pydatetime() == datetime(2026, 1, 1, 15, tzinfo=UTC)
