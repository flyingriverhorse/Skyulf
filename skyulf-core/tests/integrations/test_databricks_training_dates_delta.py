"""Real Delta source normalization before filtering and bounded driver transport."""

import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from importlib import import_module
from uuid import uuid4

import pandas as pd
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
        split_strategy="temporal",
        filter_unavailable_results=True,
        result_cutoff=datetime(2026, 3, 1, tzinfo=UTC),
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


def test_date_free_delta_snapshot_stays_pinned_and_bounded(delta_spark, source_table):
    """Random reads need only keys/features/target and preserve old versions after append."""
    delta_spark.createDataFrame(
        [("a" if i < 10 else "b", i % 10, float(i), float(2 * i)) for i in reversed(range(20))],
        "tenant string, id long, x double, target double",
    ).write.format("delta").saveAsTable(source_table)
    spec = LocalTrainingSpec(
        table=source_table,
        version=0,
        record_key_columns=("tenant", "id"),
        input_columns=("x",),
        target_column="target",
        max_rows=20,
        max_bytes=100000,
    )
    first = read_training_snapshot(delta_spark, spec)
    assert first.x.tolist() == list(range(20))
    train, heldout, excluded = split_labeled_snapshot(first, spec)
    assert len(train) == 16 and len(heldout) == 4 and excluded == 0
    delta_spark.createDataFrame([("c", 1, 20.0, 40.0)], list(first.columns)).write.format(
        "delta"
    ).mode("append").saveAsTable(source_table)
    assert read_training_snapshot(delta_spark, spec).equals(first)
    with pytest.raises(ValueError, match="max_rows"):
        read_training_snapshot(delta_spark, replace(spec, version=1))
    with pytest.raises(ValueError, match="max_bytes"):
        read_training_snapshot(delta_spark, replace(spec, max_bytes=1))


def test_random_delta_result_filtering_without_event_date(delta_spark, source_table):
    """Availability-only normalization must exclude unknown/late rows without an event mapping."""
    delta_spark.createDataFrame(
        [
            (
                i,
                float(i),
                float(i * 2) if i < 10 else None,
                "2026-03-01T00:00:00Z" if i < 10 else (None if i == 10 else "2026-04-01T00:00:00Z"),
            )
            for i in range(12)
        ],
        "id long, x double, target double, available string",
    ).write.format("delta").saveAsTable(source_table)
    spec = LocalTrainingSpec(
        table=source_table,
        version=0,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=20,
        max_bytes=100000,
        filter_unavailable_results=True,
        result_available_at_column="available",
        result_time_parsing=TrainingDateSpec(format="%Y-%m-%dT%H:%M:%S%z"),
        result_cutoff=datetime(2026, 3, 15, tzinfo=UTC),
    )
    train, heldout, excluded = split_labeled_snapshot(
        read_training_snapshot(delta_spark, spec), spec
    )
    assert excluded == 2 and set(train.x).union(heldout.x) == set(range(10))
    delta_spark.sql(f"UPDATE {source_table} SET target = 22.0 WHERE id = 11")
    later = replace(spec, version=1, result_cutoff=datetime(2026, 4, 1, tzinfo=UTC))
    assert split_labeled_snapshot(read_training_snapshot(delta_spark, later), later)[2] == 1
    assert split_labeled_snapshot(read_training_snapshot(delta_spark, spec), spec)[2] == 2


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


def test_seeded_sample_selects_before_local_transfer_and_replays(delta_spark, source_table):
    """A 100k-row pinned source must yield a deterministic bounded 10k-row training input."""
    F = import_module("pyspark.sql.functions")

    source = delta_spark.range(100000).selectExpr(
        "id", "cast(id as double) x", "cast(id * 2 as double) target"
    )
    source.repartition(3).write.format("delta").saveAsTable(source_table)
    spec = LocalTrainingSpec(
        table=source_table,
        version=0,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=10000,
        max_bytes=5000000,
        training_sample_rows=10000,
        training_sample_seed=23,
    )
    frame = read_training_snapshot(delta_spark, spec)
    assert len(frame) == 10000
    assert frame.attrs["training_selection"]["eligible_rows"] == 100000
    train, holdout, _ = split_labeled_snapshot(frame, spec)
    pinned = replace(spec, sample_key_sha256=holdout.attrs["sample_key_sha256"])
    source.orderBy(F.desc("id")).repartition(5).write.format("delta").mode("overwrite").saveAsTable(
        source_table
    )
    same_rows_new_layout = read_training_snapshot(delta_spark, replace(spec, version=1))
    assert frame.id.tolist() == same_rows_new_layout.id.tolist()
    replay = split_labeled_snapshot(read_training_snapshot(delta_spark, pinned), pinned)
    pd.testing.assert_frame_equal(train, replay[0])
    pd.testing.assert_frame_equal(holdout, replay[1])
    other = read_training_snapshot(delta_spark, replace(spec, training_sample_seed=24))
    assert frame.id.tolist() != other.id.tolist()
    with pytest.raises(ValueError, match="max_rows"):
        read_training_snapshot(delta_spark, replace(spec, training_sample_rows=None))


def test_sampling_filters_availability_before_selecting_keys(delta_spark, source_table):
    """Late labels cannot consume sample slots and source key defects cannot hide outside the sample."""
    rows = [
        (i, datetime(2026, 1 if i < 30 else 3, 1, tzinfo=UTC), float(i), float(i * 2))
        for i in range(100)
    ]
    delta_spark.createDataFrame(
        rows, "id long, available timestamp, x double, target double"
    ).write.format("delta").saveAsTable(source_table)
    spec = LocalTrainingSpec(
        table=source_table,
        version=0,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=20,
        max_bytes=100000,
        training_sample_rows=20,
        filter_unavailable_results=True,
        result_available_at_column="available",
        result_cutoff=datetime(2026, 2, 1, tzinfo=UTC),
    )
    frame = read_training_snapshot(delta_spark, spec)
    assert len(frame) == 20 and frame.id.max() < 30
    assert split_labeled_snapshot(frame, spec)[2] == 70
    delta_spark.createDataFrame(
        [rows[0]], "id long, available timestamp, x double, target double"
    ).write.format("delta").mode("append").saveAsTable(source_table)
    with pytest.raises(ValueError, match="unique"):
        read_training_snapshot(delta_spark, replace(spec, version=1))


def test_sample_key_named_count_and_nan_target_validation(delta_spark, source_table):
    """Aggregation helper names cannot collide with business keys or hide non-finite labels."""
    rows = [(i, float(i), float(i)) for i in range(10)]
    delta_spark.createDataFrame(rows, "count long, x double, target double").write.format(
        "delta"
    ).saveAsTable(source_table)
    spec = LocalTrainingSpec(
        table=source_table,
        version=0,
        record_key_columns=("count",),
        input_columns=("x",),
        target_column="target",
        max_rows=6,
        max_bytes=100000,
        training_sample_rows=6,
    )
    assert len(read_training_snapshot(delta_spark, spec)) == 6
    delta_spark.createDataFrame(
        [(99, 99.0, float("nan"))], "count long, x double, target double"
    ).write.format("delta").mode("append").saveAsTable(source_table)
    with pytest.raises(ValueError, match="nonnull targets"):
        read_training_snapshot(delta_spark, replace(spec, version=1))
