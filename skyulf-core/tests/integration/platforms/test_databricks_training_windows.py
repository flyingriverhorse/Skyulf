"""Data selection calendars must be independent of outer splits and job schedules."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

from skyulf.integrations.databricks.local_retraining import split_labeled_snapshot
from skyulf.integrations.databricks.local_workflow import resolve_training_spec, training_spec


@pytest.mark.parametrize("version", [None, 0, 7])
def test_train_resolves_latest_only_when_version_is_unset(version):
    """A trigger must never override an explicit snapshot or require a second action."""
    from skyulf.integrations.databricks.local_workflow import resolve_training_spec

    spark = _spark()
    spec = resolve_training_spec(
        spark, _config(training_version=version), datetime(2026, 9, 26, tzinfo=UTC)
    )
    assert spec.version == (9 if version is None else version)
    assert spark.sql.call_count == (1 if version is None else 0)


def test_train_preserves_explicit_result_cutoff():
    """An explicit maturity cutoff must survive both manual and cron invocations."""
    from skyulf.integrations.databricks.local_workflow import resolve_training_spec

    spec = resolve_training_spec(
        _spark(),
        _config(
            training_version=None,
            filter_unavailable_results=True,
            result_available_at_column="confirmed_at",
            result_availability_lag_hours=48,
            result_cutoff="2026-08-01T00:00:00+00:00",
        ),
        datetime(2026, 9, 26, tzinfo=UTC),
    )
    assert spec.result_cutoff == datetime(2026, 8, 1, tzinfo=UTC)


@pytest.mark.parametrize("version", [True, -1, "7", 1.5])
def test_invalid_explicit_version_never_falls_back_to_latest(version):
    """Bad pins must fail before history lookup instead of selecting unintended data."""
    spark = _spark()
    with pytest.raises(ValueError, match="training_version"):
        resolve_training_spec(spark, _config(training_version=version), datetime.now(UTC))
    spark.sql.assert_not_called()


def test_latest_is_resolved_again_for_each_new_invocation():
    """A later run must pick up appended data while the first run retains its pin."""
    spark = _spark()
    history = spark.sql.return_value.select.return_value.orderBy.return_value.first
    history.side_effect = [{"version": 9}, {"version": 10}]
    config = _config(training_version=None)
    now = datetime(2026, 9, 26, tzinfo=UTC)
    first = resolve_training_spec(spark, config, now)
    second = resolve_training_spec(spark, config, now)
    assert (first.version, second.version) == (9, 10)
    assert config["training_version"] is None


def _config(**changes):
    """Default to ordinary date-free sources; callers select calendar behavior explicitly."""
    config = {
        "training_table": "workspace.test.labels",
        "training_version": 0,
        "record_key_columns": ["id"],
        "input_columns": ["x"],
        "target_column": "target",
        "max_rows": 100,
        "max_input_mb": 1,
        "split_strategy": "random",
        "training_window_mode": "full_snapshot",
    }
    return {**config, **changes}


def _spark():
    """Only history lookup is needed to pin a monthly source version."""
    spark = Mock()
    spark.sql.return_value.select.return_value.orderBy.return_value.first.return_value = {
        "version": 9
    }
    return spark


@pytest.mark.parametrize("strategy", ["random", "temporal"])
def test_rolling_calendar_uses_named_zone_and_includes_holdout_month(strategy):
    """Business month rollover must follow the chosen calendar, not UTC or execution day."""
    config = _config(
        training_version=None,
        split_strategy=strategy,
        training_window_mode="rolling_calendar",
        event_column="event",
        monthly_lookback_months=4,
        window_timezone="Europe/Vilnius",
    )
    # Already January in Vilnius, still December in UTC.
    spec = resolve_training_spec(_spark(), config, datetime(2026, 12, 31, 22, 30, tzinfo=UTC))
    assert spec.start is not None and spec.cutoff is not None
    assert spec.start.isoformat() == "2026-09-01T00:00:00+03:00"
    assert spec.cutoff.isoformat() == "2027-01-01T00:00:00+02:00"
    if strategy == "temporal":
        assert spec.holdout_start is not None
        assert spec.holdout_start.isoformat() == "2026-12-01T00:00:00+02:00"
    else:
        assert spec.holdout_start is None and spec.test_size == 0.2
    assert spec.version == 9


def test_random_split_can_select_a_fixed_event_window():
    """Selecting recent observations must not force temporal train/test splitting."""
    config = _config(
        training_window_mode="fixed_window",
        event_column="event",
        start="2026-01-01T00:00:00+00:00",
        cutoff="2026-02-01T00:00:00+00:00",
    )
    spec = training_spec(config)
    frame = pd.DataFrame(
        {
            "id": range(20),
            "x": range(20),
            "target": range(20),
            "event": pd.date_range("2026-01-01", periods=20, tz="UTC"),
        }
    )
    train, holdout, _ = split_labeled_snapshot(frame, spec)
    assert len(train) == 16 and len(holdout) == 4
    assert set(train.x).isdisjoint(holdout.x)
    first = resolve_training_spec(_spark(), config, datetime(2026, 3, 15, tzinfo=UTC))
    later = resolve_training_spec(_spark(), config, datetime(2026, 4, 20, tzinfo=UTC))
    assert first.start == later.start == spec.start
    assert first.cutoff == later.cutoff == spec.cutoff


@pytest.mark.parametrize(
    "changes",
    [
        {"training_window_mode": "unknown"},
        {
            "training_window_mode": "rolling_calendar",
            "event_column": "event",
            "monthly_lookback_months": 4,
        },
        {
            "training_window_mode": "rolling_calendar",
            "event_column": "event",
            "monthly_lookback_months": 4,
            "window_timezone": "not/a-zone",
        },
        {"training_window_mode": "full_snapshot", "event_column": "event"},
        {"training_window_mode": "full_snapshot", "window_timezone": "UTC"},
    ],
)
def test_invalid_selection_policy_fails_before_history(changes):
    """A monthly job cannot invent source dates, silently shift calendars or ignore typos."""
    spark = _spark()
    with pytest.raises(ValueError):
        resolve_training_spec(spark, _config(**changes), datetime(2026, 9, 25, tzinfo=UTC))
    spark.sql.assert_not_called()


@pytest.mark.parametrize(
    "instant,start,holdout,cutoff",
    [
        (
            "2024-04-15T12:00:00+00:00",
            "2023-12-01T00:00:00+02:00",
            "2024-02-01T00:00:00+02:00",
            "2024-04-01T00:00:00+03:00",
        ),
        (
            "2026-11-15T12:00:00+00:00",
            "2026-07-01T00:00:00+03:00",
            "2026-09-01T00:00:00+03:00",
            "2026-11-01T00:00:00+02:00",
        ),
        (
            "2026-12-31T22:30:00+00:00",
            "2026-09-01T00:00:00+03:00",
            "2026-11-01T00:00:00+02:00",
            "2027-01-01T00:00:00+02:00",
        ),
    ],
)
def test_two_month_holdout_preserves_calendar_boundaries(instant, start, holdout, cutoff):
    """Holdout months belong to total lookback across leap years, DST and local rollover."""
    spec = resolve_training_spec(
        _spark(),
        _config(
            training_window_mode="rolling_calendar",
            split_strategy="temporal",
            event_column="event",
            monthly_lookback_months=4,
            holdout_months=2,
            window_timezone="Europe/Vilnius",
        ),
        datetime.fromisoformat(instant),
    )
    assert spec.start is not None and spec.holdout_start is not None and spec.cutoff is not None
    assert spec.start.isoformat() == start
    assert spec.holdout_start.isoformat() == holdout
    assert spec.cutoff.isoformat() == cutoff


@pytest.mark.parametrize("value", [None, True, 1.5, "2", 0, -1, 4, 121])
def test_invalid_rolling_holdout_fails_before_source_history(value):
    """Temporal holdout requires an exact integer leaving at least one training month."""
    spark = _spark()
    with pytest.raises(ValueError, match="holdout_months"):
        resolve_training_spec(
            spark,
            _config(
                training_window_mode="rolling_calendar",
                split_strategy="temporal",
                event_column="event",
                monthly_lookback_months=4,
                holdout_months=value,
                window_timezone="UTC",
            ),
            datetime(2026, 9, 25, tzinfo=UTC),
        )
    spark.sql.assert_not_called()


@pytest.mark.parametrize("instant", ["2026-03-29T04:00:00+03:00", "2026-10-25T04:00:00+02:00"])
@pytest.mark.parametrize("lag", [0, 48, 87600])
def test_result_lag_uses_elapsed_utc_hours_without_event_column(instant, lag):
    """Result maturity is independent of event windows and DST wall-clock changes."""
    now = datetime.fromisoformat(instant)
    spec = resolve_training_spec(
        _spark(),
        _config(
            filter_unavailable_results=True,
            result_available_at_column="confirmed_at",
            result_availability_lag_hours=lag,
        ),
        now,
    )
    assert spec.result_cutoff == now.astimezone(UTC) - timedelta(hours=lag)
    assert spec.event_column is None


@pytest.mark.parametrize("value", [None, True, 1.5, "2", -1, 87601])
def test_invalid_result_lag_fails_before_source_history(value):
    """Bad result lag cannot cause external reads or silently coerce a different cutoff."""
    spark = _spark()
    with pytest.raises(ValueError, match="result_availability_lag_hours"):
        resolve_training_spec(
            spark,
            _config(
                filter_unavailable_results=True,
                result_available_at_column="confirmed_at",
                result_availability_lag_hours=value,
            ),
            datetime(2026, 9, 25, tzinfo=UTC),
        )
    spark.sql.assert_not_called()


@pytest.mark.parametrize(
    "field,value", [("holdout_months", 1), ("result_availability_lag_hours", 0)]
)
def test_inactive_window_controls_require_null(field, value):
    """Inactive controls must not be silently ignored in date-free workflows."""
    with pytest.raises(ValueError, match=field):
        training_spec(_config(**{field: value}))


def test_manual_training_keeps_explicit_result_cutoff_with_lag():
    """A manual replay must preserve its saved cutoff regardless of the scheduled lag."""
    spec = training_spec(
        _config(
            filter_unavailable_results=True,
            result_available_at_column="confirmed_at",
            result_availability_lag_hours=48,
            result_cutoff="2026-09-01T00:00:00+00:00",
        )
    )
    assert spec.result_cutoff == datetime(2026, 9, 1, tzinfo=UTC)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_lagged_result_cutoff_is_inclusive_and_event_window_half_open(engine):
    """A mature label at cutoff participates, a later label and cutoff event cannot."""
    spec = resolve_training_spec(
        _spark(),
        _config(
            training_window_mode="rolling_calendar",
            split_strategy="temporal",
            event_column="event",
            monthly_lookback_months=4,
            holdout_months=2,
            window_timezone="UTC",
            filter_unavailable_results=True,
            result_available_at_column="confirmed_at",
            result_availability_lag_hours=48,
        ),
        datetime(2026, 9, 3, tzinfo=UTC),
    )
    assert spec.result_cutoff is not None
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "x": [1, 2, 3, 4, 5],
            "target": [2, 4, 6, 8, 10],
            "event": pd.to_datetime(
                ["2026-05-01", "2026-06-01", "2026-07-01", "2026-08-31", "2026-08-31"], utc=True
            ),
            "confirmed_at": [spec.result_cutoff] * 4
            + [spec.result_cutoff + timedelta(microseconds=1)],
        }
    )
    train, holdout, excluded = split_labeled_snapshot(frame, spec, engine=engine)
    assert train.x.tolist() == [1, 2]
    assert holdout.x.tolist() == [3, 4]
    assert excluded == 1
    frame.loc[4, "event"] = spec.cutoff
    with pytest.raises(ValueError, match="outside the pinned window"):
        split_labeled_snapshot(frame, spec, engine=engine)


@pytest.mark.parametrize("strategy", ["random", "temporal"])
@pytest.mark.parametrize("instant", ["2026-03-29T04:00:00+03:00", "2026-10-25T04:00:00+02:00"])
def test_rolling_days_uses_elapsed_utc_days_and_selected_event(strategy, instant):
    """Daily windows advance with each run without calendar or DST rounding."""
    now = datetime.fromisoformat(instant)
    config = _config(
        training_window_mode="rolling_days",
        split_strategy=strategy,
        event_column="observed_on",
        lookback_days=90,
        holdout_days=14 if strategy == "temporal" else None,
    )
    spec = resolve_training_spec(_spark(), config, now)
    later = resolve_training_spec(_spark(), config, now + timedelta(days=1))
    assert spec.event_column == "observed_on"
    assert spec.start is not None
    assert spec.start == now.astimezone(UTC) - timedelta(days=90)
    assert spec.cutoff == now.astimezone(UTC)
    assert spec.holdout_start == (
        now.astimezone(UTC) - timedelta(days=14) if strategy == "temporal" else None
    )
    assert later.start == spec.start + timedelta(days=1)
    assert "start" not in config


@pytest.mark.parametrize(
    "changes,field",
    [
        ({"lookback_days": None}, "lookback_days"),
        ({"lookback_days": True}, "lookback_days"),
        ({"lookback_days": 0}, "lookback_days"),
        ({"lookback_days": 1.5}, "lookback_days"),
        ({"lookback_days": "90"}, "lookback_days"),
        ({"lookback_days": 36501}, "lookback_days"),
        ({"holdout_days": None}, "holdout_days"),
        ({"holdout_days": True}, "holdout_days"),
        ({"holdout_days": 0}, "holdout_days"),
        ({"holdout_days": 90}, "holdout_days"),
        ({"holdout_days": 1.5}, "holdout_days"),
        ({"holdout_days": "14"}, "holdout_days"),
        ({"event_column": None}, "event_column"),
        ({"window_timezone": "UTC"}, "window_timezone"),
        ({"monthly_lookback_months": 4}, "monthly_lookback_months"),
        ({"holdout_months": 1}, "holdout_months"),
        ({"split_strategy": "random"}, "holdout_days"),
    ],
)
def test_invalid_daily_windows_fail_before_history(changes, field):
    """Invalid daily bounds and inactive calendar fields cannot reach source reads."""
    config = _config(
        training_window_mode="rolling_days",
        split_strategy="temporal",
        event_column="observed_on",
        lookback_days=90,
        holdout_days=14,
    )
    spark = _spark()
    with pytest.raises(ValueError, match=field):
        resolve_training_spec(spark, config | changes, datetime(2026, 10, 1, tzinfo=UTC))
    spark.sql.assert_not_called()


@pytest.mark.parametrize("field", ["lookback_days", "holdout_days"])
def test_inactive_daily_controls_require_null(field):
    """Date-free defaults must reject daily settings that would otherwise be ignored."""
    with pytest.raises(ValueError, match=field):
        training_spec(_config(**{field: 7}))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("strategy", ["random", "temporal"])
def test_daily_window_splits_selected_event_text_with_existing_parsing(engine, strategy):
    """Both engines must honor the chosen date column and existing explicit text parser."""
    spec = resolve_training_spec(
        _spark(),
        _config(
            training_window_mode="rolling_days",
            split_strategy=strategy,
            event_column="observed_on",
            event_time_parsing={"format": "%d/%m/%Y", "timezone": "UTC", "date_only": "midnight"},
            lookback_days=10,
            holdout_days=2 if strategy == "temporal" else None,
        ),
        datetime(2026, 10, 11, tzinfo=UTC),
    )
    frame = pd.DataFrame(
        {
            "id": range(10),
            "x": range(10),
            "target": range(10),
            "observed_on": [f"{day:02d}/10/2026" for day in range(1, 11)],
        }
    )
    train, holdout, excluded = split_labeled_snapshot(frame, spec, engine=engine)
    assert len(train) == 8 and len(holdout) == 2 and excluded == 0
    assert set(train.x) | set(holdout.x) == set(range(10))
    if strategy == "temporal":
        assert holdout.x.tolist() == [8, 9]
    frame.loc[9, "observed_on"] = "11/10/2026"
    with pytest.raises(ValueError, match="outside the pinned window"):
        split_labeled_snapshot(frame, spec, engine=engine)
