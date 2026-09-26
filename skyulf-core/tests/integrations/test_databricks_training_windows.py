"""Data selection calendars must be independent of outer splits and job schedules."""

from datetime import UTC, datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from skyulf.integrations.databricks.local_retraining import split_labeled_snapshot
from skyulf.integrations.databricks.local_workflow import _monthly_training_spec, _training_spec


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
        split_strategy=strategy,
        training_window_mode="rolling_calendar",
        event_column="event",
        monthly_lookback_months=4,
        window_timezone="Europe/Vilnius",
    )
    # Already January in Vilnius, still December in UTC.
    spec = _monthly_training_spec(_spark(), config, datetime(2026, 12, 31, 22, 30, tzinfo=UTC))
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
    spec = _training_spec(config)
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
    first = _monthly_training_spec(_spark(), config, datetime(2026, 3, 15, tzinfo=UTC))
    later = _monthly_training_spec(_spark(), config, datetime(2026, 4, 20, tzinfo=UTC))
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
        _monthly_training_spec(spark, _config(**changes), datetime(2026, 9, 25, tzinfo=UTC))
    spark.sql.assert_not_called()
