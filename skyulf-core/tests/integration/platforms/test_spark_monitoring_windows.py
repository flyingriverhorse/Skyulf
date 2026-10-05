"""Late-label repair runs chronologically without blocking durable current failures."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

from skyulf.integrations.databricks import spark_monitoring_windows as windows
from skyulf.integrations.databricks.monitoring_config import MonitorConfig


def config():
    """Select a valid daily report-only policy with immutable baseline identity."""
    return MonitorConfig(
        environment="test",
        project="spark",
        model_name="a.b.model",
        model_version="1",
        source_table="a.b.source",
        prediction_table="a.b.predictions",
        label_table="a.b.labels",
        result_available_at_column="available",
        execution_engine="spark",
        reference_namespace="a.mon",
        performance_policy={
            "mode": "report",
            "metric": "mae",
            "direction": "lower",
            "baseline": {"kind": "training_holdout", "model_version": "1"},
            "tolerance": 0.05,
            "tolerance_mode": "absolute",
            "window_hours": 24,
            "label_delay_hours": 0,
            "minimum_labeled_rows": 2,
            "minimum_label_coverage": 0.8,
            "consecutive_windows": 3,
        },
    )


def test_revisit_writes_oldest_first_and_excludes_latest(monkeypatch):
    """Later streak evaluation must see repaired previous windows in chronological order."""
    now = datetime(2026, 10, 5, 1, tzinfo=UTC)
    end = now.replace(hour=0)
    monkeypatch.setattr(
        windows,
        "load_spark_monitoring_reference",
        Mock(return_value=(None, None, None, {"model_version": "1"})),
    )
    observe = Mock(return_value={"status": "degraded", "action": "none"})
    persist = Mock()
    monkeypatch.setattr(windows, "observe_performance_safely", observe)
    monkeypatch.setattr(windows, "persist_report", persist)
    windows.revisit_performance_windows(None, "a.mon", [config()], now)
    assert [call.kwargs["window_end"] for call in observe.call_args_list] == [
        end - timedelta(days=2),
        end - timedelta(days=1),
    ]
    assert persist.call_count == 2


def test_missing_reference_leaves_latest_failure_to_observer(monkeypatch):
    """A missing cache must not abort before the current observer saves its failure."""
    monkeypatch.setattr(
        windows, "load_spark_monitoring_reference", Mock(side_effect=ValueError("missing cache"))
    )
    persist = Mock()
    monkeypatch.setattr(windows, "persist_report", persist)
    windows.revisit_performance_windows(
        None, "a.mon", [config()], datetime(2026, 10, 5, tzinfo=UTC)
    )
    persist.assert_not_called()
