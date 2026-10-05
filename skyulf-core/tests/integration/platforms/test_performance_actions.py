"""Persist real request outcomes separately from immutable metric observations."""

from datetime import UTC, datetime
from unittest.mock import Mock


def test_disabled_performance_does_not_touch_action_store():
    """Upgrading a drift-only project must not introduce new central writes."""
    from skyulf.integrations.databricks.performance_actions import record_performance_actions

    spark = Mock()
    record_performance_actions(
        spark,
        "cat.monitor",
        {"status": "not_requested", "models": []},
        datetime(2026, 10, 5, tzinfo=UTC),
        preview_only=True,
    )
    assert spark.mock_calls == []


def test_skipped_training_keeps_metric_reason_and_actual_action(monkeypatch):
    """No-new-data skips must remain visible after the scoring task has finished."""
    from skyulf.integrations.databricks import performance_actions as actions

    rows = []
    monkeypatch.setattr(actions, "persist_action", lambda spark, ns, row: rows.append(row))
    result = {
        "status": "not_requested",
        "models": [
            {
                "performance_monitored": True,
                "report_id": "a" * 64,
                "model_name": "cat.m.model",
                "status": "no_new_training_data",
                "triggers": ["performance"],
            }
        ],
    }
    actions.record_performance_actions(
        None, "cat.monitor", result, datetime(2026, 10, 5, tzinfo=UTC), preview_only=True
    )
    assert rows[0]["action"] == "retraining_skipped"
    assert rows[0]["action_reason"] == "no_new_training_data"
    assert rows[0]["report_id"] == "a" * 64


def test_performance_view_keeps_metric_and_request_evidence_separate():
    """The dashboard must join saved outcomes while retaining unconfigured observations."""
    from skyulf.integrations.databricks.performance_actions import performance_history_query

    sql = performance_history_query("cat.monitor")
    assert "LEFT JOIN" in sql
    assert "performance_actions" in sql
    assert "'disabled'" in sql
    assert "action_reason" in sql
    assert "baseline_value" in sql
    assert "config_digest" in sql


def test_no_trigger_records_performance_reason_instead_of_drift_reason(monkeypatch):
    """A performance skip must retain its own evidence when drift is also enabled."""
    from skyulf.integrations.databricks import performance_actions as actions

    rows = []
    monkeypatch.setattr(actions, "persist_action", lambda spark, ns, row: rows.append(row))
    actions.record_performance_actions(
        None,
        "cat.monitor",
        {
            "status": "not_requested",
            "models": [
                {
                    "performance_monitored": True,
                    "report_id": "a" * 64,
                    "status": "no_drift",
                    "performance_reason": "insufficient_labels",
                    "triggers": [],
                }
            ],
        },
        datetime(2026, 10, 5, tzinfo=UTC),
    )
    assert rows[0]["action"] == "none"
    assert rows[0]["action_reason"] == "insufficient_labels"


def test_only_ready_model_receives_shared_request_ids(monkeypatch):
    """A skipped component must not inherit another component's request identity."""
    from skyulf.integrations.databricks import performance_actions as actions

    rows = []
    monkeypatch.setattr(actions, "persist_action", lambda spark, ns, row: rows.append(row))
    actions.record_performance_actions(
        None,
        "cat.monitor",
        {
            "status": "submitted",
            "request_id": "request-a",
            "run_id": 123,
            "models": [
                {
                    "performance_monitored": True,
                    "report_id": "a" * 64,
                    "status": "ready",
                    "triggers": ["performance"],
                },
                {
                    "performance_monitored": True,
                    "report_id": "b" * 64,
                    "status": "no_drift",
                    "performance_reason": "missing_labels",
                    "triggers": [],
                },
            ],
        },
        datetime(2026, 10, 5, tzinfo=UTC),
    )
    assert [(row["request_id"], row["run_id"]) for row in rows] == [
        ("request-a", 123),
        (None, None),
    ]


def test_actual_actions_rank_ahead_of_later_previews(monkeypatch):
    """Advisory replays must not replace a saved submission in lookup or dashboard."""
    from skyulf.integrations.databricks.performance_actions import (
        load_performance_action,
        performance_history_query,
    )

    spark = Mock()
    spark.table.return_value.where.return_value.orderBy.return_value.limit.return_value.first.return_value = None
    from skyulf.integrations.databricks import performance_actions as actions

    monkeypatch.setattr(actions, "ensure_owned_object", lambda spark, name: True)
    load_performance_action(spark, "cat.monitor", "a" * 64)
    assert spark.table.return_value.where.return_value.orderBy.call_args.kwargs["ascending"] == [
        True,
        False,
        False,
    ]
    assert "ORDER BY preview ASC, recorded_at DESC, action_id DESC" in performance_history_query(
        "cat.monitor"
    )


def test_disabled_policy_does_not_borrow_drift_scoring_window():
    """Absent policy windows must stay absent in the performance evidence view."""
    from skyulf.integrations.databricks.performance_actions import performance_history_query

    sql = performance_history_query("cat.monitor")
    assert "AS window_start" in sql and "AS window_end" in sql
    assert "r.window_start" not in sql and "r.window_end" not in sql
