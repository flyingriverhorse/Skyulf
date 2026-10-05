"""Operator reports expose the saved batch without repeating drift computation."""

import json
from unittest.mock import Mock

import pytest


def test_monitor_report_has_safe_dashboard_link_and_preserves_plain_text():
    """Notebook output must not turn registry names or configured URLs into active markup."""
    from skyulf.integrations.databricks.monitoring_output import render_monitor_output

    html = render_monitor_output(
        {"results": [{"model_name": "<script>bad</script>", "status": "drift"}]},
        "https://workspace.example/dashboardsv3/id/published?a=1&b=2",
    )
    assert "Open monitoring dashboard" in html
    assert "a=1&amp;b=2" in html
    assert "&lt;script&gt;" in html and "<script>" not in html
    assert 'href="javascript:' not in render_monitor_output({}, "javascript:alert(1)")


def test_drift_report_reads_only_the_completed_batch_and_does_not_recompute():
    """A delayed report task must display the same scoring commit, not the newest model batch."""
    from skyulf.integrations.databricks.monitoring_output import run_drift_report_notebook

    dbutils, spark, display = Mock(), Mock(), Mock()
    dbutils.widgets.getAll.return_value = {"monitoring_dashboard_url": ""}
    dbutils.jobs.taskValues.get.return_value = {
        "status": "ready",
        "namespace": "ops.monitoring",
        "monitor_id": "a" * 64,
        "config_digest": "b" * 64,
        "window_start": "2026-10-01T10:00:00+00:00",
        "window_end": "2026-10-01T10:00:00.000001+00:00",
    }
    row = {
        "report_id": "report",
        "model_name": "models.schema.model",
        "model_version": "7",
        "status": "drift",
        "measured_at": "2026-10-01 10:01:00",
        "observed_at": "2026-10-01 10:00:00",
        "drifted_columns": 1,
        "report_json": json.dumps(
            {
                "metrics": [
                    {
                        "category": "drift",
                        "column_name": "x",
                        "metric_name": "psi",
                        "value": 0.4,
                        "threshold": 0.2,
                        "has_issue": True,
                        "status": "measured",
                    }
                ]
            }
        ),
    }
    spark.table.return_value.where.return_value.orderBy.return_value.limit.return_value.first.return_value.asDict.return_value = row
    result = run_drift_report_notebook(spark, dbutils, display_html=display)
    predicate = spark.table.return_value.where.call_args.args[0]
    assert "window_start" in predicate and "config_digest" in predicate
    assert "status <> 'failed'" in predicate
    assert "0.4 / 0.2" in display.call_args.args[0]
    assert result == {"report_id": "report", "status": "drift", "drifted_columns": 1}


def test_drift_report_displays_each_model_set_component(monkeypatch):
    """A successful set score must retain both targets in its visible drift report."""
    from skyulf.integrations.databricks import monitoring_output as output

    references = [{"monitor_id": "revenue"}, {"monitor_id": "risk"}]
    rows = [
        {"report_id": "revenue_report", "status": "healthy", "drifted_columns": 0},
        {"report_id": "risk_report", "status": "drift", "drifted_columns": 1},
    ]
    load = Mock(side_effect=rows)
    monkeypatch.setattr(output, "load_observation", load)
    monkeypatch.setattr(output, "render_drift_output", lambda row, url: row["report_id"])
    dbutils, display = Mock(), Mock()
    dbutils.jobs.taskValues.get.return_value = {"status": "ready", "observations": references}
    dbutils.widgets.getAll.return_value = {}
    result = output.run_drift_report_notebook(Mock(), dbutils, display_html=display)
    assert result == {"results": rows}
    assert [call.args[1] for call in load.call_args_list] == references
    assert display.call_args.args[0] == "revenue_reportrisk_report"


@pytest.mark.parametrize("status", ["disabled", "no_new_predictions"])
def test_drift_report_skips_without_a_saved_batch(status):
    """An opted-out or empty scoring run must not display an unrelated historical report."""
    from skyulf.integrations.databricks.monitoring_output import run_drift_report_notebook

    spark, dbutils = Mock(), Mock()
    dbutils.widgets.getAll.return_value = {}
    dbutils.jobs.taskValues.get.return_value = {"status": status}
    assert run_drift_report_notebook(spark, dbutils) == {"status": status}
    spark.table.assert_not_called()


def test_drift_report_rejects_unbound_reference():
    """Unvalidated task values must not become a SQL predicate or table name."""
    from skyulf.integrations.databricks.monitoring_output import load_observation

    with pytest.raises(ValueError):
        load_observation(Mock(), {"status": "ready", "namespace": "bad;sql"})


def test_drift_report_shows_saved_performance_policy_evidence():
    """Operators must see the policy verdict and exact evidence for the saved report."""
    from skyulf.integrations.databricks.monitoring_output import render_drift_output

    row = {
        "report_id": "report-9",
        "model_name": "model",
        "model_version": "7",
        "observed_at": "2026-10-05T00:00:00Z",
        "measured_at": "2026-10-05T02:00:00Z",
        "drifted_columns": 0,
        "report_json": json.dumps(
            {
                "metrics": [],
                "performance": {
                    "status": "degraded",
                    "reason": "Three completed failing windows",
                    "metric": "f1",
                    "direction": "higher",
                    "baseline_kind": "training_holdout",
                    "baseline_reference": "holdout-7",
                    "baseline_value": 0.9,
                    "current_value": 0.81,
                    "absolute_degradation": 0.09,
                    "relative_degradation": 0.1,
                    "tolerance": 0.05,
                    "tolerance_mode": "absolute",
                    "threshold_value": 0.85,
                    "labeled_rows": 80,
                    "label_coverage": 0.8,
                    "window_start": "2026-10-04T00:00:00Z",
                    "window_end": "2026-10-05T00:00:00Z",
                    "consecutive_failures": 3,
                    "required_windows": 3,
                    "evaluated_windows": ["w1", "w2", "w3"],
                    "action": "retraining_requested",
                    "request_id": "request-8",
                    "run_id": "run-12",
                },
            }
        ),
    }
    html = render_drift_output(row)
    for evidence in (
        "Performance policy evidence",
        "report-9",
        "Degraded",
        "f1",
        "0.9",
        "0.81",
        "0.09",
        "0.85",
        "0.8",
        "80",
        "2026-10-04",
        "3 / 3",
        "retraining_requested",
        "request-8",
        "run-12",
    ):
        assert evidence in html


def test_drift_report_marks_missing_policy_disabled_and_escapes_reason():
    """Absent policy and unavailable labels must never render as a healthy verdict."""
    from skyulf.integrations.databricks.monitoring_output import render_drift_output

    row = {
        "report_id": "r",
        "model_name": "m",
        "model_version": "1",
        "observed_at": "now",
        "measured_at": "now",
        "drifted_columns": 0,
    }
    row["report_json"] = json.dumps({"metrics": []})
    assert "Disabled" in render_drift_output(row)
    row["report_json"] = json.dumps(
        {
            "metrics": [],
            "performance": {
                "status": "unavailable",
                "reason": "No labels <script>alert(1)</script>",
            },
        }
    )
    html = render_drift_output(row)
    assert "Unavailable" in html and "No labels &lt;script&gt;" in html
    assert "<script>" not in html and "Healthy" not in html


def test_drift_report_shows_later_saved_retraining_action():
    """A saved action override must identify the later request without altering the metric verdict."""
    from skyulf.integrations.databricks.monitoring_output import render_drift_output

    row = {
        "report_id": "report-10",
        "model_name": "m",
        "model_version": "2",
        "observed_at": "now",
        "measured_at": "now",
        "drifted_columns": 0,
        "report_json": json.dumps(
            {
                "metrics": [],
                "performance": {
                    "status": "degraded",
                    "reason": "Metric crossed threshold",
                    "action": "eligible",
                },
            }
        ),
        "performance_action": {
            "action": "retraining_skipped",
            "action_reason": "Training data unchanged <script>",
            "request_id": "request-10",
            "run_id": "run-20",
        },
    }
    html = render_drift_output(row)
    assert "Metric crossed threshold" in html
    assert "retraining_skipped" in html
    assert "Training data unchanged &lt;script&gt;" in html
    assert "request-10" in html and "run-20" in html
    assert "<script>" not in html
