"""Execute shared investigation views against isolated saved monitoring evidence."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_store import (
    monitoring_views,
)


@pytest.fixture
def database():
    """Use recorded tables without registry calls or invented dashboard metrics."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row

    def json_value(raw, path):
        """Return SQL NULL for absent scalar metadata just like Spark."""
        value = json.loads(raw or "{}")
        for key in path[2:].split("."):
            value = value.get(key) if isinstance(value, dict) else None
        return json.dumps(value) if isinstance(value, (list, dict)) else value

    connection.create_function("get_json_object", 2, json_value)
    connection.create_function("concat", -1, lambda *values: "".join(str(v) for v in values))
    connection.create_function("test_now", 0, lambda: "2026-10-11 12:00:00")
    connection.create_function(
        "unix_timestamp",
        1,
        lambda value: (
            datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() if value else None
        ),
    )
    connection.executescript(
        "CREATE TABLE current_health (monitor_id TEXT, environment TEXT, project TEXT, "
        "model_catalog TEXT, model_schema TEXT, model_name TEXT, model_version TEXT, "
        "health_status TEXT, error_message TEXT, measured_at TEXT, observed_at TEXT, "
        "report_id TEXT, mlflow_run_id TEXT);"
        "CREATE TABLE monitoring_results (monitor_id TEXT, environment TEXT, project TEXT, "
        "model_catalog TEXT, model_schema TEXT, model_name TEXT, model_version TEXT, "
        "report_id TEXT, window_end TEXT, measured_at TEXT, evidence_json TEXT, mlflow_run_id TEXT, "
        "status TEXT DEFAULT 'healthy', report_json TEXT DEFAULT '{}');"
        "CREATE TABLE metric_history (report_id TEXT, monitor_id TEXT, model_version TEXT, "
        "category TEXT, column_name TEXT, metric_name TEXT, value REAL, threshold REAL, "
        "has_issue BOOLEAN, status TEXT);"
        "CREATE TABLE model_inventory (monitor_id TEXT, config_digest TEXT, enabled BOOLEAN, "
        "config_json TEXT, expected_interval_hours REAL);"
        "CREATE TABLE performance_history (monitor_id TEXT, config_digest TEXT, model_version TEXT, "
        "report_id TEXT, window_end TEXT, measured_at TEXT, status TEXT, reason TEXT, metric TEXT, "
        "current_value REAL, threshold_value REAL);"
    )
    yield connection
    connection.close()


def enroll(database, monitor="first", *, status="healthy", mode="off", version="2"):
    """Keep the same concrete version in different environments isolated."""
    database.execute(
        "INSERT INTO current_health VALUES (?, 'test', ?, 'models', 'risk', "
        "'models.risk.score', ?, ?, NULL, '2026-10-11 11:00:00', "
        "'2026-10-11 11:00:00', ?, 'observing')",
        (monitor, monitor, version, status, monitor + "-report"),
    )
    database.execute(
        "INSERT INTO model_inventory VALUES (?, 'current', 1, ?, 24)",
        (
            monitor,
            json.dumps(
                {"performance_policy": {"mode": mode, "window_hours": 24, "label_delay_hours": 0}}
            ),
        ),
    )


def query(database, name):
    """Execute shipped SQL with only namespace and current-time syntax adapted."""
    sql = monitoring_views("ops.monitoring")[name]
    sql = sql.replace("`ops`.`monitoring`.", "").replace("current_timestamp()", "test_now()")
    return [dict(row) for row in database.execute(sql)]


def test_issues_keep_recorded_values_and_disabled_policies_out(database):
    """Only recorded flags become metric issues; missing values cannot become zero."""
    enroll(database, status="stale")
    database.execute(
        "INSERT INTO metric_history VALUES ('first-report','first','2','drift','age',"
        "'psi',0.7,0.2,1,'measured')"
    )
    database.execute(
        "INSERT INTO metric_history VALUES ('other-report','first','2','quality','age',"
        "'missing',0.9,0.2,1,'measured')"
    )
    rows = query(database, "monitoring_issues")
    assert {row["source"] for row in rows} == {"health", "drift"}
    metric = next(row for row in rows if row["source"] == "drift")
    assert (metric["value"], metric["threshold"]) == (0.7, 0.2)
    health = next(row for row in rows if row["source"] == "health")
    assert health["value"] is None and health["reason"] == "stale"


def test_policy_issues_match_configuration_version_and_latest_window(database):
    """Latest missing labels must supersede historical loss in exactly one context."""
    enroll(database, mode="report")
    enroll(database, "second", mode="off")
    database.executemany(
        "INSERT INTO performance_history VALUES ('first', ?, ?, ?, ?, "
        "'2026-10-11 11:00:00', ?, ?, 'rmse', ?, 1.1)",
        [
            ("current", "2", "old", "2026-10-10 00:00:00", "degraded", "loss", 2.0),
            (
                "current",
                "2",
                "new",
                "2026-10-11 00:00:00",
                "unavailable",
                "insufficient_labels",
                None,
            ),
            ("old", "2", "wrong-config", "2026-10-11 01:00:00", "healthy", "healthy", 1.0),
            ("current", "1", "wrong-version", "2026-10-11 01:00:00", "healthy", "healthy", 1.0),
        ],
    )
    rows = query(database, "monitoring_issues")
    assert len(rows) == 1
    assert (rows[0]["monitor_id"], rows[0]["report_id"], rows[0]["reason"]) == (
        "first",
        "new",
        "insufficient_labels",
    )
    assert rows[0]["value"] is None


def test_unmeasurable_feature_is_visible_despite_other_healthy_metrics(database):
    """A failed individual measurement must not disappear behind aggregate health."""
    enroll(database)
    database.execute(
        "INSERT INTO metric_history VALUES ('first-report','first','2','drift','rare',"
        "'psi',NULL,0.2,0,'unavailable')"
    )
    rows = query(database, "monitoring_issues")
    assert len(rows) == 1
    assert rows[0]["reason"] == "measurement_unavailable" and rows[0]["value"] is None


def test_details_keep_versions_and_missing_metadata_separate(database):
    """Latest absent metadata cannot resurrect a different version's training lineage."""
    enroll(database)
    evidence = {
        "training_run_id": "train-one",
        "model_digest": "digest-one",
        "model_metadata": {
            "status": "recorded",
            "captured_at": "capture-one",
            "created_at": "created-one",
            "aliases": ["champion"],
            "training_experiment_id": "9",
            "algorithm": "linear_regression",
        },
    }
    database.executemany(
        "INSERT INTO monitoring_results (monitor_id,environment,project,model_catalog,model_schema,"
        "model_name,model_version,report_id,window_end,measured_at,evidence_json,mlflow_run_id) "
        "VALUES ('first','test','first','models','risk',"
        "'models.risk.score', ?, ?, '2026-10-11 00:00:00', '2026-10-11 11:00:00', ?, 'observing')",
        [("1", "report-one", json.dumps(evidence)), ("2", "report-two", "{}")],
    )
    rows = query(database, "model_version_details")
    by_version = {row["model_version"]: row for row in rows}
    assert by_version["1"]["training_run_id"] == "train-one"
    assert by_version["1"]["aliases_at_capture"] == '["champion"]'
    assert by_version["2"]["metadata_status"] == "not_recorded"
    assert by_version["2"]["training_run_id"] is None


def test_never_observed_details_remain_selectable_without_lineage(database):
    """Enrollment alone does not prove a model was measured or its metadata captured."""
    enroll(database, status="never_observed")
    row = query(database, "model_version_details")[0]
    assert row["metadata_status"] == "never_observed" and row["training_run_id"] is None


def test_unresolved_alias_failure_retains_its_report_without_borrowing_lineage(database):
    """A persisted unresolved-version failure must remain observed in its exact context."""
    enroll(database, "failed-context", status="failed", version=None)
    enroll(database, "other-context", version="2")
    database.executemany(
        "INSERT INTO monitoring_results (monitor_id,environment,project,model_catalog,model_schema,"
        "model_name,model_version,report_id,window_end,measured_at,evidence_json,status) "
        "VALUES (?, 'test', ?, 'models','risk','models.risk.score', ?, ?, "
        "'2026-10-11 00:00:00','2026-10-11 11:00:00', ?, ?)",
        [
            ("failed-context", "failed-context", None, "failed-report", "{}", "failed"),
            (
                "other-context",
                "other-context",
                "2",
                "other-report",
                '{"training_run_id":"other-training"}',
                "healthy",
            ),
        ],
    )
    rows = {row["monitor_id"]: row for row in query(database, "model_version_details")}
    failed = rows["failed-context"]
    assert failed["report_id"] == "failed-report"
    assert failed["metadata_status"] == "not_recorded"
    assert failed["model_version"] is None and failed["training_run_id"] is None
    assert rows["other-context"]["training_run_id"] == "other-training"


def test_lineage_selection_never_borrows_another_version_or_context(database):
    """Shared parameters must leave an incompatible model identity with no rows."""
    enroll(database, "first", status="never_observed", version="1")
    enroll(database, "second", status="never_observed", version="2")
    view = monitoring_views("ops.monitoring")["model_version_details"]
    database.execute(
        "CREATE VIEW model_version_details AS " + view.replace("`ops`.`monitoring`.", "")
    )
    path = (
        Path(__file__).resolve().parents[3]
        / "templates/databricks/template/{{.project_name}}/src/monitoring/monitoring.lvdash.json"
    )
    dashboard = json.loads(path.read_text(encoding="utf-8"))
    sql = "".join(
        next(d for d in dashboard["datasets"] if d["name"] == "training_lineage")["queryLines"]
    )
    assert "LIMIT " not in sql.upper() and ":lineage_" not in sql
    identity = {
        "scope_model": "models.risk.score",
        "scope_version": "1",
        "scope_monitor": "test / first / first",
    }
    row = dict(database.execute(sql, identity).fetchone())
    assert row["monitor_id"] == "first" and row["metadata_status"] == "never_observed"
    assert database.execute(sql, {**identity, "scope_version": "2"}).fetchall() == []
    assert database.execute(sql, {**identity, "scope_model": "models.risk.absent"}).fetchall() == []
    all_scope = {
        "scope_model": "All models",
        "scope_version": "All versions",
        "scope_monitor": "All contexts",
    }
    assert len(database.execute(sql, all_scope).fetchall()) == 2


def test_lineage_rows_explicitly_label_unresolved_versions(database):
    """An unresolved enrollment cannot be described as a concrete registered version."""
    enroll(database, status="failed", version=None)
    view = monitoring_views("ops.monitoring")["model_version_details"]
    database.execute(
        "CREATE VIEW model_version_details AS " + view.replace("`ops`.`monitoring`.", "")
    )
    path = (
        Path(__file__).resolve().parents[3]
        / "templates/databricks/template/{{.project_name}}/src/monitoring/monitoring.lvdash.json"
    )
    dashboard = json.loads(path.read_text(encoding="utf-8"))
    sql = "".join(
        next(d for d in dashboard["datasets"] if d["name"] == "training_lineage")["queryLines"]
    )
    dataset = next(d for d in dashboard["datasets"] if d["name"] == "training_lineage")
    defaults = {
        p["keyword"]: p["defaultSelection"]["values"]["values"][0]["value"]
        for p in dataset["parameters"]
    }
    row = dict(database.execute(sql, defaults).fetchone())
    assert row["model_version"] is None
    assert row["selection_status"] == "Selected enrollment; model version unresolved"
