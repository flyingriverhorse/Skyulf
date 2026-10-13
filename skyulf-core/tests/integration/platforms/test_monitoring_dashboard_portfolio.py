"""Multi-model dashboard evidence must retain each identity and its own policy metric."""

import json
import sqlite3
from pathlib import Path

import pytest

ASSET = (
    Path(__file__).resolve().parents[3]
    / "templates/databricks/template/{{.project_name}}/src/monitoring/monitoring.lvdash.json"
)


@pytest.mark.parametrize(
    "widget_name",
    ["drift_psi", "drift_trend", "performance_trend", "performance_policy_trend"],
)
def test_model_panels_reflow_inside_a_compact_scrollable_widget(widget_name: str) -> None:
    """Filtering to fewer identities must not leave fixed-width plots in a portfolio-sized box."""
    dashboard = json.loads(ASSET.read_text(encoding="utf-8"))
    entry = next(
        entry
        for page in dashboard["pages"]
        for entry in page["layout"]
        if entry["widget"]["name"] == widget_name
    )
    spec = entry["widget"]["spec"]
    assert spec["widgetType"] in {"bar", "scatter"}
    assert spec["facet"]["layout"] == {
        "column": {"type": "count", "value": 2},
        "row": {"type": "count", "value": 2},
    }
    assert entry["position"]["height"] <= 12
    numeric_axis = "x" if widget_name == "drift_psi" else "y"
    assert spec["encodings"][numeric_axis]["scale"]["domain"] == {"type": "per-facet"}


def _dataset(name: str) -> str:
    """Read the shipped query instead of duplicating its selection logic in tests."""
    dashboard = json.loads(ASSET.read_text(encoding="utf-8"))
    return "".join(next(d for d in dashboard["datasets"] if d["name"] == name)["queryLines"])


@pytest.mark.parametrize(
    "widget_name",
    ["drift_psi", "drift_trend", "performance_trend", "performance_policy_trend"],
)
def test_facet_identity_is_a_dataset_column(widget_name: str) -> None:
    """Cached dashboard results need a real grouping column to populate model panels."""
    dashboard = json.loads(ASSET.read_text(encoding="utf-8"))
    widget = next(
        entry["widget"]
        for page in dashboard["pages"]
        for entry in page["layout"]
        if entry["widget"]["name"] == widget_name
    )
    spec = widget["spec"]
    facet_field = (
        spec["facet"]["fieldName"]
        if "facet" in spec
        else json.loads(spec["jsonSpec"]["spec"])["facet"]["field"]
    )
    query = widget["queries"][0]["query"]
    field = next(field for field in query["fields"] if field["name"] == facet_field)
    assert field["expression"] == f"`{facet_field}`"
    assert f"AS {facet_field}" in _dataset(query["datasetName"])


def _connection() -> sqlite3.Connection:
    """Provide independent identities, including one with no measured policy."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.create_function(
        "concat", -1, lambda *parts: None if None in parts else "".join(str(p) for p in parts)
    )
    connection.execute(
        "CREATE TABLE chosen_context (monitor_id TEXT, model_name TEXT, model_version TEXT, "
        "monitor_context TEXT)"
    )
    connection.executemany(
        "INSERT INTO chosen_context VALUES (?,?,?,?)",
        [
            ("a", "model-a", "1", "prod-a"),
            ("b", "model-b", "1", "prod-b"),
            ("c", "model-a", "2", "prod-a-v2"),
            ("none", "model-c", None, "new"),
        ],
    )
    connection.execute(
        "CREATE TABLE performance_history (monitor_id TEXT, model_name TEXT, model_version TEXT, "
        "report_id TEXT, window_start INTEGER, window_end INTEGER, measured_at INTEGER, "
        "metric TEXT, status TEXT, reason TEXT, action TEXT, baseline_value REAL, "
        "current_value REAL, threshold_value REAL, direction TEXT, label_coverage REAL)"
    )
    connection.executemany(
        "INSERT INTO performance_history VALUES (?,?,?,?,0,?,?,?,?,?,?,?,?,?,?,1)",
        [
            (
                "a",
                "model-a",
                "1",
                "a-current",
                2,
                2,
                "accuracy",
                "healthy",
                "within_tolerance",
                "report",
                0.96,
                0.9,
                0.86,
                "higher",
            ),
            (
                "b",
                "model-b",
                "1",
                "b-current",
                2,
                3,
                "f1",
                "degraded",
                "loss",
                "report",
                0.86,
                0.78,
                0.81,
                "higher",
            ),
            (
                "b",
                "model-b",
                "1",
                "b-backfill",
                1,
                100,
                "accuracy",
                "healthy",
                "historical_window",
                "report",
                0.95,
                0.95,
                0.85,
                "higher",
            ),
            (
                "c",
                "model-a",
                "2",
                "c-disabled",
                3,
                3,
                None,
                "disabled",
                "policy_off",
                "none",
                None,
                None,
                None,
                None,
            ),
        ],
    )
    connection.execute("ALTER TABLE performance_history ADD COLUMN model_catalog TEXT")
    connection.execute("ALTER TABLE performance_history ADD COLUMN model_schema TEXT")
    connection.execute(
        "ALTER TABLE performance_history ADD COLUMN tolerance_mode TEXT DEFAULT 'absolute'"
    )
    connection.execute("ALTER TABLE performance_history ADD COLUMN tolerance REAL")
    connection.execute(
        "UPDATE performance_history SET tolerance = abs(baseline_value - threshold_value)"
    )
    return connection


def test_latest_policy_keeps_every_identity_and_missing_policy_evidence() -> None:
    """A newer F1 model must not hide the accuracy model, disabled version or unmeasured model."""
    tail = _dataset("performance_policy_snapshot").split("), policy_ranked AS (", 1)[1]
    with _connection() as connection:
        rows = connection.execute("WITH policy_ranked AS (" + tail).fetchall()
    assert len(rows) == 4
    by = {row["monitor_id"]: row for row in rows}
    assert set(by) == {"a", "b", "c", "none"}
    assert (by["a"]["metric"], by["a"]["current_value"]) == ("accuracy", 0.9)
    assert (by["b"]["metric"], by["b"]["current_value"]) == ("f1", 0.78)
    assert by["c"]["policy_status"] == "Policy disabled"
    assert by["none"]["policy_status"] == "No policy evidence"


def test_policy_history_selects_each_identitys_own_latest_metric() -> None:
    """Accuracy and F1 histories must coexist without a late backfill changing the chosen metric."""
    tail = _dataset("performance_policy_series").split("), selected_metric AS (", 1)[1]
    with _connection() as connection:
        rows = connection.execute("WITH selected_metric AS (" + tail).fetchall()
    assert {(r["monitor_id"], r["metric_name"]) for r in rows} == {("a", "accuracy"), ("b", "f1")}
    assert {r["report_id"] for r in rows} == {"a-current", "b-current"}
    assert len(rows) == 6


def test_unresolved_version_keeps_a_readable_portfolio_identity() -> None:
    """A failed alias lookup must not turn its entire model label into SQL NULL."""
    dashboard = json.loads(ASSET.read_text(encoding="utf-8"))
    widget = next(
        entry["widget"]
        for page in dashboard["pages"]
        for entry in page["layout"]
        if entry["widget"]["name"] == "performance_policy_summary"
    )
    expression = next(
        field["expression"]
        for field in widget["queries"][0]["query"]["fields"]
        if field["name"] == "model_identity"
    )
    tail = _dataset("performance_policy_snapshot").split("), policy_ranked AS (", 1)[1]
    with _connection() as connection:
        label = connection.execute(
            f"SELECT {expression} FROM (WITH policy_ranked AS ({tail}) WHERE monitor_id = 'none'"
        ).fetchone()[0]
    assert label == "model-c / vunresolved / none"


def test_unresolved_version_retains_its_saved_failed_observation() -> None:
    """An unresolved alias failure must retain its report and error instead of appearing unobserved."""
    tail = _dataset("confusion_status").split("), ranked_reports AS (", 1)[1]
    with _connection() as connection:
        connection.create_function(
            "get_json_object", 2, lambda value, path: json.loads(value).get(path.removeprefix("$."))
        )
        connection.execute(
            "CREATE TABLE monitoring_results (monitor_id TEXT, model_name TEXT, model_version TEXT, "
            "report_id TEXT, window_end INTEGER, measured_at INTEGER, status TEXT, "
            "report_json TEXT, error_message TEXT)"
        )
        connection.execute(
            "INSERT INTO monitoring_results VALUES "
            "('none', 'model-c', NULL, 'failed-report', 2, 3, 'failed', '{}', "
            "'Model version could not be resolved')"
        )
        rows = connection.execute("WITH ranked_reports AS (" + tail).fetchall()
    result = next(row for row in rows if row["monitor_id"] == "none")
    assert result["matrix_status"] == "failed"
    assert result["report_id"] == "failed-report"
    assert result["matrix_reason"] == "Model version could not be resolved"
