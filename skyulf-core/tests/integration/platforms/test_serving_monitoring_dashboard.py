"""Serving dashboards use saved aggregate reports without joining raw payloads."""

import json
import sqlite3
from pathlib import Path

_DASHBOARD = (
    Path(__file__).resolve().parents[3]
    / "templates/databricks/template/{{.project_name}}/src/monitoring/monitoring.lvdash.json"
)


def _data():
    """Load the shipped dashboard rather than a test-only query copy."""
    return json.loads(_DASHBOARD.read_text(encoding="utf-8"))


def _sql(name):
    """Extract a complete shipped dataset query."""
    return "".join(next(item for item in _data()["datasets"] if item["name"] == name)["queryLines"])


def test_serving_page_shows_bounded_aggregate_context_and_async_empty_state():
    """Operators need status, volume, latency, and identity without payload disclosure."""
    dashboard = _data()
    page = next(item for item in dashboard["pages"] if item["name"] == "serving")
    assert page["layoutVersion"] == "GRID_V1"
    assert [(item["position"]["y"], item["position"]["height"]) for item in page["layout"]] == [
        (0, 1),
        (1, 1),
        (2, 7),
        (9, 7),
    ]
    widgets = {item["widget"]["name"]: item for item in page["layout"]}
    assert (
        "asynchronous"
        in " ".join(widgets["serving_note"]["widget"]["multilineTextboxSpec"]["lines"]).lower()
    )
    assert {"serving_current", "serving_windows"} <= widgets.keys()
    for name in ("serving_current", "serving_windows"):
        item = widgets[name]
        assert item["position"]["x"] == 0 and item["position"]["width"] == 12
        assert item["position"]["height"] <= 8
        assert item["widget"]["spec"]["widgetType"] == "table"
        fields = {column["fieldName"] for column in item["widget"]["spec"]["encodings"]["columns"]}
        assert {
            "monitor_id",
            "endpoint",
            "served_version",
            "success_count",
            "error_count",
            "latency_mean_ms",
        } <= fields
    for name in ("serving_current", "serving_windows"):
        dataset = next(item for item in dashboard["datasets"] if item["name"] == name)
        assert len(dataset["queryLines"]) <= 8
        sql = _sql(name)
        assert "FROM model_inventory" in sql and "monitoring_results" in sql
        assert "get_json_object" in sql
        assert "request" not in sql.lower().replace("request_count", "")
        assert "response" not in sql.lower()


def test_latest_serving_status_does_not_reuse_old_success_after_failure():
    """A failed new observation must clear old request counts for that monitor."""
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row

        def json_value(raw, path):
            """Match Spark's null result for absent nested JSON fields."""
            value = json.loads(raw) if raw is not None else None
            for name in path[2:].split("."):
                value = value.get(name) if isinstance(value, dict) else None
            return str(value) if value is not None else None

        connection.create_function("get_json_object", 2, json_value)
        connection.execute(
            "CREATE TABLE model_inventory (monitor_id TEXT, model_name TEXT, enabled INTEGER, config_json TEXT, updated_at INTEGER, config_digest TEXT)"
        )
        connection.execute(
            "CREATE TABLE monitoring_results (monitor_id TEXT, report_id TEXT, window_start INTEGER, window_end INTEGER, measured_at INTEGER, status TEXT, report_json TEXT, config_digest TEXT)"
        )
        connection.executemany(
            "INSERT INTO model_inventory VALUES (?,?,?,?,?,?)",
            [
                (
                    "online",
                    "models.risk.churn",
                    1,
                    '{"model_version":"2","serving_endpoint":"risk-v2"}',
                    1,
                    "current-online",
                ),
                (
                    "other",
                    "models.risk.churn",
                    1,
                    '{"model_version":"3","serving_endpoint":"risk-v3"}',
                    2,
                    "current-other",
                ),
                ("batch", "models.risk.churn", 1, '{"model_version":"2"}', 3, "batch"),
            ],
        )
        connection.executemany(
            "INSERT INTO monitoring_results VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    "online",
                    "old",
                    0,
                    1,
                    1,
                    "healthy",
                    '{"serving":{"request_count":10,"success_count":9,"error_count":1,"latency_mean_ms":5}}',
                    "current-online",
                ),
                ("online", "new", 1, 2, 2, "failed", "{}", "current-online"),
                ("online", "policy", 2, 3, 3, "no_data", '{"performance":{}}', "current-online"),
                (
                    "online",
                    "old-policy",
                    3,
                    4,
                    4,
                    "healthy",
                    '{"serving":{"request_count":99,"success_count":99,"error_count":0}}',
                    "previous-online",
                ),
                (
                    "other",
                    "other-report",
                    0,
                    1,
                    1,
                    "healthy",
                    '{"serving":{"request_count":20,"success_count":20,"error_count":0,"latency_mean_ms":3}}',
                    "current-other",
                ),
                ("batch", "batch-report", 0, 1, 1, "healthy", "{}", "batch"),
            ],
        )
        rows = {row["monitor_id"]: row for row in connection.execute(_sql("serving_current"))}
        assert set(rows) == {"online", "other"}
        assert rows["online"]["status"] == "failed"
        assert rows["online"]["request_count"] is None
        assert rows["other"]["request_count"] == 20
        assert rows["other"]["served_version"] == "3"
        windows = list(connection.execute(_sql("serving_windows")))
        assert {(row["monitor_id"], row["report_id"]) for row in windows} == {
            ("online", "old"),
            ("online", "new"),
            ("other", "other-report"),
        }
