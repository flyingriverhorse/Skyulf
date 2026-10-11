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


def test_serving_all_models_and_exact_scope_filter_before_history_truncation():
    """Another model's newer windows cannot hide the selected model or leak into its results."""
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        connection.create_function(
            "get_json_object", 2, lambda raw, path: json.loads(raw or "{}").get(path[2:])
        )
        connection.create_function("concat", -1, lambda *args: "".join(str(v) for v in args))
        connection.execute(
            "CREATE TABLE model_inventory (monitor_id TEXT, model_name TEXT, enabled INTEGER, "
            "config_json TEXT, updated_at INTEGER, config_digest TEXT, environment TEXT, project TEXT)"
        )
        connection.execute(
            "CREATE TABLE monitoring_results (monitor_id TEXT, model_version TEXT, report_id TEXT, "
            "window_start INTEGER, window_end INTEGER, measured_at INTEGER, status TEXT, "
            "report_json TEXT, config_digest TEXT)"
        )
        for monitor, model, version, parent, enabled in (
            ("selected", "catalog.schema.a", "2", "7", 1),
            ("other", "catalog.schema.b", "2", "2", 1),
            ("disabled", "catalog.schema.a", "2", "7", 0),
        ):
            config = json.dumps(
                {"model_version": version, "model_set_version": parent, "serving_endpoint": monitor}
            )
            connection.execute(
                "INSERT INTO model_inventory VALUES (?,?,?,?,1,'config','prod','project')",
                (monitor, model, enabled, config),
            )
            connection.executemany(
                "INSERT INTO monitoring_results VALUES (?,? ,?,0,?,?,'healthy','{}','config')",
                [
                    (monitor, version, f"{monitor}-{i}", i, i)
                    for i in (range(200, 310) if monitor == "other" else [1])
                ],
            )
        parameters = {"serving_model": "", "serving_version": "", "serving_monitor": ""}
        for dataset in ("serving_current", "serving_windows"):
            assert len(list(connection.execute(_sql(dataset), parameters))) == (
                111 if dataset == "serving_windows" else 2
            )
            parameters["serving_model"] = "catalog.schema.a"
            rows = list(connection.execute(_sql(dataset), parameters))
            assert len(rows) == 1 and rows[0]["monitor_id"] == "selected"
            assert rows[0]["model_version"] == "2" and rows[0]["served_version"] == "7"
            parameters["serving_version"] = "7"
            assert list(connection.execute(_sql(dataset), parameters)) == []
            parameters.update(serving_version="2", serving_monitor="prod / project / selected")
            assert len(list(connection.execute(_sql(dataset), parameters))) == 1
            parameters["serving_monitor"] = "prod / project / other"
            assert list(connection.execute(_sql(dataset), parameters)) == []
            parameters.update(serving_model="", serving_version="", serving_monitor="")


def test_serving_page_shows_bounded_aggregate_context_and_async_empty_state():
    """Operators need status, volume, latency, and identity without payload disclosure."""
    dashboard = _data()
    page = next(item for item in dashboard["pages"] if item["name"] == "serving")
    assert page["layoutVersion"] == "GRID_V1"
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
            "monitor_context",
            "endpoint",
            "success_count",
            "error_count",
            "latency_mean_ms",
        } <= fields
    for name in ("serving_current", "serving_windows"):
        dataset = next(item for item in dashboard["datasets"] if item["name"] == name)
        assert {p["keyword"] for p in dataset["parameters"]} == {
            "serving_model",
            "serving_version",
            "serving_monitor",
        }
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
        connection.create_function("concat", -1, lambda *args: "".join(str(v) for v in args))
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
        connection.execute("ALTER TABLE model_inventory ADD environment TEXT DEFAULT 'prod'")
        connection.execute("ALTER TABLE model_inventory ADD project TEXT DEFAULT 'project'")
        connection.execute("ALTER TABLE monitoring_results ADD model_version TEXT")
        params = {
            "serving_model": "models.risk.churn",
            "serving_version": "",
            "serving_monitor": "",
        }
        rows = {
            row["monitor_id"]: row for row in connection.execute(_sql("serving_current"), params)
        }
        assert set(rows) == {"online", "other"}
        assert rows["online"]["status"] == "failed"
        assert rows["online"]["request_count"] is None
        assert rows["other"]["request_count"] == 20
        assert rows["other"]["served_version"] == "3"
        windows = list(connection.execute(_sql("serving_windows"), params))
        assert {(row["monitor_id"], row["report_id"]) for row in windows} == {
            ("online", "old"),
            ("online", "new"),
            ("other", "other-report"),
        }
