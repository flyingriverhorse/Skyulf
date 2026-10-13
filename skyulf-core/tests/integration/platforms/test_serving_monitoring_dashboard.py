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


def test_serving_operational_series_keep_legacy_counts_and_unavailable_percentiles():
    """New charts must not invent status classes or turn absent timings into zero latency."""

    def scalar(raw, path):
        """Match Spark JSON scalar access while keeping array payloads serialized."""
        value = json.loads(raw or "{}")
        for part in path[2:].split("."):
            value = value.get(part) if isinstance(value, dict) else None
        return json.dumps(value) if isinstance(value, (dict, list)) else value

    complete = {
        "request_count": 10,
        "error_count": 4,
        "status_4xx_count": 2,
        "status_5xx_count": 1,
        "status_other_count": 1,
        "latency_sample_count": 9,
        "latency_mean_ms": 15,
        "latency_p50_ms": 12,
        "latency_p95_ms": 20,
        "latency_p99_ms": 30,
        "logging_error_count": 2,
    }
    cases = [
        ("new", "healthy", complete),
        ("legacy", "healthy", {"request_count": 10, "error_count": 3, "latency_mean_ms": 5}),
        ("partial", "healthy", {"request_count": 10, "error_count": 3, "status_4xx_count": 2}),
        ("inconsistent", "healthy", complete | {"status_other_count": 9}),
        ("negative", "healthy", complete | {"status_4xx_count": -1, "status_other_count": 4}),
        ("failed", "failed", complete),
        ("zero", "healthy", {"request_count": 0, "error_count": 0}),
        ("null_logging", "healthy", complete | {"logging_error_count": None}),
        (
            "http_success",
            "degraded",
            complete
            | {
                "error_count": 0,
                "status_4xx_count": 0,
                "status_5xx_count": 0,
                "status_other_count": 0,
            },
        ),
        (
            "zero_timing",
            "healthy",
            complete | {"latency_p95_ms": 0, "latency_sample_count": 0, "logging_error_count": 0},
        ),
    ]
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        connection.create_function("get_json_object", 2, scalar)
        connection.create_function("concat", -1, lambda *parts: "".join(str(p) for p in parts))
        connection.create_function("date_format", 2, lambda value, pattern: str(value))
        connection.execute(
            "CREATE TABLE model_inventory (monitor_id, model_name, enabled, updated_at, config_digest, environment, project, config_json)"
        )
        connection.execute(
            "INSERT INTO model_inventory VALUES ('m','model',1,1,'config','demo','serving',?)",
            (json.dumps({"model_version": "1", "serving_endpoint": "endpoint"}),),
        )
        connection.execute(
            "CREATE TABLE monitoring_results (monitor_id, model_version, report_id, window_start, window_end, measured_at, status, report_json, config_digest)"
        )
        for name, status, summary in cases:
            connection.execute(
                "INSERT INTO monitoring_results VALUES ('m','1',?,0,1,2,?,?, 'config')",
                (name, status, json.dumps({"serving": summary})),
            )
        params = {"serving_model": "", "serving_version": "", "serving_monitor": ""}
        rows = {r["report_id"]: r for r in connection.execute(_sql("serving_windows"), params)}
        assert [rows["new"][key] for key in ("plot_4xx", "plot_5xx", "plot_other")] == [2, 1, 1]
        assert rows["new"]["latency_p50_ms"] == 12 and rows["new"]["latency_sample_count"] == 9
        assert float(rows["new"]["latency_p95_display"]) == 20
        assert float(rows["zero_timing"]["latency_p95_display"]) == 0
        assert str(rows["zero_timing"]["latency_samples_display"]) == "0"
        assert rows["new"]["logging_error_count"] == 2
        assert str(rows["new"]["logging_errors_display"]) == "2"
        assert rows["zero_timing"]["logging_error_count"] == 0
        assert str(rows["zero_timing"]["logging_errors_display"]) == "0"
        for name in ("legacy", "null_logging"):
            assert rows[name]["logging_error_count"] is None
            assert rows[name]["logging_errors_display"] == "Not recorded"
        assert rows["http_success"]["logging_error_count"] == 2
        assert rows["http_success"]["error_count"] == 0
        assert all(rows["http_success"][key] == 0 for key in ("plot_4xx", "plot_5xx", "plot_other"))
        series = list(connection.execute(_sql("serving_chart_series"), params))
        plotted = {(row["report_id"], row["series"]): row["metric_value"] for row in series}
        assert {key[1]: value for key, value in plotted.items() if key[0] == "new"} == {
            "4xx client": 2,
            "5xx server": 1,
            "Other / class not recorded": 1,
            "Average (mean)": 15,
            "Slow requests (p95)": 20,
        }
        for name in ("legacy", "partial", "inconsistent", "negative"):
            assert (name, "4xx client") not in plotted
            assert (name, "5xx server") not in plotted
            assert plotted[name, "Other / class not recorded"] == rows[name]["error_count"]
        assert ("legacy", "Slow requests (p95)") not in plotted
        assert plotted["zero_timing", "Slow requests (p95)"] == 0
        assert not {row["report_id"] for row in series} & {"failed", "zero"}
        assert len(plotted) == len(series)
        assert all(row["metric_value"] is not None for row in series)
        latest = list(connection.execute(_sql("serving_current"), params))
        assert len(latest) == 1 and latest[0]["report_id"] == "zero_timing"
        assert latest[0]["logging_error_count"] == 0
        assert str(latest[0]["logging_errors_display"]) == "0"
        for name, expected in (
            ("new", "2"),
            ("legacy", "Not recorded"),
            ("null_logging", "Not recorded"),
        ):
            connection.execute("DELETE FROM monitoring_results")
            summary = next(summary for case, _, summary in cases if case == name)
            connection.execute(
                "INSERT INTO monitoring_results VALUES ('m','1',?,0,1,2,'healthy',?,'config')",
                (name, json.dumps({"serving": summary})),
            )
            row = connection.execute(_sql("serving_current"), params).fetchone()
            assert str(row["logging_errors_display"]) == expected
        for name in ("legacy", "partial"):
            assert rows[name]["plot_4xx"] is rows[name]["plot_5xx"] is None
            assert rows[name]["plot_other"] == 3
            assert rows[name]["latency_p95_ms"] is rows[name]["latency_sample_count"] is None
            assert all(
                rows[name][field] == "Not recorded"
                for field in (
                    "latency_p50_display",
                    "latency_p95_display",
                    "latency_p99_display",
                    "latency_samples_display",
                )
            )
        for name in ("inconsistent", "negative"):
            assert rows[name]["plot_4xx"] is rows[name]["plot_5xx"] is None
            assert rows[name]["plot_other"] == 4
        for name in ("failed", "zero"):
            assert (
                rows[name]["plot_4xx"] is rows[name]["plot_5xx"] is rows[name]["plot_other"] is None
            )


def test_serving_all_models_and_exact_scope_filter_before_history_truncation():
    """Another model's newer windows cannot hide the selected model or leak into its results."""
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row

        def json_value(raw, path):
            """Read nested serving counters just as the native dataset does."""
            value = json.loads(raw or "{}")
            for part in path[2:].split("."):
                value = value.get(part) if isinstance(value, dict) else None
            return value

        connection.create_function("get_json_object", 2, json_value)
        connection.create_function("concat", -1, lambda *args: "".join(str(v) for v in args))
        connection.create_function("date_format", 2, lambda value, pattern: str(value))
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
                "INSERT INTO monitoring_results VALUES (?,? ,?,0,?,?,'healthy',"
                '\'{"serving":{"request_count":2,"latency_mean_ms":4,"latency_p95_ms":7}}\',\'config\')',
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

        parameters.update(serving_model="catalog.schema.a", serving_version="2")
        series = list(connection.execute(_sql("serving_chart_series"), parameters))
        assert {(row["monitor_id"], row["series"], row["metric_value"]) for row in series} == {
            ("selected", "Average (mean)", 4),
            ("selected", "Slow requests (p95)", 7),
        }
        parameters["serving_version"] = "7"
        assert not list(connection.execute(_sql("serving_chart_series"), parameters))
        parameters.update(serving_version="2", serving_monitor="prod / project / other")
        assert not list(connection.execute(_sql("serving_chart_series"), parameters))
        parameters.update(serving_model="", serving_version="", serving_monitor="")

        widget = next(
            i["widget"]
            for page in _data()["pages"]
            for i in page["layout"]
            if i["widget"]["name"] == "serving_latency_trend"
        )
        predicate = " AND ".join(f["expression"] for f in widget["queries"][0]["query"]["filters"])
        plotted_sql = f"SELECT * FROM ({_sql('serving_chart_series')}) scoped WHERE {predicate}"
        plotted = list(connection.execute(plotted_sql, parameters))
        assert len(plotted) == 26
        assert {r["report_id"] for r in plotted if r["monitor_id"] == "other"} == {
            f"other-{i}" for i in range(298, 310)
        }
        selected = list(
            connection.execute(plotted_sql, parameters | {"serving_model": "catalog.schema.a"})
        )
        assert len(selected) == 2 and {row["report_id"] for row in selected} == {"selected-1"}


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
        assert "request" not in sql.lower().replace("request_count", "").replace(
            "rows_per_request", ""
        )
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
        connection.create_function("date_format", 2, lambda value, pattern: str(value))
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
        assert rows["online"]["logging_error_count"] is None
        assert rows["online"]["logging_errors_display"] == "Not recorded"
        assert rows["other"]["request_count"] == 20
        assert rows["other"]["served_version"] == "3"
        windows = list(connection.execute(_sql("serving_windows"), params))
        assert {(row["monitor_id"], row["report_id"]) for row in windows} == {
            ("online", "old"),
            ("online", "new"),
            ("other", "other-report"),
        }


def test_serving_distribution_and_series_follow_filters_and_distinct_colors():
    """Serving charts must preserve selection scope and distinguish operational categories."""
    dashboard = _data()
    widgets = {i["widget"]["name"]: i["widget"] for p in dashboard["pages"] for i in p["layout"]}
    for name in ("filter_model", "filter_version", "filter_context", "filter_serving_window"):
        assert {"serving_latency_distribution", "serving_chart_series"} <= {
            q["query"]["datasetName"] for q in widgets[name]["queries"]
        }
    errors = widgets["serving_error_trend"]["spec"]
    assert errors["widgetType"] == "bar" and errors["mark"]["layout"] == "group"
    assert errors["encodings"]["x"]["fieldName"] == "observation_label"
    colors = {m["value"]: m["color"] for m in errors["encodings"]["color"]["scale"]["mappings"]}
    assert colors == {
        "4xx client": "#3B82F6",
        "5xx server": "#E5484D",
        "Other / class not recorded": "#7C8CA0",
    }
    assert errors["encodings"]["y"]["fieldName"] == "metric_value"
    assert errors["encodings"]["color"]["fieldName"] == "series"
    assert errors["encodings"]["color"]["legend"]["position"] == "bottom"
    assert errors["encodings"]["color"]["legend"]["title"] == "HTTP error class"
    assert errors["encodings"]["label"]["show"] is True
    assert "Blue: 4xx" in errors["frame"]["description"]
    assert "Red: 5xx" in errors["frame"]["description"]
    chart = widgets["serving_latency_trend"]["spec"]
    assert chart["widgetType"] == "bar" and chart["mark"]["layout"] == "group"
    latency = chart["encodings"]
    assert latency["x"]["fieldName"] == "observation_label"
    assert "lower is faster" in latency["y"]["displayName"]
    assert latency["y"]["fieldName"] == "metric_value"
    assert latency["color"]["fieldName"] == "series"
    for name, kind in (("serving_error_trend", "errors"), ("serving_latency_trend", "latency")):
        widget = widgets[name]
        query = widget["queries"][0]["query"]
        assert query["datasetName"] == "serving_chart_series"
        assert any(f"metric_kind = '{kind}'" in f["expression"] for f in query["filters"])
        assert widget["spec"]["encodings"]["y"]["scale"]["domain"] == {"type": "per-facet"}
        assert widget["spec"]["encodings"]["label"]["show"] is True
    assert {
        "latency_p50_display",
        "latency_p95_display",
        "latency_p99_display",
        "latency_max_ms",
        "latency_samples_display",
    } <= {f["fieldName"] for f in latency["extra"]}
    matrix = widgets["serving_latency_distribution"]["spec"]
    assert matrix["widgetType"] == "heatmap"
    assert matrix["encodings"]["color"]["scale"]["domain"] == {"min": 0, "max": 1}
    assert matrix["facet"]["fieldName"] == "model_identity"


def test_logging_exclusions_are_bound_to_tables_and_volume_tooltip_only():
    """Logging exclusions must be visible without becoming another HTTP error series."""
    widgets = {i["widget"]["name"]: i["widget"] for p in _data()["pages"] for i in p["layout"]}
    field = "logging_errors_display"
    for name in ("serving_current", "serving_windows", "serving_volume_trend"):
        widget = widgets[name]
        assert {"name": field, "expression": f"`{field}`"} in widget["queries"][0]["query"][
            "fields"
        ]
        encodings = widget["spec"]["encodings"]
        entries = encodings["extra" if name == "serving_volume_trend" else "columns"]
        column = next(item for item in entries if item["fieldName"] == field)
        assert column == {"fieldName": field, "displayName": "Requests with logging errors"}
        if name != "serving_volume_trend":
            assert (
                entries.index(column)
                == next(i for i, item in enumerate(entries) if item["fieldName"] == "error_count")
                + 1
            )
        assert (
            "payloads are excluded from drift and performance"
            in widget["spec"]["frame"]["description"]
        )
    assert widgets["serving_volume_trend"]["spec"]["encodings"]["y"]["fieldName"] == "request_count"
    assert field not in json.dumps(widgets["serving_error_trend"])
