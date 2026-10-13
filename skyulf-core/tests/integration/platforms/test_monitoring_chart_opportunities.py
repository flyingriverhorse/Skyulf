"""Execute dashboard calculations without inventing observations or merging identities."""

import json
import re
import sqlite3
from pathlib import Path

import pytest

ASSET = (
    Path(__file__).resolve().parents[3]
    / "templates/databricks/template/{{.project_name}}/src/monitoring/monitoring.lvdash.json"
)


def _dataset(name):
    """Read the actual shipped query so SQL regressions fail locally."""
    dashboard = json.loads(ASSET.read_text(encoding="utf-8"))
    return next(dataset for dataset in dashboard["datasets"] if dataset["name"] == name)


def _insert(connection, table, **values):
    """Populate only the evidence relevant to a scenario; all other fields remain null."""
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    connection.execute(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(values.values())
    )


@pytest.fixture
def database():
    """Provide recorded reports and metrics with the same identity columns as the store."""
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        connection.create_function(
            "concat", -1, lambda *parts: "".join(str(p or "") for p in parts)
        )
        connection.create_function("date_format", 2, lambda value, pattern: str(value)[:16])
        connection.create_function(
            "substring_index", 3, lambda value, sep, count: value.rsplit(sep, -count)[count]
        )
        connection.create_function("get_json_object", 2, _json_value)
        identity = (
            "monitor_id TEXT, model_catalog TEXT, model_schema TEXT, model_name TEXT, "
            "model_version TEXT, environment TEXT, project TEXT, measured_at TEXT"
        )
        observation = (
            identity + ", report_id TEXT, window_start TEXT, window_end TEXT, observed_at TEXT"
        )
        connection.execute(f"CREATE TABLE current_health ({identity})")
        connection.execute(
            f"CREATE TABLE monitoring_results ({observation}, status TEXT, report_json TEXT, "
            "current_rows INTEGER, scored_rows INTEGER, labeled_rows INTEGER, label_coverage REAL, "
            "config_digest TEXT)"
        )
        connection.execute(
            f"CREATE TABLE metric_history ({observation}, column_name TEXT, measurement_status TEXT, "
            "category TEXT, metric_name TEXT, value REAL, threshold REAL, status TEXT, has_issue INTEGER)"
        )
        connection.execute(
            "CREATE TABLE model_inventory (monitor_id TEXT, model_name TEXT, enabled INTEGER, "
            "updated_at TEXT, config_digest TEXT, environment TEXT, project TEXT, config_json TEXT)"
        )
        connection.execute(
            f"CREATE TABLE performance_history ({observation}, status TEXT, reason TEXT, metric TEXT, "
            "direction TEXT, baseline_kind TEXT, baseline_reference TEXT, baseline_value REAL, "
            "current_value REAL, absolute_degradation REAL, relative_degradation REAL, tolerance REAL, "
            "tolerance_mode TEXT, threshold_value REAL, labeled_rows INTEGER, label_coverage REAL, "
            "consecutive_failures INTEGER, required_windows INTEGER, action TEXT, action_reason TEXT, "
            "request_id TEXT, run_id TEXT)"
        )
        yield connection


def _json_value(raw, path):
    """Match nested JSON scalar lookup, including missing historical fields."""
    value = json.loads(raw or "{}")
    for part in path[2:].split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return json.dumps(value) if isinstance(value, (dict, list)) else value


def _report(connection, report_id="r1", **overrides):
    """Keep repeated windows as distinct immutable reports."""
    values = {
        "monitor_id": "monitor-a",
        "model_catalog": "catalog",
        "model_schema": "schema",
        "model_name": "model-a",
        "model_version": "1",
        "environment": "prod",
        "project": "project",
        "report_id": report_id,
        "window_start": "2026-10-01 00:00:00",
        "window_end": "2026-10-02 00:00:00",
        "measured_at": "2026-10-03 00:00:00",
        "status": "healthy",
        "report_json": "{}",
        "config_digest": "config",
    }
    values.update(overrides)
    if "report_json" not in overrides:
        values["report_json"] = json.dumps(
            {
                key: values[key]
                for key in ("current_rows", "scored_rows", "labeled_rows")
                if key in values
            }
        )
    _insert(connection, "monitoring_results", **values)
    return {
        k: values[k]
        for k in (
            "monitor_id",
            "model_catalog",
            "model_schema",
            "model_name",
            "model_version",
            "report_id",
            "window_start",
            "window_end",
            "measured_at",
            "environment",
            "project",
        )
    }


def _execute(connection, name, **overrides):
    """Run portable SQL; native warehouse tests cover Spark-specific execution too."""
    dataset = _dataset(name)
    parameters = {
        p["keyword"]: p["defaultSelection"]["values"]["values"][0]["value"]
        for p in dataset.get("parameters", [])
    }
    parameters.update(overrides)
    sql = "".join(dataset["queryLines"]).replace(" AS STRING", " AS TEXT")
    return [dict(row) for row in connection.execute(sql, parameters)]


def test_categorical_and_numeric_psi_use_the_same_saved_value_and_limit(database):
    """Categorical drift must appear beside numeric drift in both table and chart."""
    identity = _report(database)
    for feature, metric in (("age", "psi"), ("segment", "psi_categorical")):
        _insert(
            database,
            "metric_history",
            **identity,
            column_name=feature,
            category="drift",
            metric_name=metric,
            value=0.31,
            threshold=0.2,
        )
    rows = _execute(database, "drift_psi_chart")
    assert {(r["column_name"], r["series"], r["plot_value"]) for r in rows} == {
        (feature, series, value)
        for feature in ("age", "segment")
        for series, value in (("Measured PSI", 0.31), ("PSI limit", 0.2))
    }
    sql = "".join(_dataset("metric_history")["queryLines"])
    expressions = re.findall(r"MAX\(CASE WHEN metric_name .*? AS psi(?:_threshold)?[ ,\n]", sql)
    select = ", ".join(expression.rstrip(", \n") for expression in expressions)
    table = database.execute(
        f"SELECT column_name, {select} FROM metric_history GROUP BY column_name"
    )
    assert [tuple(row) for row in table] == [("age", 0.31, 0.2), ("segment", 0.31, 0.2)]


def test_serving_ratios_preserve_zero_missing_failure_and_version_identity(database):
    """A missing denominator or failed observation is not a zero error rate."""
    _insert(
        database,
        "model_inventory",
        monitor_id="monitor-a",
        model_name="model-a",
        enabled=1,
        config_digest="config",
        environment="prod",
        project="project",
        config_json=json.dumps({"model_version": "1", "serving_endpoint": "endpoint"}),
    )
    for report, counts, status, version in (
        (
            "valid",
            {"request_count": 10, "error_count": 2, "prediction_row_count": 15},
            "healthy",
            "1",
        ),
        ("zero", {"request_count": 0, "error_count": 0, "prediction_row_count": 0}, "healthy", "1"),
        ("missing", {}, "healthy", "1"),
        ("failed", {"request_count": 10, "error_count": 2}, "failed", "1"),
        ("version2", {"request_count": 10, "error_count": 0}, "healthy", "2"),
    ):
        _report(
            database,
            report,
            report_json=json.dumps({"serving": counts}),
            status=status,
            model_version=version,
        )
    rows = {r["report_id"]: r for r in _execute(database, "serving_windows")}
    assert rows["valid"]["error_rate"] == 0.2
    assert rows["valid"]["rows_per_request"] == 1.5
    assert rows["version2"]["error_rate"] == 0
    assert rows["valid"]["model_identity"] != rows["version2"]["model_identity"]
    assert all(rows[key]["error_rate"] is None for key in ("zero", "missing", "failed"))
    assert rows["version2"]["rows_per_request"] is None


def test_coverage_keeps_repeated_windows_and_hides_policy_only_counts(database):
    """Coverage belongs to each full report, without sums or failure counts presented as real data."""
    _report(database, "first", current_rows=10, scored_rows=8, labeled_rows=4, label_coverage=0.5)
    _report(database, "retry", current_rows=10, scored_rows=8, labeled_rows=0, label_coverage=0)
    _report(database, "failed", status="failed", current_rows=10, scored_rows=8, label_coverage=0.5)
    _report(
        database, "policy", status="no_data", report_json='{"performance": {"status": "healthy"}}'
    )
    _report(database, "other", model_name="model-b", monitor_id="monitor-b", scored_rows=999)
    _report(database, "legacy", current_rows=0, scored_rows=0, labeled_rows=0, report_json="{}")
    observations = _execute(database, "performance_coverage", trend_model="model-a")
    rows = {r["report_id"]: r for r in observations}
    assert {(r["series"], r["row_count"]) for r in observations if r["report_id"] == "first"} == {
        ("Labeled", 4),
        ("Not labeled at measurement", 4),
    }
    assert {(r["series"], r["row_count"]) for r in observations if r["report_id"] == "retry"} == {
        ("Labeled", 0),
        ("Not labeled at measurement", 8),
    }
    assert set(rows) == {"first", "retry", "failed", "legacy"}
    assert rows["first"]["scored_rows"] == rows["retry"]["scored_rows"] == 8
    assert rows["first"]["observation_label"] != rows["retry"]["observation_label"]
    assert rows["first"]["observation_label"].startswith("2026-10-02 00:00 #")
    assert rows["retry"]["label_coverage"] == 0
    assert rows["failed"]["scored_rows"] is rows["failed"]["label_coverage"] is None
    assert rows["legacy"]["scored_rows"] is rows["legacy"]["labeled_rows"] is None
    assert rows["legacy"]["series"] == "Coverage not measured"


def test_quality_series_keeps_reference_rates_separate_from_alert_limits(database):
    """A missing-rate baseline is not an alert boundary and nonfinite has no reference."""
    identity = _report(database)
    for feature, metric, value, threshold in (
        ("age", "missing_fraction", 0, 0.1),
        ("income", "nonfinite_fraction", 0.2, None),
    ):
        _insert(
            database,
            "metric_history",
            **identity,
            column_name=feature,
            category="quality",
            metric_name=metric,
            value=value,
            threshold=threshold,
        )
    rows = _execute(database, "quality_series")
    assert len(rows) == 1
    assert rows[0]["column_name"] == "age"
    assert rows[0]["missing_fraction"] == 0
    assert rows[0]["reference_rate"] == 0.1
    assert "monitor-a" in rows[0]["model_identity"]


def test_psi_matrix_bounds_each_identity_and_keeps_invalid_limits_null(database):
    """Older windows and low-ranked features cannot grow the heatmap without bound."""
    for day in range(1, 14):
        identity = _report(database, f"r{day}", window_end=f"2026-10-{day:02d} 00:00:00")
        for feature in range(11):
            _insert(
                database,
                "metric_history",
                **identity,
                column_name=f"f{feature:02d}",
                category="drift",
                metric_name="psi_categorical" if feature == 10 else "psi",
                value=feature / 10,
                threshold=0 if feature == 0 else 0.2,
            )
    rows = _execute(database, "drift_psi_matrix")
    assert len(rows) == 120
    assert "r1" not in {r["report_id"] for r in rows}
    assert len({r["column_name"] for r in rows}) == 10
    assert next(r["psi_ratio"] for r in rows if r["column_name"] == "f10") == 5
    database.execute("UPDATE metric_history SET threshold = 0")
    assert all(row["psi_ratio"] is None for row in _execute(database, "drift_psi_matrix"))


def test_psi_chart_keeps_short_label_collisions_separate_and_marks_only_limit_exceedance(database):
    """Short model names, hash prefixes and repeated windows must never merge heatmap cells."""
    for model, monitor in (
        ("catalog.a.risk", "123456789abc-first"),
        ("catalog.b.risk", "123456789abc-first"),
        ("catalog.a.risk", "123456789abc-second"),
    ):
        for report, value, threshold in (
            ("at", 0.2, 0.2),
            ("above", 0.3, 0.2),
            ("invalid", 0.3, 0),
        ):
            identity = _report(
                database, f"{model}-{monitor}-{report}", model_name=model, monitor_id=monitor
            )
            _insert(
                database,
                "metric_history",
                **identity,
                column_name="age",
                category="drift",
                metric_name="psi",
                value=value,
                threshold=threshold,
            )
    rows = _execute(database, "drift_psi_matrix")
    assert len(rows) == 9
    assert len({row["model_label"] for row in rows}) == 3
    assert len({(row["model_label"], row["observation_label"]) for row in rows}) == 9
    assert all("risk / v1" in row["model_label"] for row in rows)
    for row in rows:
        case = row["report_id"].rsplit("-", 1)[1]
        assert row["psi_over_limit"] == {"at": 0, "above": 1, "invalid": None}[case]
    dashboard = json.loads(ASSET.read_text(encoding="utf-8"))
    widget = next(
        c["widget"]
        for p in dashboard["pages"]
        for c in p["layout"]
        if c["widget"]["name"] == "drift_psi_heatmap"
    )
    assert widget["spec"]["widgetType"] == "heatmap"
    color = widget["spec"]["encodings"]["color"]
    assert color["fieldName"] == "psi_over_limit"
    assert color["scale"]["domain"] == {"min": 0, "max": 1}
    assert color["scale"]["colorRamp"]["colors"] == {"start": "#7C8CA0", "end": "#E5484D"}
    assert {"psi", "psi_threshold", "psi_ratio", "model_identity", "report_id", "window_end"} <= {
        f["fieldName"] for f in widget["spec"]["encodings"]["extra"]
    }
    predicate = " AND ".join(f["expression"] for f in widget["queries"][0]["query"]["filters"])
    query = "".join(_dataset("drift_psi_matrix")["queryLines"])
    plotted = list(
        database.execute(
            f"SELECT * FROM ({query}) scoped WHERE {predicate}",
            {"drift_model": "", "drift_version": "", "drift_monitor": ""},
        )
    )
    assert len(plotted) == 6 and all(row["psi_over_limit"] is not None for row in plotted)


@pytest.mark.parametrize(
    ("direction", "baseline", "limit", "current", "mode", "status", "expected"),
    [
        ("higher", 0.9, 0.8, 0.85, "absolute", "healthy", 0.5),
        ("higher", 0.9, 0.8, 0.75, "absolute", "degraded", -0.5),
        ("lower", 10, 12, 11, "relative", "healthy", 0.5),
        ("lower", 10, 12, 13, "relative", "degraded", -0.5),
        ("higher", 0.9, 0.9, 0.85, "absolute", "degraded", None),
        ("higher", 0.9, 0.8, None, "absolute", "no_data", None),
        ("lower", 10, 12, 11, "relative", "disabled", None),
        ("unknown", 10, 12, 11, "absolute", "healthy", None),
    ],
)
def test_policy_margin_uses_saved_direction_and_tolerance(
    database, direction, baseline, limit, current, mode, status, expected
):
    """Different metric units become comparable only through their own nonzero tolerance band."""
    identity = _report(database)
    _insert(
        database,
        "performance_history",
        **identity,
        metric="accuracy",
        direction=direction,
        baseline_value=baseline,
        threshold_value=limit,
        current_value=current,
        tolerance=0.1,
        tolerance_mode=mode,
        status=status,
    )
    rows = _execute(database, "performance_policy_snapshot")
    assert len(rows) == 1
    assert rows[0]["limit_margin"] == (pytest.approx(expected) if expected is not None else None)


def test_policy_margin_legend_excludes_unplottable_states_without_hiding_table_rows(database):
    """Unavailable policies belong in the table, never as empty bars or duplicate-gray legends."""
    cases = [
        ("healthy", "healthy", 0.85, 0.8),
        ("loss", "degraded", 0.75, 0.8),
        ("boundary", "degraded", 0.8, 0.8),
        ("missing", "no_data", None, 0.8),
        ("disabled", "disabled", 0.85, 0.8),
        ("zero_tolerance", "healthy", 0.85, 0.9),
    ]
    for name, status, current, limit in cases:
        identity = _report(database, name, monitor_id=name, model_name=name)
        _insert(
            database,
            "performance_history",
            **identity,
            metric="accuracy",
            direction="higher",
            baseline_value=0.9,
            threshold_value=limit,
            current_value=current,
            tolerance=0.1,
            tolerance_mode="absolute",
            status=status,
        )
    snapshot = _execute(database, "performance_policy_snapshot")
    assert len(snapshot) == len(cases)
    database.execute(
        "CREATE TABLE snapshot (model_identity TEXT, limit_margin REAL, policy_status TEXT)"
    )
    for row in snapshot:
        _insert(
            database,
            "snapshot",
            **{key: row[key] for key in ("model_identity", "limit_margin", "policy_status")},
        )
    dashboard = json.loads(ASSET.read_text(encoding="utf-8"))
    widget = next(
        c["widget"]
        for page in dashboard["pages"]
        for c in page["layout"]
        if c["widget"]["name"] == "performance_limit_margin"
    )
    filters = widget["queries"][0]["query"].get("filters", [])
    predicate = " AND ".join(item["expression"] for item in filters) or "TRUE"
    plotted = list(database.execute(f"SELECT * FROM snapshot WHERE {predicate}"))
    assert len(plotted) == 3
    assert {round(row["limit_margin"], 3) for row in plotted} == {-0.5, 0.0, 0.5}
    color = widget["spec"]["encodings"]["color"]
    assert color["displayName"] == "Policy status"
    mappings = {item["value"]: item["color"] for item in color["scale"]["mappings"]}
    assert set(mappings) == {"Within tolerance", "Performance loss"}
    assert len(set(mappings.values())) == len(mappings)


def test_new_charts_share_identity_and_date_filters_and_do_not_overlap():
    """Every chart must follow the shared selection and fit without covering another widget."""
    dashboard = json.loads(ASSET.read_text(encoding="utf-8"))
    widgets = {c["widget"]["name"]: c["widget"] for p in dashboard["pages"] for c in p["layout"]}
    expected = {
        "quality_series",
        "drift_psi_matrix",
        "performance_coverage",
        "serving_windows",
        "performance_policy",
    }
    for name in ("filter_model", "filter_version", "filter_context"):
        assert expected <= {q["query"]["datasetName"] for q in widgets[name]["queries"]}
    dates = {
        q["query"]["datasetName"]
        for name in (
            "filter_measured_at_drift_quality",
            "filter_measured_at_performance",
            "filter_serving_window",
        )
        for q in widgets[name]["queries"]
    }
    assert expected <= dates
    for page in dashboard["pages"]:
        for index, item in enumerate(page["layout"]):
            a = item["position"]
            assert 0 <= a["x"] < a["x"] + a["width"] <= 12
            for other in page["layout"][index + 1 :]:
                b = other["position"]
                overlaps = (
                    a["x"] < b["x"] + b["width"]
                    and b["x"] < a["x"] + a["width"]
                    and a["y"] < b["y"] + b["height"]
                    and b["y"] < a["y"] + a["height"]
                )
                assert not overlaps, (page["name"], item["widget"]["name"], other["widget"]["name"])


def test_daily_problem_series_keep_the_same_attention_colors_as_detail_charts():
    """A drift or performance loss must never inherit the theme's healthy green slot."""
    dashboard = json.loads(ASSET.read_text(encoding="utf-8"))
    widgets = {c["widget"]["name"]: c["widget"] for p in dashboard["pages"] for c in p["layout"]}
    for name, problem, measured, unavailable in (
        ("overview_drift_trend", "Drift detected", "Checks measured", "Checks unavailable"),
        ("overview_performance_trend", "Performance loss", "Measured", "Insufficient evidence"),
    ):
        scale = widgets[name]["spec"]["encodings"]["color"]["scale"]
        assert {item["value"]: item["color"] for item in scale["mappings"]} == {
            problem: "#E5484D",
            measured: "#3B82F6",
            unavailable: "#7C8CA0",
        }
