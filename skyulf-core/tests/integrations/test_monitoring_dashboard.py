"""Contract checks for the central, portable monitoring dashboard example."""

import json
from pathlib import Path

import yaml

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "databricks_monitoring"


def _dashboard() -> dict:
    """Load the exact Lakeview payload shipped to the bundle."""
    return json.loads((EXAMPLE / "src" / "monitoring.lvdash.json").read_text(encoding="utf-8"))


def _widgets(page: dict) -> list[dict]:
    """Expose only real widgets so field references can be checked."""
    return [item["widget"] for item in page["layout"]]


def test_dashboard_inventory_and_history_have_separate_time_scopes() -> None:
    """A history date choice must never erase unobserved inventory rows."""
    dashboard = _dashboard()
    datasets = {item["name"]: "".join(item["queryLines"]) for item in dashboard["datasets"]}
    pages = {page["name"]: page for page in dashboard["pages"]}
    assert set(pages) == {"overview", "drift_quality", "performance"}
    assert "FROM current_health" in datasets["current_health"]
    assert "FROM metric_history" in datasets["metric_history"]
    assert "WHERE measured_at" not in datasets["current_health"]
    assert "never_observed" in datasets["current_health"]
    for page in pages.values():
        assert page["pageType"] == "PAGE_TYPE_CANVAS"
        assert page["layoutVersion"] == "GRID_V1"
    assert all(
        widget["spec"]["widgetType"] != "filter-date-range-picker"
        for widget in _widgets(pages["overview"])
        if "spec" in widget
    )
    for name in ("drift_quality", "performance"):
        assert any(
            widget.get("spec", {}).get("widgetType") == "filter-date-range-picker"
            for widget in _widgets(pages[name])
        )


def test_dashboard_fields_and_filters_bind_to_real_datasets() -> None:
    """Every widget field and filter query must bind to its selected dataset."""
    dashboard = _dashboard()
    names = {item["name"] for item in dashboard["datasets"]}
    filters = {
        "model_catalog",
        "model_schema",
        "model_name",
        "model_version",
    }
    seen_filters: set[str] = set()
    for page in dashboard["pages"]:
        for widget in _widgets(page):
            queries = {item["name"]: item["query"] for item in widget.get("queries", [])}
            for query in queries.values():
                assert query["datasetName"] in names
            spec = widget.get("spec", {})
            widget_type = spec.get("widgetType")
            if widget_type and widget_type.startswith("filter-"):
                assert spec["version"] == 2
                for field in spec["encodings"]["fields"]:
                    assert field["queryName"] in queries
                    if "parameterName" in field:
                        assert field["parameterName"] in {
                            item["name"] for item in queries[field["queryName"]]["parameters"]
                        }
                        continue
                    assert field["fieldName"] in {
                        item["name"] for item in queries[field["queryName"]]["fields"]
                    }
                    seen_filters.add(field["fieldName"])
            elif widget_type:
                # Lakeview visualizations implicitly read this query name.
                assert set(queries) == {"main_query"}, widget["name"]
                selected = {item["name"] for item in next(iter(queries.values()))["fields"]}
                encodings = spec["encodings"]
                if widget_type == "pivot":
                    refs = encodings["rows"] + encodings["columns"] + encodings["cell"]["fields"]
                else:
                    refs = encodings.get("columns", []) or [
                        value for value in encodings.values() if isinstance(value, dict)
                    ]
                assert all(ref["fieldName"] in selected for ref in refs)
                assert spec["version"] == (3 if widget_type in {"line", "bar", "pivot"} else 2)
    assert filters <= seen_filters
    assert not {"environment", "project"} & seen_filters
    assert all(
        field["name"] not in {"environment", "project"}
        for page in dashboard["pages"]
        for widget in _widgets(page)
        for query in widget.get("queries", [])
        for field in query["query"].get("fields", [])
    )


def test_history_keeps_metric_identity_and_units_separate() -> None:
    """Wide rows retain observation identity and charts never mix performance units."""
    dashboard = _dashboard()
    datasets = {item["name"]: item for item in dashboard["datasets"]}
    for name in ("metric_history", "performance_history"):
        sql = "".join(datasets[name]["queryLines"])
        group = sql.split("GROUP BY", 1)[1]
        for field in ("monitor_id", "report_id", "model_version", "column_name"):
            assert field in group
        assert "AVG(" not in sql.upper()
        assert "ELSE 0" not in sql.upper()
    widgets = {w["name"]: w for p in dashboard["pages"] for w in _widgets(p)}
    for name, metrics in (
        ("drift_metrics", {"psi_display", "ks_statistic_display", "drift_status"}),
        ("quality_metrics", {"missing_fraction", "nonfinite_fraction", "observed_at"}),
    ):
        columns = {c["fieldName"] for c in widgets[name]["spec"]["encodings"]["columns"]}
        assert metrics <= columns
        assert not {"metric_name", "value"} & columns
    pivot = widgets["performance_metrics"]
    assert pivot["spec"]["widgetType"] == "pivot"
    assert pivot["spec"]["encodings"]["columns"][0]["fieldName"] == "metric_name"
    assert "observation" in {r["fieldName"] for r in pivot["spec"]["encodings"]["rows"]}
    cells_sql = "".join(datasets["performance_cells"]["queryLines"])
    assert "ORDER BY monitor_id, report_id" in cells_sql
    assert "same_time > 1" in cells_sql
    chart_sql = "".join(datasets["performance_chart"]["queryLines"])
    assert "metric_name = :performance_metric" in chart_sql
    assert "model_name = :trend_model" in chart_sql
    assert "model_version = :trend_version" in chart_sql
    assert "ORDER BY measured_at" in chart_sql
    drift_page = next(p for p in dashboard["pages"] if p["name"] == "drift_quality")
    performance_page = next(p for p in dashboard["pages"] if p["name"] == "performance")
    assert {
        w["name"]
        for w in _widgets(drift_page)
        if w.get("spec", {}).get("widgetType") in {"bar", "line"}
    } == {"drift_psi", "drift_trend"}
    assert (
        sum(w.get("spec", {}).get("widgetType") == "line" for w in _widgets(performance_page)) == 1
    )
    assert "drift_chart" not in datasets and "quality_chart" not in datasets
    assert "outcome_summary" in "".join(datasets["performance_history"]["queryLines"])


def test_charts_keep_model_selection_and_latest_observation_consistent() -> None:
    """Charts must not blend model versions or replace missing latest values with old ones."""
    dashboard = _dashboard()
    datasets = {d["name"]: "".join(d["queryLines"]) for d in dashboard["datasets"]}
    widgets = {w["name"]: w for p in dashboard["pages"] for w in _widgets(p)}
    for dataset in ("drift_psi_chart", "drift_trend"):
        assert "model_name = :drift_model" in datasets[dataset]
        assert "model_version = :drift_version" in datasets[dataset]
        assert "count(DISTINCT monitor_id)" in datasets[dataset]
        for field in ("model_name", "model_version"):
            control = widgets[f"filter_{field}_drift_quality"]
            assert control["spec"]["widgetType"] == "filter-single-select"
            assert any(q["query"]["datasetName"] == dataset for q in control["queries"])
    psi = datasets["drift_psi_chart"]
    assert "metric_name = 'psi'" in psi
    assert "threshold AS plot_value" in psi
    assert "report_rank = 1" in psi and "LIMIT 10" in psi
    assert "metric_name <> 'ks_test_p_value'" in datasets["drift_trend"]
    assert "current_timezone()" in datasets["drift_trend"]
    assert widgets["drift_trend"]["spec"]["encodings"]["x"]["fieldName"] == "measurement_time"
    latest = datasets["performance_latest"]
    assert "ORDER BY measured_at DESC, report_id DESC" in latest
    assert "value IS NOT NULL" not in latest
    for field in ("model_name", "model_version", "metric_name"):
        control = widgets[f"filter_{field}_performance"]
        assert any(q["query"]["datasetName"] == "performance_latest" for q in control["queries"])
    assert widgets["performance_latest"]["spec"]["widgetType"] == "bar"


def test_bundle_has_one_serial_manual_writer_and_stable_dashboard() -> None:
    """A retry or redeploy must not fork the central writer or dashboard identity."""
    bundle = yaml.safe_load((EXAMPLE / "databricks.yml").read_text())
    job = yaml.safe_load((EXAMPLE / "resources" / "monitoring.job.yml").read_text())
    dashboard = yaml.safe_load((EXAMPLE / "resources" / "monitoring.dashboard.yml").read_text())
    assert {
        "monitoring_catalog",
        "monitoring_schema",
        "warehouse_id",
        "wheel_path",
        "experiment_name",
    } <= set(bundle["variables"])
    assert bundle["targets"]["dev"]["mode"] == "development"
    resource = job["resources"]["jobs"]["monitoring"]
    assert resource["max_concurrent_runs"] == 1
    assert resource["queue"]["enabled"] is True
    assert "schedule" not in resource and "trigger" not in resource
    assert len(resource["tasks"]) == 1
    task = resource["tasks"][0]
    assert task["environment_key"] == resource["environments"][0]["environment_key"]
    assert resource["environments"][0]["spec"]["client"] == "4"
    assert "${var.wheel_path}" in resource["environments"][0]["spec"]["dependencies"]
    assert set(task["notebook_task"]["base_parameters"]) == {
        "monitoring_catalog",
        "monitoring_schema",
        "experiment_name",
        "as_of",
        "window_start",
        "window_end",
    }
    dashboards = dashboard["resources"]["dashboards"]
    assert list(dashboards) == ["monitoring_dashboard"]
    assert not set(dashboards).intersection(job["resources"]["jobs"])
    assert dashboards["monitoring_dashboard"]["dataset_catalog"] == "${var.monitoring_catalog}"
    assert dashboards["monitoring_dashboard"]["dataset_schema"] == "${var.monitoring_schema}"
    assert dashboards["monitoring_dashboard"]["warehouse_id"] == "${var.warehouse_id}"
