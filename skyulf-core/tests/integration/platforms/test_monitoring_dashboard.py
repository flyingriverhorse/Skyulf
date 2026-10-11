"""Contract checks for the shared monitoring dashboard shipped in the project template."""

import json
import re
import runpy
import sqlite3
from pathlib import Path

import yaml

TEMPLATE = Path(__file__).resolve().parents[3] / "templates/databricks/template/{{.project_name}}"


def test_identity_controls_are_shared_across_model_pages() -> None:
    """Moving between pages must retain model, version and context without competing controls."""
    dashboard = _dashboard()
    global_page = next(p for p in dashboard["pages"] if p["name"] == "investigation_filters")
    controls = _widgets(global_page)
    assert {w["name"] for w in controls} == {"filter_model", "filter_version", "filter_context"}
    for control in controls:
        assert control["spec"]["widgetType"] == "filter-single-select"
        bound = {q["query"]["datasetName"] for q in control["queries"]}
        assert {
            "current_health",
            "overview_inventory",
            "overview_performance",
            "overview_drift_daily",
            "overview_performance_daily",
            "monitoring_issues",
            "training_lineage",
            "drift_context",
            "performance_snapshot",
            "serving_current",
            "serving_windows",
            "serving_status",
        } <= bound
        assert not any(name.startswith("execution_") for name in bound)
    for page in dashboard["pages"]:
        if page is global_page:
            continue
        for widget in _widgets(page):
            if widget.get("spec", {}).get("widgetType", "").startswith("filter-"):
                assert not any(
                    field.get("fieldName") in {"model_name", "model_version", "monitor_context"}
                    for field in widget["spec"]["encodings"]["fields"]
                )


def test_model_details_preserve_explicit_selection_across_portfolio_pages() -> None:
    """Detail pages must retain shared model filters while naming each matching identity."""
    dashboard = _dashboard()
    datasets = {d["name"]: "".join(d["queryLines"]) for d in dashboard["datasets"]}
    for name, prefix in (("drift_context", "drift"), ("performance_snapshot", "trend")):
        assert f"model_name = :{prefix}_model" in datasets[name]
        assert f"COALESCE(:{prefix}_model, '') = '' OR" not in datasets[name]
    pages = {p["name"]: _widgets(p) for p in dashboard["pages"]}
    for name in ("drift_quality", "performance"):
        assert any(
            q["query"]["datasetName"] == "drift_context"
            for w in pages[name]
            if not w.get("spec", {}).get("widgetType", "").startswith("filter-")
            for q in w.get("queries", [])
        )
    assert not any(w["name"] == "model_inventory" for w in pages["drift_quality"])


def _dashboard() -> dict:
    """Load the exact Lakeview payload shipped to the bundle."""
    return json.loads(
        (TEMPLATE / "src/monitoring/monitoring.lvdash.json").read_text(encoding="utf-8")
    )


def _widgets(page: dict) -> list[dict]:
    """Expose only real widgets so field references can be checked."""
    return [item["widget"] for item in page["layout"]]


def test_shared_defaults_never_filter_visual_rows_by_an_all_sentinel() -> None:
    """Inherited scalar defaults must not become equality filters on portfolio rows."""
    dashboard = _dashboard()
    datasets = {item["name"]: item for item in dashboard["datasets"]}
    visual_datasets = {
        q["query"]["datasetName"]
        for page in dashboard["pages"]
        for widget in _widgets(page)
        if "spec" in widget and not widget["spec"]["widgetType"].startswith("filter-")
        for q in widget.get("queries", [])
    }
    page = next(p for p in dashboard["pages"] if p["name"] == "investigation_filters")
    for widget in _widgets(page):
        defaults = set()
        for item in widget["queries"]:
            query = item["query"]
            if query.get("fields"):
                assert query["datasetName"] not in visual_datasets
            for binding in query.get("parameters", []):
                parameter = next(
                    p
                    for p in datasets[query["datasetName"]]["parameters"]
                    if p["keyword"] == binding["keyword"]
                )
                defaults.add(parameter["defaultSelection"]["values"]["values"][0]["value"])
        assert len(defaults) == 1 and "" not in defaults


def test_execution_ids_are_enterable_without_project_named_job_options() -> None:
    """Valid one-off jobs remain inspectable when no Bundle-named jobs exist."""
    dashboard = _dashboard()
    page = next(p for p in dashboard["pages"] if p["name"] == "execution")
    controls = [w for w in _widgets(page) if w["name"].startswith("filter_execution_")]
    for control in controls:
        assert all("fields" not in query["query"] for query in control["queries"])
    choices = next(d for d in dashboard["datasets"] if d["name"] == "execution_choices")
    sql = "".join(choices["queryLines"])
    sql = (
        sql.replace("system.lakeflow.jobs", "jobs")
        .replace("system.lakeflow.job_run_timeline", "runs")
        .replace("current_timestamp() - INTERVAL 30 DAYS", "'2026-09-11'")
        .replace(" AS STRING", " AS TEXT")
    )
    with sqlite3.connect(":memory:") as connection:
        connection.executescript(
            "CREATE TABLE jobs(workspace_id TEXT, job_id TEXT, name TEXT, change_time TEXT);"
            "CREATE TABLE runs(workspace_id TEXT, job_id TEXT, period_start_time TEXT, "
            "period_end_time TEXT);"
            "INSERT INTO runs VALUES ('workspace-a','one-off','2026-10-11','2026-10-11');"
        )
        rows = connection.execute(sql).fetchall()
    assert len(rows) == 1 and rows[0][:2] == ("workspace-a", "one-off")


def test_dashboard_inventory_and_history_have_separate_time_scopes() -> None:
    """A history date choice must never erase unobserved inventory rows."""
    dashboard = _dashboard()
    datasets = {item["name"]: "".join(item["queryLines"]) for item in dashboard["datasets"]}
    pages = {page["name"]: page for page in dashboard["pages"]}
    assert set(pages) == {
        "overview",
        "drift_quality",
        "performance",
        "execution",
        "serving",
        "training_lineage",
        "investigation_filters",
    }
    assert "FROM current_health" in datasets["current_health"]
    assert "FROM metric_history" in datasets["metric_history"]
    assert "WHERE measured_at" not in datasets["current_health"]
    assert "never_observed" in datasets["current_health"]
    for page in pages.values():
        assert page["pageType"] == (
            "PAGE_TYPE_GLOBAL_FILTERS"
            if page["name"] == "investigation_filters"
            else "PAGE_TYPE_CANVAS"
        )
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
        "model_name",
        "model_version",
        "monitor_context",
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
                if widget_type == "custom-vega-viz":
                    refs = encodings["fields"]
                    assert spec["data"]["queryName"] == "main_query"
                elif widget_type == "pivot":
                    refs = encodings["rows"] + encodings["columns"] + encodings["cell"]["fields"]
                else:
                    refs = encodings.get("columns", []) or [
                        value
                        for value in encodings.values()
                        if isinstance(value, dict) and "fieldName" in value
                    ]
                    refs += encodings.get("extra", [])
                if "facet" in spec:
                    refs = [*refs, spec["facet"]]
                assert all(ref["fieldName"] in selected for ref in refs)
                assert spec["version"] == (
                    1
                    if widget_type == "custom-vega-viz"
                    else 3
                    if widget_type in {"line", "bar", "scatter", "pivot"}
                    else 2
                )
    assert filters <= seen_filters
    assert not {"environment", "project"} & seen_filters
    assert all(
        field["name"] not in {"environment", "project"}
        for page in dashboard["pages"]
        for widget in _widgets(page)
        if widget["name"] != "performance_policy_evidence"
        for query in widget.get("queries", [])
        for field in query["query"].get("fields", [])
    )


def test_history_keeps_metric_identity_and_units_separate() -> None:
    """Wide rows retain observation identity and charts never mix performance units."""
    dashboard = _dashboard()
    datasets = {item["name"]: item for item in dashboard["datasets"]}
    for name in ("metric_history", "performance_measurements"):
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
        if w.get("spec", {}).get("widgetType") in {"bar", "line", "custom-vega-viz"}
    } == {"drift_psi", "drift_trend"}
    performance_charts = {
        w["name"]: w["spec"]
        for w in _widgets(performance_page)
        if w.get("spec", {}).get("widgetType") == "scatter"
    }
    assert set(performance_charts) == {
        "performance_trend",
        "performance_policy_trend",
    }
    for chart in performance_charts.values():
        assert chart["facet"]["fieldName"] == "model_panel"
        assert chart["encodings"]["y"]["scale"]["domain"] == {"type": "per-facet"}
        # Points cannot interpolate across unavailable scores or invent zero values.
        assert chart["encodings"]["y"]["fieldName"] == "value"
    assert "drift_chart" not in datasets and "quality_chart" not in datasets
    assert "outcome_summary" in "".join(datasets["performance_measurements"]["queryLines"])


def test_charts_keep_model_selection_and_latest_observation_consistent() -> None:
    """Charts must not blend model versions or replace missing latest values with old ones."""
    dashboard = _dashboard()
    datasets = {d["name"]: "".join(d["queryLines"]) for d in dashboard["datasets"]}
    widgets = {w["name"]: w for p in dashboard["pages"] for w in _widgets(p)}
    for dataset in ("drift_psi_chart", "drift_trend"):
        assert "model_name = :drift_model" in datasets[dataset]
        assert "model_version = :drift_version" in datasets[dataset]
        assert "chosen_context" in datasets[dataset]
        assert "monitor_id" in datasets[dataset]
        for field in ("model_name", "model_version"):
            control = widgets[
                {"model_name": "filter_model", "model_version": "filter_version"}[field]
            ]
            assert control["spec"]["widgetType"] == "filter-single-select"
            assert any(q["query"]["datasetName"] == dataset for q in control["queries"])
    psi = datasets["drift_psi_chart"]
    assert "metric_name = 'psi'" in psi
    assert "threshold AS plot_value" in psi
    assert "report_rank = 1" in psi and "feature_rank <= 10" in psi
    assert "PARTITION BY monitor_id, model_name, model_version" in psi
    assert "FROM monitoring_results" in psi
    assert "window_end DESC" in psi
    assert "metric_name <> 'ks_test_p_value'" in datasets["drift_trend"]
    chart = widgets["drift_trend"]["spec"]
    encoding = chart["encodings"]
    assert chart["facet"]["fieldName"] == "model_identity"
    assert encoding["y"]["scale"]["domain"] == {"type": "per-facet"}
    assert encoding["x"]["fieldName"] == "window_end"
    assert encoding["x"]["axis"]["labelAngle"] == 0
    assert chart["mark"]["layout"] == "group"
    assert encoding["y"]["axis"]["hideLabels"] is True
    assert encoding["label"]["show"] is True
    assert encoding["y"]["format"]["decimalPlaces"]["places"] == 0
    latest = datasets["performance_latest"]
    assert "ORDER BY window_end DESC, measured_at DESC, report_id DESC" in latest
    assert "value IS NOT NULL" not in latest
    assert "FROM monitoring_results" in latest
    assert "history.report_id = latest_reports.report_id" in latest
    for field in ("model_name", "model_version", "metric_name"):
        control = widgets[
            {
                "model_name": "filter_model",
                "model_version": "filter_version",
                "metric_name": "filter_metric_name_performance",
            }[field]
        ]
        assert any(q["query"]["datasetName"] == "performance_latest" for q in control["queries"])
    assert widgets["performance_latest"]["spec"]["widgetType"] == "table"
    assert {
        column["fieldName"]
        for column in widgets["performance_latest"]["spec"]["encodings"]["columns"]
    } >= {"model_identity", "metric_name", "value", "window_end"}


def test_history_charts_and_date_filters_use_the_data_window() -> None:
    """Delayed calculations must plot and filter the data period, not the execution time."""
    dashboard = _dashboard()
    widgets = {w["name"]: w for p in dashboard["pages"] for w in _widgets(p)}
    for name in ("performance_trend", "performance_policy_trend"):
        chart = widgets[name]["spec"]
        assert chart["encodings"]["x"]["fieldName"] == "window_end"
    for name in ("filter_measured_at_drift_quality", "filter_measured_at_performance"):
        for query in widgets[name]["queries"]:
            assert query["query"]["fields"] == [
                {"name": "window_end", "expression": "`window_end`"}
            ]
    page = next(p for p in dashboard["pages"] if p["name"] == "serving")
    positions = {item["widget"]["name"]: item["position"] for item in page["layout"]}
    for name in ("serving_volume_trend", "serving_error_trend", "serving_latency_trend"):
        chart = positions[name]
        assert chart["y"] + chart["height"] <= positions["serving_identity"]["y"]


def test_bundle_has_one_serial_manual_writer_and_stable_dashboard(tmp_path) -> None:
    """One project owns the shared dashboard while existing monitoring jobs stay serialized."""
    job = yaml.safe_load((TEMPLATE / "resources/monitoring.job.yml.tmpl").read_text())
    resource = job["resources"]["jobs"]["monitoring"]
    assert resource["max_concurrent_runs"] == 1
    assert resource["queue"]["enabled"] is True
    assert resource["schedule"]["pause_status"] == "${var.monitoring_pause_status}"
    assert {task["task_key"] for task in resource["tasks"]} == {
        "monitor_model",
        "monitoring_report",
        "evaluate_retraining",
    }
    (tmp_path / "resources").mkdir()
    asset = tmp_path / "src/monitoring/monitoring.lvdash.json"
    asset.parent.mkdir(parents=True)
    asset.write_bytes((TEMPLATE / "src/monitoring/monitoring.lvdash.json").read_bytes())
    configure = runpy.run_path(str(TEMPLATE / "src/tools/configure_monitoring_dashboard.py"))[
        "configure"
    ]
    configure(tmp_path, create_dashboard=True, warehouse_id="d047a4d9aa276958")
    dashboard = yaml.safe_load((tmp_path / "resources/monitoring.dashboard.yml").read_text())
    dashboards = dashboard["targets"]["test"]["resources"]["dashboards"]
    assert list(dashboards) == ["monitoring_dashboard"]
    assert dashboards["monitoring_dashboard"]["dataset_catalog"] == "${var.monitoring_catalog}"
    assert dashboards["monitoring_dashboard"]["dataset_schema"] == "${var.monitoring_schema}"
    assert dashboards["monitoring_dashboard"]["embed_credentials"] is False


def test_performance_policy_evidence_and_trend_bind_to_saved_view() -> None:
    """Policy status and threshold series must come from the saved policy evidence."""
    dashboard = _dashboard()
    datasets = {item["name"]: "".join(item["queryLines"]) for item in dashboard["datasets"]}
    widgets = {widget["name"]: widget for page in dashboard["pages"] for widget in _widgets(page)}
    assert "FROM performance_history" in datasets["performance_policy"]
    assert "FROM performance_history" in datasets["performance_policy_series"]
    assert "UNION ALL" in datasets["performance_policy_series"]
    assert all(
        name in datasets["performance_policy_series"]
        for name in (
            "current_value",
            "baseline_value",
            "threshold_value",
            "model_name = :trend_model",
            "model_version = :trend_version",
        )
    )
    assert "FROM metric_history" in datasets["performance_measurements"]
    table = widgets["performance_policy_evidence"]
    assert table["spec"]["widgetType"] == "table"
    assert table["queries"][0]["query"]["datasetName"] == "performance_policy"
    columns = {column["fieldName"] for column in table["spec"]["encodings"]["columns"]}
    evidence_fields = {
        "status",
        "reason",
        "report_id",
        "model_name",
        "model_version",
        "metric",
        "baseline_kind",
        "baseline_reference",
        "baseline_value",
        "current_value",
        "absolute_degradation",
        "relative_degradation",
        "tolerance",
        "tolerance_mode",
        "threshold_value",
        "label_coverage",
        "labeled_rows",
        "window_start",
        "window_end",
        "consecutive_failures",
        "required_windows",
        "action",
        "action_reason",
        "request_id",
        "run_id",
    }
    assert {
        "result_label",
        "metric",
        "baseline_value",
        "current_value",
        "label_coverage",
    } <= columns
    assert {"model_identity", "window_end", "threshold_value", "action_label"} <= columns
    assert len(columns) == 9
    assert all(field in datasets["performance_policy"] for field in evidence_fields)
    chart = widgets["performance_policy_trend"]
    assert chart["spec"]["widgetType"] == "scatter"
    assert chart["spec"]["encodings"]["color"]["fieldName"] == "series"
    assert chart["queries"][0]["query"]["datasetName"] == "performance_policy_series"
    for field in ("model_name", "model_version"):
        control = widgets[
            {
                "model_name": "filter_model",
                "model_version": "filter_version",
                "metric_name": "filter_metric_name_performance",
            }[field]
        ]
        assert any(
            query["query"]["datasetName"] == "performance_policy" for query in control["queries"]
        )
    assert any(
        query["query"]["datasetName"] == "performance_policy_series"
        for query in widgets["filter_measured_at_performance"]["queries"]
    )


def test_context_selection_keeps_every_matching_observed_or_unobserved_identity() -> None:
    """All matching enrollments must survive selection, including shared model versions."""
    dashboard = _dashboard()
    datasets = {item["name"]: "".join(item["queryLines"]) for item in dashboard["datasets"]}
    assert "FROM current_health" in datasets["context_choices"]
    assert "FROM monitoring_results" in datasets["context_choices"]
    for name in ("metric_history", "quality_history", "performance_measurements"):
        assert "chosen_context" in datasets[name]
        assert "LIMIT 1" not in datasets[name]
        assert "source.monitor_id = context.monitor_id" in datasets[name]
    for name in ("drift_context",):
        assert "monitor_context" in datasets[name]
        assert "No matching enrollment" in datasets[name]
    assert {
        parameter["keyword"]
        for dataset in dashboard["datasets"]
        for parameter in dataset.get("parameters", [])
    } == {
        "scope_model",
        "scope_version",
        "scope_monitor",
        "serving_model",
        "serving_version",
        "serving_monitor",
        "drift_model",
        "drift_version",
        "drift_monitor",
        "trend_model",
        "trend_version",
        "trend_monitor",
        "performance_metric",
        "execution_workspace",
        "execution_job",
    }


def test_confusion_matrix_preserves_labels_and_missing_evidence() -> None:
    """Absent historical matrix data must not turn into perfect or zero classification results."""
    dashboard = _dashboard()
    datasets = {item["name"]: "".join(item["queryLines"]) for item in dashboard["datasets"]}
    assert "$.confusion_matrix" in datasets["confusion_cells"]
    assert "actual_index" in datasets["confusion_cells"]
    assert "predicted_index" in datasets["confusion_cells"]
    assert "not_recorded" in datasets["confusion_status"]
    status = datasets["confusion_status"]
    assert status.index("status = 'failed'") < status.index("'not_recorded'")
    assert status.index("status = 'no_data'") < status.index("'not_recorded'")
    assert "error_message" in status
    assert (
        "ORDER BY window_end DESC, measured_at DESC, report_id DESC" in datasets["confusion_cells"]
    )
    widgets = {w["name"]: w for p in dashboard["pages"] for w in _widgets(p)}
    assert widgets["confusion_matrix"]["spec"]["widgetType"] == "pivot"


def test_latest_observations_do_not_rank_policy_only_windows_as_feature_reports() -> None:
    """A newer mature-policy window must not hide a scored report or its confusion matrix."""
    dashboard = _dashboard()
    datasets = {item["name"]: "".join(item["queryLines"]) for item in dashboard["datasets"]}
    for name in (
        "context_choices",
        "drift_context",
        "drift_psi_chart",
        "performance_latest",
        "confusion_status",
        "confusion_cells",
    ):
        sql = datasets[name]
        assert "status = 'no_data'" in sql
        assert "get_json_object(report_json, '$.performance') IS NOT NULL" in sql
        assert "get_json_object(report_json, '$.current_rows') IS NULL" in sql
    assert "FROM performance_history" in datasets["performance_policy"]
    for name in ("drift_psi_chart", "performance_latest"):
        rank_source = datasets[name].split("AS report_rank", 1)[1]
        assert rank_source.startswith("\n FROM monitoring_results WHERE NOT (status = 'no_data'")


def test_execution_cost_requires_exact_scope_and_preserves_unpriced_usage() -> None:
    """Empty portable settings must never expose arbitrary workspace spend or fake exact cost."""
    dashboard = _dashboard()
    datasets = {item["name"]: "".join(item["queryLines"]) for item in dashboard["datasets"]}
    for name in ("execution_runs", "execution_cost"):
        assert ":execution_workspace" in datasets[name]
        assert ":execution_job" in datasets[name]
    cost = datasets["execution_cost"]
    assert "system.billing.usage" in cost
    assert "system.billing.list_prices" in cost
    assert "unpriced_records" in cost
    assert "currency_code = 'USD'" in cost
    assert "COALESCE(lp.pricing" not in cost
    assert "Select a workspace and job" in datasets["execution_status"]


def test_nullable_context_parameters_keep_all_and_explicit_selections() -> None:
    """All or cleared filters retain every identity while explicit mismatches stay empty."""
    datasets = _dashboard()["datasets"]
    expected = {
        dataset["name"]
        for dataset in datasets
        if "chosen_context AS" in "".join(dataset["queryLines"])
    }
    checked: set[str] = set()
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE available_contexts (monitor_id TEXT, model_name TEXT, "
            "model_version TEXT, monitor_context TEXT, latest_measurement INTEGER)"
        )
        connection.executemany(
            "INSERT INTO available_contexts VALUES (?, ?, ?, ?, ?)",
            [
                ("old", "model-a", "1", "project-a", 1),
                ("new", "model-b", "2", "project-b", 2),
                ("unobserved", "model-a", "1", "project-c", None),
            ],
        )
        for dataset in datasets:
            sql = "".join(dataset["queryLines"])
            match = re.search(r"chosen_context AS \(\n(.*?)\n\)", sql, re.S)
            if match is None:
                continue
            checked.add(dataset["name"])
            query = match.group(1)
            keywords = re.findall(r":([a-z_]+)", query)
            defaults = {
                p["keyword"]: p["defaultSelection"]["values"]["values"][0]["value"]
                for p in dataset["parameters"]
            }
            for parameters in (
                dict.fromkeys(keywords, None),
                dict.fromkeys(keywords, ""),
                defaults,
            ):
                rows = connection.execute(query, parameters).fetchall()
                assert {row[0] for row in rows} == {"old", "new", "unobserved"}, dataset["name"]
                parameters.update({key: "model-a" for key in keywords if key.endswith("_model")})
                rows = connection.execute(query, parameters).fetchall()
                assert {row[0] for row in rows} == {"old", "unobserved"}, dataset["name"]
                parameters.update({key: "2" for key in keywords if key.endswith("_version")})
                assert connection.execute(query, parameters).fetchone() is None, dataset["name"]
                parameters.update({key: "1" for key in keywords if key.endswith("_version")})
                parameters.update(
                    {key: "project-a" for key in keywords if key.endswith("_monitor")}
                )
                rows = connection.execute(query, parameters).fetchall()
                assert [row[0] for row in rows] == ["old"], dataset["name"]
                parameters.update(
                    {key: "project-b" for key in keywords if key.endswith("_monitor")}
                )
                assert connection.execute(query, parameters).fetchone() is None, dataset["name"]
    assert checked == expected and checked


def test_parameter_option_datasets_do_not_prune_sibling_selectors() -> None:
    """An empty scalar default must not remove the choices in another selector."""
    for page in _dashboard()["pages"]:
        used_options: set[str] = set()
        for widget in _widgets(page):
            fields = widget.get("spec", {}).get("encodings", {}).get("fields", [])
            if not any("parameterName" in field for field in fields):
                continue
            queries = {query["name"]: query["query"] for query in widget["queries"]}
            options = {
                queries[field["queryName"]]["datasetName"]
                for field in fields
                if "fieldName" in field and field["queryName"] == "options"
            }
            assert not options & used_options, widget["name"]
            used_options.update(options)


def test_empty_metric_selection_keeps_policy_evidence_visible() -> None:
    """A blank or unrelated chart metric must not hide the configured policy evidence."""
    dashboard = _dashboard()
    policy = next(d for d in dashboard["datasets"] if d["name"] == "performance_policy")
    assert "performance_metric" not in {p["keyword"] for p in policy["parameters"]}
    widget = next(
        w
        for page in dashboard["pages"]
        for w in _widgets(page)
        if w["name"] == "filter_metric_name_performance"
    )
    assert all(q["query"]["datasetName"] != "performance_policy" for q in widget["queries"])


def test_performance_snapshot_uses_latest_full_observation_without_inventing_counts() -> None:
    """A fresh failed measurement must clear headline counts instead of reviving old success."""
    dataset = next(d for d in _dashboard()["datasets"] if d["name"] == "performance_snapshot")
    tail = "".join(dataset["queryLines"]).split("), ranked_reports AS (", 1)[1]
    sql = (
        "WITH chosen_context AS (SELECT 'm' AS monitor_id, 'model' AS model_name, "
        "'1' AS model_version, 'prod-main' AS monitor_context "
        "UNION ALL SELECT 'other', 'model-b', '2', 'prod-other' "
        "UNION ALL SELECT 'unmeasured', 'model-c', '1', 'prod-new'), ranked_reports AS (" + tail
    )
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row

        def json_value(raw: str, path: str) -> str | None:
            """Match Spark's SQL NULL for both absent fields and explicit JSON null."""
            value = json.loads(raw).get(path[2:])
            return json.dumps(value) if value is not None else None

        connection.create_function("get_json_object", 2, json_value)
        connection.execute(
            "CREATE TABLE monitoring_results (monitor_id TEXT, model_name TEXT, model_version TEXT, "
            "report_id TEXT, window_end INTEGER, measured_at INTEGER, status TEXT, report_json TEXT, "
            "scored_rows INTEGER, labeled_rows INTEGER, label_coverage REAL)"
        )
        rows = connection.execute(sql).fetchall()
        assert len(rows) == 3 and all(row["scored_rows"] is None for row in rows)
        connection.executemany(
            "INSERT INTO monitoring_results VALUES ('m','model','1',?,?,?,?,?,?,?,?)",
            [
                ("full", 1, 1, "healthy", '{"current_rows":40}', 40, 20, 0.5),
                ("policy", 2, 2, "no_data", '{"performance":{}}', 0, 0, 0.0),
                (
                    "policy-null",
                    2,
                    3,
                    "no_data",
                    '{"performance":{},"current_rows":null}',
                    0,
                    0,
                    0.0,
                ),
            ],
        )
        connection.executemany(
            "INSERT INTO monitoring_results VALUES ('other','model-b','2',?,?,?,?,?,?,?,?)",
            [
                ("other-full", 1, 1, "healthy", '{"current_rows":60}', 60, 30, 0.5),
                ("other-backfill", 1, 100, "healthy", '{"current_rows":99}', 99, 99, 1.0),
                ("other-missing", 2, 2, "no_data", '{"current_rows":null}', None, None, None),
            ],
        )
        rows = {row["monitor_id"]: row for row in connection.execute(sql).fetchall()}
        row = rows["m"]
        assert (row["scored_rows"], row["labeled_rows"], row["label_coverage"]) == (40, 20, 0.5)
        assert rows["other"]["report_id"] == "other-missing"
        assert rows["unmeasured"]["report_id"] is None
        connection.execute(
            "INSERT INTO monitoring_results VALUES ('m','model','1','failed',3,3,'failed','{}',0,0,0)"
        )
        rows = {row["monitor_id"]: row for row in connection.execute(sql).fetchall()}
        assert set(rows) == {"m", "other", "unmeasured"}
        assert rows["m"]["report_id"] == "failed"
        assert rows["other"]["report_id"] == "other-missing"
        assert all(
            row[name] is None
            for row in rows.values()
            for name in ("scored_rows", "labeled_rows", "label_coverage")
        )


def test_policy_summary_keeps_latest_unavailable_window_and_its_own_metric() -> None:
    """Historical loss must not replace a missing latest policy check or depend on chart metric."""
    datasets = {d["name"]: d for d in _dashboard()["datasets"]}
    sql = "".join(datasets["performance_policy_snapshot"]["queryLines"])
    assert "ORDER BY window_end DESC, measured_at DESC, report_id DESC" in sql
    assert "WHERE status = 'degraded'" not in sql and "current_value IS NOT NULL" not in sql
    tail = sql.split("), policy_ranked AS (", 1)[1]
    query = (
        "WITH chosen_context AS (SELECT 'm' AS monitor_id, 'model' AS model_name, "
        "'1' AS model_version, 'prod-main' AS monitor_context), policy_ranked AS (" + tail
    )
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        connection.execute(
            "CREATE TABLE performance_history (monitor_id TEXT, model_name TEXT, model_version TEXT, "
            "report_id TEXT, window_end INTEGER, measured_at INTEGER, metric TEXT, status TEXT, "
            "reason TEXT, action TEXT, baseline_value REAL, current_value REAL, threshold_value REAL)"
        )
        connection.executemany(
            "INSERT INTO performance_history VALUES ('m','model','1',?,?,?,'rmse',?,?,'none',1,?,1.1)",
            [
                ("backfill", 1, 100, "degraded", "historical_window", 1.5),
                ("latest", 2, 2, "unavailable", "current.value must be a finite number.", None),
            ],
        )
        connection.execute("ALTER TABLE performance_history ADD COLUMN direction TEXT")
        row = connection.execute(query).fetchone()
        assert row["report_id"] == "latest" and row["current_value"] is None
        assert row["policy_status"] == "Insufficient evidence"
        connection.execute(
            "INSERT INTO performance_history VALUES "
            "('m','model','1','disabled',3,3,NULL,'disabled','policy_off','none',NULL,NULL,NULL,NULL)"
        )
        assert connection.execute(query).fetchone()["policy_status"] == "Policy disabled"
    for name in ("performance_policy_snapshot", "performance_policy_series"):
        assert "performance_metric" not in {p["keyword"] for p in datasets[name]["parameters"]}
    series = "".join(datasets["performance_policy_series"]["queryLines"])
    assert "metric AS metric_name" in series and "FROM performance_history" in series
    assert "PARTITION BY source.monitor_id, source.model_name, source.model_version" in series
    widgets = {w["name"]: w for p in _dashboard()["pages"] for w in _widgets(p)}
    assert widgets["performance_context"]["queries"][0]["query"]["datasetName"] == "drift_context"
    metric = widgets["filter_metric_name_performance"]
    assert all(q["query"]["datasetName"] != "performance_policy_series" for q in metric["queries"])


def test_performance_checks_explain_results_without_claiming_training_started() -> None:
    """A loss or eligible request must not be mislabeled as an already submitted training run."""
    dashboard = _dashboard()
    dataset = next(d for d in dashboard["datasets"] if d["name"] == "performance_policy")
    sql = "".join(dataset["queryLines"])
    labels = sql.split("source.*,", 1)[1].split("FROM (SELECT", 1)[0]
    with sqlite3.connect(":memory:") as connection:
        cases = [
            ("healthy", "none", "Within tolerance", "No training requested"),
            ("degraded", "report", "Performance loss", "Report only"),
            ("degraded", "request_eligible", "Performance loss", "Eligible; checks pending"),
            ("degraded", "retraining_requested", "Performance loss", "Training requested"),
            ("degraded", "retraining_skipped", "Performance loss", "Training skipped"),
            ("unavailable", None, "Insufficient evidence", "Not evaluated"),
            ("disabled", "none", "Policy disabled", "No training requested"),
        ]
        for status, action, result, training in cases:
            row = connection.execute(
                "SELECT " + labels + " FROM (SELECT ? AS status, ? AS action) source",
                (status, action),
            ).fetchone()
            assert row == (result, training)
    page = next(p for p in dashboard["pages"] if p["name"] == "performance")
    widgets = {w["name"]: w for w in _widgets(page)}
    assert widgets["performance_context"]["queries"][0]["query"]["datasetName"] == "drift_context"
    history = widgets["performance_policy_evidence"]["spec"]
    assert history["frame"]["title"] == "Performance checks"
    assert "monitoring_report" not in history["frame"]["description"]
    assert any(c["displayName"] == "Performance result" for c in history["encodings"]["columns"])


def test_shared_dashboard_refresh_waits_for_persisted_observations(tmp_path) -> None:
    """Native refresh follows saved evidence and policy evaluation in the existing job."""
    (tmp_path / "resources").mkdir()
    configure = runpy.run_path(str(TEMPLATE / "src/tools/configure_monitoring_dashboard.py"))[
        "configure"
    ]
    path = configure(tmp_path, dashboard_id="01f1c090238e1b6da5d633032ad9960b")
    tasks = yaml.safe_load(path.read_text())["targets"]["test"]["resources"]["jobs"]["monitoring"][
        "tasks"
    ]
    assert tasks[0]["depends_on"] == [{"task_key": "evaluate_retraining"}]
    assert tasks[1]["depends_on"] == [{"task_key": "dashboard_refresh_enabled", "outcome": "true"}]
    assert tasks[1]["dashboard_task"] == {"dashboard_id": "01f1c090238e1b6da5d633032ad9960b"}
    assert "subscription" not in tasks[1]["dashboard_task"]


def test_issue_investigation_paginates_without_truncating_filtered_total() -> None:
    """Issue filters must apply to all evidence before display pagination or counting."""
    dashboard = _dashboard()
    dataset = next(d for d in dashboard["datasets"] if d["name"] == "monitoring_issues")
    sql = "".join(dataset["queryLines"])
    assert "FROM monitoring_issues" in sql and "LIMIT " not in sql.upper()
    assert "ORDER BY priority, measured_at DESC" in sql
    widgets = {w["name"]: w for p in dashboard["pages"] for w in _widgets(p)}
    assert "itemsPerPage" not in widgets["issues_to_investigate"]["spec"]
    assert widgets["issues_to_investigate"]["spec"]["version"] == 2
    count = widgets["issues_count"]["queries"][0]["query"]
    assert count["datasetName"] == "monitoring_issues"
    assert count["fields"] == [{"name": "issue_count", "expression": "COUNT(*)"}]
    for name in ("filter_model", "filter_version", "filter_context"):
        widget = widgets[name]
        assert any(q["query"]["datasetName"] == "monitoring_issues" for q in widget["queries"])
    columns = widgets["issues_to_investigate"]["spec"]["encodings"]["columns"]
    assert any(c.get("link", {}).get("templatedURL") == "{{evidence_run_link}}" for c in columns)


def test_training_lineage_uses_concrete_identity_and_captured_alias_labels() -> None:
    """Model detail metadata must stay bound to one concrete historical identity."""
    dashboard = _dashboard()
    dataset = next(d for d in dashboard["datasets"] if d["name"] == "training_lineage")
    sql = "".join(dataset["queryLines"])
    assert "FROM model_version_details" in sql
    assert "LIMIT " not in sql.upper()
    assert "model version unresolved" in sql
    for field in (
        "model_catalog",
        "model_schema",
        "model_name",
        "model_version",
        "monitor_context",
    ):
        assert field in sql
    page = next(p for p in dashboard["pages"] if p["name"] == "training_lineage")
    text = json.dumps(page)
    assert "Aliases at capture" in text and "Captured at" in text
    assert "training_run_link" in text and "project_source_sha256" in text


def test_global_identity_filters_preserve_overview_to_lineage_scope() -> None:
    """Shared native filter state must not select a different model on the lineage page."""
    dashboard = _dashboard()
    global_page = next(p for p in dashboard["pages"] if p["pageType"] == "PAGE_TYPE_GLOBAL_FILTERS")
    widgets = {w["name"]: w for w in _widgets(global_page)}
    assert len(widgets) == 3
    for control in widgets.values():
        assert {"training_lineage", "monitoring_issues"} <= {
            q["query"]["datasetName"] for q in control["queries"]
        }
    page = next(p for p in dashboard["pages"] if p["name"] == "training_lineage")
    assert not any(
        w.get("spec", {}).get("widgetType", "").startswith("filter-") for w in _widgets(page)
    )
    for widget in _widgets(page):
        if widget.get("spec", {}).get("widgetType") == "table":
            assert {"model_name", "model_version", "monitor_context"} <= {
                c["fieldName"] for c in widget["spec"]["encodings"]["columns"]
            }
    assert not any(d["name"].startswith("lineage_") for d in dashboard["datasets"])


def test_investigation_text_widgets_use_native_multiline_serialization() -> None:
    """Native dashboard headings and operator guidance must survive widget serialization."""
    widgets = {w["name"]: w for p in _dashboard()["pages"] for w in _widgets(p)}
    names = {
        "issues_heading",
        "issues_note",
        "training_lineage_title",
        "training_lineage_intro",
        "training_lineage_note",
    }
    for name in names:
        widget = widgets[name]
        assert "textbox_spec" not in widget
        assert widget["multilineTextboxSpec"]["lines"]
        assert all(isinstance(line, str) for line in widget["multilineTextboxSpec"]["lines"])
