"""Contract checks for the central, portable monitoring dashboard example."""

import json
import re
import sqlite3
from pathlib import Path

import yaml

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "databricks_monitoring"


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
    assert set(pages) == {"overview", "drift_quality", "performance", "execution"}
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
        if w.get("spec", {}).get("widgetType") in {"bar", "line"}
    } == {"drift_psi", "drift_trend"}
    assert (
        sum(w.get("spec", {}).get("widgetType") == "line" for w in _widgets(performance_page)) == 2
    )
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
            control = widgets[f"filter_{field}_drift_quality"]
            assert control["spec"]["widgetType"] == "filter-single-select"
            assert any(q["query"]["datasetName"] == dataset for q in control["queries"])
    psi = datasets["drift_psi_chart"]
    assert "metric_name = 'psi'" in psi
    assert "threshold AS plot_value" in psi
    assert "report_rank = 1" in psi and "LIMIT 10" in psi
    assert "FROM monitoring_results" in psi
    assert "window_end DESC" in psi
    assert "metric_name <> 'ks_test_p_value'" in datasets["drift_trend"]
    assert "current_timezone()" in datasets["drift_trend"]
    assert widgets["drift_trend"]["spec"]["encodings"]["x"]["fieldName"] == "measurement_time"
    latest = datasets["performance_latest"]
    assert "ORDER BY window_end DESC, measured_at DESC, report_id DESC" in latest
    assert "value IS NOT NULL" not in latest
    assert "FROM monitoring_results" in latest
    assert "history.report_id = latest_reports.report_id" in latest
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


def test_performance_policy_evidence_and_trend_bind_to_saved_view() -> None:
    """Policy status and threshold series must come from the saved policy evidence."""
    dashboard = _dashboard()
    datasets = {item["name"]: "".join(item["queryLines"]) for item in dashboard["datasets"]}
    page = next(page for page in dashboard["pages"] if page["name"] == "performance")
    widgets = {widget["name"]: widget for widget in _widgets(page)}
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
    assert len(columns) <= 8
    assert all(field in datasets["performance_policy"] for field in evidence_fields)
    line = widgets["performance_policy_trend"]
    assert line["spec"]["widgetType"] == "line"
    assert line["spec"]["encodings"]["color"]["fieldName"] == "series"
    assert line["queries"][0]["query"]["datasetName"] == "performance_policy_series"
    for field in ("model_name", "model_version"):
        control = widgets[f"filter_{field}_performance"]
        assert any(
            query["query"]["datasetName"] == "performance_policy" for query in control["queries"]
        )
    assert any(
        query["query"]["datasetName"] == "performance_policy_series"
        for query in widgets["filter_measured_at_performance"]["queries"]
    )


def test_context_selection_keeps_unobserved_models_and_resolves_one_identity() -> None:
    """An enrolled model must remain selectable when another monitor shares its version."""
    dashboard = _dashboard()
    datasets = {item["name"]: "".join(item["queryLines"]) for item in dashboard["datasets"]}
    assert "FROM current_health" in datasets["context_choices"]
    assert "FROM monitoring_results" in datasets["context_choices"]
    for name in ("metric_history", "quality_history", "performance_measurements"):
        assert "chosen_context" in datasets[name]
        assert "LIMIT 1" in datasets[name]
        assert "source.monitor_id = context.monitor_id" in datasets[name]
    for name in ("drift_context",):
        assert "monitor_context" in datasets[name]
        assert "No matching enrollment" in datasets[name]
    assert {
        parameter["keyword"]
        for dataset in dashboard["datasets"]
        for parameter in dataset.get("parameters", [])
    } == {
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


def test_nullable_context_parameters_keep_latest_and_explicit_selections() -> None:
    """Clearing a selector must choose the latest identity instead of hiding every row."""
    datasets = _dashboard()["datasets"]
    expected = {
        dataset["name"]
        for dataset in datasets
        if any(p["keyword"].endswith("_model") for p in dataset.get("parameters", []))
    }
    checked: set[str] = set()
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE available_contexts (monitor_id TEXT, model_name TEXT, "
            "model_version TEXT, monitor_context TEXT, latest_measurement INTEGER)"
        )
        connection.executemany(
            "INSERT INTO available_contexts VALUES (?, ?, ?, ?, ?)",
            [("old", "model-a", "1", "project-a", 1), ("new", "model-b", "2", "project-b", 2)],
        )
        for dataset in datasets:
            sql = "".join(dataset["queryLines"])
            match = re.search(r"SELECT \* FROM available_contexts\n WHERE .*?LIMIT 1", sql, re.S)
            if match is None:
                continue
            checked.add(dataset["name"])
            query = match.group()
            keywords = re.findall(r":([a-z_]+)", query)
            for empty in (None, ""):
                parameters = dict.fromkeys(keywords, empty)
                assert connection.execute(query, parameters).fetchone()[0] == "new", dataset["name"]
                parameters.update({key: "model-a" for key in keywords if key.endswith("_model")})
                assert connection.execute(query, parameters).fetchone()[0] == "old", dataset["name"]
                parameters.update({key: "2" for key in keywords if key.endswith("_version")})
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
                if "fieldName" in field
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
        "'1' AS model_version), ranked_reports AS (" + tail
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
        assert connection.execute(sql).fetchone()["scored_rows"] is None
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
        row = connection.execute(sql).fetchone()
        assert (row["scored_rows"], row["labeled_rows"], row["label_coverage"]) == (40, 20, 0.5)
        connection.execute(
            "INSERT INTO monitoring_results VALUES ('m','model','1','failed',3,3,'failed','{}',0,0,0)"
        )
        row = connection.execute(sql).fetchone()
        assert all(row[name] is None for name in ("scored_rows", "labeled_rows", "label_coverage"))


def test_policy_summary_keeps_latest_unavailable_window_and_its_own_metric() -> None:
    """Historical loss must not replace a missing latest policy check or depend on chart metric."""
    datasets = {d["name"]: d for d in _dashboard()["datasets"]}
    sql = "".join(datasets["performance_policy_snapshot"]["queryLines"])
    assert "ORDER BY window_end DESC, measured_at DESC, report_id DESC" in sql
    assert "WHERE status = 'degraded'" not in sql and "current_value IS NOT NULL" not in sql
    tail = sql.split("), policy_ranked AS (", 1)[1]
    query = (
        "WITH chosen_context AS (SELECT 'm' AS monitor_id, 'model' AS model_name, "
        "'1' AS model_version), policy_ranked AS (" + tail
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
        row = connection.execute(query).fetchone()
        assert row["report_id"] == "latest" and row["current_value"] is None
        assert row["policy_status"] == "Insufficient evidence"
        connection.execute(
            "INSERT INTO performance_history VALUES "
            "('m','model','1','disabled',3,3,NULL,'disabled','policy_off','none',NULL,NULL,NULL)"
        )
        assert connection.execute(query).fetchone()["policy_status"] == "Policy disabled"
    for name in ("performance_policy_snapshot", "performance_policy_series"):
        assert "performance_metric" not in {p["keyword"] for p in datasets[name]["parameters"]}
    series = "".join(datasets["performance_policy_series"]["queryLines"])
    assert "SELECT metric AS metric_name FROM performance_history" in series
    widgets = {w["name"]: w for p in _dashboard()["pages"] for w in _widgets(p)}
    assert "performance_context" not in widgets
    metric = widgets["filter_metric_name_performance"]
    assert all(q["query"]["datasetName"] != "performance_policy_series" for q in metric["queries"])


def test_performance_checks_explain_results_without_claiming_training_started() -> None:
    """A loss or eligible request must not be mislabeled as an already submitted training run."""
    dashboard = _dashboard()
    dataset = next(d for d in dashboard["datasets"] if d["name"] == "performance_policy")
    sql = "".join(dataset["queryLines"])
    labels = sql.split("SELECT source.*,", 1)[1].split("FROM (SELECT", 1)[0]
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
    assert "performance_context" not in widgets
    history = widgets["performance_policy_evidence"]["spec"]
    assert history["frame"]["title"] == "Performance checks"
    assert "monitoring_report" not in history["frame"]["description"]
    assert any(c["displayName"] == "Performance result" for c in history["encodings"]["columns"])


def test_shared_dashboard_refresh_waits_for_persisted_observations() -> None:
    """Adding the optional central dashboard must also refresh it after measurements succeed."""
    resource = yaml.safe_load((EXAMPLE / "resources/monitoring_refresh.job.yml").read_text())
    task = resource["targets"]["dev"]["resources"]["jobs"]["monitoring"]["tasks"][0]
    assert task["depends_on"] == [{"task_key": "observe_models"}]
    assert task["dashboard_task"] == {
        "dashboard_id": "${resources.dashboards.monitoring_dashboard.id}",
        "warehouse_id": "${var.warehouse_id}",
    }
    assert "subscription" not in task["dashboard_task"]
