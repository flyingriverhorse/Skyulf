"""Execute the shipped Overview summaries against inventory and history edge cases."""

import json
import math
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

DASHBOARD = (
    Path(__file__).resolve().parents[3]
    / "templates/databricks/template/{{.project_name}}/src/monitoring/monitoring.lvdash.json"
)
NEW_DATASETS = {"overview_inventory", "overview_drift_daily", "overview_performance_daily"}


def _dashboard():
    """Load the deployable payload so fixtures cannot diverge from its queries."""
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


def _json_field(value, path):
    """Match Spark scalar JSON extraction while keeping missing values null."""
    result = json.loads(value or "null")
    for key in path.removeprefix("$.").split("."):
        result = result.get(key) if isinstance(result, dict) else None
    return None if result is None else str(result)


def _epoch(value):
    """Preserve explicit offsets when deriving the UTC day from saved instants."""
    if value is None:
        return None
    instant = datetime.fromisoformat(value)
    return (
        instant.replace(tzinfo=UTC).timestamp() if instant.tzinfo is None else instant.timestamp()
    )


@pytest.fixture
def database():
    """Model real input columns and only shim Spark scalar functions for SQLite."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.create_function("get_json_object", 2, _json_field)
    connection.create_function("unix_timestamp", 1, _epoch)
    connection.create_function("floor", 1, lambda value: math.floor(value) if value else value)
    connection.create_function("concat", -1, lambda *values: "".join(str(v) for v in values))
    connection.create_function(
        "date_add",
        2,
        lambda value, days: (
            (datetime.fromisoformat(value) + timedelta(days=days)).date().isoformat()
        ),
    )
    connection.executescript(
        "CREATE TABLE model_inventory (monitor_id TEXT, model_catalog TEXT, model_schema TEXT, "
        "model_name TEXT, environment TEXT, project TEXT, enabled BOOLEAN, config_digest TEXT, "
        "config_json TEXT);"
        "CREATE TABLE current_health (monitor_id TEXT, model_catalog TEXT, model_schema TEXT, "
        "model_name TEXT, model_version TEXT);"
        "CREATE TABLE monitoring_results (monitor_id TEXT, config_digest TEXT, model_version TEXT, "
        "report_id TEXT, window_end TEXT, measured_at TEXT, status TEXT, report_json TEXT);"
        "CREATE TABLE performance_history (monitor_id TEXT, config_digest TEXT, model_version TEXT, "
        "report_id TEXT, window_end TEXT, measured_at TEXT, status TEXT);"
        "CREATE TABLE metric_history (report_id TEXT, category TEXT, metric_name TEXT, "
        "status TEXT, has_issue BOOLEAN);"
    )
    yield connection
    connection.close()


def _enroll(
    database,
    monitor="m1",
    *,
    model="cat.schema.model",
    version="2",
    enabled=True,
    mode="report",
    source="cat.data.inputs",
    prediction="cat.data.predictions",
    label="cat.data.targets",
):
    """Allow versions and enrollment contexts to share one registered model identity."""
    config = {
        "model_version": version,
        "source_table": source,
        "prediction_table": prediction,
        "label_table": label,
        "performance_policy": {"mode": mode},
    }
    catalog, schema, _ = model.split(".")
    database.execute(
        "INSERT INTO model_inventory VALUES (?, ?, ?, ?, 'prod', ?, ?, 'current', ?)",
        (monitor, catalog, schema, model, monitor, enabled, json.dumps(config)),
    )
    database.execute(
        "INSERT INTO current_health VALUES (?, ?, ?, ?, ?)",
        (monitor, catalog, schema, model, version),
    )


def _report(
    database,
    report="r1",
    *,
    monitor="m1",
    config="current",
    version="2",
    window="2026-10-04 18:00:00",
    measured="2026-10-05 10:00:00",
    status="healthy",
    issue=False,
    metric_status="measured",
    policy_only=False,
):
    """Create separate policy and drift evidence with controllable arrival order."""
    payload = {"performance": {"status": status}} if policy_only else {"current_rows": 10}
    database.execute(
        "INSERT INTO monitoring_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            monitor,
            config,
            version,
            report,
            window,
            measured,
            "no_data" if policy_only else status,
            json.dumps(payload),
        ),
    )
    database.execute(
        "INSERT INTO performance_history VALUES (?, ?, ?, ?, ?, ?, ?)",
        (monitor, config, version, report, window, measured, status),
    )
    if not policy_only:
        database.execute(
            "INSERT INTO metric_history VALUES (?, 'drift', 'psi', ?, ?)",
            (report, metric_status, issue),
        )


def _sql(dataset):
    """Fail explicitly when a required dataset has not yet been implemented."""
    datasets = {item["name"]: item for item in _dashboard()["datasets"]}
    assert dataset in datasets, f"Overview is missing {dataset}"
    return "".join(datasets[dataset]["queryLines"]).replace("DATE '1970-01-01'", "'1970-01-01'")


def _rows(database, dataset):
    """Execute production SQL without replacing its joins, filters or ranking."""
    return [dict(row) for row in database.execute(_sql(dataset))]


def _daily(database, dataset):
    """Apply the exact chart aggregation over the actual dataset output."""
    rows = _rows(database, dataset)
    result = {}
    for row in rows:
        key = (row["window_day"], row["series"])
        value = row["context_count"]
        result.setdefault(key, None)
        if value is not None:
            result[key] = (result[key] or 0) + value
    return result


def test_distinct_counters_keep_model_versions_and_shared_tables_together(database):
    """Two enrolled versions count as one model while their contexts remain separate."""
    _enroll(database)
    _enroll(database, "m2", version="3")
    _enroll(
        database,
        "m3",
        model="other.schema.model",
        source="other.data.inputs",
        prediction=None,
        label=None,
    )
    page = next(p for p in _dashboard()["pages"] if p["name"] == "overview")
    widgets = {item["widget"]["name"]: item["widget"] for item in page["layout"]}
    counts = {}
    for name in (
        "unique_models_count",
        "enrolled_contexts_count",
        "source_tables_count",
        "prediction_tables_count",
    ):
        assert name in widgets, f"Overview is missing {name}"
        expression = widgets[name]["queries"][0]["query"]["fields"][0]["expression"]
        counts[name] = database.execute(
            f"SELECT {expression} FROM ({_sql('overview_inventory')})"
        ).fetchone()[0]
    assert counts == {
        "unique_models_count": 2,
        "enrolled_contexts_count": 3,
        "source_tables_count": 2,
        "prediction_tables_count": 1,
    }
    rows = _rows(database, "overview_inventory")
    missing = next(row for row in rows if row["monitor_id"] == "m3")
    assert missing["prediction_table"] is None
    assert missing["prediction_table_display"] == missing["label_table_display"] == "Not configured"


@pytest.mark.parametrize("dataset", ["overview_drift_daily", "overview_performance_daily"])
def test_trends_use_current_enrollment_configuration_and_concrete_version(database, dataset):
    """Old configurations, old versions and disabled enrollments cannot inflate trends."""
    _enroll(database)
    _enroll(database, "disabled", enabled=False)
    _enroll(database, "alias", version=None)
    _report(database, "old-config", config="old")
    _report(database, "old-version", version="1")
    _report(database, "disabled", monitor="disabled")
    _report(database, "unresolved", monitor="alias")
    assert _rows(database, dataset) == []


@pytest.mark.parametrize("dataset", ["overview_drift_daily", "overview_performance_daily"])
def test_daily_last_window_wins_over_late_backfills_and_retries(database, dataset):
    """One enrollment contributes once per UTC window day, ordered by event before arrival."""
    _enroll(database)
    _report(database, "backfill", window="2026-10-04 09:00:00", measured="2026-10-06 10:00:00")
    _report(database, "a", status="healthy")
    _report(database, "z", status="unavailable", metric_status="unavailable")
    rows = _rows(database, dataset)
    assert len(rows) == 3
    assert {row["report_id"] for row in rows} == {"z"}
    daily = _daily(database, dataset)
    unavailable = (
        "Checks unavailable" if dataset == "overview_drift_daily" else "Insufficient evidence"
    )
    assert daily[("2026-10-04", unavailable)] == 1


def test_performance_counts_shared_models_separately_and_never_fills_gaps(database):
    """Unavailable and unsampled days must not appear as healthy zero-loss evidence."""
    _enroll(database)
    _enroll(database, "m2")
    _enroll(database, "off", mode="off")
    _enroll(database, "never")
    _report(database, status="degraded")
    _report(database, "r2", monitor="m2", status="unavailable")
    _report(database, "off", monitor="off", status="degraded")
    _report(database, "day6", window="2026-10-06 18:00:00")
    daily = _daily(database, "overview_performance_daily")
    assert daily[("2026-10-04", "Performance loss")] == 1
    assert daily[("2026-10-04", "Measured")] == 1
    assert daily[("2026-10-04", "Insufficient evidence")] == 1
    assert {day for day, _ in daily} == {"2026-10-04", "2026-10-06"}


def test_drift_uses_feature_checks_and_excludes_policy_only_observations(database):
    """Policy loss, quality issues and diagnostic p-values are not drift detections."""
    _enroll(database)
    _enroll(database, "m2")
    _report(database, status="degraded", issue=False)
    _report(database, "r2", monitor="m2", metric_status="unavailable")
    _report(database, "policy", window="2026-10-04 23:00:00", policy_only=True)
    database.execute(
        "INSERT INTO metric_history VALUES ('r1', 'drift', 'ks_test_p_value', 'measured', 1)"
    )
    database.execute(
        "INSERT INTO metric_history VALUES ('r1', 'quality', 'missing_fraction', 'measured', 1)"
    )
    daily = _daily(database, "overview_drift_daily")
    assert daily == {
        ("2026-10-04", "Drift detected"): 0,
        ("2026-10-04", "Checks measured"): 1,
        ("2026-10-04", "Checks unavailable"): 1,
    }


@pytest.mark.parametrize("status", ["failed", "no_data"])
def test_failed_or_empty_latest_window_cannot_reuse_older_drift_checks(database, status):
    """A failed or empty full observation overrides earlier measured evidence that day."""
    _enroll(database)
    _report(database, "earlier", window="2026-10-04 09:00:00", issue=True)
    _report(database, "latest", status=status, issue=True)
    daily = _daily(database, "overview_drift_daily")
    assert daily == {
        ("2026-10-04", "Drift detected"): None,
        ("2026-10-04", "Checks measured"): 0,
        ("2026-10-04", "Checks unavailable"): 1,
    }


def test_missing_drift_metrics_are_unavailable_and_multiple_checks_do_not_inflate(database):
    """Metric-level multiplicity must not turn one enrollment into several checked contexts."""
    _enroll(database)
    _enroll(database, "m2")
    _report(database, issue=True)
    _report(database, "missing", monitor="m2")
    database.execute("DELETE FROM metric_history WHERE report_id = 'missing'")
    database.execute(
        "INSERT INTO metric_history VALUES ('r1', 'drift', 'ks_statistic', 'measured', 1)"
    )
    daily = _daily(database, "overview_drift_daily")
    assert daily == {
        ("2026-10-04", "Drift detected"): 1,
        ("2026-10-04", "Checks measured"): 1,
        ("2026-10-04", "Checks unavailable"): 1,
    }


@pytest.mark.parametrize("status", ["unavailable", "disabled", "unexpected", None])
def test_unusable_performance_status_has_no_zero_loss_claim(database, status):
    """Only measured policy verdicts can say a sampled day has no performance loss."""
    _enroll(database)
    _report(database, status=status)
    daily = _daily(database, "overview_performance_daily")
    assert daily == {
        ("2026-10-04", "Performance loss"): None,
        ("2026-10-04", "Measured"): 0,
        ("2026-10-04", "Insufficient evidence"): 1,
    }


def test_alias_resolution_requires_matching_current_health_identity(database):
    """A foreign model row cannot supply the concrete version of an unresolved alias."""
    _enroll(database, version=None)
    database.execute(
        "UPDATE current_health SET model_name = 'other.schema.model', model_version = '2'"
    )
    _report(database)
    assert _rows(database, "overview_inventory")[0]["model_version"] is None
    assert _rows(database, "overview_drift_daily") == []
    assert _rows(database, "overview_performance_daily") == []


@pytest.mark.parametrize("dataset", ["overview_drift_daily", "overview_performance_daily"])
def test_window_day_is_utc_not_measurement_arrival_or_session_day(database, dataset):
    """An offset-crossing window belongs to its UTC date even after a later retry."""
    _enroll(database)
    _report(database, window="2026-10-05T00:30:00+03:00")
    assert {row["window_day"] for row in _rows(database, dataset)} == {"2026-10-04"}


def test_overview_filters_mapping_and_chart_layout_are_complete():
    """All additions follow existing selectors and show compact, non-overlapping content."""
    page = next(p for p in _dashboard()["pages"] if p["name"] == "overview")
    widgets = {item["widget"]["name"]: item["widget"] for item in page["layout"]}
    for field in ("model_catalog", "model_schema", "model_name", "model_version"):
        bound = {q["query"]["datasetName"] for q in widgets[f"filter_{field}_overview"]["queries"]}
        assert bound >= NEW_DATASETS
    table = widgets["model_table_mapping"]
    columns = {c["fieldName"] for c in table["spec"]["encodings"]["columns"]}
    assert {
        "model_name",
        "model_version",
        "monitor_context",
        "source_table_display",
        "prediction_table_display",
        "label_table_display",
    } <= columns
    assert "monitor_id" not in columns
    for name in ("overview_drift_trend", "overview_performance_trend"):
        spec = widgets[name]["spec"]
        assert (spec["version"], spec["widgetType"]) == (3, "line")
        assert spec["mark"]["marker"]["shape"] == "circle"
    occupied = set()
    for item in page["layout"]:
        p = item["position"]
        cells = {
            (x, y)
            for x in range(p["x"], p["x"] + p["width"])
            for y in range(p["y"], p["y"] + p["height"])
        }
        assert p["x"] + p["width"] <= 12
        assert not occupied & cells
        occupied |= cells
    assert {
        "within_tolerance_count",
        "performance_loss_count",
        "healthy_count",
        "drift_count",
    } <= widgets.keys()
