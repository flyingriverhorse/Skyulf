"""Execute overview SQL against enrollment and saved-performance edge cases."""

import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

DASHBOARD = (
    Path(__file__).resolve().parents[3]
    / "templates/databricks/template/{{.project_name}}/src/monitoring/monitoring.lvdash.json"
)
NOW = "2026-10-05 12:00:00"


def _dashboard() -> dict:
    """Read the deployable dashboard rather than a duplicate query fixture."""
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


def _json_field(value: str | None, path: str) -> str | None:
    """Provide Spark's scalar JSON extraction for the SQLite execution fixture."""
    result = json.loads(value or "null")
    for key in path.removeprefix("$.").split("."):
        result = result.get(key) if isinstance(result, dict) else None
    return None if result is None else str(result)


@pytest.fixture
def database():
    """Build the real input columns without requiring a local Spark installation."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.create_function("get_json_object", 2, _json_field)
    connection.create_function("test_now", 0, lambda: NOW)
    connection.create_function("floor", 1, lambda value: math.floor(value) if value else value)
    connection.create_function(
        "unix_timestamp",
        1,
        lambda value: (
            datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp()
            if value is not None
            else None
        ),
    )
    connection.executescript(
        "CREATE TABLE model_inventory (monitor_id TEXT, model_catalog TEXT, "
        "model_schema TEXT, model_name TEXT, enabled BOOLEAN, config_digest TEXT, "
        "config_json TEXT, expected_interval_hours REAL);"
        "CREATE TABLE current_health (monitor_id TEXT, model_version TEXT);"
        "CREATE TABLE performance_history (monitor_id TEXT, config_digest TEXT, "
        "model_version TEXT, report_id TEXT, window_end TEXT, measured_at TEXT, "
        "status TEXT, reason TEXT);"
    )
    yield connection
    connection.close()


def _enroll(database, monitor="monitor-1", *, mode="report", enabled=True, version="2"):
    """Keep two enrollments of the same model distinct in every result."""
    config = {
        "model_version": version,
        "performance_policy": {
            "mode": mode,
            "window_hours": 24,
            "label_delay_hours": 24,
        },
    }
    database.execute(
        "INSERT INTO model_inventory VALUES (?, 'catalog', 'schema', "
        "'catalog.schema.model', ?, 'current', ?, 24)",
        (monitor, enabled, json.dumps(config)),
    )
    database.execute("INSERT INTO current_health VALUES (?, ?)", (monitor, version))


def _report(
    database,
    report="report-1",
    *,
    monitor="monitor-1",
    config="current",
    version="2",
    status="healthy",
    window="2026-10-04 00:00:00",
    measured="2026-10-05 11:00:00",
    reason="within_tolerance",
):
    """Add saved evidence with independently controllable event and arrival times."""
    database.execute(
        "INSERT INTO performance_history VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (monitor, config, version, report, window, measured, status, reason),
    )


def _results(database) -> list[dict]:
    """Run shipped SQL, translating only Spark's current-time function syntax."""
    datasets = {item["name"]: item for item in _dashboard()["datasets"]}
    assert "overview_performance" in datasets, "Overview lacks independent performance evidence"
    sql = "".join(datasets["overview_performance"]["queryLines"])
    sql = sql.replace("current_timestamp()", "test_now()")
    return [dict(row) for row in database.execute(sql).fetchall()]


@pytest.mark.parametrize(
    "status,label,count",
    [
        ("healthy", "Within tolerance", "within_tolerance_count"),
        ("degraded", "Performance loss", "performance_loss_count"),
        ("unavailable", "Insufficient evidence", "insufficient_evidence_count"),
        ("disabled", "Insufficient evidence", "insufficient_evidence_count"),
    ],
)
def test_overview_uses_performance_status_independently(database, status, label, count):
    """Drift health and saved policy status must not stand in for each other."""
    _enroll(database)
    _report(database, status=status)
    result = _results(database)[0]
    assert (result["performance_status"], result[count]) == (label, 1)


@pytest.mark.parametrize("config,version", [("old", "2"), ("current", "1")])
def test_old_config_or_version_is_not_current_evidence(database, config, version):
    """Re-enrollment and concrete-version changes must invalidate historical success."""
    _enroll(database)
    _report(database, config=config, version=version)
    result = _results(database)[0]
    assert (result["performance_status"], result["performance_reason"]) == (
        "Insufficient evidence",
        "no_matching_report",
    )


@pytest.mark.parametrize(
    "mode,enabled,label",
    [
        ("report", True, "Insufficient evidence"),
        ("off", True, "Policy disabled"),
        ("report", False, "Monitoring disabled"),
        ("off", False, "Monitoring disabled"),
    ],
)
def test_inventory_policy_and_enabled_state_are_authoritative(database, mode, enabled, label):
    """Missing reports remain visible and disabled enrollments cannot be healthy."""
    _enroll(database, mode=mode, enabled=enabled)
    if not enabled or mode == "off":
        _report(database)
    result = _results(database)[0]
    assert result["performance_status"] == label
    assert result["within_tolerance_count"] == result["performance_loss_count"] == 0


def test_latest_unavailable_suppresses_previous_valid_evidence(database):
    """Unavailable latest policy results must never fall back to older success."""
    _enroll(database)
    _report(database, "old", measured="2026-10-05 10:00:00")
    _report(database, "new", status="unavailable", reason="insufficient_labels")
    result = _results(database)[0]
    assert (result["report_id"], result["performance_reason"]) == ("new", "insufficient_labels")
    assert result["performance_status"] == "Insufficient evidence"


def test_late_backfill_cannot_replace_newer_window(database):
    """Late arrival is subordinate to the actual saved performance window."""
    _enroll(database)
    _report(database, "new-window", status="degraded")
    _report(database, "backfill", window="2026-10-02 00:00:00", measured=NOW)
    result = _results(database)[0]
    assert (result["report_id"], result["performance_status"]) == ("new-window", "Performance loss")


def test_report_id_breaks_equal_window_and_measurement_ties(database):
    """Ties must select one reproducible result rather than duplicate enrollment counts."""
    _enroll(database)
    _report(database, "a")
    _report(database, "z", status="unavailable", reason="missing_baseline")
    results = _results(database)
    assert len(results) == 1
    assert results[0]["report_id"] == "z"
    assert results[0]["performance_status"] == "Insufficient evidence"


def test_stale_saved_window_is_insufficient_even_when_recently_measured(database):
    """A freshly run backfill must not make old evidence look current."""
    _enroll(database)
    _report(database, window="2026-10-02 00:00:00", measured=NOW)
    result = _results(database)[0]
    assert (result["performance_status"], result["performance_reason"]) == (
        "Insufficient evidence",
        "stale_performance_window",
    )


@pytest.mark.parametrize(
    "window,measured,reason",
    [
        ("2026-10-05 00:00:00", NOW, "performance_window_not_mature"),
        ("2026-10-04 00:00:00", "2026-10-06 00:00:00", "future_measurement"),
        (None, NOW, "missing_performance_window"),
    ],
)
def test_missing_or_future_timestamps_are_insufficient(database, window, measured, reason):
    """Malformed evidence cannot count as a mature performance measurement."""
    _enroll(database)
    _report(database, window=window, measured=measured)
    result = _results(database)[0]
    assert (result["performance_status"], result["performance_reason"]) == (
        "Insufficient evidence",
        reason,
    )


def test_omitted_policy_defaults_to_disabled(database):
    """Canonical default-off configurations must not demand nonexistent evidence."""
    _enroll(database)
    database.execute("UPDATE model_inventory SET config_json = '{}' ")
    assert _results(database)[0]["performance_status"] == "Policy disabled"


def test_unresolved_alias_cannot_reuse_an_arbitrary_historical_version(database):
    """Alias enrollment needs a resolved current concrete version before matching reports."""
    _enroll(database, version=None)
    _report(database)
    result = _results(database)[0]
    assert (result["performance_status"], result["performance_reason"]) == (
        "Insufficient evidence",
        "unresolved_model_version",
    )


def test_counts_are_enrollments_and_shared_selectors_include_performance(database):
    """The same model in two enrollments must count twice and use existing filters."""
    _enroll(database)
    _enroll(database, "monitor-2")
    _report(database)
    _report(database, "report-2", monitor="monitor-2")
    results = _results(database)
    assert sum(row["within_tolerance_count"] for row in results) == 2
    page = next(page for page in _dashboard()["pages"] if page["name"] == "overview")
    widgets = {item["widget"]["name"]: item["widget"] for item in page["layout"]}
    for field in ("model_catalog", "model_schema", "model_name", "model_version"):
        assert any(
            query["query"]["datasetName"] == "overview_performance"
            for query in widgets[f"filter_{field}_overview"]["queries"]
        )
    for counter in (
        "within_tolerance_count",
        "performance_loss_count",
        "insufficient_evidence_count",
        "policy_disabled_count",
    ):
        assert widgets[counter]["queries"][0]["query"]["datasetName"] == "overview_performance"
