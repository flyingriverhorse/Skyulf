"""Shared monitoring persistence binds snapshots and preserves unobserved models."""

import json
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_config import MonitorConfig

COUNT_COLUMNS = ("reference_rows", "current_rows", "scored_rows", "labeled_rows", "drifted_columns")


def config():
    """Provide a concrete model outside the shared monitoring catalog."""
    return MonitorConfig(
        environment="test",
        project="risk",
        model_name="models.risk.score",
        model_version="2",
        source_table="features.risk.inputs",
        prediction_table="outputs.risk.predictions",
    )


def test_result_identity_retries_and_late_labels():
    """Retrying exact evidence deduplicates, while a new label snapshot retains new history."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import result_row

    now = datetime(2026, 10, 1, tzinfo=UTC)
    report = {"status": "healthy", "current_rows": 4, "metrics": [], "notes": []}
    evidence = {"prediction_version": 2, "label_version": 3}
    first = result_row(config(), "2", now, now, now, report, evidence)
    retry = result_row(config(), "2", now, now, now, report, evidence)
    changed = result_row(config(), "2", now, now, now, report, evidence | {"label_version": 4})
    assert first["report_id"] == retry["report_id"]
    assert first["report_id"] != changed["report_id"]
    assert first["model_name"] == "models.risk.score"


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({}, dict.fromkeys(COUNT_COLUMNS)),
        (dict.fromkeys(COUNT_COLUMNS), dict.fromkeys(COUNT_COLUMNS)),
        (dict.fromkeys(COUNT_COLUMNS, 0), dict.fromkeys(COUNT_COLUMNS, 0)),
        (
            {"reference_rows": "12", "current_rows": 0, "scored_rows": 7},
            {
                "reference_rows": 12,
                "current_rows": 0,
                "scored_rows": 7,
                "labeled_rows": None,
                "drifted_columns": None,
            },
        ),
    ],
    ids=["absent", "null", "zero", "partial"],
)
def test_result_counts_distinguish_unmeasured_from_zero(counts, expected):
    """Incomplete reports must not fabricate measured zeros or discard available counts."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import result_row

    now = datetime(2026, 10, 1, tzinfo=UTC)
    report = {"status": "failed", **counts}
    row = result_row(config(), "2", now, now, now, report, {})
    assert {key: row[key] for key in COUNT_COLUMNS} == expected
    assert json.loads(row["report_json"]) == report


def test_optional_metadata_refresh_preserves_observation_identity():
    """Registry metadata refreshes must not duplicate the same measured observation."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import result_row

    now = datetime(2026, 10, 1, tzinfo=UTC)
    report = {"status": "healthy", "current_rows": 4, "metrics": []}
    evidence = {"model_digest": "verified", "training_run_id": "training"}
    first = result_row(config(), "2", now, now, now, report, evidence)
    refresh = result_row(
        config(),
        "2",
        now,
        now,
        now,
        report,
        evidence | {"model_metadata": {"captured_at": "later", "aliases": ["champion"]}},
    )
    assert first["report_id"] == refresh["report_id"]


def test_inventory_keeps_namespace_and_selection():
    """The dashboard must distinguish same-named models in different catalogs."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import (
        inventory_row,
    )

    row = inventory_row(config(), datetime(2026, 10, 1, tzinfo=UTC))
    assert row["model_catalog"] == "models"
    assert row["model_schema"] == "risk"
    assert row["selection"] == "version:2"


def test_current_health_exposes_pinned_inventory_version_before_observation():
    """A newly enrolled model must remain selectable before its first report arrives."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import (
        monitoring_views,
    )

    query = monitoring_views("ops.monitoring")["current_health"]
    assert "COALESCE(r.model_version, get_json_object(i.config_json, '$.model_version'))" in query
    assert "'$.model_set_name'" in query
    assert "'$.model_set_version'" in query


def test_current_health_does_not_rank_policy_only_windows_as_feature_observations():
    """Late-label policy windows must not hide a completed scoring report or its health."""
    import json

    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import (
        monitoring_views,
        result_row,
    )

    now = datetime(2026, 10, 5, tzinfo=UTC)
    row = result_row(
        config(),
        "2",
        now,
        now,
        now,
        {"status": "no_data", "metrics": [], "performance": {"status": "unavailable"}},
        {},
    )
    assert row["current_rows"] is None
    assert "current_rows" not in json.loads(row["report_json"])
    query = monitoring_views("ops.monitoring")["current_health"]
    assert "WHERE NOT (status = 'no_data'" in query
    assert "get_json_object(report_json, '$.current_rows') IS NULL" in query
    assert "get_json_object(report_json, '$.performance') IS NOT NULL)" in query


def test_persistence_rejects_nonfinite_report_values():
    """Invalid JSON cannot become a misleading dashboard metric."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import result_row

    now = datetime(2026, 10, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        result_row(config(), "2", now, now, now, {"status": "healthy", "x": float("nan")}, {})


def test_existing_foreign_table_is_not_adopted():
    """Central setup cannot overwrite a table just because its name matches."""
    from unittest.mock import Mock

    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import (
        ensure_owned_object,
    )

    spark = Mock()
    spark.catalog.tableExists.return_value = True
    spark.sql.return_value.first.return_value = {"value": "foreign"}
    with pytest.raises(ValueError, match="ownership"):
        ensure_owned_object(spark, "ops.monitoring.model_inventory")
    assert spark.sql.call_count == 1


def test_owned_incompatible_schema_requires_explicit_migration():
    """A stale central table must fail before views are replaced or reports are written."""
    from unittest.mock import Mock

    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import (
        OWNER,
        initialize_monitoring_store,
    )

    spark = Mock()
    spark.catalog.tableExists.return_value = True
    spark.sql.return_value.first.return_value = {"value": OWNER}
    spark.table.return_value.schema.simpleString.return_value = "struct<old:string>"
    spark.createDataFrame.return_value.schema.simpleString.return_value = "struct<new:string>"
    with pytest.raises(ValueError, match="schema differs"):
        initialize_monitoring_store(spark, "ops", "monitoring")
    assert all(call.args[0].startswith("SHOW TBLPROPERTIES") for call in spark.sql.call_args_list)


@pytest.mark.parametrize("view", ["monitoring_issues", "model_version_details"])
def test_new_investigation_view_ownership_is_checked_before_any_mutation(view):
    """An existing foreign investigation view must block the whole initializer."""
    from skyulf.integrations.databricks.observability.monitoring import monitoring_store as store

    spark = Mock()
    spark.catalog.tableExists.side_effect = lambda name: name.endswith(view)
    spark.sql.return_value.first.return_value = {"value": "foreign"}
    with pytest.raises(ValueError, match="ownership"):
        store.initialize_monitoring_store(spark, "ops", "monitoring")
    assert all(call.args[0].startswith("SHOW TBLPROPERTIES") for call in spark.sql.call_args_list)


def test_inventory_reads_other_repositories_without_file_input(monkeypatch):
    """The central runner discovers all registered projects from one bounded Delta snapshot."""
    from skyulf.integrations.databricks.observability.monitoring import monitoring_store as store

    first = config()
    second = MonitorConfig.from_dict(
        first.payload() | {"project": "sales", "model_name": "other.sales.model", "enabled": False}
    )
    spark = Mock()
    rows = [store.inventory_row(item, datetime.now(UTC)) for item in (first, second)]
    spark.table.return_value.select.return_value.limit.return_value.collect.return_value = rows
    monkeypatch.setattr(store, "ensure_owned_object", lambda *args: True)
    loaded = store.load_enrolled_models(spark, "ops.monitoring", max_models=2)
    assert [item["model_name"] for item in loaded] == [first.model_name, second.model_name]
    assert loaded[1]["enabled"] is False


def test_inventory_read_is_bounded_and_rejects_tampered_config(monkeypatch):
    """An oversized or inconsistent inventory must fail instead of silently dropping monitors."""
    from skyulf.integrations.databricks.observability.monitoring import monitoring_store as store

    monkeypatch.setattr(store, "ensure_owned_object", lambda *args: True)
    row = store.inventory_row(config(), datetime.now(UTC))
    spark = Mock()
    collect = spark.table.return_value.select.return_value.limit.return_value.collect
    collect.return_value = [row, row]
    with pytest.raises(ValueError, match="limit"):
        store.load_enrolled_models(spark, "ops.monitoring", max_models=1)
    collect.return_value = [row | {"config_digest": "wrong"}]
    with pytest.raises(ValueError, match="integrity"):
        store.load_enrolled_models(spark, "ops.monitoring")
    collect.return_value = [row | {"enabled": False}]
    with pytest.raises(ValueError, match="integrity"):
        store.load_enrolled_models(spark, "ops.monitoring")


@pytest.fixture
def native_store_namespace(delta_spark):
    """Isolate real store initialization and clean only this test's owned objects."""
    from skyulf.integrations.databricks.observability.monitoring import monitoring_store as store
    from skyulf.integrations.databricks.shared._contracts import table_name

    catalog = os.environ.get("SKYULF_TEST_CATALOG", "spark_catalog")
    schema = f"skyulf_store_test_{uuid4().hex}"
    namespace = store.store_namespace(catalog, schema)
    assert not delta_spark.catalog.databaseExists(namespace)
    try:
        yield catalog, schema
    finally:
        if delta_spark.catalog.databaseExists(namespace):
            allowed = {
                "model_inventory",
                "monitoring_results",
                "performance_actions",
                "retraining_requests",
                *store.monitoring_views(namespace),
            }
            for item in delta_spark.catalog.listTables(namespace):
                if not item.isTemporary:
                    assert item.name in allowed
                    assert store.ensure_owned_object(delta_spark, f"{namespace}.{item.name}")
            delta_spark.sql(f"DROP SCHEMA {table_name(namespace)} CASCADE").collect()
            assert not delta_spark.catalog.databaseExists(namespace)


def test_native_nullable_counts_preserve_zero_health_and_legacy_replays(
    delta_spark, native_store_namespace
):
    """Real Delta and health views must distinguish unknown counts without rewriting history."""
    from skyulf.integrations.databricks.observability.monitoring import monitoring_store as store

    spark = delta_spark
    namespace = store.ensure_monitoring_store(spark, *native_store_namespace)
    now = datetime(2026, 10, 1, tzinfo=UTC)
    enrolled = config()
    zero_config = MonitorConfig.from_dict(enrolled.payload() | {"model_name": "models.risk.zero"})
    for entry in (enrolled, zero_config):
        store.enroll_monitor(spark, namespace, entry)
    legacy = store.result_row(enrolled, "2", now, now, now, {"status": "failed"}, {})
    legacy.update(dict.fromkeys(COUNT_COLUMNS, 0))
    store.persist_report(spark, namespace, legacy)
    retry = store.result_row(enrolled, "2", now, now, now, {"status": "failed"}, {})
    assert retry["report_id"] == legacy["report_id"]
    store.persist_report(spark, namespace, retry)
    later = now + timedelta(days=1)
    failed = store.result_row(enrolled, "2", later, now, later, {"status": "failed"}, {})
    zero = store.result_row(
        zero_config,
        "2",
        later,
        now,
        later,
        {"status": "no_data", **dict.fromkeys(COUNT_COLUMNS, 0)},
        {},
    )
    store.persist_report(spark, namespace, failed)
    store.persist_report(spark, namespace, zero)
    rows = {
        row.report_id: row.asDict()
        for row in spark.table(f"{namespace}.monitoring_results").collect()
    }
    assert len(rows) == 3
    assert all(rows[legacy["report_id"]][key] == 0 for key in COUNT_COLUMNS)
    assert all(rows[failed["report_id"]][key] is None for key in COUNT_COLUMNS)
    assert all(rows[zero["report_id"]][key] == 0 for key in COUNT_COLUMNS)
    assert json.loads(rows[failed["report_id"]]["report_json"]) == {"status": "failed"}
    summary = spark.sql(
        f"SELECT count(current_rows) AS measured, avg(current_rows) AS average "
        f"FROM {namespace}.monitoring_results WHERE report_id <> '{legacy['report_id']}'"
    ).first()
    assert summary.measured == 1 and summary.average == 0
    health = {
        row.model_name: row.asDict() for row in spark.table(f"{namespace}.current_health").collect()
    }
    assert health[enrolled.model_name]["health_status"] == "failed"
    assert health[enrolled.model_name]["current_rows"] is None
    assert health[zero_config.model_name]["health_status"] == "no_data"
    assert health[zero_config.model_name]["current_rows"] == 0
    latest = later + timedelta(days=1)
    policy_only = store.result_row(
        enrolled,
        "2",
        latest,
        later,
        latest,
        {"status": "no_data", "performance": {"status": "unavailable"}},
        {},
    )
    store.persist_report(spark, namespace, policy_only)
    selected = (
        spark.table(f"{namespace}.current_health")
        .where(f"model_name = '{enrolled.model_name}'")
        .first()
    )
    assert selected.report_id == failed["report_id"] and selected.current_rows is None


@pytest.mark.parametrize("operation", ["ensure_monitoring_store", "initialize_monitoring_store"])
def test_native_preprovisioned_store_accepts_requests_without_ddl(
    delta_spark, native_store_namespace, operation
):
    """Both setup paths must prepare real Delta requests that runtime can use without DDL."""
    from skyulf.integrations.databricks.lifecycle.retraining_requests import _RequestStore
    from skyulf.integrations.databricks.observability.monitoring import monitoring_store as store

    catalog, schema = native_store_namespace
    namespace = getattr(store, operation)(delta_spark, catalog, schema)
    statements = []

    def runtime_sql(statement):
        """Reject setup SQL while executing metadata reads and writes on real Delta."""
        statements.append(statement)
        assert not statement.lstrip().upper().startswith(("CREATE", "ALTER", "DROP"))
        return delta_spark.sql(statement)

    runtime = SimpleNamespace(
        sql=runtime_sql,
        catalog=delta_spark.catalog,
        table=delta_spark.table,
        createDataFrame=delta_spark.createDataFrame,
    )
    assert store.ensure_monitoring_store(runtime, catalog, schema) == namespace
    requests = _RequestStore(runtime, namespace)
    row = {
        "row_id": "request:17:" + "a" * 64,
        "job_id": 17,
        "request_id": "a" * 64,
        "status": "intent",
        "evidence_json": '{"model_version":"2"}',
        "created_at_ms": 1000,
        "submitted_at_ms": None,
        "run_id": None,
    }
    requests.record(row)
    requests.record(row | {"evidence_json": '{"later":true}'})
    assert requests.read(row["row_id"]) == row
    assert sum(statement.startswith("MERGE") for statement in statements) == 2
