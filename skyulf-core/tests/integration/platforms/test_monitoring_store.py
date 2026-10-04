"""Shared monitoring persistence binds snapshots and preserves unobserved models."""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks.monitoring_config import MonitorConfig


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
    from skyulf.integrations.databricks.monitoring_store import result_row

    now = datetime(2026, 10, 1, tzinfo=UTC)
    report = {"status": "healthy", "current_rows": 4, "metrics": [], "notes": []}
    evidence = {"prediction_version": 2, "label_version": 3}
    first = result_row(config(), "2", now, now, now, report, evidence)
    retry = result_row(config(), "2", now, now, now, report, evidence)
    changed = result_row(config(), "2", now, now, now, report, evidence | {"label_version": 4})
    assert first["report_id"] == retry["report_id"]
    assert first["report_id"] != changed["report_id"]
    assert first["model_name"] == "models.risk.score"


def test_inventory_keeps_namespace_and_selection():
    """The dashboard must distinguish same-named models in different catalogs."""
    from skyulf.integrations.databricks.monitoring_store import inventory_row

    row = inventory_row(config(), datetime(2026, 10, 1, tzinfo=UTC))
    assert row["model_catalog"] == "models"
    assert row["model_schema"] == "risk"
    assert row["selection"] == "version:2"


def test_persistence_rejects_nonfinite_report_values():
    """Invalid JSON cannot become a misleading dashboard metric."""
    from skyulf.integrations.databricks.monitoring_store import result_row

    now = datetime(2026, 10, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        result_row(config(), "2", now, now, now, {"status": "healthy", "x": float("nan")}, {})


def test_existing_foreign_table_is_not_adopted():
    """Central setup cannot overwrite a table just because its name matches."""
    from unittest.mock import Mock

    from skyulf.integrations.databricks.monitoring_store import ensure_owned_object

    spark = Mock()
    spark.catalog.tableExists.return_value = True
    spark.sql.return_value.first.return_value = {"value": "foreign"}
    with pytest.raises(ValueError, match="ownership"):
        ensure_owned_object(spark, "ops.monitoring.model_inventory")
    assert spark.sql.call_count == 1


def test_owned_incompatible_schema_requires_explicit_migration():
    """A stale central table must fail before views are replaced or reports are written."""
    from unittest.mock import Mock

    from skyulf.integrations.databricks.monitoring_store import OWNER, initialize_monitoring_store

    spark = Mock()
    spark.catalog.tableExists.return_value = True
    spark.sql.return_value.first.return_value = {"value": OWNER}
    spark.table.return_value.schema.simpleString.return_value = "struct<old:string>"
    spark.createDataFrame.return_value.schema.simpleString.return_value = "struct<new:string>"
    with pytest.raises(ValueError, match="schema differs"):
        initialize_monitoring_store(spark, "ops", "monitoring")
    assert all(call.args[0].startswith("SHOW TBLPROPERTIES") for call in spark.sql.call_args_list)


def test_inventory_reads_other_repositories_without_file_input(monkeypatch):
    """The central runner discovers all registered projects from one bounded Delta snapshot."""
    from skyulf.integrations.databricks import monitoring_store as store

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
    from skyulf.integrations.databricks import monitoring_store as store

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
