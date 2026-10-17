"""Exercise reference retention against actual Delta identities and deletion audit rows."""

import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest


@pytest.fixture
def cleanup_store(delta_spark):
    """Create and remove only a unique store owned by this test invocation."""
    from skyulf.integrations.databricks.lifecycle.retraining_requests import REQUEST_SCHEMA
    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import (
        INVENTORY_SCHEMA,
        OWNER,
        PROPERTY,
    )

    catalog = os.environ.get("SKYULF_TEST_CATALOG", "spark_catalog")
    schema = f"retention_test_{uuid4().hex}"
    namespace = f"{catalog}.{schema}"
    delta_spark.sql(f"CREATE SCHEMA {namespace}")
    try:
        for name, ddl in (
            ("model_inventory", INVENTORY_SCHEMA),
            ("retraining_requests", REQUEST_SCHEMA),
        ):
            delta_spark.sql(
                f"CREATE TABLE {namespace}.{name} ({ddl}) USING DELTA TBLPROPERTIES ('{PROPERTY}'='{OWNER}', 'delta.isolationLevel'='Serializable')"
            )
        delta_spark.sql(f"CREATE TABLE {namespace}.unrelated (id BIGINT) USING DELTA")
        yield namespace
    finally:
        assert schema.startswith("retention_test_") and len(schema) == len("retention_test_") + 32
        # This exact schema was created by this fixture, never selected from existing schemas.
        delta_spark.sql(f"DROP SCHEMA {namespace} CASCADE")


def _reference(spark, namespace, version="1"):
    """Write the same owned version-zero Delta populations and receipt as the producer."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
        MonitorConfig,
        json_digest,
    )
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_reference import (
        _reference_name,
        _write_population,
    )

    config = MonitorConfig(
        environment="test",
        project="retention",
        model_name=f"{namespace}.model",
        model_version=version,
        source_table=f"{namespace}.source",
        prediction_table=f"{namespace}.predictions",
        execution_engine="spark",
        reference_namespace=namespace,
    )
    evidence = {
        "model_name": config.model_name,
        "model_version": version,
        "model_digest": "a" * 64,
        "training_run_id": "b" * 32,
        "reference_table": config.source_table,
        "reference_version": 0,
        "dataset_id": "fixture",
        "training_evidence_sha256": "c" * 64,
    }
    tables = {
        key: _write_population(
            spark,
            _reference_name(config, evidence, key),
            spark.createDataFrame([(1, 2.0)], "id long, x double"),
        )
        for key in ("train", "seen", "source")
    }
    receipt = {
        "format": 1,
        "evidence": evidence,
        "tables": tables,
        "source_table_id": "fixture",
        "holdout": {},
    }
    name = _reference_name(config, evidence, "receipt")
    _write_population(
        spark,
        name,
        spark.createDataFrame(
            [(json.dumps(receipt), json_digest(receipt))], "payload string, digest string"
        ),
    )
    return config, name, tables


def test_native_reference_retention_dry_run_protections_and_confirmed_delete(
    delta_spark, cleanup_store
):
    """Real cleanup must preserve dependencies and write durable audit evidence before deleting."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import (
        enroll_monitor,
    )
    from skyulf.integrations.databricks.observability.monitoring.reference_cleanup import (
        cleanup_references,
    )

    namespace = cleanup_store
    catalog, schema = namespace.split(".")
    config, receipt, tables = _reference(delta_spark, namespace)
    enroll_monitor(delta_spark, namespace, config)
    values = {
        "monitoring_catalog": catalog,
        "monitoring_schema": schema,
        "monitoring_environment": "test",
        "monitoring_project": "retention",
    }
    later = datetime.now(UTC) + timedelta(days=91)
    active = cleanup_references(delta_spark, values, registry=lambda _: {}, now=later)
    assert active["references"][0]["reason"] == "enrolled model version"
    enroll_monitor(delta_spark, namespace, replace(config, enabled=False))
    alias = cleanup_references(delta_spark, values, registry=lambda _: {"champion": "1"}, now=later)
    assert "alias" in alias["references"][0]["reason"]
    young = cleanup_references(delta_spark, values, registry=lambda _: {})
    assert "younger" in young["references"][0]["reason"]
    dry = cleanup_references(delta_spark, values, registry=lambda _: {}, now=later)
    assert dry["status"] == "dry_run" and dry["deleted"] == []
    assert dry["references"][0]["decision"] == "deletable"
    assert not delta_spark.catalog.tableExists(f"{namespace}.reference_cleanup_audit")
    result = cleanup_references(
        delta_spark, values | {"cleanup_confirm": "true"}, registry=lambda _: {}, now=later
    )
    assert result["deleted"] == [receipt]
    assert all(
        not delta_spark.catalog.tableExists(name)
        for name in [receipt, *[row["name"] for row in tables.values()]]
    )
    actions = {
        row["status"] for row in delta_spark.table(f"{namespace}.reference_cleanup_audit").collect()
    }
    assert actions == {"intent", "deleted"}
    assert delta_spark.catalog.tableExists(f"{namespace}.unrelated")
    assert delta_spark.catalog.tableExists(f"{namespace}.model_inventory")


def test_native_cleanup_rejects_population_modified_after_receipt(delta_spark, cleanup_store):
    """An owned table with the same Delta ID is still unsafe when its version has advanced."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import (
        enroll_monitor,
    )
    from skyulf.integrations.databricks.observability.monitoring.reference_cleanup import (
        cleanup_references,
    )

    namespace = cleanup_store
    config, receipt, tables = _reference(delta_spark, namespace)
    enroll_monitor(delta_spark, namespace, replace(config, enabled=False))
    delta_spark.sql(f"INSERT INTO {tables['train']['name']} VALUES (2, 3.0)")
    catalog, schema = namespace.split(".")
    result = cleanup_references(
        delta_spark,
        {
            "monitoring_catalog": catalog,
            "monitoring_schema": schema,
            "monitoring_environment": "test",
            "monitoring_project": "retention",
            "cleanup_confirm": "true",
        },
        registry=lambda _: {},
        now=datetime.now(UTC) + timedelta(days=91),
    )
    assert result["deleted"] == []
    assert result["skipped"]
    assert all(
        delta_spark.catalog.tableExists(name)
        for name in [receipt, *[row["name"] for row in tables.values()]]
    )
