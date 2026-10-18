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


def _reference(spark, namespace, version="1", *, layout=1, project="retention"):
    """Write the same owned version-zero Delta populations and receipt as the producer."""
    import pandas as pd

    from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
        MonitorConfig,
        json_digest,
    )
    from skyulf.integrations.databricks.observability.monitoring.monitoring_source_evidence import (
        source_evidence,
    )
    from skyulf.integrations.databricks.observability.monitoring.reference_layout import (
        shared_source_table_name,
        source_snapshot_identity,
    )
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_reference import (
        _write_population,
        reference_table_name,
    )

    config = MonitorConfig(
        environment="test",
        project=project,
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
    rows = [(1, 2.0), (1, 2.0), (2, None), (3, 7.0)]
    source = spark.createDataFrame(rows, "id long, x double")
    frames = {
        "train": spark.createDataFrame(rows[:3], "id long, x double"),
        "seen": source,
        "source": source,
    }
    source_identity = source_snapshot_identity(
        "fixture",
        0,
        source_evidence(pd.DataFrame(rows, columns=["id", "x"]), ("id", "x"), "fixture"),
        source.schema.jsonValue(),
    )
    if layout == 2:
        frames = {
            "population": spark.createDataFrame(
                [(key, value, i < 3) for i, (key, value) in enumerate(rows)],
                "id long, x double, is_train boolean",
            ),
            "source": source,
        }
    tables = {
        key: _write_population(
            spark,
            shared_source_table_name(namespace, source_identity)
            if layout == 2 and key == "source"
            else reference_table_name(config, evidence, key),
            frame,
        )
        for key, frame in frames.items()
    }
    receipt = {
        "format": layout,
        "evidence": evidence,
        "tables": tables,
        "source_table_id": "fixture",
        "holdout": {},
    }
    if layout == 2:
        receipt.update(source_identity=source_identity, membership_column="is_train")
    name = reference_table_name(config, evidence, "receipt")
    _write_population(
        spark,
        name,
        spark.createDataFrame(
            [(json.dumps(receipt), json_digest(receipt))], "payload string, digest string"
        ),
    )
    return config, name, tables


@pytest.mark.parametrize("layout", [1, 2])
def test_native_reference_formats_roundtrip_exact_logical_populations(
    delta_spark, cleanup_store, layout
):
    """Real format-1 receipts remain readable and format-2 projections preserve duplicates and nulls."""
    from collections import Counter

    from skyulf.integrations.databricks.observability.monitoring.reference_layout import (
        reference_population_records,
    )
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_reference import (
        read_reference_metadata,
        read_reference_population,
    )

    config, receipt_name, _ = _reference(delta_spark, cleanup_store, layout=layout)
    stored = json.loads(delta_spark.table(receipt_name).first()["payload"])
    receipt = read_reference_metadata(delta_spark, config, stored["evidence"])
    records = reference_population_records(receipt)
    train = read_reference_population(delta_spark, records["train"])
    seen = read_reference_population(delta_spark, records["seen"])
    assert train.columns == seen.columns == ["id", "x"]
    assert Counter(tuple(row) for row in train.collect()) == Counter(
        [(1, 2.0), (1, 2.0), (2, None)]
    )
    assert Counter(tuple(row) for row in seen.collect()) == Counter(
        [(1, 2.0), (1, 2.0), (2, None), (3, 7.0)]
    )


def test_native_shared_source_retires_only_after_other_project(delta_spark, cleanup_store):
    """Actual Delta IDs and store-wide reverse references protect a source until its last retirement."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import (
        enroll_monitor,
    )
    from skyulf.integrations.databricks.observability.monitoring.reference_cleanup import (
        cleanup_references,
    )

    first, receipt_one, tables_one = _reference(delta_spark, cleanup_store, layout=2)
    second, receipt_two, tables_two = _reference(
        delta_spark, cleanup_store, layout=2, project="other"
    )
    assert tables_one["source"] == tables_two["source"]
    enroll_monitor(delta_spark, cleanup_store, replace(first, enabled=False))
    enroll_monitor(delta_spark, cleanup_store, replace(second, enabled=False))
    catalog, schema = cleanup_store.split(".")
    values = {
        "monitoring_catalog": catalog,
        "monitoring_schema": schema,
        "monitoring_environment": "test",
        "monitoring_project": "retention",
        "cleanup_confirm": "true",
    }
    later = datetime.now(UTC) + timedelta(days=91)
    result_one = cleanup_references(delta_spark, values, registry=lambda _: {}, now=later)
    assert result_one["deleted"] == [receipt_one]
    assert delta_spark.catalog.tableExists(tables_one["source"]["name"])
    assert delta_spark.catalog.tableExists(receipt_two)
    result_two = cleanup_references(
        delta_spark, values | {"monitoring_project": "other"}, registry=lambda _: {}, now=later
    )
    assert result_two["deleted"] == [receipt_two]
    assert not delta_spark.catalog.tableExists(tables_one["source"]["name"])


def test_native_reader_and_reuse_reject_modified_shared_source(delta_spark, cleanup_store):
    """Append-only protection permits INSERT, so latest-version validation must reject it explicitly."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_reference import (
        _write_population,
        read_reference_population,
    )

    _, _, tables = _reference(delta_spark, cleanup_store, layout=2)
    source = tables["source"]
    delta_spark.sql(f"INSERT INTO {source['name']} VALUES (4, 8.0)")
    with pytest.raises(ValueError, match="modified after creation"):
        read_reference_population(delta_spark, source)
    with pytest.raises(ValueError, match="modified after creation"):
        _write_population(
            delta_spark,
            source["name"],
            delta_spark.createDataFrame([(1, 2.0)], "id long, x double"),
        )


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
