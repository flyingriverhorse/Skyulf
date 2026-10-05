"""Owned Delta inventory and immutable observation history for shared dashboards."""

import json
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ...shared._contracts import table_name
from .monitoring_config import MonitorConfig, json_digest, qualified_name, store_namespace

OWNER = "skyulf.monitoring.v1"
PROPERTY = "skyulf.monitoring.owner"
INVENTORY_SCHEMA = (
    "monitor_id STRING, environment STRING, project STRING, model_name STRING, "
    "model_catalog STRING, model_schema STRING, selection STRING, "
    "expected_interval_hours DOUBLE, enabled BOOLEAN, config_digest STRING, "
    "config_json STRING, updated_at TIMESTAMP"
)
RESULT_SCHEMA = (
    "report_id STRING, monitor_id STRING, config_digest STRING, environment STRING, project STRING, "
    "model_name STRING, model_version STRING, model_catalog STRING, model_schema STRING, "
    "measured_at TIMESTAMP, observed_at TIMESTAMP, window_start TIMESTAMP, window_end TIMESTAMP, "
    "status STRING, reference_rows BIGINT, current_rows BIGINT, scored_rows BIGINT, "
    "labeled_rows BIGINT, label_coverage DOUBLE, drifted_columns BIGINT, "
    "error_message STRING, report_json STRING, evidence_json STRING, mlflow_run_id STRING"
)


def inventory_row(config: MonitorConfig, now: datetime) -> dict[str, Any]:
    """Keep an enrolled model visible even before its first successful measurement."""
    catalog, schema, _ = config.model_name.split(".")
    selection = (
        f"version:{config.model_version}" if config.model_version else f"alias:{config.model_alias}"
    )
    return {
        "monitor_id": config.monitor_id,
        "environment": config.environment,
        "project": config.project,
        "model_name": config.model_name,
        "model_catalog": catalog,
        "model_schema": schema,
        "selection": selection,
        "expected_interval_hours": float(config.expected_interval_hours),
        "enabled": config.enabled,
        "config_digest": json_digest(config.payload()),
        "config_json": _json(config.payload()),
        "updated_at": now,
    }


def result_row(
    config: MonitorConfig,
    version: str | None,
    as_of: datetime,
    window_start: datetime,
    window_end: datetime,
    report: dict,
    evidence: dict,
    *,
    observed_at: datetime | None = None,
    mlflow_run_id: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Fingerprint the exact cutoff, policy and snapshots; retain per-model metric units."""
    identity = {
        "config": config.payload(),
        "version": version,
        "as_of": as_of.isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "evidence": evidence,
    }
    catalog, schema, _ = config.model_name.split(".")
    return {
        "report_id": json_digest(identity),
        "monitor_id": config.monitor_id,
        "config_digest": json_digest(config.payload()),
        "environment": config.environment,
        "project": config.project,
        "model_name": config.model_name,
        "model_version": version,
        "model_catalog": catalog,
        "model_schema": schema,
        "measured_at": datetime.now(UTC),
        "observed_at": observed_at,
        "window_start": window_start,
        "window_end": window_end,
        "status": report["status"],
        **{
            key: int(report.get(key, 0))
            for key in (
                "reference_rows",
                "current_rows",
                "scored_rows",
                "labeled_rows",
                "drifted_columns",
            )
        },
        "label_coverage": report.get("label_coverage"),
        "error_message": error_message,
        "report_json": _json(report),
        "evidence_json": _json(evidence),
        "mlflow_run_id": mlflow_run_id,
    }


def _json(value: Any) -> str:
    """Reject NaN and preserve deterministic finite report artifacts."""
    return json.dumps(value, sort_keys=True, allow_nan=False)


def ensure_owned_object(spark: Any, name: str) -> bool:
    """Refuse an existing unrelated table or view before any replacement or write."""
    qualified_name(name)
    if not spark.catalog.tableExists(name):
        return False
    row = spark.sql(f"SHOW TBLPROPERTIES {table_name(name)} ('{PROPERTY}')").first()
    if row is None or row["value"] != OWNER:
        raise ValueError(f"Monitoring object ownership differs: {name}.")
    return True


def initialize_monitoring_store(spark: Any, catalog: str, schema: str) -> str:
    """Provision only the explicit central namespace, preserving foreign objects."""
    namespace = store_namespace(catalog, schema)
    names = (
        "model_inventory",
        "monitoring_results",
        "current_health",
        "metric_history",
        "performance_actions",
        "performance_history",
    )
    for name in names:
        ensure_owned_object(spark, f"{namespace}.{name}")
    for name, ddl in (("model_inventory", INVENTORY_SCHEMA), ("monitoring_results", RESULT_SCHEMA)):
        full_name = f"{namespace}.{name}"
        if spark.catalog.tableExists(full_name):
            expected = spark.createDataFrame([], schema=ddl).schema.simpleString()
            if spark.table(full_name).schema.simpleString() != expected:
                raise ValueError(f"Monitoring table schema differs: {full_name}.")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {table_name(namespace)}")
    for name, ddl in (("model_inventory", INVENTORY_SCHEMA), ("monitoring_results", RESULT_SCHEMA)):
        isolation = ", 'delta.isolationLevel' = 'Serializable'" if name == "model_inventory" else ""
        spark.sql(
            f"CREATE TABLE IF NOT EXISTS {table_name(namespace + '.' + name)} ({ddl}) "
            f"USING DELTA TBLPROPERTIES ('{PROPERTY}' = '{OWNER}'{isolation})"
        )
    inventory = table_name(f"{namespace}.model_inventory")
    if _inventory_isolation(spark, inventory) != "Serializable":
        spark.sql(
            f"ALTER TABLE {inventory} SET TBLPROPERTIES ('delta.isolationLevel' = 'Serializable')"
        )
    from .performance.performance_actions import initialize_performance_actions  # noqa: PLC0415

    initialize_performance_actions(spark, namespace)
    for name, query in monitoring_views(namespace).items():
        spark.sql(
            f"CREATE OR REPLACE VIEW {table_name(namespace + '.' + name)} "
            f"TBLPROPERTIES ('{PROPERTY}' = '{OWNER}') AS {query}"
        )
    return namespace


def ensure_monitoring_store(spark: Any, catalog: str, schema: str) -> str:
    """Create missing shared objects; reuse compatible stores without replacement DDL.

    First use requires schema/table/view creation privileges. Existing tables
    must already have the current schema and Serializable inventory isolation;
    explicit owner-run initialization remains responsible for upgrades.
    """
    from .performance.performance_actions import ACTION_SCHEMA  # noqa: PLC0415

    namespace = store_namespace(catalog, schema)
    tables = {
        "model_inventory": INVENTORY_SCHEMA,
        "monitoring_results": RESULT_SCHEMA,
        "performance_actions": ACTION_SCHEMA,
    }
    views = monitoring_views(namespace)
    missing = [
        name
        for name in (*tables, *views)
        if not _compatible_store_object(spark, namespace, name, tables.get(name))
    ]
    if missing:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {table_name(namespace)}").collect()
    for name in missing:
        _create_store_object(spark, namespace, name, tables.get(name), views.get(name))
        if not _compatible_store_object(spark, namespace, name, tables.get(name)):
            raise ValueError(f"Monitoring object missing after creation: {namespace}.{name}.")
    return namespace


def _compatible_store_object(spark: Any, namespace: str, name: str, ddl: str | None) -> bool:
    """Preflight all existing objects and verify the winner of a concurrent creation."""
    full_name = f"{namespace}.{name}"
    if not ensure_owned_object(spark, full_name):
        return False
    if ddl is not None:
        expected = spark.createDataFrame([], schema=ddl).schema.simpleString()
        if spark.table(full_name).schema.simpleString() != expected:
            raise ValueError(f"Monitoring table schema differs: {full_name}.")
    if (
        name == "model_inventory"
        and _inventory_isolation(spark, table_name(full_name)) != "Serializable"
    ):
        raise ValueError(
            "Initialize the central inventory with Serializable isolation before registration."
        )
    return True


def _create_store_object(
    spark: Any, namespace: str, name: str, ddl: str | None, query: str | None
) -> None:
    """Use conditional creation so concurrent projects cannot replace shared objects."""
    target = table_name(f"{namespace}.{name}")
    properties = f"'{PROPERTY}' = '{OWNER}'"
    if ddl is not None:
        if name == "model_inventory":
            properties += ", 'delta.isolationLevel' = 'Serializable'"
        statement = f"CREATE TABLE IF NOT EXISTS {target} ({ddl}) USING DELTA"
        statement += f" TBLPROPERTIES ({properties})"
    else:
        statement = f"CREATE VIEW IF NOT EXISTS {target} TBLPROPERTIES ({properties}) AS {query}"
    spark.sql(statement).collect()


def monitoring_views(namespace: str) -> dict[str, str]:
    """Build shared, filterable SQL without treating missing observations as healthy."""
    inventory = table_name(qualified_name(f"{namespace}.model_inventory"))
    results = table_name(qualified_name(f"{namespace}.monitoring_results"))
    current = f"""
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY monitor_id, config_digest
                ORDER BY window_end DESC, measured_at DESC, report_id DESC
            ) AS rank FROM {results}
            WHERE NOT (status = 'no_data'
                AND get_json_object(report_json, '$.current_rows') IS NULL
                AND get_json_object(report_json, '$.performance') IS NOT NULL)
        )
        SELECT i.monitor_id, i.environment, i.project, i.model_name,
            i.model_catalog, i.model_schema, i.selection, i.expected_interval_hours,
            COALESCE(r.model_version, get_json_object(i.config_json, '$.model_version'))
                AS model_version,
            get_json_object(i.config_json, '$.model_set_name') AS model_set_name,
            get_json_object(i.config_json, '$.model_set_version') AS model_set_version,
            r.report_id, r.measured_at, r.observed_at,
            r.current_rows, r.scored_rows, r.labeled_rows, r.label_coverage,
            r.drifted_columns, r.error_message, r.mlflow_run_id,
            CASE WHEN NOT i.enabled THEN 'disabled'
                WHEN r.report_id IS NULL THEN 'never_observed'
                WHEN r.status = 'failed' THEN 'failed'
                WHEN r.status = 'no_data' THEN 'no_data'
                WHEN r.observed_at IS NULL THEN 'unavailable'
                WHEN unix_timestamp(current_timestamp()) - unix_timestamp(r.observed_at)
                    > i.expected_interval_hours * 3600 THEN 'stale'
                ELSE r.status END AS health_status,
            r.status AS measurement_status
        FROM {inventory} i LEFT JOIN ranked r
            ON i.monitor_id = r.monitor_id AND i.config_digest = r.config_digest AND r.rank = 1
    """
    metrics = f"""
        SELECT r.report_id, r.monitor_id, r.environment, r.project, r.model_name,
            r.model_version, r.model_catalog, r.model_schema,
            r.measured_at, r.observed_at, r.window_start, r.window_end,
            r.status AS measurement_status, m.*
        FROM {results} r LATERAL VIEW explode(from_json(
            get_json_object(r.report_json, '$.metrics'),
            'ARRAY<STRUCT<category:STRING,column_name:STRING,metric_name:STRING,
                value:DOUBLE,threshold:DOUBLE,has_issue:BOOLEAN,status:STRING>>'
        )) metrics AS m
    """
    from .performance.performance_actions import performance_history_query  # noqa: PLC0415

    return {
        "current_health": current,
        "metric_history": metrics,
        "performance_history": performance_history_query(namespace),
    }


def enroll_monitor(
    spark: Any,
    namespace: str,
    config: MonitorConfig,
    *,
    activation_started_ms: int | None = None,
    preserve_activation: bool = False,
) -> None:
    """Upsert one producer's enrollment without creating or replacing central objects."""
    row = inventory_row(config, datetime.now(UTC))
    if activation_started_ms is not None:
        _validate_activation_order(activation_started_ms)
        row["config_json"] = _json(
            config.payload() | {"_activation_started_ms": activation_started_ms}
        )
    _merge_row(
        spark,
        f"{namespace}.model_inventory",
        row,
        INVENTORY_SCHEMA,
        "monitor_id",
        update=True,
        activation_started_ms=activation_started_ms,
        preserve_activation=preserve_activation,
    )


def persist_report(spark: Any, namespace: str, row: dict[str, Any]) -> None:
    """Insert exact observation evidence once; retries never overwrite history."""
    _merge_row(spark, f"{namespace}.monitoring_results", row, RESULT_SCHEMA, "report_id")


def load_enrolled_models(spark: Any, namespace: str, *, max_models: int = 10000) -> list[dict]:
    """Read one bounded inventory snapshot without rewriting concurrent producer changes."""
    name = qualified_name(f"{namespace}.model_inventory")
    if type(max_models) is not int or not 1 <= max_models <= 100000:
        raise ValueError("Inventory model limit must be between 1 and 100000.")
    if not ensure_owned_object(spark, name):
        raise ValueError("Initialize the central monitoring store before reading.")
    rows = spark.table(name).select("*").limit(max_models + 1).collect()
    if len(rows) > max_models:
        raise ValueError("Monitoring inventory exceeds the configured model limit.")
    configs = []
    seen = set()
    for row in rows:
        config = _inventory_config(row)
        if config.monitor_id in seen:
            raise ValueError("Monitoring inventory integrity check failed.")
        seen.add(config.monitor_id)
        configs.append(config.payload())
    return configs


def _inventory_config(row: Any) -> MonitorConfig:
    """Reject inconsistent controls so the runner and dashboard cannot disagree."""
    payload = json.loads(row["config_json"])
    if "_activation_started_ms" in payload:
        _validate_activation_order(payload.pop("_activation_started_ms"))
    config = MonitorConfig.from_dict(payload)
    expected = inventory_row(config, datetime.now(UTC))
    checked = set(expected) - {"updated_at", "config_json"}
    if any(row[key] != expected[key] for key in checked):
        raise ValueError("Monitoring inventory integrity check failed.")
    return config


def _validate_activation_order(value: Any) -> None:
    """Accept only the original lifecycle run's positive MLflow start timestamp."""
    if type(value) is not int or value <= 0:
        raise ValueError("Activation order must be a positive lifecycle start timestamp.")


def _inventory_update(
    row: dict, activation_started_ms: int | None, preserve_activation: bool
) -> str:
    """Protect activation ordering atomically, including delayed scoring and old task repairs."""
    marker = "get_json_object(t.config_json, '$._activation_started_ms')"
    if activation_started_ms is not None:
        order = f"CAST({marker} AS BIGINT)"
        return (
            f"WHEN MATCHED AND ({marker} IS NULL OR {order} < {activation_started_ms} "
            f"OR ({order} = {activation_started_ms} AND t.selection = s.selection)) "
            "THEN UPDATE SET *"
        )
    if not preserve_activation:
        return "WHEN MATCHED THEN UPDATE SET *"
    assignments = [f"t.{key} = s.{key}" for key in row if key != "config_json"]
    assignments.append(
        f"t.config_json = CASE WHEN {marker} IS NULL THEN s.config_json "
        "ELSE concat('{\"_activation_started_ms\":', "
        f"{marker}, ',', substring(s.config_json, 2)) END"
    )
    parent_match = " AND ".join(
        f"get_json_object(t.config_json, '$.{field}') "
        f"<=> get_json_object(s.config_json, '$.{field}')"
        for field in ("model_set_name", "model_set_version", "model_set_branch")
    )
    return (
        f"WHEN MATCHED AND ({marker} IS NULL OR ({parent_match})) "
        f"AND ({marker} IS NULL OR t.selection = s.selection) THEN UPDATE SET "
        + ", ".join(assignments)
    )


def _inventory_isolation(spark: Any, quoted_table: str) -> str | None:
    """Read the concurrency contract installed only by the central initializer."""
    row = spark.sql(f"SHOW TBLPROPERTIES {quoted_table} ('delta.isolationLevel')").first()
    return row["value"] if row is not None else None


def merge_with_retry(spark: Any, statement: str) -> None:
    """Retry only optimistic Delta write conflicts; preserve permissions and data errors."""
    for attempt in range(3):
        try:
            spark.sql(statement).collect()
            return
        except Exception as exc:  # noqa: BLE001 - classic and Connect wrap Delta conflicts differently
            concurrent = "Concurrent" in type(exc).__name__ or "DELTA_CONCURRENT" in str(exc)
            if not concurrent or attempt == 2:
                raise
            time.sleep(0.2 * (attempt + 1))


def _merge_row(
    spark: Any,
    name: str,
    row: dict,
    schema: str,
    key: str,
    *,
    update: bool = False,
    activation_started_ms: int | None = None,
    preserve_activation: bool = False,
) -> None:
    """Use typed data rather than interpolating caller values into SQL text."""
    if not ensure_owned_object(spark, name):
        raise ValueError("Initialize the central monitoring store before writing.")
    if update and _inventory_isolation(spark, table_name(name)) != "Serializable":
        raise ValueError(
            "Initialize the central inventory with Serializable isolation before registration."
        )
    view = f"skyulf_monitor_{uuid4().hex}"
    spark.createDataFrame([row], schema=schema).createOrReplaceTempView(view)
    matched = _inventory_update(row, activation_started_ms, preserve_activation) if update else ""
    try:
        merge_with_retry(
            spark,
            f"MERGE INTO {table_name(name)} t USING {view} s ON t.{key} = s.{key} "
            f"{matched} WHEN NOT MATCHED THEN INSERT *",
        )
    finally:
        spark.catalog.dropTempView(view)
