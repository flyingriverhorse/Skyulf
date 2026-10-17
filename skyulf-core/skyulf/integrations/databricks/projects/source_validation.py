"""Read-only source checks for a resolved training and scoring configuration."""

import json
from collections.abc import Callable
from typing import Any

from ..data.training.training_dates import TrainingDateSpec, training_date_spec
from ..shared._contracts import column_name, table_name

Query = Callable[[str], list[dict[str, Any]]]


def _raw_inputs(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Separate raw input requirements from features fetched by the native lookup runtime."""
    from ..feature_store.lifecycle_config import parse_lookup_config  # noqa: PLC0415

    inputs = list(config["input_columns"])
    lookup = parse_lookup_config(
        config.get("feature_lookup"), label=config["target_column"], input_columns=tuple(inputs)
    )
    if lookup is None:
        return inputs, []
    raw = [name for name in inputs if name not in lookup.feature_names]
    raw += [name for item in lookup.lookups for name in (*item.lookup_key, *item.timestamp_columns)]
    return list(dict.fromkeys(raw)), [
        "Feature lookup tables, point-in-time joins and fetched feature types are checked by training, not this raw-source preflight."
    ]


def _training_columns(config: dict[str, Any], external: bool) -> list[str]:
    """Reuse pre-split column declarations and retain metadata needed before fitting."""
    from ..training.fitting.candidate import validate_pre_split_step  # noqa: PLC0415

    target = config["target_column"]
    metadata = [
        config[name]
        for name in ("event_column", "weight_column", "cv_group_column")
        if config.get(name)
    ]
    protected = (
        tuple(config["record_key_columns"])
        + tuple(metadata)
        + (config.get("result_available_at_column"),)
    )
    required = metadata + [
        name
        for index, step in enumerate(config.get("pre_split_steps", ()))
        for name in validate_pre_split_step(step, index, target, protected)
    ]
    if external:
        return [
            name
            for name in required
            if name not in {target, config.get("result_available_at_column")}
        ]
    return (
        required
        + [target]
        + (
            [config["result_available_at_column"]]
            if config.get("result_available_at_column")
            else []
        )
    )


def _table_schema(query: Query, name: str) -> tuple[dict[str, str], bool]:
    """Inspect Delta metadata without treating absent permissions as an empty table."""
    quoted = table_name(name)
    detail = query(f"DESCRIBE DETAIL {quoted}")
    if len(detail) != 1 or detail[0].get("format") != "delta":
        raise ValueError(f"Source must be an existing Delta table: {name}.")
    properties = detail[0].get("properties") or {}
    if isinstance(properties, str):
        properties = json.loads(properties)
    columns = {
        row["col_name"]: row["data_type"].lower()
        for row in query(f"DESCRIBE TABLE {quoted}")
        if row.get("col_name") and not row["col_name"].startswith("#") and row.get("data_type")
    }
    return columns, str(properties.get("delta.enableChangeDataFeed", "false")).lower() == "true"


def _keys(query: Query, name: str, columns: dict[str, str], keys: list[str]) -> None:
    """Use scalar distributed probes for null/nonfinite keys and duplicate identities."""
    if not keys or set(keys) - columns.keys():
        raise ValueError(f"Source is missing record key columns: {name}.")
    invalid = []
    for key in keys:
        quoted = column_name(key)
        condition = f"{quoted} IS NULL"
        if columns[key] in {"float", "double"}:
            condition += f" OR isnan({quoted}) OR abs({quoted}) = CAST('Infinity' AS DOUBLE)"
        invalid.append(f"({condition})")
    table = table_name(name)
    if query(f"SELECT 1 AS problem FROM {table} WHERE {' OR '.join(invalid)} LIMIT 1"):
        raise ValueError(f"Source has null or nonfinite record keys: {name}.")
    grouping = ", ".join(column_name(key) for key in keys)
    if query(f"SELECT 1 AS problem FROM {table} GROUP BY {grouping} HAVING count(*) > 1 LIMIT 1"):
        raise ValueError(f"Source has duplicate record keys: {name}.")


def _require_columns(schema: dict[str, str], names: list[str], table: str) -> None:
    """Fail clearly before querying an undeclared column in a key or type check."""
    for name in names:
        column_name(name)
    if missing := set(names) - schema.keys():
        raise ValueError(f"Source {table} is missing columns: {sorted(missing)}.")


def _compatible(left: dict[str, str], right: dict[str, str], names: list[str]) -> None:
    """Refuse implicit coercion between source roles, particularly record-key joins."""
    if changed := [name for name in names if left.get(name) != right.get(name)]:
        raise ValueError(f"Source column types differ: {changed}.")


def validate_project_sources(
    config: dict[str, Any],
    query: Query,
    *,
    label_table: str | None = None,
    label_available_at_column: str | None = None,
) -> dict[str, Any]:
    """Check current source tables without training, writing data or changing CDF.

    ``query`` executes SQL and returns bounded metadata/diagnostic rows. Key
    uniqueness can scan whole tables. The report describes this invocation,
    not future inserts or another identity's permissions.
    """
    training, scoring = config["training_table"], config["score_source_table"]
    outcome = config.get("training_label_table")
    names = list(dict.fromkeys(name for name in (training, scoring, outcome, label_table) if name))
    for name in names:
        table_name(name)
    keys = list(config["record_key_columns"])
    inputs, warnings = _raw_inputs(config)
    tables: dict[str, Any] = {}
    for name in names:
        schema, cdf = _table_schema(query, name)
        _keys(query, name, schema, keys)
        tables[name] = {"columns": schema, "change_data_feed": cdf}
    if not tables[scoring]["change_data_feed"]:
        raise ValueError(f"Enable Delta change data feed before incremental scoring: {scoring}.")
    train_schema, score_schema = tables[training]["columns"], tables[scoring]["columns"]
    _require_columns(train_schema, keys + inputs, training)
    _require_columns(score_schema, keys + inputs, scoring)
    _compatible(train_schema, score_schema, keys + inputs)
    warnings += _check_outcome_columns(
        config, tables, training, outcome, label_table, label_available_at_column
    )
    warnings += [
        f"Delta change data feed is off on {name}; it is required if used for incremental scoring."
        for name in names
        if not tables[name]["change_data_feed"]
    ]
    if (
        training.casefold() == scoring.casefold()
        and config.get("initial_scoring", "full_snapshot") != "from_now"
    ):
        warnings.append(
            "Initial scoring includes training history; its performance is not independent."
        )
    return {"status": "passed", "tables": tables, "warnings": warnings}


def _check_outcome_columns(
    config: dict,
    tables: dict,
    training: str,
    outcome: str | None,
    monitoring: str | None,
    available: str | None,
) -> list[str]:
    """Validate embedded and external labels against the exact configured record keys."""
    warnings = []
    keys, target = list(config["record_key_columns"]), config["target_column"]
    train_schema = tables[training]["columns"]
    _require_columns(train_schema, _training_columns(config, outcome is not None), training)
    for name, timestamp, is_training in (
        (outcome, config.get("result_available_at_column"), True),
        (monitoring, available, False),
    ):
        if name is None:
            continue
        if not timestamp:
            raise ValueError("An outcomes table requires an availability timestamp column.")
        schema = tables[name]["columns"]
        _require_columns(schema, [*keys, target, timestamp], name)
        _compatible(train_schema, schema, keys)
        if is_training:
            warnings += _training_availability(
                schema[timestamp], f"{name}.{timestamp}", config.get("result_time_parsing")
            )
        else:
            warnings += _monitoring_availability(schema[timestamp], f"{name}.{timestamp}")
    return warnings


def _training_availability(kind: str, column: str, parsing: Any) -> list[str]:
    """Match native training date admission without requiring a local Spark installation."""
    spec = training_date_spec({} if parsing is None else parsing)
    native_calendar = spec.timezone is not None and spec.format is None
    supported = {
        "timestamp": spec == TrainingDateSpec(),
        "string": spec.format is not None,
        "timestamp_ntz": native_calendar and spec.date_only == "reject",
        "date": native_calendar and spec.date_only == "midnight",
    }
    if not supported.get(kind, False):
        raise ValueError(
            f"Training outcome availability {column} ({kind}) is incompatible with "
            "result_time_parsing; configure parsing for the declared source type."
        )
    return (
        []
        if kind == "timestamp"
        else [
            f"Training availability values in {column} are checked against result_time_parsing "
            "during training; this preflight checks only the column type and parsing settings."
        ]
    )


def _monitoring_availability(kind: str, column: str) -> list[str]:
    """Accept only monitoring's native instants or explicitly offset ISO text values."""
    if kind == "timestamp":
        return []
    if kind == "string":
        return [
            f"Monitoring availability values in {column} must be ISO timestamps with an "
            "explicit UTC offset or Z; monitoring validates the values at runtime."
        ]
    raise ValueError(
        f"Monitoring outcome availability must use TIMESTAMP or offset ISO STRING: {column}."
    )


def query_source_sql(client: Any, warehouse_id: str, sql: str) -> list[dict[str, Any]]:
    """Execute one bounded preflight query; cancel it if the warehouse cannot answer in 50s."""
    from databricks.sdk.service.sql import (  # noqa: PLC0415 - optional Databricks SDK
        ExecuteStatementRequestOnWaitTimeout,
        StatementState,
    )

    result = client.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="50s",
        on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CANCEL,
        row_limit=10000,
        byte_limit=4 * 1024 * 1024,
    )
    if result.status is None or result.status.state != StatementState.SUCCEEDED:
        raise ValueError(f"Source validation query did not complete: {_query_failure(result)}.")
    return _source_query_rows(result)


def _query_failure(result: Any) -> str:
    """Keep warehouse errors actionable without assuming a complete failure envelope."""
    if result.status and result.status.error:
        return result.status.error.message or "warehouse error"
    return "query failed or timed out; start the warehouse and retry"


def _source_query_rows(result: Any) -> list[dict[str, Any]]:
    """Reject incomplete metadata instead of incorrectly declaring a source valid."""
    _require_complete_query(result)
    names = [column.name for column in result.manifest.schema.columns or []]
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("Source validation metadata has unnamed columns.")
    return [dict(zip(names, row, strict=True)) for row in result.result.data_array or []]


def _require_complete_query(result: Any) -> None:
    """Require both complete schema metadata and a single complete result chunk."""
    if result.manifest is None or result.manifest.truncated or result.manifest.schema is None:
        raise ValueError(
            "Source validation metadata is missing or exceeds the bounded result limit."
        )
    if result.result is None or result.result.next_chunk_index is not None:
        raise ValueError("Source validation requires a complete bounded result chunk.")
