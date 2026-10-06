"""Pin serving payload populations and attest platform telemetry view semantics."""

import re
from datetime import datetime
from typing import Any

from ....data.delta_io.delta import table_identity
from ....shared._contracts import table_name
from ..monitoring_config import json_digest, qualified_name
from ..monitoring_sources import read_snapshot, snapshot_at

_PROJECTION = (
    "attributes:databricks_request_id::STRING AS databricks_request_id",
    "attributes:request_date::DATE AS request_date",
    "attributes:client_request_id::STRING AS client_request_id",
    "timestamp_millis(attributes:request_time::BIGINT) AS request_time",
    "attributes:status_code::INT AS status_code",
    "attributes:sampling_fraction::DOUBLE AS sampling_fraction",
    "attributes:execution_duration_ms::BIGINT AS execution_duration_ms",
    "body:request::STRING AS request",
    "body:response::STRING AS response",
    "attributes:served_entity_id::STRING AS served_entity_id",
    "body:logging_error_codes::ARRAY<STRING> AS logging_error_codes",
    "attributes:requester::STRING AS requester",
)
_FILTER = (
    "resource.attributes:[\"databricks.product\"]::STRING = 'custom-model-serving' "
    "AND resource.attributes:[\"telemetry.sdk.name\"]::STRING = 'payload-logger'"
)
_TOKENS = re.compile(
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|[A-Za-z_][A-Za-z0-9_]*|::|\S"
)
_QUOTED_IDENTIFIER = re.compile(r"`[A-Za-z_][A-Za-z0-9_]*`\Z")


def _sql_tokens(statement: str) -> tuple[str, ...]:
    """Ignore formatting outside literals and unquote only simple identifier tokens."""
    return tuple(
        token[1:-1] if _QUOTED_IDENTIFIER.fullmatch(token) else token
        for token in _TOKENS.findall(statement)
    )


def _view_select(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Require the observed view column aliases and one explicit SELECT body."""
    starts = [
        index for index in range(len(tokens) - 1) if tokens[index : index + 2] == ("AS", "SELECT")
    ]
    if tokens[:2] != ("CREATE", "VIEW") or len(starts) != 1:
        raise ValueError("Serving view does not match the canonical telemetry definition.")
    header = tokens[: starts[0]]
    aliases = "(" + ",".join(item.rsplit(" AS ", 1)[1] for item in _PROJECTION) + ")"
    try:
        actual = header[header.index("(") : header.index(")") + 1]
    except ValueError as error:
        raise ValueError("Serving view lacks canonical telemetry column aliases.") from error
    if actual != _sql_tokens(aliases):
        raise ValueError("Serving view does not match canonical telemetry column aliases.")
    return tokens[starts[0] + 1 :]


def _view_definition(spark: Any, source_table: str, backing: str) -> str:
    """Attest the platform SELECT instead of executing arbitrary view text."""
    row = spark.sql(f"SHOW CREATE TABLE {table_name(source_table)}").first()
    definition = row["createtab_stmt"] if row is not None else None
    if not isinstance(definition, str) or not definition:
        raise ValueError("Serving view definition is unavailable.")
    tokens = _sql_tokens(definition)
    # This canonical text is token-compared only; it is never submitted to Spark.
    expected = f"SELECT {', '.join(_PROJECTION)} FROM {table_name(backing)} WHERE {_FILTER}"  # nosec B608
    if _view_select(tokens) != _sql_tokens(expected):
        raise ValueError("Serving view does not match the canonical telemetry projection/filter.")
    return json_digest(tokens)


def _source_binding(spark: Any, source_table: str) -> tuple[str, dict]:
    """Resolve only Delta payloads or attested AI Gateway telemetry views."""
    qualified_name(source_table)
    if spark.catalog.getTable(source_table).tableType != "VIEW":
        return source_table, {}
    row = spark.sql(
        f"SHOW TBLPROPERTIES {table_name(source_table)} ('otel.sourceLogsTable')"
    ).first()
    backing = row["value"] if row is not None else None
    if not isinstance(backing, str):
        raise ValueError("Serving view requires a concrete telemetry backing table.")
    qualified_name(backing)
    definition = _view_definition(spark, source_table, backing)
    return backing, {
        "payload_backing_table": backing,
        "payload_view_definition_sha256": definition,
    }


def read_serving_payload_snapshot(
    spark: Any, source_table: str, as_of: datetime
) -> tuple[Any, dict]:
    """Pin physical logs at the cutoff and apply only the admitted platform projection."""
    physical, view = _source_binding(spark, source_table)
    identity = table_identity(spark, physical)
    version = snapshot_at(spark, physical, as_of)
    frame = read_snapshot(spark, physical, version)
    if view:
        frame = frame.filter(_FILTER).selectExpr(*_PROJECTION)
    return frame, {
        "source_table": source_table,
        "source_table_id": identity,
        "prediction_table": source_table,
        "prediction_table_id": identity,
        "prediction_version": version,
        **view,
    }


def verify_serving_payload_snapshot(spark: Any, source_table: str, evidence: dict) -> None:
    """Recheck physical identity and the complete view definition after parsing."""
    if (evidence["source_table"], evidence["prediction_table"]) != (source_table, source_table):
        raise ValueError("Serving snapshot source differs from observation evidence.")
    physical, view = _source_binding(spark, source_table)
    expected = {
        key: evidence[key]
        for key in ("payload_backing_table", "payload_view_definition_sha256")
        if key in evidence
    }
    if view != expected:
        raise ValueError("Serving payload view definition changed during observation.")
    identity = table_identity(spark, physical)
    if identity != evidence["source_table_id"] or identity != evidence["prediction_table_id"]:
        raise ValueError("Serving inference table was replaced during observation.")
