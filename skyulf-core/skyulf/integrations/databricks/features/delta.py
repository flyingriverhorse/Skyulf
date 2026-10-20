"""Persist keyed feature corrections without erasing earlier observation history."""

import json
import re
import uuid
from typing import Any

from ..shared._contracts import column_name, table_name
from .joins import validate_feature_frame

_PUBLICATION_PROPERTY = "skyulf.features.publication"


def _validate_intent(intent: Any) -> dict[str, Any]:
    """Keep persisted publication metadata small, exact and safe for SQL literals."""
    if type(intent) is not dict or set(intent) != {"generation", "snapshot_sha256"}:
        raise ValueError("Invalid feature publication metadata.")
    generation, digest = intent["generation"], intent["snapshot_sha256"]
    if type(generation) is not int or not 0 < generation < 2**63:
        raise ValueError("Invalid feature publication generation.")
    if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Invalid feature publication snapshot digest.")
    return intent


def validate_output_head(head: Any) -> None:
    """Require either a pinned Delta identity/version or an explicitly absent table."""
    if type(head) is not dict or set(head) != {"table_id", "version"}:
        raise ValueError("Invalid feature publication output head.")
    identity, version = head["table_id"], head["version"]
    if identity is None and version is None:
        return
    if identity is None:
        raise ValueError("Invalid feature publication table identity.")
    if type(identity) is not str:
        raise ValueError("Invalid feature publication table identity.")
    if not 0 < len(identity) <= 256:
        raise ValueError("Invalid feature publication table identity.")
    if type(version) is not int or version < 0:
        raise ValueError("Invalid feature publication output version.")


def publication_state(spark: Any, table: str) -> dict[str, Any] | None:
    """Read durable output identity, head and publication intent without feature rows."""
    quoted = table_name(table)
    if not spark.catalog.tableExists(table):
        return None
    detail = spark.sql(f"DESCRIBE DETAIL {quoted}").first()
    if detail is None or detail["format"] != "delta":
        raise ValueError(f"Feature output must be a Delta table: {table}.")
    head = {"table_id": detail["id"], "version": current_version(spark, table)}
    validate_output_head(head)
    properties = detail["properties"]
    intent = None
    if _PUBLICATION_PROPERTY in properties:
        raw = properties[_PUBLICATION_PROPERTY]
        if type(raw) is not str or len(raw) > 256:
            raise ValueError("Invalid feature publication metadata.")
        intent = _validate_intent(json.loads(raw))
    return {**head, "intent": intent}


def require_publication_head(
    state: dict[str, Any] | None,
    expected: dict[str, Any],
    intent: dict[str, Any],
    table: str,
) -> None:
    """Reject replaced tables and any later or competing publication generation."""
    validate_output_head(expected)
    _validate_intent(intent)
    if expected["table_id"] is not None and (
        state is None or state["table_id"] != expected["table_id"]
    ):
        raise ValueError(f"Feature output identity changed; start a new run: {table}.")
    previous = state["intent"] if state else None
    if previous and previous != intent and previous["generation"] >= intent["generation"]:
        raise ValueError(f"Feature snapshot was superseded; start a new run: {table}.")


def _publication_request(publication: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the complete caller fence before writing table metadata."""
    if type(publication) is not dict or set(publication) != {
        "generation",
        "snapshot_sha256",
        "expected",
    }:
        raise ValueError("Feature publication requires a fresh initialized snapshot.")
    intent = _validate_intent({key: publication[key] for key in ("generation", "snapshot_sha256")})
    expected = publication["expected"]
    validate_output_head(expected)
    return intent, expected


def _claim_publication(
    spark: Any, table: str, state: dict[str, Any] | None, publication: dict[str, Any]
) -> dict[str, Any]:
    """Persist an intent before MERGE; the same deterministic snapshot may retry it."""
    intent, expected = _publication_request(publication)
    require_publication_head(state, expected, intent, table)
    if state is not None and state["intent"] == intent:
        return intent
    actual = {key: state[key] if state else None for key in ("table_id", "version")}
    if actual != expected:
        raise ValueError(f"Feature output head changed; start a new run: {table}.")
    if state is not None:
        encoded = json.dumps(intent, sort_keys=True, separators=(",", ":"))
        spark.sql(
            f"ALTER TABLE {table_name(table)} SET TBLPROPERTIES "  # nosec B608
            f"('{_PUBLICATION_PROPERTY}' = '{encoded}')"
        )
        _verify_intent(spark, table, intent, expected)
    return intent


def _verify_intent(
    spark: Any, table: str, intent: dict[str, Any], expected: dict[str, Any]
) -> None:
    """A failed or ambiguous metadata write never permits an unfenced MERGE."""
    recorded = publication_state(spark, table)
    require_publication_head(recorded, expected, intent, table)
    if recorded is None or recorded["intent"] != intent:
        raise ValueError(f"Feature publication intent was not recorded: {table}.")


def _create_with_intent(spark: Any, frame: Any, table: str, intent: dict[str, Any]) -> None:
    """Commit the first rows and their stale-repair fence in one Delta CTAS."""
    view = "skyulf_feature_" + uuid.uuid4().hex
    frame.createOrReplaceTempView(view)
    encoded = json.dumps(intent, sort_keys=True, separators=(",", ":"))
    try:
        spark.sql(
            f"CREATE TABLE {table_name(table)} USING DELTA TBLPROPERTIES "  # nosec B608
            f"('delta.enableChangeDataFeed' = 'true', '{_PUBLICATION_PROPERTY}' = '{encoded}') "
            f"AS SELECT * FROM `{view}`"
        )
    finally:
        spark.catalog.dropTempView(view)


def current_version(spark: Any, table: str) -> int:
    """Read only the latest Delta commit number, never feature records."""
    row = spark.sql(f"DESCRIBE HISTORY {table_name(table)} LIMIT 1").select("version").first()
    if row is None:
        raise ValueError(f"Feature table has no Delta history: {table}.")
    return int(row[0])


def read_version(spark: Any, table: str, version: int) -> Any:
    """Require a concrete Delta version for every cross-task data dependency."""
    table_name(table)
    if type(version) is not int or version < 0:
        raise ValueError("Feature source version must be a nonnegative integer.")
    return spark.read.format("delta").option("versionAsOf", version).table(table)


def _merge_rows(spark: Any, frame: Any, table: str, grain: tuple[str, ...]) -> None:
    """Update changed values only; retries preserve unaffected rows and history."""
    view = "skyulf_feature_" + uuid.uuid4().hex
    frame.createOrReplaceTempView(view)
    names = [column_name(name) for name in frame.columns]
    condition = " AND ".join(f"target.{column_name(k)} = source.{column_name(k)}" for k in grain)
    changes = " OR ".join(f"NOT(target.{name} <=> source.{name})" for name in names)
    try:
        # Every interpolated name is allowlisted; the view name is a generated UUID.
        spark.sql(
            f"MERGE INTO {table_name(table)} AS target USING `{view}` AS source "  # nosec B608
            f"ON {condition} WHEN MATCHED AND ({changes}) THEN UPDATE SET * "
            "WHEN NOT MATCHED THEN INSERT *"
        )
    finally:
        spark.catalog.dropTempView(view)


def _validate_existing(
    spark: Any, frame: Any, table: str, keys: tuple[str, ...], timestamp: str
) -> None:
    """Require the existing grain, schema and CDF policy before claiming publication."""
    before = spark.table(table)
    validate_feature_frame(before, keys, timestamp)
    expected = {f.name: f.dataType for f in before.schema.fields}
    if {f.name: f.dataType for f in frame.schema.fields} != expected:
        raise ValueError(f"Feature table schema differs; use an explicit migration: {table}.")
    properties = spark.sql(f"DESCRIBE DETAIL {table_name(table)}").first()["properties"]
    if properties.get("delta.enableChangeDataFeed") != "true":
        raise ValueError(f"Enable Delta change data feed explicitly before using {table}.")


def publish_feature_table(
    spark: Any,
    frame: Any,
    table: str,
    keys: tuple[str, ...],
    timestamp: str,
    *,
    publication: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a CDF-enabled Delta table or upsert a checked compatible feature batch.

    A feature job serializes its own runs. Tables must have one owning producer;
    this API does not implement a distributed lock against unrelated writers.
    Output schemas are explicit and cannot evolve implicitly during a repair.
    Managed outputs require the initialized generation and pinned output head.
    The intent is durable before MERGE, including no-op batches; identical
    snapshots may replay deterministic transformations after a lost receipt.
    """
    table_name(table)
    rows = validate_feature_frame(frame, keys, timestamp)
    for name in frame.columns:
        column_name(name)
    state = publication_state(spark, table)
    if publication is None and state is not None and state["intent"] is not None:
        raise ValueError(f"Managed feature output requires a publication snapshot: {table}.")
    if state is None:
        if publication is None:
            frame.write.format("delta").mode("error").option(
                "delta.enableChangeDataFeed", "true"
            ).saveAsTable(table)
        else:
            intent = _claim_publication(spark, table, state, publication)
            _create_with_intent(spark, frame, table, intent)
            _verify_intent(spark, table, intent, {"table_id": None, "version": None})
    else:
        _validate_existing(spark, frame, table, keys, timestamp)
        if publication is not None:
            _claim_publication(spark, table, state, publication)
        _merge_rows(spark, frame, table, (*keys, timestamp))
    return {"table": table, "version": current_version(spark, table), "rows": rows}
