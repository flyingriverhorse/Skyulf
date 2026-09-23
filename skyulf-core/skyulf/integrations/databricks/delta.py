"""Guarded Delta replacement with committed receipts and explicit retries."""

import importlib
import json
from typing import Any

from ._contracts import BatchSpec, table_name
from .admission import BatchConflictError, PublishAdmission, validate_admission

__all__ = ["BatchConflictError", "DeltaPublishError", "publish_replace_period"]


class DeltaPublishError(RuntimeError):
    """Publication failed or its committed receipt could not be verified."""


def table_identity(spark: Any, table: str) -> str:
    """Read one immutable Delta table identity; targets must be provisioned explicitly."""
    detail = spark.sql(f"DESCRIBE DETAIL {table_name(table)}").first()
    if detail is None or detail["format"] != "delta":
        raise ValueError("Batch source and target must be existing Delta tables.")
    return str(detail["id"])


def history(spark: Any, table: str) -> Any:
    """Expose distributed commit history without collecting it on the driver."""
    return spark.sql(f"DESCRIBE HISTORY {table_name(table)}")


def _receipt(spark: Any, spec: BatchSpec, manifest: dict[str, Any]) -> tuple[int, dict] | None:
    """Find a bounded receipt by logical run and reject reuse for different input."""
    functions = importlib.import_module("pyspark.sql.functions")
    matches = (
        history(spark, spec.output_table)
        .where(functions.get_json_object("userMetadata", "$.run_id") == spec.run_id)
        .select("version", "userMetadata")
        .limit(2)
        .collect()
    )
    if not matches:
        return None
    if len(matches) != 1:
        raise BatchConflictError("Multiple commits use the same logical run_id.")
    recorded = json.loads(matches[0]["userMetadata"])
    if recorded.get("request_digest") != manifest["request_digest"]:
        raise BatchConflictError("run_id was already used for a different batch request.")
    return int(matches[0]["version"]), recorded


def publish_replace_period(
    spark: Any,
    frame: Any,
    spec: BatchSpec,
    *,
    manifest: dict[str, Any],
    admission: PublishAdmission,
) -> tuple[int, dict[str, Any], bool]:
    """Replace a month and verify its receipt while holding whole-table admission.

    ``expected_target_version`` prevents stale publishers, including writers that
    waited for admission. A recorded retry returns its original receipt without
    writing, even when a later run has recomputed the same period. History and
    transaction retention must cover the supported retry window. Writers outside
    this admission protocol are outside the single-writer guarantee.
    """
    admission = validate_admission(spark, admission)
    target_id = table_identity(spark, spec.output_table)
    with admission.hold(target_id):
        if target_id != table_identity(spark, spec.output_table):
            raise BatchConflictError("Target table was replaced during admission.")
        recorded = _receipt(spark, spec, manifest)
        if recorded is not None:
            return recorded[0], recorded[1], True
        latest = history(spark, spec.output_table).select("version").first()
        if latest is None or int(latest["version"]) != spec.expected_target_version:
            raise BatchConflictError("Target version changed; explicitly plan a new logical run.")
        _validate_frame(frame, spec, manifest)
        # A timezone offset in the literal avoids dependence on the session timezone.
        start, end = spec.period_start_utc.isoformat(), spec.period_end_utc.isoformat()
        predicate = (
            f"`{spec.period_column}` >= TIMESTAMP '{start}' AND "
            f"`{spec.period_column}` < TIMESTAMP '{end}'"
        )
        try:
            (
                frame.write.format("delta")
                .mode("overwrite")
                .option("replaceWhere", predicate)
                .option("mergeSchema", "false")
                .option("txnAppId", "skyulf:" + spec.run_id)
                .option("txnVersion", 0)
                .option("userMetadata", json.dumps(manifest, sort_keys=True))
                .saveAsTable(spec.output_table)
            )
        except Exception as exc:  # noqa: BLE001 - retain Spark/Delta details as the cause
            error_class = type(exc).__name__
            if "Concurrent" in error_class or "DELTA_CONCURRENT" in str(exc):
                raise BatchConflictError("Delta rejected a concurrent write.") from exc
            raise DeltaPublishError(
                "Delta write failed; inspect the chained runtime error."
            ) from exc
        recorded = _receipt(spark, spec, manifest)
        if recorded is None:
            raise DeltaPublishError("Delta returned without a verifiable commit receipt.")
        return recorded[0], recorded[1], False


def _validate_frame(frame: Any, spec: BatchSpec, manifest: dict[str, Any]) -> None:
    """Reject null or out-of-period rows, wrong ownership and misleading row counts."""
    functions = importlib.import_module("pyspark.sql.functions")
    period = frame[spec.period_column]
    if frame.schema[spec.period_column].dataType.typeName() != "timestamp":
        raise ValueError("period_column must be a Spark timestamp with timezone semantics.")
    invalid = (
        period.isNull()
        | (period < functions.lit(spec.period_start_utc))
        | (period >= functions.lit(spec.period_end_utc))
    )
    for name, expected in (
        ("run_id", spec.run_id),
        ("model_name", spec.model_name),
        ("model_version", spec.model_version),
    ):
        invalid = invalid | frame[name].isNull() | (frame[name] != functions.lit(expected))
    if frame.where(invalid).limit(1).count():
        raise ValueError("Output contains null/outside-period rows or invalid run metadata.")
    actual = frame.count()
    if actual != manifest["output_count"]:
        raise ValueError("Output count does not match the batch manifest.")
    if actual == 0 and not spec.allow_empty:
        raise ValueError("Empty period replacement requires allow_empty=True.")
