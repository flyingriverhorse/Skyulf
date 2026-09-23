"""Score only new Delta inserts with a bounded local model and atomic receipt."""

from __future__ import annotations

import hashlib
import importlib
import json
import pickle
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import pandas as pd

from ...inference.local_pipeline import LocalPipelineArtifact
from ._contracts import PREDICTION_METADATA_COLUMNS, column_name, table_name
from .admission import BatchConflictError, PublishAdmission, validate_admission
from .delta import DeltaPublishError, history, table_identity
from .local_batch import _frame_bytes
from .local_publish import _check_target, _scalar
from .local_sdk import PreparedLocalWorkflow


@dataclass(frozen=True, slots=True)
class IncrementalBatchResult:
    """Describe one committed increment or a run with no new source inserts."""

    source_start_version: int | None
    source_end_version: int
    input_count: int
    output_count: int
    commit_version: int | None
    manifest: dict[str, Any] | None
    noop: bool


def _latest(spark: Any, table: str) -> Any:
    """Read the most recent Delta commit deterministically."""
    functions = importlib.import_module("pyspark.sql.functions")
    row = (
        history(spark, table)
        .orderBy(functions.desc("version"))
        .select("version", "userMetadata")
        .first()
    )
    if row is None:
        raise ValueError("Delta table has no readable history.")
    return row


def _last_receipt(latest: Any, source_id: str, target_id: str) -> dict[str, Any] | None:
    """Trust a watermark only when the latest target commit contains it."""
    raw = latest["userMetadata"]
    if not raw:
        return None
    try:
        receipt = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if receipt.get("skyulf_mode") != "incremental_append":
        return None
    if receipt.get("source_table_id") != source_id or receipt.get("target_table_id") != target_id:
        raise BatchConflictError("Incremental target is bound to another source or target ID.")
    if receipt.get("source_end_version") is None:
        raise BatchConflictError("Incremental target receipt has no source watermark.")
    return receipt


def _validate_prepared(
    prepared: PreparedLocalWorkflow, row_keys: tuple[str, ...], period_column: str | None
) -> tuple[str, ...]:
    """Require a ready UC model and an automatic source-selection contract."""
    if not isinstance(prepared, PreparedLocalWorkflow):
        raise TypeError("prepared must be a PreparedLocalWorkflow.")
    config = prepared.config
    if (
        config.runtime != "databricks"
        or config.source.kind != "uc_table"
        or config.source.read_mode != "incremental"
        or config.source.version is not None
        or config.sink.kind != "uc_delta"
        or not config.source.table
        or not config.sink.table
        or config.source.table == config.sink.table
    ):
        raise ValueError("Incremental scoring requires unpinned UC source and UC Delta sink.")
    if (
        config.model.kind != "local_pipeline"
        or not config.model.name
        or not config.model.version
        or not prepared.preflight.ready
        or not isinstance(prepared.artifact, LocalPipelineArtifact)
        or prepared.preflight.model_version != config.model.version
        or prepared.preflight.model_digest != prepared.artifact.manifest.pipeline_sha256
    ):
        raise ValueError("Incremental scoring requires a pinned, ready local pipeline model.")
    if type(row_keys) is not tuple or not row_keys:
        raise ValueError("row_keys must be a nonempty tuple.")
    for name in row_keys:
        column_name(name)
    if period_column is not None:
        column_name(period_column)
    names = (*row_keys, *((period_column,) if period_column is not None else ()))
    if len({name.lower() for name in names}) != len(names):
        raise ValueError("row_keys and period_column must be distinct.")
    if any(name.lower() in PREDICTION_METADATA_COLUMNS for name in names):
        raise ValueError("row_keys and period_column collide with prediction metadata.")
    inputs = prepared.artifact.manifest.input_columns
    if any(key in inputs for key in row_keys) or (
        period_column is not None and period_column in inputs
    ):
        raise ValueError("Model inputs must be distinct from keys and event time.")
    if any(name in {"_change_type", "_commit_version", "_commit_timestamp"} for name in inputs):
        raise ValueError("Model inputs collide with Delta change metadata.")
    return inputs


def _bounded_frame(
    selected: Any,
    columns: tuple[str, ...],
    row_keys: tuple[str, ...],
    max_rows: int,
    max_bytes: int,
) -> pd.DataFrame:
    """Move only a bounded, projected set of source rows to the local model."""
    records: list[dict[str, Any]] = []
    decoded_bytes = 0
    for row in selected.select(*columns).orderBy(*row_keys).limit(max_rows + 1).toLocalIterator():
        if len(records) == max_rows:
            raise ValueError("Source increment exceeds max_rows.")
        record = row.asDict(recursive=True)
        decoded_bytes += len(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
        if decoded_bytes > max_bytes:
            raise ValueError("Source increment exceeds max_bytes.")
        records.append(record)
    frame = pd.DataFrame.from_records(records, columns=columns)
    if _frame_bytes(frame) > max_bytes:
        raise ValueError("Source local frame exceeds max_bytes.")
    if frame.loc[:, list(row_keys)].isna().any().any():
        raise ValueError("Source row keys must not be null.")
    if frame.duplicated(subset=list(row_keys)).any():
        raise ValueError("Source row keys must be globally unique within this increment.")
    return frame


def run_incremental_local_batch(
    spark: Any,
    prepared: PreparedLocalWorkflow,
    *,
    row_keys: tuple[str, ...],
    admission: PublishAdmission | None,
    period_column: str | None = None,
) -> IncrementalBatchResult:
    """Score the initial snapshot, then only inserts since the last target receipt.

    Source and target tables already exist. The source must have Delta CDF
    enabled before subsequent changes. All target writers must share admission;
    a target commit outside this protocol halts automatic scoring.
    """
    inputs = _validate_prepared(prepared, row_keys, period_column)
    admission = validate_admission(spark, admission)
    config = prepared.config
    source_table = config.source.table
    target_table = config.sink.table
    assert source_table is not None and target_table is not None
    source_id = table_identity(spark, source_table)
    target_id = table_identity(spark, target_table)
    if source_id == target_id:
        raise ValueError("Source and prediction target must be different Delta tables.")
    detail = spark.sql(f"DESCRIBE DETAIL {table_name(source_table)}").first()
    properties = detail["properties"] or {}
    if not any(
        key.lower() == "delta.enablechangedatafeed" and str(value).lower() == "true"
        for key, value in properties.items()
    ):
        raise ValueError("Source Delta Change Data Feed must be enabled before scoring.")
    functions = importlib.import_module("pyspark.sql.functions")

    with admission.hold(target_id):
        if (
            table_identity(spark, source_table) != source_id
            or table_identity(spark, target_table) != target_id
        ):
            raise BatchConflictError("Source or target table identity changed during admission.")
        target_latest = _latest(spark, target_table)
        previous = _last_receipt(target_latest, source_id, target_id)
        if previous is None:
            older_receipt = (
                history(spark, target_table)
                .where(
                    functions.get_json_object("userMetadata", "$.skyulf_mode")
                    == "incremental_append"
                )
                .limit(1)
                .count()
            )
            if older_receipt:
                raise BatchConflictError("Target changed outside the incremental receipt protocol.")
            if spark.table(target_table).limit(1).count():
                raise BatchConflictError(
                    "Incremental bootstrap requires an empty target with no prior predictions."
                )
        prior_version = int(previous["source_end_version"]) if previous else None
        upper_version = int(_latest(spark, source_table)["version"])
        if prior_version is not None and upper_version < prior_version:
            raise BatchConflictError("Source version moved behind the committed watermark.")
        if prior_version == upper_version:
            return IncrementalBatchResult(
                prior_version, upper_version, 0, 0, int(target_latest["version"]), previous, True
            )
        if prior_version is None:
            selected = (
                spark.read.format("delta").option("versionAsOf", upper_version).table(source_table)
            )
        else:
            selected = (
                spark.read.format("delta")
                .option("readChangeFeed", "true")
                .option("startingVersion", prior_version + 1)
                .option("endingVersion", upper_version)
                .table(source_table)
            )
            if selected.where(functions.col("_change_type") != "insert").limit(1).count():
                raise ValueError("Source updates and deletes require an explicit rescore policy.")
            selected = selected.where(functions.col("_change_type") == "insert")
        if period_column is not None:
            if selected.schema[period_column].dataType.typeName() != "timestamp":
                raise ValueError("Source event time must be a Spark timestamp.")
            if selected.where(functions.col(period_column).isNull()).limit(1).count():
                raise ValueError("Source event time must not be null.")
        frame = _bounded_frame(
            selected,
            (*row_keys, *inputs),
            row_keys,
            config.source.max_rows,
            config.source.max_bytes,
        )
        if frame.empty:
            return IncrementalBatchResult(
                prior_version, upper_version, 0, 0, int(target_latest["version"]), previous, True
            )
        target = spark.table(target_table)
        output_names = _check_target(spark, selected, target, row_keys, period_column, prepared)
        predicted = prepared.predict(frame.loc[:, list(inputs)])
        if list(predicted.columns) != list(output_names) or len(predicted) != len(frame):
            raise ValueError(
                "Local prediction schema or row count differs from the model contract."
            )
        bridge_frame = pd.concat(
            [frame.loc[:, list(row_keys)].reset_index(drop=True), predicted.reset_index(drop=True)],
            axis=1,
        )
        if _frame_bytes(bridge_frame) > config.source.max_bytes:
            raise ValueError("Prediction result exceeds max_bytes.")
        bridge_columns = (*row_keys, *output_names)
        bridge = spark.createDataFrame(
            [
                tuple(_scalar(value) for value in row)
                for row in bridge_frame.itertuples(index=False, name=None)
            ],
            schema=target.select(*bridge_columns).schema,
        )
        if (
            bridge.select(*row_keys)
            .join(target.select(*row_keys), on=list(row_keys), how="left_semi")
            .limit(1)
            .count()
        ):
            raise BatchConflictError("Source key already has a published prediction.")
        output = (
            bridge.join(selected.select(*row_keys, period_column), on=list(row_keys), how="inner")
            if period_column is not None
            else bridge
        )
        run_input = {
            "skyulf_mode": "incremental_append",
            "source_table_id": source_id,
            "target_table_id": target_id,
            "source_start_version": prior_version + 1 if prior_version is not None else None,
            "source_end_version": upper_version,
            "model_name": config.model.name,
            "model_version": config.model.version,
            "model_digest": prepared.preflight.model_digest,
            "code_version": version("skyulf-core"),
            "input_count": len(frame),
            "output_count": len(frame),
        }
        digest = hashlib.sha256(
            json.dumps(run_input, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest = {**run_input, "run_id": digest, "request_digest": digest}
        for name, value in (
            ("run_id", digest),
            ("model_name", config.model.name),
            ("model_version", config.model.version),
        ):
            output = output.withColumn(name, functions.lit(value))
        if {field.name: field.dataType for field in output.schema} != {
            field.name: field.dataType for field in target.schema
        }:
            raise ValueError("Prediction output schema must match the Delta target exactly.")
        output = output.select(*target.columns)
        if output.count() != len(frame):
            raise ValueError("Prediction keys do not match source rows.")
        if table_identity(spark, source_table) != source_id:
            raise BatchConflictError("Source table identity changed while scoring.")
        if int(_latest(spark, target_table)["version"]) != int(target_latest["version"]):
            raise BatchConflictError("Target changed while scoring; retry from the latest receipt.")
        try:
            (
                output.write.format("delta")
                .mode("append")
                .option("mergeSchema", "false")
                .option("txnAppId", f"skyulf-incremental:{source_id}:{target_id}")
                .option("txnVersion", upper_version)
                .option("userMetadata", json.dumps(manifest, sort_keys=True))
                .saveAsTable(target_table)
            )
        except Exception as exc:  # noqa: BLE001 - preserve Delta's structured cause
            if "Concurrent" in type(exc).__name__ or "DELTA_CONCURRENT" in str(exc):
                raise BatchConflictError("Delta rejected a concurrent incremental write.") from exc
            raise DeltaPublishError("Incremental Delta write failed.") from exc
        committed = _latest(spark, target_table)
        recorded = _last_receipt(committed, source_id, target_id)
        if recorded is None or recorded.get("request_digest") != digest:
            raise DeltaPublishError("Delta returned without a verifiable incremental receipt.")
        return IncrementalBatchResult(
            run_input["source_start_version"],
            upper_version,
            len(frame),
            len(frame),
            int(committed["version"]),
            recorded,
            False,
        )
