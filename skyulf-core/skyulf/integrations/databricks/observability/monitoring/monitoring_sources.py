"""Bounded monitoring reads bound to actual prediction and source Delta commits."""

import importlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from ...data.delta_io.delta import history, table_identity
from ...scoring.incremental.incremental_batch import bounded_frame
from ...shared._contracts import column_name
from .monitoring_config import MonitorConfig


def observation_window(
    as_of: datetime,
    start: datetime | None,
    end: datetime | None,
) -> tuple[datetime, datetime]:
    """Use an explicit half-open scoring-commit window and UTC observation cutoff."""
    end = as_of if end is None else end
    start = end - timedelta(days=1) if start is None else start
    for value in (as_of, start, end):
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Monitoring cutoffs must be timezone-aware.")
    if not start < end <= as_of:
        raise ValueError("Monitoring requires window_start < window_end <= as_of.")
    return start.astimezone(UTC), end.astimezone(UTC)


def snapshot_at(spark: Any, table: str, as_of: datetime) -> int:
    """Pin the most recent commit available at the observation cutoff."""
    functions = importlib.import_module("pyspark.sql.functions")
    row = (
        history(spark, table)
        .where(functions.col("timestamp") <= functions.lit(as_of))
        .orderBy(functions.desc("version"))
        .select("version")
        .first()
    )
    if row is None:
        raise ValueError(f"No Delta snapshot is available at the monitoring cutoff: {table}.")
    return int(row["version"])


def read_snapshot(spark: Any, table: str, version: int) -> Any:
    """Never fall back to latest data if a pinned snapshot has expired."""
    return spark.read.format("delta").option("versionAsOf", version).table(table)


def receipt_index(rows: list[Any]) -> dict[str, dict]:
    """Require a unique scoring receipt for each run rather than guessing source versions."""
    receipts = {}
    for row in rows:
        receipt = _commit_receipt(row)
        if receipt is None:
            continue
        run_id = receipt["run_id"]
        if run_id in receipts:
            raise ValueError("Duplicate prediction receipts for one run_id.")
        receipts[run_id] = {
            "receipt": receipt,
            "commit_version": int(row["version"]),
            "committed_us": row.get("committed_us")
            if isinstance(row, dict)
            else row["committed_us"],
        }
    return receipts


def _commit_receipt(row: Any) -> dict | None:
    """Ignore maintenance metadata but reject unreceipted writes to prediction values."""
    values = row if isinstance(row, dict) else row.asDict()
    raw = values.get("userMetadata")
    try:
        receipt = json.loads(raw) if raw else None
    except (TypeError, json.JSONDecodeError):
        receipt = None
    if isinstance(receipt, dict) and receipt.get("run_id"):
        return receipt
    if values.get("operation") in {
        "CREATE TABLE",
        "OPTIMIZE",
        "VACUUM START",
        "VACUUM END",
        "SET TBLPROPERTIES",
        "UNSET TBLPROPERTIES",
        "COMPUTE STATS",
    }:
        return None
    raise ValueError("Prediction window contains an unreceipted data or schema change.")


def validate_receipt(
    receipt: dict,
    source_id: str,
    target_id: str,
    model_name: str,
    model_version: str,
) -> int:
    """Verify source-table incarnation, prediction target and concrete model identity."""
    expected = {
        "source_table_id": source_id,
        "target_table_id": target_id,
        "model_name": model_name,
        "model_version": model_version,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Prediction receipt source, target or model identity differs.")
    version = receipt.get("source_end_version", receipt.get("source_version"))
    if type(version) is not int or version < 0:
        raise ValueError("Prediction receipt has no valid pinned source version.")
    return version


def window_receipts(spark: Any, config: MonitorConfig, start: datetime, end: datetime) -> dict:
    """Bound history transfer and reject writes whose source provenance is unavailable."""
    functions = importlib.import_module("pyspark.sql.functions")
    rows = (
        history(spark, config.prediction_table)
        .where(
            (functions.col("timestamp") >= functions.lit(start))
            & (functions.col("timestamp") < functions.lit(end))
        )
        .select(
            "version",
            "userMetadata",
            "operation",
            functions.expr("unix_micros(timestamp)").alias("committed_us"),
        )
    )
    records = rows.orderBy("version").limit(config.max_batches + 1).collect()
    if len(records) > config.max_batches:
        raise ValueError("Monitoring window exceeds max_batches.")
    return receipt_index(records)


def prediction_columns(frame: Any, keys: tuple[str, ...], probabilities: int) -> tuple[str, ...]:
    """Select only raw saved model outputs, excluding unrelated business projections."""
    columns = (*keys, "run_id", "prediction", *(f"probability_{i}" for i in range(probabilities)))
    if "scoring_status" in frame.columns:
        columns += ("scoring_status",)
    return columns


def read_current_observation(
    spark: Any,
    config: MonitorConfig,
    version: str,
    keys: tuple[str, ...],
    features: tuple[str, ...],
    *,
    probabilities: int,
    as_of: datetime,
    start: datetime,
    end: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, datetime | None]:
    """Join stored predictions to their original raw inputs by key and receipt snapshot."""
    functions = importlib.import_module("pyspark.sql.functions")
    source_id = table_identity(spark, config.source_table)
    target_id = table_identity(spark, config.prediction_table)
    prediction_version = snapshot_at(
        spark, config.prediction_table, end - timedelta(microseconds=1)
    )
    receipts = window_receipts(spark, config, start, end)
    frame = model_predictions(
        read_snapshot(spark, config.prediction_table, prediction_version),
        config,
        version,
        keys,
        probabilities,
    )
    columns = prediction_columns(frame, keys, probabilities)
    selected = frame.where(functions.col("run_id").isin(list(receipts)))
    predictions = bounded_frame(selected, columns, keys, config.max_rows, config.max_bytes)
    current, used, observed = _read_matching_features(
        spark,
        config,
        version,
        predictions,
        selected,
        receipts,
        keys,
        features,
        source_id,
        target_id,
    )
    evidence = {
        "source_table": config.source_table,
        "source_table_id": source_id,
        "prediction_table": config.prediction_table,
        "prediction_table_id": target_id,
        "prediction_version": prediction_version,
        "batches": used,
        "window_basis": "prediction_commit_timestamp",
    }
    return current, predictions.drop(columns="run_id"), evidence, observed


def _read_matching_features(
    spark: Any,
    config: MonitorConfig,
    version: str,
    predictions: pd.DataFrame,
    selected: Any,
    receipts: dict,
    keys: tuple[str, ...],
    features: tuple[str, ...],
    source_id: str,
    target_id: str,
) -> tuple[pd.DataFrame, list[dict], datetime | None]:
    """Resolve each surviving batch against its own source version, never the current source."""
    functions = importlib.import_module("pyspark.sql.functions")
    frames, used = [], []
    for run_id in predictions["run_id"].unique():
        item = receipts[run_id]
        source_version = prediction_source_version(
            item["receipt"], config, version, source_id, target_id
        )
        batch_keys = selected.where(functions.col("run_id") == run_id).select(*keys)
        source = read_snapshot(spark, config.source_table, source_version)
        joined = source.join(batch_keys, list(keys), "inner")
        native = bounded_frame(joined, (*keys, *features), keys, config.max_rows, config.max_bytes)
        if len(native) != int((predictions["run_id"] == run_id).sum()):
            raise ValueError("Prediction keys differ from their pinned source snapshot.")
        frames.append(native)
        used.append(
            {
                "run_id": run_id,
                "source_version": source_version,
                "prediction_commit_version": item["commit_version"],
                "committed_us": item["committed_us"],
            }
        )
    current = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*keys, *features])
    )
    if int(current.memory_usage(deep=True).sum()) > config.max_bytes:
        raise ValueError("Monitoring current inputs exceed max_bytes.")
    observed = (
        datetime.fromtimestamp(max(item["committed_us"] for item in used) / 1e6, UTC)
        if used
        else None
    )
    return current, used, observed


def model_predictions(
    frame: Any, config: MonitorConfig, version: str, keys: tuple[str, ...], probabilities: int
) -> Any:
    """Normalize raw component outputs without confusing similarly named model-set branches."""
    functions = importlib.import_module("pyspark.sql.functions")
    if config.model_set_name is None:
        return frame.where(
            (functions.col("model_name") == config.model_name)
            & (functions.col("model_version") == version)
        )
    frame = frame.where(
        (functions.col("model_set_name") == config.model_set_name)
        & (functions.col("model_set_version") == config.model_set_version)
    )
    outputs = ["prediction", *(f"probability_{i}" for i in range(probabilities))]
    status = f"{config.model_set_branch}__scoring_status"
    if status in frame.columns:
        outputs.append("scoring_status")
    return frame.select(
        *keys,
        "run_id",
        *(functions.col(f"{config.model_set_branch}__{name}").alias(name) for name in outputs),
    )


def prediction_source_version(
    receipt: dict, config: MonitorConfig, version: str, source_id: str, target_id: str
) -> int:
    """Verify either ordinary output identity or its explicitly selected parent release."""
    if config.model_set_name is None:
        return validate_receipt(receipt, source_id, target_id, config.model_name, version)
    normalized = receipt | {
        "model_name": receipt.get("model_set_name"),
        "model_version": receipt.get("model_set_version"),
    }
    return validate_receipt(
        normalized, source_id, target_id, config.model_set_name, str(config.model_set_version)
    )


def read_labels(
    spark: Any,
    config: MonitorConfig,
    keys: tuple[str, ...],
    target: str,
    predictions: pd.DataFrame,
    as_of: datetime,
) -> tuple[pd.DataFrame | None, dict]:
    """Read only relevant label keys from a pinned table; availability is checked by metrics."""
    if config.label_table is None:
        return None, {"label_table": None, "label_version": None}
    label_id = table_identity(spark, config.label_table)
    version = snapshot_at(spark, config.label_table, as_of)
    columns = (*keys, target, str(config.result_available_at_column))
    if predictions.empty:
        return pd.DataFrame(columns=columns), {
            "label_table": config.label_table,
            "label_version": version,
            "label_table_id": label_id,
        }
    source = read_snapshot(spark, config.label_table, version)
    available = str(config.result_available_at_column)
    timestamp = source.schema[available].dataType.typeName()
    if timestamp not in {"timestamp", "string"}:
        raise ValueError("Label availability requires a UTC timestamp or aware ISO string.")
    key_rows = [
        tuple(row) for row in predictions.loc[:, list(keys)].itertuples(index=False, name=None)
    ]
    key_schema = source.select(*keys).schema
    selected = source.join(spark.createDataFrame(key_rows, key_schema), list(keys), "inner")
    if timestamp == "timestamp":
        functions = importlib.import_module("pyspark.sql.functions")
        selected = selected.withColumn(
            available, functions.expr(f"unix_micros({column_name(available)})")
        )
    native = bounded_frame(
        selected, columns, keys, config.max_rows, config.max_bytes, require_unique_keys=False
    )
    if timestamp == "timestamp":
        native[available] = pd.to_datetime(native[available], unit="us", utc=True)
    if table_identity(spark, config.label_table) != label_id:
        raise ValueError("Label table was replaced during monitoring.")
    return native, {
        "label_table": config.label_table,
        "label_version": version,
        "label_table_id": label_id,
    }
