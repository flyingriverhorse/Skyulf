"""Read pinned monitoring populations without collecting their rows on the driver."""

import importlib
from datetime import UTC, datetime, timedelta
from functools import reduce
from typing import Any

from ....data.delta_io.delta import table_identity
from ....feature_store.monitoring import enrich_observation, validate_observation_snapshot
from ..monitoring_config import MonitorConfig
from ..monitoring_sources import (
    model_predictions,
    observation_window,
    prediction_columns,
    prediction_source_version,
    read_snapshot,
    snapshot_at,
    window_receipts,
)


def require_unique_keys(frame: Any, keys: tuple[str, ...], name: str) -> None:
    """Reject null/nonfinite or duplicate keys through distributed scalar probes."""
    functions = importlib.import_module("pyspark.sql.functions")
    if not keys or set(keys) - set(frame.columns):
        raise ValueError(f"{name} is missing record key columns.")
    invalid = []
    for key in keys:
        column = functions.col(key)
        condition = column.isNull()
        if frame.schema[key].dataType.typeName() in {"float", "double"}:
            condition = (
                condition | functions.isnan(column) | (functions.abs(column) == float("inf"))
            )
        invalid.append(condition)
    if frame.where(reduce(lambda left, right: left | right, invalid)).limit(1).count():
        raise ValueError(f"{name} has a null or nonfinite record key.")
    if frame.groupBy(*keys).count().where("count > 1").limit(1).count():
        raise ValueError(f"{name} has duplicate record keys.")


def _unchanged_identity(spark: Any, table: str, expected: str) -> None:
    """Reject same-name table replacement across planning and execution."""
    if table_identity(spark, table) != expected:
        raise ValueError(f"Monitoring table was replaced during observation: {table}.")


def _matching_features(
    spark: Any,
    config: MonitorConfig,
    version: str,
    selected: Any,
    receipts: dict,
    keys: tuple[str, ...],
    features: tuple[str, ...],
    source_id: str,
    target_id: str,
    feature_binding: dict | None = None,
) -> tuple[Any, list[dict]]:
    """Join each saved batch against its own source version using distributed keys."""
    functions = importlib.import_module("pyspark.sql.functions")
    frames, used = [], []
    for run_id, item in receipts.items():
        batch_keys = selected.where(functions.col("run_id") == run_id).select(*keys)
        if not batch_keys.limit(1).count():
            continue
        source_version = prediction_source_version(
            item["receipt"], config, version, source_id, target_id
        )
        source = read_snapshot(spark, config.source_table, source_version)
        source = enrich_observation(
            spark,
            source.join(batch_keys, list(keys), "inner"),
            keys,
            features,
            feature_binding,
            item["receipt"],
        )
        joined = source.select(*dict.fromkeys((*keys, *features)))
        require_unique_keys(joined, keys, "current")
        if joined.count() != batch_keys.count():
            raise ValueError("Prediction keys differ from their pinned source snapshot.")
        validate_observation_snapshot(spark, feature_binding)
        frames.append(joined)
        used.append(
            {
                "run_id": run_id,
                "source_version": source_version,
                "prediction_commit_version": item["commit_version"],
                "committed_us": item["committed_us"],
            }
        )
    if frames:
        return reduce(lambda left, right: left.unionByName(right), frames), used
    empty = enrich_observation(
        spark, spark.table(config.source_table).limit(0), keys, features, feature_binding, None
    ).select(*dict.fromkeys((*keys, *features)))
    return empty, used


def read_spark_observation(
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
    feature_binding: dict | None = None,
) -> tuple[Any, Any, dict, datetime | None]:
    """Return distributed current features/predictions and bounded provenance receipts."""
    observation_window(as_of, start, end)
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
    selected = frame.where(functions.col("run_id").isin(list(receipts))).select(
        *prediction_columns(frame, keys, probabilities)
    )
    require_unique_keys(selected, keys, "predictions")
    current, used = _matching_features(
        spark,
        config,
        version,
        selected,
        receipts,
        keys,
        features,
        source_id,
        target_id,
        feature_binding,
    )
    _unchanged_identity(spark, config.source_table, source_id)
    _unchanged_identity(spark, config.prediction_table, target_id)
    evidence = {
        "source_table": config.source_table,
        "source_table_id": source_id,
        "prediction_table": config.prediction_table,
        "prediction_table_id": target_id,
        "prediction_version": prediction_version,
        "batches": used,
        "window_basis": "prediction_commit_timestamp",
        "execution_engine": "spark",
    }
    if feature_binding is not None:
        evidence["feature_lookup"] = feature_binding
    observed = (
        datetime.fromtimestamp(max(item["committed_us"] for item in used) / 1e6, UTC)
        if used
        else None
    )
    return current, selected.drop("run_id"), evidence, observed


def read_spark_labels(
    spark: Any,
    config: MonitorConfig,
    keys: tuple[str, ...],
    target: str,
    predictions: Any,
    as_of: datetime,
) -> tuple[Any | None, dict]:
    """Pin and join relevant outcomes without constructing a driver-side key list."""
    if config.label_table is None:
        return None, {"label_table": None, "label_version": None}
    label_id = table_identity(spark, config.label_table)
    version = snapshot_at(spark, config.label_table, as_of)
    source = read_snapshot(spark, config.label_table, version)
    available = str(config.result_available_at_column)
    if source.schema[available].dataType.typeName() not in {"timestamp", "string"}:
        raise ValueError("Label availability requires a UTC timestamp or aware ISO string.")
    selected = source.join(predictions.select(*keys), list(keys), "left_semi").select(
        *dict.fromkeys((*keys, target, available))
    )
    _unchanged_identity(spark, config.label_table, label_id)
    return selected, {
        "label_table": config.label_table,
        "label_table_id": label_id,
        "label_version": version,
    }
