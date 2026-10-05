"""Distributed execution primitives used by the existing Delta receipt lifecycles."""

import importlib
from dataclasses import dataclass
from functools import reduce
from operator import or_
from typing import Any

from .admission import BatchConflictError


@dataclass(frozen=True)
class SparkSetExecution:
    """Hold the pinned set and explicitly selected worker runtime for one run."""

    spark: Any
    artifact: Any
    model_uri: str
    env_manager: str
    prediction_batch_rows: int
    tracking_uri: str | None = None
    registry_uri: str | None = None


@dataclass(frozen=True)
class DistributedRows:
    """Keep a Spark relation and its scalar population count without driver decoding."""

    frame: Any
    row_count: int
    execution: SparkSetExecution | None = None

    def __len__(self) -> int:
        """Expose receipt counts without materializing rows on the driver."""
        return self.row_count

    @property
    def empty(self) -> bool:
        """Identify a no-op window using the distributed count."""
        return self.row_count == 0

    def select(self, columns: list[str]) -> "DistributedRows":
        """Project publication columns without changing row membership."""
        return DistributedRows(self.frame.select(*columns), self.row_count, self.execution)


def read_distributed_rows(
    selected: Any,
    columns: tuple[str, ...],
    record_key_columns: tuple[str, ...],
    *,
    execution: SparkSetExecution | None = None,
) -> DistributedRows:
    """Check global key uniqueness and retain every selected row on Spark."""
    if not record_key_columns:
        raise ValueError("Distributed scoring requires record keys.")
    functions = importlib.import_module("pyspark.sql.functions")
    frame = selected.select(*columns)
    missing = reduce(or_, (functions.col(name).isNull() for name in record_key_columns))
    summary = frame.agg(
        functions.count(functions.lit(1)).alias("rows"),
        functions.countDistinct(functions.struct(*record_key_columns)).alias("keys"),
        functions.count(functions.when(missing, functions.lit(1))).alias("missing"),
    ).first()
    if summary["missing"]:
        raise ValueError("Source row keys must not be null.")
    if summary["keys"] != summary["rows"]:
        raise ValueError("Source row keys must be globally unique within this increment.")
    return DistributedRows(frame, int(summary["rows"]), execution)


def validate_prepared_spark(prepared: Any) -> None:
    """Reject unsafe saved behavior before source selection or target provisioning."""
    if prepared.config.inference_mode == "local":
        return
    from ...inference.partition_safety import require_partition_safe_pipeline  # noqa: PLC0415

    require_partition_safe_pipeline(prepared.artifact)


def score_distributed_single(
    spark: Any,
    prepared: Any,
    rows: DistributedRows,
    keys: tuple[str, ...],
    target: Any,
    *,
    replacing: bool,
) -> tuple[Any, dict[str, int]]:
    """Produce keyed UDF output while reusing the single-model transaction protocol."""
    from ..mlflow.spark_model import predict_spark_pyfunc  # noqa: PLC0415

    config = prepared.config
    output = predict_spark_pyfunc(
        spark,
        rows.frame,
        model_uri=f"models:/{config.model.name}/{config.model.version}",
        artifact=prepared.artifact,
        record_key_columns=keys,
        env_manager=config.spark_udf_env_manager,
        prediction_batch_rows=config.spark_udf_prediction_batch_rows,
        tracking_uri=config.model.tracking_uri,
        registry_uri=config.model.registry_uri,
    )
    reject_published_keys(output, target, keys, replacing=replacing)
    return output, {}


def reject_published_keys(
    output: Any, target: Any, keys: tuple[str, ...], *, replacing: bool
) -> None:
    """Keep existing append key collision rules for distributed predictions."""
    if not replacing and (
        output.select(*keys)
        .join(target.select(*keys), on=list(keys), how="left_semi")
        .limit(1)
        .count()
    ):
        raise BatchConflictError("Source key already has a published prediction.")


def prepare_spark_set_execution(
    spark: Any,
    artifact: Any,
    model_uri: str,
    inference_mode: str,
    env_manager: str,
    prediction_batch_rows: int,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> SparkSetExecution | None:
    """Admit every component and composition before any model-set table operation."""
    if inference_mode not in {"local", "spark"}:
        raise ValueError("inference_mode must be local or spark.")
    if env_manager not in {"local", "virtualenv"}:
        raise ValueError("spark_udf_env_manager must be local or virtualenv.")
    if type(prediction_batch_rows) is not int or not 0 < prediction_batch_rows <= 100_000:
        raise ValueError("spark_udf_prediction_batch_rows must be between 1 and 100000.")
    if inference_mode == "local":
        return None
    from ...inference.model_set_partition_safety import (  # noqa: PLC0415
        require_partition_safe_model_set,
    )

    require_partition_safe_model_set(artifact)
    return SparkSetExecution(
        spark, artifact, model_uri, env_manager, prediction_batch_rows, tracking_uri, registry_uri
    )


@dataclass(frozen=True)
class DistributedSetResult:
    """Match the existing model-set score result without a local frame or history."""

    frame: DistributedRows
    history: dict[str, Any]


def score_distributed_set(rows: DistributedRows) -> DistributedSetResult:
    """Score one immutable set package so no component publishes independently."""
    from ..mlflow.spark_model import predict_spark_pyfunc  # noqa: PLC0415

    execution = rows.execution
    if execution is None:
        raise ValueError("Distributed model-set input has no pinned execution context.")
    output = predict_spark_pyfunc(
        execution.spark,
        rows.frame,
        model_uri=execution.model_uri,
        artifact=execution.artifact,
        record_key_columns=execution.artifact.manifest.record_key_columns,
        env_manager=execution.env_manager,
        prediction_batch_rows=execution.prediction_batch_rows,
        tracking_uri=execution.tracking_uri,
        registry_uri=execution.registry_uri,
    )
    if output.count() != len(rows):
        raise ValueError("Model-set prediction row count differs from the source contract.")
    return DistributedSetResult(DistributedRows(output, len(rows), execution), {})


def complete_distributed_set(
    spark: Any,
    rows: DistributedRows,
    target: str,
    receipt: dict[str, Any],
    metadata: tuple[str, ...],
    keys: tuple[str, ...],
    write_mode: str,
) -> Any:
    """Attach provenance and validate publication schema on the distributed relation."""
    functions = importlib.import_module("pyspark.sql.functions")
    output = rows.frame
    for name in metadata:
        output = output.withColumn(name, functions.lit(receipt[name]))
    target_frame = spark.table(target)
    if {field.name: field.dataType for field in output.schema} != {
        field.name: field.dataType for field in target_frame.schema
    }:
        raise ValueError("Model-set prediction schema must match the Delta target exactly.")
    reject_published_keys(output, target_frame, keys, replacing=write_mode == "overwrite")
    return output.select(*target_frame.columns)
