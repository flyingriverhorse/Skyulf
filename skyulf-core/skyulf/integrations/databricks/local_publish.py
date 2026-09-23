"""Publish bounded local predictions through the existing guarded Delta writer."""

from __future__ import annotations

import importlib
import math
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd

from ...inference.local_pipeline import LocalPipelineArtifact
from ._contracts import BatchResult, BatchSpec
from .admission import PublishAdmission, validate_admission
from .batch import _manifest
from .delta import history, publish_replace_period, table_identity
from .local_batch import LocalSourceSpec, score_local_source
from .local_sdk import PreparedLocalWorkflow

_OUTPUT_TYPES = {"float64": "double", "int64": "long", "string": "string", "bool": "boolean"}
_METADATA = ("__skyulf_run_id", "__skyulf_model_name", "__skyulf_model_version")


def _validate_request(
    spark: Any,
    source: LocalSourceSpec,
    prepared: PreparedLocalWorkflow,
    spec: BatchSpec,
    admission: PublishAdmission | None,
) -> PublishAdmission:
    """Reject mismatched model, source and target identities before scoring."""
    if not isinstance(source, LocalSourceSpec) or not isinstance(prepared, PreparedLocalWorkflow):
        raise TypeError("Local publication requires a source spec and prepared workflow.")
    if not isinstance(spec, BatchSpec) or spec.mode != "local_pipeline":
        raise ValueError("Local publication requires BatchSpec(mode='local_pipeline').")
    admission = validate_admission(spark, admission)
    if not prepared.preflight.ready or not isinstance(prepared.artifact, LocalPipelineArtifact):
        raise ValueError("Local publication requires a ready fitted local pipeline.")
    config = prepared.config
    if config.runtime != "databricks" or config.sink.kind != "uc_delta":
        raise ValueError("Local publication requires a Databricks UC Delta sink.")
    if config.sink.table != spec.output_table or config.source.kind != "uc_table":
        raise ValueError("Configured UC source or target differs from the publication request.")
    if config.source.table != source.table or config.source.version != source.version:
        raise ValueError("Prepared source differs from the pinned Delta snapshot.")
    if source.max_rows > config.source.max_rows or source.max_bytes > config.source.max_bytes:
        raise ValueError("Source budget exceeds the prepared workflow budget.")
    if config.model.kind != "local_pipeline" or config.model.name != spec.model_name:
        raise ValueError("Publication model name differs from the prepared registry model.")
    if prepared.preflight.model_version != spec.model_version:
        raise ValueError("Publication model version differs from the pinned model.")
    if prepared.preflight.model_digest != spec.model_digest:
        raise ValueError("Publication model digest differs from the fitted artifact.")
    if spec.code_version != version("skyulf-core"):
        raise ValueError("Publication code version differs from the installed runtime.")
    if (
        spec.source_version != source.version
        or spec.row_keys != source.row_keys
        or spec.period_column != source.period_column
        or spec.period_start_utc != source.period_start.astimezone(spec.period_start_utc.tzinfo)
        or spec.period_end_utc != source.period_end.astimezone(spec.period_end_utc.tzinfo)
        or spec.business_timezone != source.business_timezone
    ):
        raise ValueError("Publication period and source contract must match exactly.")
    if source.table == spec.output_table:
        raise ValueError("Source and prediction target must be different tables.")
    return admission


def _check_target(
    spark: Any,
    source_frame: Any,
    target: Any,
    source: LocalSourceSpec,
    prepared: PreparedLocalWorkflow,
) -> tuple[str, ...]:
    """Require a precreated target with exact key, output and metadata types."""
    outputs = prepared.preflight.output_schema
    names = tuple(column.name for column in outputs)
    expected = {*source.row_keys, source.period_column, *names, *_METADATA}
    if len(expected) != len(source.row_keys) + len(names) + 4 or set(target.columns) != expected:
        raise ValueError("Prediction target columns differ from the explicit local output schema.")
    if target.schema[source.period_column].dataType.typeName() != "timestamp":
        raise ValueError("Prediction target period must be a Spark timestamp.")
    if source_frame.schema[source.period_column].dataType.typeName() != "timestamp":
        raise ValueError("Source period must be a Spark timestamp.")
    for name in source.row_keys:
        target_type = target.schema[name].dataType
        if target_type != source_frame.schema[name].dataType or target_type.typeName() not in (
            "long",
            "string",
        ):
            raise ValueError("Source and target row-key types must match and be long or string.")
    for column in outputs:
        expected_type = _OUTPUT_TYPES.get(column.dtype)
        if expected_type is None or target.schema[column.name].dataType.typeName() != expected_type:
            raise ValueError(f"Target output type differs from saved model output {column.name!r}.")
    if any(target.schema[name].dataType.typeName() != "string" for name in _METADATA):
        raise ValueError("Prediction target metadata columns must be strings.")
    return names


def _scalar(value: Any) -> Any:
    """Convert bounded pandas/NumPy scalars without changing their logical type."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError("Prediction output contains a nonfinite number.")
    return value


def run_local_batch(
    spark: Any,
    source: LocalSourceSpec,
    prepared: PreparedLocalWorkflow,
    spec: BatchSpec,
    *,
    admission: PublishAdmission | None,
) -> BatchResult:
    """Score one pinned month locally and publish only its final rows to Delta."""
    admission = _validate_request(spark, source, prepared, spec, admission)
    source_id = table_identity(spark, source.table)
    if source_id == table_identity(spark, spec.output_table):
        raise ValueError("Source and prediction target must be different Delta tables.")
    functions = importlib.import_module("pyspark.sql.functions")
    snapshot = (
        history(spark, source.table)
        .where(functions.col("version") == source.version)
        .select(functions.unix_micros("timestamp").alias("committed_us"))
        .first()
    )
    if snapshot is None or snapshot.committed_us > int(spec.as_of_utc.timestamp() * 1_000_000):
        raise ValueError("Source snapshot was not available at as_of or its history expired.")
    scored = score_local_source(spark, source, prepared)
    count = len(scored.predictions)
    if count == 0 and not spec.allow_empty:
        raise ValueError("Empty period replacement requires allow_empty=True.")
    if count != scored.diagnostics["row_count"]:
        raise ValueError("Scored row count differs from the pinned source.")
    if table_identity(spark, source.table) != source_id:
        raise ValueError("Source table was replaced while scoring the pinned snapshot.")
    source_frame = (
        spark.read.format("delta").option("versionAsOf", source.version).table(source.table)
    )
    target = spark.table(spec.output_table)
    output_names = _check_target(spark, source_frame, target, source, prepared)
    bridge_names = (*source.row_keys, *output_names)
    if list(scored.predictions.columns) != list(bridge_names):
        raise ValueError("Local prediction columns differ from the saved model output.")
    if scored.predictions.loc[:, list(source.row_keys)].isna().any().any():
        raise ValueError("Local prediction row keys must not be null.")
    if scored.predictions.duplicated(subset=list(source.row_keys)).any():
        raise ValueError("Local prediction row keys must be unique.")
    bridge_schema = target.select(*bridge_names).schema
    records = [
        tuple(_scalar(value) for value in row)
        for row in scored.predictions.itertuples(index=False, name=None)
    ]
    bridge = spark.createDataFrame(records, schema=bridge_schema)
    period = source_frame[source.period_column]
    source_period = source_frame.where(
        (period >= functions.lit(spec.period_start_utc))
        & (period < functions.lit(spec.period_end_utc))
    ).select(*source.row_keys, source.period_column)
    output = bridge.join(source_period, on=list(source.row_keys), how="inner")
    for name, value in (
        ("__skyulf_run_id", spec.run_id),
        ("__skyulf_model_name", spec.model_name),
        ("__skyulf_model_version", spec.model_version),
    ):
        output = output.withColumn(name, functions.lit(value))
    if {field.name: field.dataType for field in output.schema} != {
        field.name: field.dataType for field in target.schema
    }:
        raise ValueError("Prediction output schema must match the Delta target exactly.")
    output = output.select(*target.columns)
    if output.count() != count:
        raise ValueError("Prediction keys do not match the pinned source rows.")
    manifest = _manifest(spec, source_id, source.table, snapshot.committed_us, count, count)
    committed_version, recorded, replayed = publish_replace_period(
        spark, output, spec, manifest=manifest, admission=admission
    )
    return BatchResult(
        spec,
        recorded["input_count"],
        recorded["output_count"],
        committed_version,
        recorded,
        replayed,
    )
