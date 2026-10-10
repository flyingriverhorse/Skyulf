"""Publish bounded whole-frame predictions through the guarded Delta writer."""

import importlib
from importlib.metadata import version
from typing import Any

from skyulf.integrations.databricks.shared._batch_manifest import batch_manifest
from skyulf.integrations.databricks.shared._local_frames import output_scalar

from ....inference.fitted_pipeline import FittedPipelineArtifact
from ..data.admission import PublishAdmission, validate_admission
from ..data.delta_io.delta import history, publish_replace_period, table_identity
from ..shared._contracts import PREDICTION_METADATA_COLUMNS, BatchResult, BatchSpec
from .batch.frame_batch import SourceSpec
from .batch.frame_batch import score_source as score_local_source
from .incremental.history import bind_period_history, prediction_history
from .workflow import PreparedWorkflow, WorkflowConfig

_OUTPUT_TYPES = {"float64": "double", "int64": "long", "string": "string", "bool": "boolean"}
_METADATA = PREDICTION_METADATA_COLUMNS


def _validate_request(
    spark: Any,
    source: SourceSpec,
    prepared: PreparedWorkflow,
    spec: BatchSpec,
    admission: PublishAdmission | None,
) -> PublishAdmission:
    """Reject mismatched model, source and target identities before scoring."""
    if not isinstance(source, SourceSpec) or not isinstance(prepared, PreparedWorkflow):
        raise TypeError("Local publication requires a source spec and prepared workflow.")
    if not isinstance(spec, BatchSpec) or spec.mode != "local_pipeline":
        raise ValueError("Local publication requires BatchSpec(mode='local_pipeline').")
    admission = validate_admission(spark, admission)
    if not prepared.preflight.ready or not isinstance(prepared.artifact, FittedPipelineArtifact):
        raise ValueError("Local publication requires a ready fitted local pipeline.")
    config = _validate_publication_source(source, prepared, spec)
    _validate_publication_model(prepared, spec, config)
    _validate_publication_period(source, spec)
    return admission


def check_target(
    spark: Any,
    source_frame: Any,
    target: Any,
    record_key_columns: tuple[str, ...],
    period_column: str | None,
    prepared: PreparedWorkflow,
) -> tuple[str, ...]:
    """Require a precreated target with exact key, output and metadata types."""
    outputs = prepared.preflight.output_schema
    names = tuple(column.name for column in outputs)
    expected = {*record_key_columns, *names, *_METADATA}
    if period_column is not None:
        expected.add(period_column)
    if (
        len(expected)
        != len(record_key_columns) + len(names) + len(_METADATA) + int(period_column is not None)
        or set(target.columns) != expected
    ):
        raise ValueError("Prediction target columns differ from the explicit local output schema.")
    _check_target_controls(source_frame, target, record_key_columns, period_column)
    for column in outputs:
        expected_type = _OUTPUT_TYPES.get(column.dtype)
        if expected_type is None or target.schema[column.name].dataType.typeName() != expected_type:
            raise ValueError(f"Target output type differs from saved model output {column.name!r}.")
    if any(target.schema[name].dataType.typeName() != "string" for name in _METADATA):
        raise ValueError("Prediction target metadata columns must be strings.")
    return names


def run_frame_batch(
    spark: Any,
    source: SourceSpec,
    prepared: PreparedWorkflow,
    spec: BatchSpec,
    *,
    admission: PublishAdmission | None,
    history_state: dict[str, Any] | None = None,
) -> BatchResult:
    """Score one pinned month with pandas/Polars and publish its final rows to Delta."""
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
    with prediction_history(prepared, history_state) as temporal_session:
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
    output_names = check_target(
        spark, source_frame, target, source.record_key_columns, source.period_column, prepared
    )
    bridge = _local_prediction_bridge(spark, source, scored, output_names, target)
    period = source_frame[source.period_column]
    source_period = source_frame.where(
        (period >= functions.lit(spec.period_start_utc))
        & (period < functions.lit(spec.period_end_utc))
    ).select(*source.record_key_columns, source.period_column)
    output = bridge.join(source_period, on=list(source.record_key_columns), how="inner")
    output = _complete_local_output(output, target, spec, functions, count)
    manifest = batch_manifest(spec, source_id, source.table, snapshot.committed_us, count, count)
    manifest |= {
        key: scored.diagnostics[key]
        for key in ("predicted_count", "excluded_count")
        if key in scored.diagnostics
    }
    manifest = bind_period_history(manifest, temporal_session, history_state)
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


def _validate_publication_source(
    source: SourceSpec, prepared: PreparedWorkflow, spec: BatchSpec
) -> WorkflowConfig:
    """Validate configured target, pinned source and local input budgets."""
    config = prepared.config
    if config.runtime != "databricks" or config.sink.kind != "uc_delta":
        raise ValueError("Local publication requires a Databricks UC Delta sink.")
    if config.sink.table != spec.output_table or config.source.kind != "uc_table":
        raise ValueError("Configured UC source or target differs from the publication request.")
    if config.source.table != source.table or config.source.version != source.version:
        raise ValueError("Prepared source differs from the pinned Delta snapshot.")
    if source.max_rows > config.source.max_rows or source.max_bytes > config.source.max_bytes:
        raise ValueError("Source budget exceeds the prepared workflow budget.")
    return config


def _validate_publication_model(
    prepared: PreparedWorkflow, spec: BatchSpec, config: WorkflowConfig
) -> None:
    """Validate pinned registry identity and installed scoring code."""
    if config.model.kind != "local_pipeline" or config.model.name != spec.model_name:
        raise ValueError("Publication model name differs from the prepared registry model.")
    if prepared.preflight.model_version != spec.model_version:
        raise ValueError("Publication model version differs from the pinned model.")
    if prepared.preflight.model_digest != spec.model_digest:
        raise ValueError("Publication model digest differs from the fitted artifact.")
    if spec.code_version != version("skyulf-core"):
        raise ValueError("Publication code version differs from the installed runtime.")


def _validate_publication_period(source: SourceSpec, spec: BatchSpec) -> None:
    """Require publication boundaries to match the source contract exactly."""
    if (
        spec.source_version != source.version
        or spec.record_key_columns != source.record_key_columns
        or spec.period_column != source.period_column
        or spec.period_start_utc != source.period_start.astimezone(spec.period_start_utc.tzinfo)
        or spec.period_end_utc != source.period_end.astimezone(spec.period_end_utc.tzinfo)
        or spec.business_timezone != source.business_timezone
    ):
        raise ValueError("Publication period and source contract must match exactly.")
    if source.table == spec.output_table:
        raise ValueError("Source and prediction target must be different tables.")


def _check_target_controls(
    source_frame: Any, target: Any, record_key_columns: tuple[str, ...], period_column: str | None
) -> None:
    """Validate target timestamp and row-key types against the source."""
    if period_column is not None:
        if target.schema[period_column].dataType.typeName() != "timestamp":
            raise ValueError("Prediction target period must be a Spark timestamp.")
        if source_frame.schema[period_column].dataType.typeName() != "timestamp":
            raise ValueError("Source period must be a Spark timestamp.")
    for name in record_key_columns:
        target_type = target.schema[name].dataType
        if target_type != source_frame.schema[name].dataType or target_type.typeName() not in (
            "long",
            "string",
        ):
            raise ValueError("Source and target row-key types must match and be long or string.")


def _local_prediction_bridge(
    spark: Any, source: SourceSpec, scored: Any, output_names: tuple[str, ...], target: Any
) -> Any:
    """Check local prediction keys and construct a Spark frame with target types."""
    bridge_names = (*source.record_key_columns, *output_names)
    if list(scored.predictions.columns) != list(bridge_names):
        raise ValueError("Local prediction columns differ from the saved model output.")
    if scored.predictions.loc[:, list(source.record_key_columns)].isna().any().any():
        raise ValueError("Local prediction row keys must not be null.")
    if scored.predictions.duplicated(subset=list(source.record_key_columns)).any():
        raise ValueError("Local prediction row keys must be unique.")
    bridge_schema = target.select(*bridge_names).schema
    records = [
        tuple(output_scalar(value) for value in row)
        for row in scored.predictions.itertuples(index=False, name=None)
    ]
    bridge = spark.createDataFrame(records, schema=bridge_schema)
    return bridge


def _complete_local_output(
    output: Any, target: Any, spec: BatchSpec, functions: Any, count: int
) -> Any:
    """Attach prediction identity and verify final output schema and membership."""
    for name, value in (
        ("run_id", spec.run_id),
        ("model_name", spec.model_name),
        ("model_version", spec.model_version),
    ):
        output = output.withColumn(name, functions.lit(value))
    if {field.name: field.dataType for field in output.schema} != {
        field.name: field.dataType for field in target.schema
    }:
        raise ValueError("Prediction output schema must match the Delta target exactly.")
    output = output.select(*target.columns)
    if output.count() != count:
        raise ValueError("Prediction keys do not match the pinned source rows.")
    return output


# Preserve public imports and pickle-qualified names from earlier releases.
run_local_batch = run_frame_batch


# Preserve class imports exposed by earlier module paths.
LocalWorkflowConfig = WorkflowConfig
PreparedLocalWorkflow = PreparedWorkflow
LocalSourceSpec = SourceSpec
LocalPipelineArtifact = FittedPipelineArtifact
