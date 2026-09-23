"""Explicit scheduled batch execution through the existing Spark predictor."""

import hashlib
import importlib
import json
from dataclasses import asdict
from importlib.metadata import version
from typing import Any

from ...core.execution import ExecutionOptions, FrameSpec
from ...inference.bundle import InferenceBundle, _validate_bundle
from ...inference.spark import predict_spark
from ._contracts import BatchResult, BatchSpec, table_name
from .admission import PublishAdmission, validate_admission
from .delta import history, publish_replace_period, table_identity

__all__ = ["BatchResult", "BatchSpec", "run_batch"]


def run_batch(
    spark: Any,
    spec: BatchSpec,
    *,
    source: str,
    bundle: InferenceBundle,
    options: ExecutionOptions,
    admission: PublishAdmission | None = None,
) -> BatchResult:
    """Score a pinned Delta snapshot and commit a period under explicit admission.

    Source is a Delta table name, not an arbitrary mutable DataFrame. Its recorded
    commit must precede ``as_of``. This proves snapshot availability, not temporal
    correctness of upstream feature joins. The caller binds a concrete registry
    version to ``model_digest`` when constructing the spec. Both existing Spark
    inference modes are supported. Tables and Spark sessions remain caller-owned.
    Output stays distributed when serverless rejects persistence; classic Spark
    caches it through receipt validation and then releases the cached frame.
    """
    if not isinstance(options, ExecutionOptions) or options.engine != "spark":
        raise ValueError("run_batch requires ExecutionOptions(engine='spark').")
    if not isinstance(spec, BatchSpec):
        raise TypeError("spec must be BatchSpec.")
    if spec.mode == "local_pipeline":
        raise ValueError("run_batch requires a Spark inference mode.")
    admission = validate_admission(spark, admission)
    table_name(source)
    _validate_bundle(bundle, options)
    if bundle.semantic_digest != spec.model_digest:
        raise ValueError("Bundle digest differs from the selected model identity.")
    runtime_version = version("skyulf-core")
    if spec.code_version != runtime_version:
        raise ValueError("code_version differs from the installed skyulf-core runtime.")
    if bundle.input_stage != "raw":
        raise ValueError("Spark batch inference requires a raw bundle.")
    source_id = table_identity(spark, source)
    if source_id == table_identity(spark, spec.output_table):
        raise ValueError("Batch source and output table must be different.")
    functions = importlib.import_module("pyspark.sql.functions")
    snapshot = (
        history(spark, source)
        .where(functions.col("version") == spec.source_version)
        .select(functions.unix_micros("timestamp").alias("committed_us"))
        .first()
    )
    if snapshot is None or snapshot.committed_us > int(spec.as_of_utc.timestamp() * 1_000_000):
        raise ValueError("Source snapshot was not available at as_of or its history expired.")
    frame = spark.read.format("delta").option("versionAsOf", spec.source_version).table(source)
    if table_identity(spark, source) != source_id:
        raise ValueError("Source table was replaced while pinning the snapshot.")
    period = frame[spec.period_column]
    if frame.schema[spec.period_column].dataType.typeName() != "timestamp":
        raise ValueError("period_column requires Spark timestamp, not a string/date/timestamp_ntz.")
    if frame.where(period.isNull()).limit(1).count():
        raise ValueError("Source period_column contains null timestamps.")
    selected = frame.where(
        (period >= functions.lit(spec.period_start_utc))
        & (period < functions.lit(spec.period_end_utc))
    )
    input_count = selected.count()
    if not input_count and not spec.allow_empty:
        raise ValueError("Empty period replacement requires allow_empty=True.")
    output_names = {column.name.lower() for column in bundle.manifest.output_schema}
    if spec.period_column.lower() in output_names:
        raise ValueError("period_column collides with a model output column.")
    predictions = predict_spark(
        selected,
        bundle,
        frame_spec=FrameSpec(row_keys=spec.row_keys),
        options=options,
        mode=spec.mode,
    )
    output = predictions.join(
        selected.select(*spec.row_keys, spec.period_column), on=list(spec.row_keys), how="inner"
    )
    for name, value in (
        ("__skyulf_run_id", spec.run_id),
        ("__skyulf_model_name", spec.model_name),
        ("__skyulf_model_version", spec.model_version),
    ):
        output = output.withColumn(name, functions.lit(value))
    target = spark.table(spec.output_table)
    if {f.name: f.dataType for f in output.schema} != {f.name: f.dataType for f in target.schema}:
        raise ValueError("Output schema must match the precreated Delta target exactly.")
    output = output.select(*target.columns)
    persisted = False
    try:
        output = output.persist()
    except Exception as exc:  # noqa: BLE001 - Spark Connect wraps structured runtime errors
        condition = getattr(exc, "getCondition", None)
        if not callable(condition):
            condition = getattr(exc, "getErrorClass", None)
        if not callable(condition) or condition() != "NOT_SUPPORTED_WITH_SERVERLESS":
            raise
    else:
        persisted = True
    try:
        output_count = output.count()
        if input_count != output_count:
            raise ValueError("Prediction changed source row membership.")
        manifest = _manifest(
            spec, source_id, source, snapshot.committed_us, input_count, output_count
        )
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
    finally:
        if persisted:
            output.unpersist()


def _manifest(
    spec: BatchSpec,
    source_id: str,
    source: str,
    committed_us: int,
    input_count: int,
    output_count: int,
) -> dict[str, Any]:
    """Fingerprint the complete request and retain snapshot, code and model evidence."""
    request = asdict(spec)
    for key in ("period_start", "period_end", "as_of"):
        request[key] = getattr(spec, key + "_utc").isoformat()
    request.update(source_table_id=source_id, source_table=source)
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"))
    return dict(
        request,
        request_digest=hashlib.sha256(payload.encode()).hexdigest(),
        input_count=input_count,
        output_count=output_count,
        source_committed_us=committed_us,
        source_temporal_contract="delta_snapshot_committed_by_as_of",
    )
