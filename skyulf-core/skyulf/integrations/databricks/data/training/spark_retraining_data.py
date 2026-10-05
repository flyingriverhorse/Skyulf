"""Compare distributed training populations with exact bounded split metadata."""

import hashlib
import importlib
import json
import pickle
from dataclasses import replace
from datetime import datetime
from typing import Any, cast

import pandas as pd

from ...jobs.lifecycle.lifecycle_tasks import phase_training_spec
from ...lifecycle.local_workflow import resolve_training_spec
from ...observability.monitoring.spark.spark_monitoring_reference import (
    load_spark_monitoring_reference,
    read_reference_population,
)
from ...observability.monitoring.spark.spark_monitoring_sources import require_unique_keys
from ...training.fitting.local_retraining import (
    LocalTrainingSpec,
    _eligible_training_source,
    _partition_training_rows,
    _sample_training_source,
    training_spec_payload,
)
from ...training.weights.local_weights import extract_training_weights
from ..delta_io.delta import table_identity
from .retraining_data import _comparison_spec, _compatible_specs, _filter_weight_columns, _scalar
from .training_dates import (
    TrainingDateSpec,
    instant_from_microseconds,
    instant_microseconds,
    normalize_training_dates,
)


def _functions() -> Any:
    """Load optional Spark only when distributed work is requested."""
    return importlib.import_module("pyspark.sql.functions")


def _row_hash(values: Any) -> str:
    """Use the trainer's typed scalar encoding, including numeric widening equivalence."""
    payload = json.dumps([_scalar(value) for value in values], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _require_supported_recipe(spec: LocalTrainingSpec) -> None:
    """Reject transformations without an exact distributed eligibility adapter."""
    if set(spec.source_columns) & {"_ordinal", "_duplicates", "_survivor", "_weight_match"}:
        raise ValueError("Spark freshness source uses a reserved internal column name.")
    for step in spec.pre_split_steps:
        if step["transformer"] not in {"DropMissingRows", "ManualBounds", "Deduplicate"}:
            raise ValueError(f"Unsupported Spark freshness pre_split: {step['transformer']}.")
    if spec.weights_python_source:
        raise ValueError(
            "Spark freshness requires source weight columns; custom weights unsupported."
        )
    if any((spec.sample_key_sha256, spec.survivor_key_sha256, spec.holdout_key_sha256)):
        raise ValueError("Spark freshness current selection cannot carry saved membership digests.")


def _missing(frame: Any, name: str) -> Any:
    """Match pandas null and floating NaN semantics without casting other scalar types."""
    functions = _functions()
    result = functions.col(name).isNull()
    if frame.schema[name].dataType.typeName() in {"float", "double"}:
        result = result | functions.isnan(name)
    return result


def _drop_missing(frame: Any, params: dict) -> Any:
    """Preserve absolute, percentage and any/all missingness precedence."""
    columns = params["subset"]
    valid = sum((~_missing(frame, name)).cast("int") for name in columns)
    threshold = params.get("threshold")
    if threshold is None and params.get("missing_threshold") is not None:
        threshold = (100.0 - params["missing_threshold"]) * len(columns) / 100.0
    if threshold is None:
        threshold = 1 if params.get("how", "any") == "all" else len(columns)
    return frame.where(valid >= threshold)


def _manual_bounds(frame: Any, params: dict) -> Any:
    """Keep numeric nulls and NaNs while rejecting values outside fixed bounds."""
    functions = _functions()
    for name, bounds in params["bounds"].items():
        if frame.schema[name].dataType.typeName() not in {
            "byte",
            "short",
            "integer",
            "long",
            "float",
            "double",
            "decimal",
        }:
            raise ValueError("pre_split_steps ManualBounds requires numeric source columns.")
        condition = functions.lit(True)
        if bounds.get("lower") is not None:
            condition = condition & (functions.col(name) >= bounds["lower"])
        if bounds.get("upper") is not None:
            condition = condition & (functions.col(name) <= bounds["upper"])
        frame = frame.where(condition | _missing(frame, name))
    return frame


def _deduplicate(frame: Any, params: dict, spec: LocalTrainingSpec) -> Any:
    """Select canonical first/last survivors and reject conflicting target labels."""
    functions = _functions()
    window = importlib.import_module("pyspark.sql").Window
    subset = params["subset"]
    target_hash = functions.udf(lambda value: _row_hash([value]), "string")
    conflicts = frame.groupBy(*subset).agg(
        functions.countDistinct(target_hash(functions.col(spec.target_column))).alias("_labels")
    )
    if conflicts.where("_labels > 1").limit(1).count():
        raise ValueError("pre_split_steps Deduplicate has conflicting target labels.")
    keep = params.get("keep", "first")
    group = window.partitionBy(*subset)
    if keep in (False, "none"):
        return (
            frame.withColumn("_duplicates", functions.count("*").over(group))
            .where("_duplicates = 1")
            .drop("_duplicates")
        )
    ordering = [functions.col(name) for name in _ordering(spec)]
    if keep == "last":
        ordering = [column.desc() for column in ordering]
    return (
        frame.withColumn("_survivor", functions.row_number().over(group.orderBy(*ordering)))
        .where("_survivor = 1")
        .drop("_survivor")
    )


def _ordering(spec: LocalTrainingSpec) -> list[str]:
    """Use precisely the original stable event/key order before sklearn splitting."""
    return ([spec.event_column] if spec.event_column else []) + list(spec.record_key_columns)


def _eligible(frame: Any, spec: LocalTrainingSpec) -> Any:
    """Apply availability and supported row filters before selecting train membership."""
    floating = {
        field.name
        for field in frame.schema.fields
        if field.dataType.typeName() in {"float", "double"}
    }
    frame = _eligible_training_source(frame, spec, floating)
    for step in spec.pre_split_steps:
        if step["transformer"] == "Deduplicate":
            frame = _deduplicate(frame, step["params"], spec)
        elif step["transformer"] == "ManualBounds":
            frame = _manual_bounds(frame, step["params"])
        else:
            frame = _drop_missing(frame, step["params"])
    return frame


def _metadata_train_ordinals(metadata: pd.DataFrame, spec: LocalTrainingSpec) -> list[int]:
    """Run the existing exact split over bounded ordinals and label/group metadata only."""
    metadata = metadata.sort_values("_ordinal").reset_index(drop=True)
    if metadata[spec.target_column].isna().any():
        raise ValueError("Available labels must have nonnull targets.")
    if spec.pre_split_steps and len(metadata) < 4:
        raise ValueError("Training filters leave fewer than four eligible rows.")
    if spec.event_column:
        metadata[spec.event_column] = pd.to_datetime(
            metadata[spec.event_column], unit="us", utc=True
        )
    train, _ = _partition_training_rows(metadata, spec)
    extract_training_weights(train, spec.weight_column)
    return train["_ordinal"].tolist()


def _training_partition(frame: Any, spec: LocalTrainingSpec) -> Any:
    """Join executor-generated exact split membership back to distributed feature values."""
    functions = _functions()
    window = importlib.import_module("pyspark.sql").Window
    frame = frame.withColumn(
        "_ordinal", functions.row_number().over(window.orderBy(*_ordering(spec)))
    )
    columns = list(
        dict.fromkeys(
            name
            for name in (
                "_ordinal",
                spec.target_column,
                spec.event_column,
                spec.group_column,
                spec.weight_column,
            )
            if name is not None
        )
    )

    def select_batches(batches):
        """Collect bounded split metadata on one executor, never a feature population."""
        metadata = pd.concat(list(batches), ignore_index=True)
        if len(metadata) > spec.max_rows:
            raise ValueError("Training source exceeds max_rows.")
        yield pd.DataFrame({"_ordinal": _metadata_train_ordinals(metadata, spec)}, dtype="int64")

    membership = frame.select(*columns).coalesce(1).mapInPandas(select_batches, "_ordinal long")
    return frame.join(membership, "_ordinal", "left_semi").drop("_ordinal")


def _source_size(record: Any, spec: LocalTrainingSpec) -> int:
    """Count exact serialized records and a conservative per-row pandas allocation bound."""
    values = record.asDict(recursive=True)
    for name in (spec.event_column, spec.result_available_at_column):
        if name:
            values[name] = instant_from_microseconds(values[name])
    serialized = len(pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL))
    allocation = int(pd.DataFrame([values]).memory_usage(index=False, deep=True).sum())
    return max(serialized, allocation)


def _pandas_integer_columns(source: Any, spec: LocalTrainingSpec) -> list[str]:
    """Validate scalar transport types and identify integer columns needing inference."""
    integer_types = {"byte", "short", "integer", "long"}
    scalar_types = integer_types | {
        "boolean",
        "float",
        "double",
        "decimal",
        "string",
        "char",
        "varchar",
        "date",
        "timestamp",
        "timestamp_ntz",
        "void",
    }
    if any(field.dataType.typeName() not in scalar_types for field in source.schema.fields):
        raise ValueError("Spark freshness source requires supported scalar pandas column types.")
    dates = {spec.event_column, spec.result_available_at_column}
    return [
        field.name
        for field in source.schema.fields
        if field.dataType.typeName() in integer_types and field.name not in dates
    ]


def _pandas_source_types(source: Any, spec: LocalTrainingSpec) -> Any:
    """Reproduce column-wide pandas inference before any local eligibility filters.

    The trainer constructs pandas from Python records. An integer column with
    any null becomes float64, including its rounding of integers above 2**53.
    Inspect the whole selected source, not an Arrow batch or split partition.
    Normalized time columns stay microseconds until the split metadata bridge.
    """
    functions = _functions()
    integers = _pandas_integer_columns(source, spec)
    if not integers:
        return source
    nulls = source.agg(
        *[functions.max(functions.col(name).isNull().cast("int")).alias(name) for name in integers]
    ).first()
    widen = {name for name in integers if nulls[name]}
    return source.select(
        *[
            functions.col(name).cast("double").alias(name) if name in widen else functions.col(name)
            for name in source.columns
        ]
    )


def _read_source(spark: Any, spec: LocalTrainingSpec) -> Any:
    """Pin a projected Delta read and retain the local training row and byte budgets."""
    functions = _functions()
    source = spark.read.format("delta").option("versionAsOf", spec.version).table(spec.table)
    source = normalize_training_dates(
        source.select(*spec.source_columns),
        event_column=spec.event_column,
        result_column=spec.result_available_at_column,
        event_spec=spec.event_time_parsing,
        result_spec=spec.result_time_parsing,
    )
    if spec.event_column:
        source = source.where(
            (functions.col(spec.event_column) >= instant_microseconds(cast(datetime, spec.start)))
            & (functions.col(spec.event_column) < instant_microseconds(cast(datetime, spec.cutoff)))
        )
    if spec.training_sample_rows is not None:
        source, _ = _sample_training_source(source, spec)
    if source.limit(spec.max_rows + 1).count() > spec.max_rows:
        raise ValueError("Training source exceeds max_rows.")
    require_unique_keys(source, spec.record_key_columns, "training")
    size = functions.udf(lambda row: _source_size(row, spec), "long")
    total = source.agg(
        functions.sum(size(functions.struct(*source.columns))).alias("bytes")
    ).first()
    if int(total["bytes"] or 0) + 132 > spec.max_bytes:
        raise ValueError("Training source exceeds conservative Spark max_bytes bound.")
    return _pandas_source_types(source, spec)


def _counts(frame: Any, columns: list[str]) -> Any:
    """Hash only feature-target values on executors and retain duplicate multiplicity."""
    functions = _functions()
    digest = functions.udf(lambda row: _row_hash(tuple(row)), "string")
    return frame.select(digest(functions.struct(*columns)).alias("hash")).groupBy("hash").count()


def _content_digest(counts: Any, columns: list[str]) -> str:
    """Stream sorted count metadata on one executor to match the local JSON digest exactly."""
    prefix = json.dumps({"columns": columns}, separators=(",", ":"))[:-1] + ',"rows":['

    def digest_batches(batches):
        """Keep a constant-size digest while streaming deterministic hash-count records."""
        digest = hashlib.sha256(prefix.encode())
        delimiter = b""
        for batch in batches:
            for row_hash, count in batch.itertuples(index=False, name=None):
                digest.update(delimiter)
                digest.update(json.dumps([row_hash, int(count)], separators=(",", ":")).encode())
                delimiter = b","
        digest.update(b"]}")
        yield pd.DataFrame({"digest": [digest.hexdigest()]})

    result = (
        counts.select("hash", "count")
        .coalesce(1)
        .sortWithinPartitions("hash")
        .mapInPandas(digest_batches, "digest string")
    )
    return result.first()["digest"]


def _overlay_weights(saved: Any, current: Any, keys: tuple[str, ...], weights: set[str]) -> Any:
    """Replace weights for matched keys, including explicit nulls, without editing old values."""
    functions = _functions()
    selected = sorted(weights.intersection(current.columns))
    latest = current.select(*keys, *selected).withColumn("_weight_match", functions.lit(True))
    joined = saved.alias("old").join(latest.alias("new"), list(keys), "left")
    columns = []
    for name in saved.columns:
        value = functions.col(f"old.{name}")
        if name in selected:
            value = functions.when(
                functions.col("new._weight_match"), functions.col(f"new.{name}")
            ).otherwise(value)
        columns.append(value.alias(name))
    return joined.select(*columns)


def _baseline(
    spark: Any,
    evidence: dict,
    saved: LocalTrainingSpec,
    current: LocalTrainingSpec,
    source: Any,
    columns: list[str],
) -> Any:
    """Union original counts with weight-counterfactual counts using maximum multiplicity."""
    functions = _functions()
    records = evidence["prepared_reference"]
    baseline = _counts(read_reference_population(spark, records["seen"]), columns)
    weights = _filter_weight_columns(saved, current)
    if not weights:
        return baseline
    historical = read_reference_population(spark, records["source"])
    historical = normalize_training_dates(
        historical,
        event_column=saved.event_column,
        result_column=saved.result_available_at_column,
        event_spec=TrainingDateSpec(),
        result_spec=TrainingDateSpec(),
    )
    historical = _pandas_source_types(historical, saved)
    historical = _overlay_weights(historical, source, saved.record_key_columns, weights)
    population = _eligible(historical, _comparison_spec(saved))
    if saved.split_strategy == "temporal":
        population = population.where(
            functions.col(saved.event_column)
            < instant_microseconds(cast(datetime, saved.holdout_start))
        )
    return (
        baseline.unionByName(_counts(population, columns))
        .groupBy("hash")
        .agg(functions.max("count").alias("count"))
    )


def assess_spark_training_data(spark: Any, monitor: Any, workflow: dict, now: datetime) -> dict:
    """Assess exact training novelty without collecting current feature populations."""
    if workflow.get("training_version") is not None:
        raise ValueError(
            "On-drift retraining requires training_version=null for the latest snapshot."
        )
    artifact, saved, _, evidence = load_spark_monitoring_reference(
        spark,
        monitor,
        tracking_uri=workflow.get("tracking_uri", "databricks"),
        registry_uri=workflow.get("registry_uri", "databricks-uc"),
    )
    engine = workflow["engine"]
    if engine != artifact.manifest.fitted_engine:
        raise ValueError("On-drift training engine differs from the active model.")
    identity = table_identity(spark, workflow["training_table"])
    if identity != evidence["prepared_reference_source_table_id"]:
        raise ValueError("Training source physical identity differs from the prepared reference.")
    current = resolve_training_spec(spark, workflow, now)
    _compatible_specs(saved, current)
    current = phase_training_spec(
        training_spec_payload(current, engine),
        workflow.get("pipeline", {}).get("project_python_source"),
    )
    _require_supported_recipe(current)
    _require_supported_recipe(
        replace(saved, sample_key_sha256=None, survivor_key_sha256=None, holdout_key_sha256=None)
    )
    if table_identity(spark, current.table) != identity:
        raise ValueError("Training source table was replaced while resolving its snapshot.")
    source = _read_source(spark, current)
    train = _training_partition(_eligible(source, current), current)
    columns = [*current.input_columns, current.target_column]
    counts = _counts(train, columns)
    result = _assessment(
        counts, _baseline(spark, evidence, saved, current, source, columns), columns
    )
    if table_identity(spark, current.table) != identity:
        raise ValueError("Training source table was replaced during freshness assessment.")
    return result | {
        "source_table": current.table,
        "source_version": current.version,
        "source_table_id": identity,
        "baseline_model_version": evidence["model_version"],
    }


def _assessment(counts: Any, baseline: Any, columns: list[str]) -> dict:
    """Return bounded scalar novelty and training-row evidence only."""
    functions = _functions()
    compared = counts.alias("current").join(baseline.alias("old"), "hash", "left")
    summary = compared.agg(
        functions.sum("current.count").alias("rows"),
        functions.sum(
            functions.greatest(
                functions.col("current.count")
                - functions.coalesce(functions.col("old.count"), functions.lit(0)),
                functions.lit(0),
            )
        ).alias("changed"),
    ).first()
    changed = int(summary["changed"] or 0)
    return {
        "status": "ready" if changed else "no_new_training_data",
        "training_rows": int(summary["rows"] or 0),
        "changed_rows": changed,
        "content_sha256": _content_digest(counts, columns),
    }
