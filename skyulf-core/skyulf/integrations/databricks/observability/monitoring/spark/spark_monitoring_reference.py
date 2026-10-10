"""Prepare version-bound references once and read immutable distributed snapshots."""

import importlib
import json
from dataclasses import replace
from typing import Any
from uuid import uuid4

import pandas as pd

from .....mlflow.shared._client import make_registry_client, require_mlflow
from ....data.delta_io.delta import table_identity
from ....jobs.shared.notebook_diagnostics import notebook_task
from ....shared._contracts import table_name
from ....training.fitting.candidate import read_training_snapshot, split_labeled_snapshot
from ....training.shared.training_evidence import validate_training_evidence
from ..monitoring_config import MonitorConfig, json_digest, qualified_name
from ..monitoring_performance import measure_holdout_values, performance_contract
from ..monitoring_reference import load_monitoring_artifact, reference_document
from ..monitoring_source_evidence import validate_source_evidence
from ..monitoring_sources import read_snapshot
from ..monitoring_store import OWNER, PROPERTY, ensure_owned_object


def reference_id(config: MonitorConfig, evidence: dict) -> str:
    """Bind prepared populations to immutable model and verified training identities."""
    reference_config = replace(config, serving_endpoint=None) if config.serving_endpoint else config
    return json_digest(
        {"monitor_id": reference_config.monitor_id, "evidence": evidence, "format": 1}
    )


def _reference_name(config: MonitorConfig, evidence: dict, suffix: str) -> str:
    """Keep artifact-specific tables separate from mutable monitoring observations."""
    if config.reference_namespace is None:
        raise ValueError("Spark monitoring requires reference_namespace.")
    return qualified_name(
        f"{config.reference_namespace}.monitor_ref_{reference_id(config, evidence)}_{suffix}"
    )


def _snapshot_identity(spark: Any, name: str) -> dict:
    """Pin the first atomic population write rather than trusting mutable table contents."""
    return {"name": name, "id": table_identity(spark, name), "version": 0}


def _write_population(spark: Any, name: str, frame: Any) -> dict:
    """Create an owned append-only reference atomically; never overwrite existing data."""
    if ensure_owned_object(spark, name):
        saved = read_snapshot(spark, name, 0)
        if saved.schema != frame.schema or (
            saved.exceptAll(frame).limit(1).count() or frame.exceptAll(saved).limit(1).count()
        ):
            raise ValueError(f"Incomplete prepared reference differs from verified data: {name}.")
        return _snapshot_identity(spark, name)
    view = f"skyulf_reference_{uuid4().hex}"
    frame.createOrReplaceTempView(view)
    try:
        # Table identifiers are validated/quoted, properties are constants, and
        # the temporary view suffix is generated UUID hex rather than input text.
        spark.sql(
            f"CREATE TABLE {table_name(name)} USING DELTA "  # nosec B608
            f"TBLPROPERTIES ('{PROPERTY}' = '{OWNER}', 'delta.appendOnly' = 'true') "
            f"AS SELECT * FROM {view}"
        )
    finally:
        spark.catalog.dropTempView(view)
    return _snapshot_identity(spark, name)


def read_reference_population(spark: Any, record: dict) -> Any:
    """Reject replaced reference tables and read only the recorded immutable Delta version."""
    name = qualified_name(record["name"])
    if not ensure_owned_object(spark, name) or table_identity(spark, name) != record["id"]:
        raise ValueError("Prepared monitoring reference table identity differs.")
    return read_snapshot(spark, name, record["version"])


def read_reference_metadata(spark: Any, config: MonitorConfig, evidence: dict) -> dict:
    """Require explicit preparation for old models instead of quietly replaying training."""
    name = _reference_name(config, evidence, "receipt")
    if not ensure_owned_object(spark, name):
        raise ValueError(
            "Spark reference is not prepared; run prepare_spark_monitoring_reference first."
        )
    rows = read_snapshot(spark, name, 0).limit(2).collect()
    if len(rows) != 1:
        raise ValueError("Prepared monitoring reference receipt is invalid.")
    receipt = json.loads(rows[0]["payload"])
    if rows[0]["digest"] != json_digest(receipt):
        raise ValueError("Prepared monitoring reference receipt digest differs.")
    return receipt


def _prepared_holdout(artifact: Any, spec: Any, config: MonitorConfig, receipt: dict) -> dict:
    """Adapt cached exact holdout metrics to the selected policy without replaying inference."""
    policy = config.performance_policy
    if not policy or policy["mode"] == "off":
        return {}
    values = receipt["holdout"]
    evidence = receipt["evidence"]
    return {
        "performance_baseline": {
            "model_version": evidence["model_version"],
            "metric": policy["metric"],
            "contract_digest": performance_contract(artifact, spec, config),
            "value": values["values"].get(policy["metric"]),
            "labeled_rows": values["labeled_rows"],
            "label_coverage": values["label_coverage"],
            "reference": f"runs:/{evidence['training_run_id']}/training_filter_evidence.json",
            "kind": "training_holdout",
        }
    }


def load_spark_monitoring_reference(
    spark: Any, config: MonitorConfig, *, tracking_uri: str | None, registry_uri: str | None
) -> tuple[Any, Any, Any, dict]:
    """Read prepared train features and metric summaries without local population reads."""
    artifact, spec, _, evidence = load_monitoring_artifact(
        config, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    receipt = read_reference_metadata(spark, config, evidence)
    if receipt["evidence"] != evidence:
        raise ValueError("Prepared monitoring reference model/training identity differs.")
    train = read_reference_population(spark, receipt["tables"]["train"])
    baseline = _prepared_holdout(artifact, spec, config, receipt)
    return (
        artifact,
        spec,
        train,
        evidence
        | baseline
        | {
            "prepared_reference": receipt["tables"],
            "prepared_reference_source_table_id": receipt.get("source_table_id"),
        },
    )


def _prepared_frame(spark: Any, frame: pd.DataFrame, source_schema: Any, spec: Any) -> Any:
    """Retain transport types and recover typed null columns from the pinned source."""
    arrow = importlib.import_module("pyarrow")
    conversion = importlib.import_module("pyspark.sql.pandas.types")
    types = importlib.import_module("pyspark.sql.types")
    schema = conversion.from_arrow_schema(arrow.Schema.from_pandas(frame, preserve_index=False))
    dates = {spec.event_column, spec.result_available_at_column}
    for field in schema.fields:
        if isinstance(field.dataType, types.NullType):
            field.dataType = (
                types.TimestampType() if field.name in dates else source_schema[field.name].dataType
            )
    return spark.createDataFrame(frame, schema=schema)


def prepare_spark_monitoring_reference(
    spark: Any,
    config: MonitorConfig,
    *,
    tracking_uri: str | None = "databricks",
    registry_uri: str | None = "databricks-uc",
) -> dict:
    """Prepare a bounded local-trained model once using its training, not monitoring, budget.

    This explicit activation/migration operation verifies the exact original split.
    Ordinary Spark observations never call it. Training remains a bounded local
    operation; raising a monitoring row cap is neither required nor permitted.
    """
    identity = f"{config.model_name}/{config.model_version or config.model_alias}"
    with notebook_task(f"reference.load_model {identity}", None):
        artifact, spec, filters, evidence = load_monitoring_artifact(
            config, tracking_uri=tracking_uri, registry_uri=registry_uri
        )
    name = _reference_name(config, evidence, "receipt")
    if ensure_owned_object(spark, name):
        with notebook_task(f"reference.reuse {identity}", None):
            receipt = read_reference_metadata(spark, config, evidence)
            if receipt["evidence"] != evidence:
                raise ValueError("Prepared monitoring reference model/training identity differs.")
            for record in receipt["tables"].values():
                read_reference_population(spark, record)
        return receipt
    with notebook_task(f"reference.read_training {identity}", None):
        source_id = table_identity(spark, spec.table)
        source = read_training_snapshot(spark, spec)
        if table_identity(spark, spec.table) != source_id:
            raise ValueError("Training source was replaced while preparing monitoring reference.")
        client = make_registry_client(require_mlflow(), tracking_uri, registry_uri)
        source_receipt = reference_document(
            client, evidence["training_run_id"], "monitoring_source_evidence.json"
        )
        validate_source_evidence(source_receipt, source, spec.source_columns, spec.dataset_id)
        train, holdout, _ = split_labeled_snapshot(
            source, spec, engine=artifact.manifest.fitted_engine
        )
        validate_training_evidence(
            filters,
            spec,
            project_source_sha256=artifact.manifest.project_source_sha256,
            heldout=holdout,
        )
    columns = [*spec.input_columns, spec.target_column]
    old_population = (
        train[columns]
        if spec.split_strategy == "temporal"
        else pd.concat([train[columns], holdout[columns]], ignore_index=True)
    )
    frames = {"train": train[columns], "source": source, "seen": old_population}
    source_schema = read_snapshot(spark, spec.table, spec.version).schema
    tables = {}
    for key, frame in frames.items():
        with notebook_task(f"reference.write_{key} {identity}", None):
            tables[key] = _write_population(
                spark,
                _reference_name(config, evidence, key),
                _prepared_frame(spark, frame, source_schema, spec),
            )
    with notebook_task(f"reference.measure_holdout {identity}", None):
        holdout_values = measure_holdout_values(artifact, spec, holdout)
    receipt = {
        "format": 1,
        "evidence": evidence,
        "tables": tables,
        "source_table_id": source_id,
        "holdout": holdout_values,
    }
    payload = json.dumps(receipt, sort_keys=True, allow_nan=False)
    with notebook_task(f"reference.write_receipt {identity}", None):
        _write_population(
            spark,
            name,
            spark.createDataFrame(
                [(payload, json_digest(receipt))], "payload STRING, digest STRING"
            ),
        )
    return receipt
