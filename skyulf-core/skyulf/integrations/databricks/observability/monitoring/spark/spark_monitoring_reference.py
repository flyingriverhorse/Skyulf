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
from ....shared._contracts import column_name, table_name
from ....training.fitting.candidate import read_training_snapshot, split_labeled_snapshot
from ....training.shared.training_evidence import validate_training_evidence
from ..monitoring_config import MonitorConfig, json_digest, qualified_name
from ..monitoring_details import observation_identity
from ..monitoring_performance import measure_holdout_values, performance_contract
from ..monitoring_reference import load_monitoring_artifact, reference_document
from ..monitoring_source_evidence import validate_source_evidence
from ..monitoring_sources import read_snapshot
from ..monitoring_store import OWNER, PROPERTY, ensure_monitoring_store, ensure_owned_object
from ..reference_admission import (
    hold_reference_mutation,
    reference_mutation_holder,
    require_reference_mutation,
)
from ..reference_cleanup import immutable_detail
from ..reference_layout import (
    combine_reference_population,
    reference_population_records,
    shared_source_table_name,
    source_snapshot_identity,
)


def reference_id(config: MonitorConfig, evidence: dict) -> str:
    """Bind prepared populations to immutable model and verified training identities."""
    reference_config = replace(config, serving_endpoint=None) if config.serving_endpoint else config
    return json_digest(
        {
            "monitor_id": reference_config.monitor_id,
            "evidence": observation_identity(evidence),
            "format": 1,
        }
    )


def reference_table_name(config: MonitorConfig, evidence: dict, suffix: str) -> str:
    """Keep artifact-specific tables separate from mutable monitoring observations."""
    if config.reference_namespace is None:
        raise ValueError("Spark monitoring requires reference_namespace.")
    return qualified_name(
        f"{config.reference_namespace}.monitor_ref_{reference_id(config, evidence)}_{suffix}"
    )


def _snapshot_identity(spark: Any, name: str) -> dict:
    """Pin the first atomic population write rather than trusting mutable table contents."""
    return {"name": name, "id": str(immutable_detail(spark, name)["id"]), "version": 0}


def _write_population(spark: Any, name: str, frame: Any) -> dict:
    """Create an owned append-only reference atomically; never overwrite existing data."""
    if not ensure_owned_object(spark, name):
        _create_population(spark, name, frame)
    record = _snapshot_identity(spark, name)
    saved = read_snapshot(spark, name, 0)
    if saved.schema != frame.schema or (
        saved.exceptAll(frame).limit(1).count() or frame.exceptAll(saved).limit(1).count()
    ):
        raise ValueError(f"Incomplete prepared reference differs from verified data: {name}.")
    immutable_detail(spark, name, record["id"])
    return record


def _create_population(spark: Any, name: str, frame: Any) -> None:
    """Allow one atomic creator; the caller verifies even a concurrent winner's contents."""
    view = f"skyulf_reference_{uuid4().hex}"
    frame.createOrReplaceTempView(view)
    try:
        # Table identifiers are validated/quoted, properties are constants, and
        # the temporary view suffix is generated UUID hex rather than input text.
        spark.sql(
            f"CREATE TABLE IF NOT EXISTS {table_name(name)} USING DELTA "  # nosec B608
            f"TBLPROPERTIES ('{PROPERTY}' = '{OWNER}', 'delta.appendOnly' = 'true') "
            f"AS SELECT * FROM {view}"
        )
    finally:
        spark.catalog.dropTempView(view)


def read_reference_population(spark: Any, record: dict) -> Any:
    """Reject replaced reference tables and read only the recorded immutable Delta version."""
    name = qualified_name(record["name"])
    if record["version"] != 0:
        raise ValueError("Prepared monitoring reference records a mutable version.")
    immutable_detail(spark, name, record["id"])
    frame = read_snapshot(spark, name, record["version"])
    membership = record.get("membership_column")
    if membership is not None:
        if frame.schema[membership].dataType.simpleString() != "boolean":
            raise ValueError("Prepared monitoring reference membership is not boolean.")
        if record.get("train_only") is True:
            frame = frame.where(f"{column_name(membership)} = true")
        frame = frame.drop(membership)
    immutable_detail(spark, name, record["id"])
    return frame


def read_reference_metadata(spark: Any, config: MonitorConfig, evidence: dict) -> dict:
    """Require explicit preparation for old models instead of quietly replaying training."""
    name = reference_table_name(config, evidence, "receipt")
    if not ensure_owned_object(spark, name):
        raise ValueError(
            "Spark reference is not prepared; run prepare_spark_monitoring_reference first."
        )
    receipt_id = str(immutable_detail(spark, name)["id"])
    rows = read_snapshot(spark, name, 0).limit(2).collect()
    if len(rows) != 1:
        raise ValueError("Prepared monitoring reference receipt is invalid.")
    receipt = json.loads(rows[0]["payload"])
    if rows[0]["digest"] != json_digest(receipt):
        raise ValueError("Prepared monitoring reference receipt digest differs.")
    immutable_detail(spark, name, receipt_id)
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
    if observation_identity(receipt["evidence"]) != observation_identity(evidence):
        raise ValueError("Prepared monitoring reference model/training identity differs.")
    records = reference_population_records(receipt)
    train = read_reference_population(spark, records["train"])
    baseline = _prepared_holdout(artifact, spec, config, receipt)
    return (
        artifact,
        spec,
        train,
        evidence
        | baseline
        | {
            "prepared_reference": records,
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
    reference_holder: dict | None = None,
) -> dict:
    """Prepare a bounded local-trained model once using its training, not monitoring, budget.

    This explicit activation/migration operation verifies the exact original split.
    Ordinary Spark observations never call it. Training remains a bounded local
    operation; raising a monitoring row cap is neither required nor permitted.
    Expensive reconstruction precedes the shared publication admission. Generated jobs
    pass their validated job run, task and workspace via ``reference_holder`` for recovery.
    """
    if config.reference_namespace is None:
        raise ValueError("Spark monitoring requires reference_namespace.")
    holder = reference_mutation_holder(reference_holder or {})
    catalog, schema = config.reference_namespace.split(".")
    namespace = ensure_monitoring_store(spark, catalog, schema)
    identity = f"{config.model_name}/{config.model_version or config.model_alias}"
    with notebook_task(f"reference.load_model {identity}", None):
        artifact, spec, filters, evidence = load_monitoring_artifact(
            config, tracking_uri=tracking_uri, registry_uri=registry_uri
        )
    name = reference_table_name(config, evidence, "receipt")
    if ensure_owned_object(spark, name):
        with hold_reference_mutation(spark, namespace, **holder):
            receipt = _reuse_reference(spark, config, evidence, identity)
            if receipt is not None:
                return receipt
    receipt, frames = _prepare_reference(
        spark, artifact, spec, filters, evidence, identity, tracking_uri, registry_uri
    )
    with hold_reference_mutation(spark, namespace, **holder) as owner:
        reused = _reuse_reference(spark, config, evidence, identity)
        if reused is not None:
            return reused
        _verify_source_identity(spark, spec, receipt)
        return _publish_reference(spark, config, spec, receipt, frames, identity, owner)


def _reuse_reference(spark, config, evidence, identity) -> dict | None:
    """Read any winner's completed receipt only while deletion is excluded."""
    if not ensure_owned_object(spark, reference_table_name(config, evidence, "receipt")):
        return None
    with notebook_task(f"reference.reuse {identity}", None):
        receipt = read_reference_metadata(spark, config, evidence)
        if observation_identity(receipt["evidence"]) != observation_identity(evidence):
            raise ValueError("Prepared monitoring reference model/training identity differs.")
        for record in reference_population_records(receipt).values():
            read_reference_population(spark, record)
    return receipt


def _prepare_reference(
    spark, artifact, spec, filters, evidence, identity, tracking_uri, registry_uri
):
    """Reconstruct and validate bounded local populations without taking the store admission."""
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
    population, membership = combine_reference_population(
        train, holdout, columns, spec.split_strategy
    )
    from ....data.training.training_labels import join_training_labels  # noqa: PLC0415

    source_schema = join_training_labels(
        spark, read_snapshot(spark, spec.table, spec.version), spec
    ).schema
    source_columns: list[str] = source_receipt["columns"]
    prepared_source = _prepared_frame(spark, source.loc[:, source_columns], source_schema, spec)
    source_identity = source_snapshot_identity(
        source_id, spec.version, source_receipt, prepared_source.schema.jsonValue()
    )
    frames = {
        "population": _prepared_frame(spark, population, source_schema, spec),
        "source": prepared_source,
    }
    with notebook_task(f"reference.measure_holdout {identity}", None):
        holdout_values = measure_holdout_values(artifact, spec, holdout)
    receipt = {
        "format": 2,
        "evidence": evidence,
        "membership_column": membership,
        "source_table_id": source_id,
        "source_identity": source_identity,
        "holdout": holdout_values,
    }
    return receipt, frames


def _verify_source_identity(spark, spec, receipt) -> None:
    """Rebind reconstructed rows to the same physical source and pinned immutable version."""
    if table_identity(spark, spec.table) != receipt["source_table_id"] or (
        spec.version != receipt["source_identity"]["version"]
    ):
        raise ValueError("Training source was replaced while preparing monitoring reference.")


def _publish_reference(spark, config, spec, receipt, frames, identity, owner) -> dict:
    """Atomically create children and then the receipt while shared deletion is excluded."""
    evidence = receipt["evidence"]
    names = {
        "population": reference_table_name(config, evidence, "population"),
        "source": shared_source_table_name(
            str(config.reference_namespace), receipt["source_identity"]
        ),
    }
    tables = {}
    for key, frame in frames.items():
        with notebook_task(f"reference.write_{key} {identity}", None):
            require_reference_mutation(spark, config.reference_namespace, owner)
            tables[key] = _write_population(
                spark,
                names[key],
                frame,
            )
    _verify_source_identity(spark, spec, receipt)
    receipt = receipt | {"tables": tables}
    payload = json.dumps(receipt, sort_keys=True, allow_nan=False)
    with notebook_task(f"reference.write_receipt {identity}", None):
        require_reference_mutation(spark, config.reference_namespace, owner)
        _write_population(
            spark,
            reference_table_name(config, evidence, "receipt"),
            spark.createDataFrame(
                [(payload, json_digest(receipt))], "payload STRING, digest STRING"
            ),
        )
    return receipt
