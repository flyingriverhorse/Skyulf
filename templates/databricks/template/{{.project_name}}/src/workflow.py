# Databricks notebook source
"""Run one bounded Skyulf local training, comparison, alias or scoring action."""

import json
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from skyulf.integrations.databricks import (
    InputSource,
    LocalTrainingSpec,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    prepare_local_workflow,
    read_training_snapshot,
    run_incremental_local_batch,
    split_labeled_snapshot,
    train_local_candidate,
)
from skyulf.integrations.databricks.delta import table_identity
from skyulf.integrations.databricks.delta_admission import DeltaTableAdmission
from skyulf.integrations.mlflow.promotion import (
    DeltaAliasAdmission,
    alias_resource_id,
    promote_candidate,
    stage_challenger,
)
from skyulf.integrations.mlflow.registry import resolve_model
from skyulf.integrations.mlflow.validation import compare_registered_local_models

_TABLE_FIELDS = (
    "training_table",
    "score_source_table",
    "prediction_table",
    "score_admission_table",
    "alias_admission_table",
    "model_name",
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){2}\Z")
_OUTPUT_TYPES = {
    "float64": ("double", "DOUBLE"),
    "int64": ("long", "BIGINT"),
    "string": ("string", "STRING"),
    "bool": ("boolean", "BOOLEAN"),
}


@dataclass(frozen=True, slots=True)
class ProvisionResult:
    """Report the UC objects verified or created by one explicit setup run."""

    source_table: str
    prediction_table: str
    prediction_created: bool
    score_admission_table: str
    score_admission_created: bool
    alias_admission_table: str | None
    alias_admission_created: bool


def resolve_target_config(config: dict[str, Any], bindings: dict[str, str]) -> dict[str, Any]:
    """Bind only UC object names to one validated Bundle target."""
    for name in ("catalog", "input_schema", "output_schema", "metadata_schema"):
        if not _IDENTIFIER.fullmatch(bindings.get(name, "")):
            raise ValueError(f"Invalid {name} for a Unity Catalog identifier.")
    suffix = bindings.get("resource_suffix", "")
    if suffix and (not suffix.startswith("_") or not _IDENTIFIER.fullmatch(suffix)):
        raise ValueError("Invalid resource_suffix for a Unity Catalog identifier.")
    resolved = config.copy()
    for name in _TABLE_FIELDS:
        value = config[name]
        if name == "alias_admission_table" and value is None:
            continue
        if type(value) is not str:
            raise ValueError(f"{name} must be a string.")
        for key, replacement in bindings.items():
            value = value.replace("{" + key + "}", replacement)
        if not _TABLE_NAME.fullmatch(value):
            raise ValueError(f"{name} must resolve to a three-part UC name.")
        if name in {
            "prediction_table",
            "score_admission_table",
            "alias_admission_table",
            "model_name",
        }:
            schema = "output_schema" if name == "prediction_table" else "metadata_schema"
            expected = f"{bindings['catalog']}.{bindings[schema]}."
            if not value.startswith(expected):
                raise ValueError(f"{name} must use the active target's {schema}.")
            if suffix and not value.endswith(suffix):
                raise ValueError(f"{name} must include the active target's resource_suffix.")
        resolved[name] = value
    return resolved


def _scoring_config(config: dict[str, Any]) -> LocalWorkflowConfig:
    """Bind the same pinned local model and source for setup and scoring."""
    return LocalWorkflowConfig(
        runtime="databricks",
        engine=config["engine"],
        source=InputSource(
            kind="uc_table",
            table=config["score_source_table"],
            read_mode="incremental",
            max_rows=config["max_rows"],
            max_bytes=config["max_bytes"],
        ),
        model=ModelSelection(
            kind="local_pipeline",
            name=config["model_name"],
            version=config["model_version"],
            tracking_uri=config.get("tracking_uri", "databricks"),
            registry_uri=config.get("registry_uri", "databricks-uc"),
        ),
        sink=OutputSink(kind="uc_delta", table=config["prediction_table"]),
    )


def _prediction_columns(
    config: dict[str, Any], prepared: Any, source: Any
) -> tuple[tuple[str, str, str], ...]:
    """Derive an exact Delta target schema from source keys and saved model outputs."""
    keys = tuple(config["row_keys"])
    inputs = tuple(prepared.artifact.manifest.input_columns)
    if not keys or tuple(config["input_columns"]) != inputs:
        raise ValueError("Configured keys or model inputs differ from the fitted model.")
    names = (*keys, *inputs)
    if any(not _IDENTIFIER.fullmatch(name) for name in names):
        raise ValueError("Source keys and model inputs need simple column identifiers.")
    if len({name.lower() for name in names}) != len(names):
        raise ValueError("Source keys and model inputs must be distinct.")
    if not set(names).issubset(source.columns):
        raise ValueError("Scoring source lacks a row key or fitted model input column.")
    columns = []
    for key in keys:
        kind = source.schema[key].dataType.typeName()
        if kind not in {"string", "long"}:
            raise ValueError(f"Source row key {key!r} must be STRING or BIGINT.")
        columns.append((key, kind, "STRING" if kind == "string" else "BIGINT"))
    for output in prepared.preflight.output_schema:
        if not _IDENTIFIER.fullmatch(output.name) or output.dtype not in _OUTPUT_TYPES:
            raise ValueError("Saved model output has an unsupported name or type.")
        kind, sql_type = _OUTPUT_TYPES[output.dtype]
        columns.append((output.name, kind, sql_type))
    columns.extend((name, "string", "STRING") for name in ("run_id", "model_name", "model_version"))
    if len({name.lower() for name, _, _ in columns}) != len(columns):
        raise ValueError("Prediction columns collide with keys or metadata.")
    return tuple(columns)


def _check_existing_table(spark: Any, name: str, columns: tuple[tuple[str, str, str], ...]) -> None:
    """Reject an existing prediction target with a different schema."""
    actual = {field.name: field.dataType.typeName() for field in spark.table(name).schema.fields}
    expected = {column: kind for column, kind, _ in columns}
    if actual != expected:
        raise ValueError(f"Existing prediction table {name} differs from the model output schema.")


def _check_existing_control(spark: Any, name: str, target_id: str) -> None:
    """Require exactly one idle admission row bound to the intended resource."""
    frame = spark.table(name)
    actual = {field.name: field.dataType.typeName() for field in frame.schema.fields}
    if actual != {"target_id": "string", "owner": "string"}:
        raise ValueError(f"Existing admission table {name} has an invalid schema.")
    rows = frame.limit(2).collect()
    if len(rows) != 1 or rows[0]["target_id"] != target_id or rows[0]["owner"] is not None:
        raise ValueError(f"Existing admission table {name} is not idle for this resource.")


def _create_control(spark: Any, name: str, target_id: str) -> None:
    """Create an admission table and its single row in one Delta CTAS statement."""
    spark.sql(
        f"CREATE TABLE {name} USING DELTA AS "
        "SELECT :target_id AS target_id, CAST(NULL AS STRING) AS owner",
        args={"target_id": target_id},
    ).collect()


def provision_local_tables(spark: Any, config: dict[str, Any], prepared: Any) -> ProvisionResult:
    """Preflight an existing source/model, then create only missing output state."""
    source_name = config["score_source_table"]
    training_name = config["training_table"]
    target_name = config["prediction_table"]
    control_name = config["score_admission_table"]
    lifecycle = config.get("include_lifecycle", "no")
    if lifecycle not in {"no", "yes"}:
        raise ValueError("include_lifecycle must be 'yes' or 'no'.")
    for name in (source_name, training_name, target_name, control_name, config["model_name"]):
        if not _TABLE_NAME.fullmatch(name):
            raise ValueError("Setup needs valid three-part Unity Catalog names.")
    if lifecycle == "yes" and not config.get("alias_admission_table"):
        raise ValueError("Lifecycle setup needs an alias admission table.")
    if lifecycle == "yes" and not _TABLE_NAME.fullmatch(config["alias_admission_table"]):
        raise ValueError("Lifecycle setup needs a valid alias admission table name.")
    if source_name == target_name:
        raise ValueError("Prediction output must differ from the source table.")
    for name in {source_name, training_name}:
        if not spark.catalog.tableExists(name):
            raise ValueError(f"Existing input source table is missing: {name}.")
    if not prepared.preflight.ready:
        raise ValueError("The pinned registered model failed Skyulf preflight.")
    source = spark.table(source_name)
    columns = _prediction_columns(config, prepared, source)
    if (
        source.select(*config["row_keys"], *config["input_columns"])
        .limit(config["max_rows"] + 1)
        .count()
        > config["max_rows"]
    ):
        raise ValueError("Initial scoring source exceeds the configured max_rows budget.")
    detail = spark.sql(f"DESCRIBE DETAIL {source_name}").first()
    properties = detail["properties"] or {}
    if not any(
        key.lower() == "delta.enablechangedatafeed" and str(value).lower() == "true"
        for key, value in properties.items()
    ):
        raise ValueError("Scoring source must have Delta Change Data Feed enabled.")
    target_exists = spark.catalog.tableExists(target_name)
    control_exists = spark.catalog.tableExists(control_name)
    if control_exists and not target_exists:
        raise ValueError("Admission table exists while prediction target is missing.")
    if target_exists:
        _check_existing_table(spark, target_name, columns)
        if control_exists:
            _check_existing_control(spark, control_name, table_identity(spark, target_name))
    alias_name = config.get("alias_admission_table") if lifecycle == "yes" else None
    alias_exists = bool(alias_name and spark.catalog.tableExists(alias_name))
    if alias_exists:
        _check_existing_control(spark, alias_name, alias_resource_id(config["model_name"]))
    if not target_exists:
        definition = ", ".join(f"{name} {sql_type}" for name, _, sql_type in columns)
        spark.sql(f"CREATE TABLE {target_name} ({definition}) USING DELTA").collect()
    if not control_exists:
        _create_control(spark, control_name, table_identity(spark, target_name))
    if alias_name and not alias_exists:
        _create_control(spark, alias_name, alias_resource_id(config["model_name"]))
    return ProvisionResult(
        source_name,
        target_name,
        not target_exists,
        control_name,
        not control_exists,
        alias_name,
        bool(alias_name and not alias_exists),
    )


def _training_spec(config: dict[str, Any]) -> LocalTrainingSpec:
    """Keep the evaluation split and source snapshot identical across actions."""
    return LocalTrainingSpec(
        table=config["training_table"],
        version=config["training_version"],
        start=datetime.fromisoformat(config["start"]),
        holdout_start=datetime.fromisoformat(config["holdout_start"]),
        cutoff=datetime.fromisoformat(config["cutoff"]),
        event_column=config["event_column"],
        label_time_column=config["label_time_column"],
        row_keys=tuple(config["row_keys"]),
        input_columns=tuple(config["input_columns"]),
        target_column=config["target_column"],
        max_rows=config["max_rows"],
        max_bytes=config["max_bytes"],
    )


def _comparison(spark: Any, config: dict[str, Any]) -> tuple[Any, Any]:
    """Evaluate two pinned registry versions on one pinned held-out frame."""
    spec = _training_spec(config)
    frame = read_training_snapshot(spark, spec)
    _, heldout, _ = split_labeled_snapshot(frame, spec)
    if config["engine"] == "polars":
        heldout = pl.from_pandas(heldout)
    model_name = config["model_name"]
    tracking_uri = config.get("tracking_uri", "databricks")
    registry_uri = config.get("registry_uri", "databricks-uc")
    candidate = resolve_model(
        model_name,
        version=config["candidate_version"],
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    champion = resolve_model(
        model_name,
        version=config["champion_version"],
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    report = compare_registered_local_models(
        candidate,
        champion,
        heldout,
        target_column=spec.target_column,
        dataset_id=spec.dataset_id,
        metric=config["metric"],
        min_improvement=config["min_improvement"],
        max_rows=spec.max_rows,
        max_bytes=spec.max_bytes,
        quality_threshold=config.get("quality_threshold"),
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    return report, heldout


def run_action(
    spark: Any,
    config: dict[str, Any],
    action: str,
    *,
    experiment_name: str | None = None,
    artifact_path: str | Path | None = None,
) -> Any:
    """Delegate each explicit job action to the corresponding Skyulf service."""
    tracking_uri = config.get("tracking_uri", "databricks")
    registry_uri = config.get("registry_uri", "databricks-uc")
    if action == "train":
        if experiment_name is None or artifact_path is None:
            raise ValueError("Training needs an experiment and temporary artifact path.")
        return train_local_candidate(
            spark,
            _training_spec(config),
            config["pipeline"],
            model_name=config["model_name"],
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            experiment_name=experiment_name,
            run_name="candidate_training",
            artifact_path=artifact_path,
            metric=config["metric"],
            min_improvement=config["min_improvement"],
            engine=config["engine"],
            champion_version=config.get("champion_version"),
            quality_threshold=config.get("quality_threshold"),
        )
    if action in {"setup", "score"}:
        prepared = prepare_local_workflow(_scoring_config(config))
        if action == "setup":
            return provision_local_tables(spark, config, prepared)
        return run_incremental_local_batch(
            spark,
            prepared,
            row_keys=tuple(config["row_keys"]),
            admission=DeltaTableAdmission(spark, config["score_admission_table"]),
        )
    if action in {"compare", "stage", "promote"}:
        if action != "compare" and config["alias_admission_table"] is None:
            raise ValueError("Alias control is not configured; enable the lifecycle jobs first.")
        report, heldout = _comparison(spark, config)
        if action == "compare":
            return report
        if not report.eligible:
            raise ValueError(f"Candidate is not eligible for {action}: {report.reason}.")
        options = {
            "target_column": config["target_column"],
            "expected_champion_version": config["champion_version"],
            "admission": DeltaAliasAdmission(spark, config["alias_admission_table"]),
            "max_rows": config["max_rows"],
            "max_bytes": config["max_bytes"],
            "tracking_uri": tracking_uri,
            "registry_uri": registry_uri,
        }
        if action == "stage":
            return stage_challenger(
                report,
                heldout,
                expected_challenger_version=config.get("expected_challenger_version"),
                **options,
            )
        return promote_candidate(report, heldout, **options)
    raise ValueError(f"Unknown workflow action: {action}.")


def main() -> None:
    """Read deployed configuration and job widgets only at the notebook boundary."""
    widgets = globals()["dbutils"].widgets
    config = json.loads(Path(widgets.get("config_path")).read_text(encoding="utf-8"))
    config = resolve_target_config(
        config,
        {
            name: widgets.get(name)
            for name in (
                "catalog",
                "input_schema",
                "output_schema",
                "metadata_schema",
                "resource_suffix",
            )
        },
    )
    action = widgets.get("action")
    with tempfile.TemporaryDirectory(prefix="skyulf-bundle-") as directory:
        result = run_action(
            globals()["spark"],
            config,
            action,
            experiment_name=widgets.get("experiment_name") if action == "train" else None,
            artifact_path=Path(directory) / "artifact" if action == "train" else None,
        )
    output = json.dumps(asdict(result), default=str, allow_nan=False)
    print(output)
    globals()["dbutils"].notebook.exit(output)


if __name__ == "__main__":
    main()
