# Databricks notebook source
"""Run one bounded Skyulf local training or incremental scoring action."""

import json
import re
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skyulf.integrations.databricks import (
    InputSource,
    LocalTrainingSpec,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    prepare_local_workflow,
    run_incremental_local_batch,
    train_local_candidate,
)
from skyulf.integrations.databricks.admission import SingleWriterAdmission
from skyulf.integrations.mlflow.registry import RegistryModelNotFoundError, resolve_model

_TABLE_FIELDS = (
    "training_table",
    "score_source_table",
    "prediction_table",
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
        if type(value) is not str:
            raise ValueError(f"{name} must be a string.")
        for key, replacement in bindings.items():
            value = value.replace("{" + key + "}", replacement)
        if not _TABLE_NAME.fullmatch(value):
            raise ValueError(f"{name} must resolve to a three-part UC name.")
        if name in {"prediction_table", "model_name"}:
            schema = "output_schema" if name == "prediction_table" else "metadata_schema"
            expected = f"{bindings['catalog']}.{bindings[schema]}."
            if not value.startswith(expected):
                raise ValueError(f"{name} must use the active target's {schema}.")
            if suffix and not value.endswith(suffix):
                raise ValueError(f"{name} must include the active target's resource_suffix.")
        resolved[name] = value
    return resolved


def _scoring_config(config: dict[str, Any]) -> LocalWorkflowConfig:
    """Bind the pinned local model and source for incremental scoring."""
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


def _scoring_target(config: dict[str, Any]) -> str:
    """Choose a stable append target or a physical model-version generation."""
    mode = config.get("model_change_mode", "incremental_append")
    if type(mode) is not str or mode not in {"incremental_append", "full_rebuild"}:
        raise ValueError("model_change_mode must be incremental_append or full_rebuild.")
    base = config["prediction_table"]
    version = config["model_version"]
    if not _TABLE_NAME.fullmatch(base):
        raise ValueError("prediction_table must be a three-part UC name.")
    if (
        type(version) is not str
        or not version.isascii()
        or not version.isdecimal()
        or version[0] == "0"
    ):
        raise ValueError("model_version must be a concrete positive version.")
    if mode == "full_rebuild":
        return f"{base}_v{version}"
    return base


def _managed_prediction_view_exists(spark: Any, logical: str) -> bool:
    """Reject a table or unrelated view before creating prediction resources."""
    if not spark.catalog.tableExists(logical):
        return False
    if spark.catalog.getTable(logical).tableType.upper() != "VIEW":
        raise ValueError("Full rebuild needs a view name, but a physical table already uses it.")
    marker = spark.sql(f"SHOW TBLPROPERTIES {logical} ('skyulf.mode')").first()
    if marker is None or marker["value"] != "full_rebuild":
        raise ValueError("Existing prediction view is not managed by Skyulf full rebuild.")
    return True


def _activate_prediction_view(spark: Any, logical: str, generation: str) -> None:
    """Expose a complete generation without replacing prior prediction tables."""
    if (
        not _TABLE_NAME.fullmatch(logical)
        or not _TABLE_NAME.fullmatch(generation)
        or not re.fullmatch(re.escape(logical) + r"_v[1-9][0-9]*", generation)
    ):
        raise ValueError("Prediction view and generation names must match one UC target.")
    physical = spark.table(generation)
    if not physical.limit(1).count():
        raise ValueError("A full-rebuild generation must contain predictions before activation.")
    if not _managed_prediction_view_exists(spark, logical):
        spark.sql(
            f"CREATE VIEW {logical} TBLPROPERTIES ('skyulf.mode' = 'full_rebuild') "
            f"AS SELECT * FROM {generation}"
        ).collect()
        return
    active_columns = tuple(
        (field.name, field.dataType.typeName()) for field in spark.table(logical).schema.fields
    )
    candidate_columns = tuple(
        (field.name, field.dataType.typeName()) for field in physical.schema.fields
    )
    if active_columns != candidate_columns:
        raise ValueError("New prediction generation differs from the active view schema.")
    definition = spark.sql(f"SHOW CREATE TABLE {logical}").first()
    if definition is not None:
        sql = definition["createtab_stmt"].casefold().replace("`", "")
        if re.search(r"\bfrom\s+" + re.escape(generation.casefold()) + r"\b", sql):
            return
    spark.sql(f"ALTER VIEW {logical} AS SELECT * FROM {generation}").collect()


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


def _generation_properties(config: dict[str, Any], prepared: Any) -> dict[str, str]:
    """Bind a full-rebuild table to one concrete saved model artifact."""
    digest = prepared.preflight.model_digest
    if type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Pinned model has no valid artifact digest.")
    return {
        "skyulf.mode": "full_rebuild",
        "skyulf.model_name": config["model_name"],
        "skyulf.model_version": config["model_version"],
        "skyulf.model_digest": digest,
    }


def provision_prediction_table(spark: Any, config: dict[str, Any], prepared: Any) -> bool:
    """Preflight an existing source/model and create only a missing output table."""
    source_name = config["score_source_table"]
    target_name = config["prediction_table"]
    for name in (source_name, target_name, config["model_name"]):
        if not _TABLE_NAME.fullmatch(name):
            raise ValueError("Scoring needs valid three-part Unity Catalog names.")
    if source_name == target_name:
        raise ValueError("Prediction output must differ from the source table.")
    if not spark.catalog.tableExists(source_name):
        raise ValueError(f"Existing input source table is missing: {source_name}.")
    if not prepared.preflight.ready:
        raise ValueError("The pinned registered model failed Skyulf preflight.")
    source = spark.table(source_name)
    columns = _prediction_columns(config, prepared, source)
    detail = spark.sql(f"DESCRIBE DETAIL {source_name}").first()
    properties = detail["properties"] or {}
    if not any(
        key.lower() == "delta.enablechangedatafeed" and str(value).lower() == "true"
        for key, value in properties.items()
    ):
        raise ValueError("Scoring source must have Delta Change Data Feed enabled.")
    target_exists = spark.catalog.tableExists(target_name)
    generation = config.get("model_change_mode") == "full_rebuild"
    generation_properties = _generation_properties(config, prepared) if generation else {}
    if target_exists:
        if generation:
            actual_properties = (
                spark.sql(f"DESCRIBE DETAIL {target_name}").first()["properties"] or {}
            )
            if any(
                actual_properties.get(key) != value for key, value in generation_properties.items()
            ):
                raise ValueError(
                    "Existing prediction generation belongs to another model or workflow."
                )
        _check_existing_table(spark, target_name, columns)
    if not target_exists:
        if (
            source.select(*config["row_keys"], *config["input_columns"])
            .limit(config["max_rows"] + 1)
            .count()
            > config["max_rows"]
        ):
            raise ValueError("Initial scoring source exceeds the configured max_rows budget.")
        definition = ", ".join(f"{name} {sql_type}" for name, _, sql_type in columns)
        table_properties = ""
        if generation:
            properties = ", ".join(
                f"'{key}' = '{value}'" for key, value in generation_properties.items()
            )
            table_properties = f" TBLPROPERTIES ({properties})"
        spark.sql(
            f"CREATE TABLE {target_name} ({definition}) USING DELTA{table_properties}"
        ).collect()
    return not target_exists


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


def _monthly_training_spec(spark: Any, config: dict[str, Any], now: datetime) -> LocalTrainingSpec:
    """Pin one Delta version and a UTC calendar window for monthly training."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Monthly training needs a timezone-aware run instant.")
    lookback = config.get("monthly_lookback_months")
    if type(lookback) is not int or not 2 <= lookback <= 120:
        raise ValueError("monthly_lookback_months must be an integer from 2 to 120.")
    cutoff = now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def months_before(count: int) -> datetime:
        """Preserve the first-of-month boundary across year rollover."""
        index = cutoff.year * 12 + cutoff.month - 1 - count
        return datetime(index // 12, index % 12 + 1, 1, tzinfo=UTC)

    table = config["training_table"]
    if not _TABLE_NAME.fullmatch(table):
        raise ValueError("training_table must be a three-part UC name.")
    latest = (
        spark.sql(f"DESCRIBE HISTORY {table}")
        .select("version")
        .orderBy("version", ascending=False)
        .first()
    )
    if latest is None or type(latest["version"]) is not int:
        raise ValueError("Training source has no concrete Delta version.")
    return LocalTrainingSpec(
        table=table,
        version=latest["version"],
        start=months_before(lookback),
        holdout_start=months_before(1),
        cutoff=cutoff,
        event_column=config["event_column"],
        label_time_column=config["label_time_column"],
        row_keys=tuple(config["row_keys"]),
        input_columns=tuple(config["input_columns"]),
        target_column=config["target_column"],
        max_rows=config["max_rows"],
        max_bytes=config["max_bytes"],
    )


def _monthly_champion_version(config: dict[str, Any]) -> str | None:
    """Resolve the current champion once while allowing first-model training."""
    try:
        champion = resolve_model(
            config["model_name"],
            alias="champion",
            tracking_uri=config.get("tracking_uri", "databricks"),
            registry_uri=config.get("registry_uri", "databricks-uc"),
        )
    except RegistryModelNotFoundError:
        return None
    return champion.version


def run_action(
    spark: Any,
    config: dict[str, Any],
    action: str,
    *,
    experiment_name: str | None = None,
    artifact_path: str | Path | None = None,
    now: datetime | None = None,
) -> Any:
    """Delegate training or scoring to Skyulf's existing services."""
    tracking_uri = config.get("tracking_uri", "databricks")
    registry_uri = config.get("registry_uri", "databricks-uc")
    if action in {"train", "train_monthly"}:
        if experiment_name is None or artifact_path is None:
            raise ValueError("Training needs an experiment and temporary artifact path.")
        monthly = action == "train_monthly"
        spec = (
            _monthly_training_spec(spark, config, now or datetime.now(UTC))
            if monthly
            else _training_spec(config)
        )
        champion_version = (
            _monthly_champion_version(config) if monthly else config.get("champion_version")
        )
        return train_local_candidate(
            spark,
            spec,
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
            champion_version=champion_version,
            quality_threshold=config.get("quality_threshold"),
        )
    if action == "score":
        target = _scoring_target(config)
        if config.get("model_change_mode", "incremental_append") == "full_rebuild":
            _managed_prediction_view_exists(spark, config["prediction_table"])
        score_config = {**config, "prediction_table": target}
        prepared = prepare_local_workflow(_scoring_config(score_config))
        provision_prediction_table(spark, score_config, prepared)
        result = run_incremental_local_batch(
            spark,
            prepared,
            row_keys=tuple(config["row_keys"]),
            admission=SingleWriterAdmission(),
        )
        if config.get("model_change_mode", "incremental_append") == "full_rebuild":
            _activate_prediction_view(spark, config["prediction_table"], target)
        return result
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
            experiment_name=widgets.get("experiment_name") if action.startswith("train") else None,
            artifact_path=Path(directory) / "artifact" if action.startswith("train") else None,
        )
    output = json.dumps(asdict(result), default=str, allow_nan=False)
    print(output)
    globals()["dbutils"].notebook.exit(output)


if __name__ == "__main__":
    main()
