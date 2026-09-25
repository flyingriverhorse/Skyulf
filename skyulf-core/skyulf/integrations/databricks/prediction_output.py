"""Validate and publish local model prediction tables and full-rebuild views."""

import re
from typing import Any

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

_TABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){2}\Z")

_OUTPUT_TYPES = {
    "float64": ("double", "DOUBLE"),
    "int64": ("long", "BIGINT"),
    "string": ("string", "STRING"),
    "bool": ("boolean", "BOOLEAN"),
}


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
    keys = tuple(config["record_key_columns"])
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
            source.select(*config["record_key_columns"], *config["input_columns"])
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
