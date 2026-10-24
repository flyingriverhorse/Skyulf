"""Validate and publish local model prediction tables and full-rebuild views."""

import re
from typing import Any

IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

TABLE_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){2}\Z")

OUTPUT_TYPES = {
    "float64": ("double", "DOUBLE"),
    "int64": ("long", "BIGINT"),
    "string": ("string", "STRING"),
    "bool": ("boolean", "BOOLEAN"),
}


def scoring_target(config: dict[str, Any]) -> str:
    """Choose a stable append target or a physical model-version generation."""
    mode = config.get("model_change_mode", "incremental_append")
    if type(mode) is not str or mode not in {"incremental_append", "full_rebuild"}:
        raise ValueError("model_change_mode must be incremental_append or full_rebuild.")
    base = config["prediction_table"]
    version = config["model_version"]
    if not TABLE_NAME_PATTERN.fullmatch(base):
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


def managed_prediction_view_exists(spark: Any, logical: str) -> bool:
    """Reject a table or unrelated view before creating prediction resources."""
    if not spark.catalog.tableExists(logical):
        return False
    if spark.catalog.getTable(logical).tableType.upper() != "VIEW":
        raise ValueError("Full rebuild needs a view name, but a physical table already uses it.")
    marker = spark.sql(f"SHOW TBLPROPERTIES {logical} ('skyulf.mode')").first()
    if marker is None or marker["value"] != "full_rebuild":
        raise ValueError("Existing prediction view is not managed by Skyulf full rebuild.")
    return True


def activate_prediction_view(spark: Any, logical: str, generation: str) -> None:
    """Expose a complete generation without replacing prior prediction tables."""
    if (
        not TABLE_NAME_PATTERN.fullmatch(logical)
        or not TABLE_NAME_PATTERN.fullmatch(generation)
        or not re.fullmatch(re.escape(logical) + r"_v[1-9][0-9]*", generation)
    ):
        raise ValueError("Prediction view and generation names must match one UC target.")
    physical = spark.table(generation)
    exists = managed_prediction_view_exists(spark, logical)
    if exists:
        _validate_generation_schema(spark, logical, physical)
        if _view_reads_generation(spark, logical, generation):
            return
    if not physical.limit(1).count() and not _empty_start_generation(spark, generation):
        raise ValueError("A full-rebuild generation must contain predictions before activation.")
    if not exists:
        spark.sql(
            f"CREATE VIEW {logical} TBLPROPERTIES ('skyulf.mode' = 'full_rebuild') "
            f"AS SELECT * FROM {generation}"
        ).collect()
        return
    spark.sql(f"ALTER VIEW {logical} AS SELECT * FROM {generation}").collect()


def prune_prediction_generations(
    spark: Any,
    logical: str,
    active: str,
    *,
    model_name: str,
    keep: int | None = 2,
) -> list[str]:
    """Drop older full-rebuild generations, keeping the newest ``keep`` and the active one.

    Only ``<logical>_v<n>`` tables tagged as this model's full-rebuild output are
    dropped; ``keep=None`` keeps every generation. Returns the dropped table names.
    """
    if keep is None:
        return []
    if type(keep) is not int or keep < 1:
        raise ValueError("keep_prediction_generations must be a positive integer or null.")
    catalog, schema, base = logical.split(".")
    pattern = re.compile(re.escape(base) + r"_v([1-9][0-9]*)", re.IGNORECASE)
    listed = spark.sql(f"SHOW TABLES IN `{catalog}`.`{schema}` LIKE '{base}_v*'").collect()
    versions = {
        f"{catalog}.{schema}.{row['tableName']}": int(match[1])
        for row in listed
        if (match := pattern.fullmatch(row["tableName"]))
    }
    newest = sorted(versions, key=versions.__getitem__, reverse=True)
    kept = {name.lower() for name in newest[:keep]} | {active.lower()}
    dropped = []
    for name in newest[keep:]:
        if name.lower() in kept or not _owned_generation(spark, name, model_name):
            continue
        spark.sql(f"DROP TABLE IF EXISTS {_quoted(name)}").collect()
        dropped.append(name)
    return dropped


def _owned_generation(spark: Any, name: str, model_name: str) -> bool:
    """Return True for a full-rebuild generation written for ``model_name``."""
    rows = spark.sql(f"SHOW TBLPROPERTIES {_quoted(name)}").collect()
    properties = {row["key"]: row["value"] for row in rows}
    return (
        properties.get("skyulf.mode") == "full_rebuild"
        and properties.get("skyulf.model_name") == model_name
    )


def _quoted(name: str) -> str:
    """Backtick each part of a validated three-part name."""
    return ".".join(f"`{part}`" for part in name.split("."))


def active_prediction_generation(spark: Any, logical: str) -> str | None:
    """Resolve only a managed same-target generation for boundary inheritance."""
    if not TABLE_NAME_PATTERN.fullmatch(logical):
        raise ValueError("Prediction view must be a three-part UC name.")
    if not managed_prediction_view_exists(spark, logical):
        return None
    row = spark.sql(f"SHOW CREATE TABLE {logical}").first()
    sql = row["createtab_stmt"].replace("`", "") if row is not None else ""
    match = re.search(
        r"\bAS\s+SELECT\s+\*\s+FROM\s+(" + re.escape(logical) + r"_v[1-9][0-9]*)\s*;?\s*$",
        sql,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError("Managed prediction view does not select one complete model generation.")
    return match[1]


def _empty_start_generation(spark: Any, generation: str) -> bool:
    """Allow an empty from_now result only after its atomic receipt was committed."""
    import json  # noqa: PLC0415

    from ...data.delta_io.delta import prediction_commit, table_identity  # noqa: PLC0415
    from ..incremental.incremental_batch import latest_source_version  # noqa: PLC0415
    from ..incremental.initial_scoring import start_fields  # noqa: PLC0415

    latest = prediction_commit(spark, generation, latest_source_version(spark, generation))
    try:
        receipt = json.loads(latest["userMetadata"] or "{}")
    except (TypeError, ValueError):
        return False
    if (
        receipt.get("skyulf_mode") != "incremental_append"
        or receipt.get("initial_scoring") != "from_now"
        or receipt.get("target_table_id") != table_identity(spark, generation)
        or receipt.get("output_count") != 0
    ):
        return False
    start_fields(receipt, "from_now", receipt.get("source_end_version"))
    return True


def _view_reads_generation(spark: Any, logical: str, generation: str) -> bool:
    """Keep a recovered empty generation active without treating it as a new release."""
    definition = spark.sql(f"SHOW CREATE TABLE {logical}").first()
    if definition is None:
        return False
    sql = definition["createtab_stmt"].casefold().replace("`", "")
    return re.search(r"\bfrom\s+" + re.escape(generation.casefold()) + r"\b", sql) is not None


def _prediction_columns(
    config: dict[str, Any], prepared: Any, source: Any
) -> tuple[tuple[str, str, str], ...]:
    """Derive an exact Delta target schema from source keys and saved model outputs."""
    keys = _prediction_source_keys(config, prepared, source)
    columns = []
    for key in keys:
        kind = source.schema[key].dataType.typeName()
        if kind not in {"string", "long"}:
            raise ValueError(f"Source row key {key!r} must be STRING or BIGINT.")
        columns.append((key, kind, "STRING" if kind == "string" else "BIGINT"))
    for output in prepared.preflight.output_schema:
        if not IDENTIFIER_PATTERN.fullmatch(output.name) or output.dtype not in OUTPUT_TYPES:
            raise ValueError("Saved model output has an unsupported name or type.")
        kind, sql_type = OUTPUT_TYPES[output.dtype]
        columns.append((output.name, kind, sql_type))
    columns.extend((name, "string", "STRING") for name in ("run_id", "model_name", "model_version"))
    if len({name.lower() for name, _, _ in columns}) != len(columns):
        raise ValueError("Prediction columns collide with keys or metadata.")
    return tuple(columns)


def check_existing_table(spark: Any, name: str, columns: tuple[tuple[str, str, str], ...]) -> None:
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
    source_name, target_name = _validate_prediction_request(spark, config, prepared)
    source = spark.table(source_name)
    columns = _prediction_columns(config, prepared, source)
    _require_change_feed(spark, source_name)
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
        check_existing_table(spark, target_name, columns)
    if not target_exists:
        _create_prediction_table(
            spark, config, source, target_name, columns, generation, generation_properties
        )
    return not target_exists


def _validate_generation_schema(spark: Any, logical: str, physical: Any) -> None:
    """Require a new generation to preserve the active prediction view schema."""
    active_columns = tuple(
        (field.name, field.dataType.typeName()) for field in spark.table(logical).schema.fields
    )
    candidate_columns = tuple(
        (field.name, field.dataType.typeName()) for field in physical.schema.fields
    )
    if active_columns != candidate_columns:
        raise ValueError("New prediction generation differs from the active view schema.")


def _prediction_source_keys(config: dict[str, Any], prepared: Any, source: Any) -> tuple[str, ...]:
    """Validate source key identifiers and saved input column membership."""
    keys = tuple(config["record_key_columns"])
    inputs = tuple(prepared.artifact.manifest.input_columns)
    if not keys or tuple(config["input_columns"]) != inputs:
        raise ValueError("Configured keys or model inputs differ from the fitted model.")
    names = (*keys, *inputs)
    if any(not IDENTIFIER_PATTERN.fullmatch(name) for name in names):
        raise ValueError("Source keys and model inputs need simple column identifiers.")
    if len({name.lower() for name in names}) != len(names):
        raise ValueError("Source keys and model inputs must be distinct.")
    if not set(names).issubset(source.columns):
        raise ValueError("Scoring source lacks a row key or fitted model input column.")
    return keys


def _validate_prediction_request(
    spark: Any, config: dict[str, Any], prepared: Any
) -> tuple[str, str]:
    """Check existing source, target identity and model readiness before provisioning."""
    source_name = config["score_source_table"]
    target_name = config["prediction_table"]
    for name in (source_name, target_name, config["model_name"]):
        if not TABLE_NAME_PATTERN.fullmatch(name):
            raise ValueError("Scoring needs valid three-part Unity Catalog names.")
    if source_name == target_name:
        raise ValueError("Prediction output must differ from the source table.")
    if not spark.catalog.tableExists(source_name):
        raise ValueError(f"Existing input source table is missing: {source_name}.")
    if not prepared.preflight.ready:
        raise ValueError("The pinned registered model failed Skyulf preflight.")
    return source_name, target_name


def _require_change_feed(spark: Any, source_name: str) -> None:
    """Require enabled Delta change data feed before provisioning prediction output."""
    detail = spark.sql(f"DESCRIBE DETAIL {source_name}").first()
    properties = detail["properties"] or {}
    if not any(
        key.lower() == "delta.enablechangedatafeed" and str(value).lower() == "true"
        for key, value in properties.items()
    ):
        raise ValueError("Scoring source must have Delta Change Data Feed enabled.")


def _create_prediction_table(
    spark: Any,
    config: dict[str, Any],
    source: Any,
    target_name: str,
    columns: tuple[tuple[str, str, str], ...],
    generation: bool,
    generation_properties: dict[str, str],
) -> None:
    """Check local bootstrap size and create a target with pinned generation properties."""
    if (
        config.get("inference_mode", "local") != "spark"
        and config.get("initial_scoring", "full_snapshot") != "from_now"
        and source.select(*config["record_key_columns"], *config["input_columns"])
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
    spark.sql(f"CREATE TABLE {target_name} ({definition}) USING DELTA{table_properties}").collect()
