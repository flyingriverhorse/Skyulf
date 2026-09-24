# Databricks notebook source
"""Run one bounded Skyulf local training, comparison, alias or scoring action."""

import json
import re
import tempfile
from dataclasses import asdict
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
from skyulf.integrations.databricks.delta_admission import DeltaTableAdmission
from skyulf.integrations.mlflow.promotion import (
    DeltaAliasAdmission,
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
    if action == "score":
        scoring = LocalWorkflowConfig(
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
                tracking_uri=tracking_uri,
                registry_uri=registry_uri,
            ),
            sink=OutputSink(kind="uc_delta", table=config["prediction_table"]),
        )
        prepared = prepare_local_workflow(scoring)
        return run_incremental_local_batch(
            spark,
            prepared,
            row_keys=tuple(config["row_keys"]),
            admission=DeltaTableAdmission(spark, config["score_admission_table"]),
        )
    if action in {"compare", "stage", "promote"}:
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
