"""Offline validation and explicit migration of local Databricks Bundle settings."""

import math
import re
from copy import deepcopy
from datetime import datetime
from typing import Any

from ...config_validation import validate_pipeline_config
from ...modeling.base import BaseModelCalculator
from ...registry import NodeRegistry
from ..mlflow.validation import _CLASSIFICATION, _MINIMIZE, _REGRESSION
from ._contracts import PREDICTION_METADATA_COLUMNS
from .local_retraining import LocalTrainingSpec
from .local_sdk import ModelSelection
from .prediction_output import _IDENTIFIER, _TABLE_NAME

_ACTIONS = {"train", "train_monthly", "score", "approve", "reject", "rollback"}
_FIELDS = {
    "config_version",
    "task",
    "engine",
    "training_table",
    "score_source_table",
    "prediction_table",
    "model_name",
    "model_version",
    "model_change_mode",
    "score_model_selection",
    "promotion_policy",
    "score_handoff",
    "champion_version",
    "risk_category",
    "row_keys",
    "input_columns",
    "target_column",
    "event_column",
    "label_time_column",
    "training_version",
    "start",
    "holdout_start",
    "cutoff",
    "monthly_lookback_months",
    "max_rows",
    "max_bytes",
    "metric",
    "min_improvement",
    "quality_threshold",
    "pipeline",
    "tracking_uri",
    "registry_uri",
}


def _choice(config: dict[str, Any], key: str, choices: set[str]) -> str:
    """Reject mistyped and unknown choices with their exact configuration field."""
    value = config.get(key)
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{key} must be one of {', '.join(sorted(choices))}.")
    return value


def _finite(value: Any, name: str) -> float:
    """Require an actual finite numeric value rather than coercing strings or booleans."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return value


def _columns(config: dict[str, Any]) -> None:
    """Require explicit, distinct source identities and feature/label roles."""
    names = []
    for key in ("row_keys", "input_columns"):
        value = config.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(f"{key} must be a nonempty list of source column names.")
        names.extend(value)
    names.extend(config.get(key) for key in ("target_column", "event_column", "label_time_column"))
    checked: list[str] = []
    for name in names:
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
            raise ValueError("Source columns must be simple column identifiers.")
        checked.append(name)
    if len({name.casefold() for name in checked}) != len(checked):
        raise ValueError("Key, input, target, event and label columns must be distinct.")
    if any(name.lower() in PREDICTION_METADATA_COLUMNS for name in config["row_keys"]):
        raise ValueError("row_keys contain reserved prediction metadata names.")
    if any(
        name.lower() in {"_change_type", "_commit_version", "_commit_timestamp"} for name in checked
    ):
        raise ValueError("Source columns contain reserved Delta change metadata names.")


def _training_contract(config: dict[str, Any], action: str) -> None:
    """Reuse the pinned training specification without requiring dates for scoring."""
    if action == "train_monthly":
        months = config.get("monthly_lookback_months")
        if type(months) is not int or not 2 <= months <= 120:
            raise ValueError("monthly_lookback_months must be an integer from 2 to 120.")
    if action != "train":
        return
    version = config.get("training_version")
    if type(version) is not int or version < 0:
        raise ValueError(
            "Set training_version to an explicit nonnegative Delta version before train."
        )
    dates = {}
    for key in ("start", "holdout_start", "cutoff"):
        raw = config.get(key)
        if not isinstance(raw, str) or not raw:
            raise ValueError(
                f"Set {key} to an explicit timezone-aware training boundary before train."
            )
        try:
            dates[key] = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"{key} must be an ISO timestamp with timezone.") from exc
    LocalTrainingSpec(
        table=config["training_table"],
        version=version,
        **dates,
        event_column=config["event_column"],
        label_time_column=config["label_time_column"],
        row_keys=tuple(config["row_keys"]),
        input_columns=tuple(config["input_columns"]),
        target_column=config["target_column"],
        max_rows=config["max_rows"],
        max_bytes=config["max_bytes"],
    )


def validate_workflow_config(config: dict[str, Any], *, action: str) -> dict[str, Any]:
    """Validate a resolved project without starting Spark, fitting or accessing MLflow.

    Historical dates are allowed for deliberate snapshot replays. Generated
    projects leave manual dates/version unset so examples cannot silently train
    on an obsolete snapshot. Saved-evidence actions use their own pinned data.
    """
    if not isinstance(config, dict):
        raise ValueError("Workflow configuration must be an object.")
    if (
        "model_selection_mode" in config
        or type(config.get("config_version")) is not int
        or config.get("config_version") != 1
    ):
        raise ValueError(
            "config_version must be 1; migrate the project and regenerate/redeploy its jobs."
        )
    unknown = set(config) - _FIELDS
    if unknown:
        raise ValueError(f"Unknown workflow settings: {', '.join(sorted(unknown))}.")
    if action not in _ACTIONS:
        raise ValueError("Unknown workflow action.")
    task = _choice(config, "task", {"regression", "classification"})
    _choice(config, "engine", {"pandas", "polars"})
    selection = _choice(config, "score_model_selection", {"champion", "pinned_version"})
    policy = _choice(config, "promotion_policy", {"automatic", "manual_approval"})
    _choice(config, "score_handoff", {"disabled", "after_alias_change"})
    _choice(config, "model_change_mode", {"incremental_append", "full_rebuild"})
    for key in ("training_table", "score_source_table", "prediction_table", "model_name"):
        value = config.get(key)
        if not isinstance(value, str) or not _TABLE_NAME.fullmatch(value):
            raise ValueError(f"{key} must be a resolved three-part Unity Catalog name.")
    if config["prediction_table"].casefold() in {
        config[key].casefold() for key in ("training_table", "score_source_table")
    }:
        raise ValueError("prediction_table must not overwrite a training/scoring source.")
    for key in ("max_rows", "max_bytes"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    _columns(config)
    champion = config.get("champion_version")
    if champion is not None and (
        not isinstance(champion, str) or not re.fullmatch(r"[1-9][0-9]*", champion)
    ):
        raise ValueError("champion_version must be null or a concrete positive integer string.")
    risk = config.get("risk_category")
    if risk is not None and (not isinstance(risk, str) or len(risk.encode("utf-8")) > 256):
        raise ValueError("risk_category must be text of at most 256 UTF-8 bytes.")
    version = config.get("model_version")
    if version is not None and (
        not isinstance(version, str) or not re.fullmatch(r"[1-9][0-9]*", version)
    ):
        raise ValueError("model_version must be a concrete positive integer string.")
    if selection == "pinned_version" and version is None:
        raise ValueError("Pinned scoring requires model_version.")
    ModelSelection(
        kind="local_pipeline",
        name=config["model_name"],
        version=version if selection == "pinned_version" else None,
        alias="champion" if selection == "champion" else None,
        tracking_uri=config.get("tracking_uri"),
        registry_uri=config.get("registry_uri"),
    )
    metric = config.get("metric")
    if not isinstance(metric, str) or metric not in (
        _REGRESSION if task == "regression" else _CLASSIFICATION
    ):
        raise ValueError(f"metric must be a supported heldout metric for {task}.")
    if _finite(config.get("min_improvement"), "min_improvement") < 0:
        raise ValueError("min_improvement must be nonnegative.")
    threshold = config.get("quality_threshold")
    if threshold is None:
        if policy == "automatic":
            raise ValueError("Automatic promotion requires quality_threshold.")
    else:
        threshold = _finite(threshold, "quality_threshold")
        if metric in _MINIMIZE and threshold < 0:
            raise ValueError("quality_threshold must be nonnegative for error/loss metrics.")
        if task == "classification" and metric != "heldout_log_loss":
            lower = -1 if metric == "heldout_matthews_corrcoef" else 0
            if not lower <= threshold <= 1:
                raise ValueError(f"quality_threshold for {metric} must be between {lower} and 1.")
        if metric in {"heldout_r2", "heldout_explained_variance"} and threshold > 1:
            raise ValueError(f"quality_threshold for {metric} must be at most 1.")
    pipeline = config.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError("pipeline must be a Core pipeline configuration object.")
    validate_pipeline_config(pipeline)
    model = pipeline.get("modeling")
    if not isinstance(model, dict) or not isinstance(model.get("type"), str):
        raise ValueError("pipeline.modeling.type must identify a registered Core model.")
    calculator = NodeRegistry.get_calculator(model["type"])
    if not issubclass(calculator, BaseModelCalculator) or calculator().problem_type != task:
        raise ValueError("Configured model does not match the declared task.")
    for step in pipeline.get("preprocessing", []):
        if issubclass(NodeRegistry.get_calculator(step["transformer"]), BaseModelCalculator):
            raise ValueError("pipeline.preprocessing cannot contain a model calculator.")
    _training_contract(config, action)
    return deepcopy(config)


def migrate_workflow_config(
    config: dict[str, Any], *, task: str, score_handoff: str
) -> dict[str, Any]:
    """Return an explicit migration; caller saves it and regenerates/redeploys job definitions."""
    if not isinstance(config, dict):
        raise ValueError("Workflow configuration must be an object.")
    if "config_version" in config and (
        type(config["config_version"]) is not int or config["config_version"] != 1
    ):
        raise ValueError("Cannot migrate an unknown config_version.")
    if "task" in config and config["task"] != task:
        raise ValueError("Migration must not change the existing task.")
    migrated = deepcopy(config)
    if "model_selection_mode" in migrated:
        if {"score_model_selection", "promotion_policy"}.intersection(migrated):
            raise ValueError("Remove mixed legacy/new policies before migration.")
        legacy = _choice(migrated, "model_selection_mode", {"pinned_version", "auto_champion"})
        migrated.pop("model_selection_mode")
        migrated.update(
            score_model_selection="champion" if legacy == "auto_champion" else "pinned_version",
            promotion_policy="automatic" if legacy == "auto_champion" else "manual_approval",
        )
    if "score_handoff" in migrated and migrated["score_handoff"] != score_handoff:
        raise ValueError("Migration must not silently change the existing score_handoff.")
    migrated.update(config_version=1, task=task, score_handoff=score_handoff)
    # Binding names and manual training dates remain a caller-owned step.
    _choice(migrated, "task", {"regression", "classification"})
    _choice(migrated, "score_handoff", {"disabled", "after_alias_change"})
    _choice(migrated, "score_model_selection", {"champion", "pinned_version"})
    _choice(migrated, "promotion_policy", {"automatic", "manual_approval"})
    return migrated


def validate_deployed_contract(config: dict[str, Any], parameters: dict[str, str]) -> None:
    """Require notebook/job generation to agree with the project's handoff contract."""
    if parameters.get("workflow_contract") != "1" or parameters.get(
        "deployed_score_handoff"
    ) != config.get("score_handoff"):
        raise ValueError(
            "Project and job definitions disagree; regenerate/redeploy the Bundle together."
        )
