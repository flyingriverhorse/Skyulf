"""Offline validation and explicit migration of local Databricks Bundle settings."""

import json
import math
import re
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from ...config_validation import validate_pipeline_config
from ...modeling.base import BaseModelCalculator
from ...registry import NodeRegistry
from ..mlflow.validation import _CLASSIFICATION, _MINIMIZE, _REGRESSION
from ._contracts import PREDICTION_METADATA_COLUMNS, input_budget_bytes
from .local_cv import CV_FIELDS, LocalCVSpec
from .local_sdk import ModelSelection
from .local_workflow import _training_spec, _training_window_mode
from .prediction_output import _IDENTIFIER, _TABLE_NAME
from .training_dates import training_date_spec

_ACTIONS = {"train", "train_monthly", "score", "approve", "reject", "rollback"}
_FIELDS = {
    "pre_split_steps",
    *CV_FIELDS,
    "training_sample_rows",
    "training_sample_seed",
    "training_window_mode",
    "window_timezone",
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
    "record_key_columns",
    "input_columns",
    "target_column",
    "event_column",
    "result_available_at_column",
    "event_time_parsing",
    "result_time_parsing",
    "training_version",
    "split_strategy",
    "test_size",
    "random_state",
    "stratify",
    "filter_unavailable_results",
    "result_cutoff",
    "start",
    "holdout_start",
    "cutoff",
    "monthly_lookback_months",
    "max_rows",
    "max_input_mb",
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
    for key in ("record_key_columns", "input_columns"):
        value = config.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(f"{key} must be a nonempty list of source column names.")
        names.extend(value)
    names.append(config.get("target_column"))
    names.extend(
        config[key]
        for key in ("event_column", "result_available_at_column")
        if config.get(key) is not None
    )
    checked: list[str] = []
    for name in names:
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
            raise ValueError("Source columns must be simple column identifiers.")
        checked.append(name)
    if len({name.casefold() for name in checked}) != len(checked):
        raise ValueError("Key, input, target, event and label columns must be distinct.")
    if any(name.lower() in PREDICTION_METADATA_COLUMNS for name in config["record_key_columns"]):
        raise ValueError("record_key_columns contain reserved prediction metadata names.")
    if any(
        name.lower() in {"_change_type", "_commit_version", "_commit_timestamp"} for name in checked
    ):
        raise ValueError("Source columns contain reserved Delta change metadata names.")


def _training_contract(config: dict[str, Any], action: str) -> None:
    """Validate explicit policies while requiring manual pins only for manual training."""
    settings = dict(config)
    strategy = config.get("split_strategy", "random")
    mode = _training_window_mode(config)
    if config.get("stratify") is True and config["task"] != "classification":
        raise ValueError("stratify requires a classification task.")
    if action == "train":
        version = config.get("training_version")
        if type(version) is not int or version < 0:
            raise ValueError(
                "Set training_version to an explicit nonnegative Delta version before train."
            )
    else:
        settings["training_version"] = 0
        # These actions use saved evidence or derive fresh boundaries at invocation.
        if mode != "full_snapshot" and (action != "train_monthly" or mode == "rolling_calendar"):
            boundaries = {"start": 1, "cutoff": 3}
            if strategy == "temporal":
                boundaries["holdout_start"] = 2
            for key, month in boundaries.items():
                settings[key] = datetime(2000, month, 1, tzinfo=UTC).isoformat()
        if config.get("filter_unavailable_results") is True:
            settings["result_cutoff"] = datetime(2000, 3, 1, tzinfo=UTC).isoformat()
    _training_spec(settings)


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
    if type(config.get("max_rows")) is not int or config["max_rows"] <= 0:
        raise ValueError("max_rows must be a positive integer.")
    input_budget_bytes(config.get("max_input_mb"))
    _columns(config)
    for field in ("event_time_parsing", "result_time_parsing"):
        training_date_spec(config.get(field) if config.get(field) is not None else {})
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
    LocalCVSpec.from_workflow(config).validate_pipeline(
        pipeline, target_column=config["target_column"], event_column=config.get("event_column")
    )
    return deepcopy(config)


def preview_workflow_config(config: dict[str, Any], *, action: str = "score") -> str:
    """Describe resolved settings offline using the same preflight as job execution.

    The default checks the configuration without requiring manual training pins.
    Pass ``action='train'`` or ``'train_monthly'`` for action-specific validation.
    This cannot check source values, installed worker dependencies or permissions.
    """
    checked = validate_workflow_config(config, action=action)
    model = checked["pipeline"]["modeling"]
    cv = LocalCVSpec.from_workflow(checked)
    manual_status = "configured (source data and permissions not checked)"
    try:
        validate_workflow_config(checked, action="train")
    except ValueError as exc:
        manual_status = f"needs configuration: {exc}"
    sample = checked.get("training_sample_rows")
    window = _training_window_mode(checked)
    monthly = action == "train_monthly"
    version = "latest snapshot at invocation" if monthly else checked.get("training_version")
    observation_window = f"[{checked.get('start')}, {checked.get('cutoff')})"
    holdout_start = checked.get("holdout_start")
    result_cutoff = checked.get("result_cutoff")
    if monthly:
        if window == "rolling_calendar":
            observation_window = "completed calendar months at invocation"
            if checked.get("split_strategy") == "temporal":
                holdout_start = "last completed calendar month"
        if checked.get("filter_unavailable_results"):
            result_cutoff = "invocation time"
    lines = [
        "Skyulf workflow preview",
        "No data read, training, registry mutation or deployment.",
        f"Engine: {checked['engine']} | Task: {checked['task']}",
        f"Training source: {checked['training_table']} @ {version}",
        f"Record keys: {', '.join(checked['record_key_columns'])}",
        f"Features: {', '.join(checked['input_columns'])} | Target: {checked['target_column']}",
        f"Source selection: {window} | Event column: {checked.get('event_column')}",
        f"Window: {observation_window}",
        f"Calendar: {checked.get('monthly_lookback_months')} months, "
        f"timezone={checked.get('window_timezone')}",
        f"Result availability: {checked.get('result_available_at_column')} | "
        f"filter={checked.get('filter_unavailable_results', False)} | "
        f"cutoff={result_cutoff}",
        f"Training sample: {sample if sample is not None else 'all eligible rows'} "
        f"(includes final holdout); seed={checked.get('training_sample_seed', 42)}",
        f"Local input limits: {checked['max_rows']} rows, {checked['max_input_mb']} MiB "
        "(not total training memory)",
        "Phase order: source window -> optional seeded sample -> bounded read -> "
        "availability -> fixed cleanup and training filters -> final split -> fold-local preprocessing/model.",
        "With sampling, availability is selected before the seeded sample; without "
        "sampling, it is selected after the bounded read.",
        f"Final holdout: {checked.get('split_strategy', 'random')} | "
        f"fraction={checked.get('test_size', 0.2)} | start={holdout_start}",
        f"Manual training: {manual_status}",
        "Pre-split cleanup (fixed normalization and training eligibility; edit build_pre_split_steps()):",
    ]
    for index, step in enumerate(checked.get("pre_split_steps", []), 1):
        lines.append(
            f"  {index}. {step['name']} -> {step['transformer']} "
            f"{json.dumps(step.get('params', {}), sort_keys=True)}"
        )
    if not checked.get("pre_split_steps"):
        lines.append("  No pre-split cleanup steps.")
    lines.append(
        "Fixed feature cleanup is saved as a pipeline prefix and applied once to raw model inputs; "
        "training row exclusions are not repeated during scoring."
    )
    lines.append("Fold-local preprocessing (after final split; edit build_preprocessing()):")
    for index, step in enumerate(checked["pipeline"].get("preprocessing", []), 1):
        lines.append(
            f"  {index}. {step['name']} -> {step['transformer']} "
            f"{json.dumps(step.get('params', {}), sort_keys=True)}"
        )
    if not checked["pipeline"].get("preprocessing"):
        lines.append("  No preprocessing steps.")
    lines.extend(
        [
            f"Model: {model['type']} | Explicit params: {json.dumps(model.get('params', {}))}",
            "Unspecified model parameters use Core defaults.",
            f"CV: {cv.method}, {cv.folds} folds, training partition only; "
            "preprocessing refitted per fold, no parameter search."
            if cv.enabled
            else "CV: disabled (final holdout evaluation still runs).",
            f"Promotion: {checked['promotion_policy']} | Metric: {checked['metric']} | "
            f"Threshold: {checked.get('quality_threshold')} | "
            f"Minimum improvement: {checked['min_improvement']}",
            f"Scoring source: {checked['score_source_table']}",
            f"Score model: {checked['model_name']} | {checked['score_model_selection']} | "
            f"pin={checked.get('model_version')} | handoff={checked['score_handoff']}",
            f"Prediction output: {checked['prediction_table']} | {checked['model_change_mode']}",
            "Scoring is never sampled. Validate source values and worker dependencies separately.",
        ]
    )
    return "\n".join(lines)


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
