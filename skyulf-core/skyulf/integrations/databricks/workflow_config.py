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
from ..mlflow.validation import validate_quality_policy
from ._contracts import PREDICTION_METADATA_COLUMNS, input_budget_bytes
from .decision_thresholds import threshold_policy
from .evaluation_chart_data import chart_settings
from .local_cv import CV_FIELDS, LocalCVSpec
from .local_explanations import validate_explanation_config
from .local_sdk import ModelSelection
from .local_search import base_model_config, prepare_search_pipeline
from .local_workflow import training_settings, training_spec, training_window_mode
from .prediction_output import IDENTIFIER_PATTERN, TABLE_NAME_PATTERN
from .training_dates import training_date_spec
from .weight_config import WEIGHT_FIELDS, validate_weight_roles

_ACTIONS = {"train", "score", "approve", "reject", "rollback"}
WORKFLOW_FIELDS = {
    *WEIGHT_FIELDS,
    "evaluation_charts",
    "training_layout",
    "competition",
    "competition_max_trials",
    "competition_max_candidates",
    "pre_split_steps",
    *CV_FIELDS,
    "training_sample_rows",
    "training_sample_seed",
    "training_window_mode",
    "window_timezone",
    "config_version",
    "task",
    "engine",
    "inference_mode",
    "spark_udf_env_manager",
    "spark_udf_prediction_batch_rows",
    "training_table",
    "score_source_table",
    "prediction_table",
    "model_name",
    "model_version",
    "model_change_mode",
    "auto_rebuild_on_cdf_expiry",
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
    "holdout_months",
    "lookback_days",
    "holdout_days",
    "result_availability_lag_hours",
    "max_rows",
    "max_input_mb",
    "metric",
    "min_improvement",
    "quality_threshold",
    "quality_gates",
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
        for key in ("event_column", "result_available_at_column", "cv_group_column")
        if config.get(key) is not None
    )
    _validate_column_roles(names, config["record_key_columns"])
    validate_weight_roles(config)


def _validate_column_roles(names: list[Any], record_key_columns: list[str]) -> None:
    """Reject ambiguous or reserved source column roles in their declared order."""
    checked: list[str] = []
    for name in names:
        if not isinstance(name, str) or not IDENTIFIER_PATTERN.fullmatch(name):
            raise ValueError("Source columns must be simple column identifiers.")
        checked.append(name)
    if len({name.casefold() for name in checked}) != len(checked):
        raise ValueError("Key, input, target, event and label columns must be distinct.")
    if any(name.lower() in PREDICTION_METADATA_COLUMNS for name in record_key_columns):
        raise ValueError("record_key_columns contain reserved prediction metadata names.")
    if any(
        name.lower() in {"_change_type", "_commit_version", "_commit_timestamp"} for name in checked
    ):
        raise ValueError("Source columns contain reserved Delta change metadata names.")


def _training_contract(config: dict[str, Any], action: str) -> None:
    """Validate training selection offline, allowing unset versions to resolve at invocation."""
    settings = dict(config)
    strategy = config.get("split_strategy", "random")
    mode = training_window_mode(config)
    if config.get("stratify") is True and config["task"] != "classification":
        raise ValueError("stratify requires a classification task.")
    if action == "train":
        settings = training_settings(config, datetime(2000, 3, 1, tzinfo=UTC))
    else:
        settings["training_version"] = 0
        # These actions use saved evidence or derive fresh boundaries at invocation.
        if mode != "full_snapshot":
            boundaries = {"start": 1, "cutoff": 3}
            if strategy == "temporal":
                boundaries["holdout_start"] = 2
            for key, month in boundaries.items():
                settings[key] = datetime(2000, month, 1, tzinfo=UTC).isoformat()
        if config.get("filter_unavailable_results") is True:
            settings["result_cutoff"] = datetime(2000, 3, 1, tzinfo=UTC).isoformat()
    training_spec(settings)


def _validate_workflow_fields(config: dict[str, Any], action: str) -> None:
    """Reject obsolete or unknown settings before checking their values."""
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
    unknown = set(config) - WORKFLOW_FIELDS
    if unknown:
        raise ValueError(f"Unknown workflow settings: {', '.join(sorted(unknown))}.")
    if action not in _ACTIONS:
        raise ValueError("Unknown workflow action.")
    if type(config.get("auto_rebuild_on_cdf_expiry", False)) is not bool:
        raise ValueError("auto_rebuild_on_cdf_expiry must be a boolean.")


def _validate_inference_settings(config: dict[str, Any]) -> None:
    """Keep inference independent of training and bound each worker prediction call."""
    mode = (
        _choice(config, "inference_mode", {"local", "spark"})
        if "inference_mode" in config
        else "local"
    )
    if mode == "spark" and config["engine"] != "pandas":
        raise ValueError("inference_mode=spark requires engine=pandas.")
    if "spark_udf_env_manager" in config:
        _choice(config, "spark_udf_env_manager", {"local", "virtualenv"})
    batch = config.get("spark_udf_prediction_batch_rows", 10000)
    if type(batch) is not int or not 1 <= batch <= 100000:
        raise ValueError("spark_udf_prediction_batch_rows must be 1..100000.")


def _validate_workflow_sources(config: dict[str, Any]) -> None:
    """Validate source names, input limits and column roles before any reader opens."""
    for key in ("training_table", "score_source_table", "prediction_table", "model_name"):
        value = config.get(key)
        if not isinstance(value, str) or not TABLE_NAME_PATTERN.fullmatch(value):
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


def _validate_champion_metadata(config: dict[str, Any]) -> None:
    """Validate optional champion pins and risk metadata before scoring selection."""
    champion = config.get("champion_version")
    if champion is not None and (
        not isinstance(champion, str) or not re.fullmatch(r"[1-9][0-9]*", champion)
    ):
        raise ValueError("champion_version must be null or a concrete positive integer string.")
    risk = config.get("risk_category")
    if risk is not None and (not isinstance(risk, str) or len(risk.encode("utf-8")) > 256):
        raise ValueError("risk_category must be text of at most 256 UTF-8 bytes.")


def _validate_workflow_model_selection(config: dict[str, Any], selection: str) -> None:
    """Require valid pins, risk metadata and registry selection settings."""
    _validate_champion_metadata(config)
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


def _validate_workflow_quality(config: dict[str, Any], task: str, policy: str) -> None:
    """Reject incompatible metric bounds and incomplete automatic promotion policies."""
    validate_quality_policy(
        config.get("metric", ""),
        config.get("quality_threshold"),
        config.get("quality_gates"),
        task=task,
    )
    if _finite(config.get("min_improvement"), "min_improvement") < 0:
        raise ValueError("min_improvement must be nonnegative.")
    if config.get("quality_threshold") is None and policy == "automatic":
        raise ValueError("Automatic promotion requires quality_threshold.")


def validate_workflow_pipeline(
    config: dict[str, Any], task: str, *, allow_empty_model: bool = False
) -> None:
    """Check the model task and preprocessing types against the Core registry."""
    pipeline = config.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError("pipeline must be a Core pipeline configuration object.")
    validate_pipeline_config(pipeline)
    _validate_pipeline_model(pipeline, task, allow_empty_model=allow_empty_model)
    if pipeline.get("modeling"):
        threshold_policy(pipeline)
    for step in pipeline.get("preprocessing", []):
        if issubclass(NodeRegistry.get_calculator(step["transformer"]), BaseModelCalculator):
            raise ValueError("pipeline.preprocessing cannot contain a model calculator.")
    validate_explanation_config(pipeline)


def _validate_pipeline_model(
    pipeline: dict[str, Any], task: str, *, allow_empty_model: bool
) -> None:
    """Validate declared models, permitting only an explicit empty saved-model declaration."""
    model = pipeline.get("modeling")
    if allow_empty_model and type(model) is dict and not model:
        return
    if not isinstance(model, dict) or not isinstance(model.get("type"), str):
        raise ValueError("pipeline.modeling.type must identify a registered Core model.")
    calculator = NodeRegistry.get_calculator(base_model_config(pipeline)["type"])
    if not issubclass(calculator, BaseModelCalculator) or calculator().problem_type != task:
        raise ValueError("Configured model does not match the declared task.")


def validate_workflow_config(config: dict[str, Any], *, action: str) -> dict[str, Any]:
    """Validate a resolved project without starting Spark, fitting or accessing MLflow.

    Historical dates and explicit versions allow deliberate snapshot replays.
    An unset version selects the latest snapshot at invocation. Saved-evidence
    actions use their own pinned data.
    """
    _validate_workflow_fields(config, action)
    chart_settings(config.get("evaluation_charts"))
    _validate_layout(config, action)
    task = _choice(config, "task", {"regression", "classification"})
    _choice(config, "engine", {"pandas", "polars"})
    _validate_inference_settings(config)
    selection = _choice(config, "score_model_selection", {"champion", "pinned_version"})
    policy = _choice(config, "promotion_policy", {"automatic", "manual_approval"})
    _choice(config, "score_handoff", {"disabled", "after_alias_change"})
    _choice(config, "model_change_mode", {"incremental_append", "full_rebuild"})
    _validate_workflow_sources(config)
    _validate_workflow_model_selection(config, selection)
    _validate_workflow_quality(config, task, policy)
    validate_workflow_pipeline(
        config,
        task,
        allow_empty_model=(
            action != "train" and config.get("training_layout", "single_model") == "single_model"
        ),
    )
    _training_contract(config, action)
    cv = LocalCVSpec.from_workflow(config)
    if action == "train":
        _validate_bundle_cv_holdout(config, cv)
        # Saved-model actions never execute the editable project training hooks.
        cv.validate_pipeline(
            config["pipeline"],
            target_column=config["target_column"],
            event_column=config.get("event_column"),
        )
    return deepcopy(config)


def validate_project_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Check shared training/scoring settings without executing editable model hooks.

    Generated single-model settings intentionally leave modeling empty until the
    Python recipe is loaded. Shared training dates and CV holdout rules still
    need their actual configured values checked during static smoke validation.
    """
    checked = validate_workflow_config(config, action="score")
    _training_contract(checked, "train")
    _validate_bundle_cv_holdout(checked, LocalCVSpec.from_workflow(checked))
    return checked


def _validate_layout(config: dict[str, Any], action: str) -> None:
    """Admit explicit layouts and verify resolved competition before remote work."""
    layout = config.get("training_layout", "single_model")
    if layout not in {"single_model", "multi_target", "model_competition"}:
        raise ValueError("Unknown training_layout.")
    if "competition" in config and layout != "model_competition":
        raise ValueError("Competition candidates require training_layout=model_competition.")
    if layout == "model_competition" and action == "train":
        from .competition_project import validate_competition_config  # noqa: PLC0415
        from .local_competition import validate_competition_budget  # noqa: PLC0415

        validate_competition_config(config)
        validate_competition_budget(config)


def _validate_bundle_cv_holdout(config: dict[str, Any], cv: LocalCVSpec) -> None:
    """Check policy isolation without resolving source versions or runtime dates."""
    if (
        cv.enabled
        and cv.method == "nested_cv"
        and cv.temporal
        and config.get("split_strategy") != "temporal"
    ):
        raise ValueError("Nested temporal CV requires a temporal final holdout.")
    if cv.group_column and config.get("stratify", False):
        raise ValueError("Group holdout uses whole groups; set stratify=false.")


def _preview_window(checked: dict[str, Any]) -> tuple[str, Any, Any]:
    """Describe runtime and explicit time boundaries without reading source data."""
    window = training_window_mode(checked)
    observation_window = f"[{checked.get('start')}, {checked.get('cutoff')})"
    holdout_start = checked.get("holdout_start")
    result_cutoff = checked.get("result_cutoff")
    if window == "rolling_calendar":
        observation_window = "completed calendar months at invocation"
        if checked.get("split_strategy") == "temporal":
            months = checked.get("holdout_months", 1)
            holdout_start = (
                "last completed calendar month"
                if months == 1
                else f"last {months} completed calendar months"
            )
    elif window == "rolling_days":
        observation_window = (
            f"last {checked['lookback_days']} elapsed UTC days before invocation (exclusive)"
        )
        if checked.get("split_strategy") == "temporal":
            holdout_start = f"invocation time UTC minus {checked['holdout_days']} elapsed days"
    if checked.get("filter_unavailable_results") and result_cutoff is None:
        result_cutoff = (
            f"invocation time UTC minus {checked.get('result_availability_lag_hours', 0)} "
            "elapsed hours (inclusive)"
        )
    return observation_window, holdout_start, result_cutoff


def _preview_training_source(checked: dict[str, Any], training_status: str) -> list[str]:
    """Describe source selection, resource bounds and the final holdout."""
    sample = checked.get("training_sample_rows")
    window = training_window_mode(checked)
    version = checked.get("training_version")
    if version is None:
        version = "latest snapshot at invocation"
    observation_window, holdout_start, result_cutoff = _preview_window(checked)
    return [
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
        "Job clocks and pause settings live in Bundle variables, independently of data selection. "
        "The same train action runs manually or at any cron frequency; "
        "training_version=null pins the latest snapshot once per invocation.",
        "PAUSED stops clock triggers; an unchanged score can finish as a no-op. "
        "Overlapping runs queue behind the same job's single active run; "
        "scheduled scoring and lifecycle handoff share that score job.",
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
        f"Training: {training_status}",
        "Pre-split cleanup (fixed normalization and training eligibility; edit build_pre_split_steps()):",
    ]


def _preview_steps(steps: list[dict[str, Any]], empty_message: str) -> list[str]:
    """Render an ordered Core recipe consistently in either preprocessing phase."""
    if not steps:
        return [empty_message]
    return [
        f"  {index}. {step['name']} -> {step['transformer']} "
        f"{json.dumps(step.get('params', {}), sort_keys=True)}"
        for index, step in enumerate(steps, 1)
    ]


def _preview_model_and_scoring(checked: dict[str, Any], cv: LocalCVSpec) -> list[str]:
    """Describe the model, quality policy and scoring destination."""
    model = checked["pipeline"]["modeling"]
    if model["type"] == "hyperparameter_tuner":
        model_lines = _preview_search(checked, cv)
    else:
        model_lines = [
            f"Model: {model['type']} | Explicit params: {json.dumps(model.get('params', {}))}",
            "Unspecified model parameters use Core defaults.",
            f"CV: {cv.method}, {cv.folds} folds, training partition only; "
            "preprocessing refitted per fold, no parameter search."
            if cv.enabled
            else "CV: disabled (final holdout evaluation still runs).",
        ]
    return [
        *model_lines,
        *_preview_explanations(checked["pipeline"]),
        *_preview_scoring_rules(checked["pipeline"]),
        f"Promotion: {checked['promotion_policy']} | Metric: {checked['metric']} | "
        f"Threshold: {checked.get('quality_threshold')} | "
        f"Minimum improvement (absolute): {checked['min_improvement']}",
        f"Additional quality gates: {json.dumps(checked.get('quality_gates') or {}, sort_keys=True)}",
        f"Scoring source: {checked['score_source_table']}",
        f"Score model: {checked['model_name']} | {checked['score_model_selection']} | "
        f"pin={checked.get('model_version')} | handoff={checked['score_handoff']}",
        f"Prediction output: {checked['prediction_table']} | {checked['model_change_mode']}",
        "Scoring is never sampled. Validate source values and worker dependencies separately.",
    ]


def _preview_search(checked: dict[str, Any], cv: LocalCVSpec) -> list[str]:
    """Describe the validated effective search without claiming fitted results."""
    pipeline = checked["pipeline"]
    model = pipeline["modeling"]
    effective = prepare_search_pipeline(
        pipeline,
        cv,
        target_column=checked["target_column"],
        event_column=checked.get("event_column"),
    )["modeling"]
    base = model["base_model"]
    strategy = model.get("strategy", "random")
    source = (
        "Core default search space" if not model.get("search_space") else "explicit search space"
    )
    space = effective["search_space"]
    count = math.prod(len(values) for values in space.values())
    budget = (
        f"{count} candidates | max_candidates={effective.get('max_candidates', 1000)}"
        if strategy in {"grid", "halving_grid"}
        else f"n_trials={effective.get('n_trials', 10)} | {len(space)} search axes"
    )
    lines = [
        f"Model: {base['type']} | Explicit params: {json.dumps(base.get('params', {}), sort_keys=True)}",
        "Unspecified model parameters use Core defaults.",
        f"Search: {strategy} | {source} | {budget}; preprocessing refitted per candidate fold.",
        f"Search objective: {model['metric']} (Core tuning on training partition only). "
        f"Final holdout and promotion metric: {checked['metric']}.",
        f"Search random_state={effective.get('random_state', 42)} | "
        f"cv_random_state={cv.random_state} (independent seeds).",
    ]
    _append_search_cv_preview(lines, cv)
    if effective.get("tune_threshold"):
        lines.append(
            "Decision threshold: binary inner out-of-fold selection; outer and final holdout labels stay untouched."
        )
    if strategy == "optuna" and effective.get("timeout") is not None:
        lines.append(
            f"Optuna timeout: {effective['timeout']} seconds is a soft study limit; "
            "it does not interrupt an in-flight fit."
        )
    return lines


def _append_search_cv_preview(lines: list[str], cv: LocalCVSpec) -> None:
    """Explain the effective search fold policy and repeated nested-search budgets."""
    if cv.enabled:
        lines.append(f"Search CV: {cv.method}, {cv.folds} folds, training partition only.")
    else:
        lines.append("Search CV: disabled; Core uses one single training-only shuffle split.")
    if cv.method == "nested_cv" and cv.enabled:
        inner = cv.inner_folds or (min(3, cv.folds - 1) if cv.folds > 2 else 2)
        lines.append(
            f"Nested CV: independent {inner}-fold inner search inside each of {cv.folds} outer folds, "
            f"then a separate final training search; policy={cv.nested_type}. Search budgets apply to each search."
        )

    if cv.temporal:
        lines.append(
            f"Temporal CV: gap={cv.gap} rows, test_size={cv.test_size}, max_train_size={cv.max_train_size}; stable event ordering."
        )
    if cv.group_column:
        lines.append(
            f"Group CV: {cv.group_column} is split metadata; final holdout isolates whole groups."
        )


def _preview_explanations(pipeline: dict[str, Any]) -> list[str]:
    """Report the opted-in SHAP sample, feature and display budgets."""
    settings = pipeline.get("explainability")
    if settings is None:
        return ["Explanations: disabled."]
    samples = settings.get("max_samples", 100)
    features = settings.get("max_features", 30)
    display = settings.get("max_display_samples", min(10, samples))
    return [
        f"Explanations: SHAP, max_samples={samples}, max_features={features}, "
        f"max_display_samples={display}; computed from the fitted training artifact when available."
    ]


def preview_workflow_config(config: dict[str, Any], *, action: str = "score") -> str:
    """Describe resolved settings offline using the same preflight as job execution.

    The default checks the configuration without requiring manual training pins.
    Pass ``action='train'`` to validate training selection as well.
    This cannot check source values, installed worker dependencies or permissions.
    """
    checked = validate_workflow_config(config, action=action)
    cv = LocalCVSpec.from_workflow(checked)
    training_status = "configured (source data and permissions not checked)"
    try:
        validate_workflow_config(checked, action="train")
    except ValueError as exc:
        training_status = f"needs configuration: {exc}"
    lines = _preview_training_source(checked, training_status)
    lines.extend(
        _preview_steps(checked.get("pre_split_steps", []), "  No pre-split cleanup steps.")
    )
    lines.append(
        "Fixed feature cleanup is saved as a pipeline prefix and applied once to raw model inputs; "
        "training row exclusions are not repeated during scoring."
    )
    lines.append("Fold-local preprocessing (after final split; edit build_preprocessing()):")
    lines.extend(
        _preview_steps(checked["pipeline"].get("preprocessing", []), "  No preprocessing steps.")
    )
    lines.extend(_preview_model_and_scoring(checked, cv))
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
    _migrate_selection_policy(migrated)
    if "score_handoff" in migrated and migrated["score_handoff"] != score_handoff:
        raise ValueError("Migration must not silently change the existing score_handoff.")
    migrated.update(config_version=1, task=task, score_handoff=score_handoff)
    migrated.setdefault("inference_mode", "local")
    # Binding names and manual training dates remain a caller-owned step.
    _choice(migrated, "task", {"regression", "classification"})
    _choice(migrated, "score_handoff", {"disabled", "after_alias_change"})
    _choice(migrated, "score_model_selection", {"champion", "pinned_version"})
    _choice(migrated, "promotion_policy", {"automatic", "manual_approval"})
    return migrated


def _migrate_selection_policy(migrated: dict[str, Any]) -> None:
    """Translate legacy policy fields without merging contradictory generations."""
    if "model_selection_mode" in migrated:
        if {"score_model_selection", "promotion_policy"}.intersection(migrated):
            raise ValueError("Remove mixed legacy/new policies before migration.")
        legacy = _choice(migrated, "model_selection_mode", {"pinned_version", "auto_champion"})
        migrated.pop("model_selection_mode")
        migrated.update(
            score_model_selection="champion" if legacy == "auto_champion" else "pinned_version",
            promotion_policy="automatic" if legacy == "auto_champion" else "manual_approval",
        )


def validate_deployed_contract(config: dict[str, Any], parameters: dict[str, str]) -> None:
    """Require notebook/job generation to agree with handoff and recovery policy."""
    if parameters.get("workflow_contract") not in {"2", "3"} or parameters.get(
        "deployed_score_handoff"
    ) != config.get("score_handoff"):
        raise ValueError(
            "Project and job definitions disagree; regenerate/redeploy the Bundle together."
        )
    recovery = str(config.get("auto_rebuild_on_cdf_expiry", False)).lower()
    if parameters.get("deployed_auto_rebuild_on_cdf_expiry", "false") != recovery:
        raise ValueError(
            "Project and recovery tasks disagree; regenerate/redeploy the Bundle together."
        )


def _preview_scoring_rules(pipeline: dict[str, Any]) -> list[str]:
    """Show saved policy declarations without executing eligibility or output callbacks."""
    config = pipeline.get("project_scoring")
    if config is None:
        return ["Project scoring rules: disabled (ordinary prediction schema)."]
    return [
        "Project scoring rules: " + json.dumps(config, sort_keys=True),
        "Every input key receives predicted/excluded status; exclusions carry reasons.",
        "Rules affect scoring only; training filters and holdout metrics remain independent.",
    ]
