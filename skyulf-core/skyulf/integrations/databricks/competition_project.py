"""Freeze bounded model competition recipes under one shared workflow contract."""

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from ...inference.project_code import (
    MAX_PROJECT_SOURCE_BYTES,
    load_project_module,
    project_source_digest,
)
from ._project_files import modeling_hook, project_source, read_source, renamed_modeling_hook
from .competition_evaluation import competition_metric, validate_competition_preprocessing
from .local_cv import LocalCVSpec
from .local_search import base_model_config
from .project import resolve_project_workflow, strict_json_value, validate_project_steps
from .weight_config import capture_model_weights, validate_weight_roles
from .workflow_config import WORKFLOW_FIELDS, validate_workflow_pipeline

_CANDIDATE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")


def _competition_settings(config: dict[str, Any]) -> tuple[LocalCVSpec, str, int]:
    """Require shared CV and bounded candidate/trial limits without coercion."""
    if config.get("training_layout") != "model_competition":
        raise ValueError("Competition requires training_layout=model_competition.")
    if config.get("cv_enabled") is not True:
        raise ValueError("Model competition requires cv_enabled=true.")
    limits = {
        "competition_max_candidates": (8, 2, 32),
        "competition_max_trials": (100, 1, 10_000),
    }
    for field, (default, minimum, maximum) in limits.items():
        value = config.get(field, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{field} must be an integer from {minimum} to {maximum}.")
    metric = competition_metric(config.get("metric", ""), config.get("task", ""))
    return LocalCVSpec.from_workflow(config), metric, config.get("competition_max_candidates", 8)


def _candidate_mapping(value: Any, limit: int) -> dict[str, Any]:
    """Admit explicit safe identities and at least two requested competitors."""
    if type(value) is not dict or not 2 <= len(value) <= limit:
        raise ValueError(f"Competition candidates must contain 2 to {limit} named candidates.")
    for name in value:
        if not isinstance(name, str) or _CANDIDATE_NAME.fullmatch(name) is None:
            raise ValueError("Competition candidate names need a letter and safe ASCII characters.")
    return value


def _candidate_definition(value: Any) -> dict[str, Any]:
    """Keep candidate overrides limited to the model and a preprocessing recipe."""
    if type(value) is not dict or "modeling" not in value:
        raise ValueError("Each candidate must define modeling.")
    if set(value) - {"modeling", "preprocessing_recipe"}:
        raise ValueError(
            "Unknown candidate setting; shared workflow settings cannot be overridden."
        )
    recipe = value.get("preprocessing_recipe", "default")
    if not isinstance(recipe, str) or not recipe.strip():
        raise ValueError("Candidate preprocessing_recipe must be a nonempty name.")
    return value


def _bind_candidate_metric(pipeline: dict[str, Any], metric: str) -> None:
    """Bind omitted tuner objectives and reject a different selection metric."""
    selected = base_model_config(pipeline)
    if set(selected) - {"type", "params", "node_id"}:
        raise ValueError("Unknown candidate model setting; shared workflow settings are immutable.")
    modeling = pipeline["modeling"]
    if modeling.get("type") == "hyperparameter_tuner":
        if "metric" in modeling and modeling["metric"] != metric:
            raise ValueError("Candidate metric must match the shared competition metric.")
        modeling["metric"] = metric


def _validate_candidate_pipeline(
    pipeline: dict[str, Any], config: dict[str, Any], cv: LocalCVSpec, metric: str
) -> None:
    """Validate task, preprocessing and authoritative CV without mutating saved recipes."""
    if set(pipeline) & WORKFLOW_FIELDS:
        raise ValueError("Candidate pipelines cannot override shared workflow settings.")
    validate_competition_preprocessing(pipeline)
    copied = deepcopy(pipeline)
    _bind_candidate_metric(copied, metric)
    validate_workflow_pipeline({"pipeline": copied}, config["task"])
    cv.validate_pipeline(
        copied,
        target_column=config["target_column"],
        event_column=config.get("event_column"),
    )


def validate_competition_config(config: dict[str, Any]) -> None:
    """Preflight resolved SDK candidates before source, tracking or registry access."""
    cv, metric, limit = _competition_settings(config)
    competition = config.get("competition")
    if type(competition) is not dict:
        raise ValueError("Competition requires resolved candidates.")
    candidates = _candidate_mapping(competition.get("candidates"), limit)
    for candidate in candidates.values():
        if type(candidate) is not dict or set(candidate) != {"pipeline"}:
            raise ValueError("Resolved candidate settings must contain only pipeline.")
        if type(candidate["pipeline"]) is not dict:
            raise ValueError("Candidate pipeline must be a JSON object.")
        _validate_candidate_pipeline(candidate["pipeline"], config, cv, metric)


def _load_candidates(
    path: Path, task: str, limit: int
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Capture and execute one bounded candidates hook with strict JSON output."""
    hook = renamed_modeling_hook(modeling_hook(path, "model_competition.py"), "candidates.py")
    source = read_source(hook)
    module = load_project_module(source)
    factory = getattr(module, "build_candidates", None)
    if not callable(factory):
        raise ValueError(f"{hook.name} must define build_candidates(task).")
    candidates = strict_json_value(factory(task=task), hook.name)
    if len(json.dumps(candidates, allow_nan=False).encode("utf-8")) > MAX_PROJECT_SOURCE_BYTES:
        raise ValueError(f"{hook.name} returned more than 64 KiB of configuration.")
    settings = {"weight_column": module.WEIGHT_COLUMN} if hasattr(module, "WEIGHT_COLUMN") else {}
    return _candidate_mapping(candidates, limit), source, capture_model_weights(source, settings)


def load_competition_project(config: dict[str, Any], path: str | Path) -> dict[str, Any]:
    """Resolve each model through the existing project hooks and freeze common eligibility."""
    cv, metric, limit = _competition_settings(config)
    validate_project_steps(config)
    candidates, source, weights = _load_candidates(Path(path), config["task"], limit)
    config = {**deepcopy(config), **weights}
    validate_weight_roles(config)
    shared_source = _competition_source(project_source(Path(path)))
    result = deepcopy(config)
    resolved = {}
    common_steps = None
    for name, value in candidates.items():
        candidate = _candidate_definition(value)
        local = deepcopy(config)
        local["training_layout"] = "single_model"
        local["pipeline"]["modeling"] = deepcopy(candidate["modeling"])
        _bind_candidate_metric(local["pipeline"], metric)
        loaded = resolve_project_workflow(
            local,
            path,
            shared_source,
            preprocessing_recipe=candidate.get("preprocessing_recipe", "default"),
        )
        _validate_candidate_pipeline(loaded["pipeline"], config, cv, metric)
        if common_steps is None:
            common_steps = loaded["pre_split_steps"]
            result["pipeline"] = deepcopy(loaded["pipeline"])
        elif loaded["pre_split_steps"] != common_steps:
            raise ValueError("Competition candidates must use identical shared pre_split_steps.")
        resolved[name] = {"pipeline": loaded["pipeline"]}
    result["pre_split_steps"] = common_steps
    result["competition"] = {
        "candidates": resolved,
        "candidates_source": source,
        "candidates_sha256": project_source_digest(source),
    }
    return result


def _competition_source(source: str) -> str:
    """Embed one snapshot while keeping shared filter classes under its canonical identity."""
    return (
        f"__skyulf_common_source__ = {source!r}\n"
        "exec(compile(__skyulf_common_source__, __file__, 'exec'), globals())\n"
        "from skyulf.inference.project_code import load_project_module as _skyulf_load_common\n"
        "_skyulf_common_module = _skyulf_load_common(__skyulf_common_source__)\n"
        "if callable(getattr(_skyulf_common_module, 'build_pre_split_steps', None)):\n"
        "    build_pre_split_steps = _skyulf_common_module.build_pre_split_steps\n"
    )
