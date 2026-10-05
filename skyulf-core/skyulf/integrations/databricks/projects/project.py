"""Resolve trusted project Python recipes into the existing Core config."""

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from ....inference.project_code import (
    MAX_PROJECT_SOURCE_BYTES,
    load_project_module,
    project_source_digest,
)
from ....inference.project_scoring import validate_scoring_config
from ..training.fitting.local_ensemble import ENSEMBLE_MODELS
from ..training.thresholds.decision_thresholds import threshold_policy
from ..training.tuning.local_search import bounded_space
from ..training.weights.weight_config import capture_model_weights, validate_weight_roles
from ._project_files import modeling_hook, project_source, read_source
from ._project_recipes import bind_recipe_source, recipe_label, recipe_steps


def strict_json_value(value: Any, filename: str = "ensemble.py") -> Any:
    """Copy only finite JSON values from trusted project hook output."""
    if type(value) is dict:
        return _strict_json_object(value, filename)
    if type(value) is list:
        return [strict_json_value(item, filename) for item in value]
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"{filename} must return finite JSON values.")


def _strict_json_object(value: dict[Any, Any], filename: str = "ensemble.py") -> dict[str, Any]:
    """Copy nested hook mappings while rejecting non-string JSON keys."""
    if any(type(key) is not str for key in value):
        raise ValueError(f"{filename} must return a JSON object with string keys.")
    return {key: strict_json_value(item, filename) for key, item in value.items()}


def _load_single_model(config: dict[str, Any], path: Path) -> dict[str, Any]:
    """Resolve editable model parameters once, keeping saved pipelines self-contained."""
    if config.get("training_layout", "single_model") != "single_model":
        return config
    hook = modeling_hook(path, "single_model.py")
    if not hook.is_file():
        return config
    if config["pipeline"].get("modeling") != {}:
        raise ValueError("Configure modeling in single_model.py; leave pipeline.modeling empty.")
    source = read_source(hook)
    module = load_project_module(source)
    factory = getattr(module, "build_modeling", None)
    if not callable(factory):
        raise ValueError("single_model.py must define build_modeling().")
    modeling = strict_json_value(factory(), hook.name)
    if type(modeling) is not dict:
        raise ValueError("single_model.py build_modeling() must return a JSON object.")
    if len(json.dumps(modeling, allow_nan=False).encode("utf-8")) > MAX_PROJECT_SOURCE_BYTES:
        raise ValueError("single_model.py returned more than 64 KiB of parameters.")
    result = deepcopy(config)
    result["pipeline"]["modeling"] = modeling
    if hasattr(module, "DECISION_THRESHOLD"):
        result["pipeline"]["decision_threshold"] = strict_json_value(
            module.DECISION_THRESHOLD, hook.name
        )
        threshold_policy(result["pipeline"])
    if hasattr(module, "WEIGHT_COLUMN"):
        result.update(capture_model_weights(source, {"weight_column": module.WEIGHT_COLUMN}))
    return result


def _load_ensemble_hook(result: dict[str, Any], preprocessing_path: Path) -> None:
    """Apply an optional ensemble recipe before resolving any tuning override."""
    modeling = result["pipeline"].get("modeling", {})
    selected = (
        modeling.get("base_model", {})
        if modeling.get("type") == "hyperparameter_tuner"
        else modeling
    )
    if selected.get("type") not in ENSEMBLE_MODELS:
        return
    hook_path = modeling_hook(preprocessing_path, "ensemble.py")
    if not hook_path.is_file():
        return
    with hook_path.open("rb") as stream:
        payload = stream.read(MAX_PROJECT_SOURCE_BYTES + 1)
    if len(payload) > MAX_PROJECT_SOURCE_BYTES:
        raise ValueError("ensemble.py source exceeds 64 KiB.")
    source = payload.decode("utf-8")
    module = load_project_module(source)
    factory = getattr(module, "build_ensemble_params", None)
    if not callable(factory):
        raise ValueError("ensemble.py must define build_ensemble_params().")
    overrides = factory(model_type=selected["type"])
    if overrides is not None:
        if type(overrides) is not dict:
            raise ValueError("build_ensemble_params() must return a JSON object or None.")
        overrides = strict_json_value(overrides)
        if len(json.dumps(overrides, allow_nan=False).encode("utf-8")) > MAX_PROJECT_SOURCE_BYTES:
            raise ValueError("ensemble.py returned more than 64 KiB of parameters.")
        params = selected.get("params", {})
        if type(params) is not dict:
            raise ValueError("base model params must be a JSON object.")
        selected["params"] = {**params, **overrides}
    result["pipeline"]["ensemble_python_source"] = source
    result["pipeline"]["ensemble_python_sha256"] = project_source_digest(source)


def _load_search_hook(result: dict[str, Any], preprocessing_path: Path) -> None:
    """Resolve an optional sibling tuning factory only for new tuner recipes."""
    modeling = result["pipeline"].get("modeling", {})
    if modeling.get("type") != "hyperparameter_tuner":
        return
    hook_path = modeling_hook(preprocessing_path, "tuning.py")
    if not hook_path.is_file():
        return
    with hook_path.open("rb") as stream:
        payload = stream.read(MAX_PROJECT_SOURCE_BYTES + 1)
    if len(payload) > MAX_PROJECT_SOURCE_BYTES:
        raise ValueError("tuning.py source exceeds 64 KiB.")
    source = payload.decode("utf-8")
    digest = project_source_digest(source)
    module = load_project_module(source)
    factory = getattr(module, "build_search_space", None)
    if not callable(factory):
        raise ValueError("tuning.py must define build_search_space().")
    selected = modeling.get("base_model", {})
    space = factory(
        model_type=selected["type"],
        strategy=modeling.get("strategy", "random"),
        params=deepcopy(selected.get("params", {})),
    )
    if space is not None:
        try:
            modeling["search_space"] = bounded_space(space)
        except ValueError as exc:
            raise ValueError(f"tuning.py returned invalid search_space: {exc}") from exc
    result["pipeline"]["search_python_source"] = source
    result["pipeline"]["search_python_sha256"] = digest


def load_project_workflow(
    config: dict[str, Any],
    path: str | Path,
    *,
    preprocessing_recipe: str | None = None,
    pre_split_recipe: str | None = None,
    weight_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use Python-defined steps for training/preview and capture their exact source.

    Score and lifecycle approval load the saved artifact instead of this file.
    A nonempty JSON chain is rejected rather than silently overwritten.
    Named selections call the corresponding builder with ``recipe=...`` and
    remain bound into saved source for artifact and training-plan replay.
    ``None`` preserves the original zero-argument builder contract.
    Single-model projects may provide ``single_model.py:build_modeling()`` with
    an empty JSON modeling object. Resolved parameters persist in the existing
    pipeline configuration; saved plans replay that configuration directly.
    """
    if config.get("training_layout") == "model_competition":
        from .competition_project import load_competition_project  # noqa: PLC0415

        if preprocessing_recipe is not None or pre_split_recipe is not None:
            raise ValueError(
                "Competition selects candidate preprocessing and shared default pre-split."
            )
        return load_competition_project(config, path)
    validate_project_steps(config)
    config = {**deepcopy(config), **deepcopy(weight_settings or {})}
    config = _load_single_model(config, Path(path))
    validate_weight_roles(config)
    return resolve_project_workflow(
        config,
        path,
        project_source(Path(path)),
        preprocessing_recipe=preprocessing_recipe,
        pre_split_recipe=pre_split_recipe,
    )


def validate_project_steps(config: dict[str, Any]) -> None:
    """Reject JSON step chains before reading source that would replace them."""
    if config.get("pipeline", {}).get("preprocessing"):
        raise ValueError("Configure preprocessing in the Python file; leave the JSON list empty.")
    if config.get("pre_split_steps"):
        raise ValueError("Configure pre_split_steps in the Python file; leave the JSON list empty.")


def resolve_project_workflow(
    config: dict[str, Any],
    path: str | Path,
    source: str,
    *,
    preprocessing_recipe: str | None = None,
    pre_split_recipe: str | None = None,
) -> dict[str, Any]:
    """Resolve an already captured source snapshot through the existing project hooks."""
    source = bind_recipe_source(
        source,
        {"preprocessing_recipe": preprocessing_recipe, "pre_split_recipe": pre_split_recipe},
    )
    module = load_project_module(source)
    steps = recipe_steps(
        module, "build_preprocessing", "preprocessing_recipe", preprocessing_recipe
    )
    pre_split_steps = recipe_steps(
        module, "build_pre_split_steps", "pre_split_recipe", pre_split_recipe, optional=True
    )
    result = deepcopy(config)
    result["pipeline"]["preprocessing"] = steps
    result["pipeline"]["project_python_source"] = source
    result["pipeline"]["feature_recipes"] = {
        "preprocessing": recipe_label(module, "build_preprocessing", preprocessing_recipe),
        "pre_split": recipe_label(module, "build_pre_split_steps", pre_split_recipe),
    }
    result["pre_split_steps"] = deepcopy(pre_split_steps)
    _load_scoring_hook(result, module, source)
    _load_ensemble_hook(result, Path(path))
    _load_search_hook(result, Path(path))
    return result


def _load_scoring_hook(result: dict[str, Any], module: Any, source: str) -> None:
    """Resolve optional scoring rules once and bind their parameters to the model."""
    factory = getattr(module, "build_scoring", None)
    if factory is None:
        return
    if not callable(factory):
        raise ValueError("build_scoring must be a function returning a scoring policy.")
    config = factory()
    if isinstance(config, dict) and "reuse_pre_split" in config:
        from ..scoring.shared.scoring_pre_split import resolve_pre_split_scoring  # noqa: PLC0415

        config = resolve_pre_split_scoring(config, result)
    result["pipeline"].pop("project_scoring", None)
    if config is not None:
        result["pipeline"]["project_scoring"] = validate_scoring_config(config, source)
