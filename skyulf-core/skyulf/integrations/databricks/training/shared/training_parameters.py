"""Readable experiment parameters derived from the fitted model and pinned split."""

from __future__ import annotations

import math
from datetime import UTC
from typing import TYPE_CHECKING, Any

import numpy as np

from ..fitting.ensemble import ENSEMBLE_MODELS
from ..tuning.search import base_model_config
from ..tuning.search_results import parameter_preview

if TYPE_CHECKING:
    from .....inference.fitted_pipeline import FittedPipelineArtifact
    from ..fitting.candidate import TrainingSpec


def _parameter_value(value: Any) -> Any:
    """Describe constructor values without serializing fitted state or object reprs."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, dict):
        return {str(key): _parameter_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_parameter_value(item) for item in value]
    return {"python_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _ensemble_parameters(model: Any, selected: dict[str, Any]) -> dict[str, Any]:
    """Report actual member order and distinguish internal stacking CV from search CV."""
    if selected["type"] not in ENSEMBLE_MODELS:
        return {}
    params = {
        "strategy": selected["type"].split("_")[0],
        "base_models": [name for name, _estimator in model.estimators],
        "calibrate_base_models": selected.get("params", {}).get("calibrate_base_models", False),
    }
    for name in ("voting", "weights", "cv", "passthrough", "stack_method"):
        if hasattr(model, name):
            params[name] = _parameter_value(getattr(model, name))
    if hasattr(model, "final_estimator_"):
        final = model.final_estimator_
        params["final_estimator"] = type(final).__name__
        params["final_estimator_params"] = _parameter_value(final.get_params(deep=True))
    return params


def _split_parameters(spec: TrainingSpec) -> dict[str, Any]:
    """Expose only active split controls; temporal membership uses explicit UTC boundaries."""
    params: dict[str, Any] = {"strategy": spec.split_strategy, "group_column": spec.group_column}
    if spec.split_strategy == "random":
        params.update(
            test_size=spec.test_size, random_state=spec.random_state, stratify=spec.stratify
        )
    else:
        params["event_column"] = spec.event_column
        for name in ("start", "holdout_start", "cutoff"):
            value = getattr(spec, name)
            params[name] = None if value is None else value.astimezone(UTC).isoformat()
    return params


def _preview(value: Any, section: str) -> str:
    """Use native text for scalar strings and JSON for structured parameter values."""
    if isinstance(value, str) and len(value.encode("utf-8")) <= 500:
        return value
    return parameter_preview(value, section, artifact_file="training_parameters.json")


def log_training_parameters(
    run: Any,
    artifact: FittedPipelineArtifact,
    spec: TrainingSpec,
    config: dict[str, Any],
) -> None:
    """Persist comparison-friendly parameters with complete JSON summaries beside them.

    Estimator-valued constructor arguments are represented by their Python type;
    deep get_params entries retain their individual constructor settings. The
    fitted model artifact remains authoritative for non-JSON objects.
    """
    selected = base_model_config(dict(artifact.pipeline.config))
    estimator = artifact.pipeline.model_estimator
    assert estimator is not None
    model = estimator._unwrap_tuned_model()
    groups: dict[str, Any] = {
        "model_params": _parameter_value(model.get_params(deep=True)),
        "ensemble": _ensemble_parameters(model, selected),
        "split": _split_parameters(spec),
    }
    if "training_weights" in artifact.pipeline.config:
        groups["training_weights"] = dict(artifact.pipeline.config)["training_weights"]
    recipes = config.get("feature_recipes", {})
    summary = {
        "model_type": selected["type"],
        "preprocessing_recipe": recipes.get("preprocessing", "inline"),
        "pre_split_recipe": recipes.get("pre_split", "inline"),
        "preprocessing_steps": [step["name"] for step in config.get("preprocessing", [])],
        "pre_split_steps": [step["name"] for step in spec.pre_split_steps],
        **groups,
    }
    if "class_weight" in selected.get("params", {}):
        summary["configured_class_weight"] = _parameter_value(selected["params"]["class_weight"])
    run.client.log_dict(run.run_id, summary, "training_parameters.json")
    params = {name: _preview(value, name) for name, value in summary.items()}
    for group, values in groups.items():
        separator = "_" if group == "split" else "."
        for name, value in values.items():
            key = f"{group}{separator}{name}"
            if len(key) <= 250:
                params[key] = _preview(value, f"{group}.{name}")
    run.log_params(params)
