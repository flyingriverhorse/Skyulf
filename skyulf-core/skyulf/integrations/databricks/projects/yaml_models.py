"""Adapt compact model declarations to existing self-contained training recipes."""

from copy import deepcopy
from typing import Any

from .yaml_config import MODEL_FIELDS


def declaration_source(document: dict[str, Any]) -> str:
    """Freeze literal declarations as inert Python for existing provenance/replay contracts."""
    return f"YAML_DECLARATIONS = {document!r}\n"


def model_entries(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Apply shallow explicit overrides; nested model parameters belong to one entry."""
    return {
        name: {**deepcopy(document["defaults"]), **deepcopy(entry)}
        for name, entry in document["models"].items()
    }


def model_pipeline(entry: dict[str, Any]) -> dict[str, Any]:
    """Translate model plus optional tuning into the existing Core representation."""
    modeling = deepcopy(entry["model"])
    if "tuning" in entry:
        modeling = {
            "type": "hyperparameter_tuner",
            "base_model": modeling,
            **deepcopy(entry["tuning"]),
        }
    pipeline = {"preprocessing": [], "modeling": modeling}
    for key in ("decision_threshold", "explainability"):
        if key in entry:
            pipeline[key] = deepcopy(entry[key])
    return pipeline


def single_model(document: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Resolve one model and retain shared explainability and source-level workflow values."""
    from ..training.weights.weight_config import capture_model_weights  # noqa: PLC0415

    entries = model_entries(document)
    if len(entries) != 1:
        raise ValueError("single_model training.yml must declare exactly one model.")
    if config["pipeline"].get("modeling"):
        raise ValueError("Model settings defined in both workflow.json and training.yml.")
    entry = next(iter(entries.values()))
    if "features_path" in entry:
        raise ValueError("features_path is supported only for multi_target YAML models.")
    result = {
        **deepcopy(config),
        **{key: value for key, value in entry.items() if key not in MODEL_FIELDS},
    }
    result["pipeline"] = {**deepcopy(config["pipeline"]), **model_pipeline(entry)}
    result.update(capture_model_weights(declaration_source(document), entry))
    return result


def competition_candidates(document: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Keep competition workflow settings shared while allowing independent estimators."""
    from ..training.weights.weight_config import capture_model_weights  # noqa: PLC0415

    allowed = {"model", "tuning", "decision_threshold", "preprocessing_recipe", "explainability"}
    unsupported = set(document["defaults"]) & {"features_path", "pre_split_recipe"}
    if unsupported:
        raise ValueError(f"Competition does not support YAML defaults: {sorted(unsupported)}.")
    candidates = {}
    for name, overrides in document["models"].items():
        if set(overrides) - allowed:
            raise ValueError(
                "Competition model overrides must not change shared workflow settings."
            )
        entry = {**deepcopy(document["defaults"]), **deepcopy(overrides)}
        pipeline = model_pipeline(entry)
        candidates[name] = {
            key: pipeline[key]
            for key in ("modeling", "decision_threshold", "explainability")
            if key in pipeline
        }
        candidates[name]["preprocessing_recipe"] = entry.get("preprocessing_recipe", "default")
    source = declaration_source(document)
    return candidates, source, capture_model_weights(source, document["defaults"])


def training_branches(document: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Resolve per-target settings without sharing mutable nested dictionaries."""
    branches = {}
    for name, entry in model_entries(document).items():
        workflow = {key: value for key, value in entry.items() if key not in MODEL_FIELDS}
        workflow["pipeline"] = model_pipeline(entry)
        branches[name] = {
            "workflow": workflow,
            **{
                key: entry[key]
                for key in ("features_path", "preprocessing_recipe", "pre_split_recipe")
                if key in entry
            },
        }
    return branches, declaration_source(document)


def static_workflows(document: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build inert model configurations for smoke checks without loading custom features."""
    layout = config.get("training_layout", "single_model")
    if layout == "single_model":
        return [single_model(document, config)]
    if layout == "model_competition":
        candidates, _, _ = competition_candidates(document)
        return [
            {
                **deepcopy(config),
                "pipeline": {
                    **deepcopy(config["pipeline"]),
                    **{
                        key: value
                        for key, value in candidate.items()
                        if key != "preprocessing_recipe"
                    },
                },
            }
            for candidate in candidates.values()
        ]
    branches, _ = training_branches(document)
    return [{**deepcopy(config), **entry["workflow"]} for entry in branches.values()]
