"""Read optional YAML settings with explicit ownership and bounded, inert parsing."""

import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from ._project_files import modeling_hook, read_source
from .workflow_config import WORKFLOW_FIELDS

INFERENCE_FIELDS = {
    "inference_mode",
    "spark_udf_env_manager",
    "spark_udf_prediction_batch_rows",
    "score_source_table",
    "prediction_table",
    "model_version",
    "model_change_mode",
    "auto_rebuild_on_cdf_expiry",
    "score_model_selection",
    "score_handoff",
}
MODEL_FIELDS = {
    "model",
    "tuning",
    "decision_threshold",
    "features_path",
    "preprocessing_recipe",
    "pre_split_recipe",
    "explainability",
}
TRAINING_FIELDS = (
    WORKFLOW_FIELDS
    - INFERENCE_FIELDS
    - {
        "pipeline",
        "competition",
        "weights_python_source",
        "weights_python_sha256",
        "reserved_weight_columns",
        "pre_split_steps",
    }
)
TUNING_FIELDS = {
    "strategy",
    "metric",
    "n_trials",
    "random_state",
    "tune_threshold",
    "max_candidates",
    "timeout",
    "strategy_params",
    "search_space",
}


def _unique_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys before a JSON or YAML setting can disappear."""
    result = {}
    for key, value in pairs:
        if type(key) is not str:
            raise ValueError("Configuration mapping keys must be strings.")
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}.")
        result[key] = value
    return result


def _json_value(value: Any, depth: int = 0) -> None:
    """Limit YAML to finite JSON values without timestamps, tags or recursive aliases."""
    if depth > 32:
        raise ValueError("YAML nesting must not exceed 32 levels.")
    if type(value) is dict:
        for item in value.values():
            _json_value(item, depth + 1)
    elif type(value) is list:
        for item in value:
            _json_value(item, depth + 1)
    elif (
        value is not None
        and type(value) not in {str, bool, int}
        and (type(value) is not float or not math.isfinite(value))
    ):
        raise ValueError("YAML settings require finite JSON values; quote dates as strings.")


def read_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Read at most 64 KiB of safe YAML, rejecting aliases and duplicate mapping keys."""
    import yaml  # noqa: PLC0415 - optional project-configuration dependency

    class UniqueLoader(yaml.SafeLoader):
        """Reject all duplicate keys, including nested model parameter declarations."""

        def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, Any]:
            """Build a mapping without silently replacing an earlier declaration."""
            return _unique_mapping(
                [
                    (self.construct_object(key, deep=deep), self.construct_object(value, deep=deep))
                    for key, value in node.value
                ]
            )

    source = read_source(Path(path))
    try:
        _check_tokens(yaml, source)
        result = yaml.load(source, Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"{Path(path).name}: invalid YAML: {exc}") from exc
    if type(result) is not dict:
        raise ValueError(f"{Path(path).name} must contain a mapping.")
    _json_value(result)
    return result


def _check_tokens(yaml: Any, source: str) -> None:
    """Bound parser depth before YAML builds its recursively nested node structure."""
    opening = (
        yaml.BlockMappingStartToken,
        yaml.BlockSequenceStartToken,
        yaml.FlowMappingStartToken,
        yaml.FlowSequenceStartToken,
    )
    closing = (yaml.BlockEndToken, yaml.FlowMappingEndToken, yaml.FlowSequenceEndToken)
    depth = 0
    for token in yaml.scan(source):
        if isinstance(token, yaml.AliasToken):
            raise ValueError("YAML aliases are unsupported; use defaults and model overrides.")
        if isinstance(token, opening):
            depth += 1
        elif isinstance(token, closing):
            depth -= 1
        if depth > 32:
            raise ValueError("YAML nesting must not exceed 32 levels.")


def _fields(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    """Require a mapping and reject misspelled fields before applying any defaults."""
    if type(value) is not dict:
        raise ValueError(f"{label} must be a mapping.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} settings: {', '.join(sorted(map(str, unknown)))}.")
    return value


def _version(document: dict[str, Any], label: str) -> None:
    """Version YAML independently from the existing workflow runtime contract."""
    if type(document.get("version")) is not int or document["version"] != 1:
        raise ValueError(f"{label} version must be 1.")


def _model_entry(value: Any, label: str) -> dict[str, Any]:
    """Validate declaration structure while existing model validators own parameter values."""
    entry = _fields(value, TRAINING_FIELDS | MODEL_FIELDS, label)
    if "model" in entry:
        model = _fields(entry["model"], {"type", "params"}, f"{label}.model")
        if not isinstance(model.get("type"), str) or not model["type"]:
            raise ValueError(f"{label}.model.type must be a nonempty string.")
        if type(model.get("params", {})) is not dict:
            raise ValueError(f"{label}.model.params must be a mapping.")
    if "tuning" in entry:
        _fields(entry["tuning"], TUNING_FIELDS, f"{label}.tuning")
    return entry


def read_training_config(directory: str | Path) -> dict[str, Any] | None:
    """Return validated defaults and named model overrides, or the legacy selection."""
    path = Path(directory) / "training.yml"
    if not path.exists():
        return None
    document = _fields(read_yaml_mapping(path), {"version", "defaults", "models"}, path.name)
    _version(document, path.name)
    defaults = _model_entry(document.get("defaults", {}), "training defaults")
    models = document.get("models")
    if type(models) is not dict or not models:
        raise ValueError("training.yml models must be a nonempty named mapping.")
    for name, value in models.items():
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name) is None:
            raise ValueError("YAML model names need a letter and safe ASCII characters.")
        entry = _model_entry(value, f"model {name}")
        if "model" not in defaults and "model" not in entry:
            raise ValueError(f"model {name} requires model.type or a shared default model.")
    return {"version": 1, "defaults": defaults, "models": models}


def read_inference_config(directory: str | Path) -> dict[str, Any] | None:
    """Read scoring settings and an optional model-set destination declaration."""
    path = Path(directory) / "inference.yml"
    if not path.exists():
        return None
    document = _fields(
        read_yaml_mapping(path), INFERENCE_FIELDS | {"version", "model_set"}, path.name
    )
    _version(document, path.name)
    if "model_set" in document:
        from ..model_sets.model_set_project import _validate_set_settings  # noqa: PLC0415

        _validate_set_settings(document["model_set"])
    return document


def project_training_config(path: Path) -> dict[str, Any] | None:
    """Locate YAML beside an organized src/features or legacy preprocessing module."""
    hook = modeling_hook(path, "single_model.py")
    root = hook.parent.parent.parent
    declarations = read_training_config(root / "config")
    if declarations is not None:
        _training_python_owners(root)
    return declarations


def _training_python_owners(root: Path) -> None:
    """Reject simultaneous Python model definitions at every recipe-loading entrypoint."""
    _check_python_owners(
        root,
        (
            "single_model.py",
            "model_competition.py",
            "candidates.py",
            "multi_model.py",
            "branches.py",
            "tuning.py",
            "ensemble.py",
        ),
        "training.yml",
    )


def _merge_owner(config: dict[str, Any], settings: dict[str, Any], filename: str) -> None:
    """Admit disjoint ownership only, even when duplicate values would be identical."""
    conflicts = set(config) & set(settings)
    if conflicts:
        raise ValueError(
            f"Settings defined in both workflow.json and {filename}: {sorted(conflicts)}."
        )
    config.update(deepcopy(settings))


def _check_python_owners(root: Path, filenames: tuple[str, ...], owner: str) -> None:
    """Check file presence without importing or executing editable Python code."""
    for name in filenames:
        if (root / "src/modeling" / name).exists():
            raise ValueError(f"Settings defined in both {owner} and src/modeling/{name}.")


def _check_model_owners(config: dict[str, Any], training: dict[str, Any]) -> None:
    """Keep per-model overrides inside YAML's explicitly declared shared defaults."""
    pipeline = config.get("pipeline", {})
    if pipeline.get("modeling"):
        raise ValueError("Model settings defined in both workflow.json and training.yml.")
    for entry in [training["defaults"], *training["models"].values()]:
        conflicts = (set(entry) & set(config)) | (
            set(entry) & set(pipeline) & {"decision_threshold", "explainability"}
        )
        if conflicts:
            raise ValueError(
                f"Settings defined in both workflow.json and training.yml: {sorted(conflicts)}."
            )


def read_workflow_config(path: str | Path) -> dict[str, Any]:
    """Combine disjoint JSON/YAML owners at the shared job, preview and smoke boundary."""
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_mapping)
    if type(config) is not dict:
        raise ValueError("Workflow configuration must be an object.")
    training = read_training_config(path.parent)
    if training is not None:
        _training_python_owners(path.parent.parent)
        _check_model_owners(config, training)
        _merge_owner(
            config,
            {key: value for key, value in training["defaults"].items() if key not in MODEL_FIELDS},
            "training.yml",
        )
    inference = read_inference_config(path.parent)
    if inference is not None:
        _merge_owner(
            config,
            {key: value for key, value in inference.items() if key not in {"version", "model_set"}},
            "inference.yml",
        )
        if "model_set" in inference:
            _check_python_owners(path.parent.parent, ("model_set.py",), "inference.yml")
    return config
