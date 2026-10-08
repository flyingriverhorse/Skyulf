"""Resolve and replay fixed pre-split recipes as explicit scoring eligibility."""

from copy import deepcopy
from typing import Any

import pandas as pd
import polars as pl

from .....inference.project_code import (
    is_registered_project_step,
    load_project_module,
    project_step_source,
)
from .....preprocessing.function_steps import FILTER_STEP, resolve_function
from ...training.fitting.local_pre_split import FIXED_TYPES, projected_fixed_steps


def resolve_pre_split_scoring(
    request: dict[str, Any], workflow: dict[str, Any]
) -> dict[str, Any] | None:
    """Freeze reuse switches into saved steps, rejecting unknown scoring dependencies."""
    fields = {"reuse_pre_split", "skip_target_steps"}
    if (
        set(request) not in (fields, fields | {"eligibility", "outputs"})
        or request["reuse_pre_split"] is not True
    ):
        raise ValueError("Pre-split scoring requires reuse_pre_split=True and skip_target_steps.")
    skip = request["skip_target_steps"]
    if type(skip) is not bool:
        raise ValueError("SKIP_TARGET_PRE_SPLIT_STEPS must be boolean.")
    custom = {key: deepcopy(request.get(key, [])) for key in ("eligibility", "outputs")}
    steps = workflow["pre_split_steps"]
    if not steps:
        return custom if "eligibility" in request else None
    target = workflow["target_column"]
    inputs = tuple(workflow["input_columns"])
    selected, skipped = _select_steps(steps, target, inputs, skip)
    return {
        **custom,
        "pre_split": {
            "steps": selected,
            "target_column": target,
            "engine": workflow.get("engine", "pandas"),
            "skipped_target_steps": skipped,
        },
    }


def _step_columns(step: dict[str, Any], target: str) -> list[str]:
    """Use the existing admission rules so scoring cannot admit learned pre-split steps."""
    from ...training.fitting.local_retraining import (  # noqa: PLC0415 - preserve lazy dependency boundary
        validate_pre_split_step,
    )

    return list(validate_pre_split_step(step, 0, target, ()))


def _select_steps(
    steps: list[dict[str, Any]], target: str, inputs: tuple[str, ...], skip: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    """Skip whole target-reading filters and project fixed edits onto scoring inputs."""
    selected, skipped = [], []
    for step in steps:
        columns = _step_columns(step, target)
        if target in columns:
            if not skip:
                raise ValueError(
                    f"Pre-split step {step['name']!r} reads target {target!r}. "
                    "Set SKIP_TARGET_PRE_SPLIT_STEPS=True or SCORING_MODE='custom'."
                )
            skipped.append(step["name"])
            if step["transformer"] not in FIXED_TYPES:
                continue
        _require_input_columns([name for name in columns if name != target], inputs, step["name"])
        if step["transformer"] in FIXED_TYPES:
            selected.extend(projected_fixed_steps((step,), inputs))
        else:
            selected.append(deepcopy(step))
    return selected, skipped


def _require_input_columns(columns: list[str], inputs: tuple[str, ...], name: str) -> None:
    """Fail early when a training-only source field cannot be supplied by the scorer."""
    missing = sorted(set(columns) - set(inputs))
    if missing:
        raise ValueError(
            f"Reused pre-split step {name!r} needs scoring input_columns {missing}. "
            "Include them in input_columns (drop from model features in preprocessing if needed) "
            "or select SCORING_MODE='custom'."
        )


def validate_pre_split_scoring(config: Any, module: Any) -> None:
    """Revalidate saved steps and restore custom registrations from saved source only."""
    fields = {"steps", "target_column", "engine", "skipped_target_steps"}
    if type(config) is not dict or set(config) != fields:
        raise ValueError("Invalid saved pre-split scoring fields.")
    if config["engine"] not in ("pandas", "polars"):
        raise ValueError("Pre-split scoring engine must be pandas or polars.")
    if not isinstance(config["target_column"], str) or not config["target_column"]:
        raise ValueError("Pre-split scoring requires a target column.")
    if type(config["steps"]) is not list or type(config["skipped_target_steps"]) is not list:
        raise ValueError("Pre-split scoring steps and skipped names must be lists.")
    _register_saved_filters(config["steps"], module)
    for step in config["steps"]:
        if config["target_column"] in _step_columns(step, config["target_column"]):
            raise ValueError("Saved pre-split scoring must not read target values.")


def _register_saved_filters(steps: list[dict[str, Any]], module: Any) -> None:
    """Restore project class registrations while keeping saved parameters authoritative."""
    custom = [step for step in steps if "pre_split" in step]
    owner = _saved_filter_owner(module, custom)
    _register_filter_classes(
        [step for step in custom if step["transformer"] != FILTER_STEP], module
    )
    if any(not _owned_by(project_step_source(step), owner.__name__) for step in custom):
        raise ValueError("Scoring filters must belong to the saved project package.")
    _register_filter_functions(custom)


def _register_filter_functions(steps: list[dict[str, Any]]) -> None:
    """Import only saved filter functions before admission, without rebuilding recipes."""
    for step in steps:
        if step["transformer"] != FILTER_STEP:
            continue
        try:
            function = resolve_function(step["params"]["function"])
        except (KeyError, ImportError, AttributeError, TypeError) as exc:
            raise ValueError(
                "Saved pre-split function is absent from captured project source."
            ) from exc
        if not callable(function):
            raise ValueError("Saved pre-split function must be callable.")


def _register_filter_classes(classes: list[dict[str, Any]], module: Any) -> None:
    """Rebuild class-based filters so their saved project registrations exist again."""
    if not classes or all(is_registered_project_step(step["transformer"]) for step in classes):
        return
    factory = getattr(module, "build_pre_split_steps", None)
    if not callable(factory):
        raise ValueError("Saved project must define build_pre_split_steps for custom filters.")
    factory()


def _owned_by(source: str, owner: str) -> bool:
    """Accept a filter defined in the saved project module or one of its submodules."""
    return source == owner or source.startswith(owner + ".")


def _saved_filter_owner(module: Any, custom: list[dict[str, Any]]) -> Any:
    """Admit canonical shared filters only when their exact builder and steps are captured."""
    source = getattr(module, "__skyulf_common_source__", None)
    if source is None or not custom:
        return module
    owner = load_project_module(source)
    factory = getattr(owner, "build_pre_split_steps", None)
    if not callable(factory) or factory is not getattr(module, "build_pre_split_steps", None):
        raise ValueError("Shared scoring filters require the captured canonical builder.")
    expected = factory()
    if not isinstance(expected, list) or any(step not in expected for step in custom):
        raise ValueError("Shared scoring filters differ from the captured canonical recipe.")
    return owner


def pre_split_exclusion_reasons(frame: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """Run the saved recipe on a copy and report the first step excluding each position."""
    from ...training.fitting.local_retraining import (  # noqa: PLC0415 - avoid import cycle
        apply_pre_split_step,
    )

    reasons = pd.Series(pd.NA, index=frame.index, dtype="string")
    key = "__skyulf_scoring_position"
    while key in frame.columns:
        key += "_"
    working = frame.assign(**{key: range(len(frame))})
    native = pl.from_pandas(working) if config["engine"] == "polars" else working
    for step in config["steps"]:
        if len(native) == 0:
            break
        before = native[key].to_list()
        native = apply_pre_split_step(
            native, step, keys=[key], target_column=config["target_column"]
        )
        after = set(native[key].to_list())
        removed = [position for position in before if position not in after]
        reasons.loc[removed] = f"pre_split:{step['name']}"
    return reasons
