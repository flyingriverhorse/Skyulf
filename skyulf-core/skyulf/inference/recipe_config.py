"""Resolve captured declarative recipes without reading live project configuration."""

import keyword
from copy import deepcopy
from importlib import import_module
from typing import Any

from ..config_validation import validate_preprocessing_steps


def custom_reference(value: Any) -> tuple[str, str]:
    """Require a project-relative module and a top-level factory, without expressions."""
    parts = value.split(".") if isinstance(value, str) else []
    if len(parts) < 2 or any(
        not part.isidentifier() or keyword.iskeyword(part) or part.startswith("__")
        for part in parts
    ):
        raise ValueError("custom must name a project module and factory, e.g. preprocessing.fill.")
    return ".".join(parts[:-1]), parts[-1]


def _custom_step(package_name: str, declaration: dict[str, Any]) -> dict[str, Any]:
    """Call one factory from the saved package and validate its returned Core step."""
    relative, name = custom_reference(declaration["custom"])
    module_name = f"{package_name}.{relative}"
    try:
        module = import_module(module_name)
        factory = getattr(module, name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot load custom recipe factory {declaration['custom']!r}.") from exc
    if not callable(factory) or getattr(factory, "__module__", None) != module_name:
        raise ValueError("custom factory must be defined in its captured project module.")
    step = deepcopy(factory(**deepcopy(declaration.get("params", {}))))
    if type(step) is not dict:
        raise ValueError("custom factory must return one Core step dictionary.")
    if "name" in declaration:
        step["name"] = declaration["name"]
    validate_preprocessing_steps([step])
    return step


def build_recipe(
    package_name: str, recipes: dict[str, list[dict[str, Any]]], recipe: str = "default"
) -> list[dict[str, Any]]:
    """Return independent Core steps in declaration order from the saved recipe version."""
    if recipe == "none":
        return []
    if recipe not in recipes:
        raise ValueError(f"Unknown feature recipe: {recipe!r}. Choose from {list(recipes)}.")
    return [
        _custom_step(package_name, step) if "custom" in step else deepcopy(step)
        for step in recipes[recipe]
    ]
