"""Bind explicit recipe choices into the executable source saved with a model."""

from inspect import signature
from types import ModuleType
from typing import Any

from ....config_validation import validate_preprocessing_steps


def recipe_label(module: ModuleType, factory_name: str, selected: str | None) -> str:
    """Name an explicit recipe or a builder's declared default without invoking it again."""
    if selected is not None:
        return selected
    factory = getattr(module, factory_name, None)
    if factory is None:
        return "none"
    try:
        parameter = signature(factory).parameters.get("recipe")
    except (TypeError, ValueError):
        return "builder_default"
    if parameter is not None and isinstance(parameter.default, str):
        return parameter.default
    return "builder_default"


def bind_recipe_source(source: str, selections: dict[str, str | None]) -> str:
    """Include selections in source identity and preserve zero-argument saved replay."""
    factories = {
        "preprocessing_recipe": "build_preprocessing",
        "pre_split_recipe": "build_pre_split_steps",
    }
    bindings = []
    for option, recipe in selections.items():
        if recipe is None:
            continue
        if not isinstance(recipe, str) or not recipe.strip():
            raise ValueError(f"{option} must be a nonempty recipe name or None.")
        factory = factories[option]
        error = f"{option}={recipe!r} requires callable {factory}(recipe=...)."
        bindings.append(
            f"if not callable(globals().get({factory!r})):\n"
            f"    raise ValueError({error!r})\n"
            f"{factory} = _skyulf_recipe_partial({factory}, recipe={recipe!r})\n"
        )
    if not bindings:
        return source
    return (
        source + "\nfrom functools import partial as _skyulf_recipe_partial\n" + "".join(bindings)
    )


def recipe_steps(
    module: ModuleType, name: str, option: str, recipe: str | None, *, optional: bool = False
) -> list[dict[str, Any]]:
    """Resolve one builder without falling back when a selected recipe is invalid."""
    factory = getattr(module, name, None)
    if factory is None and optional:
        return []
    if not callable(factory):
        messages = {
            "build_preprocessing": "preprocessing.py must define build_preprocessing().",
            "build_pre_split_steps": (
                "build_pre_split_steps must be a function returning Core steps."
            ),
        }
        raise ValueError(messages[name])
    try:
        steps = factory()
    except (TypeError, ValueError, KeyError) as exc:
        if recipe is None:
            raise
        raise ValueError(f"Invalid {option}={recipe!r} for {name}(): {exc}") from exc
    if not isinstance(steps, list):
        raise ValueError(f"{name}() must return a list of Core steps.")
    validate_preprocessing_steps(steps)
    return steps
