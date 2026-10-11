"""Capture separate YAML phase recipes alongside their trusted project functions."""

import ast
import re
from pathlib import Path
from typing import Any

from ....config_validation import validate_preprocessing_steps
from ....inference.project_code import project_source_digest
from ....inference.recipe_config import custom_reference
from ._project_files import contained_project_file, project_source, read_source
from .yaml_config import read_yaml_mapping

_FACTORIES = {"preprocessing": "build_preprocessing", "pre_split": "build_pre_split_steps"}


def _known_fields(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    """Reject misspelled declarations before any editable Python gets executed."""
    if type(value) is not dict:
        raise ValueError(f"{label} must be a mapping.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} settings: {sorted(map(str, unknown))}.")
    return value


def _custom_source(root: Path, reference: Any) -> None:
    """Require the referenced module to be a captured project file, not an external import."""
    module, _ = custom_reference(reference)
    relative = module.replace(".", "/")
    if relative.split("/")[0] == "groups":
        raise ValueError("custom recipes cannot import upstream feature groups.")
    filename = relative + ".py"
    if (root / relative).is_dir():
        filename = relative + "/__init__.py"
    try:
        contained_project_file(root, filename)
    except ValueError as exc:
        raise ValueError(f"custom module {module!r} must be inside the feature package.") from exc


def _step(value: Any, root: Path, label: str) -> dict[str, Any]:
    """Validate either an existing Core step or an explicit project factory call."""
    custom = isinstance(value, dict) and "custom" in value
    fields = (
        {"name", "custom", "params"} if custom else {"name", "transformer", "params", "pre_split"}
    )
    step = _known_fields(value, fields, f"{label} {'custom' if custom else 'step'}")
    if type(step.get("params", {})) is not dict:
        raise ValueError(f"{label} params must be a mapping.")
    if "name" in step and (type(step["name"]) is not str or not step["name"].strip()):
        raise ValueError(f"{label} name must be a nonempty string.")
    if custom:
        _custom_source(root, step["custom"])
    else:
        validate_preprocessing_steps([step])
    return step


def _recipes(document: dict[str, Any], root: Path, phase: str) -> dict[str, list[dict[str, Any]]]:
    """Require a default recipe and bounded safe YAML lists with one declaration owner."""
    _known_fields(document, {"version", "recipes"}, f"{phase}.yml")
    if type(document.get("version")) is not int or document["version"] != 1:
        raise ValueError(f"{phase}.yml version must be 1.")
    entries = document.get("recipes")
    if type(entries) is not dict or "default" not in entries:
        raise ValueError(f"{phase}.yml recipes must be a mapping containing default.")
    result = {}
    for name, steps in entries.items():
        result[name] = _recipe(name, steps, root, phase)
    return result


def _recipe(name: str, steps: Any, root: Path, phase: str) -> list[dict[str, Any]]:
    """Validate one ordered recipe without executing custom factories."""
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name) is None:
        raise ValueError(f"{phase}.yml recipe names need a letter and safe ASCII characters.")
    if type(steps) is not list:
        raise ValueError(f"{phase}.yml recipe {name} must be a list.")
    if name == "none" and steps:
        raise ValueError(f"{phase}.yml recipe none must be empty.")
    return [_step(step, root, f"{phase}.yml recipes.{name}[{i}]") for i, step in enumerate(steps)]


def _binds_name(statement: ast.stmt, name: str) -> bool:
    """Recognize direct builder definitions, assignments and imports without execution."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return statement.name == name
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return any((alias.asname or alias.name) == name for alias in statement.names)
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        return any(isinstance(target, ast.Name) and target.id == name for target in targets)
    return False


def _python_owner(root: Path, phase: str) -> None:
    """Allow custom functions but reject a second Python recipe builder for a YAML phase."""
    for filename in ("__init__.py", f"{phase}.py"):
        if not (root / filename).exists():
            continue
        tree = ast.parse(read_source(contained_project_file(root, filename)), filename=filename)
        if any(_binds_name(statement, _FACTORIES[phase]) for statement in tree.body):
            raise ValueError(f"Recipes defined in both {phase}.yml and {filename}.")


def _declarations(root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Read each optional phase config from the package's project, with containment checks."""
    directory = root.parent.parent / "config"
    declarations = {}
    for phase in _FACTORIES:
        path = directory / f"{phase}.yml"
        if not path.exists() and not path.is_symlink():
            continue
        if not directory.resolve().is_relative_to(root.parent.parent.resolve()):
            raise ValueError("Recipe configuration must remain inside the project.")
        document = read_yaml_mapping(contained_project_file(directory, path.name))
        declarations[phase] = _recipes(document, root, phase)
        _python_owner(root, phase)
    return declarations


def feature_project_source(path: Path) -> str:
    """Snapshot recipes and functions for training, static checks and disk-free replay.

    YAML is parsed only here. The saved source contains literal declarations and
    exports the same builders used by existing training-plan restore consumers.
    Python-only projects and previously saved model packages keep their behavior.
    """
    declarations = _declarations(path) if path.is_dir() else {}
    source = project_source(path, exclude_feature_groups=True)
    if not declarations:
        return source
    source += (
        "\nfrom functools import partial as _skyulf_yaml_partial\n"
        "from skyulf.inference.recipe_config import build_recipe as _skyulf_yaml_recipe\n"
    )
    for phase, recipes in declarations.items():
        factory = _FACTORIES[phase]
        error = f"Recipes defined in both {phase}.yml and Python {factory}."
        source += (
            f"if {factory!r} in globals():\n    raise ValueError({error!r})\n"
            f"{factory} = _skyulf_yaml_partial(_skyulf_yaml_recipe, __name__, {recipes!r})\n"
        )
    project_source_digest(source)
    return source
