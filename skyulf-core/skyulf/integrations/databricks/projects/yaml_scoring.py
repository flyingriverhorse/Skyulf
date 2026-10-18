"""Compile optional scoring YAML into the existing saved project hook contracts."""

import ast
import inspect
import math
from pathlib import Path
from typing import Any, cast

from ....inference import scoring_rules
from ....inference._model_set_operations import _inputs, _weights
from ....inference.project_scoring import _identifier, _validate_columns
from ._project_files import contained_project_file, read_source
from .yaml_config import read_yaml_mapping
from .yaml_recipes import _binds_name, _custom_source, _known_fields

_BUILDERS = ("build_model_rules", "build_scoring", "build_combined_rules")
_BUILTINS = {
    "required_values": scoring_rules._skyulf_yaml_required_values,
    "value_range": scoring_rules._skyulf_yaml_value_range,
    "prediction_band": scoring_rules._skyulf_yaml_prediction_band,
}


def _text(value: Any, label: str) -> str:
    """Require an explicit nonempty input name without silently coercing YAML scalars."""
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value


def _required_values(params: dict[str, Any]) -> None:
    """Validate required observed columns without executing the captured callback."""
    _known_fields(params, {"columns"}, "required_values params")
    columns = params.get("columns")
    if type(columns) is not list or not columns:
        raise ValueError("required_values columns must be a nonempty list.")
    for column in columns:
        _text(column, "required_values column")
    if len(columns) != len(set(columns)):
        raise ValueError("required_values columns must be unique.")


def _finite(value: Any) -> bool:
    """Accept finite native numeric scalars without booleans or coercible strings."""
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _value_range(params: dict[str, Any]) -> None:
    """Require at least one finite, inclusive and ordered numeric bound."""
    _known_fields(params, {"column", "minimum", "maximum"}, "value_range params")
    _text(params.get("column"), "value_range column")
    lower, upper = params.get("minimum"), params.get("maximum")
    bounds = [value for value in (lower, upper) if value is not None]
    if not bounds or not all(_finite(value) for value in bounds):
        raise ValueError("value_range requires finite limits and at least one bound.")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("value_range requires ordered limits.")


def _prediction_band(params: dict[str, Any], columns: list[dict[str, str]]) -> None:
    """Reject ambiguous band cut points, labels or declared output schemas early."""
    _known_fields(params, {"column", "thresholds", "labels", "output"}, "prediction_band params")
    _text(params.get("column", "prediction"), "prediction_band column")
    thresholds = params.get("thresholds")
    if type(thresholds) is not list or not all(_finite(value) for value in thresholds):
        raise ValueError("prediction_band thresholds must be a finite numeric list.")
    thresholds = cast(list[float], thresholds)
    if any(left >= right for left, right in zip(thresholds, thresholds[1:], strict=False)):
        raise ValueError("prediction_band thresholds must be strictly increasing.")
    labels = params.get("labels")
    if type(labels) is not list or len(labels) != len(thresholds) + 1:
        raise ValueError("prediction_band requires one more label than thresholds.")
    for label in labels:
        _text(label, "prediction_band label")
    if columns != [{"name": params.get("output", "band"), "dtype": "string"}]:
        raise ValueError("prediction_band columns must declare its single string output.")


def _builtin(rule: dict[str, Any], kind: str) -> str:
    """Validate one known builtin and select its self-contained saved callback."""
    selector = "check" if kind == "eligibility" else "rule"
    name = rule.get(selector)
    allowed = {"required_values", "value_range"} if kind == "eligibility" else {"prediction_band"}
    if type(name) is not str or name not in allowed:
        raise ValueError(f"Unknown scoring {selector}: {name!r}.")
    if rule["version"] != "1":
        raise ValueError("Builtin scoring rules require version '1'.")
    if name == "required_values":
        _required_values(rule["params"])
    elif name == "value_range":
        _value_range(rule["params"])
    else:
        _prediction_band(rule["params"], rule["columns"])
    return _BUILTINS[name].__name__


def _dependencies(value: Any) -> None:
    """Require direct component names before binding them to the fitted model set."""
    if type(value) is not list or not value or not all(_identifier(name) for name in value):
        raise ValueError("Combined required_components must be a nonempty list of names.")
    if len(value) != len(set(value)):
        raise ValueError("Combined required_components must be unique.")


def _operation(rule: dict[str, Any]) -> None:
    """Reuse arithmetic parameter checks; real component binding occurs at set assembly."""
    if rule["operation"] not in ("weighted_sum", "weighted_mean") or rule["version"] != "1":
        raise ValueError("Combined operation must be weighted_sum or weighted_mean, version '1'.")
    if len(rule["columns"]) != 1 or rule["columns"][0]["dtype"] != "float64":
        raise ValueError("Combined operation requires one float64 output column.")
    _inputs(rule)
    _weights(rule)


def _rule_fields(value: Any, kind: str) -> dict[str, Any]:
    """Normalize defaults while retaining the existing scoring rule schema."""
    selector = {"eligibility": "check", "outputs": "rule", "combined": "operation"}[kind]
    fields = {"name", "version", "params", selector, "custom"}
    if kind != "eligibility":
        fields.add("columns")
    if kind == "combined":
        fields.add("required_components")
    raw = _known_fields(value, fields, f"scoring {kind} rule")
    if ("custom" in raw) == (selector in raw):
        raise ValueError(f"Scoring {kind} rule needs exactly one of custom or {selector}.")
    rule = {"version": "1", "params": {}, **raw}
    if not _identifier(rule.get("name")):
        raise ValueError("Scoring rule name must be a simple identifier.")
    _text(rule["version"], "Scoring rule version")
    if type(rule["params"]) is not dict:
        raise ValueError("Scoring rule params must be a mapping.")
    if kind != "eligibility":
        _columns(rule.get("columns"))
    return rule


def _columns(columns: Any) -> None:
    """Validate scalar YAML dtype values before the existing schema membership checks."""
    if isinstance(columns, list):
        for column in columns:
            if isinstance(column, dict) and not isinstance(column.get("dtype"), str):
                raise ValueError("Scoring output column dtype must be a string.")
    _validate_columns(columns)


def _rule(value: Any, root: Path, kind: str) -> dict[str, Any]:
    """Translate a declarative selector to the established function or operation field."""
    rule = _rule_fields(value, kind)
    if kind == "combined":
        _dependencies(rule.get("required_components"))
    if "custom" in rule:
        _custom_source(root, rule["custom"])
        rule["function"] = rule.pop("custom")
    elif kind == "combined":
        _operation(rule)
    else:
        rule["function"] = _builtin(rule, kind)
        rule.pop("check" if kind == "eligibility" else "rule")
    return rule


def _unique_rules(rules: list[dict[str, Any]]) -> None:
    """Reject duplicate rule identities and output names before any project execution."""
    names = [rule["name"].casefold() for rule in rules]
    columns = [col["name"].casefold() for rule in rules for col in rule.get("columns", [])]
    if len(names) != len(set(names)) or len(columns) != len(set(columns)):
        raise ValueError("Scoring rule names and output columns must be unique.")


def _envelope(document: dict[str, Any]) -> tuple[str, bool]:
    """Reject invalid modes and flags without executing user source."""
    _known_fields(
        document,
        {"version", "mode", "skip_target_pre_split_steps", "eligibility", "outputs", "combined"},
        "scoring.yml",
    )
    if type(document.get("version")) is not int or document["version"] != 1:
        raise ValueError("scoring.yml version must be 1.")
    mode = document.get("mode", "pre_split")
    if mode not in ("pre_split", "custom", "combined"):
        raise ValueError("scoring.yml mode must be pre_split, custom or combined.")
    skip = document.get("skip_target_pre_split_steps", False)
    if type(skip) is not bool:
        raise ValueError("skip_target_pre_split_steps must be boolean.")
    if mode == "custom" and skip:
        raise ValueError("skip_target_pre_split_steps does not apply in custom mode.")
    return mode, skip


def _document(document: dict[str, Any], root: Path) -> dict[str, Any]:
    """Validate each ordered list without importing project code."""
    mode, skip = _envelope(document)
    result: dict[str, Any] = {"mode": mode, "skip_target_pre_split_steps": skip}
    for kind in ("eligibility", "outputs", "combined"):
        values = document.get(kind, [])
        if type(values) is not list:
            raise ValueError(f"scoring.yml {kind} must be a list.")
        result[kind] = [_rule(value, root, kind) for value in values]
    if mode == "pre_split" and (result["eligibility"] or result["outputs"]):
        raise ValueError("pre_split mode cannot contain custom eligibility or outputs.")
    _unique_rules(result["eligibility"] + result["outputs"])
    _unique_rules(result["combined"])
    return result


def _python_owners(root: Path) -> None:
    """Reject competing builders anywhere in the captured feature package before imports."""
    for filename in root.rglob("*.py"):
        relative = filename.relative_to(root).as_posix()
        if relative.split("/")[0] == "groups":
            continue
        tree = ast.parse(read_source(contained_project_file(root, relative)), filename=relative)
        if _competing_binding(tree):
            raise ValueError(f"Rules defined in both scoring.yml and Python {relative}.")


def _competing_binding(tree: ast.AST) -> bool:
    """Include destructured assignments, conditional exports and reserved callback symbols."""
    names = {*_BUILDERS, *(function.__name__ for function in _BUILTINS.values())}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in names:
            return True
        if isinstance(node, ast.stmt) and any(_binds_name(node, name) for name in names):
            return True
    return False


def _callback_source(custom: dict[str, Any]) -> str:
    """Snapshot selected builtins and reject exported symbol collisions."""
    callbacks = {rule.get("function") for rules in custom.values() for rule in rules}
    selected = [function for function in _BUILTINS.values() if function.__name__ in callbacks]
    names = (*_BUILDERS, *(function.__name__ for function in selected))
    result = "\n"
    for name in names:
        result += f"if {name!r} in globals():\n    raise ValueError('Rules defined in both scoring.yml and Python: {name}.')\n"
    result += "\n".join(inspect.getsource(function) for function in selected)
    return result


def _bindings(document: dict[str, Any]) -> str:
    """Save fresh literal builders and exact builtin definitions for disk-free replay."""
    custom = {kind: document[kind] for kind in ("eligibility", "outputs")}
    reuse = {"reuse_pre_split": True, "skip_target_steps": document["skip_target_pre_split_steps"]}
    mode = document["mode"]
    model = custom if mode == "custom" else reuse
    if mode == "combined":
        model = {**reuse, **custom}
    result = _callback_source(custom)
    result += f'\ndef build_model_rules():\n    """Return this saved version of the model policy."""\n    return {model!r}\n'
    result += f'\ndef build_combined_rules():\n    """Return this saved version of combined model rules."""\n    return {document["combined"]!r}\n'
    return result


def scoring_source(root: Path) -> str:
    """Read optional scoring YAML only during project capture and freeze its policy."""
    if not root.is_dir():
        _require_package_for_yaml(root)
        return ""
    directory = root.parent.parent / "config"
    path = directory / "scoring.yml"
    if not path.exists() and not path.is_symlink():
        return ""
    if (
        directory.is_symlink()
        or directory.is_junction()
        or not directory.resolve().is_relative_to(root.parent.parent.resolve())
    ):
        raise ValueError("Scoring configuration must remain inside the project.")
    document = _document(read_yaml_mapping(contained_project_file(directory, path.name)), root)
    _python_owners(root)
    return _bindings(document)


def _require_package_for_yaml(path: Path) -> None:
    """Reject YAML beside a legacy file without consulting unrelated ancestor projects."""
    project = path.parent.parent if path.parent.name == "src" else path.parent
    declaration = project / "config/scoring.yml"
    if declaration.exists() or declaration.is_symlink():
        raise ValueError("scoring.yml requires a feature package directory, not a Python file.")
