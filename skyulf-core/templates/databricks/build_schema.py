"""Generate the Databricks CLI wizard schema from topic files in schema/."""

import argparse
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys, including nested prompt conditions."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_object(path: Path) -> dict[str, Any]:
    """Read a source object with actionable file context on invalid JSON."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc


def render_schema(source: Path) -> str:
    """Combine metadata and prompt definitions without silently replacing fields."""
    metadata = _read_object(source / "metadata.json")
    if "properties" in metadata:
        raise ValueError("metadata.json must not define properties; use topic files.")
    properties: dict[str, Any] = {}
    for path in sorted(source.glob("*.json")):
        if path.name == "metadata.json":
            continue
        fields = _read_object(path)
        fields = _expand_topic(path.name, fields)
        duplicates = properties.keys() & fields.keys()
        if duplicates:
            raise ValueError(
                f"Duplicate properties in {path.name}: {', '.join(sorted(duplicates))}"
            )
        properties |= fields
    if not properties:
        raise ValueError("No prompt properties found in schema topic files.")
    properties = _weight_questions(properties)
    _validate_prompt_orders(properties)
    metadata["properties"] = dict(sorted(properties.items(), key=lambda item: item[1]["order"]))
    return _format_schema(metadata)


def _validate_prompt_orders(properties: dict[str, Any]) -> None:
    """Reject ambiguous CLI sequencing after all derived questions are inserted."""
    owners: dict[int, str] = {}
    for name, field in properties.items():
        order = field["order"]
        if order in owners:
            raise ValueError(f"Duplicate prompt order {order}: {owners[order]}, {name}")
        owners[order] = name


def _weight_prefix(name: str) -> str:
    """Keep independent branch weight decisions separate from shared competition weights."""
    match = re.match(r"branch_[1-8]_", name)
    return match.group() if match else ""


def _model_choice(name: str) -> bool:
    """Recognize existing model selections without maintaining a second model catalog."""
    return bool(
        re.search(r"(?:regression|classification)_model(?:_[1-8])?$", name)
        or re.search(
            r"ensemble_(?:[1-8]_)?(?:regression|classification)_(?:base_[0-9]+|final)$", name
        )
    )


def _supported_choices(name: str, choices: list[str]) -> list[str]:
    """Derive menus using the same configured-model policy as fitting."""
    from skyulf.modeling.capabilities import (  # noqa: PLC0415 - standalone schema-only copies
        model_supports_sample_weight,
    )
    from skyulf.registry import NodeRegistry  # noqa: PLC0415 - optional build dependencies

    if "ensemble_" not in name:
        for choice in choices:
            try:
                NodeRegistry.get_calculator(choice)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Install all template model dependencies before generation: {choice}"
                ) from exc
        return [choice for choice in choices if model_supports_sample_weight(choice)]
    task = "classifier" if "classification" in name else "regressor"
    calculator = NodeRegistry.get_calculator("voting_" + task)
    missing = set(choices) - set(calculator.BASE_ESTIMATORS)
    if missing:
        raise ValueError(
            f"Install all template model dependencies before generation: {sorted(missing)}"
        )
    return [
        choice
        for choice in choices
        if model_supports_sample_weight("voting_" + task, {"base_estimators": [choice]})
    ]


def _weight_condition(value: Any, choices: set[str]) -> Any:
    """Make dependent ensemble prompts inspect the active weighted or ordinary menu."""
    if isinstance(value, list):
        return [_weight_condition(item, choices) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _weight_condition(item, choices) for key, item in value.items()}
    properties = result.get("properties", {})
    selected = set(properties).intersection(choices)
    if not selected:
        return result
    remaining = {key: item for key, item in properties.items() if key not in selected}
    alternatives = [_active_model_condition(key, properties[key]) for key in sorted(selected)]
    return {
        **{key: item for key, item in result.items() if key != "properties"},
        "allOf": [{"properties": remaining}, *alternatives],
    }


def _active_model_condition(name: str, constraint: Any) -> dict[str, Any]:
    """Require the selected menu's value instead of an inactive default."""
    flag = _weight_prefix(name) + "sample_weight_enabled"
    return {
        "anyOf": [
            {"properties": {flag: {"const": "false"}, name: constraint}},
            {"properties": {flag: {"const": "true"}, name + "_weighted": constraint}},
        ]
    }


def _weight_controls(prefix: str, order: int, skip: dict[str, Any]) -> dict[str, Any]:
    """Ask opt-in and source column before the corresponding model selections."""
    flag = prefix + "sample_weight_enabled"
    return {
        flag: {
            "order": order,
            "type": "string",
            "default": "false",
            "enum": ["false", "true"],
            "description": "Use sample weights? false = No (default); true = Yes.",
            "skip_prompt_if": skip,
        },
        prefix + "weight_column": {
            "order": order + 1,
            "type": "string",
            "default": "training_weight",
            "description": "Training-only sample-weight column (excluded from model features).",
            "pattern": "^[A-Za-z_][A-Za-z0-9_]*$",
            "skip_prompt_if": {"anyOf": [skip, {"properties": {flag: {"const": "false"}}}]},
        },
    }


def _weight_questions(properties: dict[str, Any]) -> dict[str, Any]:
    """Generate filtered parallel menus; CLI initialization remains Python-free."""
    choices = {name for name in properties if _model_choice(name)}
    if not choices:
        return properties
    result = {}
    for name, original in properties.items():
        field = deepcopy(original)
        scaled_order = field["order"] * 10
        if not float(scaled_order).is_integer():
            raise ValueError(f"{name}: prompt order must have at most one decimal place.")
        field["order"] = int(scaled_order)
        if "skip_prompt_if" in field:
            field["skip_prompt_if"] = _weight_condition(field["skip_prompt_if"], choices)
        result[name] = field
        if name in choices:
            _add_weighted_choice(result, name, field)
    result.update(
        _weight_controls("", 537, {"properties": {"training_layout": {"const": "multi_target"}}})
    )
    for slot in range(1, 9):
        prefix = f"branch_{slot}_"
        order = result[prefix + "task"]["order"] + 1
        skip = result[prefix + "task"]["skip_prompt_if"]
        result.update(_weight_controls(prefix, order, skip))
    return result


def _add_weighted_choice(result: dict[str, Any], name: str, field: dict[str, Any]) -> None:
    """Retain legacy fields while showing exactly one supported selection prompt."""
    weighted = deepcopy(field)
    weighted["enum"] = _supported_choices(name, field["enum"])
    if weighted["default"] not in weighted["enum"]:
        weighted["default"] = weighted["enum"][0]
    weighted["order"] += 1
    flag = _weight_prefix(name) + "sample_weight_enabled"
    skip = field.get("skip_prompt_if", {})
    field["skip_prompt_if"] = {"anyOf": [skip, {"properties": {flag: {"const": "true"}}}]}
    weighted["skip_prompt_if"] = {"anyOf": [skip, {"properties": {flag: {"const": "false"}}}]}
    result[name + "_weighted"] = weighted


def _format_schema(schema: dict[str, Any]) -> str:
    """Keep each generated prompt on one line while retaining JSON member order."""
    encode = json.JSONEncoder(ensure_ascii=False, separators=(",", ":")).encode
    root_fields = [
        f"  {encode(name)}: {encode(value)}"
        for name, value in schema.items()
        if name != "properties"
    ]
    properties = ",\n".join(
        f"    {encode(name)}: {encode(field)}" for name, field in schema["properties"].items()
    )
    root_fields.append('  "properties": {\n' + properties + "\n  }")
    return "{\n" + ",\n".join(root_fields) + "\n}\n"


def _competition_ensemble_slots(fields: dict[str, Any]) -> dict[str, Any]:
    """Repeat the ensemble question group for the eight supported candidate slots."""
    if any(not name.startswith("competition_ensemble_SLOT_") for name in fields):
        raise ValueError("Ensemble question names must start with competition_ensemble_SLOT_.")
    result = {}
    for slot in range(1, 9):
        group = json.loads(json.dumps(fields).replace("SLOT", str(slot)))
        for name, field in group.items():
            field["order"] += 73 + (slot - 1) * len(fields)
            result[name] = field
    return result


def _expand_topic(name: str, fields: dict[str, Any]) -> dict[str, Any]:
    """Expand only the wizard's explicit model and branch question groups."""
    if name == "competition_ensemble.json":
        return _ensemble_questions(fields)
    if name == "branches.json":
        return _branch_slots(fields, offset=0)
    return fields


def _without_candidate_count(value: Any) -> Any:
    """Single-model questions need no competition slot admission condition."""
    if isinstance(value, list):
        return [_without_candidate_count(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _without_candidate_count(item)
            for key, item in value.items()
            if key != "competition_candidate_count"
        }
    return value


def _ensemble_questions(fields: dict[str, Any]) -> dict[str, Any]:
    """Reuse the same ensemble controls for single models, candidates and branches."""
    result = _competition_ensemble_slots(fields)
    payload = json.dumps(fields)
    single = payload.replace("competition_ensemble_SLOT_", "single_ensemble_")
    single = single.replace("Candidate SLOT ensemble", "Single model ensemble")
    single = single.replace('"model_competition"', '"single_model"')
    for task in ("regression", "classification"):
        single = single.replace(f"competition_{task}_model_SLOT", f"{task}_model")
    single_fields = _without_candidate_count(json.loads(single))
    for field in single_fields.values():
        field["order"] += 596
    result |= single_fields
    branch = payload.replace("competition_ensemble_SLOT_", "branch_SLOT_ensemble_")
    branch = branch.replace("Candidate SLOT ensemble", "Branch SLOT ensemble")
    branch = branch.replace('"model_competition"', '"multi_target"')
    branch = branch.replace("competition_candidate_count", "branch_count")
    branch = branch.replace('"task":', '"branch_SLOT_task":')
    for task in ("regression", "classification"):
        branch = branch.replace(f"competition_{task}_model_SLOT", f"branch_SLOT_{task}_model")
    return result | _branch_slots(json.loads(branch), offset=150)


def _branch_slots(fields: dict[str, Any], *, offset: int) -> dict[str, Any]:
    """Reserve one ordered settings group and ensemble group per branch."""
    if any(not name.startswith("branch_SLOT_") for name in fields):
        raise ValueError("Branch question names must start with branch_SLOT_.")
    if any(not 0 <= field["order"] < 150 for field in fields.values()):
        raise ValueError("Branch question local order must be below 150.")
    result = {}
    for slot in range(1, 9):
        group = json.loads(json.dumps(fields).replace("SLOT", str(slot)))
        for name, field in group.items():
            field["order"] += 1001 + (slot - 1) * 300 + offset
            result[name] = field
    return result


def _model_space_catalog(root: Path, *, check: bool) -> int:
    """Keep the committed Core-derived CLI helper fresh alongside wizard sources."""
    builder = root / "build_model_spaces.py"
    if not builder.is_file():
        return 0  # Standalone schema-only copies need no Core catalog.
    command = [sys.executable, str(builder)]
    if check:
        command.append("--check")
    return subprocess.run(command, check=False).returncode


def main() -> int:
    """Regenerate the committed schema or check it without writing anything."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Fail if the generated schema is stale"
    )
    options = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root / "databricks_template_schema.json"
    rendered = render_schema(root / "schema")
    if options.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            print(
                "Template schema is out of date; run build_schema.py without --check.",
                file=sys.stderr,
            )
            return 1
        print("Template schema is up to date.")
        return _model_space_catalog(root, check=True)
    output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Generated {output.name}")
    return _model_space_catalog(root, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
