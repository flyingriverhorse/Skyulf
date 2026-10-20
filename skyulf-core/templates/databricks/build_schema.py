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


def _source_properties(source: Path) -> dict[str, Any]:
    """Merge topic files and derived questions; rules stay as written except hidden settings."""
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
    return _hide_hyperparameters(_weight_questions(properties))


# Ensemble settings that tune a chosen model rather than choose one. Base count, base and
# final models stay asked; `_weighted` names are sample-weight menus, not voting weights.
_HYPERPARAMETER_PROMPTS = re.compile(
    r".*ensemble_(?:[1-8]_)?(?:(?:regression|classification)_weight_[0-9]+"
    r"|voting|cv|passthrough|calibrate|calibration_cv)"
)


def _hide_hyperparameters(properties: dict[str, Any]) -> dict[str, Any]:
    """Keep hyperparameters as answer-file inputs whose defaults reach config/training.yml."""
    for name, field in properties.items():
        if _HYPERPARAMETER_PROMPTS.fullmatch(name):
            field["skip_prompt_if"] = {}  # Matches every answer set: never asked, still settable.
    return properties


def render_schema(source: Path) -> str:
    """Combine metadata and prompt definitions without silently replacing fields."""
    metadata = _read_object(source / "metadata.json")
    if "properties" in metadata:
        raise ValueError("metadata.json must not define properties; use topic files.")
    properties = _default_first(_quick_setup(_cli_rules(_source_properties(source))))
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
    label = f"Branch {prefix.split('_')[1]} | " if prefix else ""
    return {
        flag: {
            "order": order,
            "type": "string",
            "default": "false",
            "enum": ["false", "true"],
            "description": label + "Use sample weights? false = No (default); true = Yes.",
            "skip_prompt_if": skip,
        },
        prefix + "weight_column": {
            "order": order + 1,
            "type": "string",
            "default": "training_weight",
            "description": label
            + "Training-only sample-weight column (excluded from model features).",
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


_QUICK_PROMPTS = re.compile(
    r"training_layout|project_name|setup_mode|task|source_table_name|record_key_columns"
    r"|input_columns|target_column|branch_count|monitoring|monitoring_schema"
    r"|label_table_name|label_available_at_column"
    r"|source_layout|score_source_table_name|include_source_sql|initial_scoring"
    r"|branch_[1-8]_(?:task|target_column|input_columns)"
)


def _quick_setup(properties: dict[str, Any]) -> dict[str, Any]:
    """Let quick setup accept each optional default while keeping facts that have none."""
    if "setup_mode" not in properties:
        return properties
    # Required: a missing answer must ask the question, not silently accept its default.
    quick = {"properties": {"setup_mode": {"const": "quick"}}, "required": ["setup_mode"]}
    for name, field in properties.items():
        skip = field.get("skip_prompt_if")
        if _QUICK_PROMPTS.fullmatch(name) or skip == {}:
            continue  # An empty condition already hides the prompt in every mode.
        options = [] if skip is None else skip["anyOf"] if set(skip) == {"anyOf"} else [skip]
        field["skip_prompt_if"] = {"anyOf": [*options, quick]} if options else quick
    return properties


# The CLI evaluates only properties (const, enum, pattern), required and anyOf in a skip rule;
# it drops not and allOf without an error. Each rule is therefore rewritten as OR of ANDs
# over the allowed choices of earlier questions. An alternative maps a question to
# (allowed choices, answer must exist); [] never matches and [{}] always matches.
_ALWAYS: list[dict[str, Any]] = [{}]


def _allowed(constraint: dict[str, Any], choices: list[str]) -> frozenset[str]:
    """List the choices one const/enum/pattern constraint accepts."""
    unknown = set(constraint) - {"const", "enum", "pattern"}
    if unknown:
        raise ValueError(f"Unsupported skip rule constraint: {sorted(unknown)}")
    return frozenset(
        choice
        for choice in choices
        if ("const" not in constraint or choice == constraint["const"])
        and ("enum" not in constraint or choice in constraint["enum"])
        and ("pattern" not in constraint or re.search(constraint["pattern"], choice))
    )


def _both(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require two alternative lists together; contradictory combinations disappear."""
    result = []
    for one in left:
        for two in right:
            merged = dict(one)
            for name, (allowed, must) in two.items():
                if name in merged:
                    allowed, must = merged[name][0] & allowed, merged[name][1] or must
                merged[name] = (allowed, must)
            if all(allowed for allowed, _ in merged.values()):
                result.append(merged)
    return result


def _rule_parts(
    rule: dict[str, Any], domains: dict[str, list[str]], negate: bool
) -> list[list[dict[str, Any]]]:
    """Translate each keyword of one rule object, pushing a pending NOT into it."""
    parts: list[list[dict[str, Any]]] = []
    for name, constraint in rule.get("properties", {}).items():
        allowed = _allowed(constraint, domains[name])
        # A missing answer satisfies a constraint, so its negation needs the answer.
        choice = (frozenset(domains[name]) - allowed, True) if negate else (allowed, False)
        parts.append([{name: choice}])
    for key in ("anyOf", "allOf"):
        if key not in rule:
            continue
        options = [_alternatives(item, domains, negate) for item in rule[key]]
        if (key == "anyOf") != negate:
            parts.append([item for option in options for item in option])
            continue
        combined = _ALWAYS
        for option in options:
            combined = _both(combined, option)
        parts.append(combined)
    if "not" in rule:
        parts.append(_alternatives(rule["not"], domains, not negate))
    return parts


def _alternatives(
    rule: dict[str, Any], domains: dict[str, list[str]], negate: bool = False
) -> list[dict[str, Any]]:
    """Rewrite one rule as alternatives; De Morgan turns NOT(a AND b) into NOT a OR NOT b."""
    unknown = set(rule) - {"properties", "anyOf", "allOf", "not"}
    if unknown:
        raise ValueError(f"Unsupported skip rule keyword: {sorted(unknown)}")
    parts = _rule_parts(rule, domains, negate)
    if negate:
        return [item for part in parts for item in part]
    combined = _ALWAYS
    for part in parts:
        combined = _both(combined, part)
    return combined


def _merge_once(options: list[dict[str, Any]]) -> bool:
    """Join two alternatives that differ only in one question's allowed choices."""
    for first, one in enumerate(options):
        for second in range(first + 1, len(options)):
            two = options[second]
            different = [name for name in one if one[name] != two.get(name)]
            if one.keys() != two.keys() or len(different) != 1:
                continue
            name = different[0]
            if one[name][1] == two[name][1]:
                options[first] = one | {name: (one[name][0] | two[name][0], one[name][1])}
                del options[second]
                return True
    return False


def _implied(one: dict[str, Any], other: dict[str, Any]) -> bool:
    """Tell whether every answer set matching one alternative also matches the other."""
    return all(
        name in one and one[name][0] <= allowed and (one[name][1] or not must)
        for name, (allowed, must) in other.items()
    )


def _tidy(options: list[dict[str, Any]], domains: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Drop constraints that allow every choice, then merged and implied alternatives."""
    cleaned: list[dict[str, Any]] = []
    for option in options:
        kept = {
            name: choice for name, choice in option.items() if choice[0] != frozenset(domains[name])
        }
        if not kept:
            return _ALWAYS
        if kept not in cleaned:
            cleaned.append(kept)
    if _merge_once(cleaned):
        return _tidy(cleaned, domains)
    return [
        one
        for one in cleaned
        if not any(other != one and _implied(one, other) for other in cleaned)
    ]


def _without_prefixes(prefixes: set[str]) -> str:
    """Match text starting with none of these equal-length prefixes, without lookahead."""
    heads = sorted({prefix[0] for prefix in prefixes})
    options = ["[^" + "".join(heads) + "]"]
    for head in heads:
        rest = {prefix[1:] for prefix in prefixes if prefix[0] == head and len(prefix) > 1}
        if rest:
            options.append(head + "(?:" + _without_prefixes(rest) + ")")
    return "|".join(options)


def _choice_rule(allowed: frozenset[str], choices: list[str]) -> dict[str, Any]:
    """Write a set of choices compactly; a long all-but-these list becomes a checked pattern."""
    listed = [choice for choice in choices if choice in allowed]
    if len(listed) == 1:
        return {"const": listed[0]}
    excluded = [choice for choice in choices if choice not in allowed]
    for length in range(1, 1 + min(map(len, excluded))):
        prefixes = {choice[:length] for choice in excluded}
        if any(choice[:length] in prefixes for choice in listed):
            continue
        pattern = "^(?:" + _without_prefixes(prefixes) + ")"
        exact = all(bool(re.search(pattern, choice)) == (choice in allowed) for choice in choices)
        if exact and len(pattern) + 60 < len(json.dumps(listed)):
            return {"pattern": pattern}
        break
    return {"enum": listed}


def _write_rule(options: list[dict[str, Any]], domains: dict[str, list[str]]) -> dict[str, Any]:
    """Emit alternatives with the keywords the CLI evaluates and nothing else."""
    written = [
        {
            "properties": {
                name: _choice_rule(allowed, domains[name]) for name, (allowed, _) in one.items()
            }
        }
        for one in options
    ]
    return written[0] if len(written) == 1 else {"anyOf": written}


def _earlier_choices(name: str, rule: Any, properties: dict[str, Any]) -> None:
    """Require rules to read only list answers given before their own question."""
    if isinstance(rule, list):
        for item in rule:
            _earlier_choices(name, item, properties)
    if not isinstance(rule, dict):
        return
    for key, value in rule.items():
        for other in value if key == "properties" else ():
            field = properties.get(other, {})
            choices = field.get("enum")
            if not choices or not all(type(choice) is str for choice in choices):
                raise ValueError(f"{name}: skip rule needs a list question, not {other}.")
            if field["order"] >= properties[name]["order"]:
                raise ValueError(f"{name}: skip rule reads the later question {other}.")
        _earlier_choices(name, value, properties)


def _cli_rules(properties: dict[str, Any]) -> dict[str, Any]:
    """Rewrite every skip rule into the subset the Databricks CLI evaluates."""
    domains = {name: field["enum"] for name, field in properties.items() if "enum" in field}
    for name, field in properties.items():
        rule = field.get("skip_prompt_if")
        if not rule:
            continue  # Absent always asks and {} never asks; the CLI reads both correctly.
        _earlier_choices(name, rule, properties)
        options = _tidy(_alternatives(rule, domains), domains)
        if not options:
            raise ValueError(f"{name}: skip rule can never match; remove it.")
        field["skip_prompt_if"] = {} if options == _ALWAYS else _write_rule(options, domains)
    return properties


def _default_first(properties: dict[str, Any]) -> dict[str, Any]:
    """Offer each default first: the CLI highlights the first choice and has no list default."""
    for field in properties.values():
        choices = field.get("enum")
        if choices and field.get("default") in choices:
            default = field["default"]
            field["enum"] = [default, *(choice for choice in choices if choice != default)]
    return properties


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
    result = {}
    for slot in range(1, 9):
        group = json.loads(json.dumps(fields).replace("SLOT", str(slot)))
        for name, field in group.items():
            field["order"] += 73 + (slot - 1) * len(fields)
            result[name] = field
    return result


def _ensemble_base_questions(fields: dict[str, Any]) -> dict[str, Any]:
    """Expand each task's first model and weight prompts without duplicate definitions."""
    fields = deepcopy(fields)
    for task in ("regression", "classification"):
        prefix = f"competition_ensemble_SLOT_{task}_"
        models = fields[prefix + "base_1"]["enum"]
        for kind in ("base", "weight"):
            prototype = fields.pop(prefix + kind + "_1")
            for position, model in enumerate(models, 1):
                name = f"{prefix}{kind}_{position}"
                if name in fields:
                    raise ValueError(f"Duplicate ensemble question: {name}")
                field = deepcopy(prototype)
                field["order"] += 2 * (position - 1)
                field["description"] = field["description"].replace("model 1", f"model {position}")
                if kind == "base":
                    field["default"] = model
                properties = field["skip_prompt_if"]["not"]["anyOf"][0]["properties"]
                properties[prefix + "base_count"]["enum"] = [
                    str(count) for count in range(max(2, position), len(models) + 1)
                ]
                fields[name] = field
    return fields


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
    if any(not name.startswith("competition_ensemble_SLOT_") for name in fields):
        raise ValueError("Ensemble question names must start with competition_ensemble_SLOT_.")
    fields = _ensemble_base_questions(fields)
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
