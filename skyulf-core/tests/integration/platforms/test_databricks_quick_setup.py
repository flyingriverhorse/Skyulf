"""Quick setup asks only for facts that have no usable default."""

import json
import re
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[3] / "templates/databricks"
TEMPLATE = ROOT / "template/{{.project_name}}"
PROPERTIES = json.loads((ROOT / "databricks_template_schema.json").read_text(encoding="utf-8"))[
    "properties"
]
QUICK = {"properties": {"setup_mode": {"const": "quick"}}, "required": ["setup_mode"]}
SHARED = {
    "training_layout",
    "project_name",
    "setup_mode",
    "source_table_name",
    "record_key_columns",
}
ONE_TARGET = SHARED | {"task", "input_columns", "target_column"}


def _asked(**answers):
    """List the prompts an operator sees once every earlier answer or default is known."""
    values = {name: field["default"] for name, field in PROPERTIES.items()} | answers
    return [
        name
        for name, field in PROPERTIES.items()
        if not Draft7Validator(field.get("skip_prompt_if", False)).is_valid(values)
    ]


def _cli_valid(schema, values):
    """Apply only what Databricks CLI 1.17 evaluates: required, enum, pattern, const, anyOf."""
    if any(name not in values for name in schema.get("required", [])):
        return False
    for name, value in values.items():
        rule = schema.get("properties", {}).get(name, {})
        if "enum" in rule and value not in rule["enum"]:
            return False
        if "pattern" in rule and not re.search(rule["pattern"], value):
            return False
        if "const" in rule and value != rule["const"]:
            return False
    return "anyOf" not in schema or any(_cli_valid(item, values) for item in schema["anyOf"])


def _cli_asked(properties, **answers):
    """Replay the CLI: each prompt sees only the values assigned before it."""
    values, asked = {}, []
    for name, field in properties.items():
        rule = field.get("skip_prompt_if")
        if rule is not None and _cli_valid(rule, values):
            values[name] = field["default"]
            continue
        asked.append(name)
        values[name] = answers.get(name, field["default"])
    return asked


def _without_quick_rule(field):
    """Recover the rule a prompt had before quick setup existed."""
    rule = field.get("skip_prompt_if")
    if rule == QUICK:
        return {key: value for key, value in field.items() if key != "skip_prompt_if"}
    if isinstance(rule, dict) and rule.get("anyOf", [None])[-1] == QUICK:
        return field | {"skip_prompt_if": rule["anyOf"][0]}
    return field


@pytest.mark.parametrize("layout", ["single_model", "model_competition"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_quick_setup_asks_one_target_only_for_names_task_and_columns(layout, task):
    """A first project must not require forty answers that all have working defaults."""
    asked = _asked(setup_mode="quick", training_layout=layout, task=task)
    assert set(asked) == ONE_TARGET


@pytest.mark.parametrize("count", ["2", "3"])
def test_quick_setup_asks_each_configured_branch_for_its_own_task_and_columns(count):
    """Independent targets have no shared default, so every active branch still asks."""
    asked = _asked(setup_mode="quick", training_layout="multi_target", branch_count=count)
    branches = {
        f"branch_{slot}_{name}"
        for slot in range(1, int(count) + 1)
        for name in ("task", "target_column", "input_columns")
    }
    assert set(asked) == SHARED | {"branch_count"} | branches


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_detailed_setup_stays_the_default_and_keeps_its_questions(layout):
    """Existing answer files omit the new choice and must keep the full guided path."""
    assert PROPERTIES["setup_mode"]["default"] == "detailed"
    detailed = set(_asked(training_layout=layout))
    quick = set(_asked(setup_mode="quick", training_layout=layout))
    assert quick < detailed
    assert {"compute_mode", "retraining_mode", "deployment_identity"} <= detailed
    assert len(detailed) > 40


def test_quick_choice_is_asked_before_every_question_it_hides():
    """The CLI evaluates a skip rule against earlier answers only."""
    gate = PROPERTIES["setup_mode"]["order"]
    assert "skip_prompt_if" not in PROPERTIES["setup_mode"]
    hidden = [
        name
        for name, field in PROPERTIES.items()
        if '"setup_mode"' in json.dumps(field.get("skip_prompt_if", {}))
    ]
    assert len(hidden) > 2000
    assert all(PROPERTIES[name]["order"] > gate for name in hidden)
    assert all("default" in PROPERTIES[name] for name in hidden)


def test_quick_rule_is_only_added_as_an_alternative_to_each_existing_rule():
    """Replacing a rule would reveal settings that were hidden, or hide ones that were asked."""
    always_hidden = [
        name for name, field in PROPERTIES.items() if field.get("skip_prompt_if") == {}
    ]
    assert {"catalog", "schema", "model_params", "search_space"} <= set(always_hidden)
    for name, field in PROPERTIES.items():
        rule = field.get("skip_prompt_if")
        if rule in (None, {}, QUICK):
            continue
        assert set(rule) == {"anyOf"} or '"setup_mode"' not in json.dumps(rule), name
        if '"setup_mode"' in json.dumps(rule):
            assert rule["anyOf"][1] == QUICK and len(rule["anyOf"]) == 2, name
            assert '"setup_mode"' not in json.dumps(rule["anyOf"][0]), name


def test_quick_rule_needs_an_answer_instead_of_matching_a_missing_one():
    """A partial answer set must ask the question rather than silently take its default."""
    rule = PROPERTIES["compute_mode"]["skip_prompt_if"]
    assert not Draft7Validator(rule).is_valid({})
    assert not _cli_valid(rule, {})
    assert _cli_valid(rule, {"setup_mode": "quick"})


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_quick_setup_holds_under_the_rules_the_cli_itself_evaluates(layout):
    """The CLI ignores not/allOf, so the promise must not depend on a full JSON Schema reader."""
    before = {name: _without_quick_rule(field) for name, field in PROPERTIES.items()}
    detailed = _cli_asked(PROPERTIES, training_layout=layout)
    assert [name for name in detailed if name != "setup_mode"] == [
        name for name in _cli_asked(before, training_layout=layout) if name != "setup_mode"
    ]
    quick = _cli_asked(PROPERTIES, setup_mode="quick", training_layout=layout)
    assert {"training_layout", "project_name", "setup_mode", "source_table_name"} <= set(quick)
    assert set(quick) <= ONE_TARGET | {"branch_count"} | {
        name for name in PROPERTIES if re.fullmatch(r"branch_[1-8]_(task|\w+_columns?)", name)
    }
    assert len(quick) < len(detailed)


def test_upgrade_steps_have_one_central_home_that_first_run_points_to():
    """Steps written after a project exists cannot live in that project's own copy."""
    repository = ROOT.parents[2]
    changelog = (repository / "changelog/0.9.x.md").read_text(encoding="utf-8")
    release = changelog.split("\n## v0.9.4\n")[1].split("\n## ")[0]
    assert "### Upgrade steps for generated Databricks projects" in release
    assert "initialize_monitoring_store" in release
    first_run = (TEMPLATE / "START_HERE.md.tmpl").read_text(encoding="utf-8")
    assert "changelog/0.9.x.md" in first_run
    assert "Upgrade steps for generated Databricks projects" in first_run
