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
    "source_layout",
    "include_source_sql",
    "initial_scoring",
    "record_key_columns",
    # Monitoring defaults to off; its schema is asked only after "monitoring: true".
    "monitoring",
}
LABELS = {"label_table_name", "label_available_at_column"}
ONE_TARGET = SHARED | {"task", "input_columns", "target_column"}
LAYOUTS = ["single_model", "model_competition", "multi_target"]


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


def _cli_asked(**answers):
    """Replay the CLI: each prompt sees only the values assigned before it."""
    values, asked = {}, []
    for name, field in PROPERTIES.items():
        rule = field.get("skip_prompt_if")
        if rule is not None and _cli_valid(rule, values):
            values[name] = field["default"]
            continue
        asked.append(name)
        values[name] = answers.get(name, field["default"])
    return asked


def _keywords(node, found):
    """Collect every schema keyword used outside the per-question constraints."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties":
                found |= {keyword for constraint in value.values() for keyword in constraint}
            else:
                found.add(key)
                _keywords(value, found)
    for item in node if isinstance(node, list) else ():
        _keywords(item, found)
    return found


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


@pytest.mark.parametrize("layout", LAYOUTS)
def test_quick_setup_is_default_and_detailed_keeps_its_questions(layout):
    """First-time users get the short wizard; detailed setup keeps every relevant choice."""
    assert PROPERTIES["setup_mode"]["default"] == "quick"
    detailed = set(_asked(setup_mode="detailed", training_layout=layout))
    quick = set(_asked(setup_mode="quick", training_layout=layout))
    assert set(_cli_asked(training_layout=layout)) == quick
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
    # About 1700: ensemble weights and voting/stacking settings are hidden in every mode.
    assert len(hidden) > 1500
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
        if rule in (None, {}, QUICK) or '"setup_mode"' not in json.dumps(rule):
            continue
        assert set(rule) == {"anyOf"} and rule["anyOf"][-1] == QUICK, name
        assert '"setup_mode"' not in json.dumps(rule["anyOf"][:-1]), name


def test_quick_rule_needs_an_answer_instead_of_matching_a_missing_one():
    """A partial answer set must ask the question rather than silently take its default."""
    rule = PROPERTIES["compute_mode"]["skip_prompt_if"]
    assert not Draft7Validator(rule).is_valid({})
    assert not _cli_valid(rule, {})
    assert _cli_valid(rule, {"setup_mode": "quick"})


def test_skip_rules_use_only_what_the_cli_evaluates():
    """The CLI drops not/allOf/oneOf silently, which hid every branch and candidate question."""
    used = set()
    for field in PROPERTIES.values():
        _keywords(field.get("skip_prompt_if", {}), used)
    assert used <= {"anyOf", "required", "const", "enum", "pattern"}


@pytest.mark.parametrize("mode", ["detailed", "quick"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_cli_asks_exactly_the_questions_the_schema_describes(layout, task, mode):
    """A full JSON Schema reader and the CLI's partial one must walk the same wizard."""
    answers = {"training_layout": layout, "task": task, "setup_mode": mode}
    assert _cli_asked(**answers) == _asked(**answers)
    rich = answers | {"cv_enabled": "true", "search_settings": "custom", "branch_count": "3"}
    assert _cli_asked(**rich) == _asked(**rich)


@pytest.mark.parametrize(
    ("layout", "expected"),
    [("multi_target", "branch_count"), ("model_competition", "competition_candidate_count")],
)
def test_cli_reaches_branch_and_candidate_questions(layout, expected):
    """These questions were unreachable while their rules relied on an ignored keyword."""
    asked = _cli_asked(setup_mode="detailed", training_layout=layout)
    assert expected in asked
    assert "search_strategy" in _cli_asked(
        setup_mode="detailed", training_layout="single_model", search_settings="custom"
    )


@pytest.mark.parametrize(
    ("answers", "prefix"),
    [
        (
            {"training_layout": "single_model", "regression_model": "voting_regressor"},
            "single_ensemble_",
        ),
        (
            {
                "training_layout": "model_competition",
                "competition_regression_model_1": "voting_regressor",
            },
            "competition_ensemble_1_",
        ),
        (
            {"training_layout": "multi_target", "branch_1_regression_model": "stacking_regressor"},
            "branch_1_ensemble_",
        ),
    ],
)
def test_ensembles_ask_member_models_but_not_their_settings(answers, prefix):
    """Weights and voting/stacking settings would multiply the wizard; members stay a choice."""
    hidden = re.compile(
        r".*ensemble_(?:[1-8]_)?(?:\w+_weight_\d+|voting|cv|passthrough|calibrat\w+)"
    )
    settings = [name for name in PROPERTIES if hidden.fullmatch(name)]
    assert len(settings) > 500
    assert all(PROPERTIES[name]["skip_prompt_if"] == {} for name in settings)
    assert all("default" in PROPERTIES[name] for name in settings)
    asked = _cli_asked(setup_mode="detailed", task="regression", **answers)
    members = {name for name in asked if name.startswith(prefix)}
    assert {name.removeprefix(prefix) for name in members} >= {
        "regression_base_count",
        "regression_base_1",
        "regression_base_2",
    }
    assert not members & set(settings)


def test_each_list_offers_its_default_first():
    """The CLI highlights the first choice, so Enter must mean the documented default."""
    wrong = [
        name
        for name, field in PROPERTIES.items()
        if "enum" in field and field["enum"][0] != field["default"]
    ]
    assert not wrong


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


def test_quick_setup_asks_for_the_monitoring_schema_once_monitoring_is_on():
    """Turning monitoring on in quick setup must still ask where the shared store lives."""
    asked = _asked(setup_mode="quick", monitoring="true")
    assert "monitoring_schema" in asked
    # The old catalog answer remains only as a removed-answer guard that accepts blank.
    assert PROPERTIES["monitoring_catalog"]["enum"] == [""]
    assert "monitoring_catalog" not in asked


@pytest.mark.parametrize(
    ("source_layout", "monitoring", "asks_labels"),
    [
        ("single_table", "false", False),
        ("single_table", "true", True),
        ("separate_outcomes", "false", True),
        ("separate_all", "false", True),
    ],
)
def test_outcome_table_is_asked_only_when_something_reads_it(
    source_layout, monitoring, asks_labels
):
    """One embedded-label table without monitoring has no reader for an outcomes table."""
    asked = _cli_asked(source_layout=source_layout, monitoring=monitoring)
    assert (set(asked) >= LABELS) is asks_labels
    assert LABELS.isdisjoint(asked) is not asks_labels
    assert asked.index("monitoring") < asked.index("source_table_name")


def test_first_single_model_batch_project_asks_twelve_questions():
    """The default quick path stays short; each prompt here has no usable default."""
    asked = _cli_asked(training_layout="single_model")
    assert len(asked) == 12
