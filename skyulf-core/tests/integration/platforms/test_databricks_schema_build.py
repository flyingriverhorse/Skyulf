"""Keep modular wizard sources and the CLI's generated schema synchronized."""

import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "templates/databricks"


def test_committed_schema_matches_sources():
    """CI must catch forgotten generation after any wizard source edit."""
    build = runpy.run_path(str(ROOT / "build_schema.py"))["render_schema"]
    assert build(ROOT / "schema") == (ROOT / "databricks_template_schema.json").read_text(
        encoding="utf-8"
    )


def test_ensemble_expansion_has_unique_ordered_candidate_fields():
    """Repeating the question group must leave no placeholders or overlapping prompt orders."""
    rendered = (ROOT / "databricks_template_schema.json").read_text(encoding="utf-8")
    properties = json.loads(rendered)["properties"]
    orders = [field["order"] for field in properties.values()]
    assert "SLOT" not in rendered
    assert len(orders) == len(set(orders))
    assert all(isinstance(order, int) for order in orders)
    assert (
        properties["source_table_name"]["order"]
        < properties["score_source_table_name"]["order"]
        < properties["prediction_table_name"]["order"]
    )
    assert (
        properties["target_column"]["order"]
        < properties["training_version"]["order"]
        < properties["cv_enabled"]["order"]
    )
    for slot in range(1, 9):
        name = f"competition_ensemble_{slot}_regression_base_count"
        assert name in properties
        assert properties[name]["order"] < properties[f"competition_ensemble_{slot}_cv"]["order"]


def test_ensemble_group_rejects_names_without_slot(tmp_path):
    """An accidentally shared question must not be overwritten once per candidate."""
    build = runpy.run_path(str(ROOT / "build_schema.py"))["render_schema"]
    (tmp_path / "metadata.json").write_text("{}")
    (tmp_path / "competition_ensemble.json").write_text('{"shared": {"order": 0}}')
    with pytest.raises(ValueError, match="must start with competition_ensemble_SLOT_"):
        build(tmp_path)


@pytest.mark.parametrize("duplicate", ["within_file", "across_files"])
def test_duplicate_properties_fail_without_overwriting(duplicate, tmp_path):
    """Duplicate names must fail loudly instead of silently replacing user prompts."""
    build = runpy.run_path(str(ROOT / "build_schema.py"))["render_schema"]
    (tmp_path / "metadata.json").write_text("{}")
    first = '{"project_name": {"order": 0}}'
    if duplicate == "within_file":
        first = '{"project_name": {"order": 0}, "project_name": {"order": 1}}'
    else:
        (tmp_path / "second.json").write_text(first)
    (tmp_path / "first.json").write_text(first)
    with pytest.raises(ValueError, match="Duplicate.*project_name"):
        build(tmp_path)


def test_schema_properties_follow_prompt_order(tmp_path):
    """Source filenames must not change the intended wizard order."""
    build = runpy.run_path(str(ROOT / "build_schema.py"))["render_schema"]
    (tmp_path / "metadata.json").write_text('{"welcome_message": "Hello"}')
    (tmp_path / "a.json").write_text('{"later": {"order": 2}}')
    (tmp_path / "z.json").write_text('{"first": {"order": 1}}')
    result = json.loads(build(tmp_path))
    assert result["welcome_message"] == "Hello"
    assert list(result["properties"]) == ["first", "later"]


def test_duplicate_prompt_orders_fail_with_both_field_names(tmp_path):
    """Ambiguous wizard sequencing must fail during generation, before CLI use."""
    build = runpy.run_path(str(ROOT / "build_schema.py"))["render_schema"]
    (tmp_path / "metadata.json").write_text("{}")
    (tmp_path / "prompts.json").write_text('{"source": {"order": 7}, "scoring": {"order": 7}}')
    with pytest.raises(ValueError, match="Duplicate prompt order 7: source, scoring"):
        build(tmp_path)


def test_schema_rejects_prompt_order_precision_instead_of_truncating(tmp_path):
    """A malformed insertion position must not silently change wizard sequencing."""
    build = runpy.run_path(str(ROOT / "build_schema.py"))["render_schema"]
    source = tmp_path / "schema"
    shutil.copytree(ROOT / "schema", source)
    path = source / "project.json"
    fields = json.loads(path.read_text(encoding="utf-8"))
    fields["training_version"]["order"] = 12.19
    path.write_text(json.dumps(fields), encoding="utf-8")
    with pytest.raises(ValueError, match="training_version.*one decimal place"):
        build(source)


def test_schema_keeps_each_prompt_on_one_line(tmp_path):
    """Compact output must retain nested conditions, escaping and JSON member order."""
    build = runpy.run_path(str(ROOT / "build_schema.py"))["render_schema"]
    metadata = {"welcome_message": "Hello\nworld", "version": 1}
    first = {
        "order": 1,
        "description": 'A "quoted" café prompt',
        "skip_prompt_if": {"z": [False, None], "a": {"const": "a\\b"}},
    }
    later = {"order": 2, "enum": ["b", "a"], "default": "b"}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "prompts.json").write_text(
        json.dumps({"later": later, "first": first}), encoding="utf-8"
    )

    rendered = build(tmp_path)

    assert rendered.splitlines() == [
        "{",
        '  "welcome_message": "Hello\\nworld",',
        '  "version": 1,',
        '  "properties": {',
        '    "first": {"order":1,"description":"A \\"quoted\\" café prompt",'
        '"skip_prompt_if":{"z":[false,null],"a":{"const":"a\\\\b"}}},',
        '    "later": {"order":2,"enum":["b","a"],"default":"b"}',
        "  }",
        "}",
    ]
    assert rendered.endswith("\n")
    assert json.loads(rendered) == metadata | {"properties": {"first": first, "later": later}}


def test_check_is_read_only_and_build_works_outside_template_directory(tmp_path):
    """Maintainers can regenerate from any working directory; CI never rewrites files."""
    root = tmp_path / "template"
    root.mkdir()
    shutil.copyfile(ROOT / "build_schema.py", root / "build_schema.py")
    shutil.copytree(ROOT / "schema", root / "schema")
    output = root / "databricks_template_schema.json"
    output.write_text("stale\n")
    command = [sys.executable, str(root / "build_schema.py")]
    stale = subprocess.run(command + ["--check"], cwd=tmp_path, capture_output=True, text=True)
    assert stale.returncode == 1
    assert "out of date" in stale.stderr
    assert output.read_text() == "stale\n"
    generated = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert generated.returncode == 0, generated.stderr
    check = subprocess.run(command + ["--check"], cwd=tmp_path, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    assert json.loads(output.read_text()) == json.loads(
        (ROOT / "databricks_template_schema.json").read_text()
    )


def test_weight_prompt_precedes_models_and_filters_supported_choices():
    """Weighted menus must use the same configured-model capability as training."""
    from skyulf.modeling.capabilities import model_supports_sample_weight

    schema = json.loads((ROOT / "databricks_template_schema.json").read_text())
    properties = schema["properties"]
    assert properties["sample_weight_enabled"]["default"] == "false"
    assert (
        properties["sample_weight_enabled"]["order"]
        < properties["competition_regression_model_1"]["order"]
    )
    for task in ("regression", "classification"):
        original = properties[f"{task}_model"]["enum"]
        weighted = properties[f"{task}_model_weighted"]["enum"]
        assert weighted == [name for name in original if model_supports_sample_weight(name)]
        assert properties[f"{task}_model_weighted"]["default"] in weighted


@pytest.mark.parametrize("prefix", ["", "branch_1_"])
def test_weighted_prompt_conditions_use_only_active_model_menu(prefix):
    """Inactive unweighted defaults must not reveal unrelated ensemble questions."""
    from jsonschema import Draft7Validator

    properties = json.loads((ROOT / "databricks_template_schema.json").read_text())["properties"]
    values = {key: field["default"] for key, field in properties.items()}
    values.update(training_layout="multi_target" if prefix else "single_model")
    values[prefix + "task"] = "classification"
    values[prefix + "sample_weight_enabled"] = "true"
    values[prefix + "classification_model"] = "logistic_regression"
    values[prefix + "classification_model_weighted"] = "voting_classifier"
    ensemble = prefix + "ensemble_" if prefix else "single_ensemble_"
    assert not Draft7Validator(
        properties[ensemble + "classification_base_1_weighted"]["skip_prompt_if"]
    ).is_valid(values)
    assert Draft7Validator(
        properties[ensemble + "classification_base_1"]["skip_prompt_if"]
    ).is_valid(values)
    values[prefix + "classification_model_weighted"] = "logistic_regression"
    assert Draft7Validator(
        properties[ensemble + "classification_base_1_weighted"]["skip_prompt_if"]
    ).is_valid(values)


def test_schema_rejects_unavailable_ensemble_dependency(monkeypatch):
    """A missing optional member must not silently fall back while building supported menus."""
    from skyulf.modeling.ensemble import BASE_ESTIMATORS_CLF

    select = runpy.run_path(str(ROOT / "build_schema.py"))["_supported_choices"]
    monkeypatch.delitem(BASE_ESTIMATORS_CLF, "random_forest")
    with pytest.raises(ValueError, match="model dependencies"):
        select("single_ensemble_classification_base_1", ["random_forest"])
