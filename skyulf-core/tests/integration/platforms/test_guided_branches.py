"""Guided branch setup retains independent models, searches and evaluation policies."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[3] / "templates/databricks"


def _prototype_properties():
    """Expand the source prototype so its predicates can be checked before rendering."""
    source = (ROOT / "schema/branches.json").read_text(encoding="utf-8")
    return {
        name: field
        for slot in range(1, 9)
        for name, field in json.loads(source.replace("SLOT", str(slot))).items()
    }


def _visible(**overrides):
    """Evaluate branch prompts with the defaults the initializer will supply."""
    properties = _prototype_properties()
    values = {name: field["default"] for name, field in properties.items()}
    values.update(training_layout="multi_target", branch_count="2")
    values.update(overrides)
    return {
        name
        for name, field in properties.items()
        if not Draft7Validator(field["skip_prompt_if"]).is_valid(values)
    }


@pytest.mark.parametrize("count", ["2", "3", "8"])
def test_branch_prompts_follow_count_and_independent_tasks(count):
    """Mixed targets must expose their own models and no unused branch slots."""
    shown = _visible(branch_count=count, branch_2_task="classification")
    for slot in range(1, 9):
        task = "classification" if slot == 2 else "regression"
        other = "regression" if slot == 2 else "classification"
        assert (f"branch_{slot}_{task}_model" in shown) == (slot <= int(count))
        assert f"branch_{slot}_{other}_model" not in shown
        assert f"branch_{slot}_search_space" not in shown
        assert (f"branch_{slot}_search_strategy" in shown) == (slot <= int(count))
    assert not any(name.endswith("model_params") for name in shown)


@pytest.mark.parametrize("layout", ["single_model", "model_competition"])
def test_branch_questions_are_hidden_for_other_layouts(layout):
    """A different layout must never request branch-specific configuration."""
    assert not _visible(training_layout=layout)


def test_branch_model_menus_follow_the_existing_model_catalog():
    """The copied prototype menus must stay aligned with the authoritative model choices."""
    models = json.loads((ROOT / "schema/models.json").read_text())
    props = _prototype_properties()
    for task in ("regression", "classification"):
        assert props[f"branch_1_{task}_model"]["enum"] == models[f"{task}_model"]["enum"]


def test_branch_cv_and_strategy_predicates_are_independent():
    """Time, group and Optuna controls should only open for their own branch."""
    shown = _visible(
        branch_1_cv_enabled="true",
        branch_1_cv_type="nested_cv",
        branch_1_cv_nested_type="time_series_split",
        branch_1_search_strategy="optuna",
        branch_1_search_settings="custom",
        branch_2_cv_enabled="true",
        branch_2_cv_type="group_k_fold",
    )
    assert {
        "branch_1_cv_inner_folds",
        "branch_1_cv_gap",
        "branch_1_event_column",
        "branch_1_search_sampler",
        "branch_1_search_timeout",
        "branch_2_cv_group_column",
    } <= shown
    assert (
        not {
            "branch_1_cv_group_column",
            "branch_1_search_tune_threshold",
            "branch_2_cv_gap",
            "branch_2_search_sampler",
            "branch_2_event_column",
        }
        & shown
    )


def _generate(tmp_path, **overrides):
    """Render through the real CLI only when the operator explicitly opts in."""
    from test_databricks_bundle_generation import CLI, PROFILE, _generate_project

    if not CLI or not PROFILE:
        pytest.skip("Set SKYULF_BUNDLE_CLI_TEST_PROFILE for real CLI generation.")
    return _generate_project(tmp_path, training_layout="multi_target", **overrides)


def _factory(project):
    """Load generated YAML through the same branch adapter as the notebook."""
    from skyulf.integrations.databricks.projects.yaml_config import read_training_config
    from skyulf.integrations.databricks.projects.yaml_models import training_branches

    document = read_training_config(project / "config")
    assert document is not None
    return lambda: training_branches(document)[0]


def _load_configs(project):
    """Resolve real project recipes with explicit deployment bindings and no cloud calls."""
    from skyulf.integrations.databricks.jobs.training.branch_notebook import (
        load_training_branch_configs,
    )

    return load_training_branch_configs(
        {
            "config_path": str(project / "config/training.yml"),
            "catalog": "workspace",
            "input_schema": "inputs",
            "output_schema": "outputs",
            "metadata_schema": "models",
            "resource_suffix": "_dev",
            "workflow_contract": "3",
            "deployed_score_handoff": "disabled",
        }
    )


@pytest.mark.parametrize("count", ["2", "8"])
def test_cli_branches_work_without_hand_edits_and_return_fresh_copies(tmp_path, count):
    """Initialized multi-target projects must contain every requested independent branch."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config
    from skyulf.integrations.databricks.training.tuning.cv import CVSpec

    project = _generate(tmp_path, branch_count=count)
    factory = _factory(project)
    entries = factory()
    assert len(entries) == int(count)
    entries["branch_1"]["workflow"]["pipeline"]["preprocessing"].append({"changed": True})
    assert factory()["branch_1"]["workflow"]["pipeline"]["preprocessing"] == []
    configs = _load_configs(project)
    for name, config in configs.items():
        checked = validate_workflow_config(config, action="train")
        CVSpec.from_workflow(checked).validate_pipeline(
            checked["pipeline"], target_column=checked["target_column"]
        )
        assert checked["model_name"] == f"workspace.models.sm33_generated_{name}_dev"
        assert checked["target_column"] == name.replace("branch", "target")


@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "halving_random", "optuna"])
def test_cli_mixed_branches_keep_independent_search_and_ensemble_settings(tmp_path, strategy):
    """Changing one branch's tuning must preserve its sibling's model, space and recipe."""
    from skyulf.integrations.databricks.training.tuning.cv import CVSpec

    project = _generate(
        tmp_path,
        branch_1_name="revenue",
        branch_1_regression_model="ridge_regression",
        branch_1_preprocessing_recipe="example_imputer",
        branch_1_pre_split_recipe="example_complete_inputs",
        branch_1_input_columns="feature_value, category",
        branch_1_search_strategy=strategy,
        branch_1_search_space='{"alpha": [0.1, 1.0]}',
        branch_1_search_n_trials="2",
        branch_1_search_settings="custom",
        branch_1_search_factor="2",
        branch_1_search_sampler="random",
        branch_1_search_pruning="false",
        branch_1_search_pruner="none",
        branch_1_quality_threshold="2.5",
        branch_1_min_improvement=0.1,
        branch_2_name="churn",
        branch_2_task="classification",
        branch_2_classification_model="voting_classifier",
        branch_2_ensemble_voting="hard",
        branch_2_search_strategy="grid",
        branch_2_search_space='{"logistic_regression__C": [0.5, 1.0]}',
        branch_2_classification_search_metric="f1",
        branch_2_classification_metric="heldout_f1",
        branch_2_quality_threshold="0.8",
        branch_2_quality_gates='{"heldout_accuracy": 0.75}',
    )
    configs = _load_configs(project)
    for config in configs.values():
        CVSpec.from_workflow(config).validate_pipeline(
            config["pipeline"], target_column=config["target_column"]
        )
    revenue, churn = configs["revenue"], configs["churn"]
    left, right = revenue["pipeline"]["modeling"], churn["pipeline"]["modeling"]
    assert left["strategy"] == strategy and left["n_trials"] == 2
    assert left["search_space"] == {"alpha": [0.1, 1.0]}
    assert left["base_model"]["type"] == "ridge_regression"
    assert revenue["pipeline"]["preprocessing"][0]["transformer"] == "SimpleImputer"
    assert len(revenue["pre_split_steps"]) == 1
    assert revenue["pre_split_steps"][0]["pre_split"]["learns_from_data"] is False
    assert revenue["quality_threshold"] == 2.5 and revenue["min_improvement"] == 0.1
    assert right["strategy"] == "grid" and right["metric"] == "f1"
    assert right["base_model"]["type"] == "voting_classifier"
    assert right["base_model"]["params"]["voting"] == "hard"
    assert right["search_space"] == {"logistic_regression__C": [0.5, 1.0]}
    assert churn["pipeline"]["preprocessing"] == []
    assert churn["pre_split_steps"] == []
    assert churn["metric"] == "heldout_f1" and churn["quality_threshold"] == 0.8
    assert churn["quality_gates"] == {"heldout_accuracy": 0.75}


@pytest.mark.parametrize("policy", ["k_fold", "group_k_fold", "time_series_split"])
def test_cli_nested_branch_cv_clears_inherited_policy(tmp_path, policy):
    """Ordinary siblings must not inherit a nested branch's group or temporal metadata."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config
    from skyulf.integrations.databricks.training.tuning.cv import CVSpec

    settings = {"branch_1_cv_group_column": "customer"} if policy == "group_k_fold" else {}
    if policy == "time_series_split":
        settings.update(branch_1_event_column="event_time", branch_1_cv_gap="1")
    project = _generate(
        tmp_path,
        branch_1_cv_enabled="true",
        branch_1_cv_type="nested_cv",
        branch_1_cv_nested_type=policy,
        branch_1_cv_inner_folds="2",
        **settings,
    )
    configs = _load_configs(project)
    for config in configs.values():
        checked = validate_workflow_config(config, action="train")
        CVSpec.from_workflow(checked).validate_pipeline(
            checked["pipeline"],
            target_column=checked["target_column"],
            event_column=checked.get("event_column"),
        )
    left, right = configs["branch_1"], configs["branch_2"]
    assert left["cv_inner_folds"] == 2 and left["cv_nested_type"] == policy
    assert CVSpec.from_workflow(right).inner_folds is None
    assert CVSpec.from_workflow(right).nested_type == "auto"
    assert right.get("cv_group_column") is None and right.get("event_column") is None
    if policy == "time_series_split":
        assert left["split_strategy"] == "temporal" and left["cv_shuffle"] is False
        assert left["training_window_mode"] == "rolling_calendar"
    assert right["split_strategy"] == "random" and right["cv_enabled"] is False


def test_cli_duplicate_branch_names_fail_before_silent_overwrite(tmp_path):
    """Two equal names must fail instead of silently dropping a requested target."""
    project = _generate(tmp_path, branch_1_name="duplicate", branch_2_name="duplicate")
    with pytest.raises(ValueError, match="Duplicate configuration key"):
        _factory(project)
