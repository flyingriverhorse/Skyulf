"""Competition setup shares evaluation controls and keeps recipes in project code."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[3] / "templates/databricks"


def _setup(**overrides):
    """Read actual initializer predicates with competition selected."""
    properties = json.loads((ROOT / "databricks_template_schema.json").read_text())["properties"]
    values = {name: item["default"] for name, item in properties.items()}
    values.update(training_layout="model_competition", **overrides)
    visible = {
        name
        for name, item in properties.items()
        if not Draft7Validator(item.get("skip_prompt_if", False)).is_valid(values)
    }
    return properties, visible


def test_competition_prompts_share_target_and_hide_individual_model_settings():
    """Guided candidates share search controls while retaining one target and CV policy."""
    properties, visible = _setup()
    assert "model_competition" in properties["training_layout"]["enum"]
    assert {"task", "target_column", "input_columns", "cv_type", "cv_folds"} <= visible
    assert {"regression_metric", "quality_threshold", "promotion_policy"} <= visible
    hidden = {"cv_enabled", "regression_model", "classification_model", "model_params"}
    hidden.update({"regression_search_metric", "classification_search_metric", "search_space"})
    assert not visible & hidden
    assert {"competition_candidate_count", "search_strategy", "search_n_trials"} <= visible


@pytest.mark.parametrize("method", ["time_series_split", "nested_cv"])
def test_competition_temporal_policy_opens_clock_without_cv_toggle(method):
    """Mandatory competition CV must expose its temporal clock despite the old false default."""
    _, visible = _setup(cv_type=method, cv_nested_type="time_series_split")
    assert {
        "event_column",
        "event_time_kind",
        "window_timezone",
        "holdout_months",
        "monthly_lookback_months",
        "cv_gap",
        "cv_test_size",
        "cv_max_train_size",
    } <= visible
    assert not visible & {"split_strategy", "cv_shuffle", "cv_random_state", "cv_group_column"}


@pytest.mark.parametrize("method", ["group_k_fold", "nested_cv"])
def test_competition_group_policy_shares_one_group_column(method):
    """Group CV controls must remain available for the single shared evaluation plan."""
    _, visible = _setup(cv_type=method, cv_nested_type="stratified_group_k_fold")
    assert "cv_group_column" in visible
    assert "cv_gap" not in visible


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("compute_mode", ["serverless", "policy_cluster"])
def test_cli_competition_keeps_two_jobs_and_task_matching_placeholder(tmp_path, task, compute_mode):
    """Real CLI rendering must enable common CV and retain the existing lifecycle graph."""
    from test_databricks_bundle_generation import CLI, PROFILE, _generate_project, _read_jobs

    from skyulf.integrations.databricks.project import load_project_workflow

    if not CLI or not PROFILE:
        pytest.skip("Set SKYULF_BUNDLE_CLI_TEST_PROFILE to opt into installed CLI generation.")
    project = _generate_project(
        tmp_path,
        training_layout="model_competition",
        task=task,
        compute_mode=compute_mode,
        cluster_policy_name="existing_policy",
    )
    config = json.loads((project / "config/workflow.json").read_text())
    assert config["training_layout"] == "model_competition"
    assert config["cv_enabled"] is True
    assert config["competition_max_trials"] == 1000
    assert config["competition_max_candidates"] == 8
    assert config["pipeline"]["modeling"]["type"] == (
        "logistic_regression" if task == "classification" else "ridge_regression"
    )
    jobs = _read_jobs(project)
    assert "train_and_tune" in {task["task_key"] for task in jobs["train"]["tasks"]}
    assert all(task["task_key"] != "train_models" for task in jobs["train"]["tasks"])
    assert (project / "src/modeling/model_competition.py").is_file()
    loaded = load_project_workflow(config, project / "src/features")
    suffix = "classifier" if task == "classification" else "regressor"
    first = "logistic_regression" if task == "classification" else "ridge_regression"
    assert set(loaded["competition"]["candidates"]) == {
        f"{first}_1",
        f"random_forest_{suffix}_2",
        f"voting_{suffix}_3",
    }
    for entry in loaded["competition"]["candidates"].values():
        modeling = entry["pipeline"]["modeling"]
        assert modeling["metric"] == config["metric"].removeprefix("heldout_")
        assert modeling["n_trials"] == 10 and modeling["search_space"]
    preview = subprocess.run(
        [sys.executable, str(project / "src/tools/preview.py"), "--action", "train"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert preview.returncode == 0, preview.stdout + preview.stderr
    for name in loaded["competition"]["candidates"]:
        assert f"Candidate: {name}" in preview.stdout
