"""Real CLI regressions for weighted competition graphs and policy-cluster YAML."""

import json
import runpy

import pytest
from test_databricks_bundle_generation import (
    CLI,
    PROFILE,
    _generate_project,
    _read_bundle,
    _read_jobs,
)

pytestmark = pytest.mark.skipif(
    not PROFILE or not CLI,
    reason="Set SKYULF_BUNDLE_CLI_TEST_PROFILE for installed CLI generation.",
)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("weighted,count", [("false", 3), ("true", 3), ("true", 8)])
def test_competition_graph_uses_selected_candidate_names(tmp_path, task, weighted, count):
    """Weighted answers must reach initialization, training, SHAP and winner selection."""
    from skyulf.integrations.databricks.training_node_notebook import validate_model_task_names

    selected = (
        ["decision_tree_classifier", "logistic_regression", "random_forest_classifier"]
        if task == "classification"
        else ["decision_tree_regressor", "linear_regression", "random_forest_regressor"]
    )
    answers = {}
    for slot in range(1, count + 1):
        # Deliberately keep the unused answers different from the weighted ones.
        answers[f"competition_{task}_model_{slot}"] = selected[slot % 3]
        answers[f"competition_{task}_model_{slot}_weighted"] = selected[(slot + 1) % 3]
    project = _generate_project(
        tmp_path,
        task=task,
        training_layout="model_competition",
        sample_weight_enabled=weighted,
        weight_column="training_weight",
        competition_candidate_count=str(count),
        shap_enabled="true",
        **answers,
    )
    definitions = runpy.run_path(str(project / "src/modeling/model_competition.py"))
    names = list(definitions["build_candidates"](task))
    suffix = "_weighted" if weighted == "true" else ""
    expected = [
        f"{answers[f'competition_{task}_model_{slot}{suffix}']}_{slot}"
        for slot in range(1, count + 1)
    ]
    assert names == expected
    assert definitions["WEIGHT_COLUMN"] == ("training_weight" if weighted == "true" else None)
    tasks = {item["task_key"]: item for item in _read_jobs(project)["train"]["tasks"]}
    initialize = tasks["initialize_run"]["notebook_task"]["base_parameters"]
    validate_model_task_names(initialize, set(names))
    assert json.loads(initialize["model_keys_json"]) == names
    assert tasks["select_best_model"]["depends_on"] == [
        {"task_key": f"train_{name}"} for name in names
    ]
    for name in names:
        train = tasks[f"train_{name}"]["notebook_task"]["base_parameters"]
        shap = tasks[f"shap_{name}"]
        assert train["model_key"] == name
        assert shap["depends_on"] == [{"task_key": f"train_{name}"}]
        assert shap["notebook_task"]["base_parameters"]["reference_json"] == (
            "{{tasks.train_" + name + ".values.reference_json}}"
        )
    assert {name for name in tasks if name.startswith("train_")} == {
        f"train_{name}" for name in names
    }


@pytest.mark.parametrize(
    "policy,key,value",
    [
        ('Approved "ML" policy', "CostCenter", 'R&D "east"'),
        (r"team\finance", "CostCenter", r"team\finance"),
        ("Approved policy", "true", "true"),
        ("Approved policy", "null", "null"),
    ],
)
def test_policy_cluster_answers_round_trip_as_yaml_strings(tmp_path, policy, key, value):
    """User policy names and tag strings cannot become YAML syntax, escapes or booleans."""
    project = _generate_project(
        tmp_path,
        compute_mode="policy_cluster",
        cluster_policy_name=policy,
        cost_tag_key=key,
        cost_tag_value=value,
    )
    variables = _read_bundle(project)["variables"]
    assert variables["cluster_policy_id"]["lookup"]["cluster_policy"] == policy
    assert variables["cost_tag_value"]["default"] == value
    jobs = _read_jobs(project)
    assert "job_clusters" not in jobs["monitoring"]
    for job in (jobs["train"], jobs["score"]):
        cluster = job["job_clusters"][0]["new_cluster"]
        assert cluster["custom_tags"] == {key: "${var.cost_tag_value}"}
