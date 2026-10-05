"""Optional branch training preserves the two-job bundle and target isolation."""

import ast
import json
import os
import re
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

TEMPLATE = Path(__file__).resolve().parents[3] / "templates/databricks/template/{{.project_name}}"


def test_bundle_wheel_references_match_core_build_version():
    """Generated jobs must install the wheel produced by the current Core release."""
    core = Path(__file__).resolve().parents[3]
    metadata = tomllib.loads((core / "pyproject.toml").read_text())
    assert "version" in metadata["project"]["dynamic"]
    setup = next(
        node
        for node in ast.walk(ast.parse((core / "setup.py").read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    version = ast.literal_eval(next(item.value for item in setup.keywords if item.arg == "version"))
    declared = json.loads((TEMPLATE / "deployment/artifact.json").read_text())
    assert declared["version"] == version
    bundle = yaml.safe_load(
        (TEMPLATE / "databricks.yml.tmpl").read_text().replace("{{.project_name}}", "wheel_check")
    )
    artifact = bundle["artifacts"]["skyulf"]
    assert artifact["type"] == "whl"
    assert artifact["files"] == [{"source": "dist/skyulf/*.whl"}]
    assert artifact["build"] == "uv run --no-project python src/tools/build_wheel.py"
    for filename in ("train.job.yml.tmpl", "score.job.yml.tmpl"):
        source = (TEMPLATE / "resources" / filename).read_text()
        references = re.findall(r"(?m)^\s*-\s*(?:whl:\s*)?(\S+\.whl)\s*$", source)
        assert references and set(references) == {"../dist/skyulf/*.whl"}
    assert f"Skyulf {version} Bundle:" in (TEMPLATE / "README.md.tmpl").read_text()


def _project(tmp_path, workflow_config):
    """Build an editable project with separate feature recipes for distinct targets."""
    config = deepcopy(workflow_config)
    config.update(promotion_policy="manual_approval", score_handoff="disabled")
    config["pipeline"] = {
        "preprocessing": [],
        "modeling": {"type": "ridge_regression", "params": {}},
    }
    config["pre_split_steps"] = []
    for name in ("training_table", "score_source_table"):
        config[name] = "{catalog}.{input_schema}.source"
    config["model_name"] = "{catalog}.{metadata_schema}.base{resource_suffix}"
    config["prediction_table"] = "{catalog}.{output_schema}.predictions{resource_suffix}"
    path = tmp_path / "config/workflow.json"
    path.parent.mkdir()
    path.write_text(json.dumps(config))
    modeling = tmp_path / "src/modeling"
    modeling.mkdir(parents=True)
    entries = {}
    for name, model, task, metric in (
        ("revenue", "ridge_regression", "regression", "heldout_rmse"),
        ("cost", "linear_regression", "regression", "heldout_mae"),
        ("churn", "logistic_regression", "classification", "heldout_f1"),
    ):
        features = tmp_path / "src/features" / name
        features.mkdir(parents=True)
        features.joinpath("__init__.py").write_text(
            f"BRANCH = {name!r}\ndef build_preprocessing():\n    return []\n"
        )
        entries[name] = {
            "workflow": {
                "target_column": name,
                "task": task,
                "metric": metric,
                "input_columns": [f"{name}_feature"],
                "model_name": "{catalog}.{metadata_schema}." + name + "{resource_suffix}",
                "pipeline": {"preprocessing": [], "modeling": {"type": model, "params": {}}},
            },
            "features_path": f"../features/{name}",
        }
    modeling.joinpath("branches.py").write_text(
        f"def build_training_branches():\n    return {entries!r}\n"
    )
    values = {
        "config_path": str(path),
        "catalog": "workspace",
        "input_schema": "inputs",
        "output_schema": "outputs",
        "metadata_schema": "models",
        "resource_suffix": "_dev",
        "workflow_contract": "3",
        "deployed_score_handoff": "disabled",
        "job_id": "1",
        "job_run_id": "2",
        "repair_count": "0",
        "execution_count": "1",
        "lifecycle_action": "train",
        "experiment_name": "/test/branches",
    }
    return values, entries


def test_branch_loader_preserves_independent_recipes_and_bindings(tmp_path, workflow_config):
    """Different targets must retain their model, source package and active deployment suffix."""
    from skyulf.integrations.databricks.jobs.training.branch_notebook import (
        load_training_branch_configs,
    )

    values, _ = _project(tmp_path, workflow_config)
    before = Path(values["config_path"]).read_text()
    configs = load_training_branch_configs(values)
    assert set(configs) == {"revenue", "cost", "churn"}
    for name, config in configs.items():
        assert config["model_name"] == f"workspace.models.{name}_dev"
        assert config["target_column"] == name
        assert config["input_columns"] == [f"{name}_feature"]
        assert name in config["pipeline"]["project_python_source"]
    configs["revenue"]["pipeline"]["modeling"]["params"]["alpha"] = 2
    assert configs["cost"]["pipeline"]["modeling"]["params"] == {}
    assert configs["churn"]["task"] == "classification"
    assert Path(values["config_path"]).read_text() == before


@pytest.mark.parametrize("change", [{"promotion_policy": "automatic"}, {"score_handoff": "always"}])
def test_branch_loader_rejects_unsupported_policies(tmp_path, workflow_config, change):
    """Multi-target training must never silently enable promotion or scoring."""
    from skyulf.integrations.databricks.jobs.training.branch_notebook import (
        load_training_branch_configs,
    )

    values, _ = _project(tmp_path, workflow_config)
    path = Path(values["config_path"])
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    values["deployed_score_handoff"] = change.get("score_handoff", "disabled")
    with pytest.raises(ValueError, match="manual_approval|disabled"):
        load_training_branch_configs(values)


@pytest.mark.parametrize(
    "change", [{"repair_count": "1"}, {"execution_count": "2"}, {"lifecycle_action": "approve"}]
)
def test_branch_notebook_rejects_repairs_and_operator_actions(tmp_path, workflow_config, change):
    """An unsupported invocation must fail before editable code or cloud services execute."""
    from skyulf.integrations.databricks.jobs.training.branch_notebook import (
        run_branch_training_notebook,
    )

    values, _ = _project(tmp_path, workflow_config)
    values.update(change)
    dbutils = SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: values))
    with pytest.raises(ValueError, match="repair/retry|train"):
        run_branch_training_notebook(None, dbutils)


@pytest.mark.skipif(
    not os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE"), reason="CLI generation opt-in required"
)
@pytest.mark.parametrize("compute", ["serverless", "policy_cluster"])
@pytest.mark.parametrize("policy", ["manual_approval", "automatic"])
@pytest.mark.parametrize("handoff", ["disabled", "after_alias_change"])
def test_multi_target_cli_graph_has_same_three_jobs(tmp_path, compute, policy, handoff):
    """Actual Go expansion must create independent fits and retain a single complete-set join."""
    from test_databricks_bundle_generation import _generate_project, _read_jobs, _synced_sources

    project = _generate_project(
        tmp_path,
        training_layout="multi_target",
        compute_mode=compute,
        model_set_promotion_policy=policy,
        score_handoff=handoff,
    )
    jobs = _read_jobs(project)
    assert set(jobs) == {"train", "score", "monitoring"}
    train = jobs["train"]
    assert train["max_concurrent_runs"] == 1
    assert len(train["tasks"]) == 12
    tasks = {item["task_key"]: item for item in train["tasks"]}
    assert tasks["monitoring_allowed"]["depends_on"] == [{"task_key": "training_report"}]
    assert tasks["monitoring_allowed"]["condition_task"] == {
        "op": "EQUAL_TO",
        "left": "${bundle.mode}",
        "right": "production",
    }
    assert tasks["register_monitor"]["depends_on"] == [
        {"task_key": "monitoring_allowed", "outcome": "true"}
    ]
    assert (
        tasks["register_monitor"]["notebook_task"]["notebook_path"]
        == "../src/jobs/register_set_monitor.py"
    )
    assert tasks["scoring_requested"]["run_if"] == "NONE_FAILED"
    assert tasks["scoring_requested"]["depends_on"] == [
        {"task_key": "monitoring_allowed", "outcome": "false"},
        {"task_key": "register_monitor"},
    ]
    assert tasks["run_batch_scoring"]["depends_on"] == [
        {"task_key": "scoring_requested", "outcome": "true"}
    ]
    assert tasks["run_batch_scoring"]["run_job_task"]["job_id"] == "${resources.jobs.score.id}"
    task = train["tasks"][0]
    assert task["task_key"] == "initialize_run" and task["max_retries"] == 0
    assert task["notebook_task"]["notebook_path"] == "../src/jobs/initialize_models.py"
    parameters = task["notebook_task"]["base_parameters"]
    assert parameters["deployed_score_handoff"] == handoff
    assert parameters["repair_count"] == "{{job.repair_count}}"
    assert parameters["execution_count"] == "{{task.execution_count}}"
    assert task["environment_key" if compute == "serverless" else "job_cluster_key"] == "skyulf"
    assert (
        jobs["score"]["tasks"][0]["notebook_task"]["notebook_path"] == "../src/jobs/score_models.py"
    )
    bundle = yaml.safe_load((project / "databricks.yml").read_text())
    assert "src/modeling/multi_model.py" in _synced_sources(project, bundle)
    from skyulf.inference.project_code import load_project_module

    factory = load_project_module((project / "src/modeling/model_set.py").read_text())
    assert factory.build_model_set()["model_name"].endswith("sm33_generated_set{resource_suffix}")
    assert factory.build_model_set()["combined_rules_path"] == "../features"
    assert factory.build_model_set()["publication"] == {"mode": "all"}
    assert factory.build_model_set()["promotion_policy"] == policy


def test_branch_factory_is_generated_from_initializer_answers():
    """The former disabled sample must be replaced by an executable generated declaration."""
    assert not (TEMPLATE / "src/modeling/branches.py").exists()
    assert (TEMPLATE / "src/modeling/multi_model.py.tmpl").is_file()


@pytest.mark.skipif(
    not os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE"), reason="CLI generation opt-in required"
)
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_generated_branches_validate_with_either_base_task(tmp_path, workflow_config, task):
    """Selected branches must clear inherited policies and validate ordinary and ensemble models."""
    from test_databricks_bundle_generation import _generate_project

    from skyulf.integrations.databricks.jobs.training.branch_notebook import (
        load_training_branch_configs,
    )
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    values, _ = _project(tmp_path, workflow_config)
    project = _generate_project(
        tmp_path,
        training_layout="multi_target",
        task=task,
        stratify="true" if task == "classification" else "false",
        cv_enabled="true",
        cv_type="stratified_k_fold" if task == "classification" else "k_fold",
        branch_count="4",
        branch_1_name="revenue",
        branch_1_target_column="revenue_target",
        branch_1_cv_enabled="false",
        branch_2_name="cost",
        branch_2_target_column="cost_target",
        branch_2_cv_enabled="false",
        branch_3_name="churn",
        branch_3_task="classification",
        branch_3_target_column="churn_target",
        branch_3_cv_enabled="false",
        branch_4_name="demand_ensemble",
        branch_4_target_column="demand_target",
        branch_4_regression_model="voting_regressor",
        branch_4_cv_enabled="false",
    )
    values["config_path"] = str(project / "config/workflow.json")
    configs = load_training_branch_configs(values)
    checked = {
        name: validate_workflow_config(config, action="train") for name, config in configs.items()
    }
    assert set(checked) == {"revenue", "cost", "churn", "demand_ensemble"}
    assert {config["target_column"] for config in checked.values()} == {
        "revenue_target",
        "cost_target",
        "churn_target",
        "demand_target",
    }
    ensemble = checked["demand_ensemble"]["pipeline"]["modeling"]["base_model"]
    assert ensemble["type"] == "voting_regressor"
    assert ensemble["params"]["base_estimators"] == ["linear_regression", "ridge"]
    assert ensemble["params"]["weights"] == [1, 1]
    assert all(
        config["cv_enabled"] is False and config["stratify"] is False for config in checked.values()
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("model_name", "workspace.models.revenue", "resource_suffix"),
        ("model_name", "workspace.other.revenue_dev", "metadata_schema"),
        ("features_path", "../../../outside", "inside"),
    ],
)
def test_branch_loader_rejects_escaping_bindings(tmp_path, workflow_config, field, value, match):
    """Overlays must not escape deployment model ownership or the synced project."""
    from skyulf.integrations.databricks.jobs.training.branch_notebook import (
        load_training_branch_configs,
    )

    values, entries = _project(tmp_path, workflow_config)
    selected = entries["revenue"] if field == "features_path" else entries["revenue"]["workflow"]
    selected[field] = value
    path = tmp_path / "src/modeling/branches.py"
    path.write_text(f"def build_training_branches():\n    return {entries!r}\n")
    with pytest.raises(ValueError, match=match):
        load_training_branch_configs(values)


@pytest.mark.parametrize("explanations", [False, True])
def test_branch_notebook_passes_resolved_configs_to_service(
    tmp_path, workflow_config, monkeypatch, explanations
):
    """The real notebook adapter must prepare once, pass all components and display escaped output."""
    from skyulf.integrations.databricks.jobs.shared import job_runtime
    from skyulf.integrations.databricks.jobs.training.branch_notebook import (
        run_branch_training_notebook,
    )
    from skyulf.integrations.databricks.training import local_branches

    values, entries = _project(tmp_path, workflow_config)
    if explanations:
        entries["cost"]["workflow"]["pipeline"]["explainability"] = {"method": "shap"}
        (tmp_path / "src/modeling/branches.py").write_text(
            f"def build_training_branches():\n    return {entries!r}\n"
        )
    show_explanations = Mock()
    monkeypatch.setattr(job_runtime, "_display_training_explanations", show_explanations)
    prepared = ("prepared_revenue", "prepared_cost", "prepared_churn")
    path = Path(values["config_path"])
    path.write_text(
        json.dumps({**json.loads(path.read_text()), "tracking_uri": None, "registry_uri": None})
    )
    prepare = Mock(return_value=prepared)

    @dataclass
    class Result:
        """Expose the service's JSON-compatible parent and component output contract."""

        parent_run_id: str
        source_table: str
        source_version: int
        components: dict

    train = Mock(
        return_value=Result("parent", "workspace.inputs.source", 4, {"<cost>": {"version": "7"}})
    )
    monkeypatch.setattr(local_branches, "prepare_training_branches", prepare)
    monkeypatch.setattr(local_branches, "train_local_branches", train)
    display = Mock()
    dbutils = SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: values), notebook=Mock())
    output = run_branch_training_notebook(
        "spark", dbutils, display_html=display, exit_notebook=False
    )
    assert prepare.call_count == train.call_count == 1
    configs = prepare.call_args.args[1]
    assert set(configs) == {"revenue", "cost", "churn"}
    assert configs["churn"]["model_name"] == "workspace.models.churn_dev"
    assert train.call_args.args == ("spark", prepared)
    assert train.call_args.kwargs["experiment_name"] == "/test/branches"
    assert train.call_args.kwargs["tracking_uri"] == "databricks"
    assert train.call_args.kwargs["registry_uri"] == "databricks-uc"
    assert json.loads(output)["components"] == {"<cost>": {"version": "7"}}
    assert "&lt;cost&gt;" in display.call_args.args[0]
    if explanations:
        assert show_explanations.call_args.args[1] == "databricks"
    else:
        show_explanations.assert_not_called()
    dbutils.notebook.exit.assert_not_called()
