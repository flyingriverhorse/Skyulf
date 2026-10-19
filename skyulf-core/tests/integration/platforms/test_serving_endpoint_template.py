"""The optional serving job is independent of batch and online-feature setup."""

import json
from pathlib import Path

import pytest
import yaml
from test_databricks_bundle_generation import _generate_project

ROOT = Path(__file__).resolve().parents[3] / "templates/databricks"


def test_serving_endpoint_wizard_has_independent_logging_and_safe_defaults():
    """Choosing plain serving must not inherit online-feature payload logging."""
    fields = json.loads((ROOT / "databricks_template_schema.json").read_text())["properties"]
    assert "include_serving_endpoint" not in fields
    assert fields["prediction_delivery"]["default"] == "batch"
    assert fields["serving_endpoint_logging_mode"]["default"] == "disabled"
    assert fields["serving_follow_champion"]["default"] == "true"
    assert fields["serving_logging_mode"]["default"] == "telemetry"
    assert "quick" in json.dumps(fields["prediction_delivery"]["skip_prompt_if"])
    assert "training_layout" not in json.dumps(fields["prediction_delivery"])


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_generated_serving_endpoint_supports_each_layout(tmp_path, layout):
    """A selected endpoint has its own runtime, target bindings and concrete-version handoff."""
    project = _generate_project(tmp_path, training_layout=layout, prediction_delivery="both")
    config = yaml.safe_load((project / "config/serving.yml").read_text())["serving_endpoint"]
    assert config["enabled"] is True
    assert config["logging_mode"] == "disabled"
    assert config["follow_champion"] is True
    assert config["endpoint_name"] == "sm33-generated-{bundle_target}-endpoint{resource_suffix}"
    assert config["model_name"].endswith(
        "sm33_generated_set{resource_suffix}"
        if layout == "multi_target"
        else "sm33_generated_model{resource_suffix}"
    )
    assert config["sql_function"]["name"].endswith("sm33_generated_score{resource_suffix}")
    job = yaml.safe_load((project / "resources/serving_endpoint.job.yml").read_text())["resources"][
        "jobs"
    ]["serving_endpoint"]
    assert job["max_concurrent_runs"] == 1 and job["queue"]["enabled"] is True
    assert job["timeout_seconds"] == "${var.serving_timeout_seconds}"
    params = job["tasks"][0]["notebook_task"]["base_parameters"]
    assert params["model_version"] == "{{job.parameters.model_version}}"
    assert params["bundle_target"] == "${bundle.target}"
    assert params["resource_suffix"] == "${var.resource_suffix}"
    train = yaml.safe_load((project / "resources/train.job.yml").read_text())["resources"]["jobs"][
        "train"
    ]
    assert train["timeout_seconds"] == "${var.train_timeout_seconds}"
    tasks = {task["task_key"]: task for task in train["tasks"]}
    assert tasks["serving_allowed"]["condition_task"]["left"] == "${bundle.mode}"
    assert tasks["serving_allowed"]["condition_task"]["right"] == "production"
    assert (
        tasks["serving_requested"]["condition_task"]["left"]
        == "{{tasks.training_report.values.serving_requested}}"
    )
    dispatch = tasks["run_serving_endpoint"]["notebook_task"]
    assert dispatch["notebook_path"] == "../src/jobs/dispatch_serving.py"
    assert (
        dispatch["base_parameters"]["model_version"]
        == "{{tasks.training_report.values.serving_model_version}}"
    )
    assert dispatch["base_parameters"]["serving_job_id"] == "${resources.jobs.serving_endpoint.id}"
    assert (project / "src/jobs/serving_endpoint.py").is_file()


def test_unselected_endpoint_generates_no_optional_artifacts(tmp_path):
    """Existing projects stay free of the optional job and its settings."""
    project = _generate_project(tmp_path)
    assert not (project / "resources/serving_endpoint.job.yml").exists()
    assert not (project / "src/jobs/serving_endpoint.py").exists()
    assert not (project / "config/serving.yml").exists()
    assert "serving_requested" not in (project / "resources/train.job.yml").read_text()


def test_serving_job_passes_target_monitoring_flag(tmp_path):
    """Serving enrollment must receive the same monitoring switch as the deployed target."""
    project = _generate_project(tmp_path, prediction_delivery="both")
    job = yaml.safe_load((project / "resources/serving_endpoint.job.yml").read_text())["resources"][
        "jobs"
    ]["serving_endpoint"]
    params = job["tasks"][0]["notebook_task"]["base_parameters"]
    assert params["monitoring_enabled"] == "${var.monitoring_enabled}"


@pytest.mark.parametrize("compute_mode", ["serverless", "policy_cluster"])
def test_serving_job_respects_compute_and_cost_attribution(tmp_path, compute_mode):
    """Endpoint maintenance must run with the project's permitted compute and cost identity."""
    project = _generate_project(
        tmp_path, prediction_delivery="both", compute_mode=compute_mode, cost_tag_key="team"
    )
    job = yaml.safe_load((project / "resources/serving_endpoint.job.yml").read_text())["resources"][
        "jobs"
    ]["serving_endpoint"]
    task = job["tasks"][0]
    params = task["notebook_task"]["base_parameters"]
    variables = yaml.safe_load((project / "deployment/variables.yml").read_text())["variables"]
    assert "cost_tag_value" in variables
    assert params["cost_tag_key"] == "team"
    assert params["cost_tag_value"] == "${var.cost_tag_value}"
    if compute_mode == "serverless":
        assert task["environment_key"] == "serving"
        assert job["budget_policy_id"] == "${var.serverless_budget_policy_id}"
        assert "job_clusters" not in job
        assert job["environments"][0]["spec"]["client"] == "${var.serverless_environment_version}"
    else:
        assert "environment_key" not in task and "environments" not in job
        assert task["job_cluster_key"] == "serving"
        cluster = job["job_clusters"][0]["new_cluster"]
        assert cluster["policy_id"] == "${var.cluster_policy_id}"
        assert cluster["custom_tags"] == {"team": "${var.cost_tag_value}"}
        assert cluster["spark_version"] == "${var.spark_version}"
        assert cluster["node_type_id"] == "${var.node_type_id}"
        assert task["libraries"] == [
            {"requirements": "${workspace.file_path}/deployment/serving-requirements.txt"}
        ]


def test_manual_serving_keeps_training_timeout_and_score_handoff(tmp_path):
    """YAML-only handoff edits need an existing dispatch path without changing batch scheduling."""
    project = _generate_project(
        tmp_path, prediction_delivery="both", serving_follow_champion="false"
    )
    train = (project / "resources/train.job.yml").read_text()
    assert "timeout_seconds: ${var.train_timeout_seconds}" in train
    assert "run_batch_scoring" in train
    assert "run_serving_endpoint" in train
    assert 'deployed_serving_endpoint: "true"' in train
    assert "deployed_serving_handoff" not in train
    assert (project / "src/jobs/dispatch_serving.py").is_file()


def test_daily_rollout_hands_off_only_a_completed_promotion(tmp_path):
    """A rollout alias change also updates the independently owned stable endpoint."""
    project = _generate_project(tmp_path, prediction_delivery="both", include_daily_rollout="true")
    job = yaml.safe_load((project / "resources/daily_rollout.job.yml").read_text())["resources"][
        "jobs"
    ]["daily_rollout"]
    tasks = {task["task_key"]: task for task in job["tasks"]}
    assert tasks["serving_allowed"]["condition_task"]["left"] == "${bundle.mode}"
    assert (
        tasks["serving_requested"]["condition_task"]["left"]
        == "{{tasks.daily_rollout.values.serving_requested}}"
    )
    assert (
        tasks["run_serving_endpoint"]["notebook_task"]["base_parameters"]["model_version"]
        == "{{tasks.daily_rollout.values.serving_model_version}}"
    )


def test_serving_logging_prefix_is_separate_from_online_setup(tmp_path):
    """Two endpoints must never write into one inference payload sink by default."""
    project = _generate_project(
        tmp_path,
        prediction_delivery="both",
        include_online_serving="true",
        serving_endpoint_logging_mode="telemetry",
    )
    config = yaml.safe_load((project / "config/serving.yml").read_text())
    assert (
        config["serving_endpoint"]["logging_table_prefix"]
        == "sm33_generated_serving{resource_suffix}"
    )
    assert config["online_serving"]["endpoint"]["logging_table_prefix"] == "sm33_generated_endpoint"


def test_polars_endpoint_option_is_not_offered_or_silently_generated(tmp_path):
    """Existing pinned-package admission requires pandas; the wizard must disclose it early."""
    from test_databricks_bundle_generation import _initialize_project

    fields = json.loads((ROOT / "databricks_template_schema.json").read_text())["properties"]
    assert "polars" in json.dumps(fields["prediction_delivery"]["skip_prompt_if"])
    result, _ = _initialize_project(tmp_path, engine="polars", prediction_delivery="both")
    assert result.returncode != 0 and "requires_engine_pandas" in result.stderr


@pytest.mark.parametrize(
    "identity,run_as,caller",
    [
        ("shared_service_principal", "shared_service_principal", "shared_service_principal"),
        ("separate_service_principals", "score_service_principal", "train_service_principal"),
    ],
)
def test_serving_job_uses_target_identity_and_grants_dispatcher_run_access(
    tmp_path, identity, run_as, caller
):
    """A service-principal training job must be allowed to start the serving child."""
    project = _generate_project(tmp_path, prediction_delivery="both", deployment_identity=identity)
    targets = yaml.safe_load((project / "deployment/targets.yml").read_text())["targets"]
    variables = yaml.safe_load((project / "deployment/variables.yml").read_text())["variables"]
    for name in ("test", "syst", "prod"):
        job = targets[name]["resources"]["jobs"]["serving_endpoint"]
        assert job["run_as"]["service_principal_name"] == "${var." + run_as + "}"
        assert job["permissions"] == "${var.serving_permissions}"
    assert {
        "level": "CAN_MANAGE_RUN",
        "service_principal_name": "${var." + caller + "}",
    } in variables["serving_permissions"]["default"]
