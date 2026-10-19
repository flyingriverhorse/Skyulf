"""Shared-store defaults and the independent reference-retention schedule."""

import json

import pytest
import yaml
from test_databricks_bundle_generation import CLI_ONLY, TEMPLATE_ROOT, _generate_project
from test_databricks_deployment import (
    _resolve,
    offline_workspace,  # noqa: F401 - register the shared pytest fixture
)


def test_monitoring_schema_default_is_shared():
    """New projects should reuse a stable store without inventing a per-project schema."""
    schema = json.loads((TEMPLATE_ROOT / "databricks_template_schema.json").read_text())
    assert schema["properties"]["monitoring_schema"]["default"] == "skyulf_monitoring"


@CLI_ONLY
@pytest.mark.parametrize(
    ("settings", "catalog", "schema"),
    [
        (
            {"monitoring_catalog": "ops", "omit_fields": ("monitoring_schema",)},
            "ops",
            "skyulf_monitoring",
        ),
        ({"monitoring_catalog": "ops", "monitoring_schema": ""}, "ops", "skyulf_monitoring"),
        ({"monitoring_catalog": "ops", "monitoring_schema": "team_store"}, "ops", "team_store"),
        (
            {
                "monitoring_catalog": "ops",
                "monitoring_schema": "",
                "performance_policy_mode": "report",
                "label_table_name": "outcomes",
                "label_available_at_column": "available_at",
            },
            "ops",
            "skyulf_monitoring",
        ),
        ({"monitoring_catalog": ""}, "", ""),
        ({"monitoring_catalog": "", "monitoring_schema": "unused_store"}, "", ""),
    ],
)
def test_rendered_monitoring_destination_defaults(tmp_path, settings, catalog, schema):
    """A catalog selects the shared default while a blank catalog keeps monitoring fully off."""
    project = _generate_project(tmp_path, **settings)
    variables = yaml.safe_load((project / "deployment/variables.yml").read_text())["variables"]
    assert variables["monitoring_catalog"]["default"] == catalog
    assert variables["monitoring_schema"]["default"] == schema
    assert variables["monitoring_enabled"]["default"] == ("true" if catalog else "false")
    assert (project / "resources/monitoring_cleanup.job.yml").exists() is bool(catalog)


@CLI_ONLY
@pytest.mark.parametrize(
    "identity", ["deployer", "shared_service_principal", "separate_service_principals"]
)
def test_cleanup_job_has_independent_clock_and_existing_monitoring_identity(tmp_path, identity):
    """Retention must not run observation/retraining and must preserve the monitoring permissions."""
    project = _generate_project(
        tmp_path,
        monitoring_catalog="ops",
        deployment_identity=identity,
        manage_job_permissions="true",
        personal_development_targets="true",
        rescore_when_history_expires="true",
    )
    variables = yaml.safe_load((project / "deployment/variables.yml").read_text())["variables"]
    assert variables["monitoring_cleanup_cron"]["default"] == "0 0 4 ? * SUN"
    assert variables["monitoring_cleanup_confirm"]["default"] == "true"
    assert variables["monitoring_cleanup_pause_status"]["default"] == "UNPAUSED"
    assert variables["monitoring_reference_retention_days"]["default"] == "90"
    job = yaml.safe_load((project / "resources/monitoring_cleanup.job.yml").read_text())[
        "resources"
    ]["jobs"]["monitoring_cleanup"]
    assert job["schedule"] == {
        "quartz_cron_expression": "${var.monitoring_cleanup_cron}",
        "timezone_id": "UTC",
        "pause_status": "${var.monitoring_cleanup_pause_status}",
    }
    assert job["max_concurrent_runs"] == 1
    assert len(job["tasks"]) == 1
    task = job["tasks"][0]
    assert task["notebook_task"]["notebook_path"] == "../src/jobs/monitor_project.py"
    parameters = task["notebook_task"]["base_parameters"]
    assert parameters["job_run_id"] == "{{job.run_id}}"
    assert parameters["task_key"] == "cleanup_references"
    assert parameters["workspace_host"] == "${workspace.host}"
    assert parameters["monitoring_action"] == "cleanup_references"
    assert parameters["monitoring_project"] == "${bundle.name}"
    assert parameters["monitoring_environment"] == "${bundle.target}"
    assert parameters["cleanup_confirm"] == "{{job.parameters.cleanup_confirm}}"
    assert {item["name"]: item["default"] for item in job["parameters"]} == {
        "cleanup_confirm": "${var.monitoring_cleanup_confirm}",
        "reference_retention_days": "${var.monitoring_reference_retention_days}",
    }
    for resource, task_key in (
        ("monitoring", "monitor_model"),
        ("score", "prepare_monitoring"),
        ("score", "score"),
        ("score", "recover_predictions"),
    ):
        producer = yaml.safe_load((project / f"resources/{resource}.job.yml").read_text())[
            "resources"
        ]["jobs"][resource]
        task = next(item for item in producer["tasks"] if item["task_key"] == task_key)
        owner = task["notebook_task"]["base_parameters"]
        assert owner["job_run_id"] == "{{job.run_id}}"
        assert owner["task_key"] == task_key
        assert owner["workspace_host"] == "${workspace.host}"
    targets = yaml.safe_load((project / "deployment/targets.yml").read_text())["targets"]
    for target in targets.values():
        if target["mode"] == "development":
            assert target["variables"]["monitoring_cleanup_pause_status"] == "PAUSED"
            assert target["resources"]["jobs"]["monitoring_cleanup"]["name"] == "monitoring_cleanup"
        else:
            jobs = target["resources"]["jobs"]
            assert jobs["monitoring_cleanup"] == jobs["monitoring"]


@CLI_ONLY
@pytest.mark.parametrize("target", ["test", "test_development"])
def test_cleanup_schedule_and_destination_resolve_per_target(tmp_path, target, offline_workspace):
    """Real Bundle resolution must retain custom stores, dry runs, schedules and development pauses."""
    project = _generate_project(
        tmp_path, monitoring_catalog="ops", personal_development_targets="true"
    )
    path = project / "deployment/targets.yml"
    document = yaml.safe_load(path.read_text())
    document["targets"]["test"]["variables"].update(
        {
            "monitoring_catalog": "custom_ops",
            "monitoring_schema": "existing_store",
            "monitoring_cleanup_cron": "0 30 5 ? * MON",
            "monitoring_cleanup_confirm": "false",
            "monitoring_cleanup_pause_status": "PAUSED",
            "monitoring_reference_retention_days": "120",
        }
    )
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    resolved = _resolve(project, target, offline_workspace)
    job = resolved["resources"]["jobs"]["monitoring_cleanup"]
    assert job["schedule"]["pause_status"] == "PAUSED"
    parameters = {item["name"]: item["default"] for item in job["parameters"]}
    task = job["tasks"][0]["notebook_task"]["base_parameters"]
    if target == "test":
        assert parameters == {"cleanup_confirm": "false", "reference_retention_days": "120"}
        assert task["monitoring_catalog"] == "custom_ops"
        assert task["monitoring_schema"] == "existing_store"
        assert job["schedule"]["quartz_cron_expression"] == "0 30 5 ? * MON"
    else:
        assert task["monitoring_enabled"] == "false"
        assert task["monitoring_deployment_mode"] == "development"
        assert parameters == {"cleanup_confirm": "true", "reference_retention_days": "90"}
