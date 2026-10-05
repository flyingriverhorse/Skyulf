"""Generated repositories independently bind to the shared monitoring inventory."""

import os
from pathlib import Path

import pytest
import yaml


def test_separate_spark_monitoring_template_exists():
    """New projects need independent scheduled monitoring instead of inline score compute."""
    root = Path(__file__).resolve().parents[3] / "templates/databricks/template/{{.project_name}}"
    job = (root / "resources/monitoring.job.yml.tmpl").read_text(encoding="utf-8")
    assert "monitoring_task_timeout_seconds" in job
    assert "monitoring_cron" in job
    assert "monitoring_request" in job
    assert "../src/jobs/monitor_project.py" in job
    assert "../src/jobs/retrain_on_drift.py" in job


@pytest.mark.skipif(not os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE"), reason="CLI opt-in")
@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("recovery", ["false", "true"])
@pytest.mark.parametrize("compute", ["serverless", "policy_cluster"])
def test_generated_projects_bind_independent_monitoring_destination(
    tmp_path, layout, recovery, compute
):
    """Every scoring branch receives the central destination independently of model namespaces."""
    from test_databricks_bundle_generation import _generate_project

    project = _generate_project(
        tmp_path, training_layout=layout, auto_rebuild_on_cdf_expiry=recovery, compute_mode=compute
    )
    variables = yaml.safe_load((project / "deployment/variables.yml").read_text())["variables"]
    targets = yaml.safe_load((project / "deployment/targets.yml").read_text())["targets"]
    for target in targets.values():
        if target["mode"] == "development":
            assert target["variables"]["monitoring_enabled"] == "false"
            assert target["variables"]["on_drift"] == "disabled"
            assert target["variables"]["monitoring_performance_policies"] == "{}"
        else:
            assert target["variables"].get("monitoring_enabled", "true") == "true"
    assert variables["monitoring_enabled"]["default"] == "true"
    assert "monitoring_execution_engine" not in variables
    assert variables["monitoring_catalog"]["default"] == ""
    assert variables["monitoring_schema"]["default"] == ""
    assert variables["monitoring_drift_thresholds"]["default"] == "{}"
    assert variables["monitoring_performance_policies"]["default"] == "{}"
    assert variables["on_drift"]["default"] == "disabled"
    monitoring_job = yaml.safe_load((project / "resources/monitoring.job.yml").read_text())[
        "resources"
    ]["jobs"]["monitoring"]
    monitoring_task = next(
        task for task in monitoring_job["tasks"] if task["task_key"] == "monitor_model"
    )
    monitoring_params = monitoring_task["notebook_task"]["base_parameters"]
    assert monitoring_params["as_of_unix_ms"] == "{{job.start_time.timestamp_ms}}"
    assert "as_of" not in monitoring_params
    job = yaml.safe_load((project / "resources/score.job.yml").read_text())["resources"]["jobs"][
        "score"
    ]
    tasks = [task for task in job["tasks"] if "notebook_task" in task]
    for task in tasks:
        params = task["notebook_task"]["base_parameters"]
        assert params["monitoring_execution_engine"] == "spark"
        assert params["catalog"] == "${var.catalog}"
        assert params["monitoring_catalog"] == "${var.monitoring_catalog}"
        assert params["monitoring_schema"] == "${var.monitoring_schema}"
        assert params["monitoring_environment"] == "${bundle.target}"
        assert params["monitoring_deployment_mode"] == "${bundle.mode}"
        assert params["monitoring_project"] == "${bundle.name}"
        assert params["monitoring_drift_thresholds"] == "${var.monitoring_drift_thresholds}"
        assert params["monitoring_performance_policies"] == "${var.monitoring_performance_policies}"
    assert len(tasks) == (4 if recovery == "true" else 2)
    assert not {
        "drift_report",
        "check_retraining",
        "retraining_needed",
        "retrain_on_drift",
        "retraining_skipped",
    }.intersection(task["task_key"] for task in job["tasks"])
    monitor = next(task for task in tasks if task["task_key"] == "prepare_monitoring")
    assert monitor["depends_on"] == [{"task_key": "monitoring_allowed", "outcome": "true"}]
    allowed = next(task for task in job["tasks"] if task["task_key"] == "monitoring_allowed")
    assert allowed["depends_on"] == [
        {"task_key": "scoring_report" if recovery == "true" else "score"}
    ]
    assert allowed["condition_task"] == {
        "op": "EQUAL_TO",
        "left": "${bundle.mode}",
        "right": "production",
    }
    assert monitor["notebook_task"]["notebook_path"] == "../src/jobs/prepare_monitoring.py"
    assert "monitoring_invocation_id" not in monitor["notebook_task"]["base_parameters"]
    assert (
        monitor["notebook_task"]["base_parameters"]["monitoring_dashboard_url"]
        == "${var.monitoring_dashboard_url}"
    )
    ready = next(task for task in job["tasks"] if task["task_key"] == "monitoring_ready")
    assert ready["depends_on"] == [{"task_key": "prepare_monitoring"}]
    assert ready["condition_task"] == {
        "op": "EQUAL_TO",
        "left": "{{tasks.prepare_monitoring.values.monitoring_ready}}",
        "right": "true",
    }
    dispatch = next(task for task in job["tasks"] if task["task_key"] == "monitor_model")
    assert dispatch["depends_on"] == [{"task_key": "monitoring_ready", "outcome": "true"}]
    assert dispatch["run_job_task"] == {
        "job_id": "${resources.jobs.monitoring.id}",
        "job_parameters": {
            "monitoring_request": "{{tasks.prepare_monitoring.values.monitoring_request_json}}",
            "score_model_version": "{{job.parameters.score_model_version}}",
        },
    }
    assert monitoring_params["monitoring_execution_engine"] == "spark"
    dependencies = (
        job["environments"][0]["spec"]["dependencies"]
        if compute == "serverless"
        else [item.get("requirements") for item in monitor["libraries"]]
    )
    requirements_reference = "${workspace.file_path}/deployment/requirements.txt"
    assert (
        "-r " + requirements_reference if compute == "serverless" else requirements_reference
    ) in dependencies
    requirements = (project / "deployment/requirements.txt").read_text().splitlines()
    assert "imbalanced-learn==0.14.1" in requirements
    assert "sklearn-compat==0.1.5" in requirements
    training = yaml.safe_load((project / "resources/train.job.yml").read_text())["resources"][
        "jobs"
    ]["train"]
    training_tasks = {task["task_key"]: task for task in training["tasks"]}
    assert training_tasks["monitoring_allowed"]["depends_on"] == [{"task_key": "training_report"}]
    assert training_tasks["monitoring_allowed"]["condition_task"] == allowed["condition_task"]
    assert training_tasks["register_monitor"]["depends_on"] == [
        {"task_key": "monitoring_allowed", "outcome": "true"}
    ]
    assert training_tasks["scoring_requested"]["run_if"] == "NONE_FAILED"
    assert training_tasks["scoring_requested"]["depends_on"] == [
        {"task_key": "monitoring_allowed", "outcome": "false"},
        {"task_key": "register_monitor"},
    ]
    reference_task = "initialize_run" if layout == "multi_target" else "training_report"
    registration = training_tasks["register_monitor"]["notebook_task"]
    assert registration["base_parameters"]["monitoring_execution_engine"] == "spark"
    assert registration["base_parameters"]["reference_json"] == (
        "{{tasks." + reference_task + ".values.reference_json}}"
    )
    notebook = "register_set_monitor.py" if layout == "multi_target" else "register_monitor.py"
    assert registration["notebook_path"] == "../src/jobs/" + notebook
    assert registration["base_parameters"]["monitoring_drift_thresholds"] == (
        "${var.monitoring_drift_thresholds}"
    )
    assert registration["base_parameters"]["monitoring_performance_policies"] == (
        "${var.monitoring_performance_policies}"
    )


def test_source_templates_propagate_performance_mapping_to_every_monitoring_task():
    """Every producer and retraining path must receive the same mapping as drift thresholds."""
    project = Path(__file__).parents[3] / "templates/databricks/template/{{.project_name}}"
    for name in ("score.job.yml.tmpl", "train.job.yml.tmpl"):
        text = (project / "resources" / name).read_text()
        assert text.count(
            "monitoring_performance_policies: ${var.monitoring_performance_policies}"
        ) == (text.count("monitoring_drift_thresholds: ${var.monitoring_drift_thresholds}"))
    targets = (project / "deployment/targets.yml.tmpl").read_text()
    assert targets.count("monitoring_performance_policies: '{}'") == 3
    variables = (project / "deployment/variables.yml.tmpl").read_text()
    assert "  monitoring_performance_policies:" in variables
    assert "    default: '{}'" in variables
