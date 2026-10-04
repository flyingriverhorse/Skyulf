"""Generated repositories independently bind to the shared monitoring inventory."""

import os

import pytest
import yaml


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
        else:
            assert target["variables"].get("monitoring_enabled", "true") == "true"
    assert variables["monitoring_enabled"]["default"] == "true"
    assert variables["monitoring_catalog"]["default"] == ""
    assert variables["monitoring_schema"]["default"] == ""
    assert variables["monitoring_drift_thresholds"]["default"] == "{}"
    assert variables["on_drift"]["default"] == "disabled"
    job = yaml.safe_load((project / "resources/score.job.yml").read_text())["resources"]["jobs"][
        "score"
    ]
    tasks = [task for task in job["tasks"] if "notebook_task" in task]
    for task in tasks:
        params = task["notebook_task"]["base_parameters"]
        if task["task_key"] == "retraining_skipped":
            continue
        if task["task_key"] == "drift_report":
            assert params["monitoring_dashboard_url"] == "${var.monitoring_dashboard_url}"
            continue
        assert params["catalog"] == "${var.catalog}"
        assert params["monitoring_catalog"] == "${var.monitoring_catalog}"
        assert params["monitoring_schema"] == "${var.monitoring_schema}"
        assert params["monitoring_environment"] == "${bundle.target}"
        assert params["monitoring_deployment_mode"] == "${bundle.mode}"
        assert params["monitoring_project"] == "${bundle.name}"
        assert params["monitoring_drift_thresholds"] == "${var.monitoring_drift_thresholds}"
    assert len(tasks) == (8 if recovery == "true" else 6)
    retrain = next(task for task in tasks if task["task_key"] == "retrain_on_drift")
    assert retrain["depends_on"] == [{"task_key": "retraining_needed", "outcome": "true"}]
    check = next(task for task in tasks if task["task_key"] == "check_retraining")
    assert check["depends_on"] == [{"task_key": "drift_report"}]
    condition = next(task for task in job["tasks"] if task["task_key"] == "retraining_needed")
    assert condition["depends_on"] == [{"task_key": "check_retraining"}]
    assert condition["condition_task"] == {
        "op": "EQUAL_TO",
        "left": "{{tasks.check_retraining.values.retraining_needed}}",
        "right": "true",
    }
    skipped = next(task for task in tasks if task["task_key"] == "retraining_skipped")
    assert skipped["depends_on"] == [{"task_key": "retraining_needed", "outcome": "false"}]
    for name in ("check_retraining", "retraining_skipped"):
        assert (project / "src/jobs" / (name + ".py")).is_file()
    params = retrain["notebook_task"]["base_parameters"]
    assert params["train_job_name"] == "${bundle.name}_train"
    assert params["score_job_name"] == "{{job.name}}"
    assert params["on_drift"] == "${var.on_drift}"
    monitor = next(task for task in tasks if task["task_key"] == "monitor_model")
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
    assert monitor["notebook_task"]["notebook_path"] == "../src/jobs/monitor_model.py"
    assert (
        monitor["notebook_task"]["base_parameters"]["monitoring_dashboard_url"]
        == "${var.monitoring_dashboard_url}"
    )
    drift = next(task for task in tasks if task["task_key"] == "drift_report")
    assert drift["depends_on"] == [{"task_key": "monitor_model"}]
    assert drift["notebook_task"]["notebook_path"] == "../src/jobs/drift_report.py"
    dependencies = (
        job["environments"][0]["spec"]["dependencies"]
        if compute == "serverless"
        else [item.get("pypi", {}).get("package") for item in monitor["libraries"]]
    )
    assert "imbalanced-learn>=0.14.1,<1.0.0" in dependencies
    assert "sklearn-compat==0.1.5" in dependencies
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
    assert registration["base_parameters"]["reference_json"] == (
        "{{tasks." + reference_task + ".values.reference_json}}"
    )
    notebook = "register_set_monitor.py" if layout == "multi_target" else "register_monitor.py"
    assert registration["notebook_path"] == "../src/jobs/" + notebook
    assert registration["base_parameters"]["monitoring_drift_thresholds"] == (
        "${var.monitoring_drift_thresholds}"
    )
