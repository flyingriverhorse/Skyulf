"""Endpoint projects train and monitor without inventing a batch prediction destination."""

import pytest
import yaml
from test_databricks_bundle_generation import _generate_project, _read_validated_config

from skyulf.integrations.databricks.lifecycle.workflow import resolve_target_config, run_action
from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config


@pytest.fixture
def endpoint_config(workflow_config):
    """Use the existing training contract while removing the unused batch destination."""
    config = {**workflow_config, "engine": "pandas", "prediction_delivery": "endpoint"}
    config.pop("prediction_table")
    config["score_handoff"] = "disabled"
    return config


@pytest.mark.parametrize("action", ["train", "approve", "reject", "rollback"])
def test_endpoint_only_validates_without_prediction_table(endpoint_config, action):
    """Training and operator actions need no Delta prediction sink for REST/SQL serving."""
    assert validate_workflow_config(endpoint_config, action=action) == endpoint_config


def test_endpoint_only_rejects_batch_execution_before_side_effects(endpoint_config):
    """A caller cannot bypass the generated graph and accidentally run batch scoring."""
    with pytest.raises(ValueError, match="endpoint.*batch scoring"):
        run_action(None, endpoint_config, "score")


def test_both_delivery_rejects_polars_outside_the_wizard(workflow_config):
    """Editing YAML cannot bypass endpoint admission enforced during template generation."""
    config = {**workflow_config, "engine": "polars", "prediction_delivery": "both"}
    with pytest.raises(ValueError, match="Endpoint delivery requires engine=pandas"):
        validate_workflow_config(config, action="train")


@pytest.mark.parametrize("value", [None, True, [], {}, "auto"])
def test_invalid_delivery_is_rejected_explicitly(endpoint_config, value):
    """Malformed delivery settings cannot leak low-level membership/type errors."""
    with pytest.raises(ValueError, match="prediction_delivery"):
        validate_workflow_config({**endpoint_config, "prediction_delivery": value}, action="train")


@pytest.mark.parametrize(
    "field,value",
    [
        ("prediction_table", None),
        ("prediction_table", "main.ml.outputs"),
        ("score_handoff", "after_alias_change"),
        ("engine", "polars"),
    ],
)
def test_endpoint_only_refuses_conflicting_batch_or_engine_settings(endpoint_config, field, value):
    """Endpoint-only configuration fails before execution instead of silently ignoring batch settings."""
    with pytest.raises(ValueError, match="Endpoint"):
        validate_workflow_config({**endpoint_config, field: value}, action="train")


def test_endpoint_only_resolves_real_sources_without_fake_output(endpoint_config):
    """Target binding preserves genuine training/source/model identities without a dummy table."""
    bindings = {
        "catalog": "workspace",
        "input_schema": "test",
        "output_schema": "test",
        "metadata_schema": "test",
        "resource_suffix": "",
    }
    for key in ("training_table", "score_source_table", "model_name"):
        endpoint_config[key] = "{catalog}.{metadata_schema}." + key
    resolved = resolve_target_config(endpoint_config, bindings)
    assert resolved["model_name"] == "workspace.test.model_name"
    assert "prediction_table" not in resolved


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("delivery", ["batch", "endpoint", "both"])
def test_generated_delivery_has_only_selected_jobs(tmp_path, layout, delivery):
    """All layouts keep training and monitoring without dangling job or notebook references."""
    project = _generate_project(
        tmp_path,
        training_layout=layout,
        prediction_delivery=delivery,
        deployment_identity="shared_service_principal",
    )
    batch = delivery != "endpoint"
    endpoint = delivery != "batch"
    assert (project / "resources/score.job.yml").exists() == batch
    assert (project / "resources/serving_endpoint.job.yml").exists() == endpoint
    assert (project / "resources/monitoring.job.yml").is_file()
    training_guide = (project / "TRAINING.md").read_text(encoding="utf-8")
    assert "promotion_policy=" in training_guide
    for setting in (
        "score_model_selection=",
        "follow_champion:",
        "rescore_when_history_expires:",
    ):
        assert (setting in training_guide) == batch
    train = (project / "resources/train.job.yml").read_text()
    assert ("run_batch_scoring" in train) == batch
    assert ("run_serving_endpoint" in train) == endpoint
    targets = yaml.safe_load((project / "deployment/targets.yml").read_text())["targets"]
    for target in targets.values():
        assert ("score" in target.get("resources", {}).get("jobs", {})) == batch
    if not batch:
        for name in (
            "score.py",
            "score_models.py",
            "recover_predictions.py",
            "scoring_report.py",
            "register_monitor.py",
            "register_set_monitor.py",
        ):
            assert not (project / "src/jobs" / name).exists()
        inference = yaml.safe_load((project / "config/inference.yml").read_text())
        assert inference["prediction_delivery"] == "endpoint"
        assert "score_source_table" in inference
        assert "prediction_table" not in inference and "score_handoff" not in inference
        if "model_set" in inference:
            assert "prediction_table" not in inference["model_set"]
        assert not (project / "deployment/score-requirements.txt").exists()
    config = _read_validated_config(project, action="train")
    assert config.get("prediction_delivery", "batch") == delivery
    from skyulf.integrations.databricks.projects.project_checks import check_project

    bindings = {
        "catalog": "test_catalog",
        "input_schema": "inputs",
        "output_schema": "outputs",
        "metadata_schema": "metadata",
        "resource_suffix": "_dev",
    }
    assert check_project(project, bindings)["status"] == "passed"
    if not batch:
        from skyulf.integrations.databricks.jobs.shared.job_runtime import read_notebook_config
        from skyulf.integrations.databricks.model_sets.model_set_project import (
            load_project_model_set,
        )
        from skyulf.integrations.databricks.projects.project import load_project_workflow
        from skyulf.integrations.databricks.projects.workflow_config import preview_workflow_config

        values = {
            **bindings,
            "config_path": str(project / "config/training.yml"),
            "workflow_contract": "3",
            "deployed_score_handoff": "disabled",
        }
        resolved = read_notebook_config(values)
        loaded = load_project_workflow(resolved, project / "src/features")
        preview = preview_workflow_config(loaded, action="train")
        assert "endpoint only" in preview and "Prediction output:" not in preview
        if layout == "multi_target":
            settings = load_project_model_set(values, resolved)
            assert settings is not None
            assert settings["model_name"].endswith("_set_dev")
            assert "prediction_table" not in settings
