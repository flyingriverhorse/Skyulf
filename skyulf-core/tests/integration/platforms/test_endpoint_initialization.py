"""Generated endpoint notebooks validate settings before starting lifecycle mutations."""

import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from test_databricks_bundle_generation import _generate_project

pytest.importorskip("mlflow")


class InitializationReached(Exception):
    """Stop at the first remote lifecycle operation after real configuration loading."""


@pytest.fixture(params=["single_model", "model_competition", "multi_target"])
def endpoint_initialization(tmp_path, monkeypatch, request):
    """Render each training graph and intercept only its downstream lifecycle mutation."""
    from skyulf.integrations.databricks.jobs.lifecycle import lifecycle_tasks
    from skyulf.integrations.databricks.jobs.training import training_node_notebook

    project = _generate_project(
        tmp_path,
        training_layout=request.param,
        prediction_delivery="endpoint",
        deployment_identity="shared_service_principal",
    )
    job = yaml.safe_load((project / "resources/train.job.yml").read_text())["resources"]["jobs"][
        "train"
    ]
    task = job["tasks"][0]
    values = dict(task["notebook_task"]["base_parameters"])
    values.update(
        job_id="10",
        job_run_id="20",
        workspace_host="https://example.cloud.databricks.com",
        repair_count="0",
        execution_count="1",
        lifecycle_action="train",
        candidate_version="",
        expected_champion_version="",
        rejection_reason="",
        promotion_receipt_json="",
        allow_unloadable_champion="",
        config_path=str(project / "config/training.yml"),
        serving_config_path=str(project / "config/serving.yml"),
        experiment_name="/test/endpoint",
        catalog="workspace",
        input_schema="inputs",
        output_schema="outputs",
        metadata_schema="models",
        resource_suffix="",
        bundle_project="sm33_generated",
        bundle_target="prod",
        bundle_mode="production",
        cost_tag_key="team",
        cost_tag_value="risk",
        monitoring_enabled="true",
        monitoring_catalog="ops",
        monitoring_schema="monitoring",
        monitoring_environment="prod",
        monitoring_deployment_mode="production",
        monitoring_project="sm33_generated",
        monitoring_drift_thresholds="{}",
        monitoring_performance_policies="{}",
        monitoring_label_table="",
        monitoring_result_available_at_column="",
        monitoring_expected_interval_hours="24",
    )
    execute = Mock(side_effect=InitializationReached)
    if request.param == "multi_target":
        monkeypatch.setattr(training_node_notebook, "initialize_branch_training", execute)
    else:
        monkeypatch.setattr(lifecycle_tasks, "run_lifecycle_phase", execute)
    dbutils = SimpleNamespace(
        widgets=SimpleNamespace(getAll=lambda: values),
        jobs=SimpleNamespace(taskValues=Mock()),
        notebook=Mock(),
    )
    notebook = project / "resources" / task["notebook_task"]["notebook_path"]
    return notebook, values, dbutils, execute


def _initialize(case):
    """Execute the actual generated entry point without contacting cloud services."""
    notebook, _, dbutils, _ = case
    runpy.run_path(
        str(notebook), run_name="__main__", init_globals={"spark": None, "dbutils": dbutils}
    )


def test_endpoint_monitoring_initializes_without_batch_output(endpoint_initialization):
    """Monitoring-enabled endpoint projects must reach initialization without prediction_table."""
    with pytest.raises(InitializationReached):
        _initialize(endpoint_initialization)
    execute = endpoint_initialization[3]
    execute.assert_called_once()
    options = execute.call_args.kwargs
    configs = options.get("configs", {"single": options.get("config")})
    assert all(config["prediction_delivery"] == "endpoint" for config in configs.values())
    assert all("prediction_table" not in config for config in configs.values())


@pytest.mark.parametrize("action", ["train", "approve"])
def test_invalid_active_serving_stops_before_initialization(endpoint_initialization, action):
    """Invalid endpoint ownership must fail before fitting or changing a champion alias."""
    _, values, dbutils, execute = endpoint_initialization
    values.update(cost_tag_value="REPLACE_WITH_COST_TAG", lifecycle_action=action)
    if action == "approve":
        values.update(
            candidate_version="2", expected_champion_version="1", comparison_sha256="a" * 64
        )
    with pytest.raises(ValueError, match="cost_tag_value"):
        _initialize(endpoint_initialization)
    execute.assert_not_called()
    dbutils.notebook.exit.assert_not_called()


@pytest.mark.parametrize("gate", ["disabled", "manual"])
def test_inactive_serving_does_not_block_training(endpoint_initialization, gate):
    """Unused serving settings must not break runs that manage the endpoint by hand."""
    _, values, _, execute = endpoint_initialization
    values["cost_tag_value"] = "REPLACE_WITH_COST_TAG"
    path = Path(values["serving_config_path"])
    document = yaml.safe_load(path.read_text())
    field = "enabled" if gate == "disabled" else "follow_champion"
    document["serving_endpoint"][field] = False
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(InitializationReached):
        _initialize(endpoint_initialization)
    execute.assert_called_once()
