"""Real CLI generation must produce directly executable YAML projects without migration."""

import os
import runpy
import shutil
from pathlib import Path

import pytest
from test_databricks_bundle_generation import _generate_project

pytestmark = pytest.mark.skipif(
    os.environ.get("SKYULF_BUNDLE_OFFLINE_CLI") != "1" or not shutil.which("databricks"),
    reason="Opt into local CLI rendering with SKYULF_BUNDLE_OFFLINE_CLI=1.",
)


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("compute", ["serverless", "policy_cluster"])
def test_new_bundles_are_yaml_only_without_migration(tmp_path, layout, compute):
    """Every generated layout must expose one editable YAML owner and working runtime loaders."""
    from skyulf.integrations.databricks.jobs.training.branch_notebook import (
        load_training_branch_configs,
    )
    from skyulf.integrations.databricks.projects.project import load_project_workflow
    from skyulf.integrations.databricks.projects.project_checks import check_project
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    project = _generate_project(tmp_path, training_layout=layout, compute_mode=compute)
    assert {path.name for path in (project / "config").iterdir()} == {
        "features.yml",
        "training.yml",
        "inference.yml",
    }
    assert not (project / "src/modeling").exists()
    assert not (project / "src/tools/migrate_config.py").exists()
    path = project / "config/training.yml"
    config = read_workflow_config(path)
    bindings = {
        "catalog": "workspace",
        "input_schema": "default",
        "output_schema": "default",
        "metadata_schema": "default",
        "resource_suffix": "",
    }
    assert check_project(project, bindings)["status"] == "passed"
    if layout == "multi_target":
        branches = load_training_branch_configs(
            {
                "config_path": str(path),
                "workflow_contract": "3",
                "deployed_score_handoff": config["score_handoff"],
                **bindings,
            }
        )
        assert len(branches) == 2
        assert all(
            branch["pipeline"]["modeling"]["type"] == "hyperparameter_tuner"
            for branch in branches.values()
        )
    else:
        resolved = load_project_workflow(config, project / "src/features")
        assert resolved["pipeline"]["modeling"]["type"] == "hyperparameter_tuner"
    refreshed = runpy.run_path(str(project / "src/tools/refresh_training_graph.py"))["refresh"](
        project
    )
    assert len(refreshed) == {"single_model": 0, "model_competition": 3, "multi_target": 2}[layout]
    assert all(
        "config/workflow.json" not in source.read_text()
        for source in (project / "resources").glob("*.yml")
    )


def test_direct_yaml_configuration_accepts_explicit_pipeline_structure(tmp_path):
    """Internal pipeline structure must be expressible without a second JSON configuration file."""
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    path = tmp_path / "training.yml"
    path.write_text(
        "version: 1\nmodels: {main: {model: {type: ridge_regression}}}\n"
        "pipeline: {preprocessing: []}\n",
        encoding="utf-8",
    )
    config = read_workflow_config(path)
    assert config["pipeline"] == {"preprocessing": [], "modeling": {}}


def test_default_yaml_matches_offline_contract_snapshot(tmp_path):
    """Keep offline default tests tied to actual Go template output, not a second renderer."""
    from skyulf.integrations.databricks.projects.yaml_config import read_yaml_mapping

    project = _generate_project(tmp_path)
    fixture = Path(__file__).parent / "fixtures/databricks_yaml_defaults/config"
    for name in ("training.yml", "inference.yml"):
        assert read_yaml_mapping(project / "config" / name) == read_yaml_mapping(fixture / name)


@pytest.mark.parametrize("name", ["on", "null", "true", "No"])
def test_yaml_generation_quotes_identifier_values(tmp_path, name):
    """Valid SQL identifiers must not turn into YAML booleans or nulls."""
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    project = _generate_project(
        tmp_path, input_columns=name, target_column="answer", record_key_columns="id"
    )
    config = read_workflow_config(project / "config/training.yml")
    assert config["input_columns"] == [name]


@pytest.mark.parametrize("inference", ["local", "spark"])
@pytest.mark.parametrize("compute", ["serverless", "policy_cluster"])
def test_yaml_generation_preserves_selected_scoring_environment(tmp_path, inference, compute):
    """Selected worker environments and optional labels must survive real template rendering."""
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    project = _generate_project(
        tmp_path,
        inference_mode=inference,
        compute_mode=compute,
        risk_category="High",
        record_key_columns="customer_id",
    )
    config = read_workflow_config(project / "config/training.yml")
    assert config["risk_category"] == "High"
    assert config["record_key_columns"] == ["customer_id"]
    if inference == "spark":
        assert config["spark_udf_env_manager"] == (
            "local" if compute == "serverless" else "virtualenv"
        )
    else:
        assert "spark_udf_env_manager" not in config
