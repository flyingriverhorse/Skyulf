"""Validate optional feature jobs through real CLI generation and local target resolution."""

import os

import pytest
import yaml
from test_databricks_bundle_generation import CLI, OFFLINE_CLI, _generate_project
from test_databricks_deployment import _resolve, offline_workspace  # noqa: F401 - pytest fixture

from skyulf.integrations.databricks.features.config import parse_feature_config
from skyulf.integrations.databricks.features.graph import (
    feature_targets_snippet,
    missing_feature_controls,
    refresh_feature_graph,
)
from skyulf.integrations.databricks.features.runtime import transform_source
from skyulf.integrations.databricks.projects.project_checks import check_project

pytestmark = pytest.mark.skipif(
    not CLI or not (os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE") or OFFLINE_CLI),
    reason="CLI opt-in",
)


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_feature_job_resolves_for_each_layout_and_identity(tmp_path, layout, offline_workspace):
    """An optional data producer must inherit real target identities without adding model tasks."""
    project = _generate_project(
        tmp_path,
        training_layout=layout,
        compute_mode="serverless",
        deployment_identity="separate_service_principals",
        manage_job_permissions="true",
        personal_development_targets="true",
    )
    assert not (project / "src/jobs/feature_task.py").exists()
    assert (project / "config/training.yml").is_file()
    text = (project / "config/features.yml").read_text(encoding="utf-8")
    assert parse_feature_config(yaml.safe_load(text)) is None
    _, marker, example = text.partition("# version: 1\n")
    assert marker, "The generated configuration must include an activatable example."
    config = yaml.safe_load("version: 1\n" + "\n".join(line[2:] for line in example.splitlines()))
    plan = parse_feature_config(config)
    assert plan is not None
    for group in plan.groups:
        source, digest = transform_source(project, group)
        compile(source, group.transform.split(":")[0], "exec")
        assert len(digest) == 64
    smoke = check_project(
        project,
        {
            "catalog": "workspace",
            "input_schema": "test",
            "output_schema": "test",
            "metadata_schema": "test",
            "resource_suffix": "",
        },
    )
    assert smoke["status"] == "passed"
    (project / "config/features.yml").write_text(yaml.safe_dump(config))
    assert refresh_feature_graph(project) == ["activity", "company"]
    bindings = {
        "catalog": "workspace",
        "input_schema": "test",
        "output_schema": "test",
        "metadata_schema": "test",
        "resource_suffix": "",
    }
    missing = missing_feature_controls(project)
    assert set(missing) == {"test", "syst", "prod"}  # personal targets keep no run_as/ACL
    with pytest.raises(ValueError, match="deployment/targets.yml"):
        check_project(project, bindings)
    targets_path = project / "deployment/targets.yml"
    document = yaml.safe_load(targets_path.read_text())
    for name, target in yaml.safe_load(feature_targets_snippet(missing))["targets"].items():
        document["targets"][name]["resources"]["jobs"].update(target["resources"]["jobs"])
    targets_path.write_text(yaml.safe_dump(document, sort_keys=False))
    assert check_project(project, bindings)["status"] == "passed"
    shared = _resolve(project, "test", offline_workspace)
    personal = _resolve(project, "test_development", offline_workspace)
    for resolved in (shared, personal):
        jobs = resolved["resources"]["jobs"]
        features = jobs["features"]
        assert features.get("run_as") == jobs["train"].get("run_as")
        assert features.get("permissions") == jobs["train"].get("permissions")
        tasks = {task["task_key"]: task for task in features["tasks"]}
        assert set(tasks) == {
            "initialize_features",
            "feature_activity",
            "feature_company",
            "merge_features",
        }
        for name in ("activity", "company"):
            assert tasks[f"feature_{name}"]["depends_on"] == [{"task_key": "initialize_features"}]
        assert {item["task_key"] for item in tasks["merge_features"]["depends_on"]} == {
            "feature_activity",
            "feature_company",
        }
        assert (project / "src/jobs/feature_task.py").is_file()
    assert (
        personal["resources"]["jobs"]["features"]["name"] == "[dev alice] sm33_generated_features"
    )
