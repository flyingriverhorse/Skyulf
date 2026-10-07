"""Validate optional feature jobs through real CLI generation and local target resolution."""

import os

import pytest
import yaml
from test_databricks_bundle_generation import CLI, OFFLINE_CLI, _generate_project
from test_databricks_deployment import _resolve, offline_workspace  # noqa: F401 - pytest fixture

from skyulf.integrations.databricks.features.graph import refresh_feature_graph
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
    for name in ("company", "activity"):
        (project / f"src/features/groups/{name}.py").write_text(
            "def compute(frame):\n    return frame\n", encoding="utf-8"
        )
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
    config = {
        "version": 1,
        "base_table": "workspace.test.observations",
        "output_table": "workspace.test.merged",
        "keys": ["id"],
        "timestamp": "at",
        "groups": {
            name: {
                "source_table": f"workspace.test.raw_{name}",
                "output_table": f"workspace.test.features_{name}",
                "transform": f"src/features/groups/{name}.py:compute",
                "columns": [f"value_{name}"],
            }
            for name in ("company", "activity")
        },
    }
    (project / "config/features.yml").write_text(yaml.safe_dump(config))
    assert refresh_feature_graph(project) == ["activity", "company"]
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
    assert personal["resources"]["jobs"]["features"]["name"] == "dev_alice_features"
