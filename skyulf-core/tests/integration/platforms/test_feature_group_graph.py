"""Feature DAGs pin inputs once and isolate independently repairable domains."""

import os
import subprocess
from copy import deepcopy
from dataclasses import asdict
from typing import Any, cast

import pytest
import yaml

from skyulf.integrations.databricks.features.config import parse_feature_config
from skyulf.integrations.databricks.features.graph import build_feature_job, refresh_feature_graph


def test_refresh_creates_and_removes_only_optional_feature_files(tmp_path):
    """A ready-table project must not retain unused feature notebooks or job resources."""
    plan = _plan()
    assert plan is not None
    config = {"version": 1, **asdict(plan)}
    config["groups"] = {
        g.name: {k: v for k, v in asdict(g).items() if k != "name"} for g in plan.groups
    }
    (tmp_path / "config").mkdir()
    (tmp_path / "resources").mkdir()
    (tmp_path / "deployment").mkdir()
    (tmp_path / "deployment/targets.yml").write_text(
        yaml.safe_dump(
            {
                "targets": {
                    "prod": {
                        "resources": {
                            "jobs": {
                                "train": {
                                    "run_as": {
                                        "service_principal_name": "${var.train_service_principal}"
                                    },
                                    "permissions": "${var.train_permissions}",
                                }
                            }
                        }
                    },
                    "test_development": {
                        "mode": "development",
                        "resources": {"jobs": {"train": {"name": "train"}}},
                    },
                }
            }
        )
    )
    path = tmp_path / "config/features.yml"
    path.write_text(yaml.safe_dump(config))
    (tmp_path / "resources/train.job.yml").write_text(
        yaml.safe_dump(
            {
                "resources": {
                    "jobs": {
                        "train": {
                            "tasks": [
                                {
                                    "task_key": "initialize_run",
                                    "environment_key": "skyulf",
                                    "notebook_task": {
                                        "notebook_path": "../src/jobs/initialize_run.py"
                                    },
                                }
                            ]
                        }
                    }
                }
            }
        )
    )
    assert refresh_feature_graph(tmp_path) == ["activity", "company"]
    assert (tmp_path / "src/jobs/feature_task.py").is_file()
    assert (tmp_path / "resources/features.job.yml").is_file()
    generated = yaml.safe_load((tmp_path / "resources/features.job.yml").read_text())
    target_jobs = generated["targets"]
    assert target_jobs["prod"]["resources"]["jobs"]["features"]["run_as"] == {
        "service_principal_name": "${var.train_service_principal}"
    }
    assert (
        target_jobs["prod"]["resources"]["jobs"]["features"]["permissions"]
        == "${var.train_permissions}"
    )
    assert target_jobs["test_development"]["resources"]["jobs"]["features"]["name"] == "features"
    path.write_text("version: 1\ngroups: {}\n")
    assert refresh_feature_graph(tmp_path) == []
    assert not (tmp_path / "src/jobs/feature_task.py").exists()
    assert not (tmp_path / "resources/features.job.yml").exists()


def _plan():
    """Keep table declarations explicit, separate from job compute defaults."""
    return parse_feature_config(
        {
            "version": 1,
            "base_table": "workspace.demo.observations",
            "output_table": "workspace.demo.merged",
            "keys": ["id"],
            "timestamp": "at",
            "groups": {
                name: {
                    "source_table": f"workspace.demo.raw_{name}",
                    "output_table": f"workspace.demo.features_{name}",
                    "transform": f"src/feature_groups/{name}.py:compute",
                    "columns": [f"value_{name}"],
                }
                for name in ("company", "activity")
            },
        }
    )


@pytest.mark.parametrize("compute", ["serverless", "cluster"])
def test_parallel_tasks_preserve_compute_and_permissions(compute):
    """Feature jobs inherit deployment controls and expose each group as its own task."""
    source: dict[str, Any] = {
        "name": "example_train",
        "max_concurrent_runs": 1,
        "permissions": [{"group_name": "team", "level": "CAN_MANAGE_RUN"}],
        "tasks": [
            {
                "task_key": "initialize_run",
                "timeout_seconds": 300,
                "max_retries": 0,
                "notebook_task": {
                    "base_parameters": {
                        "catalog": "${var.catalog}",
                        "input_schema": "${var.input_schema}",
                    }
                },
            }
        ],
    }
    if compute == "serverless":
        source["environments"] = [{"environment_key": "skyulf", "spec": {"client": "1"}}]
        source["tasks"][0]["environment_key"] = "skyulf"
    else:
        source["job_clusters"] = [{"job_cluster_key": "skyulf", "new_cluster": {"num_workers": 2}}]
        prototype = cast(dict[str, Any], source["tasks"][0])
        prototype.update(job_cluster_key="skyulf", libraries=[{"whl": "../dist/*.whl"}])
    before = deepcopy(source)
    plan = _plan()
    assert plan is not None
    job = build_feature_job(plan, source)
    tasks = {task["task_key"]: task for task in job["tasks"]}
    assert tasks["feature_company"]["depends_on"] == [{"task_key": "initialize_features"}]
    assert tasks["feature_activity"]["depends_on"] == [{"task_key": "initialize_features"}]
    assert tasks["merge_features"]["depends_on"] == [
        {"task_key": "feature_company"},
        {"task_key": "feature_activity"},
    ]
    assert job["permissions"] == source["permissions"]
    assert job.get("environments") == source.get("environments")
    assert job.get("job_clusters") == source.get("job_clusters")
    assert source == before


@pytest.mark.parametrize("enabled", [False, True])
def test_refresh_rejects_resource_parent_outside_project(tmp_path, enabled):
    """An external resource directory must never be overwritten or removed by refresh."""
    root, external = tmp_path / "project", tmp_path / "external"
    (root / "config").mkdir(parents=True)
    external.mkdir()
    config = "version: 1\ngroups: {}\n"
    if enabled:
        config = """version: 1
base_table: workspace.demo.observations
output_table: workspace.demo.merged
keys: [id]
timestamp: at
groups:
  activity:
    source_table: workspace.demo.raw
    output_table: workspace.demo.features
    transform: src/feature_groups/activity.py:compute
    columns: [value]
"""
    (root / "config/features.yml").write_text(config, encoding="utf-8")
    target = external / "features.job.yml"
    original = (
        "# Generated by refresh_feature_graph.py; edit config/features.yml.\nowned: outside\n"
    )
    target.write_text(original, encoding="utf-8")
    (external / "train.job.yml").write_text(
        "resources:\n  jobs:\n    train:\n      tasks:\n        - task_key: train\n          notebook_task: {}\n",
        encoding="utf-8",
    )
    link = root / "resources"
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(external)], check=True, capture_output=True
        )
    else:
        link.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="within|outside|project"):
        refresh_feature_graph(root)
    assert target.read_text(encoding="utf-8") == original
