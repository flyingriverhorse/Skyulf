"""Opt-in real CLI generation checks for the deployable two-job operator graph."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

PROFILE = os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE")
CLI = shutil.which("databricks")
pytestmark = pytest.mark.skipif(
    not PROFILE or not CLI,
    reason="Set SKYULF_BUNDLE_CLI_TEST_PROFILE to explicitly opt into installed CLI generation.",
)


@pytest.mark.parametrize("selection", ["pinned_version", "champion"])
@pytest.mark.parametrize("policy", ["manual_approval", "automatic"])
@pytest.mark.parametrize("handoff", ["disabled", "after_alias_change"])
@pytest.mark.parametrize("compute", ["serverless", "policy_cluster"])
def test_cli_emits_independent_policies_and_serialized_operator_graph(
    tmp_path, selection, policy, handoff, compute
):
    """Render real Go templates so broken dynamic references or job dependencies cannot hide."""
    root = Path(__file__).resolve().parents[2] / "templates/databricks"
    schema = json.loads((root / "databricks_template_schema.json").read_text())
    values = {key: spec["default"] for key, spec in schema["properties"].items()}
    monthly = handoff == "after_alias_change"
    values.update(
        project_name="sm32_generated",
        score_model_selection=selection,
        promotion_policy=policy,
        score_handoff=handoff,
        compute_mode=compute,
        engine="polars" if policy == "automatic" else "pandas",
        quality_threshold="100.0",
        retraining_mode="monthly_paused" if monthly else "manual",
    )
    inputs = tmp_path / "init.json"
    inputs.write_text(json.dumps(values), encoding="utf-8")
    generated = subprocess.run(
        [
            str(CLI),
            "bundle",
            "init",
            str(root),
            "--config-file",
            str(inputs),
            "--output-dir",
            str(tmp_path / "output"),
            "--profile",
            str(PROFILE),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert generated.returncode == 0, generated.stdout + generated.stderr
    project = tmp_path / "output" / "sm32_generated"
    jobs = yaml.safe_load((project / "resources/workflow.jobs.yml").read_text())["resources"][
        "jobs"
    ]
    config = json.loads((project / "config/workflow.json").read_text())
    assert set(jobs) == {"train", "score"}
    for job in jobs.values():
        assert job["max_concurrent_runs"] == 1 and job["queue"]["enabled"] is True
    tasks = {task["task_key"]: task for task in jobs["train"]["tasks"]}
    assert set(tasks) == {"train", "should_score", "score_after_lifecycle"}
    assert tasks["should_score"]["depends_on"] == [{"task_key": "train"}]
    assert tasks["should_score"]["condition_task"] == {
        "op": "EQUAL_TO",
        "left": "{{tasks.train.values.score_requested}}",
        "right": "true",
    }
    assert tasks["score_after_lifecycle"]["depends_on"] == [
        {"task_key": "should_score", "outcome": "true"}
    ]
    assert tasks["score_after_lifecycle"]["run_job_task"] == {
        "job_id": "${resources.jobs.score.id}"
    }
    defaults = {item["name"]: item["default"] for item in jobs["train"]["parameters"]}
    assert defaults["lifecycle_action"] == ("train_monthly" if monthly else "train")
    assert set(defaults) == {
        "lifecycle_action",
        "candidate_version",
        "expected_champion_version",
        "rejection_reason",
        "promotion_receipt_json",
    }
    parameters = tasks["train"]["notebook_task"]["base_parameters"]
    for name in defaults:
        assert parameters[name] == "{{job.parameters." + name + "}}"
    score_task = jobs["score"]["tasks"][0]
    assert score_task["notebook_task"]["notebook_path"] == "../src/score.py"
    assert "lifecycle_action" not in score_task["notebook_task"]["base_parameters"]
    assert "parameters" not in jobs["score"]
    assert (project / "src/score.py").is_file()
    assert config["score_model_selection"] == selection
    assert config["promotion_policy"] == policy
    assert config["score_handoff"] == handoff
    assert config["quality_threshold"] == 100.0  # Manual gates must not be erased.
    assert "model_selection_mode" not in config
    if monthly:
        assert jobs["train"]["schedule"]["pause_status"] == "PAUSED"
    else:
        assert "schedule" not in jobs["train"]
    if compute == "serverless":
        assert tasks["train"]["environment_key"] == "skyulf"
        assert "job_clusters" not in jobs["train"]
    else:
        assert tasks["train"]["job_cluster_key"] == "skyulf"
        assert "environments" not in jobs["train"]
