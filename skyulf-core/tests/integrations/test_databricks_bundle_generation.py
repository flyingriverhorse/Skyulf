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


def _initialize_project(tmp_path, **overrides):
    """Render the actual template through the installed CLI without cloud writes."""
    root = Path(__file__).resolve().parents[2] / "templates/databricks"
    schema = json.loads((root / "databricks_template_schema.json").read_text())
    values = {key: spec["default"] for key, spec in schema["properties"].items()}
    values.update(project_name="sm33_generated", **overrides)
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
    return generated, tmp_path / "output" / "sm33_generated"


def _generate_project(tmp_path, **overrides):
    """Return a successfully rendered project for its behavior checks."""
    generated, project = _initialize_project(tmp_path, **overrides)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    return project


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_rows", 0),
        ("max_bytes", -1),
        ("max_rows", "0"),
        ("max_bytes", "-1"),
        ("max_rows", "1.5"),
        ("max_bytes", "1e3"),
        ("record_key_columns_json", '[\f"id"]'),
        ("input_columns_json", '["x",\f"y"]'),
        ("record_key_columns_json", '["id"], "task": "classification"'),
    ],
)
def test_cli_rejects_invalid_limits_and_non_json_column_arrays(tmp_path, field, value):
    """Invalid init values must fail before writing malformed or unusable config."""
    generated, _ = _initialize_project(tmp_path, **{field: value})
    assert generated.returncode != 0
    assert field in generated.stderr


def _read_validated_config(project):
    """Check generated settings against the same preflight used by the notebook."""
    from skyulf.integrations.databricks.local_workflow import resolve_target_config
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    config = json.loads((project / "config/workflow.json").read_text())
    resolved = resolve_target_config(
        config,
        {
            "catalog": "test_catalog",
            "input_schema": "inputs",
            "output_schema": "outputs",
            "metadata_schema": "metadata",
            "resource_suffix": "_dev",
        },
    )
    validate_workflow_config(resolved, action="score")
    return config


@pytest.mark.parametrize(
    ("task", "model", "metric"),
    [
        ("regression", "linear_regression", "heldout_rmse"),
        ("classification", "logistic_regression", "heldout_accuracy"),
    ],
)
def test_cli_initializes_task_without_stale_training_inputs(tmp_path, task, model, metric):
    """A generated task must select compatible defaults without pinning example data."""
    project = _generate_project(tmp_path, task=task)
    config = _read_validated_config(project)
    assert config["config_version"] == 1
    assert config["task"] == task
    assert config["pipeline"]["modeling"]["type"] == model
    assert config["metric"] == metric
    assert config["training_version"] is None
    assert all(config[key] is None for key in ("start", "holdout_start", "cutoff"))
    assert config["record_key_columns"] == ["entity_id"]
    assert config["input_columns"] == ["feature_value"]
    assert config["training_table"] == "{catalog}.{input_schema}.sm33_generated_source"
    assert config["score_source_table"] == config["training_table"]


def test_cli_preserves_existing_sources_composite_keys_and_explicit_split(tmp_path):
    """Real source mappings and limits must survive initialization without invented columns."""
    project = _generate_project(
        tmp_path,
        task="classification",
        source_table_name="labeled_customers",
        score_source_table_name="new_customers",
        record_key_columns_json='[\n "customer_id",\t"observation_id"\r\n]',
        input_columns_json='[\t"income", "age"\n]',
        target_column="churn",
        event_column="observed_at",
        result_available_at_column="labeled_at",
        training_version="17",
        start="2026-06-01T00:00:00+00:00",
        holdout_start="2026-07-01T00:00:00+00:00",
        cutoff="2026-08-01T00:00:00+00:00",
        max_rows="500",
        max_bytes="1048576",
        metric="heldout_f1",
        quality_threshold="0.8",
    )
    config = _read_validated_config(project)
    assert config["training_table"] == "{catalog}.{input_schema}.labeled_customers"
    assert config["score_source_table"] == "{catalog}.{input_schema}.new_customers"
    assert config["record_key_columns"] == ["customer_id", "observation_id"]
    assert config["input_columns"] == ["income", "age"]
    assert config["target_column"] == "churn"
    assert config["event_column"] == "observed_at"
    assert config["result_available_at_column"] == "labeled_at"
    assert config["training_version"] == 17
    assert config["start"] == "2026-06-01T00:00:00+00:00"
    assert config["holdout_start"] == "2026-07-01T00:00:00+00:00"
    assert config["cutoff"] == "2026-08-01T00:00:00+00:00"
    assert config["max_rows"] == 500 and config["max_bytes"] == 1048576
    assert config["metric"] == "heldout_f1" and config["quality_threshold"] == 0.8


def test_cli_reuses_named_training_source_and_existing_single_key(tmp_path):
    """An omitted scoring table and composite key must preserve the simpler setup."""
    project = _generate_project(
        tmp_path, source_table_name="customers", record_key_columns_json='["customer_id"]'
    )
    config = _read_validated_config(project)
    assert config["record_key_columns"] == ["customer_id"]
    assert config["training_table"] == "{catalog}.{input_schema}.customers"
    assert config["score_source_table"] == config["training_table"]


@pytest.mark.parametrize("selection", ["pinned_version", "champion"])
@pytest.mark.parametrize("policy", ["manual_approval", "automatic"])
@pytest.mark.parametrize("handoff", ["disabled", "after_alias_change"])
@pytest.mark.parametrize("compute", ["serverless", "policy_cluster"])
def test_cli_emits_independent_policies_and_serialized_operator_graph(
    tmp_path, selection, policy, handoff, compute
):
    """Render real Go templates so broken dynamic references or job dependencies cannot hide."""
    monthly = handoff == "after_alias_change"
    project = _generate_project(
        tmp_path,
        score_model_selection=selection,
        promotion_policy=policy,
        score_handoff=handoff,
        compute_mode=compute,
        engine="polars" if policy == "automatic" else "pandas",
        quality_threshold="100.0",
        retraining_mode="monthly_paused" if monthly else "manual",
    )
    jobs = yaml.safe_load((project / "resources/workflow.jobs.yml").read_text())["resources"][
        "jobs"
    ]
    config = _read_validated_config(project)
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
        "job_id": "${resources.jobs.score.id}",
        "job_parameters": {"score_model_version": ""},
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
    assert jobs["score"]["parameters"] == [{"name": "score_model_version", "default": ""}]
    score_parameters = score_task["notebook_task"]["base_parameters"]
    assert score_parameters["score_model_version"] == "{{job.parameters.score_model_version}}"
    for notebook_parameters in (parameters, score_parameters):
        assert notebook_parameters["workflow_contract"] == "1"
        assert notebook_parameters["deployed_score_handoff"] == handoff
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
