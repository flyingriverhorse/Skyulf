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
        ("max_input_mb", -1),
        ("max_rows", "0"),
        ("max_input_mb", "-1"),
        ("max_rows", "1.5"),
        ("max_input_mb", "1e3"),
        ("record_key_columns", "\fid"),
        ("input_columns", "x,\fy"),
        ("record_key_columns", 'id, "task": "classification"'),
    ],
)
def test_cli_rejects_invalid_limits_and_column_text(tmp_path, field, value):
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


def test_cli_plain_column_lists_preserve_order_and_empty_preprocessing(tmp_path):
    """Operators can enter comma-separated names without inserting JSON syntax."""
    project = _generate_project(
        tmp_path,
        record_key_columns="customer_id, observation_id",
        input_columns="income, age, balance",
        regression_model="random_forest_regressor",
        regression_metric="heldout_mae",
    )
    config = _read_validated_config(project)
    assert config["record_key_columns"] == ["customer_id", "observation_id"]
    assert config["input_columns"] == ["income", "age", "balance"]
    assert config["pipeline"] == {
        "preprocessing": [],
        "modeling": {"type": "random_forest_regressor", "params": {}},
    }
    assert config["metric"] == "heldout_mae"


def test_cli_named_prediction_output_is_separate_from_prediction_input(tmp_path):
    """Choosing the output name must not replace the input table or its target bindings."""
    project = _generate_project(
        tmp_path,
        score_source_table_name="customers_to_score",
        prediction_table_name="customer_predictions",
    )
    config = _read_validated_config(project)
    assert config["score_source_table"] == "{catalog}.{input_schema}.customers_to_score"
    assert (
        config["prediction_table"]
        == "{catalog}.{output_schema}.customer_predictions{resource_suffix}"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("record_key_columns", "customer_id,,event_id"),
        ("input_columns", 'income, age"'),
        ("input_columns", "income,"),
        ("regression_model", "random_forest_classifier"),
        ("classification_metric", "heldout_rmse"),
    ],
)
def test_cli_rejects_invalid_plain_columns_and_cross_task_choices(tmp_path, field, value):
    """The friendlier input must still reject malformed names and wrong-task selections."""
    generated, _ = _initialize_project(tmp_path, **{field: value})
    assert generated.returncode != 0
    assert field in generated.stderr


def test_cli_six_month_schedule_keeps_data_window_independent(tmp_path):
    """A six-month job cadence must not silently impose a six-month training window."""
    project = _generate_project(
        tmp_path,
        retraining_mode="scheduled",
        retraining_cron_expression="0 0 3 1 1,7 ?",
        retraining_timezone_id="Europe/Copenhagen",
        training_window_mode="full_snapshot",
    )
    bundle = yaml.safe_load((project / "databricks.yml").read_text())
    jobs = yaml.safe_load((project / "resources/workflow.jobs.yml").read_text())["resources"][
        "jobs"
    ]
    assert bundle["variables"]["retraining_cron_expression"]["default"] == "0 0 3 1 1,7 ?"
    assert bundle["variables"]["retraining_timezone_id"]["default"] == "Europe/Copenhagen"
    assert jobs["train"]["schedule"]["pause_status"] == "UNPAUSED"
    assert _read_validated_config(project)["training_window_mode"] == "full_snapshot"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_cli_guided_pipeline_cv_and_offline_preview(tmp_path, engine, task):
    """The real initializer must preserve selected Core steps/model/CV on both engines."""
    import sys

    import numpy as np
    import pandas as pd
    import polars as pl

    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline
    from skyulf.integrations.databricks.local_batch import fit_local_workflow

    model = "random_forest_classifier" if task == "classification" else "random_forest_regressor"
    project = _generate_project(
        tmp_path,
        engine=engine,
        task=task,
        training_version="0",
        **{f"{task}_model": model},
        cv_enabled="true",
        cv_folds="3",
        cv_type="k_fold",
        training_sample_rows="500",
        training_sample_seed="19",
    )
    config = _read_validated_config(project)
    jobs = yaml.safe_load((project / "resources/workflow.jobs.yml").read_text())["resources"][
        "jobs"
    ]
    for job in jobs.values():
        for entry in job["tasks"]:
            if "notebook_task" in entry:
                assert entry["max_retries"] == 0
                assert entry["disable_auto_optimization"] is True
    assert config["pipeline"]["modeling"] == {"type": model, "params": {}}
    assert config["pipeline"]["preprocessing"] == []
    # Operators add their own steps after initialization; test that edited path.
    steps = [
        {
            "name": "impute",
            "transformer": "SimpleImputer",
            "params": {"columns": ["feature_value"], "strategy": "mean"},
        },
        {
            "name": "scale",
            "transformer": "StandardScaler",
            "params": {"columns": ["feature_value"]},
        },
    ]
    source_path = project / "src/preprocessing.py"
    with source_path.open("a", encoding="utf-8") as stream:
        stream.write("\n\ndef build_preprocessing():\n    return " + repr(steps) + "\n")
    from skyulf.integrations.databricks.project import load_project_workflow

    config = load_project_workflow(config, source_path)
    assert config["cv_enabled"] is True and config["cv_folds"] == 3
    assert config["training_sample_rows"] == 500 and config["training_sample_seed"] == 19
    preview = subprocess.run(
        [sys.executable, str(project / "src/preview.py"), "--action", "train"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=project,
    )
    assert preview.returncode == 0, preview.stdout + preview.stderr
    assert f"Engine: {engine}" in preview.stdout
    assert model in preview.stdout
    assert "training partition only" in preview.stdout
    frame = pd.DataFrame(
        {
            "feature_value": [float(i) if i % 7 else np.nan for i in range(40)],
            "target": [i % 2 if task == "classification" else i * 2.0 for i in range(40)],
        }
    )
    data = pl.from_pandas(frame) if engine == "polars" else frame
    artifact = tmp_path / "pipeline.pkl"
    fit_local_workflow(
        config["pipeline"],
        SplitDataset(train=data, test=data[:0]),
        target_column="target",
        artifact_path=artifact,
        max_rows=100,
        max_bytes=1048576,
    )
    incoming = pd.DataFrame({"feature_value": [None, 5.0, 8.0]})
    if engine == "polars":
        incoming = pl.from_pandas(incoming)
    predictions = predict_local_pipeline(incoming, load_local_pipeline(artifact))
    assert len(predictions) == 3
    assert np.isfinite(np.asarray(predictions)).all()


def test_cli_random_window_and_non_utc_source_rules(tmp_path):
    """Random holdout and source calendar selection are independent initializer choices."""
    project = _generate_project(
        tmp_path,
        training_window_mode="fixed_window",
        split_strategy="random",
        event_column="observed_at",
        event_time_kind="text",
        event_text_kind="local_datetime",
        event_time_format="%d/%m/%Y %H:%M",
        event_time_timezone="Europe/Copenhagen",
        training_version="0",
        start="2026-06-01T00:00:00+02:00",
        cutoff="2026-09-01T00:00:00+02:00",
        filter_unavailable_results="true",
        result_available_at_column="confirmed_at",
        result_time_kind="text",
        result_text_kind="date",
        result_time_format="%Y-%m-%d",
        result_time_timezone="Europe/Vilnius",
        result_date_only="midnight",
        result_cutoff="2026-09-15T00:00:00+03:00",
    )
    config = _read_validated_config(project)
    assert config["training_window_mode"] == "fixed_window"
    assert config["monthly_lookback_months"] is None and config["window_timezone"] is None
    assert config["event_time_parsing"] == {
        "format": "%d/%m/%Y %H:%M",
        "timezone": "Europe/Copenhagen",
        "date_only": "reject",
    }
    assert config["result_time_parsing"]["date_only"] == "midnight"


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
        record_key_columns=" customer_id,\tobservation_id ",
        input_columns="\tincome, age ",
        target_column="churn",
        split_strategy="temporal",
        filter_unavailable_results="true",
        event_column="observed_at",
        result_available_at_column="labeled_at",
        training_version="17",
        start="2026-06-01T00:00:00+00:00",
        holdout_start="2026-07-01T00:00:00+00:00",
        cutoff="2026-08-01T00:00:00+00:00",
        result_cutoff="2026-09-01T00:00:00+00:00",
        max_rows="500",
        max_input_mb="1",
        classification_metric="heldout_f1",
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
    assert config["result_cutoff"] == "2026-09-01T00:00:00+00:00"
    assert config["split_strategy"] == "temporal"
    assert config["filter_unavailable_results"] is True
    assert config["max_rows"] == 500 and config["max_input_mb"] == 1
    assert config["metric"] == "heldout_f1" and config["quality_threshold"] == 0.8


def test_cli_reuses_named_training_source_and_existing_single_key(tmp_path):
    """An omitted scoring table and composite key must preserve the simpler setup."""
    project = _generate_project(
        tmp_path, source_table_name="customers", record_key_columns="customer_id"
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
        retraining_mode="scheduled" if monthly else "manual",
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
        assert jobs["train"]["schedule"]["pause_status"] == "UNPAUSED"
    else:
        assert "schedule" not in jobs["train"]
    if compute == "serverless":
        assert tasks["train"]["environment_key"] == "skyulf"
        assert "job_clusters" not in jobs["train"]
    else:
        assert tasks["train"]["job_cluster_key"] == "skyulf"
        assert "environments" not in jobs["train"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("availability", [False, True])
def test_cli_generates_date_free_training_contract(tmp_path, engine, task, availability):
    """The real initializer must support date-free training and independent delayed results."""
    from skyulf.integrations.databricks.local_workflow import resolve_target_config
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    project = _generate_project(
        tmp_path,
        engine=engine,
        task=task,
        training_version="7",
        filter_unavailable_results="true" if availability else "false",
        result_available_at_column="confirmed_at" if availability else "",
        result_cutoff="2026-09-01T00:00:00+00:00" if availability else "",
        stratify="true" if task == "classification" else "false",
    )
    config = _read_validated_config(project)
    resolved = resolve_target_config(
        config,
        {
            "catalog": "workspace",
            "input_schema": "inputs",
            "output_schema": "outputs",
            "metadata_schema": "metadata",
            "resource_suffix": "",
        },
    )
    validate_workflow_config(resolved, action="train")
    assert config["split_strategy"] == "random"
    assert config["training_window_mode"] == "full_snapshot"
    assert config["window_timezone"] is None
    assert config["cv_enabled"] is False and config["cv_folds"] == 5
    assert config["training_sample_rows"] is None
    assert config["engine"] == engine
    assert config["test_size"] == 0.2 and config["random_state"] == 42
    assert config["stratify"] == (task == "classification")
    assert config["event_column"] is None
    assert all(
        config[key] is None
        for key in ("start", "holdout_start", "cutoff", "monthly_lookback_months")
    )
    assert config["filter_unavailable_results"] == availability
    assert config["result_available_at_column"] == ("confirmed_at" if availability else None)


def test_cli_preserves_conflicting_fields_for_preflight_rejection(tmp_path):
    """Initialization must not silently discard a supplied date mapping in random mode."""
    from skyulf.integrations.databricks.local_workflow import resolve_target_config
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    project = _generate_project(tmp_path, event_column="observed_at", training_version="7")
    config = json.loads((project / "config/workflow.json").read_text())
    assert config["event_column"] == "observed_at"
    resolved = resolve_target_config(
        config,
        {
            "catalog": "workspace",
            "input_schema": "inputs",
            "output_schema": "outputs",
            "metadata_schema": "metadata",
            "resource_suffix": "",
        },
    )
    with pytest.raises(ValueError, match="event_column|[Rr]andom"):
        validate_workflow_config(resolved, action="train")


@pytest.mark.parametrize(
    "filename",
    [
        "date-free-init.example.json",
        "random-delayed-results-init.example.json",
        "temporal-delayed-results-init.example.json",
        "guided-classification-init.example.json",
        "random-window-init.example.json",
    ],
)
def test_cli_training_examples_pass_manual_preflight(tmp_path, filename):
    """Published examples must render and validate as usable manual training policies."""
    from skyulf.integrations.databricks.local_workflow import resolve_target_config
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    root = Path(__file__).resolve().parents[2] / "templates/databricks/examples"
    inputs = json.loads((root / filename).read_text())
    inputs.pop("project_name")
    config = _read_validated_config(_generate_project(tmp_path, **inputs))
    resolved = resolve_target_config(
        config,
        {
            "catalog": "workspace",
            "input_schema": "inputs",
            "output_schema": "outputs",
            "metadata_schema": "metadata",
            "resource_suffix": "",
        },
    )
    checked = validate_workflow_config(resolved, action="train")
    assert checked["training_version"] == 12
    assert checked["split_strategy"] == inputs["split_strategy"]
