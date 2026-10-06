"""Reject invalid project settings before registry, training or table side effects."""

from copy import deepcopy

import pytest


@pytest.mark.parametrize("action", ["train", "score", "approve"])
def test_legacy_workflow_defaults_to_local_inference(workflow_config, action):
    """Older Polars projects keep their local scoring route after the new choice."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    assert validate_workflow_config(workflow_config, action=action) == workflow_config


def test_migration_records_local_inference_explicitly(workflow_config):
    """A deliberate migration must preserve the older project's local route."""
    from skyulf.integrations.databricks.projects.workflow_config import migrate_workflow_config

    migrated = migrate_workflow_config(
        workflow_config, task="regression", score_handoff="after_alias_change"
    )
    assert migrated["inference_mode"] == "local"


@pytest.mark.parametrize("action", ["train", "score", "approve"])
def test_spark_inference_requires_pandas_training(workflow_config, action):
    """Contradictory training and inference modes fail before any job side effects."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    with pytest.raises(ValueError, match="inference_mode=spark requires engine=pandas"):
        validate_workflow_config({**workflow_config, "inference_mode": "spark"}, action=action)
    settings = {**workflow_config, "engine": "pandas", "inference_mode": "spark"}
    assert validate_workflow_config(settings, action=action) == settings


@pytest.mark.parametrize(
    "field,value",
    [
        ("inference_mode", "distributed"),
        ("spark_udf_env_manager", "conda"),
        ("spark_udf_prediction_batch_rows", 0),
        ("spark_udf_prediction_batch_rows", 100001),
        ("spark_udf_prediction_batch_rows", True),
    ],
)
def test_spark_runtime_settings_reject_invalid_values(workflow_config, field, value):
    """Worker setup and prediction batch limits must be explicit and bounded."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    settings = {**workflow_config, "engine": "pandas", "inference_mode": "spark", field: value}
    with pytest.raises(ValueError, match=field):
        validate_workflow_config(settings, action="score")


def test_spark_runtime_settings_accept_explicit_values(workflow_config):
    """The selected worker environment and batch size reach the scoring route."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    settings = {
        **workflow_config,
        "engine": "pandas",
        "inference_mode": "spark",
        "spark_udf_env_manager": "virtualenv",
        "spark_udf_prediction_batch_rows": 10000,
    }
    assert validate_workflow_config(settings, action="score") == settings


@pytest.mark.parametrize("action", ["train", "score", "approve"])
@pytest.mark.parametrize("enabled", [False, True])
def test_cdf_expiry_recovery_requires_explicit_boolean(workflow_config, action, enabled):
    """Optional recovery must remain independent of model-change and lifecycle policy."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    settings = {**workflow_config, "auto_rebuild_on_cdf_expiry": enabled}
    assert validate_workflow_config(settings, action=action) == settings


@pytest.mark.parametrize("value", [None, "true", "false", 0, 1, [], {}])
def test_cdf_expiry_recovery_rejects_coerced_values(workflow_config, value):
    """Truthy strings and numbers must never authorize an automatic full rescore."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    with pytest.raises(ValueError, match="auto_rebuild_on_cdf_expiry must be a boolean"):
        validate_workflow_config(
            {**workflow_config, "auto_rebuild_on_cdf_expiry": value}, action="score"
        )


@pytest.mark.parametrize("enabled,deployed", [(True, None), (True, "false"), (False, "true")])
def test_cdf_recovery_rejects_json_only_graph_changes(workflow_config, enabled, deployed):
    """An enabled runtime must have matching recovery tasks in the deployed score job."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_deployed_contract

    parameters = {"workflow_contract": "3", "deployed_score_handoff": "after_alias_change"}
    if deployed is not None:
        parameters["deployed_auto_rebuild_on_cdf_expiry"] = deployed
    with pytest.raises(ValueError, match="regenerate/redeploy"):
        validate_deployed_contract(
            {**workflow_config, "auto_rebuild_on_cdf_expiry": enabled}, parameters
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_cdf_recovery_accepts_matching_deployed_graph(workflow_config, enabled):
    """Regenerated notebook parameters explicitly match the JSON recovery policy."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_deployed_contract

    assert (
        validate_deployed_contract(
            {**workflow_config, "auto_rebuild_on_cdf_expiry": enabled},
            {
                "workflow_contract": "3",
                "deployed_score_handoff": "after_alias_change",
                "deployed_auto_rebuild_on_cdf_expiry": str(enabled).lower(),
            },
        )
        is None
    )


def test_workflow_accepts_optional_quality_gates(workflow_config):
    """Additional absolute bounds must survive offline validation without new defaults."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    settings = {**workflow_config, "quality_gates": {"heldout_r2": 0.5}}
    assert validate_workflow_config(settings, action="train") == settings
    with pytest.raises(ValueError, match="quality_gates"):
        validate_workflow_config(
            {**settings, "quality_gates": {"heldout_accuracy": 0.8}}, action="train"
        )


@pytest.mark.parametrize("action", ["train", "score"])
def test_config_accepts_exact_window_controls_and_rejects_schedule_fields(workflow_config, action):
    """Data policy belongs in workflow JSON while job clock settings stay in the Bundle."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    settings = {**workflow_config, "holdout_months": 2, "result_availability_lag_hours": 48}
    assert validate_workflow_config(settings, action=action) == settings
    for field in ("holdout_months", "result_availability_lag_hours"):
        with pytest.raises(ValueError, match=field):
            validate_workflow_config({**settings, field: True}, action=action)
    with pytest.raises(ValueError, match="Unknown workflow settings"):
        validate_workflow_config({**settings, "scoring_mode": "scheduled"}, action=action)


@pytest.mark.parametrize("action", ["train", "score", "approve"])
def test_random_workflow_allows_null_inactive_dates_and_explicit_snapshot(workflow_config, action):
    """Regenerated ordinary-table projects require no time columns or calendar windows."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    settings = {**workflow_config, "split_strategy": "random", "filter_unavailable_results": False}
    settings.update(training_window_mode="full_snapshot", window_timezone=None)
    for field in (
        "start",
        "holdout_start",
        "cutoff",
        "event_column",
        "event_time_parsing",
        "result_cutoff",
        "result_available_at_column",
        "result_time_parsing",
        "monthly_lookback_months",
    ):
        settings[field] = None
    assert validate_workflow_config(settings, action=action) == settings
    with pytest.raises(ValueError, match="inactive event"):
        validate_workflow_config({**settings, "event_column": "event_time"}, action=action)
    with pytest.raises(ValueError, match="object"):
        validate_workflow_config({**settings, "event_time_parsing": []}, action=action)


def test_random_stratification_requires_classification_in_workflow(workflow_config):
    """Preflight must not interpret repeated continuous targets as stratification classes."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    settings = {
        **workflow_config,
        "split_strategy": "random",
        "stratify": True,
        "training_window_mode": "full_snapshot",
        "window_timezone": None,
        "filter_unavailable_results": False,
    }
    for field in (
        "start",
        "holdout_start",
        "cutoff",
        "event_column",
        "result_cutoff",
        "result_available_at_column",
        "monthly_lookback_months",
    ):
        settings[field] = None
    with pytest.raises(ValueError, match="classification"):
        validate_workflow_config(settings, action="train")


@pytest.mark.parametrize("action", ["train", "score"])
def test_source_date_rules_are_validated_offline(workflow_config, action):
    """Invalid parsing cannot wait until a monthly run has opened its Delta source."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    config = {
        **workflow_config,
        "event_time_parsing": {"format": "%d/%m/%Y %H:%M", "timezone": "Asia/Tokyo"},
    }
    assert validate_workflow_config(config, action=action) == config
    with pytest.raises(ValueError, match="timezone"):
        validate_workflow_config(
            {**config, "result_time_parsing": {"format": "%Y-%m-%d %H:%M"}}, action=action
        )


def test_column_names_remain_canonical_through_validation(workflow_config):
    """Validated settings must use the same column names as the saved workflow."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    before = deepcopy(workflow_config)
    validated = validate_workflow_config(workflow_config, action="train")
    assert validated == before
    validated["record_key_columns"].append("another_key")
    assert workflow_config == before


@pytest.mark.parametrize("old_name", ["row_keys", "label_time_column", "max_bytes"])
def test_retired_column_names_are_unknown_settings(workflow_config, old_name):
    """Removed field names must not silently override the current training contract."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    with pytest.raises(ValueError, match="Unknown workflow settings"):
        validate_workflow_config({**workflow_config, old_name: "unused"}, action="train")


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"config_version": 2}, "config_version"),
        ({"config_version": True}, "config_version"),
        ({"max_rows": True}, "max_rows"),
        ({"max_input_mb": -1}, "max_input_mb"),
        ({"metric": "heldout_accuracy"}, "metric"),
        ({"quality_threshold": float("nan")}, "finite"),
        ({"min_improvement": float("inf")}, "finite"),
        ({"min_improvement": -1}, "min_improvement"),
        ({"quality_threshold": -1}, "threshold"),
        ({"prediction_table": "workspace.test.source"}, "source"),
        ({"record_key_columns": ["id", "ID"]}, "distinct"),
        ({"input_columns": ["target"]}, "distinct"),
        ({"record_key_columns": "id"}, "record_key_columns"),
        ({"unknown_option": True}, "unknown_option"),
        ({"engine": "spark"}, "engine"),
        ({"model_version": "latest"}, "model_version"),
        ({"promotion_policy": "automatic", "quality_threshold": None}, "quality_threshold"),
        ({"champion_version": "latest"}, "champion_version"),
        ({"risk_category": {"label": "low"}}, "risk_category"),
        (
            {
                "pipeline": {
                    "preprocessing": [{"name": "bad", "transformer": "linear_regression"}],
                    "modeling": {"type": "linear_regression"},
                }
            },
            "preprocessing",
        ),
    ],
)
def test_invalid_project_settings_are_rejected_offline(workflow_config, changes, message):
    """Malformed settings cannot survive until expensive training or output publication."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    with pytest.raises(ValueError, match=message):
        validate_workflow_config({**workflow_config, **changes}, action="train")


@pytest.mark.parametrize("field", ["start", "holdout_start", "cutoff"])
def test_fixed_window_needs_boundaries_only_for_training(workflow_config, field):
    """Saved-evidence actions do not require dates, but fixed-window training does."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    settings = {
        **workflow_config,
        "training_window_mode": "fixed_window",
        "monthly_lookback_months": None,
        "window_timezone": None,
        field: None,
    }
    with pytest.raises(ValueError, match=field):
        validate_workflow_config(settings, action="train")
    for action in ("score", "approve", "reject", "rollback"):
        assert validate_workflow_config(settings, action=action)[field] is None


def test_train_allows_runtime_version_and_calendar_resolution(workflow_config):
    """One train action must work without manual pins for rolling data selection."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    settings = {
        **workflow_config,
        "training_version": None,
        "start": None,
        "holdout_start": None,
        "cutoff": None,
        "result_cutoff": None,
    }
    assert validate_workflow_config(settings, action="train") == settings


def test_removed_monthly_action_is_rejected(workflow_config):
    """Old job definitions must not silently retain different training semantics."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    with pytest.raises(ValueError, match="action"):
        validate_workflow_config(workflow_config, action="train_monthly")


def test_core_pipeline_and_task_validation_are_reused(workflow_config):
    """Changing the task must select a compatible model and metric without another model list."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    settings = {
        **workflow_config,
        "task": "classification",
        "metric": "heldout_accuracy",
        "quality_threshold": 0.8,
    }
    with pytest.raises(ValueError, match="model.*task|task.*model"):
        validate_workflow_config(settings, action="train")
    settings["pipeline"] = {"preprocessing": [], "modeling": {"type": "logistic_regression"}}
    assert validate_workflow_config(settings, action="train")["task"] == "classification"
    with pytest.raises(ValueError, match="threshold"):
        validate_workflow_config({**settings, "quality_threshold": 5}, action="train")


@pytest.mark.parametrize(
    "legacy, selection, policy",
    [
        ("pinned_version", "pinned_version", "manual_approval"),
        ("auto_champion", "champion", "automatic"),
    ],
)
def test_migration_is_explicit_and_does_not_invent_snapshot_or_modify_input(
    workflow_config, legacy, selection, policy
):
    """Old coupled policy mapping must be reviewable and require a chosen handoff."""
    from skyulf.integrations.databricks.projects.workflow_config import migrate_workflow_config

    old = workflow_config
    for key in (
        "config_version",
        "task",
        "score_model_selection",
        "promotion_policy",
        "score_handoff",
    ):
        old.pop(key)
    old.update(model_selection_mode=legacy, training_version=None, start=None)
    before = deepcopy(old)
    migrated = migrate_workflow_config(old, task="regression", score_handoff="disabled")
    assert old == before
    assert migrated["score_model_selection"] == selection and migrated["promotion_policy"] == policy
    assert migrated["config_version"] == 1 and "model_selection_mode" not in migrated
    assert migrated["start"] is None and migrated["training_version"] is None


def test_deployed_contract_refuses_json_only_handoff_changes(workflow_config):
    """Editing the config alone cannot silently change deployed handoff expectations."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_deployed_contract

    with pytest.raises(ValueError, match="redeploy"):
        validate_deployed_contract(
            workflow_config, {"workflow_contract": "2", "deployed_score_handoff": "disabled"}
        )
    with pytest.raises(ValueError, match="regenerate|redeploy"):
        validate_deployed_contract(workflow_config, {})
    assert (
        validate_deployed_contract(
            workflow_config,
            {"workflow_contract": "2", "deployed_score_handoff": "after_alias_change"},
        )
        is None
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"config_version": 2},
        {"config_version": True},
        {"promotion_policy": "typo"},
        {"score_model_selection": None},
        {"task": "classification"},
    ],
)
def test_migration_does_not_downgrade_or_reinterpret_project(workflow_config, changes):
    """Migration must reject future versions and contradictory existing task or policy choices."""
    from skyulf.integrations.databricks.projects.workflow_config import migrate_workflow_config

    with pytest.raises(ValueError):
        migrate_workflow_config(
            {**workflow_config, **changes}, task="regression", score_handoff="after_alias_change"
        )


@pytest.mark.parametrize(
    "change",
    [
        {"record_key_columns": ["run_id"]},
        {"input_columns": ["_commit_version"]},
    ],
)
def test_reserved_output_and_cdf_columns_fail_before_compute(workflow_config, change):
    """Project initialization must expose key collisions instead of waiting for scoring."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    with pytest.raises(ValueError, match="reserved"):
        validate_workflow_config({**workflow_config, **change}, action="score")


@pytest.mark.parametrize("megabytes", [1, 64, 256])
def test_input_megabytes_reach_training_and_scoring_as_bytes(workflow_config, megabytes):
    """Readable Bundle units must preserve the same byte budget on both execution paths."""
    from skyulf.integrations.databricks.lifecycle.local_workflow import (
        _scoring_config,
        training_spec,
    )
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    config = {key: value for key, value in workflow_config.items() if key != "max_bytes"}
    config["max_input_mb"] = megabytes
    checked = validate_workflow_config(config, action="train")
    assert training_spec(checked).max_bytes == megabytes * 1024 * 1024
    assert _scoring_config(checked).source.max_bytes == megabytes * 1024 * 1024
    assert "max_bytes" not in checked


@pytest.mark.parametrize("value", [None, 0, -1, True, "64", 1.5, float("inf")])
def test_input_megabytes_reject_invalid_units_before_execution(workflow_config, value):
    """An invalid size must never become an unlimited or incorrectly scaled byte budget."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    config = {key: item for key, item in workflow_config.items() if key != "max_bytes"}
    config["max_input_mb"] = value
    with pytest.raises(ValueError, match="max_input_mb"):
        validate_workflow_config(config, action="train")


@pytest.mark.parametrize("action", ["score", "approve", "reject", "rollback"])
def test_saved_model_actions_do_not_expand_editable_training_search(workflow_config, action):
    """Saved model replay must not depend on unexecuted project ensemble overrides."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config

    workflow_config["pipeline"]["modeling"] = {
        "type": "hyperparameter_tuner",
        "base_model": {"type": "voting_regressor", "params": {}},
        "strategy": "grid",
        "metric": "rmse",
        "max_candidates": 10,
        "search_space": {},
    }
    with pytest.raises(ValueError, match="above max_candidates"):
        validate_workflow_config(workflow_config, action="train")
    assert validate_workflow_config(workflow_config, action=action) == workflow_config
