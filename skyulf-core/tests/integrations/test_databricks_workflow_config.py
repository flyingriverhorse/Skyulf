"""Reject invalid project settings before registry, training or table side effects."""

from copy import deepcopy

import pytest


def test_column_names_remain_canonical_through_validation(workflow_config):
    """Validated settings must use the same column names as the saved workflow."""
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    before = deepcopy(workflow_config)
    validated = validate_workflow_config(workflow_config, action="train")
    assert validated == before
    validated["record_key_columns"].append("another_key")
    assert workflow_config == before


@pytest.mark.parametrize("old_name", ["row_keys", "label_time_column"])
def test_retired_column_names_are_unknown_settings(workflow_config, old_name):
    """Removed field names must not silently override the current training contract."""
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    with pytest.raises(ValueError, match="Unknown workflow settings"):
        validate_workflow_config({**workflow_config, old_name: "unused"}, action="train")


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"config_version": 2}, "config_version"),
        ({"config_version": True}, "config_version"),
        ({"max_rows": True}, "max_rows"),
        ({"max_bytes": -1}, "max_bytes"),
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
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    with pytest.raises(ValueError, match=message):
        validate_workflow_config({**workflow_config, **changes}, action="train")


@pytest.mark.parametrize("field", ["start", "holdout_start", "cutoff", "training_version"])
def test_missing_manual_snapshot_is_required_only_for_manual_training(workflow_config, field):
    """Scoring and saved-evidence actions must not require new manual training dates."""
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    settings = {**workflow_config, field: None}
    with pytest.raises(ValueError, match=field):
        validate_workflow_config(settings, action="train")
    for action in ("score", "approve", "reject", "rollback", "train_monthly"):
        assert validate_workflow_config(settings, action=action)[field] is None


def test_core_pipeline_and_task_validation_are_reused(workflow_config):
    """Changing the task must select a compatible model and metric without another model list."""
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

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
    from skyulf.integrations.databricks.workflow_config import migrate_workflow_config

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
    from skyulf.integrations.databricks.workflow_config import validate_deployed_contract

    with pytest.raises(ValueError, match="redeploy"):
        validate_deployed_contract(
            workflow_config, {"workflow_contract": "1", "deployed_score_handoff": "disabled"}
        )
    with pytest.raises(ValueError, match="regenerate|redeploy"):
        validate_deployed_contract(workflow_config, {})
    assert (
        validate_deployed_contract(
            workflow_config,
            {"workflow_contract": "1", "deployed_score_handoff": "after_alias_change"},
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
    from skyulf.integrations.databricks.workflow_config import migrate_workflow_config

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
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    with pytest.raises(ValueError, match="reserved"):
        validate_workflow_config({**workflow_config, **change}, action="score")
