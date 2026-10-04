"""Training layout determines whether the wizard or branch recipes own model choices."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

SCHEMA = (
    Path(__file__).resolve().parents[3] / "templates/databricks/databricks_template_schema.json"
)
SHARED_FIELDS = {
    "training_layout",
    "project_name",
    "engine",
    "catalog",
    "schema",
    "source_table_name",
    "record_key_columns",
    "training_version",
    "max_rows",
    "max_input_mb",
    "compute_mode",
    "deployment_identity",
    "manage_job_permissions",
    "personal_development_targets",
    "auto_rebuild_on_cdf_expiry",
    "evaluation_charts_enabled",
    "cluster_policy_name",
    "spark_version",
    "node_type_id",
    "cost_tag_key",
    "cost_tag_value",
    "retraining_mode",
    "retraining_cron_expression",
    "retraining_timezone_id",
    "retraining_pause_status",
    "model_set_output_mode",
    "model_set_name",
    "model_set_promotion_policy",
    "score_handoff",
    "shap_enabled",
    "source_change_policy",
    "model_change_mode",
    "model_set_table_name",
    "model_view_prefix",
    "combined_view_name",
}


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {
            "task": "classification",
            "cv_enabled": "true",
            "cv_type": "nested_cv",
            "cv_nested_type": "stratified_group_k_fold",
            "search_settings": "custom",
            "search_strategy": "optuna",
            "filter_unavailable_results": "true",
        },
        {
            "cv_enabled": "true",
            "cv_type": "time_series_split",
            "training_window_mode": "fixed_window",
            "search_settings": "custom",
            "search_strategy": "halving_grid",
            "scoring_mode": "scheduled",
        },
    ],
)
def test_multi_target_prompts_for_shared_setup_and_independent_branches(settings):
    """Model-set setup must collect per-branch settings without unrelated root-model prompts."""
    properties = json.loads(SCHEMA.read_text())["properties"]
    values = {name: field["default"] for name, field in properties.items()}
    values.update(settings, training_layout="multi_target")
    visible = {
        name
        for name, field in properties.items()
        if not Draft7Validator(field.get("skip_prompt_if", False)).is_valid(values)
    }
    branch_fields = {name for name in properties if name.startswith("branch_")}
    assert visible <= SHARED_FIELDS | branch_fields
    assert {"branch_count", "branch_1_name", "branch_2_target_column"} <= visible
    assert {
        "project_name",
        "engine",
        "source_table_name",
        "record_key_columns",
        "training_version",
        "compute_mode",
        "retraining_mode",
    } <= visible
    # Deployment namespaces are editable target bindings, not interactive model prompts.
    assert {"catalog", "schema"}.isdisjoint(visible)


def test_multi_target_keeps_scheduled_training_and_policy_compute_questions():
    """Selecting branch training must retain the shared deployment controls it needs."""
    properties = json.loads(SCHEMA.read_text())["properties"]
    values = {name: field["default"] for name, field in properties.items()}
    values.update(
        training_layout="multi_target", retraining_mode="scheduled", compute_mode="policy_cluster"
    )
    fields = {
        "retraining_cron_expression",
        "retraining_timezone_id",
        "retraining_pause_status",
        "cluster_policy_name",
        "spark_version",
        "node_type_id",
        "cost_tag_key",
        "cost_tag_value",
    }
    assert all(
        not Draft7Validator(properties[name].get("skip_prompt_if", False)).is_valid(values)
        for name in fields
    )


@pytest.mark.parametrize(
    "layout,mode,expected",
    [
        ("single_model", "separate_views", set()),
        ("multi_target", "all", {"model_set_output_mode", "model_set_table_name"}),
        ("multi_target", "combined_only", {"model_set_output_mode", "model_set_table_name"}),
        (
            "multi_target",
            "separate_views",
            {
                "model_set_output_mode",
                "model_set_table_name",
                "model_view_prefix",
                "combined_view_name",
            },
        ),
    ],
)
def test_model_set_output_questions_follow_selected_layout(layout, mode, expected):
    """Only separate views need consumer names; single-model setup remains unchanged."""
    properties = json.loads(SCHEMA.read_text())["properties"]
    names = {
        "model_set_name",
        "source_change_policy",
        "model_set_output_mode",
        "model_set_table_name",
        "model_view_prefix",
        "combined_view_name",
    }
    if layout == "multi_target":
        expected = expected | {"model_set_name", "source_change_policy"}
    visible = {
        name
        for name in names
        if not Draft7Validator(properties[name]["skip_prompt_if"]).is_valid(
            {"training_layout": layout, "model_set_output_mode": mode}
        )
    }
    assert visible == expected


@pytest.mark.parametrize(
    "name,valid",
    [("", True), ("profit_models", True), ("other.schema.set", False), ("bad-name", False)],
)
def test_model_set_name_accepts_default_or_simple_custom_name(name, valid):
    """Custom registry names must preserve namespace bindings and safe template rendering."""
    definition = json.loads(SCHEMA.read_text())["properties"]["model_set_name"]
    assert Draft7Validator(definition).is_valid(name) is valid


def test_multi_target_exposes_independent_model_and_source_change_choices():
    """Choosing how new models replace results must remain separate from source corrections."""
    properties = json.loads(SCHEMA.read_text())["properties"]
    values = {name: field["default"] for name, field in properties.items()}
    values["training_layout"] = "multi_target"
    for name in ("model_change_mode", "source_change_policy"):
        assert not Draft7Validator(properties[name].get("skip_prompt_if", False)).is_valid(values)
