"""Guided search setup keeps ordinary training and search contracts distinct."""

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft7Validator

TEMPLATE_ROOT = Path(__file__).resolve().parents[3] / "templates/databricks"


def _properties():
    """Read the actual initializer questions so visibility regressions are caught."""
    return json.loads((TEMPLATE_ROOT / "databricks_template_schema.json").read_text())["properties"]


def _visible(properties, name, values):
    """Apply the same skip predicate used by the template initializer."""
    return not Draft7Validator(properties[name].get("skip_prompt_if", False)).is_valid(values)


def test_shap_prompts_are_opt_in_and_bounded():
    """Optional explanations must not ask budgets or install work by default."""
    properties = _properties()
    values = {name: item["default"] for name, item in properties.items()}
    assert values["shap_enabled"] == "false"
    budgets = ("shap_max_samples", "shap_max_features", "shap_max_display_samples")
    assert all(not _visible(properties, name, values) for name in budgets)
    values["shap_enabled"] = "true"
    assert all(_visible(properties, name, values) for name in budgets)
    assert Draft7Validator(properties["shap_max_samples"]).is_valid("200")
    assert not Draft7Validator(properties["shap_max_samples"]).is_valid("201")
    assert not Draft7Validator(properties["shap_max_features"]).is_valid("51")
    assert Draft7Validator(properties["shap_max_display_samples"]).is_valid("0")


def test_nested_policy_questions_follow_active_metadata_and_task():
    """Guided setup must expose only the metadata required by its nested policy."""
    properties = _properties()
    values = {name: item["default"] for name, item in properties.items()}
    values.update(cv_enabled="true", cv_type="nested_cv", cv_nested_type="time_series_split")
    assert _visible(properties, "cv_nested_type", values)
    assert all(
        _visible(properties, name, values)
        for name in ("cv_gap", "cv_test_size", "cv_max_train_size")
    )
    assert not _visible(properties, "cv_group_column", values)
    values.update(cv_nested_type="stratified_group_k_fold", task="classification")
    assert _visible(properties, "cv_group_column", values)
    assert _visible(properties, "search_tune_threshold", values)
    assert not _visible(properties, "cv_gap", values)


def test_search_questions_follow_strategy_and_keep_cv_single():
    """Search setup should expose relevant controls with Core defaults first."""
    properties = _properties()
    values = {name: item["default"] for name, item in properties.items()}
    assert "search_mode" not in properties
    assert _visible(properties, "search_strategy", values)
    assert _visible(properties, "search_settings", values)
    assert all(
        not _visible(properties, field, values) for field in ("search_space", "search_timeout")
    )
    assert _visible(properties, "search_n_trials", values)
    values.update(task="regression", search_strategy="grid")
    assert all(
        _visible(properties, field, values)
        for field in (
            "search_strategy",
            "search_max_candidates",
            "search_random_state",
            "regression_search_metric",
        )
    )
    assert not _visible(properties, "search_n_trials", values)
    assert not _visible(properties, "search_space", values)
    assert not _visible(properties, "search_timeout", values)
    assert not _visible(properties, "classification_search_metric", values)
    values.update(search_strategy="optuna", task="classification")
    assert values["search_settings"] == "default"
    assert not _visible(properties, "search_timeout", values)
    values["search_settings"] = "custom"
    assert _visible(properties, "search_timeout", values)
    assert _visible(properties, "search_n_trials", values)
    assert not _visible(properties, "search_max_candidates", values)
    assert _visible(properties, "classification_search_metric", values)
    assert not _visible(properties, "regression_search_metric", values)
    assert all(not name.startswith("search_cv_") for name in properties)
    assert set(properties["search_strategy"]["enum"]) == {
        "grid",
        "random",
        "optuna",
        "halving_grid",
        "halving_random",
    }
    values["search_strategy"] = "halving_random"
    assert all(
        _visible(properties, name, values)
        for name in ("search_factor", "search_resource", "search_min_resources")
    )
    assert not _visible(properties, "search_estimator_min_resources", values)
    values["search_resource"] = "n_estimators"
    assert _visible(properties, "search_estimator_min_resources", values)
    assert not _visible(properties, "search_min_resources", values)
    values["search_resource"] = "n_samples"
    assert all(
        not _visible(properties, name, values)
        for name in ("search_sampler", "search_pruner", "search_timeout")
    )
    values["search_strategy"] = "optuna"
    assert all(
        _visible(properties, name, values)
        for name in ("search_sampler", "search_pruner", "search_timeout")
    )
    assert all(
        not _visible(properties, name, values)
        for name in ("search_factor", "search_resource", "search_min_resources")
    )


def test_ordinary_model_hides_ensemble_and_raw_parameter_prompts():
    """Ordinary training should not ask for unrelated composition or raw JSON settings."""
    properties = _properties()
    values = {name: item["default"] for name, item in properties.items()}
    assert not _visible(properties, "model_params", values)
    assert all(
        not _visible(properties, name, values)
        for name in properties
        if name.startswith("single_ensemble_")
    )


@pytest.mark.parametrize(
    ("task", "model", "controls"),
    [
        ("regression", "voting_regressor", {"regression_weight_1", "regression_weight_2"}),
        ("regression", "stacking_regressor", {"regression_final", "cv", "passthrough"}),
        (
            "classification",
            "voting_classifier",
            {"classification_weight_1", "classification_weight_2", "voting", "calibrate"},
        ),
        (
            "classification",
            "stacking_classifier",
            {"classification_final", "cv", "passthrough", "calibrate"},
        ),
    ],
)
def test_single_ensemble_prompts_follow_selected_model(task, model, controls):
    """Each ensemble needs its own base menus and controls without the other task's prompts."""
    properties = _properties()
    values = {name: item["default"] for name, item in properties.items()}
    values.update(training_layout="single_model", task=task)
    values[f"{task}_model"] = model
    expected = controls | {f"{task}_base_count", f"{task}_base_1", f"{task}_base_2"}
    visible = {
        name.removeprefix("single_ensemble_")
        for name in properties
        if name.startswith("single_ensemble_") and _visible(properties, name, values)
    }
    assert visible == expected
    assert not _visible(properties, "model_params", values)
    assert Draft7Validator(properties["model_params"]).is_valid('{"n_jobs": 1}')


def test_search_budget_and_seed_inputs_are_bounded():
    """Guided values must reject out-of-budget trials and malformed search seeds."""
    properties = _properties()
    assert properties["search_n_trials"]["pattern"] == "^(?:[1-9]|[1-9][0-9]{1,2}|1000)$"
    assert properties["search_max_candidates"]["pattern"] == "^(?:[1-9]|[1-9][0-9]{1,3}|10000)$"
    assert properties["search_timeout"]["pattern"] == "^(?:null|[1-9][0-9]*)$"
    assert properties["search_random_state"]["pattern"] == "^(?:0|[1-9][0-9]*)$"


def test_cv_questions_follow_enabled_method():
    """Fold settings should appear only when they can affect candidate scoring."""
    properties = _properties()
    values = {name: item["default"] for name, item in properties.items()}
    assert not _visible(properties, "cv_folds", values)
    assert not _visible(properties, "cv_type", values)
    values["cv_enabled"] = "true"
    assert all(
        _visible(properties, name, values)
        for name in ("cv_folds", "cv_type", "cv_shuffle", "cv_random_state")
    )
    values["cv_type"] = "time_series_split"
    assert not _visible(properties, "cv_shuffle", values)
    assert not _visible(properties, "cv_random_state", values)
    values["cv_type"] = "shuffle_split"
    assert not _visible(properties, "cv_shuffle", values)
    assert _visible(properties, "cv_random_state", values)


def test_nested_temporal_hides_unused_randomization_questions():
    """Temporal nested splitting must not ask for shuffling or its unused seed."""
    properties = _properties()
    values = {name: item["default"] for name, item in properties.items()}
    values.update(cv_enabled="true", cv_type="nested_cv", cv_nested_type="time_series_split")
    assert not _visible(properties, "cv_shuffle", values)
    assert not _visible(properties, "cv_random_state", values)


@pytest.mark.parametrize("method", ["time_series_split", "nested_cv"])
def test_temporal_cv_opens_required_data_questions_with_default_window(method):
    """Selecting chronological CV must expose its clock and final holdout window."""
    properties = _properties()
    values = {name: item["default"] for name, item in properties.items()}
    values.update(cv_enabled="true", cv_type=method, cv_nested_type="time_series_split")
    assert not _visible(properties, "split_strategy", values)
    for name in (
        "event_column",
        "event_time_kind",
        "window_timezone",
        "monthly_lookback_months",
        "holdout_months",
    ):
        assert _visible(properties, name, values), name
        assert properties[name]["order"] > properties["cv_nested_type"]["order"]


def test_search_example_names_compatible_model_axis():
    """The guided example should use Core defaults and a native search metric."""
    example = json.loads((TEMPLATE_ROOT / "examples/tuning-init.example.json").read_text())
    assert "search_mode" not in example
    assert example["regression_model"] == "random_forest_regressor"
    assert example["regression_search_metric"] == "rmse"
    assert "search_space" not in example


def test_model_definitions_are_synced_with_generated_bundle(tmp_path):
    """Inline model and search settings must reach the training workspace together."""
    template = TEMPLATE_ROOT / "template/{{.project_name}}"
    bundle = (template / "databricks.yml.tmpl").read_text(encoding="utf-8")
    sync = yaml.safe_load(bundle.split("sync:\n", 1)[1].split("\nvariables:", 1)[0])
    modeling = tmp_path / "config"
    modeling.mkdir(parents=True)
    definitions = set()
    for name in ("training.yml", "inference.yml"):
        assert (template / "config" / f"{name}.tmpl").is_file()
        generated = modeling / name
        generated.write_text("version: 1\n", encoding="utf-8")
        definitions.add(generated)
    synced = {path for pattern in sync["include"] for path in tmp_path.glob(pattern)}
    assert definitions <= synced
