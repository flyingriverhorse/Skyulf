"""Wizard answers that Core would reject are hidden or refused when the project is generated."""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft7Validator
from test_databricks_bundle_generation import (
    CLI,
    OFFLINE_CLI,
    PROFILE,
    _generate_project,
    _initialize_project,
    _read_validated_config,
)

TEMPLATE_ROOT = Path(__file__).resolve().parents[3] / "templates/databricks"
needs_cli = pytest.mark.skipif(
    not (PROFILE or OFFLINE_CLI) or not CLI,
    reason="Set SKYULF_BUNDLE_OFFLINE_CLI=1 to render the template with the installed CLI.",
)


def _properties():
    """Read the generated initializer questions exactly as the CLI receives them."""
    return json.loads((TEMPLATE_ROOT / "databricks_template_schema.json").read_text())["properties"]


def _visible(properties, name, values):
    """Apply the CLI skip predicate to one answer set."""
    return not Draft7Validator(properties[name].get("skip_prompt_if", False)).is_valid(values)


def _template_regex(relative, marker):
    """Read the regular expression a template guard uses, so tests compare it with Core."""
    text = (TEMPLATE_ROOT / relative).read_text(encoding="utf-8")
    return re.compile(re.search(marker + r' := "([^"]+)"', text).group(1))


def _menu_models(properties):
    """List every model a wizard menu can generate, including weighted and branch menus."""
    return sorted(
        {
            choice
            for name, field in properties.items()
            if re.search(r"(?:regression|classification)_model(?:_[1-8])?(?:_weighted)?$", name)
            for choice in field["enum"]
        }
    )


def test_estimator_resource_guard_matches_core_model_parameters():
    """A stale model list would let n_estimators halving reach Core for a model without it."""
    from skyulf.integrations.databricks.training.tuning.search import instantiate_model
    from skyulf.registry import NodeRegistry

    pattern = _template_regex("library/modeling.tmpl", r"\$estimator_resource_models")
    for model in _menu_models(_properties()):
        calculator = NodeRegistry.get_calculator(model)()
        estimator = instantiate_model(cast(Any, calculator).model_class, calculator.default_params)
        supported = "n_estimators" in estimator.get_params(deep=True)
        assert bool(pattern.search(model)) is supported, model


def test_competition_metric_guard_matches_core_admission():
    """A stale metric list would generate competitions that Core refuses before training."""
    from skyulf.integrations.databricks.training.competition.competition_evaluation import (
        competition_metric,
    )

    pattern = _template_regex(
        "template/{{.project_name}}/config/training.yml.tmpl", r"\$unsupported_competition_metrics"
    )
    properties = _properties()
    for task in ("regression", "classification"):
        for metric in properties[f"{task}_metric"]["enum"]:
            try:
                competition_metric(metric, task)
                rejected = False
            except ValueError:
                rejected = True
            assert bool(pattern.search(metric)) is rejected, (task, metric)


@pytest.mark.parametrize("prefix", ["", "branch_1_"])
def test_estimator_max_resources_default_is_a_core_bound(prefix):
    """The old "null" default produced max_resources: null, which Core rejects for n_estimators."""
    field = _properties()[prefix + "search_max_resources"]
    validator = Draft7Validator(field)
    assert field["default"] == "100" and validator.is_valid(field["default"])
    assert not any(validator.is_valid(value) for value in ("null", "auto", "0", "1"))


@pytest.mark.parametrize(
    ("layout", "prefix"), [("single_model", ""), ("multi_target", "branch_1_")]
)
def test_full_snapshot_window_hides_the_split_question(layout, prefix):
    """Full-snapshot windows require random splitting, so the temporal choice must not be offered."""
    properties = _properties()
    values = {name: field["default"] for name, field in properties.items()}
    values.update(setup_mode="detailed", training_layout=layout)
    window, split = prefix + "training_window_mode", prefix + "split_strategy"
    assert properties[window]["order"] < properties[split]["order"]
    assert _visible(properties, split, values)
    values[window] = "full_snapshot"
    assert not _visible(properties, split, values)


@pytest.mark.parametrize(
    ("answers", "visible"),
    [
        ({}, True),
        ({"training_layout": "model_competition"}, True),
        ({"training_layout": "multi_target"}, False),
        ({"engine": "polars"}, False),
    ],
)
def test_online_feature_lookup_is_offered_only_for_pandas_single_or_competition(answers, visible):
    """Core accepts online features only for pandas single-model or competition training."""
    properties = _properties()
    values = {name: field["default"] for name, field in properties.items()}
    values.update(setup_mode="detailed", **answers)
    assert _visible(properties, "include_online_feature_lookup", values) is visible


@pytest.mark.parametrize(
    ("setup", "engine", "visible"),
    [("detailed", "pandas", True), ("detailed", "polars", False), ("quick", "pandas", False)],
)
def test_prediction_delivery_defaults_to_batch_and_hides_unsupported_choices(
    setup, engine, visible
):
    """Quick and Polars setup must keep batch delivery without offering unsupported endpoints."""
    properties = _properties()
    values = {name: field["default"] for name, field in properties.items()}
    values.update(setup_mode=setup, engine=engine)
    delivery = properties["prediction_delivery"]
    assert delivery["default"] == "batch"
    assert delivery["enum"] == ["batch", "endpoint", "both"]
    assert "include_serving_endpoint" not in properties
    assert properties["engine"]["order"] < delivery["order"]
    assert delivery["order"] < properties["batch_compute"]["order"]
    assert _visible(properties, "prediction_delivery", values) is visible
    assert _visible(properties, "batch_compute", values) is visible


def test_prediction_delivery_is_chosen_before_every_dependent_question():
    """The CLI must know delivery before it decides which questions to omit."""
    properties = _properties()
    order = properties["prediction_delivery"]["order"]
    dependent = [
        field
        for field in properties.values()
        if '"prediction_delivery"' in json.dumps(field.get("skip_prompt_if", {}))
    ]
    assert dependent
    assert all(order < field["order"] for field in dependent)


_DUPLICATE_BASES = "ERROR_ensemble_base_models_must_differ"
_ESTIMATOR_RESOURCE = "ERROR_search_resource_n_estimators_requires_a_model_with_n_estimators"
_COMPETITION_METRIC = "ERROR_metric_is_not_supported_for_model_competition"
_STRATIFIED = "ERROR_stratified_cv_requires_task_classification"
_FULL_SNAPSHOT = "ERROR_training_window_mode_full_snapshot_requires_random_split"
_ONLINE = "ERROR_online_feature_lookup_requires_engine_pandas_and_single_model_or_competition"
_HALVING = {"search_strategy": "halving_random", "search_settings": "custom"}
_BRANCH_HALVING = {f"branch_1_{key}": value for key, value in _HALVING.items()}


@needs_cli
@pytest.mark.parametrize(
    ("guard", "answers"),
    [
        (
            _DUPLICATE_BASES,
            {
                "regression_model": "stacking_regressor",
                "single_ensemble_regression_base_2": "linear_regression",
            },
        ),
        (
            _DUPLICATE_BASES,
            {
                "task": "classification",
                "training_layout": "model_competition",
                "competition_classification_model_1": "voting_classifier",
                "competition_ensemble_1_classification_base_count": "3",
                "competition_ensemble_1_classification_base_3": "logistic_regression",
            },
        ),
        (
            _DUPLICATE_BASES,
            {
                "training_layout": "multi_target",
                "branch_1_regression_model": "stacking_regressor",
                "branch_1_ensemble_regression_base_2": "linear_regression",
            },
        ),
        (
            _ESTIMATOR_RESOURCE,
            {
                **_HALVING,
                "search_resource": "n_estimators",
                "regression_model": "linear_regression",
            },
        ),
        (
            _ESTIMATOR_RESOURCE,
            {
                **_BRANCH_HALVING,
                "training_layout": "multi_target",
                "branch_1_search_resource": "n_estimators",
                "branch_1_regression_model": "linear_regression",
            },
        ),
        (
            _COMPETITION_METRIC,
            {"training_layout": "model_competition", "regression_metric": "heldout_mape"},
        ),
        (
            _COMPETITION_METRIC,
            {
                "training_layout": "model_competition",
                "task": "classification",
                "classification_metric": "heldout_precision_weighted",
            },
        ),
        (
            _COMPETITION_METRIC,
            {
                "training_layout": "model_competition",
                "task": "classification",
                "classification_metric": "heldout_recall_weighted",
            },
        ),
        (_STRATIFIED, {"cv_enabled": "true", "cv_type": "stratified_k_fold"}),
        (
            _STRATIFIED,
            {"cv_enabled": "true", "cv_type": "stratified_group_k_fold", "cv_group_column": "g"},
        ),
        (
            _STRATIFIED,
            {"cv_enabled": "true", "cv_type": "nested_cv", "cv_nested_type": "stratified_k_fold"},
        ),
        (_STRATIFIED, {"training_layout": "model_competition", "cv_type": "stratified_k_fold"}),
        (
            _STRATIFIED,
            {
                "training_layout": "multi_target",
                "branch_1_cv_enabled": "true",
                "branch_1_cv_type": "stratified_k_fold",
            },
        ),
        (_FULL_SNAPSHOT, {"training_window_mode": "full_snapshot", "split_strategy": "temporal"}),
        (
            _FULL_SNAPSHOT,
            {
                "training_layout": "multi_target",
                "branch_1_training_window_mode": "full_snapshot",
                "branch_1_split_strategy": "temporal",
            },
        ),
        (_ONLINE, {"include_online_feature_lookup": "true", "training_layout": "multi_target"}),
        (_ONLINE, {"include_online_feature_lookup": "true", "engine": "polars"}),
    ],
)
def test_cli_refuses_answers_that_core_rejects(tmp_path, guard, answers):
    """These answer files used to render projects that failed only at smoke or preview time."""
    generated, _ = _initialize_project(tmp_path, **answers)
    assert generated.returncode != 0
    assert guard in generated.stdout + generated.stderr


@needs_cli
@pytest.mark.parametrize(
    "answers",
    [
        {"regression_model": "stacking_regressor", "single_ensemble_regression_base_count": "3"},
        {
            "training_layout": "model_competition",
            "competition_regression_model_1": "voting_regressor",
        },
        {"training_layout": "multi_target", "branch_1_regression_model": "stacking_regressor"},
        {
            **_HALVING,
            "search_resource": "n_estimators",
            "regression_model": "random_forest_regressor",
        },
        {"regression_metric": "heldout_mape"},
        {"task": "classification", "classification_metric": "heldout_precision_weighted"},
        {"task": "classification", "cv_enabled": "true", "cv_type": "stratified_k_fold"},
        {"cv_enabled": "true", "cv_type": "nested_cv", "cv_nested_type": "k_fold"},
        {"cv_type": "stratified_k_fold"},
        {"training_window_mode": "full_snapshot"},
    ],
)
def test_cli_keeps_valid_neighbouring_answers(tmp_path, answers):
    """The guards must reject only the unsupported combination, not its valid neighbours."""
    project = _generate_project(tmp_path, **answers)
    assert _read_validated_config(project, action="train")


@needs_cli
def test_cli_keeps_online_feature_lookup_stub_for_pandas_single_model(tmp_path):
    """Online serving still generates the lookup declarations users must complete."""
    project = _generate_project(tmp_path, include_online_feature_lookup="true")
    text = (project / "config/training.yml").read_text(encoding="utf-8")
    assert "feature_lookup:" in text and "online_features:" in text


def _preview(project, action="train"):
    """Run the generated preview tool as a user would."""
    return subprocess.run(
        [sys.executable, str(project / "src/tools/preview.py"), "--action", action],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=project,
    )


@needs_cli
@pytest.mark.parametrize("enabled", ["false", "true"])
def test_cli_multi_target_preview_passes_cdf_recovery_setting(tmp_path, enabled):
    """Preview must pass the deployed recovery flag, or Core reports the tasks disagree."""
    project = _generate_project(
        tmp_path, training_layout="multi_target", rescore_when_history_expires=enabled
    )
    result = _preview(project)
    assert result.returncode == 0, result.stdout + result.stderr


@needs_cli
def test_cli_preview_reports_unreadable_config_without_traceback(tmp_path):
    """A hand-edited config with a duplicate key must give one actionable line, not a traceback."""
    project = _generate_project(tmp_path)
    path = project / "config/training.yml"
    path.write_text(path.read_text(encoding="utf-8") + "version: 1\n", encoding="utf-8")
    result = _preview(project)
    assert result.returncode == 2
    assert "Configuration error:" in result.stderr and "Traceback" not in result.stderr
