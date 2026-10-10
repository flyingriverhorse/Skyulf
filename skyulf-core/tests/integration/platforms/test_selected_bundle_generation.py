"""Real CLI generation retains selected settings and their runtime defaults."""

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from test_databricks_bundle_generation import (
    _generate_project,
    _read_modeling,
    _read_validated_config,
)

from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

pytestmark = pytest.mark.skipif(
    not (
        os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE")
        or os.environ.get("SKYULF_BUNDLE_OFFLINE_CLI") == "1"
    ),
    reason="CLI opt-in",
)


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_only_yaml_model_settings_are_generated(tmp_path, layout):
    """Unused layout hooks must not distract operators or become accidental owners."""
    project = _generate_project(tmp_path, training_layout=layout)
    assert not (project / "src/modeling").exists()
    assert (project / "config/training.yml").is_file()
    assert (project / "config/inference.yml").is_file()
    assert not (project / "selection").exists()
    assert _read_validated_config(project, action="train")["training_layout"] == layout


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_unused_optional_training_settings_are_absent(tmp_path, layout):
    """Date-free random training keeps defaults without irrelevant null placeholders."""
    from skyulf.integrations.databricks.lifecycle.workflow import training_settings, training_spec
    from skyulf.integrations.databricks.training.tuning.cv import CVSpec

    project = _generate_project(
        tmp_path,
        training_layout=layout,
        training_window_mode="full_snapshot",
        split_strategy="random",
        cv_enabled="false",
        cv_type="k_fold",
        filter_unavailable_results="false",
        event_column="",
        result_available_at_column="",
    )
    config = _read_validated_config(project, action="train")
    optional = {
        "lookback_days",
        "holdout_days",
        "window_timezone",
        "holdout_months",
        "monthly_lookback_months",
        "result_availability_lag_hours",
        "cv_group_column",
        "cv_test_size",
        "cv_max_train_size",
        "event_column",
        "result_available_at_column",
        "start",
        "holdout_start",
        "cutoff",
        "result_cutoff",
    }
    assert optional.isdisjoint(config)
    assert config["random_state"] == 42
    assert config["filter_unavailable_results"] is False
    config = {**config, "training_table": "workspace.default.source"}
    legacy = {**config, **dict.fromkeys(optional)}
    now = datetime(2026, 10, 5, tzinfo=UTC)
    assert training_spec(training_settings(config, now)) == training_spec(
        training_settings(legacy, now)
    )
    assert CVSpec.from_workflow(config) == CVSpec.from_workflow(legacy)


@pytest.mark.parametrize("strategy", ["random", "grid", "optuna"])
def test_unused_tuning_timeout_is_absent(tmp_path, strategy):
    """Default tuning must not expose a timeout setting with no selected value."""
    project = _generate_project(tmp_path, search_strategy=strategy)
    assert "timeout" not in _read_modeling(project)


def test_selected_optional_settings_and_semantic_none_are_retained(tmp_path):
    """Configured values and explicit model None sentinels must survive generation."""
    project = _generate_project(
        tmp_path,
        search_strategy="optuna",
        search_settings="custom",
        search_timeout="30",
        training_window_mode="rolling_days",
        lookback_days="45",
        split_strategy="temporal",
        holdout_days="7",
        event_column="event_time",
        event_time_timezone="UTC",
        model_params='{"alpha": 1.0, "max_iter": null}',
    )
    config = read_workflow_config(project / "config/training.yml")
    model = _read_modeling(project)
    assert config["lookback_days"] == 45
    assert config["holdout_days"] == 7
    assert config["event_column"] == "event_time"
    assert config.get("weight_column") is None
    assert model["timeout"] == 30
    assert model["base_model"]["params"]["max_iter"] is None


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize(
    "shap,charts,recovery",
    [
        ("false", "false", "false"),
        ("true", "false", "false"),
        ("false", "true", "false"),
        ("false", "false", "true"),
        ("true", "true", "true"),
    ],
)
def test_generated_notebooks_match_selected_job_tasks(tmp_path, layout, shap, charts, recovery):
    """Every deployed notebook must exist and unused layout notebooks stay absent."""
    project = _generate_project(
        tmp_path,
        training_layout=layout,
        shap_enabled=shap,
        evaluation_charts_enabled=charts,
        auto_rebuild_on_cdf_expiry=recovery,
    )
    referenced = set()
    for resource in (project / "resources").glob("*.yml"):
        jobs = yaml.safe_load(resource.read_text())["resources"]["jobs"]
        for job in jobs.values():
            for task in job["tasks"]:
                if "notebook_task" in task:
                    relative = task["notebook_task"]["notebook_path"]
                    referenced.add((resource.parent / Path(relative)).resolve())
    generated = {path.resolve() for path in (project / "src/jobs").rglob("*.py")}
    assert generated == referenced
