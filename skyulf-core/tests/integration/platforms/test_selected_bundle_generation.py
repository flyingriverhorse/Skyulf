"""Real CLI generation retains selected settings and their runtime defaults."""

import json
import os
import runpy
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from test_databricks_bundle_generation import _generate_project, _read_validated_config

pytestmark = pytest.mark.skipif(
    not os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE"), reason="CLI opt-in"
)


@pytest.mark.parametrize(
    "layout,expected",
    [
        ("single_model", {"single_model.py"}),
        ("model_competition", {"model_competition.py"}),
        ("multi_target", {"multi_model.py", "model_set.py"}),
    ],
)
def test_only_selected_modeling_files_are_generated(tmp_path, layout, expected):
    """Unused layout hooks must not distract operators or become accidental owners."""
    project = _generate_project(tmp_path, training_layout=layout)
    assert {path.name for path in (project / "src/modeling").glob("*.py")} == expected
    assert not (project / "selection").exists()
    assert _read_validated_config(project, action="train")["training_layout"] == layout


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_unused_optional_training_settings_are_absent(tmp_path, layout):
    """Date-free random training keeps defaults without irrelevant null placeholders."""
    from skyulf.integrations.databricks.lifecycle.local_workflow import (
        training_settings,
        training_spec,
    )
    from skyulf.integrations.databricks.training.tuning.local_cv import LocalCVSpec

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
    assert LocalCVSpec.from_workflow(config) == LocalCVSpec.from_workflow(legacy)


@pytest.mark.parametrize("strategy", ["random", "grid", "optuna"])
def test_unused_tuning_timeout_is_absent(tmp_path, strategy):
    """Default tuning must not expose a timeout setting with no selected value."""
    project = _generate_project(tmp_path, search_strategy=strategy)
    module = runpy.run_path(str(project / "src/modeling/single_model.py"))
    assert "timeout" not in module["build_modeling"]()


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
    config = json.loads((project / "config/workflow.json").read_text())
    module = runpy.run_path(str(project / "src/modeling/single_model.py"))
    assert config["lookback_days"] == 45
    assert config["holdout_days"] == 7
    assert config["event_column"] == "event_time"
    assert module["WEIGHT_COLUMN"] is None
    assert module["build_modeling"]()["timeout"] == 30
    assert module["build_modeling"]()["base_model"]["params"]["max_iter"] is None


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
