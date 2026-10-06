"""Offline setup explains the actual Core pipeline without contacting Databricks."""

from copy import deepcopy

import pytest

from skyulf.integrations.databricks.projects.workflow_config import preview_workflow_config


def test_preview_preserves_step_order_and_separates_cv_from_holdout(workflow_config):
    """Operators must see the executed order and which data each evaluation uses."""
    workflow_config.update(cv_enabled=True, cv_folds=3, cv_type="k_fold")
    workflow_config["pipeline"]["preprocessing"] = [
        {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}},
        {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
    ]
    original = deepcopy(workflow_config)
    report = preview_workflow_config(workflow_config, action="train")
    assert report.index("1. fill") < report.index("2. scale")
    assert "k_fold, 3 folds" in report
    assert "training partition only" in report
    assert "Final holdout: temporal" in report
    assert "workspace.test.predictions" in report
    assert "No data read, training, registry mutation or deployment" in report
    assert workflow_config == original


def test_preview_does_not_call_a_draft_train_ready(workflow_config):
    """An incomplete fixed window must not be presented as ready for training."""
    workflow_config.update(
        training_window_mode="fixed_window",
        monthly_lookback_months=None,
        window_timezone=None,
        start=None,
    )
    report = preview_workflow_config(workflow_config)
    assert "Training: needs configuration" in report
    assert "start" in report
    with pytest.raises(ValueError, match="start"):
        preview_workflow_config(workflow_config, action="train")


def test_preview_describes_runtime_selection_when_unset(workflow_config):
    """Latest and rolling selections apply equally to manual and scheduled training."""
    workflow_config.update(training_version=None, result_cutoff=None)
    report = preview_workflow_config(workflow_config, action="train")
    assert "latest snapshot at invocation" in report
    assert "completed calendar months at invocation" in report
    assert "cutoff=invocation time" in report
    assert "last completed calendar month" in report
    assert workflow_config["holdout_start"] not in report
    assert workflow_config["start"] not in report


def test_preview_explains_holdout_lag_and_independent_job_clocks(workflow_config):
    """Preview must distinguish selected rows from scheduled triggers and queued work."""
    workflow_config.update(holdout_months=2, result_availability_lag_hours=48, result_cutoff=None)
    report = preview_workflow_config(workflow_config, action="train")
    assert "last 2 completed calendar months" in report
    assert "UTC minus 48 elapsed hours (inclusive)" in report
    assert "Bundle variables" in report
    assert "any cron frequency" in report
    assert "PAUSED" in report and "no-op" in report and "queue" in report


@pytest.mark.parametrize(
    "task,model", [("classification", "linear_regression"), ("regression", "logistic_regression")]
)
def test_preview_rejects_model_task_mismatch(workflow_config, task, model):
    """Selecting a registered ID must still enforce the declared learning task."""
    workflow_config.update(
        task=task,
        metric="heldout_accuracy" if task == "classification" else "heldout_rmse",
        quality_threshold=0.8,
    )
    workflow_config["pipeline"]["modeling"]["type"] = model
    with pytest.raises(ValueError, match="declared task"):
        preview_workflow_config(workflow_config)
