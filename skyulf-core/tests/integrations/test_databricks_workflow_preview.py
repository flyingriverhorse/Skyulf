"""Offline setup explains the actual Core pipeline without contacting Databricks."""

from copy import deepcopy

import pytest

from skyulf.integrations.databricks.workflow_config import preview_workflow_config


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
    """A structurally valid draft still needs a pinned version for manual training."""
    workflow_config["training_version"] = None
    report = preview_workflow_config(workflow_config)
    assert "Manual training: needs configuration" in report
    assert "training_version" in report
    with pytest.raises(ValueError, match="training_version"):
        preview_workflow_config(workflow_config, action="train")


def test_monthly_preview_describes_runtime_selection_instead_of_manual_pins(workflow_config):
    """A scheduled run must not appear to reuse stale manual dates or a pinned snapshot."""
    report = preview_workflow_config(workflow_config, action="train_monthly")
    assert "latest snapshot at invocation" in report
    assert "completed calendar months at invocation" in report
    assert "cutoff=invocation time" in report
    assert "last completed calendar month" in report
    assert workflow_config["holdout_start"] not in report
    assert workflow_config["start"] not in report


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
