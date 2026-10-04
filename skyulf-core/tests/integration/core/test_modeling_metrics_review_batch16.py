"""Expose missing evaluation metrics and preserve complete nested CV coverage."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from skyulf.modeling._evaluation.classification import evaluate_classification_model
from skyulf.modeling.cross_validation import perform_cross_validation
from skyulf.registry import NodeRegistry


@pytest.mark.parametrize("all_classes", [False, True])
def test_evaluation_explains_undefined_macro_auc(all_classes):
    """A missing holdout class must have a serialized reason instead of a vanished macro AUC."""
    X = pd.DataFrame({"x": range(18)})
    y = np.repeat([0, 1, 2], 6)
    model = LogisticRegression().fit(X, y)
    rows = np.arange(18 if all_classes else 12)
    report = evaluate_classification_model(model, X.iloc[rows], y[rows])
    payload = json.loads(report.model_dump_json())
    if all_classes:
        assert "roc_auc_ovr" in payload["metrics"]
        assert "roc_auc_ovr" not in payload["omitted_metrics"]
    else:
        assert "roc_auc_ovr" not in payload["metrics"]
        assert payload["omitted_metrics"]["roc_auc_ovr"]
        assert "accuracy" in payload["metrics"]


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_fixed_nested_cv_has_full_metric_coverage(task):
    """Fixed nested evaluation must retain the metric family reported by ordinary CV."""
    X = pd.DataFrame({"x": np.arange(36), "z": np.arange(36) % 5})
    y = pd.Series(np.arange(36) % 2 if task == "classification" else np.arange(36) ** 2)
    family = "logistic_regression" if task == "classification" else "ridge_regression"
    report = perform_cross_validation(
        NodeRegistry.get_calculator(family)(),
        NodeRegistry.get_applier(family)(),
        X,
        y,
        {},
        n_folds=3,
        inner_folds=2,
        cv_type="nested_cv",
        cv_nested_type="k_fold",
        random_state=19,
    )
    required = (
        {"accuracy", "f1_weighted", "log_loss"}
        if task == "classification"
        else {"mse", "mae", "r2"}
    )
    assert required <= report["aggregated_metrics"].keys()
    for name in required:
        assert report["aggregated_metrics"][name]["valid_folds"] == 3
        assert report["aggregated_metrics"][name]["total_folds"] == 3
        assert all(name in fold["metrics"] for fold in report["folds"])
    assert report["fixed_parameters"] is True
    assert report["total_trials"] == 4


def test_cv_exposes_partial_auc_fold_coverage():
    """A one-class holdout cannot make a partial AUC look like a three-fold average."""
    X = pd.DataFrame({"x": np.arange(12), "z": np.arange(12) % 3})
    y = pd.Series([0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1])
    report = perform_cross_validation(
        NodeRegistry.get_calculator("logistic_regression")(),
        NodeRegistry.get_applier("logistic_regression")(),
        X,
        y,
        {},
        n_folds=3,
        shuffle=False,
    )
    auc = report["aggregated_metrics"]["roc_auc"]
    assert auc["valid_folds"] == 2
    assert auc["total_folds"] == 3
    assert report["aggregated_metrics"]["accuracy"]["valid_folds"] == 3
