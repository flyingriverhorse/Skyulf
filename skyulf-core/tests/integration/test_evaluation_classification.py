"""Tests for skyulf.modeling._evaluation.classification (evaluate_classification_model)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.modeling._evaluation.classification import (
    _compute_confusion_matrix,
    evaluate_classification_model,
)
from skyulf.modeling._evaluation.schemas import ModelEvaluationReport


@pytest.fixture
def binary_fitted():
    """Deterministic binary classification model + held-out test data."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 80), "f2": rng.normal(0, 1, 80)})
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))
    model = LogisticRegression().fit(X, y)
    return model, X, y


@pytest.fixture
def multiclass_fitted():
    """Deterministic 3-class classification model + held-out test data."""
    rng = np.random.RandomState(1)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 90), "f2": rng.normal(0, 1, 90)})
    y = pd.Series(np.tile([0, 1, 2], 30))
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    return model, X, y


def test_evaluate_binary_returns_model_evaluation_report(binary_fitted):
    """The report should be a ModelEvaluationReport with classification populated."""
    model, X, y = binary_fitted
    report = evaluate_classification_model(model, X, y, dataset_name="test")
    assert isinstance(report, ModelEvaluationReport)
    assert report.dataset_name == "test"
    assert report.classification is not None
    assert report.regression is None


def test_evaluate_binary_confusion_matrix_shape(binary_fitted):
    """Binary classification confusion matrix should be a 2x2 matrix with matching labels."""
    model, X, y = binary_fitted
    report = evaluate_classification_model(model, X, y)
    assert report.classification is not None
    cm = report.classification.confusion_matrix
    assert cm is not None
    assert len(cm.labels) == 2
    assert len(cm.matrix) == 2
    assert all(len(row) == 2 for row in cm.matrix)


def test_evaluate_binary_confusion_matrix_total_equals_sample_count(binary_fitted):
    """Sum of all confusion-matrix cells should equal the number of test samples."""
    model, X, y = binary_fitted
    report = evaluate_classification_model(model, X, y)
    assert report.classification is not None
    cm = report.classification.confusion_matrix
    assert cm is not None
    total = sum(sum(row) for row in cm.matrix)
    assert total == len(y)


def test_evaluate_binary_produces_one_roc_and_pr_curve(binary_fitted):
    """Binary classification should produce exactly one ROC curve and one PR curve."""
    model, X, y = binary_fitted
    report = evaluate_classification_model(model, X, y)
    assert report.classification is not None
    assert len(report.classification.roc_curves) == 1
    assert len(report.classification.pr_curves) == 1


def test_evaluate_binary_roc_curve_auc_matches_metrics(binary_fitted):
    """The ROC curve's auc field should match the scalar roc_auc metric."""
    model, X, y = binary_fitted
    report = evaluate_classification_model(model, X, y)
    assert report.classification is not None
    roc_curve_auc = report.classification.roc_curves[0].auc
    assert roc_curve_auc == pytest.approx(report.metrics["roc_auc"])


def test_evaluate_multiclass_produces_curve_per_class(multiclass_fitted):
    """Multiclass classification should produce one ROC/PR curve per class."""
    model, X, y = multiclass_fitted
    report = evaluate_classification_model(model, X, y)
    assert report.classification is not None
    assert len(report.classification.roc_curves) == 3
    assert len(report.classification.pr_curves) == 3


def test_evaluate_multiclass_confusion_matrix_is_3x3(multiclass_fitted):
    """Multiclass confusion matrix should be 3x3 for a 3-class problem."""
    model, X, y = multiclass_fitted
    report = evaluate_classification_model(model, X, y)
    assert report.classification is not None
    cm = report.classification.confusion_matrix
    assert cm is not None
    assert len(cm.labels) == 3
    assert all(len(row) == 3 for row in cm.matrix)


def test_evaluate_model_without_predict_proba_has_no_curves():
    """A model lacking predict_proba should yield empty ROC/PR curve lists."""
    rng = np.random.RandomState(2)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 40), "f2": rng.normal(0, 1, 40)})
    y = pd.Series((X["f1"] > 0).astype(int))
    model = LinearSVC().fit(X, y)
    report = evaluate_classification_model(model, X, y)
    assert report.classification is not None
    assert report.classification.roc_curves == []
    assert report.classification.pr_curves == []


def test_evaluate_predict_proba_exception_is_swallowed():
    """If predict_proba raises, evaluation must proceed without probability curves."""
    rng = np.random.RandomState(2)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 40), "f2": rng.normal(0, 1, 40)})
    y = pd.Series((X["f1"] > 0).astype(int))
    base_model = LogisticRegression().fit(X, y)

    class _BrokenProbaModel:
        def __init__(self, inner):
            self._inner = inner
            self.classes_ = inner.classes_

        def predict(self, X):
            return self._inner.predict(X)

        def predict_proba(self, X):
            raise RuntimeError("boom")

    report = evaluate_classification_model(_BrokenProbaModel(base_model), X, y)
    assert report.classification is not None
    assert report.classification.roc_curves == []
    assert report.classification.pr_curves == []


def test_evaluate_classes_fallback_to_np_unique_when_model_has_no_classes_attr():
    """When the model exposes no classes_, classes must be derived from np.unique(y_test)."""
    rng = np.random.RandomState(2)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 40), "f2": rng.normal(0, 1, 40)})
    y = pd.Series((X["f1"] > 0).astype(int))
    base_model = LogisticRegression().fit(X, y)

    class _NoClassesModel:
        def __init__(self, inner):
            self._inner = inner

        def predict(self, X):
            return self._inner.predict(X)

    report = evaluate_classification_model(_NoClassesModel(base_model), X, y)
    assert report.classification is not None
    cm = report.classification.confusion_matrix
    assert cm is not None
    assert len(cm.labels) == 2


def test_evaluate_sanitizes_nan_metrics():
    """Any NaN/inf scalar metrics should be dropped from the final report."""
    rng = np.random.RandomState(3)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 40), "f2": rng.normal(0, 1, 40)})
    y = pd.Series((X["f1"] > 0).astype(int))
    model = LogisticRegression().fit(X, y)
    report = evaluate_classification_model(model, X, y)
    assert all(np.isfinite(v) for v in report.metrics.values())


def test_compute_confusion_matrix_direct():
    """_compute_confusion_matrix should build a ConfusionMatrixData matching sklearn's output."""
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    cm_data = _compute_confusion_matrix(y_true, y_pred, ["0", "1"], label_values=[0, 1])
    assert cm_data.labels == ["0", "1"]
    # 0 predicted as 0 once, as 1 once; 1 predicted as 1 twice, as 0 once.
    assert cm_data.matrix == [[1, 1], [1, 2]]


def test_compute_confusion_matrix_without_label_values_uses_labels_directly():
    """Without label_values, `labels` itself is used to match y_true/y_pred (legacy behavior)."""
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "a"]
    cm_data = _compute_confusion_matrix(y_true, y_pred, ["a", "b"])
    assert cm_data.matrix == [[1, 1], [1, 1]]


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing values — closer to production data than the small
    synthetic fixtures used elsewhere in this file.
    """

    def test_evaluate_churn_classifier_on_customers_data(self) -> None:
        df = load_sample_dataset("customers")
        # LogisticRegression can't handle NaN, so rows with missing
        # age/income are dropped rather than assumed clean.
        df = df.dropna(subset=["age", "income"])
        X = df[["age", "income"]]
        y = df["churned"]
        model = LogisticRegression().fit(X, y)

        report = evaluate_classification_model(model, X, y, dataset_name="customers")
        assert isinstance(report, ModelEvaluationReport)
        assert report.classification is not None
        assert "accuracy" in report.metrics
        assert all(np.isfinite(v) for v in report.metrics.values())
