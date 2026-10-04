"""Classification reports must respect the trained class axis on sparse holdouts."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from skyulf.modeling._evaluation.classification import evaluate_classification_model
from skyulf.modeling._evaluation.metrics import calculate_classification_metrics
from skyulf.modeling.classification import (
    DecisionTreeClassifierApplier,
    DecisionTreeClassifierCalculator,
)
from skyulf.modeling.cross_validation import perform_cross_validation


@pytest.mark.parametrize("labels", [[1, 2], ["no", "yes"]])
@pytest.mark.parametrize("class_index", [0, 1])
def test_single_class_binary_holdout_omits_roc_and_preserves_pr(labels, class_index):
    """A missing ROC axis must not publish NaN coordinates or discard finite PR data."""
    X = pd.DataFrame({"x": np.arange(40, dtype=float)})
    y = pd.Series(np.repeat(labels, 20))
    model = LogisticRegression().fit(X, y)
    held_out = y == labels[class_index]

    report = evaluate_classification_model(model, X[held_out], y[held_out])

    assert "roc_auc" not in report.metrics
    assert report.classification is not None
    assert report.classification.roc_curves == []
    assert report.metrics["pr_auc"] == pytest.approx(float(class_index))
    assert len(report.classification.pr_curves) == 1
    curve = report.classification.pr_curves[0]
    assert curve.auc == pytest.approx(float(class_index))
    assert curve.points
    assert all(math.isfinite(point.x) and math.isfinite(point.y) for point in curve.points)


@pytest.mark.parametrize("labels", [[0, 1, 2], ["a", "b", "c"]])
@pytest.mark.parametrize("extra_in_predictions", [False, True])
def test_confusion_matrix_keeps_unseen_classes(labels, extra_in_predictions):
    """Every evaluated row must appear in the matrix, including unseen classes."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    model = DecisionTreeClassifier(random_state=0).fit(X, [labels[0]] * 2 + [labels[1]] * 2)
    y = pd.Series([labels[0], labels[1], labels[2], labels[2]])
    predictions = np.array([labels[0], labels[1], labels[0], labels[1]])
    if extra_in_predictions:
        y, predictions = pd.Series(predictions), y.to_numpy()

    report = evaluate_classification_model(model, X, y, predictions=predictions)

    assert report.classification is not None
    cm = report.classification.confusion_matrix
    assert cm is not None
    assert cm.labels == [str(label) for label in labels]
    assert np.sum(cm.matrix) == len(y)
    assert np.trace(cm.matrix) / len(y) == report.metrics["accuracy"] == 0.5
    assert [curve.name for curve in report.classification.pr_curves] == [f"PR (Class {labels[1]})"]


def test_single_class_multiclass_holdout_preserves_defined_metrics(caplog):
    """An undefined multiclass AUC must not abort valid holdout evaluation."""
    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = pd.Series(np.repeat([0, 1, 2], 4))
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    held_out = y == 0

    metrics = calculate_classification_metrics(model, X[held_out], y[held_out])
    report = evaluate_classification_model(model, X[held_out], y[held_out])

    assert metrics["accuracy"] == report.metrics["accuracy"] == 1.0
    assert "roc_auc_ovo_weighted" not in metrics
    assert "roc_auc_ovo_weighted" in caplog.text
    assert report.classification is not None
    assert report.classification.roc_curves == []
    assert report.classification.confusion_matrix is not None
    assert np.sum(report.classification.confusion_matrix.matrix) == 4
    assert all(math.isfinite(value) for value in report.metrics.values())


def test_unshuffled_cv_with_single_class_holdouts_completes():
    """One undefined probability metric must not discard entire CV folds."""
    y = pd.Series(np.tile(np.repeat([0, 1, 2], 4), 2))
    X = pd.DataFrame({"x": y.astype(float)})

    result = perform_cross_validation(
        DecisionTreeClassifierCalculator(),
        DecisionTreeClassifierApplier(),
        X,
        y,
        config={"random_state": 0},
        n_folds=6,
        shuffle=False,
    )

    assert len(result["folds"]) == 6
    assert result["aggregated_metrics"]["accuracy"]["mean"] == 1.0
