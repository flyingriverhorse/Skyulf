"""Classification reports must respect the trained class axis on sparse holdouts."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from skyulf.modeling._evaluation.classification import evaluate_classification_model


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
