"""Diagnostics reflect actual predictions, class order and estimator capabilities."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
from skyulf.integrations.databricks.observability.charts.evaluation_charts import (
    diagnostic_charts,
    model_charts,
)


def test_regression_charts_use_paired_holdout_errors():
    """Residual plots must show prediction minus observation without silently reordering rows."""
    figures, skipped = diagnostic_charts(
        "regression", [1, 3, 5], pd.DataFrame({"prediction": [2, 2, 6]}), (), 20
    )
    assert set(figures) == {"actual_vs_predicted", "residuals", "residual_distribution"}
    assert list(figures["residuals"].axes[0].collections[0].get_offsets()[:, 1]) == [1, -1, 1]
    assert not skipped


def test_binary_probability_curves_follow_manifest_positive_label():
    """Positive probability is the second manifest class, even with nonnumeric labels."""
    predictions = pd.DataFrame(
        {
            "prediction": ["yes", "no", "yes", "no"],
            "probability_0": [0.1, 0.8, 0.2, 0.9],
            "probability_1": [0.9, 0.2, 0.8, 0.1],
        }
    )
    figures, skipped = diagnostic_charts(
        "classification", ["yes", "no", "yes", "no"], predictions, ("no", "yes"), 20
    )
    assert {"confusion_matrix", "class_metrics", "roc_pr", "calibration"} == set(figures)
    assert "yes" in figures["roc_pr"].axes[0].get_legend_handles_labels()[1][0]
    assert not skipped


def test_probability_free_model_keeps_classification_diagnostics():
    """Unsupported probability output must not be fabricated or prevent confusion matrices."""
    figures, skipped = diagnostic_charts(
        "classification", [10, 30, 10], pd.DataFrame({"prediction": [10, 30, 30]}), (10, 30), 20
    )
    assert set(figures) == {"confusion_matrix", "class_metrics"}
    assert any("probabilit" in note for note in skipped)


def test_multiclass_missing_sample_class_is_disclosed():
    """A bounded sample can omit classes, so undefined ROC curves must be skipped explicitly."""
    predictions = pd.DataFrame(
        {
            "prediction": ["a", "b", "a", "b"],
            "probability_0": [0.8, 0.1, 0.7, 0.2],
            "probability_1": [0.1, 0.8, 0.2, 0.7],
            "probability_2": [0.1] * 4,
        }
    )
    figures, skipped = diagnostic_charts(
        "classification", ["a", "b", "a", "b"], predictions, ("a", "b", "c"), 20
    )
    assert "roc_pr" in figures
    assert any("c" in note and "absent" in note for note in skipped)


def test_importance_requires_exact_transformed_feature_alignment():
    """Tree importance cannot assign plausible names to mismatched feature vectors."""
    figures, skipped = model_charts(
        SimpleNamespace(feature_importances_=np.array([0.1, 0.9])), ("small", "large"), (), 20
    )
    assert set(figures) == {"feature_importance"}
    assert [
        label.get_text() for label in figures["feature_importance"].axes[0].get_yticklabels()
    ] == ["small", "large"]
    assert not skipped
    with pytest.raises(ValueError, match="feature"):
        model_charts(SimpleNamespace(feature_importances_=np.array([1.0])), ("a", "b"), (), 20)


def test_linear_coefficients_keep_sign_and_unsupported_models_are_explicit():
    """Signed linear effects are distinct from native tree importance, and no fake plot is made."""
    figures, _ = model_charts(SimpleNamespace(coef_=np.array([-3.0, 2.0])), ("a", "b"), (), 20)
    widths = [bar.get_width() for bar in figures["coefficients"].axes[0].patches]
    assert sorted(widths) == [-3.0, 2.0]
    figures, skipped = model_charts(SimpleNamespace(), ("a",), (), 20)
    assert not figures
    assert any("importance" in note for note in skipped)


@pytest.fixture(autouse=True)
def close_figures():
    """Release real Matplotlib figures between independent rendering tests."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.mark.parametrize("classes", [3, 4])
def test_linear_svc_coefficients_name_pairwise_boundaries(classes):
    """Multiclass SVC coefficients compare pairs, not individual one-versus-rest classes."""
    from sklearn.datasets import make_classification
    from sklearn.svm import SVC

    features, labels = make_classification(
        n_samples=60,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=classes,
        random_state=42,
    )
    model = SVC(kernel="linear").fit(features, labels)
    figures, skipped = model_charts(
        model, tuple(f"f{i}" for i in range(5)), tuple(model.classes_), 5
    )
    assert len(figures) == classes * (classes - 1) // 2
    assert all("versus" in figure._suptitle.get_text() for figure in figures.values())
    assert not skipped
