"""Tests for skyulf.modeling._evaluation.metrics — verified against sklearn/hand-computed values."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from skyulf.modeling._evaluation import metrics as metrics_mod
from skyulf.modeling._evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
)


@pytest.fixture
def binary_data():
    """Small, deterministic binary-classification data with a fitted LogisticRegression."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {"f1": rng.normal(0, 1, 60), "f2": rng.normal(0, 1, 60)},
    )
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int), name="target")
    model = LogisticRegression().fit(X, y)
    return model, X, y


@pytest.fixture
def multiclass_data():
    """Small deterministic 3-class dataset with a fitted DecisionTreeClassifier."""
    rng = np.random.RandomState(1)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 90), "f2": rng.normal(0, 1, 90)})
    y = pd.Series(np.tile([0, 1, 2], 30), name="target")
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    return model, X, y


def test_classification_metrics_accuracy_matches_sklearn(binary_data):
    """accuracy metric must equal sklearn's accuracy_score on the same predictions."""
    model, X, y = binary_data
    metrics = calculate_classification_metrics(model, X, y)
    expected = accuracy_score(y, model.predict(X))
    assert metrics["accuracy"] == pytest.approx(expected)


def test_classification_metrics_f1_weighted_matches_sklearn(binary_data):
    """f1_weighted metric must equal sklearn's weighted f1_score."""
    model, X, y = binary_data
    metrics = calculate_classification_metrics(model, X, y)
    expected = f1_score(y, model.predict(X), average="weighted", zero_division=0)
    assert metrics["f1_weighted"] == pytest.approx(expected)


def test_classification_metrics_binary_adds_unweighted_variants(binary_data):
    """Binary classification should include unweighted precision/recall/f1 keys."""
    model, X, y = binary_data
    metrics = calculate_classification_metrics(model, X, y)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


def test_classification_metrics_binary_non_01_string_labels_still_produce_precision_recall_f1():
    """Regression test: binary labels that aren't literally {0, 1} (e.g. string
    labels) must not silently drop precision/recall/f1. Previously
    average="binary" relied on sklearn's default pos_label=1, which raises for
    non-{0,1} labels — swallowed by a bare `except Exception: pass`."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 60), "f2": rng.normal(0, 1, 60)})
    y = pd.Series(np.where(X["f1"] + X["f2"] > 0, "yes", "no"), name="target")
    model = LogisticRegression().fit(X, y)

    metrics = calculate_classification_metrics(model, X, y)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    predictions = model.predict(X)
    pos_label = model.classes_[1]
    expected_f1 = f1_score(y, predictions, average="binary", pos_label=pos_label)
    assert metrics["f1"] == pytest.approx(expected_f1)


def test_classification_metrics_binary_negative_positive_int_labels():
    """Non-{0,1} integer binary labels (e.g. {-1, 1}) must also produce
    precision/recall/f1 using the correct positive-class label."""
    rng = np.random.RandomState(2)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 60), "f2": rng.normal(0, 1, 60)})
    y = pd.Series(np.where(X["f1"] + X["f2"] > 0, 1, -1), name="target")
    model = LogisticRegression().fit(X, y)

    metrics = calculate_classification_metrics(model, X, y)
    predictions = model.predict(X)
    pos_label = model.classes_[1]
    expected_precision = f1_score(y, predictions, average="binary", pos_label=pos_label)
    assert metrics["f1"] == pytest.approx(expected_precision)


def test_classification_metrics_binary_roc_auc_matches_sklearn(binary_data):
    """roc_auc for binary classification should match sklearn's roc_auc_score."""
    model, X, y = binary_data
    metrics = calculate_classification_metrics(model, X, y)
    proba = model.predict_proba(X)
    expected = roc_auc_score(y, proba[:, 1])
    assert metrics["roc_auc"] == pytest.approx(expected)


def test_classification_metrics_multiclass_has_ovr_and_ovo_variants(multiclass_data):
    """Multiclass predictions should produce OVR/OVO roc_auc variants, not binary keys."""
    model, X, y = multiclass_data
    metrics = calculate_classification_metrics(model, X, y)
    assert "roc_auc_ovr" in metrics
    assert "roc_auc_ovo" in metrics
    assert "roc_auc_ovr_weighted" in metrics
    assert "roc_auc_ovo_weighted" in metrics
    assert "roc_auc" not in metrics


def test_classification_metrics_multiclass_survives_fold_missing_a_trained_class(multiclass_data):
    """Regression test: multiclass roc_auc/pr_auc must not silently disappear
    when the evaluated split doesn't contain every class the model was
    trained on (common with small/imbalanced CV folds). Previously
    roc_auc_score raised "Number of classes in y_true not equal to the number
    of columns in y_score", swallowed by a bare except."""
    model, X, y = multiclass_data
    # Evaluate on a subset containing only 2 of the 3 trained classes.
    mask = y != 2
    X_subset, y_subset = X[mask], y[mask]

    metrics = calculate_classification_metrics(model, X_subset, y_subset)
    assert "roc_auc_ovr" in metrics
    assert "roc_auc_ovo" in metrics
    assert "pr_auc_weighted" in metrics


def test_classification_metrics_multiclass_pr_auc_weighted_present(multiclass_data):
    """Multiclass predictions should include a weighted PR-AUC computed via label_binarize."""
    model, X, y = multiclass_data
    metrics = calculate_classification_metrics(model, X, y)
    assert "pr_auc_weighted" in metrics
    assert 0.0 <= metrics["pr_auc_weighted"] <= 1.0


def test_classification_metrics_matthews_corrcoef_bounded(binary_data):
    """matthews_corrcoef should be within its valid [-1, 1] range."""
    model, X, y = binary_data
    metrics = calculate_classification_metrics(model, X, y)
    assert -1.0 <= metrics["matthews_corrcoef"] <= 1.0


def test_classification_metrics_no_predict_proba_skips_probability_metrics():
    """Models without predict_proba should not produce log_loss/roc_auc keys."""
    from sklearn.svm import LinearSVC

    rng = np.random.RandomState(2)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 40), "f2": rng.normal(0, 1, 40)})
    y = pd.Series((X["f1"] > 0).astype(int))
    model = LinearSVC().fit(X, y)
    metrics = calculate_classification_metrics(model, X, y)
    assert "log_loss" not in metrics
    assert "roc_auc" not in metrics
    assert "accuracy" in metrics


def test_regression_metrics_mae_matches_sklearn():
    """mae metric must equal sklearn's mean_absolute_error."""
    rng = np.random.RandomState(3)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 50)})
    y = pd.Series(2 * X["f1"] + 1 + rng.normal(0, 0.1, 50))
    model = LinearRegression().fit(X, y)
    metrics = calculate_regression_metrics(model, X, y)
    expected = mean_absolute_error(y, model.predict(X))
    assert metrics["mae"] == pytest.approx(expected)


def test_regression_metrics_rmse_is_sqrt_of_mse():
    """rmse must equal the square root of mse for consistency."""
    rng = np.random.RandomState(4)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 50)})
    y = pd.Series(3 * X["f1"] - 2 + rng.normal(0, 0.1, 50))
    model = LinearRegression().fit(X, y)
    metrics = calculate_regression_metrics(model, X, y)
    assert metrics["rmse"] == pytest.approx(metrics["mse"] ** 0.5)


def test_regression_metrics_r2_matches_sklearn():
    """r2 metric must equal sklearn's r2_score."""
    rng = np.random.RandomState(5)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 50)})
    y = pd.Series(X["f1"] * 5 + rng.normal(0, 0.2, 50))
    model = LinearRegression().fit(X, y)
    metrics = calculate_regression_metrics(model, X, y)
    expected = r2_score(y, model.predict(X))
    assert metrics["r2"] == pytest.approx(expected)


def test_regression_metrics_mse_matches_sklearn():
    """mse metric must equal sklearn's mean_squared_error."""
    rng = np.random.RandomState(6)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 30)})
    y = pd.Series(X["f1"] + rng.normal(0, 0.05, 30))
    model = LinearRegression().fit(X, y)
    metrics = calculate_regression_metrics(model, X, y)
    expected = mean_squared_error(y, model.predict(X))
    assert metrics["mse"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Exception-swallowing branches (all guarded blocks must not propagate errors)
# ---------------------------------------------------------------------------


def test_binary_unweighted_metrics_exception_is_swallowed(monkeypatch, binary_data):
    """If precision_score raises for the binary block, precision/recall/f1 must be absent."""
    model, X, y = binary_data
    original = metrics_mod.precision_score

    def flaky(y_true, y_pred, average="binary", **kwargs):
        if average == "binary":
            raise ValueError("boom")
        return original(y_true, y_pred, average=average, **kwargs)

    monkeypatch.setattr(metrics_mod, "precision_score", flaky)
    result = calculate_classification_metrics(model, X, y)
    assert "precision" not in result
    assert "precision_weighted" in result


def test_geometric_mean_score_exception_is_swallowed(monkeypatch, binary_data):
    """If geometric_mean_score raises, g_score must simply be absent from the result."""
    model, X, y = binary_data

    def flaky(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(metrics_mod, "geometric_mean_score", flaky)
    result = calculate_classification_metrics(model, X, y)
    assert "g_score" not in result
    assert "accuracy" in result


def test_log_loss_exception_is_swallowed(monkeypatch, binary_data):
    """If log_loss raises, log_loss must be absent but roc_auc/pr_auc still computed."""
    model, X, y = binary_data

    def flaky(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(metrics_mod, "log_loss", flaky)
    result = calculate_classification_metrics(model, X, y)
    assert "log_loss" not in result
    assert "roc_auc" in result


def test_multiclass_pr_auc_weighted_exception_is_swallowed(monkeypatch, multiclass_data):
    """If roc_auc_score raises inside the multiclass block, roc_auc_ovo must be absent."""
    model, X, y = multiclass_data

    def flaky(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(metrics_mod, "roc_auc_score", flaky)
    result = calculate_classification_metrics(model, X, y)
    assert "roc_auc_ovo" not in result
    assert "accuracy" in result


def test_predict_proba_outer_exception_is_swallowed(monkeypatch, binary_data):
    """If predict_proba itself raises, the whole probability block must be skipped safely."""
    model, X, y = binary_data

    class _BrokenProbaModel:
        def __init__(self, inner):
            self._inner = inner

        def predict(self, X):
            return self._inner.predict(X)

        def predict_proba(self, X):
            raise RuntimeError("boom")

    broken = _BrokenProbaModel(model)
    result = calculate_classification_metrics(broken, X, y)
    assert "roc_auc" not in result
    assert "accuracy" in result


def test_multiclass_classes_fallback_when_model_lacks_classes_attr(monkeypatch, multiclass_data):
    """When model.classes_ is missing/mismatched, classes must fall back to np.arange."""
    model, X, y = multiclass_data

    class _NoClassesModel:
        def __init__(self, inner):
            self._inner = inner

        def predict(self, X):
            return self._inner.predict(X)

        def predict_proba(self, X):
            return self._inner.predict_proba(X)

    wrapped = _NoClassesModel(model)
    result = calculate_classification_metrics(wrapped, X, y)
    assert "pr_auc_weighted" in result


def test_imblearn_import_failure_leaves_geometric_mean_score_none(monkeypatch):
    """If imblearn.metrics is unimportable, geometric_mean_score must fall back to None."""
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "imblearn.metrics", None)
    try:
        importlib.reload(metrics_mod)
        assert metrics_mod.geometric_mean_score is None
    finally:
        monkeypatch.delitem(sys.modules, "imblearn.metrics", raising=False)
        importlib.reload(metrics_mod)


def test_regression_metrics_returns_all_expected_keys():
    """calculate_regression_metrics should always return a fixed set of keys."""
    rng = np.random.RandomState(7)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 20)})
    y = pd.Series(X["f1"] + 1)
    model = LinearRegression().fit(X, y)
    metrics = calculate_regression_metrics(model, X, y)
    assert set(metrics.keys()) == {"mae", "mse", "rmse", "r2", "mape", "explained_variance"}
