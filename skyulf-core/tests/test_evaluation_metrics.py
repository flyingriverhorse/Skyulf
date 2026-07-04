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


def test_regression_metrics_returns_all_expected_keys():
    """calculate_regression_metrics should always return a fixed set of keys."""
    rng = np.random.RandomState(7)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 20)})
    y = pd.Series(X["f1"] + 1)
    model = LinearRegression().fit(X, y)
    metrics = calculate_regression_metrics(model, X, y)
    assert set(metrics.keys()) == {"mae", "mse", "rmse", "r2", "mape", "explained_variance"}
