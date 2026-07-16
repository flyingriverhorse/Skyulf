"""Unit tests for `skyulf.modeling._explainability.compute_shap_explanation`."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from skyulf.modeling._explainability import compute_shap_explanation


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Small deterministic binary-classification feature/target split."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.random(60),
            "b": rng.random(60),
            "c": rng.random(60),
        }
    )
    y = (X["a"] + X["b"] > 1).astype(int)
    return X, y


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Small deterministic regression feature/target split where `a` dominates."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.random(60),
            "b": rng.random(60),
            "c": rng.random(60),
        }
    )
    y = X["a"] * 3 + X["b"] * 0.1 + rng.random(60) * 0.01
    return X, y


@pytest.fixture
def multiclass_data() -> tuple[pd.DataFrame, pd.Series]:
    """Small deterministic 3-class classification feature/target split."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.random(60),
            "b": rng.random(60),
            "c": rng.random(60),
        }
    )
    y = pd.cut(X["a"] + X["b"] + X["c"], bins=3, labels=[0, 1, 2]).astype(int)
    return X, y


def test_random_forest_classifier_shape_and_scores(classification_data):
    """Tree-based binary-classification output has the expected top-level shape."""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)

    result = compute_shap_explanation(model, X, max_display_samples=5)

    assert result is not None
    assert set(result["feature_names"]) == {"a", "b", "c"}
    assert set(result["mean_abs_importance"].keys()) == {"a", "b", "c"}
    assert all(v >= 0 for v in result["mean_abs_importance"].values())
    assert max(result["mean_abs_importance"].values()) > 0
    assert len(result["samples"]) == 5
    sample = result["samples"][0]
    assert set(sample.keys()) == {"base_value", "feature_values", "shap_values"}
    assert set(sample["feature_values"].keys()) == {"a", "b", "c"}
    assert set(sample["shap_values"].keys()) == {"a", "b", "c"}


def test_random_forest_classifier_waterfall_reconstructs_predicted_proba(classification_data):
    """base_value + sum(shap_values) reconstructs the positive-class probability."""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)

    result = compute_shap_explanation(model, X, max_display_samples=5)

    assert result is not None
    for i, sample in enumerate(result["samples"]):
        reconstructed = sample["base_value"] + sum(sample["shap_values"].values())
        actual_proba = model.predict_proba(X.iloc[[i]])[0][1]
        assert reconstructed == pytest.approx(actual_proba, abs=1e-3)


def test_linear_regression_waterfall_reconstructs_prediction(regression_data):
    """base_value + sum(shap_values) reconstructs the regression prediction exactly."""
    X, y = regression_data
    model = LinearRegression().fit(X, y)

    result = compute_shap_explanation(model, X, max_display_samples=5)

    assert result is not None
    for i, sample in enumerate(result["samples"]):
        reconstructed = sample["base_value"] + sum(sample["shap_values"].values())
        actual = model.predict(X.iloc[[i]])[0]
        assert reconstructed == pytest.approx(actual, abs=1e-3)


def test_linear_regression_ranks_dominant_feature(regression_data):
    """For a regression target dominated by `a`, SHAP ranks `a` above `b` and `c`."""
    X, y = regression_data
    model = LinearRegression().fit(X, y)

    result = compute_shap_explanation(model, X)

    assert result is not None
    importance = result["mean_abs_importance"]
    assert importance["a"] > importance["b"] > importance["c"] > 0


def test_logistic_regression_returns_result(classification_data):
    """Linear classifier SHAP output covers every input feature."""
    X, y = classification_data
    model = LogisticRegression().fit(X, y)

    result = compute_shap_explanation(model, X, max_display_samples=3)

    assert result is not None
    assert set(result["mean_abs_importance"].keys()) == {"a", "b", "c"}
    assert len(result["samples"]) == 3


def test_multiclass_waterfall_reconstructs_predicted_class_proba(multiclass_data):
    """For 3+ classes, each row's base+shap reconstructs *its own predicted* class proba."""
    X, y = multiclass_data
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)

    result = compute_shap_explanation(model, X, max_display_samples=5)

    assert result is not None
    preds = model.predict(X.iloc[:5])
    probas = model.predict_proba(X.iloc[:5])
    classes = list(model.classes_)
    for i, sample in enumerate(result["samples"]):
        reconstructed = sample["base_value"] + sum(sample["shap_values"].values())
        actual = probas[i][classes.index(preds[i])]
        assert reconstructed == pytest.approx(actual, abs=1e-3)


def test_returns_none_on_model_failure(classification_data):
    """Any exception during SHAP computation is swallowed and `None` is returned."""
    X, _ = classification_data

    class _Broken:
        def predict(self, X):
            raise RuntimeError("boom")

    result = compute_shap_explanation(_Broken(), X)

    assert result is None


def test_returns_none_for_empty_dataframe():
    """An empty DataFrame yields `None` rather than raising."""
    empty = pd.DataFrame(columns=["a", "b"])

    result = compute_shap_explanation(RandomForestClassifier(), empty)

    assert result is None


def test_caps_computation_sample_size(classification_data):
    """Large input frames are sub-sampled to `max_samples` rows before explaining."""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

    result = compute_shap_explanation(model, X, max_samples=10, max_display_samples=10)

    assert result is not None
    assert len(result["samples"]) <= 10


def test_caps_display_sample_count(classification_data):
    """The `samples` list never exceeds `max_display_samples`, even with more rows available."""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

    result = compute_shap_explanation(model, X, max_display_samples=7)

    assert result is not None
    assert len(result["samples"]) == 7


def test_returns_none_when_shap_not_installed(classification_data, monkeypatch: pytest.MonkeyPatch):
    """If `shap` can't be imported, the function degrades to `None` instead of raising."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("no shap installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    X, y = classification_data
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    result = compute_shap_explanation(model, X)

    assert result is None
