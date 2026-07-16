"""Unit tests for `skyulf.modeling._explainability.compute_shap_summary`."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from skyulf.modeling._explainability import compute_shap_summary


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


def test_random_forest_classifier_returns_non_negative_scores(classification_data):
    """Tree-based multi-class SHAP output is reduced to non-negative per-feature scores."""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)

    result = compute_shap_summary(model, X)

    assert result is not None
    assert set(result.keys()) == {"a", "b", "c"}
    assert all(v >= 0 for v in result.values())
    assert max(result.values()) > 0


def test_logistic_regression_returns_summary(classification_data):
    """Linear classifier SHAP output covers every input feature."""
    X, y = classification_data
    model = LogisticRegression().fit(X, y)

    result = compute_shap_summary(model, X)

    assert result is not None
    assert set(result.keys()) == {"a", "b", "c"}


def test_linear_regression_ranks_dominant_feature(regression_data):
    """For a regression target dominated by `a`, SHAP ranks `a` above `b` and `c`."""
    X, y = regression_data
    model = LinearRegression().fit(X, y)

    result = compute_shap_summary(model, X)

    assert result is not None
    assert result["a"] > result["b"] > result["c"] > 0


def test_returns_none_on_model_failure(classification_data):
    """Any exception during SHAP computation is swallowed and `None` is returned."""
    X, _ = classification_data

    class _Broken:
        def predict(self, X):
            raise RuntimeError("boom")

    result = compute_shap_summary(_Broken(), X)

    assert result is None


def test_returns_none_for_empty_dataframe():
    """An empty DataFrame yields `None` rather than raising."""
    empty = pd.DataFrame(columns=["a", "b"])

    result = compute_shap_summary(RandomForestClassifier(), empty)

    assert result is None


def test_caps_sample_size(classification_data):
    """Large input frames are sub-sampled to `max_samples` rows before explaining."""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

    result = compute_shap_summary(model, X, max_samples=10)

    assert result is not None
    assert set(result.keys()) == {"a", "b", "c"}


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

    result = compute_shap_summary(model, X)

    assert result is None
