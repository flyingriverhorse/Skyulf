"""Pin deterministic explanations and native estimator weight semantics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.utils.class_weight import compute_sample_weight

from skyulf.modeling.classification import LogisticRegressionCalculator


@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_generic_shap_repeats_for_same_fitted_model(kind):
    """Permutation explanations must not depend on unrelated NumPy random draws."""
    pytest.importorskip("shap")
    from skyulf.modeling._explainability import compute_shap_explanation

    rng = np.random.default_rng(143)
    X = pd.DataFrame(rng.normal(size=(10, 12)), columns=[f"x{i}" for i in range(12)])
    y = (X.x0 * X.x1 + X.x2).to_numpy()
    model = GaussianNB().fit(X, y > np.median(y)) if kind == "classification" else SVR().fit(X, y)
    state = np.random.get_state()
    try:
        np.random.seed(5)
        first = compute_shap_explanation(model, X, max_samples=5)
        np.random.seed(918)
        second = compute_shap_explanation(model, X, max_samples=5)
    finally:
        np.random.set_state(state)
    assert first is not None and second is not None
    assert first["samples"]
    assert first == second


@pytest.mark.parametrize("labels", [[0] * 4 + [1] * 2, ["a"] * 4 + ["b"] * 2])
def test_balanced_logistic_keeps_native_weighted_class_frequency(labels):
    """Native balancing must not be replaced by a count-only external multiplier."""
    X = pd.DataFrame({"x": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]})
    y = np.asarray(labels)
    weights = np.array([1.0, 2.0, 0.0, 3.0, 7.0, 1.0])
    expected_weights = weights.copy()
    params = {"class_weight": "balanced", "max_iter": 1000, "random_state": 42}
    actual = LogisticRegressionCalculator().fit(X, y, {"params": params}, sample_weight=weights)
    native = LogisticRegression(**params).fit(X.to_numpy(), y, sample_weight=weights)
    counted = LogisticRegression(max_iter=1000, random_state=42).fit(
        X.to_numpy(), y, sample_weight=weights * compute_sample_weight("balanced", y)
    )
    np.testing.assert_allclose(
        actual.predict_proba(X.to_numpy()), native.predict_proba(X.to_numpy())
    )
    np.testing.assert_array_equal(weights, expected_weights)
    assert not np.allclose(actual.coef_, counted.coef_)
