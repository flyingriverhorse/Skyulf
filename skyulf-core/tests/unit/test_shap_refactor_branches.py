"""Public SHAP explanation contracts around retries and optional interactions."""

import logging
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from skyulf.modeling._explainability import compute_shap_explanation

shap = pytest.importorskip("shap")


@pytest.fixture
def regression_frame():
    """Use two distinct rows so explanation reconstruction has a known target."""
    return pd.DataFrame({"a": [1.0, 3.0], "b": [2.0, 4.0]})


@pytest.mark.parametrize("exceptions_available", [True, False])
def test_exact_tree_additivity_retry_preserves_explanation(
    regression_frame, monkeypatch, caplog, exceptions_available
):
    """A numerical additivity failure must retain usable feature contributions."""
    model = DecisionTreeRegressor(random_state=0).fit(regression_frame, [4.0, 8.0])
    original_call = shap.TreeExplainer.__call__
    error_type = shap.utils._exceptions.ExplainerError if exceptions_available else RuntimeError
    checks = []

    def fail_additivity_check(self, X, **kwargs):
        """Emulate a SHAP version rejecting only its redundant consistency check."""
        checks.append(kwargs.get("check_additivity", True))
        if checks[-1]:
            raise error_type("Additivity check failed for this tree")
        return original_call(self, X, **kwargs)

    monkeypatch.setattr(shap.TreeExplainer, "__call__", fail_additivity_check)
    if not exceptions_available:
        monkeypatch.setitem(sys.modules, "shap.utils._exceptions", None)

    with caplog.at_level(logging.WARNING):
        result = compute_shap_explanation(model, regression_frame)

    assert result is not None
    reconstructed = [
        row["base_value"] + sum(row["shap_values"].values()) for row in result["samples"]
    ]
    assert reconstructed == pytest.approx([4.0, 8.0])
    assert result["mean_abs_importance"]
    assert checks == [True, False]
    assert "retrying with check_additivity=False" in caplog.text
    assert "Failed to compute SHAP explanation" not in caplog.text


@pytest.mark.parametrize(
    ("tree_model", "exceptions_available", "message"),
    [
        (True, True, "additivity is not the cause: invalid input"),
        (True, False, "invalid input"),
        (False, True, "Additivity check failed"),
        (False, False, "Additivity check failed"),
    ],
    ids=["unrelated-typed-error", "unrelated-legacy-error", "generic-typed", "generic-legacy"],
)
def test_unsafe_explanation_failures_are_logged_without_retry(
    regression_frame, monkeypatch, caplog, tree_model, exceptions_available, message
):
    """Only exact-tree additivity errors may disable SHAP's consistency check."""
    model = DecisionTreeRegressor(random_state=0) if tree_model else LinearRegression()
    model.fit(regression_frame, [4.0, 8.0])
    error_type = shap.utils._exceptions.ExplainerError if not tree_model else ValueError
    checks = []

    def fail_explanation(self, X, **kwargs):
        """Record whether the integration incorrectly attempts an unsafe retry."""
        checks.append(kwargs.get("check_additivity", True))
        raise error_type(message)

    explainer_class = shap.TreeExplainer if tree_model else shap.LinearExplainer
    monkeypatch.setattr(explainer_class, "__call__", fail_explanation)
    if not exceptions_available:
        monkeypatch.setitem(sys.modules, "shap.utils._exceptions", None)

    with caplog.at_level(logging.WARNING):
        result = compute_shap_explanation(model, regression_frame)

    assert result is None
    assert checks == [True]
    assert message in caplog.text
    assert "Failed to compute SHAP explanation" in caplog.text
    assert "retrying with check_additivity=False" not in caplog.text


def test_failed_additivity_retry_remains_best_effort(regression_frame, monkeypatch, caplog):
    """A second explainer failure must not escape and interrupt model training."""
    model = DecisionTreeRegressor(random_state=0).fit(regression_frame, [4.0, 8.0])
    checks = []

    def fail_both_attempts(self, X, **kwargs):
        """Raise distinct errors so the final warning identifies the retry failure."""
        checks.append(kwargs.get("check_additivity", True))
        if checks[-1]:
            raise shap.utils._exceptions.ExplainerError("Additivity check failed")
        raise RuntimeError("tree explanation could not be computed")

    monkeypatch.setattr(shap.TreeExplainer, "__call__", fail_both_attempts)

    with caplog.at_level(logging.WARNING):
        result = compute_shap_explanation(model, regression_frame)

    assert result is None
    assert checks == [True, False]
    assert "retrying with check_additivity=False" in caplog.text
    assert "Failed to compute SHAP explanation" in caplog.text
    assert "tree explanation could not be computed" in caplog.text


@pytest.mark.parametrize(
    "shape",
    [(2, 1, 2, 2), (2, 2, 1, 2), (2, 1, 2), (2, 2, 1), (2, 2)],
    ids=[
        "class-first-feature-axis",
        "class-second-feature-axis",
        "first-feature-axis",
        "second-feature-axis",
        "missing-axis",
    ],
)
def test_malformed_interactions_preserve_main_explanation(regression_frame, monkeypatch, shape):
    """Malformed optional interactions must not discard valid sample explanations."""
    model = DecisionTreeRegressor(random_state=0).fit(regression_frame, [4.0, 8.0])
    monkeypatch.setattr(
        shap.TreeExplainer, "shap_interaction_values", lambda self, X: np.ones(shape)
    )

    result = compute_shap_explanation(model, regression_frame)

    assert result is not None
    assert result["interactions"] is None
    assert result["feature_names"] == ["a", "b"]
    reconstructed = [
        row["base_value"] + sum(row["shap_values"].values()) for row in result["samples"]
    ]
    assert reconstructed == pytest.approx([4.0, 8.0])


@pytest.mark.parametrize(
    ("classification", "expected_matrix"),
    [(False, [[2.0, 3.0], [3.0, 4.0]]), (True, [[4.0, 6.0], [6.0, 8.0]])],
)
def test_interactions_preserve_magnitude_across_rows_and_classes(
    regression_frame, monkeypatch, classification, expected_matrix
):
    """Opposite signs must not cancel feature interaction strength during averaging."""
    model_class = DecisionTreeClassifier if classification else DecisionTreeRegressor
    model = model_class(random_state=0).fit(regression_frame, [0, 1])
    raw = np.array([[[1.0, -2.0], [-2.0, 3.0]], [[-3.0, 4.0], [4.0, -5.0]]])
    if classification:
        raw = np.stack([raw, -3.0 * raw], axis=-1)
    monkeypatch.setattr(shap.TreeExplainer, "shap_interaction_values", lambda self, X: raw)

    result = compute_shap_explanation(model, regression_frame)

    assert result is not None
    assert result["interactions"] == {"feature_names": ["a", "b"], "matrix": expected_matrix}


def test_zero_display_limit_keeps_summary_and_interactions(regression_frame):
    """Disabling per-row display must still return global model explanations."""
    model = DecisionTreeRegressor(random_state=0).fit(regression_frame, [4.0, 8.0])

    result = compute_shap_explanation(model, regression_frame, max_display_samples=0)

    assert result is not None
    assert result["samples"] == []
    assert sum(result["mean_abs_importance"].values()) == pytest.approx(2.0)
    assert result["interactions"] is not None
