"""Tests targeting gap lines in skyulf.modeling.classification (CalibratedClassifier resolver)."""

import numpy as np
import pandas as pd
import pytest

from skyulf.modeling.classification import CalibratedClassifierCalculator


@pytest.fixture
def clf_data():
    """Small deterministic binary classification dataset."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 60), "f2": rng.normal(0, 1, 60)})
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))
    return X, y


def test_resolve_base_estimator_with_falsy_config_returns_empty_dict():
    """A None/empty config should short-circuit to an empty dict without resolving anything."""
    result = CalibratedClassifierCalculator._resolve_base_estimator(None)
    assert result == {}
    result_empty = CalibratedClassifierCalculator._resolve_base_estimator({})
    assert result_empty == {}


def test_resolve_base_estimator_flat_config_injects_estimator_instance():
    """A flat config with 'base_estimator' key should resolve to an 'estimator' instance."""
    resolved = CalibratedClassifierCalculator._resolve_base_estimator(
        {"base_estimator": "random_forest"}
    )
    assert "estimator" in resolved
    assert resolved["estimator"].__class__.__name__ == "RandomForestClassifier"


def test_resolve_base_estimator_nested_params_config_injects_estimator_instance():
    """A nested {'params': {...}} config should have 'estimator' injected inside 'params'."""
    resolved = CalibratedClassifierCalculator._resolve_base_estimator(
        {"params": {"base_estimator": "decision_tree"}, "type": "calibrated_classifier"}
    )
    assert "params" in resolved
    assert "estimator" in resolved["params"]
    assert resolved["params"]["estimator"].__class__.__name__ == "DecisionTreeClassifier"


def test_resolve_base_estimator_unknown_key_falls_back_to_logistic_regression(caplog):
    """An unrecognised base_estimator key should fall back to logistic_regression with a warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="skyulf.modeling.classification"):
        resolved = CalibratedClassifierCalculator._resolve_base_estimator(
            {"base_estimator": "not_a_real_estimator"}
        )
    assert resolved["estimator"].__class__.__name__ == "LogisticRegression"
    assert any("Unknown base_estimator" in r.message for r in caplog.records)


def test_calibrated_classifier_fit_predict_round_trip(clf_data):
    """End-to-end: fitting CalibratedClassifierCalculator should produce a working model."""
    from skyulf.modeling.classification import CalibratedClassifierApplier

    X, y = clf_data
    calc = CalibratedClassifierCalculator()
    model = calc.fit(X, y, {"base_estimator": "logistic_regression", "cv": 3})
    preds = CalibratedClassifierApplier().predict(X, model)
    assert len(preds) == len(y)
    assert set(preds.unique()).issubset({0, 1})
