"""Tests targeting gap lines in skyulf.modeling.classification (CalibratedClassifier resolver)."""

import numpy as np
import pandas as pd
import pytest
from tests.utils.reload_guard import reload_module_preserving_registry

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


def test_prepare_tuning_params_captures_base_estimator_from_flat_config():
    """prepare_tuning_params must absorb the base_estimator key from a flat config."""
    calc = CalibratedClassifierCalculator()
    calc.prepare_tuning_params({"base_estimator": "random_forest", "method": "sigmoid"})
    assert calc._tuning_base_config == {"base_estimator": "random_forest"}


def test_prepare_tuning_params_captures_base_estimator_from_nested_config():
    """prepare_tuning_params must absorb the base_estimator key from a nested params config."""
    calc = CalibratedClassifierCalculator()
    calc.prepare_tuning_params({"params": {"base_estimator": "gradient_boosting"}})
    assert calc._tuning_base_config == {"base_estimator": "gradient_boosting"}


def test_prepare_tuning_params_ignores_non_structural_keys():
    """prepare_tuning_params must only capture STRUCTURAL_TUNING_KEYS, not other config keys."""
    calc = CalibratedClassifierCalculator()
    calc.prepare_tuning_params({"method": "isotonic", "cv": 5})
    assert calc._tuning_base_config == {}


def test_default_params_resolves_base_estimator_to_instance():
    """After prepare_tuning_params, default_params must expose the resolved estimator instance."""
    from sklearn.ensemble import RandomForestClassifier

    calc = CalibratedClassifierCalculator()
    calc.prepare_tuning_params({"base_estimator": "random_forest"})
    params = calc.default_params
    assert isinstance(params["estimator"], RandomForestClassifier)


def test_default_params_falls_back_to_logistic_regression_for_unknown_key():
    """An unknown base_estimator key must fall back to LogisticRegression in default_params."""
    from sklearn.linear_model import LogisticRegression

    calc = CalibratedClassifierCalculator()
    calc.prepare_tuning_params({"base_estimator": "not_a_real_estimator"})
    params = calc.default_params
    assert isinstance(params["estimator"], LogisticRegression)


def test_default_params_returns_default_when_no_tuning_base_config():
    """Without prepare_tuning_params, default_params must return the default LogisticRegression."""
    from sklearn.linear_model import LogisticRegression

    calc = CalibratedClassifierCalculator()
    params = calc.default_params
    assert isinstance(params["estimator"], LogisticRegression)


def test_default_params_does_not_mutate_internal_state():
    """default_params must return a copy; mutating it must not corrupt the calculator's state."""
    calc = CalibratedClassifierCalculator()
    params = calc.default_params
    params["estimator"] = "corrupted"
    assert calc.default_params["estimator"].__class__.__name__ == "LogisticRegression"


def test_silent_lgbm_logger_info_and_warning_are_no_ops():
    """_SilentLgbmLogger.info/.warning must be callable no-ops (silences native lgbm logs)."""
    from skyulf.modeling.classification import _SilentLgbmLogger

    logger_instance = _SilentLgbmLogger()
    assert logger_instance.info("some native message") is None
    assert logger_instance.warning("some native warning") is None


def test_classification_xgboost_import_failure_sets_flag_false(monkeypatch):
    """Simulating an unimportable xgboost must leave XGBOOST_AVAILABLE False after reload."""
    import skyulf.modeling.classification as clf_mod

    with reload_module_preserving_registry(clf_mod, monkeypatch, "xgboost") as mod:
        assert mod.XGBOOST_AVAILABLE is False
    assert clf_mod.XGBOOST_AVAILABLE is True


def test_classification_lightgbm_import_failure_sets_flag_false(monkeypatch):
    """Simulating an unimportable lightgbm must leave LIGHTGBM_AVAILABLE False after reload."""
    import skyulf.modeling.classification as clf_mod

    with reload_module_preserving_registry(clf_mod, monkeypatch, "lightgbm") as mod:
        assert mod.LIGHTGBM_AVAILABLE is False
    assert clf_mod.LIGHTGBM_AVAILABLE is True
