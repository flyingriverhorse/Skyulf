"""Gap-closing tests for skyulf.modeling.ensemble (_BaseEnsembleCalculator internals)."""

import numpy as np
import pandas as pd
import pytest

from skyulf.modeling.ensemble import (
    StackingClassifierCalculator,
    VotingClassifierCalculator,
)


@pytest.fixture
def clf_data():
    """Small deterministic binary classification dataset."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 60), "f2": rng.normal(0, 1, 60)})
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))
    return X, y


def test_build_tuning_search_space_without_tune_base_models_returns_default():
    """When tune_base_models is falsy, only the ensemble's own default space is returned."""
    calc = VotingClassifierCalculator()
    space = calc.build_tuning_search_space({"params": {"tune_base_models": False}}, "random")
    assert "voting" in space
    assert not any("__" in k for k in space)


def test_build_tuning_search_space_with_tune_base_models_expands_nested_keys():
    """tune_base_models=True should expand nested <name>__<param> keys per base learner."""
    calc = VotingClassifierCalculator()
    space = calc.build_tuning_search_space(
        {
            "params": {
                "tune_base_models": True,
                "base_estimators": ["random_forest"],
            }
        },
        "random",
    )
    assert any(k.startswith("random_forest__") for k in space)


def test_inject_tuning_base_config_merges_into_flat_config():
    """_inject_tuning_base_config should restore captured base_estimators into a flat config."""
    calc = VotingClassifierCalculator()
    calc.prepare_tuning_params({"params": {"base_estimators": ["random_forest"]}})
    merged = calc._inject_tuning_base_config({"C": 1.0})
    assert merged["base_estimators"] == ["random_forest"]


def test_inject_tuning_base_config_merges_into_nested_config():
    """_inject_tuning_base_config should restore base_estimators inside a nested params dict."""
    calc = VotingClassifierCalculator()
    calc.prepare_tuning_params({"params": {"base_estimators": ["random_forest"]}})
    merged = calc._inject_tuning_base_config({"params": {"voting": "hard"}})
    assert merged["params"]["base_estimators"] == ["random_forest"]


def test_inject_tuning_base_config_noop_when_base_estimators_already_present():
    """If base_estimators is already in the config, the captured config should not overwrite it."""
    calc = VotingClassifierCalculator()
    calc.prepare_tuning_params({"params": {"base_estimators": ["random_forest"]}})
    config = {"params": {"base_estimators": ["decision_tree"]}}
    merged = calc._inject_tuning_base_config(config)
    assert merged is config


def test_build_estimators_falls_back_to_defaults_for_non_list_keys():
    """A non-list/tuple `keys` argument should fall back to DEFAULT_KEYS."""
    calc = VotingClassifierCalculator()
    estimators = calc._build_estimators("not_a_list")
    names = [name for name, _ in estimators]
    assert names == list(calc.DEFAULT_KEYS)


def test_apply_params_logs_warning_on_invalid_param(caplog):
    """Invalid sklearn params passed via set_params should log a warning, not raise."""
    import logging

    from sklearn.linear_model import LogisticRegression

    with caplog.at_level(logging.WARNING, logger="skyulf.modeling.ensemble"):
        result = VotingClassifierCalculator._apply_params(
            LogisticRegression(), {"totally_bogus_param": 1}, "logistic_regression"
        )
    assert isinstance(result, LogisticRegression)
    assert any("Invalid params" in r.message for r in caplog.records)


def test_resolve_final_estimator_unknown_key_falls_back_with_warning(caplog):
    """An unrecognised final_estimator key should fall back to the family default with a warning."""
    import logging

    calc = StackingClassifierCalculator()
    with caplog.at_level(logging.WARNING, logger="skyulf.modeling.ensemble"):
        estimator = calc._resolve_final_estimator("not_a_real_estimator")
    assert (
        estimator.__class__.__name__
        == calc.BASE_ESTIMATORS[calc.DEFAULT_FINAL_KEY]().__class__.__name__
    )
    assert any("Unknown final_estimator" in r.message for r in caplog.records)


def test_clean_meta_keys_drops_invalid_n_jobs():
    """A non-numeric n_jobs value should be dropped instead of crashing set_params."""
    calc = VotingClassifierCalculator()
    bucket = {"n_jobs": "not-a-number", "estimators": []}
    calc._clean_meta_keys(bucket)
    assert "n_jobs" not in bucket


def test_resolve_estimators_nested_config_returns_nested_shape():
    """_resolve_estimators on a nested {'params': {...}} config should preserve nesting."""
    calc = VotingClassifierCalculator()
    resolved = calc._resolve_estimators({"params": {"base_estimators": ["random_forest"]}})
    assert "params" in resolved
    assert "estimators" in resolved["params"]


def test_voting_classifier_fits_with_defaults_end_to_end(clf_data):
    """Full fit with default base learners should produce a working voting ensemble."""
    from skyulf.modeling.ensemble import VotingClassifierApplier

    X, y = clf_data
    calc = VotingClassifierCalculator()
    model = calc.fit(X, y, {})
    preds = VotingClassifierApplier().predict(X, model)
    assert len(preds) == len(y)
