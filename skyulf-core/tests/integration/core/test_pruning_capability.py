"""Data-free capability checks must match the estimator Optuna actually receives."""

from copy import deepcopy

import pytest
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import SGDClassifierCalculator
from skyulf.modeling.sklearn_wrapper import SklearnCalculator


@pytest.mark.parametrize(
    ("model", "defaults", "space", "reason"),
    [
        (SGDClassifier, {"max_iter": 5}, {}, None),
        (MLPClassifier, {"max_iter": 5}, {}, None),
        (MLPClassifier, {}, {"solver": ["adam", "lbfgs"]}, "lbfgs"),
        (MLPClassifier, {}, {"solver": ["adam", "sgd"]}, None),
        (GaussianNB, {}, {}, "epoch budget"),
        (SGDClassifier, {"max_iter": 0}, {}, "epoch budget"),
        (SGDClassifier, {"max_iter": True}, {}, "epoch budget"),
        (SGDClassifier, {"early_stopping": True}, {}, "early_stopping"),
        (SGDClassifier, {"early_stopping": True}, {"early_stopping": [False]}, None),
        (SGDClassifier, {}, {"early_stopping": [False, True]}, "early_stopping"),
        (SGDClassifier, {"class_weight": "balanced"}, {}, "balanced"),
        (SGDClassifier, {}, {"class_weight": [None, "balanced"]}, "balanced"),
        (SGDClassifier, {}, {"max_iter": [5]}, "searched max_iter"),
        (MLPClassifier, {}, {"class_weight": [None]}, "Pipeline"),
    ],
)
def test_capability_uses_actual_defaults_and_outer_weight_wrapper(model, defaults, space, reason):
    """Defaults and routed nonnative weights must govern both the preview and runtime."""
    calculator = SklearnCalculator(model, defaults, "classification")
    config = TuningConfig(strategy="optuna", search_space=space)
    before = deepcopy((calculator.default_params, config))
    actual = TuningCalculator(calculator).pruning_support_reason(config)
    if reason is None:
        assert actual is None
    else:
        assert actual is not None and reason in actual
    assert (calculator.default_params, config) == before


@pytest.mark.parametrize("pruner", ["none", "median", "hyperband"])
def test_capability_is_independent_of_the_selected_pruner(pruner, monkeypatch):
    """A disabled selection must remain selectable again when the model supports pruning."""

    def fail_fit(*args, **kwargs):
        """Capability inspection must never execute training."""
        pytest.fail("capability check attempted fitting")

    monkeypatch.setattr(SGDClassifier, "fit", fail_fit)
    monkeypatch.setattr(SGDClassifier, "partial_fit", fail_fit)
    config = TuningConfig(strategy="optuna", strategy_params={"pruner": pruner})
    assert TuningCalculator(SGDClassifierCalculator()).pruning_support_reason(config) is None


def test_explicit_pruning_false_has_a_visible_reason():
    """Legacy opt-out flags must not advertise pruning that runtime would suppress."""
    config = TuningConfig(strategy="optuna", strategy_params={"pruning": False})
    reason = TuningCalculator(SGDClassifierCalculator()).pruning_support_reason(config)
    assert reason is not None and "pruning=False" in reason
