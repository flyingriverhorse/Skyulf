"""Tests for skyulf.modeling.naive_bayes (MultinomialNB / BernoulliNB nodes)."""

import numpy as np
import pandas as pd

from skyulf.modeling.naive_bayes import (
    BernoulliNBApplier,
    BernoulliNBCalculator,
    MultinomialNBApplier,
    MultinomialNBCalculator,
)
from skyulf.registry import NodeRegistry


def _count_data():
    """Small deterministic non-negative count matrix suitable for MultinomialNB/BernoulliNB."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randint(0, 5, size=(40, 3)), columns=["t1", "t2", "t3"])  # ty: ignore[invalid-argument-type]
    y = pd.Series(rng.randint(0, 2, size=40))
    return X, y


def test_multinomial_nb_registered_in_node_registry():
    """multinomial_nb should be registered with its calculator/applier pair."""
    assert NodeRegistry.get_calculator("multinomial_nb") is MultinomialNBCalculator
    assert NodeRegistry.get_applier("multinomial_nb") is MultinomialNBApplier


def test_bernoulli_nb_registered_in_node_registry():
    """bernoulli_nb should be registered with its calculator/applier pair."""
    assert NodeRegistry.get_calculator("bernoulli_nb") is BernoulliNBCalculator
    assert NodeRegistry.get_applier("bernoulli_nb") is BernoulliNBApplier


def test_multinomial_nb_problem_type_is_classification():
    """MultinomialNBCalculator.problem_type should report 'classification'."""
    assert MultinomialNBCalculator().problem_type == "classification"


def test_bernoulli_nb_problem_type_is_classification():
    """BernoulliNBCalculator.problem_type should report 'classification'."""
    assert BernoulliNBCalculator().problem_type == "classification"


def test_multinomial_nb_fit_predict_round_trip():
    """MultinomialNBCalculator should fit on counts and predict binary labels."""
    X, y = _count_data()
    model = MultinomialNBCalculator().fit(X, y, {})
    preds = MultinomialNBApplier().predict(X, model)
    assert len(preds) == len(y)
    assert set(preds.unique()).issubset({0, 1})


def test_bernoulli_nb_fit_predict_round_trip():
    """BernoulliNBCalculator should fit on counts/binary features and predict labels."""
    X, y = _count_data()
    model = BernoulliNBCalculator().fit(X, y, {"alpha": 1.0, "binarize": 0.5})
    preds = BernoulliNBApplier().predict(X, model)
    assert len(preds) == len(y)
    assert set(preds.unique()).issubset({0, 1})


def test_multinomial_nb_default_params():
    """MultinomialNBCalculator's default_params should match the documented defaults."""
    calc = MultinomialNBCalculator()
    assert calc.default_params == {"alpha": 1.0, "fit_prior": True}


def test_bernoulli_nb_default_params():
    """BernoulliNBCalculator's default_params should match the documented defaults."""
    calc = BernoulliNBCalculator()
    assert calc.default_params == {"alpha": 1.0, "binarize": 0.0, "fit_prior": True}
