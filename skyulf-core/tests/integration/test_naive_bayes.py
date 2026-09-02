"""Integration tests for the MultinomialNB and BernoulliNB modeling nodes.

Covers the remaining Naive Bayes gap classes (MultinomialNBCalculator/Applier,
BernoulliNBCalculator/Applier) end-to-end through StatefulEstimator:
fit -> predict -> evaluate, plus parameter propagation and registry wiring.

Unlike test_modeling_all.py, the fixture uses non-negative Poisson count
features because MultinomialNB cannot be fit on negative values (see
skyulf.modeling.naive_bayes).
"""

from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.naive_bayes import (
    BernoulliNBApplier,
    BernoulliNBCalculator,
    MultinomialNBApplier,
    MultinomialNBCalculator,
)
from skyulf.registry import NodeRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def counts_dataset() -> SplitDataset:
    """Binary classification on non-negative Poisson count features.

    Per-class Poisson rate matrices give the classes a learnable structure
    while guaranteeing non-negative integers, as required by MultinomialNB.
    """
    rng = np.random.RandomState(0)
    n = 200
    y = rng.randint(0, 2, n)
    rates = np.where(y[:, None] == 0, [[2.0, 1.0, 3.0]], [[4.0, 0.5, 1.5]])
    X = rng.poisson(rates)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])  # ty: ignore[invalid-argument-type]
    df["target"] = y
    return SplitDataset(train=df.iloc[:160], test=df.iloc[160:], validation=None)


def _build_estimator(
    calculator: MultinomialNBCalculator | BernoulliNBCalculator,
    applier: MultinomialNBApplier | BernoulliNBApplier,
) -> StatefulEstimator:
    """Pair a calculator/applier into a StatefulEstimator keyed by node id."""
    node_id = (
        "multinomial_nb" if isinstance(calculator, MultinomialNBCalculator) else "bernoulli_nb"
    )
    return StatefulEstimator(node_id=node_id, calculator=calculator, applier=applier)


# ===========================================================================
# MULTINOMIAL NB
# ===========================================================================


class TestMultinomialNB:
    def test_fit_predict(self, counts_dataset: SplitDataset) -> None:
        estimator = _build_estimator(MultinomialNBCalculator(), MultinomialNBApplier())
        preds = estimator.fit_predict(counts_dataset, target_column="target", config={})

        assert len(preds["train"]) == 160
        assert len(preds["test"]) == 40
        assert set(preds["test"].unique()) <= {0, 1}
        assert isinstance(estimator.model, MultinomialNB)

    def test_evaluate_reports_classification_metrics(self, counts_dataset: SplitDataset) -> None:
        estimator = _build_estimator(MultinomialNBCalculator(), MultinomialNBApplier())
        estimator.fit_predict(counts_dataset, target_column="target", config={})
        report = estimator.evaluate(counts_dataset, target_column="target")

        assert report["problem_type"] == "classification"
        test_report = report["splits"]["test"]
        assert test_report is not None
        assert "accuracy" in test_report.metrics
        assert 0.0 <= test_report.metrics["accuracy"] <= 1.0

    def test_predict_proba_rows_sum_to_one(self, counts_dataset: SplitDataset) -> None:
        estimator = _build_estimator(MultinomialNBCalculator(), MultinomialNBApplier())
        estimator.fit_predict(counts_dataset, target_column="target", config={})

        # The applier treats every input column as a feature, so the target
        # column must be stripped first (fit_predict does this internally).
        test_features = cast(pd.DataFrame, counts_dataset.test).drop(columns=["target"])
        proba = estimator.applier.predict_proba(test_features, estimator.model)

        assert proba is not None
        proba = cast(pd.DataFrame, proba)
        assert set(map(str, proba.columns)) == {"0", "1"}
        assert np.allclose(proba.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
        assert (proba.to_numpy() >= 0).all()

    def test_alpha_param_is_applied(self, counts_dataset: SplitDataset) -> None:
        estimator = _build_estimator(MultinomialNBCalculator(), MultinomialNBApplier())
        estimator.fit_predict(
            counts_dataset, target_column="target", config={"params": {"alpha": 0.25}}
        )

        assert cast(MultinomialNB, estimator.model).alpha == 0.25

    def test_fit_prior_false_uses_uniform_priors(self, counts_dataset: SplitDataset) -> None:
        """fit_prior=False must override empirical priors with a uniform one."""
        estimator = _build_estimator(MultinomialNBCalculator(), MultinomialNBApplier())
        estimator.fit_predict(
            counts_dataset, target_column="target", config={"params": {"fit_prior": False}}
        )

        # sklearn 1.8 exposes class_log_prior_ (log of the class priors).
        model = cast(MultinomialNB, estimator.model)
        assert np.allclose(model.class_log_prior_, np.log(np.full(2, 0.5)))

    def test_calculator_metadata(self) -> None:
        assert MultinomialNBCalculator().problem_type == "classification"
        assert MultinomialNBCalculator().default_params == {"alpha": 1.0, "fit_prior": True}


# ===========================================================================
# BERNOULLI NB
# ===========================================================================


class TestBernoulliNB:
    def test_fit_predict(self, counts_dataset: SplitDataset) -> None:
        estimator = _build_estimator(BernoulliNBCalculator(), BernoulliNBApplier())
        preds = estimator.fit_predict(counts_dataset, target_column="target", config={})

        assert len(preds["train"]) == 160
        assert len(preds["test"]) == 40
        assert set(preds["test"].unique()) <= {0, 1}
        assert isinstance(estimator.model, BernoulliNB)

    def test_evaluate_reports_classification_metrics(self, counts_dataset: SplitDataset) -> None:
        estimator = _build_estimator(BernoulliNBCalculator(), BernoulliNBApplier())
        estimator.fit_predict(counts_dataset, target_column="target", config={})
        report = estimator.evaluate(counts_dataset, target_column="target")

        assert report["problem_type"] == "classification"
        test_report = report["splits"]["test"]
        assert test_report is not None
        assert "accuracy" in test_report.metrics

    def test_binarize_param_is_applied(self, counts_dataset: SplitDataset) -> None:
        """BernoulliNB-specific: binarize threshold must reach the sklearn model."""
        estimator = _build_estimator(BernoulliNBCalculator(), BernoulliNBApplier())
        estimator.fit_predict(
            counts_dataset, target_column="target", config={"params": {"binarize": 0.5}}
        )

        assert cast(BernoulliNB, estimator.model).binarize == 0.5

    def test_alpha_param_is_applied(self, counts_dataset: SplitDataset) -> None:
        estimator = _build_estimator(BernoulliNBCalculator(), BernoulliNBApplier())
        estimator.fit_predict(
            counts_dataset, target_column="target", config={"params": {"alpha": 0.5}}
        )

        assert cast(BernoulliNB, estimator.model).alpha == 0.5

    def test_calculator_metadata(self) -> None:
        assert BernoulliNBCalculator().problem_type == "classification"
        assert BernoulliNBCalculator().default_params == {
            "alpha": 1.0,
            "binarize": 0.0,
            "fit_prior": True,
        }


# ===========================================================================
# REGISTRY WIRING
# ===========================================================================


class TestNaiveBayesRegistry:
    def test_registry_returns_concrete_classes(self) -> None:
        assert NodeRegistry.get_calculator("multinomial_nb") is MultinomialNBCalculator
        assert NodeRegistry.get_applier("multinomial_nb") is MultinomialNBApplier
        assert NodeRegistry.get_calculator("bernoulli_nb") is BernoulliNBCalculator
        assert NodeRegistry.get_applier("bernoulli_nb") is BernoulliNBApplier

    def test_nodes_are_registered_as_models(self) -> None:
        models = NodeRegistry.list_models()
        assert "multinomial_nb" in models
        assert "bernoulli_nb" in models

    def test_registry_metadata(self) -> None:
        metadata = NodeRegistry.get_all_metadata()

        assert metadata["multinomial_nb"]["category"] == "Modeling"
        assert metadata["bernoulli_nb"]["category"] == "Modeling"
        assert metadata["multinomial_nb"]["params"] == {"alpha": 1.0, "fit_prior": True}
