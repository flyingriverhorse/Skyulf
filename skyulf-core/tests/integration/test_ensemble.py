"""Integration tests for the ensemble meta-estimator nodes.

Covers VotingRegressor, StackingClassifier, and VotingClassifier end-to-end
through StatefulEstimator (fit -> predict -> proba), plus the structural
config resolution in ``_BaseEnsembleCalculator._resolve_estimators``:
nested ``<name>__<param>`` absorption, unknown-key skip/fallback, and the
voting/stacking meta-key cleanup that keeps cross-family params from leaking
into the sklearn constructors.

Config uses the nested ``{"params": {...}}`` shape (the model-training
payload) throughout, mirroring how the backend submits ensemble runs.
"""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import StackingClassifier, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression

from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.ensemble import (
    StackingClassifierApplier,
    StackingClassifierCalculator,
    StackingRegressorApplier,
    StackingRegressorCalculator,
    VotingClassifierApplier,
    VotingClassifierCalculator,
    VotingRegressorApplier,
    VotingRegressorCalculator,
)
from skyulf.registry import NodeRegistry

N_TRAIN = 240
N_TEST = 60


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clf_dataset() -> SplitDataset:
    """Binary classification on 5 Gaussian features with class-dependent means."""
    rng = np.random.RandomState(0)
    n = N_TRAIN + N_TEST
    y = rng.randint(0, 2, n)
    means = np.where(
        y[:, None] == 0,
        np.array([1.0, 0.0, 2.0, 0.5, 1.5]),
        np.array([0.0, 1.0, 1.0, 1.5, 0.5]),
    )
    X = means + rng.normal(0.0, 1.0, (n, 5))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return SplitDataset(train=df.iloc[:N_TRAIN], test=df.iloc[N_TRAIN:], validation=None)


@pytest.fixture
def reg_dataset() -> SplitDataset:
    """Regression on 5 uniform features with a linear ground-truth target."""
    rng = np.random.RandomState(0)
    n = N_TRAIN + N_TEST
    X = rng.uniform(-2.0, 2.0, (n, 5))
    weights = np.array([1.5, -1.0, 2.0, 0.5, -0.5])
    y = X @ weights + rng.normal(0.0, 0.5, n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return SplitDataset(train=df.iloc[:N_TRAIN], test=df.iloc[N_TRAIN:], validation=None)


def _build_estimator(calculator: Any, applier: Any, node_id: str) -> StatefulEstimator:
    """Pair a calculator/applier into a StatefulEstimator keyed by node id."""
    return StatefulEstimator(node_id=node_id, calculator=calculator, applier=applier)


# ===========================================================================
# VOTING REGRESSOR
# ===========================================================================


class TestVotingRegressor:
    def test_fit_predict_default_base_models(self, reg_dataset: SplitDataset) -> None:
        estimator = _build_estimator(
            VotingRegressorCalculator(), VotingRegressorApplier(), "voting_regressor"
        )
        preds = estimator.fit_predict(reg_dataset, target_column="target", config={})

        assert len(preds["train"]) == N_TRAIN
        assert len(preds["test"]) == N_TEST
        assert np.isfinite(preds["test"].to_numpy()).all()
        model = cast(VotingRegressor, estimator.model)
        base_names = list(model.named_estimators_.keys())
        assert base_names == ["linear_regression", "random_forest", "gradient_boosting"]

    def test_weights_dropped_when_length_mismatch(self, reg_dataset: SplitDataset) -> None:
        """A 2-entry weights list for 3 default bases must be dropped, not fatal."""
        estimator = _build_estimator(
            VotingRegressorCalculator(), VotingRegressorApplier(), "voting_regressor"
        )
        estimator.fit_predict(
            reg_dataset, target_column="target", config={"params": {"weights": [0.5, 0.5]}}
        )

        model = cast(VotingRegressor, estimator.model)
        assert model.weights is None
        assert len(model.estimators_) == 3

    def test_weights_kept_when_length_matches(self, reg_dataset: SplitDataset) -> None:
        estimator = _build_estimator(
            VotingRegressorCalculator(), VotingRegressorApplier(), "voting_regressor"
        )
        estimator.fit_predict(
            reg_dataset,
            target_column="target",
            config={"params": {"weights": [0.2, 0.3, 0.5]}},
        )

        model = cast(VotingRegressor, estimator.model)
        assert model.weights == [0.2, 0.3, 0.5]

    def test_unknown_base_estimators_fall_back_to_defaults(self, reg_dataset: SplitDataset) -> None:
        """An all-unknown selection must fall back to DEFAULT_KEYS, not fit empty."""
        estimator = _build_estimator(
            VotingRegressorCalculator(), VotingRegressorApplier(), "voting_regressor"
        )
        estimator.fit_predict(
            reg_dataset,
            target_column="target",
            config={"params": {"base_estimators": ["bogus_one", "bogus_two"]}},
        )

        model = cast(VotingRegressor, estimator.model)
        assert list(model.named_estimators_.keys()) == [
            "linear_regression",
            "random_forest",
            "gradient_boosting",
        ]

    def test_partial_unknown_base_estimators_skipped(self, reg_dataset: SplitDataset) -> None:
        estimator = _build_estimator(
            VotingRegressorCalculator(), VotingRegressorApplier(), "voting_regressor"
        )
        estimator.fit_predict(
            reg_dataset,
            target_column="target",
            config={"params": {"base_estimators": ["bogus", "linear_regression", "random_forest"]}},
        )

        model = cast(VotingRegressor, estimator.model)
        assert list(model.named_estimators_.keys()) == ["linear_regression", "random_forest"]

    def test_nested_base_estimator_params_absorbed(self, reg_dataset: SplitDataset) -> None:
        """Tuner-style ``random_forest__n_estimators`` keys reach the base model."""
        estimator = _build_estimator(
            VotingRegressorCalculator(), VotingRegressorApplier(), "voting_regressor"
        )
        estimator.fit_predict(
            reg_dataset,
            target_column="target",
            config={
                "params": {
                    "base_estimators": ["linear_regression", "random_forest"],
                    "random_forest__n_estimators": 17,
                }
            },
        )

        model = cast(VotingRegressor, estimator.model)
        random_forest = model.named_estimators_["random_forest"]
        assert cast(int, random_forest.get_params()["n_estimators"]) == 17


# ===========================================================================
# STACKING CLASSIFIER
# ===========================================================================


class TestStackingClassifier:
    def test_fit_predict_defaults(self, clf_dataset: SplitDataset) -> None:
        estimator = _build_estimator(
            StackingClassifierCalculator(), StackingClassifierApplier(), "stacking_classifier"
        )
        preds = estimator.fit_predict(clf_dataset, target_column="target", config={})

        assert len(preds["train"]) == N_TRAIN
        assert len(preds["test"]) == N_TEST
        assert set(preds["test"].unique()) <= {0, 1}
        model = cast(StackingClassifier, estimator.model)
        base_names = list(model.named_estimators_.keys())
        assert base_names == ["random_forest", "gradient_boosting", "svc"]
        assert model.cv == 5
        assert isinstance(model.final_estimator_, LogisticRegression)

    def test_predict_proba_rows_sum_to_one(self, clf_dataset: SplitDataset) -> None:
        estimator = _build_estimator(
            StackingClassifierCalculator(), StackingClassifierApplier(), "stacking_classifier"
        )
        estimator.fit_predict(clf_dataset, target_column="target", config={})

        test_features = cast(pd.DataFrame, clf_dataset.test).drop(columns=["target"])
        proba = estimator.applier.predict_proba(test_features, estimator.model)

        assert proba is not None
        assert np.allclose(proba.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
        assert (proba.to_numpy() >= 0).all()

    def test_unknown_final_estimator_falls_back_to_default(self, clf_dataset: SplitDataset) -> None:
        estimator = _build_estimator(
            StackingClassifierCalculator(), StackingClassifierApplier(), "stacking_classifier"
        )
        estimator.fit_predict(
            clf_dataset,
            target_column="target",
            config={"params": {"final_estimator": "bogus_final", "cv": 3}},
        )

        model = cast(StackingClassifier, estimator.model)
        assert isinstance(model.final_estimator_, LogisticRegression)
        assert model.cv == 3

    def test_voting_and_weights_cleaned_for_stacking(self, clf_dataset: SplitDataset) -> None:
        """Voting-family meta-keys must not alter stacking behavior."""
        estimator = _build_estimator(
            StackingClassifierCalculator(), StackingClassifierApplier(), "stacking_classifier"
        )
        estimator.fit_predict(
            clf_dataset,
            target_column="target",
            config={"params": {"voting": "soft", "weights": [1, 1, 1], "cv": 3}},
        )

        model = cast(StackingClassifier, estimator.model)
        # voting/weights are not StackingClassifier params; if they had leaked
        # into the constructor, sklearn would have rejected the kwargs.
        assert not hasattr(model, "voting")
        assert not hasattr(model, "weights")
        assert model.cv == 3
        assert isinstance(model.final_estimator_, LogisticRegression)

    def test_nested_final_estimator_params_absorbed(self, clf_dataset: SplitDataset) -> None:
        estimator = _build_estimator(
            StackingClassifierCalculator(), StackingClassifierApplier(), "stacking_classifier"
        )
        estimator.fit_predict(
            clf_dataset,
            target_column="target",
            config={
                "params": {
                    "final_estimator": "logistic_regression",
                    "final_estimator__C": 0.5,
                    "cv": 3,
                }
            },
        )

        model = cast(StackingClassifier, estimator.model)
        final = cast(LogisticRegression, model.final_estimator_)
        assert final.C == 0.5


# ===========================================================================
# VOTING CLASSIFIER
# ===========================================================================


class TestVotingClassifier:
    def test_fit_predict_defaults(self, clf_dataset: SplitDataset) -> None:
        estimator = _build_estimator(
            VotingClassifierCalculator(), VotingClassifierApplier(), "voting_classifier"
        )
        preds = estimator.fit_predict(clf_dataset, target_column="target", config={})

        assert len(preds["train"]) == N_TRAIN
        assert len(preds["test"]) == N_TEST
        assert set(preds["test"].unique()) <= {0, 1}
        model = cast(VotingClassifier, estimator.model)
        base_names = list(model.named_estimators_.keys())
        assert base_names == ["random_forest", "logistic_regression", "gradient_boosting"]
        assert model.voting == "soft"

    def test_stacking_meta_keys_cleaned_for_voting(self, clf_dataset: SplitDataset) -> None:
        """Stacking-family meta-keys (final_estimator, cv) must not reach voting."""
        estimator = _build_estimator(
            VotingClassifierCalculator(), VotingClassifierApplier(), "voting_classifier"
        )
        estimator.fit_predict(
            clf_dataset,
            target_column="target",
            config={
                "params": {
                    "final_estimator": "logistic_regression",
                    "cv": 5,
                    "voting": "hard",
                }
            },
        )

        model = cast(VotingClassifier, estimator.model)
        # final_estimator/cv were cleaned, so no stacking artifacts remain on
        # the fitted VotingClassifier (and fit would have crashed if they
        # reached the constructor).
        assert not hasattr(model, "final_estimator_")
        assert not hasattr(model, "cv")
        assert model.voting == "hard"


# ===========================================================================
# REGISTRY WIRING
# ===========================================================================


class TestEnsembleRegistry:
    @pytest.mark.parametrize(
        ("node_id", "calculator_cls", "applier_cls"),
        [
            ("voting_classifier", VotingClassifierCalculator, VotingClassifierApplier),
            ("stacking_classifier", StackingClassifierCalculator, StackingClassifierApplier),
            ("voting_regressor", VotingRegressorCalculator, VotingRegressorApplier),
            ("stacking_regressor", StackingRegressorCalculator, StackingRegressorApplier),
        ],
    )
    def test_node_registered(self, node_id: str, calculator_cls: Any, applier_cls: Any) -> None:
        assert NodeRegistry.get_calculator(node_id) is calculator_cls
        assert NodeRegistry.get_applier(node_id) is applier_cls
        assert node_id in NodeRegistry.get_all_metadata()
