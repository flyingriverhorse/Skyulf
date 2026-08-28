"""Integration tests for the CalibratedClassifier and SGDClassifier modeling nodes.

Covers the remaining classification gap classes (CalibratedClassifierCalculator/
Applier, SGDClassifierCalculator/Applier) end-to-end through StatefulEstimator:
fit -> predict -> evaluate, base-estimator resolution, parameter propagation,
and registry wiring.
"""

import logging
from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.classification import (
    CalibratedClassifierApplier,
    CalibratedClassifierCalculator,
    SGDClassifierApplier,
    SGDClassifierCalculator,
)
from skyulf.registry import NodeRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def blobs_dataset() -> SplitDataset:
    """Binary classification on two well-separated 3D Gaussian blobs.

    100 samples per class with a fixed 4.0 offset, so every linear (and most
    non-linear) base estimator reaches near-perfect accuracy quickly.
    """
    rng = np.random.RandomState(42)
    n = 100
    base = rng.normal(0.0, 1.0, size=(n, 3))
    X = np.vstack([base, base + 4.0])
    y = np.array([0] * n + [1] * n)
    df = pd.DataFrame(X, columns=["f0", "f1", "f2"])  # ty: ignore[invalid-argument-type]
    df["target"] = y
    order = rng.permutation(len(df))
    df = df.iloc[order].reset_index(drop=True)
    return SplitDataset(train=df.iloc[:160], test=df.iloc[160:], validation=None)


def _build_estimator(
    calculator: CalibratedClassifierCalculator | SGDClassifierCalculator,
    applier: CalibratedClassifierApplier | SGDClassifierApplier,
) -> StatefulEstimator:
    """Pair a calculator/applier into a StatefulEstimator keyed by node id."""
    node_id = (
        "calibrated_classifier"
        if isinstance(calculator, CalibratedClassifierCalculator)
        else "sgd_classifier"
    )
    return StatefulEstimator(node_id=node_id, calculator=calculator, applier=applier)


# ===========================================================================
# CALIBRATED CLASSIFIER
# ===========================================================================


class TestCalibratedClassifier:
    def test_fit_predict(self, blobs_dataset: SplitDataset) -> None:
        estimator = _build_estimator(
            CalibratedClassifierCalculator(), CalibratedClassifierApplier()
        )
        preds = estimator.fit_predict(blobs_dataset, target_column="target", config={})

        assert len(preds["train"]) == 160
        assert len(preds["test"]) == 40
        assert set(preds["test"].unique()) <= {0, 1}

        model = cast(CalibratedClassifierCV, estimator.model)
        assert isinstance(model, CalibratedClassifierCV)
        # Default config resolves to the logistic_regression base estimator
        # with cv=5, so five calibrated folds must exist.
        assert len(model.calibrated_classifiers_) == 5
        assert isinstance(model.calibrated_classifiers_[0].estimator, LogisticRegression)

        test_labels = cast(pd.DataFrame, blobs_dataset.test)["target"]
        accuracy = (cast(pd.Series, preds["test"]).to_numpy() == test_labels.to_numpy()).mean()
        assert accuracy > 0.9

    def test_evaluate_reports_classification_metrics(self, blobs_dataset: SplitDataset) -> None:
        estimator = _build_estimator(
            CalibratedClassifierCalculator(), CalibratedClassifierApplier()
        )
        estimator.fit_predict(blobs_dataset, target_column="target", config={})
        report = estimator.evaluate(blobs_dataset, target_column="target")

        assert report["problem_type"] == "classification"
        test_report = report["splits"]["test"]
        assert test_report is not None
        assert "accuracy" in test_report.metrics
        assert test_report.metrics["accuracy"] > 0.9

    def test_predict_proba_rows_sum_to_one(self, blobs_dataset: SplitDataset) -> None:
        estimator = _build_estimator(
            CalibratedClassifierCalculator(), CalibratedClassifierApplier()
        )
        estimator.fit_predict(blobs_dataset, target_column="target", config={})

        # The applier treats every input column as a feature, so the target
        # column must be stripped first (fit_predict does this internally).
        test_features = cast(pd.DataFrame, blobs_dataset.test).drop(columns=["target"])
        proba = estimator.applier.predict_proba(test_features, estimator.model)

        assert proba is not None
        assert set(map(str, proba.columns)) == {"0", "1"}
        assert np.allclose(proba.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
        assert (proba.to_numpy() >= 0).all()

    def test_nested_base_estimator_string_resolves(self, blobs_dataset: SplitDataset) -> None:
        """Nested params: the 'random_forest' key must become a fitted RandomForest."""
        estimator = _build_estimator(
            CalibratedClassifierCalculator(), CalibratedClassifierApplier()
        )
        estimator.fit_predict(
            blobs_dataset,
            target_column="target",
            config={"params": {"base_estimator": "random_forest"}},
        )

        model = cast(CalibratedClassifierCV, estimator.model)
        assert isinstance(model.calibrated_classifiers_[0].estimator, RandomForestClassifier)

    def test_flat_base_estimator_string_resolves(self, blobs_dataset: SplitDataset) -> None:
        """Flat legacy config: 'decision_tree' must still resolve to its factory."""
        estimator = _build_estimator(
            CalibratedClassifierCalculator(), CalibratedClassifierApplier()
        )
        estimator.fit_predict(
            blobs_dataset, target_column="target", config={"base_estimator": "decision_tree"}
        )

        model = cast(CalibratedClassifierCV, estimator.model)
        assert isinstance(model.calibrated_classifiers_[0].estimator, DecisionTreeClassifier)

    def test_unknown_base_estimator_falls_back_with_warning(
        self, blobs_dataset: SplitDataset, caplog: pytest.LogCaptureFixture
    ) -> None:
        estimator = _build_estimator(
            CalibratedClassifierCalculator(), CalibratedClassifierApplier()
        )
        with caplog.at_level(logging.WARNING, logger="skyulf.modeling.classification"):
            estimator.fit_predict(
                blobs_dataset,
                target_column="target",
                config={"params": {"base_estimator": "totally_bogus"}},
            )

        model = cast(CalibratedClassifierCV, estimator.model)
        assert isinstance(model.calibrated_classifiers_[0].estimator, LogisticRegression)
        assert "Unknown base_estimator" in caplog.text

    def test_cv_param_is_applied(self, blobs_dataset: SplitDataset) -> None:
        estimator = _build_estimator(
            CalibratedClassifierCalculator(), CalibratedClassifierApplier()
        )
        estimator.fit_predict(blobs_dataset, target_column="target", config={"params": {"cv": 3}})

        model = cast(CalibratedClassifierCV, estimator.model)
        assert len(model.calibrated_classifiers_) == 3

    def test_isotonic_method_is_applied(self, blobs_dataset: SplitDataset) -> None:
        estimator = _build_estimator(
            CalibratedClassifierCalculator(), CalibratedClassifierApplier()
        )
        estimator.fit_predict(
            blobs_dataset, target_column="target", config={"params": {"method": "isotonic"}}
        )

        model = cast(CalibratedClassifierCV, estimator.model)
        assert model.calibrated_classifiers_[0].method == "isotonic"

    def test_invalid_method_raises_value_error(self, blobs_dataset: SplitDataset) -> None:
        estimator = _build_estimator(
            CalibratedClassifierCalculator(), CalibratedClassifierApplier()
        )
        with pytest.raises(ValueError, match="must be"):
            estimator.fit_predict(
                blobs_dataset,
                target_column="target",
                config={"params": {"method": "bogus"}},
            )


# ===========================================================================
# SGD CLASSIFIER
# ===========================================================================


class TestSGDClassifier:
    def test_fit_predict(self, blobs_dataset: SplitDataset) -> None:
        estimator = _build_estimator(SGDClassifierCalculator(), SGDClassifierApplier())
        preds = estimator.fit_predict(blobs_dataset, target_column="target", config={})

        assert len(preds["train"]) == 160
        assert len(preds["test"]) == 40
        assert set(preds["test"].unique()) <= {0, 1}
        assert isinstance(estimator.model, SGDClassifier)

        model = cast(SGDClassifier, estimator.model)
        assert model.loss == "log_loss"
        assert model.penalty == "l2"

        test_labels = cast(pd.DataFrame, blobs_dataset.test)["target"]
        accuracy = (cast(pd.Series, preds["test"]).to_numpy() == test_labels.to_numpy()).mean()
        assert accuracy > 0.9

    def test_evaluate_reports_classification_metrics(self, blobs_dataset: SplitDataset) -> None:
        estimator = _build_estimator(SGDClassifierCalculator(), SGDClassifierApplier())
        estimator.fit_predict(blobs_dataset, target_column="target", config={})
        report = estimator.evaluate(blobs_dataset, target_column="target")

        assert report["problem_type"] == "classification"
        test_report = report["splits"]["test"]
        assert test_report is not None
        assert "accuracy" in test_report.metrics
        assert test_report.metrics["accuracy"] > 0.9

    def test_predict_proba_rows_sum_to_one(self, blobs_dataset: SplitDataset) -> None:
        estimator = _build_estimator(SGDClassifierCalculator(), SGDClassifierApplier())
        estimator.fit_predict(blobs_dataset, target_column="target", config={})

        test_features = cast(pd.DataFrame, blobs_dataset.test).drop(columns=["target"])
        proba = estimator.applier.predict_proba(test_features, estimator.model)

        assert proba is not None
        assert set(map(str, proba.columns)) == {"0", "1"}
        assert np.allclose(proba.sum(axis=1).to_numpy(), 1.0, atol=1e-6)

    def test_hinge_loss_disables_predict_proba(self, blobs_dataset: SplitDataset) -> None:
        """Hinge loss is not probability-capable, so the applier must return None."""
        estimator = _build_estimator(SGDClassifierCalculator(), SGDClassifierApplier())
        preds = estimator.fit_predict(
            blobs_dataset, target_column="target", config={"params": {"loss": "hinge"}}
        )

        assert cast(SGDClassifier, estimator.model).loss == "hinge"
        assert len(preds["test"]) == 40

        test_features = cast(pd.DataFrame, blobs_dataset.test).drop(columns=["target"])
        assert estimator.applier.predict_proba(test_features, estimator.model) is None

    def test_alpha_param_is_applied(self, blobs_dataset: SplitDataset) -> None:
        estimator = _build_estimator(SGDClassifierCalculator(), SGDClassifierApplier())
        estimator.fit_predict(
            blobs_dataset, target_column="target", config={"params": {"alpha": 0.5}}
        )

        assert cast(SGDClassifier, estimator.model).alpha == 0.5

    def test_calculator_metadata(self) -> None:
        assert SGDClassifierCalculator().problem_type == "classification"
        # `random_state` is not part of the static defaults: the default seed is
        # injected at fit-resolution time (`_inject_default_seed`, finding F-21).
        assert SGDClassifierCalculator().default_params == {
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 0.0001,
            "l1_ratio": 0.15,
            "max_iter": 1000,
        }
        assert SGDClassifierCalculator()._resolve_fit_params({})["random_state"] == 42


# ===========================================================================
# REGISTRY WIRING
# ===========================================================================


class TestClassificationGapsRegistry:
    def test_registry_returns_concrete_classes(self) -> None:
        assert (
            NodeRegistry.get_calculator("calibrated_classifier") is CalibratedClassifierCalculator
        )
        assert NodeRegistry.get_applier("calibrated_classifier") is CalibratedClassifierApplier
        assert NodeRegistry.get_calculator("sgd_classifier") is SGDClassifierCalculator
        assert NodeRegistry.get_applier("sgd_classifier") is SGDClassifierApplier

    def test_nodes_are_registered_as_models(self) -> None:
        models = NodeRegistry.list_models()
        assert "calibrated_classifier" in models
        assert "sgd_classifier" in models

    def test_registry_metadata(self) -> None:
        metadata = NodeRegistry.get_all_metadata()

        assert metadata["calibrated_classifier"]["category"] == "Modeling"
        assert metadata["sgd_classifier"]["category"] == "Modeling"
        assert metadata["calibrated_classifier"]["params"] == {
            "base_estimator": "logistic_regression",
            "method": "sigmoid",
            "cv": 5,
        }
        # Seed lives in the injected default (F-21), not in the static metadata.
        assert metadata["sgd_classifier"]["params"] == {
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 0.0001,
            "l1_ratio": 0.15,
            "max_iter": 1000,
        }

    def test_calculator_problem_types(self) -> None:
        assert CalibratedClassifierCalculator().problem_type == "classification"
        assert SGDClassifierCalculator().problem_type == "classification"
