"""Serving must reject invalid threshold weights before scoring predictions."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from backend.ml_pipeline.deployment.service import DeploymentService, OverrideThresholdMismatch


@pytest.mark.parametrize("source", ["override", "saved"])
@pytest.mark.parametrize(
    "thresholds", [{"a": 1.0, "m": 1.0}, {"a": 1.0, "m": 1.0, "z": 1.0, "x": 1.0}]
)
def test_deployment_rejects_incomplete_or_extra_class_keys(source, thresholds):
    """Class coverage errors must be explicit for active saved configurations as well as overrides."""
    job = SimpleNamespace(
        tuned_thresholds_enabled=True, tuned_thresholds={"thresholds": thresholds}
    )
    with pytest.raises(OverrideThresholdMismatch, match="classes"):
        DeploymentService._resolve_thresholds_for_predict(
            thresholds if source == "override" else None, job, ["z", "a", "m"]
        )


@pytest.mark.parametrize("source", ["override", "saved"])
@pytest.mark.parametrize("invalid", [0.0, -0.1, float("nan"), float("inf"), -float("inf")])
def test_deployment_rejects_invalid_multiclass_thresholds(source, invalid):
    """Neither overrides nor legacy saved sets may silently turn invalid denominators into labels."""
    thresholds = {"z": 2.0, "a": invalid, "m": 4.0}
    job = SimpleNamespace(
        tuned_thresholds_enabled=True, tuned_thresholds={"thresholds": thresholds}
    )
    with pytest.raises(OverrideThresholdMismatch, match="threshold"):
        DeploymentService._resolve_thresholds_for_predict(
            thresholds if source == "override" else None, job, np.array(["z", "a", "m"])
        )


@pytest.mark.parametrize("source", ["override", "saved"])
def test_deployment_preserves_class_order_and_positive_weights_above_one(source):
    """Actual estimator probability columns must map to their labels despite JSON key order."""
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    estimator = DummyClassifier(strategy="prior").fit(features, ["z", "a", "m"])
    thresholds = {"z": 4.0, "m": 2.0, "a": 3.0}
    job = SimpleNamespace(
        tuned_thresholds_enabled=True, tuned_thresholds={"thresholds": thresholds}
    )
    resolved = DeploymentService._resolve_thresholds_for_predict(
        thresholds if source == "override" else None, job, estimator.classes_
    )
    predictions, applied = DeploymentService._predict_and_decode(
        estimator, features, None, None, thresholds=resolved
    )
    assert predictions == ["m", "m", "m"]
    assert applied == thresholds


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_deployment_binary_boundary_pair_predicts_with_exact_probabilities(threshold):
    """Saved binary cutoff endpoints must behave like >= even when an estimator emits zero or one."""
    features = pd.DataFrame({"x": [0.0, 1.0]})
    estimator = DummyClassifier(strategy="constant", constant="yes").fit(features, ["no", "yes"])
    thresholds = {"yes": threshold, "no": 1.0 - threshold}
    resolved = DeploymentService._resolve_thresholds_for_predict(
        thresholds, None, estimator.classes_
    )
    with np.errstate(divide="raise", invalid="raise"):
        predictions, applied = DeploymentService._predict_and_decode(
            estimator, features, None, None, thresholds=resolved
        )
    assert predictions == ["yes", "yes"]
    assert applied == thresholds
