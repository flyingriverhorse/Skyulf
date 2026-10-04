"""Pin configured capability parity and recursive rejection before model fitting."""

import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from skyulf.modeling._sample_weights import SampleWeightError, ensure_sample_weight_support
from skyulf.modeling.capabilities import model_supports_sample_weight
from skyulf.modeling.classification import _SeededCalibratedClassifierCV


@pytest.mark.parametrize(
    "model_type",
    [
        "voting_classifier",
        "stacking_classifier",
        "voting_regressor",
        "stacking_regressor",
        "calibrated_classifier",
    ],
)
def test_configured_defaults_support_weights(model_type):
    """Integration selectors must offer the same weighted composites as runtime."""
    assert model_supports_sample_weight(model_type)


@pytest.mark.parametrize(
    "model_type,params",
    [
        ("voting_classifier", {"base_estimators": ["knn"]}),
        ("stacking_regressor", {"final_estimator": "knn"}),
        ("stacking_classifier", {"final_estimator": "knn"}),
        ("calibrated_classifier", {"estimator": KNeighborsClassifier()}),
        ("voting_classifier", {"base_estimators": ["knn"], "calibrate_base_models": True}),
    ],
)
def test_configured_unsupported_children_rejected(model_type, params):
    """Config choices cannot hide unsupported children behind a supported parent."""
    assert not model_supports_sample_weight(model_type, params)


@pytest.mark.parametrize(
    "model",
    [
        StackingClassifier([("lr", LogisticRegression())], final_estimator=KNeighborsClassifier()),
        VotingClassifier([("nested", CalibratedClassifierCV(KNeighborsClassifier()))]),
        _SeededCalibratedClassifierCV(estimator=KNeighborsClassifier()),
    ],
)
def test_recursive_estimator_rejection(model):
    """Unsupported final and nested calibration learners fail before their fit starts."""
    with pytest.raises(SampleWeightError, match="KNeighborsClassifier.*sample_weight"):
        ensure_sample_weight_support(model)


@pytest.mark.parametrize(
    "model",
    [
        VotingClassifier([("lr", LogisticRegression())]),
        StackingClassifier([("lr", LogisticRegression())]),
        CalibratedClassifierCV(LogisticRegression()),
    ],
)
def test_composite_class_and_user_weights_combine_once(model):
    """Nonnative balanced weights multiply the user's vector once at the parent fit."""
    import numpy as np

    from skyulf.modeling._class_weights import sample_weight_for_fit

    weights = sample_weight_for_fit(model, "balanced", [0, 0, 0, 1], [1, 2, 3, 4])
    assert weights is not None
    np.testing.assert_allclose(weights, [2 / 3, 4 / 3, 2, 8])
