"""Regress propagation of SMOTE settings through combined Tomek resampling."""

from typing import Any

import pandas as pd
import pytest
from sklearn.datasets import make_classification

pytest.importorskip("imblearn", reason="imbalanced-learn is optional")

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

from skyulf.preprocessing.resampling import OversamplingApplier, OversamplingCalculator


@pytest.mark.parametrize("method", ["smote", "smote_tomek"])
@pytest.mark.parametrize("strategy, minority_count", [("auto", 5), (0.8, 4), ({1: 4}, 4)])
def test_smote_tomek_honors_neighbors_for_two_minority_rows(
    method: str, strategy: Any, minority_count: int
) -> None:
    """Configured one-neighbor SMOTE must work in the combined sampler on small classes."""
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0]})
    target = pd.Series([0, 0, 0, 0, 0, 1, 1], name="target")
    config = {
        "method": method,
        "k_neighbors": 1,
        "sampling_strategy": strategy,
        "random_state": 7,
    }
    artifact = OversamplingCalculator().fit((features, target), config)

    actual_features, actual_target = OversamplingApplier().apply((features, target), artifact)

    expected_features, expected_target = SMOTE(
        k_neighbors=1, sampling_strategy=strategy, random_state=7
    ).fit_resample(features, target)
    pd.testing.assert_frame_equal(actual_features, expected_features)
    pd.testing.assert_series_equal(actual_target, expected_target)
    assert actual_target.value_counts().to_dict() == {0: 5, 1: minority_count}


@pytest.mark.parametrize("custom_neighbors", [None, 1])
def test_smote_tomek_preserves_seed_and_tomek_cleaning(custom_neighbors: int | None) -> None:
    """Forwarding neighbor settings must retain deterministic sampling and Tomek removal."""
    values, labels = make_classification(
        n_samples=80,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.75, 0.25],
        class_sep=0.4,
        random_state=52,
    )
    features = pd.DataFrame(values, columns=["x", "z"])
    target = pd.Series(labels, name="target")
    config = {"method": "smote_tomek", "random_state": 17}
    if custom_neighbors is not None:
        config["k_neighbors"] = custom_neighbors
    smote = SMOTE(random_state=17, k_neighbors=custom_neighbors or 5)
    expected_features, expected_target = SMOTETomek(
        random_state=17, smote=smote if custom_neighbors is not None else None
    ).fit_resample(features, target)
    uncleaned_features, _ = smote.fit_resample(features, target)
    artifact = OversamplingCalculator().fit((features, target), config)

    actual_features, actual_target = OversamplingApplier().apply((features, target), artifact)

    pd.testing.assert_frame_equal(actual_features, expected_features)
    pd.testing.assert_series_equal(actual_target, expected_target)
    assert len(actual_features) < len(uncleaned_features)
