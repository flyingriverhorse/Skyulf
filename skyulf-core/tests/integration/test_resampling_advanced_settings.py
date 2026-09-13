"""Saved resampling configuration must control the actual imbalanced-learn algorithm."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.svm import SVC

pytest.importorskip("imblearn")
from imblearn.over_sampling import SVMSMOTE, KMeansSMOTE
from imblearn.under_sampling import NearMiss

from skyulf.preprocessing.resampling import (
    OversamplingApplier,
    OversamplingCalculator,
    UndersamplingApplier,
    UndersamplingCalculator,
    _build_oversampler,
    _build_undersampler,
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["svm_smote", "kmeans_smote", "nearmiss"])
def test_resampling_matches_direct_configured_sampler(engine: str, method: str) -> None:
    """Custom estimators and neighborhood sizes must determine the generated training rows."""
    features, labels = make_classification(
        n_samples=120,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        weights=[0.75, 0.25],
        class_sep=0.6,
        random_state=4,
    )
    frame = pd.DataFrame(features, columns=["a", "b", "c"])
    target = pd.Series(labels, name="label")
    config: dict[str, Any] = {"method": method, "random_state": 7, "n_jobs": 1}
    if method == "svm_smote":
        estimator = SVC(C=10, gamma=1)
        config["svm_estimator"] = estimator
        expected_sampler = SVMSMOTE(random_state=7, svm_estimator=estimator)
    elif method == "kmeans_smote":
        estimator = KMeans(n_clusters=1, random_state=7, n_init=1)
        config.update(kmeans_estimator=estimator, cluster_balance_threshold=0.0)
        expected_sampler = KMeansSMOTE(
            random_state=7,
            k_neighbors=5,
            kmeans_estimator=estimator,
            cluster_balance_threshold=0.0,
            n_jobs=1,
        )
    else:
        config["n_neighbors"] = 1
        expected_sampler = NearMiss(n_neighbors=1, n_jobs=1)
    expected_X, expected_y = expected_sampler.fit_resample(frame, target)
    if engine == "polars":
        frame, target = pl.from_pandas(frame), pl.from_pandas(target)
    calculator = UndersamplingCalculator() if method == "nearmiss" else OversamplingCalculator()
    applier = UndersamplingApplier() if method == "nearmiss" else OversamplingApplier()
    artifact = calculator.fit((frame, target), config)

    actual_X, actual_y = applier.apply((frame, target), artifact)

    np.testing.assert_allclose(actual_X.to_numpy(), expected_X.to_numpy())
    np.testing.assert_array_equal(actual_y.to_numpy(), expected_y.to_numpy())
    assert len(actual_y) != len(labels)


@pytest.mark.parametrize(
    "method",
    ["kmeans_smote", "smote_tomek", "nearmiss", "tomek_links", "edited_nearest_neighbours"],
)
def test_parallel_capable_samplers_honor_saved_job_limit(method: str) -> None:
    """A requested worker limit must reach samplers which expose n_jobs."""
    builder = (
        _build_oversampler if method in {"kmeans_smote", "smote_tomek"} else _build_undersampler
    )
    sampler = builder(method, {"n_jobs": 2})

    assert sampler.get_params()["n_jobs"] == 2


@pytest.mark.parametrize(
    "method", ["random_over", "smote", "adasyn", "borderline_smote", "svm_smote"]
)
def test_oversamplers_without_parallel_option_remain_constructible(method: str) -> None:
    """Shared artifact defaults must not become unsupported constructor arguments."""
    sampler = _build_oversampler(method, {"n_jobs": 2})

    assert sampler.sampling_strategy == "auto"


@pytest.mark.parametrize(
    "method, setting", [("svm_smote", "svm_estimator"), ("kmeans_smote", "kmeans_estimator")]
)
def test_invalid_configured_estimator_fails_instead_of_using_default(
    method: str, setting: str
) -> None:
    """An unusable estimator setting must surface an error rather than silently training defaults."""
    frame = pd.DataFrame({"x": range(30)})
    target = pd.Series([0] * 20 + [1] * 10)
    artifact = OversamplingCalculator().fit((frame, target), {"method": method, setting: "invalid"})

    with pytest.raises(ValueError, match=setting):
        OversamplingApplier().apply((frame, target), artifact)
