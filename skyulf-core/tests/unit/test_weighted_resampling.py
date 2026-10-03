"""Keep sampling geometry independent of row weights and preserve sampled weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("imblearn")

from sklearn.datasets import make_classification

from skyulf.modeling._sample_weights import SampleWeightError
from skyulf.preprocessing.resampling import (
    OversamplingApplier,
    OversamplingCalculator,
    UndersamplingApplier,
    UndersamplingCalculator,
    _build_oversampler,
    _build_undersampler,
)


def _inputs():
    """Unequal weights and duplicate indexes expose accidental index-based alignment."""
    X, y = make_classification(
        n_samples=120,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        weights=[0.75, 0.25],
        class_sep=1.2,
        random_state=42,
    )
    return pd.DataFrame(X, index=np.zeros(len(X))), pd.Series(y), np.arange(1, 121.0)


def _run(method, config, engine="pandas"):
    """Execute the weighted hook using the real registered node pair."""
    from skyulf.preprocessing._weighted_resampling import fit_resample_weighted

    X, y, weights = _inputs()
    over = method in {
        "random_over",
        "smote",
        "adasyn",
        "borderline_smote",
        "svm_smote",
        "kmeans_smote",
        "smote_tomek",
    }
    calculator = OversamplingCalculator() if over else UndersamplingCalculator()
    applier = OversamplingApplier() if over else UndersamplingApplier()
    if engine == "polars":
        X.columns = [f"x{i}" for i in range(X.shape[1])]
        X, y = pl.from_pandas(X), pl.Series("target", y)
    return fit_resample_weighted(calculator, applier, (X, y), {"method": method, **config}, weights)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "method",
    [
        "random_over",
        "random_under_sampling",
        "nearmiss",
        "tomek_links",
        "edited_nearest_neighbours",
    ],
)
def test_selected_rows_keep_exact_source_weights(method, engine):
    """Sampler-selected positions must determine all output weights, including repeats."""
    X, y, weights = _inputs()
    builder = _build_oversampler if method == "random_over" else _build_undersampler
    sampler = builder(method, {"random_state": 19})
    expected_X, expected_y = sampler.fit_resample(X, y)
    _, (actual_X, actual_y), actual_weights = _run(method, {"random_state": 19}, engine)
    np.testing.assert_allclose(np.asarray(actual_X), expected_X)
    np.testing.assert_array_equal(np.asarray(actual_y), expected_y)
    np.testing.assert_array_equal(actual_weights, weights[sampler.sample_indices_])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("policy", ["class_mean", "uniform"])
@pytest.mark.parametrize(
    "method", ["smote", "adasyn", "borderline_smote", "svm_smote", "kmeans_smote", "smote_tomek"]
)
def test_synthetic_weights_are_explicit_and_do_not_change_geometry(method, policy, engine):
    """Weighted resampling must produce exactly the ordinary sampler's feature rows."""
    X, y, weights = _inputs()
    config = {
        "random_state": 19,
        "synthetic_weight": policy,
        "kmeans_estimator": 2,
        "cluster_balance_threshold": 0.01,
        "n_jobs": 1,
    }
    sampler = _build_oversampler(method, config)
    expected_X, expected_y = sampler.fit_resample(X, y)
    artifact, (actual_X, actual_y), actual_weights = _run(method, config, engine)
    np.testing.assert_allclose(np.asarray(actual_X), expected_X)
    np.testing.assert_array_equal(np.asarray(actual_y), expected_y)
    if method == "smote_tomek":
        pre_X, pre_y = sampler.smote_.fit_resample(X, y)
        selected = sampler.tomek_.sample_indices_
    else:
        pre_X, pre_y = expected_X, expected_y
        selected = np.arange(len(pre_y))
    synthetic_y = np.asarray(pre_y)[len(y) :]
    synthetic_w = (
        np.ones(len(synthetic_y))
        if policy == "uniform"
        else np.array([weights[np.asarray(y) == label].mean() for label in synthetic_y])
    )
    expected_weights = np.concatenate([weights, synthetic_w])[selected]
    assert len(pre_X) >= len(X)
    assert artifact["synthetic_weight"] == policy
    np.testing.assert_allclose(actual_weights, expected_weights)


@pytest.mark.parametrize("policy", [None, "guess"])
def test_synthetic_sampling_requires_known_explicit_policy(policy):
    """A caller must choose the meaning of weights for newly invented rows."""
    with pytest.raises(SampleWeightError, match="synthetic_weight"):
        _run("smote", {"synthetic_weight": policy})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("target_kind", ["native", "array", "list"])
@pytest.mark.parametrize("method", ["random_over", "random_under_sampling", "smote"])
def test_weighted_sampling_accepts_supported_input_containers(engine, wrapped, target_kind, method):
    """Weights cannot narrow the ordinary sampler's supported frame and target types."""
    from skyulf.engines.pandas_engine import SkyulfPandasWrapper
    from skyulf.engines.polars_engine import SkyulfPolarsWrapper
    from skyulf.preprocessing._weighted_resampling import fit_resample_weighted

    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([0, 0, 0, 0, 1, 1], name="label")
    if engine == "polars":
        X, y = pl.from_pandas(X), pl.from_pandas(y)
    if target_kind == "array":
        y = np.asarray(y)
    elif target_kind == "list":
        y = list(y)
    if wrapped:
        X = SkyulfPolarsWrapper(X) if isinstance(X, pl.DataFrame) else SkyulfPandasWrapper(X)
    over = method != "random_under_sampling"
    calculator = OversamplingCalculator() if over else UndersamplingCalculator()
    applier = OversamplingApplier() if over else UndersamplingApplier()
    config = {
        "method": method,
        "random_state": 19,
        "k_neighbors": 1,
        "synthetic_weight": "class_mean",
    }
    expected_X, expected_y = applier.apply((X, y), dict(calculator.fit((X, y), config)))
    _, (actual_X, actual_y), weights = fit_resample_weighted(
        calculator, applier, (X, y), config, [10, 20, 30, 40, 50, 60]
    )
    expected_native = expected_X.to_native() if hasattr(expected_X, "to_native") else expected_X
    actual_native = actual_X.to_native() if hasattr(actual_X, "to_native") else actual_X
    assert type(actual_X) is type(expected_X)
    np.testing.assert_array_equal(np.asarray(actual_native), np.asarray(expected_native))
    np.testing.assert_array_equal(np.asarray(actual_y), np.asarray(expected_y))
    expected_weights = (
        [10, 20, 30, 40, 50, 60, 55, 55]
        if method == "smote"
        else (np.asarray(expected_native["x"]) + 1) * 10
    )
    np.testing.assert_array_equal(weights, expected_weights)


@pytest.mark.parametrize("dtype", ["Float64", "Float32"])
@pytest.mark.parametrize("policy", ["class_mean", "uniform"])
def test_synthetic_sampling_preserves_nullable_numeric_prefix(dtype, policy):
    """Finite nullable feature columns must retain their original rows and weights."""
    from skyulf.preprocessing._weighted_resampling import fit_resample_weighted

    X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5], "z": [10, 11, 12, 13, 14, 15]}, dtype=dtype)
    y = pd.Series([0, 0, 0, 0, 1, 1])
    config = {"method": "smote", "k_neighbors": 1, "random_state": 19, "synthetic_weight": policy}
    expected_X, expected_y = _build_oversampler("smote", config).fit_resample(X, y)
    _, (actual_X, actual_y), weights = fit_resample_weighted(
        OversamplingCalculator(), OversamplingApplier(), (X, y), config, [10, 20, 30, 40, 50, 60]
    )
    pd.testing.assert_frame_equal(actual_X, expected_X)
    np.testing.assert_array_equal(actual_y, expected_y)
    synthetic = [55, 55] if policy == "class_mean" else [1, 1]
    np.testing.assert_array_equal(weights, [10, 20, 30, 40, 50, 60, *synthetic])


@pytest.mark.parametrize("change", ["reordered", "rounded", "labels"])
def test_synthetic_prefix_verification_rejects_changed_original_rows(change):
    """Nullable support must not accept reordered rows, changed values, or changed labels."""
    from skyulf.preprocessing._weighted_resampling import _append_weights

    X = pd.DataFrame(
        {"x": pd.Series([2**53, 2**53 + 1], dtype="Int64"), "z": pd.Series([1, 2], dtype="Float64")}
    )
    y = pd.Series([0, 1])
    output_X, output_y = X.copy(), y.copy()
    if change == "reordered":
        output_X = output_X.iloc[::-1]
    elif change == "rounded":
        output_X.loc[1, "x"] = 2**53
    else:
        output_y = output_y.iloc[::-1]
    with pytest.raises(SampleWeightError, match="original"):
        _append_weights(X, y, np.array([10, 20]), output_X, output_y, "uniform")
