import numpy as np
import pandas as pd
import pytest
from skyulf.preprocessing.resampling import (
    OversamplingApplier,
    OversamplingCalculator,
    UndersamplingApplier,
    UndersamplingCalculator,
)


def test_oversampling_smote():
    # Create imbalanced dataset
    # Class 0: 90 samples, Class 1: 10 samples
    X = np.random.rand(100, 2)
    y = np.array([0] * 90 + [1] * 10)
    df = pd.DataFrame(X, columns=["f1", "f2"])
    df["target"] = y

    calc = OversamplingCalculator()
    applier = OversamplingApplier()

    config = {
        "method": "smote",
        "target_column": "target",
        "sampling_strategy": "auto",  # Resample to 50-50
        "k_neighbors": 2,
    }

    artifacts = calc.fit(df, config)

    # Check artifacts
    assert artifacts["type"] == "oversampling"
    assert artifacts["method"] == "smote"

    # Apply
    df_res = applier.apply(df, artifacts)

    # Check counts
    counts = df_res["target"].value_counts()
    assert counts[0] == 90
    assert counts[1] == 90  # Should be balanced

    # Check shape
    assert len(df_res) == 180


def test_undersampling_random():
    # Create imbalanced dataset
    # Class 0: 90 samples, Class 1: 10 samples
    X = np.random.rand(100, 2)
    y = np.array([0] * 90 + [1] * 10)
    df = pd.DataFrame(X, columns=["f1", "f2"])
    df["target"] = y

    calc = UndersamplingCalculator()
    applier = UndersamplingApplier()

    config = {
        "method": "random_under_sampling",
        "target_column": "target",
        "sampling_strategy": "auto",
    }

    artifacts = calc.fit(df, config)

    # Check artifacts
    assert artifacts["type"] == "undersampling"
    assert artifacts["method"] == "random_under_sampling"

    # Apply
    df_res = applier.apply(df, artifacts)

    # Check counts
    counts = df_res["target"].value_counts()
    assert counts[0] == 10  # Should be undersampled to minority class count
    assert counts[1] == 10

    # Check shape
    assert len(df_res) == 20


def test_undersampling_nearmiss():
    # Create imbalanced dataset
    X = np.random.rand(100, 2)
    y = np.array([0] * 90 + [1] * 10)
    df = pd.DataFrame(X, columns=["f1", "f2"])
    df["target"] = y

    calc = UndersamplingCalculator()
    applier = UndersamplingApplier()

    config = {"method": "nearmiss", "target_column": "target", "version": 1}

    artifacts = calc.fit(df, config)
    df_res = applier.apply(df, artifacts)

    counts = df_res["target"].value_counts()
    assert counts[0] == 10
    assert counts[1] == 10


def test_resampling_with_tuple_input():
    # Test with (X, y) tuple input which is standard in V2 pipeline
    X = pd.DataFrame(np.random.rand(100, 2), columns=["f1", "f2"])
    y = pd.Series([0] * 90 + [1] * 10, name="target")

    calc = OversamplingCalculator()
    applier = OversamplingApplier()

    config = {"method": "smote", "k_neighbors": 2}

    # Fit expects df or tuple. If tuple, config doesn't strictly need target_column if y is provided separately,
    # but calculator stores what's in config.
    artifacts = calc.fit((X, y), config)

    # Apply
    res = applier.apply((X, y), artifacts)

    assert isinstance(res, tuple)
    X_res, y_res = res

    assert len(X_res) == 180
    assert len(y_res) == 180
    assert y_res.value_counts()[0] == 90
    assert y_res.value_counts()[1] == 90


def test_resampling_missing_target_in_df():
    # If target column is specified but missing in DF and no y provided
    df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    calc = OversamplingCalculator()
    applier = OversamplingApplier()

    config = {"method": "smote", "target_column": "target"}
    artifacts = calc.fit(df, config)

    # Apply
    res = applier.apply(df, artifacts)

    # Should return original df because it can't find target
    assert len(res) == 2
    pd.testing.assert_frame_equal(res, df)
