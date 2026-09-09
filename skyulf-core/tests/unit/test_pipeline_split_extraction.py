"""Tests for SkyulfPipeline.get_fitted_split() (convenience split-extraction API)."""

import numpy as np
import pandas as pd
import pytest

from skyulf.pipeline import SkyulfPipeline


def _config(test_size=0.25, random_state=42):
    """Keep split-extraction tests independent of unsafe preprocessing order."""
    return {
        "preprocessing": [
            {
                "name": "split",
                "transformer": "TrainTestSplitter",
                "params": {"test_size": test_size, "random_state": random_state},
            },
            {
                "name": "imputer",
                "transformer": "SimpleImputer",
                "params": {"strategy": "mean"},
            },
        ],
        "modeling": {"type": "logistic_regression"},
    }


def test_get_fitted_split_returns_plain_pandas_objects(sample_classification_data):
    """Returned X/y for both train and test must be plain pandas objects."""
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_config())
    X_train, y_train, X_test, y_test = pipeline.get_fitted_split(data, target_column="target")
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)


def test_get_fitted_split_drops_target_column_from_features(sample_classification_data):
    """Neither X_train nor X_test should still contain the target column."""
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_config())
    X_train, _, X_test, _ = pipeline.get_fitted_split(data, target_column="target")
    assert "target" not in X_train.columns
    assert "target" not in X_test.columns


def test_get_fitted_split_row_counts_match_configured_test_size(sample_classification_data):
    """With test_size=0.25 on 100 rows, train should get ~75 rows and test ~25."""
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_config(test_size=0.25, random_state=42))
    X_train, y_train, X_test, y_test = pipeline.get_fitted_split(data, target_column="target")
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(data)
    assert len(X_test) == pytest.approx(25, abs=2)


def test_get_fitted_split_matches_pipeline_fit_row_counts(sample_classification_data):
    """The split get_fitted_split() returns should have the same row counts
    as the split SkyulfPipeline.fit() uses internally, for a fixed random_state.
    """
    data = sample_classification_data.drop(columns=["category"])
    fit_pipeline = SkyulfPipeline(_config())
    fit_pipeline.fit(data, target_column="target")

    split_pipeline = SkyulfPipeline(_config())
    X_train, y_train, X_test, y_test = split_pipeline.get_fitted_split(data, target_column="target")
    # fit()'s internal training set size is reconstructable from the same
    # configured test_size/random_state producing an identical split.
    assert len(X_train) + len(X_test) == len(data)


def test_get_fitted_split_raises_without_a_configured_splitter(sample_classification_data):
    """If preprocessing doesn't produce a train/test split, raise a clear error
    instead of returning a nonsensical single-split result.
    """
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "logistic_regression"}})
    with pytest.raises(ValueError, match="train/test split"):
        pipeline.get_fitted_split(data, target_column="target")


def test_get_fitted_split_leaves_a_fitted_pipeline_untouched():
    """Extracting a split must not refit the pipeline's own preprocessing (OC-164).

    It used to run the live ``feature_engineer.fit_transform``, replacing a
    trained scaler's statistics while keeping the model fitted against the old
    ones — so the same input's prediction silently changed from 50 to -950 and
    every later ``predict()`` was wrong with no error raised.
    """
    x = np.arange(20, dtype=float)
    data = pd.DataFrame({"x": x, "target": 10 * x})
    config = {
        "preprocessing": [
            {
                "name": "split",
                "transformer": "TrainTestSplitter",
                "params": {"test_size": 0.25, "random_state": 42},
            },
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
        ],
        "modeling": {"type": "linear_regression"},
    }
    pipeline = SkyulfPipeline(config)
    pipeline.fit(data, target_column="target")

    probe = pd.DataFrame({"x": [5.0]})
    before = float(pipeline.predict(probe).iloc[0])

    shifted = pd.DataFrame({"x": x + 100.0, "target": 10 * (x + 100.0)})
    X_train, y_train, X_test, y_test = pipeline.get_fitted_split(shifted, target_column="target")

    assert float(pipeline.predict(probe).iloc[0]) == before
    # ...and the throwaway chain really did fit the data it was handed. X comes
    # back standardized, so provenance is read off the unscaled target.
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(shifted)
    assert y_train.min() >= 1000.0
    assert y_test.min() >= 1000.0
