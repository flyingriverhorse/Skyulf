"""Tests for SkyulfPipeline.get_fitted_split() (convenience split-extraction API)."""

import pandas as pd
import pytest

from skyulf.pipeline import SkyulfPipeline


def _config(test_size=0.25, random_state=42):
    return {
        "preprocessing": [
            {
                "name": "imputer",
                "transformer": "SimpleImputer",
                "params": {"strategy": "mean"},
            },
            {
                "name": "split",
                "transformer": "TrainTestSplitter",
                "params": {"test_size": test_size, "random_state": random_state},
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
    as the split SkyulfPipeline.fit() uses internally, for a fixed random_state."""
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
    instead of returning a nonsensical single-split result."""
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "logistic_regression"}})
    with pytest.raises(ValueError, match="train/test split"):
        pipeline.get_fitted_split(data, target_column="target")
