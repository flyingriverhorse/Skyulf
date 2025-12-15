"""Tests for preprocessing module."""

import numpy as np
import pandas as pd
from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.pipeline import FeatureEngineer


def test_feature_engineer_simple_flow(sample_classification_data):
    """Test a simple feature engineering pipeline."""
    config = [
        {
            "name": "imputer",
            "transformer": "SimpleImputer",
            "params": {"columns": ["feature1"], "strategy": "mean"},
        },
        {
            "name": "scaler",
            "transformer": "StandardScaler",
            "params": {"columns": ["feature1", "feature2"]},
        },
    ]

    fe = FeatureEngineer(config)

    # Fit Transform
    transformed_data, metrics = fe.fit_transform(sample_classification_data)

    assert isinstance(transformed_data, pd.DataFrame)
    assert transformed_data["feature1"].isna().sum() == 0

    # Check if scaling happened (mean approx 0)
    assert np.abs(transformed_data["feature1"].mean()) < 0.1

    # Transform (Inference)
    new_data = sample_classification_data.copy()
    inference_data = fe.transform(new_data)

    assert isinstance(inference_data, pd.DataFrame)
    assert inference_data["feature1"].isna().sum() == 0


def test_feature_engineer_with_splitting(sample_classification_data):
    """Test pipeline with splitting steps."""
    config = [
        {
            "name": "target_split",
            "transformer": "feature_target_split",
            "params": {"target_column": "target"},
        },
        {
            "name": "train_test_split",
            "transformer": "TrainTestSplitter",
            "params": {"test_size": 0.2, "random_state": 42},
        },
        {
            "name": "imputer",
            "transformer": "SimpleImputer",
            "params": {"columns": ["feature1"], "strategy": "mean"},
        },
    ]

    fe = FeatureEngineer(config)

    # Fit Transform
    transformed_data, metrics = fe.fit_transform(sample_classification_data)

    assert isinstance(transformed_data, SplitDataset)
    assert transformed_data.train is not None
    assert transformed_data.test is not None

    # Check structure of train (should be tuple X, y because of feature_target_split)
    assert isinstance(transformed_data.train, tuple)
    X_train, y_train = transformed_data.train
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)

    # Check imputation on X_train
    assert X_train["feature1"].isna().sum() == 0


def test_one_hot_encoding(sample_classification_data):
    """Test OneHotEncoder."""
    config = [
        {
            "name": "encoder",
            "transformer": "OneHotEncoder",
            "params": {"columns": ["category"]},
        }
    ]

    fe = FeatureEngineer(config)
    transformed_data, _ = fe.fit_transform(sample_classification_data)

    # Check for new columns
    assert "category_A" in transformed_data.columns
    assert "category_B" in transformed_data.columns
    assert (
        "category" not in transformed_data.columns
    )  # Should be dropped by default usually, or kept depending on impl
    # Checking implementation: OneHotEncoder usually drops original if not configured otherwise.

    # Test inference
    inference_data = fe.transform(sample_classification_data.iloc[:5])
    assert "category_A" in inference_data.columns
