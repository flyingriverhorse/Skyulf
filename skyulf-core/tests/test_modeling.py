"""Tests for modeling module."""

import pandas as pd
from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.classification import (
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
)
from skyulf.modeling.regression import (
    RandomForestRegressorApplier,
    RandomForestRegressorCalculator,
)


def test_logistic_regression_training(sample_classification_data):
    """Test Logistic Regression training and prediction."""
    # Prepare data
    # Drop missing for this test or fill them
    data = sample_classification_data.fillna(0)
    # Drop non-numeric for simple test
    data = data.drop(columns=["category"])

    dataset = SplitDataset(train=data.iloc[:80], test=data.iloc[80:], validation=None)

    estimator = StatefulEstimator(
        node_id="test_lr",
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
    )

    config = {"params": {"C": 0.5}}

    # Fit Predict
    predictions = estimator.fit_predict(dataset, target_column="target", config=config)

    assert "train" in predictions
    assert "test" in predictions
    assert len(predictions["train"]) == 80
    assert len(predictions["test"]) == 20

    # Evaluate
    report = estimator.evaluate(dataset, target_column="target")
    assert report["problem_type"] == "classification"
    assert "accuracy" in report["splits"]["test"].metrics


def test_random_forest_regression(sample_regression_data):
    """Test Random Forest Regression."""
    data = sample_regression_data.drop(columns=["category"])

    dataset = SplitDataset(train=data.iloc[:80], test=data.iloc[80:], validation=None)

    estimator = StatefulEstimator(
        node_id="test_rf",
        calculator=RandomForestRegressorCalculator(),
        applier=RandomForestRegressorApplier(),
    )

    config = {"params": {"n_estimators": 10}}

    estimator.fit_predict(dataset, target_column="target", config=config)

    report = estimator.evaluate(dataset, target_column="target")
    assert report["problem_type"] == "regression"
    assert "mse" in report["splits"]["test"].metrics


def test_cross_validation(sample_classification_data):
    """Test Cross Validation."""
    data = sample_classification_data.fillna(0).drop(columns=["category"])

    dataset = SplitDataset(train=data, test=pd.DataFrame())

    estimator = StatefulEstimator(
        node_id="test_cv",
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
    )

    cv_results = estimator.cross_validate(
        dataset=dataset, target_column="target", config={}, n_folds=3
    )

    assert "aggregated_metrics" in cv_results
    assert "accuracy" in cv_results["aggregated_metrics"]
    assert len(cv_results["folds"]) == 3
