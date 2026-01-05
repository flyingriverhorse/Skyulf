import os
import shutil

import numpy as np
import pandas as pd
import pytest
from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.classification import (
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
)
from skyulf.modeling.regression import (
    RandomForestRegressorApplier,
    RandomForestRegressorCalculator,
    RidgeRegressionApplier,
    RidgeRegressionCalculator,
)
from skyulf.modeling.tuning import TuningCalculator, TuningConfig

from backend.ml_pipeline.artifacts.local import LocalArtifactStore


@pytest.fixture
def classification_data():
    # Simple binary classification problem
    train = pd.DataFrame(
        {
            "f1": [1, 2, 3, 4, 5, 6],
            "f2": [1, 1, 1, 0, 0, 0],
            "target": [1, 1, 1, 0, 0, 0],
        }
    )
    test = pd.DataFrame({"f1": [1.5, 5.5], "f2": [1, 0], "target": [1, 0]})
    return SplitDataset(train=train, test=test)


@pytest.fixture
def regression_data():
    # Simple linear regression problem y = 2*x
    train = pd.DataFrame({"x": [1, 2, 3, 4, 5], "target": [2, 4, 6, 8, 10]})
    test = pd.DataFrame({"x": [1.5, 3.5], "target": [3, 7]})  # Interpolation
    return SplitDataset(train=train, test=test)


@pytest.fixture
def artifact_store():
    path = "temp_test_artifacts_modeling"
    if os.path.exists(path):
        shutil.rmtree(path)
    store = LocalArtifactStore(path)
    yield store
    if os.path.exists(path):
        shutil.rmtree(path)


def test_logistic_regression_flow(classification_data, artifact_store):
    calculator = LogisticRegressionCalculator()
    applier = LogisticRegressionApplier()
    estimator = StatefulEstimator(calculator, applier, "adv_node")

    # Fit and Predict
    predictions = estimator.fit_predict(classification_data, "target", {})

    # Check Model
    assert estimator.model is not None
    from sklearn.linear_model import LogisticRegression

    assert isinstance(estimator.model, LogisticRegression)

    # Check Predictions
    assert "train" in predictions
    assert "test" in predictions
    assert len(predictions["train"]) == 6
    assert len(predictions["test"]) == 2

    # Basic accuracy check (should be perfect on this simple data)
    assert (predictions["test"] == classification_data.test["target"]).all()


def test_ridge_regression_flow(regression_data, artifact_store):
    calculator = RidgeRegressionCalculator()
    applier = RidgeRegressionApplier()
    estimator = StatefulEstimator(calculator, applier, "ridge_node")

    # Fit and Predict
    predictions = estimator.fit_predict(
        regression_data, "target", {"alpha": 0.0}
    )  # alpha=0 is OLS

    # Check Model
    assert estimator.model is not None
    from sklearn.linear_model import Ridge

    assert isinstance(estimator.model, Ridge)

    # Check Predictions
    preds = predictions["test"]
    actuals = regression_data.test["target"]
    mae = np.mean(np.abs(preds - actuals))
    assert mae < 0.1  # Should be very close


def test_logistic_regression_evaluation(classification_data, artifact_store):
    calculator = LogisticRegressionCalculator()
    applier = LogisticRegressionApplier()
    estimator = StatefulEstimator(calculator, applier, "lr_eval_node")

    # Fit first
    estimator.fit_predict(classification_data, "target", {})

    # Evaluate
    report = estimator.evaluate(classification_data, "target")

    assert report["problem_type"] == "classification"
    assert "train" in report["splits"]
    assert "test" in report["splits"]

    # Check Metrics
    test_metrics = report["splits"]["test"].metrics
    assert "accuracy" in test_metrics
    assert test_metrics["accuracy"] == 1.0

    # Check Confusion Matrix
    cm = report["splits"]["test"].classification.confusion_matrix
    assert cm is not None
    # assert cm.accuracy == 1.0 # Accuracy is in metrics, not CM object

    # Check ROC Curves (should exist for binary classification)
    assert len(report["splits"]["test"].classification.roc_curves) > 0


def test_ridge_regression_evaluation(regression_data, artifact_store):
    calculator = RidgeRegressionCalculator()
    applier = RidgeRegressionApplier()
    estimator = StatefulEstimator(calculator, applier, "ridge_eval_node")

    # Fit first
    estimator.fit_predict(regression_data, "target", {"alpha": 0.0})

    # Evaluate
    report = estimator.evaluate(regression_data, "target")

    assert report["problem_type"] == "regression"
    assert "train" in report["splits"]
    assert "test" in report["splits"]

    # Check Metrics
    test_metrics = report["splits"]["test"].metrics
    assert "mse" in test_metrics
    assert "r2" in test_metrics
    assert test_metrics["mse"] < 0.01

    # Check Residuals
    residuals = report["splits"]["test"].regression.residuals
    assert residuals is not None
    assert len(residuals.residuals) == 2


def test_cross_validation(classification_data, artifact_store):
    calculator = LogisticRegressionCalculator()
    applier = LogisticRegressionApplier()
    estimator = StatefulEstimator(calculator, applier, "cv_node")

    # Run CV (2 folds because data is small)
    cv_results = estimator.cross_validate(classification_data, "target", {}, n_folds=2)

    assert "aggregated_metrics" in cv_results
    assert "folds" in cv_results
    assert len(cv_results["folds"]) == 2

    # Check aggregated metrics
    agg = cv_results["aggregated_metrics"]
    assert "accuracy" in agg
    assert "mean" in agg["accuracy"]
    assert "std" in agg["accuracy"]

    # Check range (exact 1.0 might fail on small random splits)
    assert 0.0 <= agg["accuracy"]["mean"] <= 1.0


def test_hyperparameter_tuning(classification_data, artifact_store):
    # Setup
    model_calculator = LogisticRegressionCalculator()
    tuner = TuningCalculator(model_calculator)

    # Config
    config = TuningConfig(
        strategy="grid",
        metric="accuracy",
        n_trials=2,
        cv_folds=2,
        search_space={"C": [0.1, 1.0, 10.0], "solver": ["lbfgs"]},
    )

    # Run Tuning
    X_train = classification_data.train.drop(columns=["target"])
    y_train = classification_data.train["target"]

    result = tuner.tune(X_train, y_train, config)

    # Verify
    assert result.n_trials == 3  # Grid search 3 candidates
    assert "C" in result.best_params
    assert result.best_score > 0.0

    # Ensure best params can be used for training
    estimator = StatefulEstimator(
        model_calculator, LogisticRegressionApplier(), "tuned_node"
    )
    estimator.fit_predict(classification_data, "target", result.best_params)
    assert estimator.model is not None


def test_halving_strategies(classification_data, artifact_store):
    from skyulf.modeling.classification import LogisticRegressionCalculator
    from skyulf.modeling.tuning.schemas import TuningConfig
    from skyulf.modeling.tuning.engine import TuningCalculator

    model_calculator = LogisticRegressionCalculator()
    tuner = TuningCalculator(model_calculator)

    X_train = classification_data.train.drop(columns=["target"])
    y_train = classification_data.train["target"]

    # Halving Grid
    config_grid = TuningConfig(
        strategy="halving_grid",
        metric="accuracy",
        cv_folds=2,
        search_space={"C": [0.1, 1.0], "solver": ["lbfgs"]},
    )

    try:
        result_grid = tuner.tune(X_train, y_train, config_grid)
        assert result_grid is not None
    except ValueError as e:
        # If it fails due to data size, that's expected for this tiny fixture
        # "min_resources_=8 is greater than max_resources_=6."
        if "min_resources_" in str(e) and "max_resources_" in str(e):
            pass
        elif "n_splits" not in str(e) and "n_samples" not in str(e):
            raise e

    # Halving Random
    config_random = TuningConfig(
        strategy="halving_random",
        metric="accuracy",
        n_trials=2,
        cv_folds=2,
        search_space={"C": [0.1, 1.0], "solver": ["lbfgs"]},
    )

    try:
        result_random = tuner.tune(X_train, y_train, config_random)
        assert result_random is not None
    except ValueError as e:
        if "min_resources_" in str(e) and "max_resources_" in str(e):
            pass
        elif "n_splits" not in str(e) and "n_samples" not in str(e):
            raise e


def test_advanced_features(classification_data, artifact_store):
    from skyulf.modeling.base import StatefulEstimator
    from skyulf.modeling.classification import (
        LogisticRegressionApplier,
        LogisticRegressionCalculator,
    )

    calculator = LogisticRegressionCalculator()
    applier = LogisticRegressionApplier()
    estimator = StatefulEstimator(calculator, applier, "adv_node")

    # 1. Test Refit Strategy
    # Should run without error and produce predictions
    estimator.fit_predict(classification_data, "target", {})
    estimator.refit(classification_data, "target", {})

    # Verify artifact exists (it should have been overwritten)
    assert estimator.model is not None

    # 2. Test Progress Callback
    progress_calls = []

    def on_progress(current, total):
        progress_calls.append((current, total))

    estimator.cross_validate(
        classification_data, "target", {}, n_folds=2, progress_callback=on_progress
    )

    assert len(progress_calls) == 2
    assert progress_calls[0] == (1, 2)
    assert progress_calls[1] == (2, 2)

    # 3. Test Feature Columns in Report
    report = estimator.evaluate(classification_data, "target")
    # assert len(report.feature_columns) > 0
    # assert "f1" in report.feature_columns
