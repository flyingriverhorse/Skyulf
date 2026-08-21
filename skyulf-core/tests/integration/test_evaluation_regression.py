"""Tests for skyulf.modeling._evaluation.regression (evaluate_regression_model)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.modeling._evaluation.regression import evaluate_regression_model
from skyulf.modeling._evaluation.schemas import ModelEvaluationReport


@pytest.fixture
def regression_fitted():
    """Deterministic linear regression model + test data with known residual structure."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 50)})
    y = pd.Series(2 * X["f1"] + 1 + rng.normal(0, 0.05, 50))
    model = LinearRegression().fit(X, y)
    return model, X, y


def test_evaluate_regression_returns_model_evaluation_report(regression_fitted):
    """The report should be a ModelEvaluationReport with regression populated."""
    model, X, y = regression_fitted
    report = evaluate_regression_model(model, X, y, dataset_name="holdout")
    assert isinstance(report, ModelEvaluationReport)
    assert report.dataset_name == "holdout"
    assert report.regression is not None
    assert report.classification is None


def test_evaluate_regression_residuals_length_matches_input(regression_fitted):
    """Residuals/predicted/actual lists should each have one entry per sample."""
    model, X, y = regression_fitted
    report = evaluate_regression_model(model, X, y)
    assert report.regression is not None
    residuals = report.regression.residuals
    assert residuals is not None
    assert len(residuals.predicted) == len(y)
    assert len(residuals.residuals) == len(y)
    assert len(residuals.actual) == len(y)


def test_evaluate_regression_residuals_equal_actual_minus_predicted(regression_fitted):
    """Each residual value must equal actual - predicted for that sample."""
    model, X, y = regression_fitted
    report = evaluate_regression_model(model, X, y)
    assert report.regression is not None
    residuals = report.regression.residuals
    assert residuals is not None
    computed = np.array(residuals.actual) - np.array(residuals.predicted)
    np.testing.assert_allclose(residuals.residuals, computed, rtol=1e-6, atol=1e-9)


def test_evaluate_regression_prediction_error_is_none(regression_fitted):
    """prediction_error is documented as derivable client-side; should stay None."""
    model, X, y = regression_fitted
    report = evaluate_regression_model(model, X, y)
    assert report.regression is not None
    assert report.regression.prediction_error is None


def test_evaluate_regression_downsamples_large_datasets():
    """Datasets larger than 1000 rows should be downsampled to at most 1000 residual points."""
    rng = np.random.RandomState(1)
    n = 1500
    X = pd.DataFrame({"f1": rng.normal(0, 1, n)})
    y = pd.Series(X["f1"] * 2 + rng.normal(0, 0.1, n))
    model = LinearRegression().fit(X, y)
    report = evaluate_regression_model(model, X, y)
    assert report.regression is not None
    assert report.regression.residuals is not None
    assert len(report.regression.residuals.predicted) == 1000


def test_evaluate_regression_metrics_include_r2(regression_fitted):
    """Scalar metrics attached to the report should include a valid r2 value."""
    model, X, y = regression_fitted
    report = evaluate_regression_model(model, X, y)
    assert "r2" in report.metrics
    assert report.metrics["r2"] <= 1.0


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing values — closer to production data than the small
    synthetic fixtures used elsewhere in this file.
    """

    def test_evaluate_income_regression_on_customers_data(self) -> None:
        df = load_sample_dataset("customers")
        # LinearRegression can't handle NaN, so rows with missing age/income
        # are dropped rather than assumed clean.
        df = df.dropna(subset=["age", "income"])
        X = df[["age"]]
        y = df["income"]
        model = LinearRegression().fit(X, y)

        report = evaluate_regression_model(model, X, y, dataset_name="customers")
        assert isinstance(report, ModelEvaluationReport)
        assert report.regression is not None
        assert report.regression.residuals is not None
        assert len(report.regression.residuals.predicted) == len(y)
        assert "r2" in report.metrics
