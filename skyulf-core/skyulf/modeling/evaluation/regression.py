"""Regression evaluation logic."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from ...engines import SkyulfDataFrame
from ...modeling.sklearn_wrapper import SklearnBridge
from .common import sanitize_metrics
from .metrics import calculate_regression_metrics
from .schemas import ModelEvaluationReport, RegressionEvaluation, ResidualsData


def evaluate_regression_model(
    model: Any,
    X_test: Union[pd.DataFrame, SkyulfDataFrame],
    y_test: Union[pd.Series, Any],
    X_train: Optional[Union[pd.DataFrame, SkyulfDataFrame]] = None,
    y_train: Optional[Union[pd.Series, Any]] = None,
    dataset_name: str = "test",
) -> ModelEvaluationReport:
    """Evaluate a regression model and return a structured report."""

    # Convert to Numpy for compatibility
    X_test_np, y_test_np = SklearnBridge.to_sklearn((X_test, y_test))

    # Calculate scalar metrics
    metrics = calculate_regression_metrics(model, X_test, y_test)

    # Generate predictions
    y_pred = model.predict(X_test_np)

    # Ensure numpy arrays
    y_true_arr = y_test_np
    y_pred_arr = np.array(y_pred)

    # Calculate residuals
    residuals = y_true_arr - y_pred_arr

    # Create residuals data
    # We downsample if there are too many points to keep the payload size reasonable
    max_points = 1000
    if len(y_true_arr) > max_points:
        indices = np.random.choice(len(y_true_arr), max_points, replace=False)
        y_true_sample = y_true_arr[indices]
        y_pred_sample = y_pred_arr[indices]
        residuals_sample = residuals[indices]
    else:
        y_true_sample = y_true_arr
        y_pred_sample = y_pred_arr
        residuals_sample = residuals

    residuals_data = ResidualsData(
        predicted=y_pred_sample.tolist(),
        residuals=residuals_sample.tolist(),
        actual=y_true_sample.tolist(),
    )

    # Prediction Error Plot (Actual vs Predicted)
    # We can reuse the sampled data for this

    regression_eval = RegressionEvaluation(
        residuals=residuals_data,
        prediction_error=None,  # Can be derived from residuals data on frontend if needed
    )

    return ModelEvaluationReport(
        dataset_name=dataset_name,
        metrics=sanitize_metrics(metrics),
        classification=None,
        regression=regression_eval,
    )
