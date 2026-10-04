"""Held-out metrics for a saved pandas/Polars local pipeline."""

from collections.abc import Callable
from typing import Any

import pandas as pd
import polars as pl
from joblib import parallel_config

from ..modeling._evaluation.common import sanitize_metrics
from ..modeling._evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
)
from ..preprocessing._target_labels import encoded_labels
from .local_pipeline import LocalPipelineArtifact, predict_local_pipeline


def evaluate_local_holdout(
    artifact: LocalPipelineArtifact,
    heldout: pd.DataFrame | pl.DataFrame,
    *,
    target_column: str,
    on_predictions: Callable[[Any, pd.DataFrame], None] | None = None,
) -> dict[str, float]:
    """Measure a saved local pipeline on labeled rows excluded from fitting.

    The caller owns the split. This function never fits preprocessing or a model;
    it predicts with the loaded artifact and records only held-out metrics.
    Binary F1 uses the artifact's second class as the positive label.
    An optional observer receives actual labels and predictions after scoring,
    allowing integrations to sample the exact outputs without repeating inference.
    Evaluation uses sequential joblib prediction to keep parallel forest sums
    reproducible for exact registry evidence checks. Saved model parameters and
    normal training/batch prediction parallelism remain unchanged. This does not
    guarantee determinism for arbitrary models or other numerical runtimes.
    """
    _validate_holdout(artifact, heldout, target_column)
    columns = list(artifact.manifest.input_columns)
    features = (
        heldout.select(columns) if isinstance(heldout, pl.DataFrame) else heldout.loc[:, columns]
    )
    actual = heldout[target_column].to_numpy()
    with parallel_config(backend="sequential"):
        predictions = predict_local_pipeline(features, artifact)
    raw_metrics = _holdout_metrics(artifact, features, heldout, target_column, actual, predictions)
    if on_predictions is not None:
        on_predictions(actual, predictions)
    return {f"heldout_{name}": value for name, value in sanitize_metrics(raw_metrics).items()}


def _validate_holdout(
    artifact: LocalPipelineArtifact, heldout: pd.DataFrame | pl.DataFrame, target_column: str
) -> None:
    """Require labeled evaluation rows with a target outside the model inputs."""
    if not isinstance(artifact, LocalPipelineArtifact):
        raise TypeError("artifact must be a LocalPipelineArtifact.")
    if not isinstance(heldout, pd.DataFrame | pl.DataFrame):
        raise TypeError("heldout must be a pandas or Polars DataFrame.")
    if len(heldout) < 2:
        raise ValueError("heldout must contain at least two labeled rows.")
    if target_column not in heldout.columns or target_column in artifact.manifest.input_columns:
        raise ValueError("heldout must contain a separate target column.")


def _holdout_metrics(
    artifact: LocalPipelineArtifact,
    features: pd.DataFrame | pl.DataFrame,
    heldout: pd.DataFrame | pl.DataFrame,
    target_column: str,
    actual: Any,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Score existing predictions using the saved model's task and probability contract."""
    estimator = artifact.pipeline.model_estimator
    if estimator is None:
        raise ValueError("Local artifact has no fitted model.")
    model = estimator._unwrap_tuned_model()
    scoring_args: dict[str, Any] = {
        "X_np": features.to_numpy(),
        "y_np": actual,
        "predictions": predictions["prediction"].to_numpy(),
    }
    if artifact.manifest.task == "regression":
        raw_metrics = calculate_regression_metrics(
            model, features, heldout[target_column], **scoring_args
        )
    else:
        # Score on the fitted class axis: custom ordinal orders need not match
        # sklearn's sorted original labels. Observers still receive raw labels.
        classes = model.classes_
        scoring_args["y_np"] = encoded_labels(artifact.pipeline, actual, classes)
        scoring_args["predictions"] = encoded_labels(
            artifact.pipeline, scoring_args["predictions"], classes
        )
        probability_columns = (
            [f"probability_{position}" for position in range(len(artifact.manifest.classes))]
            if artifact.manifest.classification_probabilities
            else []
        )
        raw_metrics = calculate_classification_metrics(
            model,
            features,
            heldout[target_column],
            proba=(
                predictions.loc[:, probability_columns].to_numpy() if probability_columns else None
            ),
            **scoring_args,
        )
    return raw_metrics
