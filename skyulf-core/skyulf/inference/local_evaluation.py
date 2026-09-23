"""Held-out metrics for a saved pandas/Polars local pipeline."""

import pandas as pd
import polars as pl

from ..modeling._evaluation.common import sanitize_metrics
from ..modeling._evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
)
from .local_pipeline import LocalPipelineArtifact, predict_local_pipeline


def evaluate_local_holdout(
    artifact: LocalPipelineArtifact,
    heldout: pd.DataFrame | pl.DataFrame,
    *,
    target_column: str,
) -> dict[str, float]:
    """Measure a saved local pipeline on labeled rows excluded from fitting.

    The caller owns the split. This function never fits preprocessing or a model;
    it predicts with the loaded artifact and records only held-out metrics.
    Binary F1 uses the artifact's second class as the positive label.
    """
    if not isinstance(artifact, LocalPipelineArtifact):
        raise TypeError("artifact must be a LocalPipelineArtifact.")
    if not isinstance(heldout, pd.DataFrame | pl.DataFrame):
        raise TypeError("heldout must be a pandas or Polars DataFrame.")
    if len(heldout) < 2:
        raise ValueError("heldout must contain at least two labeled rows.")
    if target_column not in heldout.columns or target_column in artifact.manifest.input_columns:
        raise ValueError("heldout must contain a separate target column.")
    columns = list(artifact.manifest.input_columns)
    features = (
        heldout.select(columns) if isinstance(heldout, pl.DataFrame) else heldout.loc[:, columns]
    )
    actual = heldout[target_column].to_numpy()
    predictions = predict_local_pipeline(features, artifact)
    estimator = artifact.pipeline.model_estimator
    if estimator is None:
        raise ValueError("Local artifact has no fitted model.")
    model = estimator._unwrap_tuned_model()
    scoring_args = {
        "X_np": features.to_numpy(),
        "y_np": actual,
        "predictions": predictions["prediction"].to_numpy(),
    }
    if artifact.manifest.task == "regression":
        raw_metrics = calculate_regression_metrics(
            model, features, heldout[target_column], **scoring_args
        )
    else:
        probability_columns = [
            f"probability_{position}" for position in range(len(artifact.manifest.classes))
        ]
        raw_metrics = calculate_classification_metrics(
            model,
            features,
            heldout[target_column],
            proba=predictions.loc[:, probability_columns].to_numpy(),
            **scoring_args,
        )
    return {f"heldout_{name}": value for name, value in sanitize_metrics(raw_metrics).items()}
