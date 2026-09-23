"""Held-out metrics for a saved pandas/Polars local pipeline."""

import math

import pandas as pd
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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
    predicted = predict_local_pipeline(features, artifact)["prediction"].to_numpy()
    if artifact.manifest.task == "regression":
        return {
            "heldout_mae": float(mean_absolute_error(actual, predicted)),
            "heldout_rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
            "heldout_r2": float(r2_score(actual, predicted)),
        }
    metrics = {
        "heldout_accuracy": float(accuracy_score(actual, predicted)),
        "heldout_f1_weighted": float(
            f1_score(actual, predicted, average="weighted", zero_division=0)
        ),
    }
    if len(artifact.manifest.classes) == 2:
        metrics["heldout_f1"] = float(
            f1_score(actual, predicted, pos_label=artifact.manifest.classes[1], zero_division=0)
        )
    return metrics
