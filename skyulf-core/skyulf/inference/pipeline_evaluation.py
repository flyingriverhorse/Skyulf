"""Held-out metrics for a saved pandas/Polars fitted pipeline."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from joblib import parallel_config

from ..modeling._evaluation.common import sanitize_metrics
from ..modeling._evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
)
from ..preprocessing._helpers import select_rows_by_position
from ..preprocessing._target_labels import encoded_labels
from .fitted_pipeline import FittedPipelineArtifact, predict_pipeline
from .pipeline_scoring import score_pipeline


def evaluate_holdout(
    artifact: FittedPipelineArtifact,
    heldout: pd.DataFrame | pl.DataFrame,
    *,
    target_column: str,
    on_predictions: Callable[[Any, pd.DataFrame], None] | None = None,
) -> dict[str, float]:
    """Measure a saved fitted pipeline on labeled rows excluded from fitting.

    The caller owns the split. This function never fits preprocessing or a model;
    it predicts with the loaded artifact and records only held-out metrics.
    Binary F1 uses the artifact's second class as the positive label.
    An optional observer receives actual labels and predictions after scoring,
    allowing integrations to sample the exact outputs without repeating inference.
    Fitted outlier filters contribute metrics only for predicted rows, with
    total/scored/excluded counts. The observer still receives every input row.
    Evaluation uses sequential joblib prediction to keep parallel forest sums
    reproducible for exact registry evidence checks. Saved model parameters and
    normal training/batch prediction parallelism remain unchanged. This does not
    guarantee determinism for arbitrary models or other numerical runtimes.
    """
    features, actual, predictions = _holdout_predictions(artifact, heldout, target_column)
    raw_metrics = _evaluated_metrics(
        artifact, features, heldout, target_column, actual, predictions
    )
    if on_predictions is not None:
        on_predictions(actual, predictions)
    return {f"heldout_{name}": value for name, value in sanitize_metrics(raw_metrics).items()}


def _holdout_predictions(
    artifact: FittedPipelineArtifact, heldout: pd.DataFrame | pl.DataFrame, target_column: str
) -> tuple[Any, np.ndarray, pd.DataFrame]:
    """Predict the entire supplied holdout once before choosing any metric population."""
    _validate_holdout(artifact, heldout, target_column)
    columns = list(artifact.manifest.input_columns)
    features = (
        heldout.select(columns) if isinstance(heldout, pl.DataFrame) else heldout.loc[:, columns]
    )
    predict = (
        score_pipeline if artifact.pipeline.config.get("project_scoring") else predict_pipeline
    )
    with parallel_config(backend="sequential"):
        predictions = predict(features, artifact)
    return features, heldout[target_column].to_numpy(), predictions


def _scored_mask(predictions: pd.DataFrame) -> np.ndarray:
    """Keep positional coverage independent of duplicate input labels."""
    if "scoring_status" in predictions:
        return predictions["scoring_status"].eq("predicted").to_numpy(dtype=bool)
    return np.ones(len(predictions), dtype=bool)


def evaluate_holdout_comparison(
    candidate: FittedPipelineArtifact,
    champion: FittedPipelineArtifact | None,
    heldout: pd.DataFrame | pl.DataFrame,
    *,
    target_column: str,
    include_coverage: bool = False,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """Compare common predicted rows while reporting each model's own raw coverage.

    Each model predicts once on the original holdout. Metric selection never
    repeats preprocessing on a reduced frame or substitutes common coverage for
    the candidate's own scored fraction.
    """
    artifacts = [candidate] if champion is None else [candidate, champion]
    outputs = [_holdout_predictions(item, heldout, target_column) for item in artifacts]
    common = np.logical_and.reduce([_scored_mask(output[2]) for output in outputs])
    positions = np.flatnonzero(common)
    if len(positions) < 2:
        raise ValueError("Comparison requires at least two common predicted holdout rows.")
    coverage = include_coverage or any("scoring_status" in output[2] for output in outputs)
    metrics = []
    for artifact, (features, actual, predicted) in zip(artifacts, outputs, strict=True):
        raw = _evaluated_metrics(
            artifact,
            features,
            heldout,
            target_column,
            actual,
            predicted,
            positions=positions,
            include_coverage=coverage,
        )
        metrics.append({f"heldout_{name}": value for name, value in sanitize_metrics(raw).items()})
    return metrics[0], metrics[1] if champion is not None else None


def _evaluated_metrics(
    artifact: FittedPipelineArtifact,
    features: pd.DataFrame | pl.DataFrame,
    heldout: pd.DataFrame | pl.DataFrame,
    target_column: str,
    actual: Any,
    predictions: pd.DataFrame,
    *,
    positions: np.ndarray | None = None,
    include_coverage: bool = False,
) -> dict[str, Any]:
    """Evaluate only predicted outcomes and make outlier coverage explicit."""
    coverage = include_coverage or "scoring_status" in predictions
    if not coverage:
        return _holdout_metrics(artifact, features, heldout, target_column, actual, predictions)
    own_rows = int(_scored_mask(predictions).sum())
    comparing = positions is not None
    if positions is None:
        positions = np.flatnonzero(_scored_mask(predictions))
    if len(positions) < 2:
        raise ValueError("Outlier-filtered holdout must contain at least two scored rows.")
    metrics = _holdout_metrics(
        artifact,
        select_rows_by_position(features, positions),
        select_rows_by_position(heldout, positions),
        target_column,
        actual[positions],
        predictions.iloc[positions],
    )
    return {
        **metrics,
        "total_rows": len(predictions),
        "scored_rows": own_rows,
        "excluded_rows": len(predictions) - own_rows,
        "scored_fraction": own_rows / len(predictions),
        **({"comparison_rows": len(positions)} if comparing else {}),
    }


def _validate_holdout(
    artifact: FittedPipelineArtifact, heldout: pd.DataFrame | pl.DataFrame, target_column: str
) -> None:
    """Require labeled evaluation rows with a target outside the model inputs."""
    if not isinstance(artifact, FittedPipelineArtifact):
        raise TypeError("artifact must be a FittedPipelineArtifact.")
    if not isinstance(heldout, pd.DataFrame | pl.DataFrame):
        raise TypeError("heldout must be a pandas or Polars DataFrame.")
    if len(heldout) < 2:
        raise ValueError("heldout must contain at least two labeled rows.")
    if target_column not in heldout.columns or target_column in artifact.manifest.input_columns:
        raise ValueError("heldout must contain a separate target column.")


def _holdout_metrics(
    artifact: FittedPipelineArtifact,
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
