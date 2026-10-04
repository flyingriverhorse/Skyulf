"""Apply saved project eligibility and output rules around ordinary model inference."""

from typing import Any

import pandas as pd
import polars as pl

from ..preprocessing.time_series.history import current_history, propose_history
from ._manifest import ColumnSpec, label_dtype
from .local_pipeline import LocalPipelineArtifact, predict_local_pipeline, validate_local_input
from .project_scoring import run_project_scoring, scoring_output_columns


def prediction_output_schema(artifact: LocalPipelineArtifact) -> tuple[ColumnSpec, ...]:
    """Describe raw estimator outputs independently of optional scoring policies."""
    manifest = artifact.manifest
    label = "float64" if manifest.task == "regression" else label_dtype(manifest.classes)
    columns = [ColumnSpec(name="prediction", dtype=label)]
    if manifest.task == "classification" and manifest.classification_probabilities:
        columns.extend(
            ColumnSpec(name=f"probability_{i}", dtype="float64")
            for i in range(len(manifest.classes))
        )
    return tuple(columns)


def scoring_output_schema(artifact: LocalPipelineArtifact) -> tuple[ColumnSpec, ...]:
    """Include declared rules and explicit row outcomes only for opted-in models."""
    columns = prediction_output_schema(artifact)
    config = artifact.pipeline.config.get("project_scoring")
    if config is None:
        return columns
    additions = (
        *scoring_output_columns(config),
        ("scoring_status", "string"),
        ("exclusion_reason", "string"),
    )
    return (*columns, *(ColumnSpec(name=name, dtype=dtype) for name, dtype in additions))


def _preserve_history(artifact: LocalPipelineArtifact) -> None:
    """Retain the previous tail when explicit eligibility excludes the entire batch."""
    for step in artifact.pipeline.feature_engineer.fitted_steps:
        params = step["artifact"]
        if params.get("history_mode") == "carry":
            propose_history(params, current_history(params))


def score_local_pipeline(
    frame: pd.DataFrame | pl.DataFrame, artifact: LocalPipelineArtifact
) -> pd.DataFrame:
    """Return exactly one outcome per requested row using the saved rule versions."""
    config = artifact.pipeline.config.get("project_scoring")
    if config is None:
        return predict_local_pipeline(frame, artifact)
    validate_local_input(frame, artifact)
    predicted = False

    def predict(eligible: Any) -> pd.DataFrame:
        """Delegate eligible rows to the existing engine and fitted-state contract."""
        nonlocal predicted
        predicted = True
        return predict_local_pipeline(eligible, artifact)

    result = run_project_scoring(
        frame,
        predict,
        source=artifact.pipeline.config["project_python_source"],
        config=config,
        row_keys=[],
        prediction_dtypes={
            column.name: column.dtype for column in prediction_output_schema(artifact)
        },
    )
    if not predicted:
        _preserve_history(artifact)
    return result


def scoring_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Separate successful estimates from explicit exclusions and total output records."""
    if "scoring_status" not in frame:
        return {}
    predicted = int(frame["scoring_status"].eq("predicted").sum())
    excluded = int(frame["scoring_status"].eq("excluded").sum())
    if predicted + excluded != len(frame):
        raise ValueError("Scoring status must describe every requested row.")
    return {"predicted_count": predicted, "excluded_count": excluded}
