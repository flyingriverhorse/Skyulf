"""Apply saved project eligibility and output rules around ordinary model inference."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl

from ..preprocessing.time_series._history_state import decode_rows, ordered_groups
from ..preprocessing.time_series.history import (
    TemporalHistorySession,
    current_history,
    propose_history,
)
from ._manifest import ColumnSpec, label_dtype
from .local_pipeline import LocalPipelineArtifact, predict_local_pipeline, validate_local_input
from .project_scoring import _typed_column, run_project_scoring, scoring_output_columns


@dataclass(frozen=True)
class LocalPrediction:
    """Scoring results and detached continuation proposed for atomic publication."""

    frame: pd.DataFrame
    history: dict[str, Any] | None


def _validate_history_request(artifact: Any, state: Any, bootstrap: Any) -> None:
    """Reject ambiguous resets and malformed public continuation arguments."""
    if not isinstance(artifact, LocalPipelineArtifact):
        raise TypeError("Expected a LocalPipelineArtifact.")
    if type(bootstrap) is not bool or (bootstrap and state is not None):
        raise ValueError("Temporal bootstrap must be boolean and cannot accompany saved history.")
    if state is not None and not isinstance(state, dict):
        raise ValueError("Saved temporal history must be a state object.")


def local_history_session(
    artifact: LocalPipelineArtifact,
    state: dict[str, Any] | None = None,
    *,
    bootstrap: bool = False,
) -> TemporalHistorySession | nullcontext[None]:
    """Bind carry history to this model, optionally rebuilding from supplied observations."""
    _validate_history_request(artifact, state, bootstrap)
    identities = [
        step["artifact"]["history_id"]
        for step in artifact.pipeline.feature_engineer.fitted_steps
        if step["artifact"].get("history_mode") == "carry"
    ]
    if not identities:
        if state is not None:
            raise ValueError("Saved temporal history requires a model with carry steps.")
        return nullcontext(None)
    model_id = artifact.manifest.pipeline_sha256
    if bootstrap:
        state = {"version": 1, "model_id": model_id, "steps": {key: [] for key in identities}}
    return TemporalHistorySession(model_id, state)


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
    """Validate and retain prior tails when no rows reach temporal preprocessing."""
    for step in artifact.pipeline.feature_engineer.fitted_steps:
        params = step["artifact"]
        if params.get("history_mode") == "carry":
            rows = current_history(params)
            ordered_groups(decode_rows(rows, params), params)
            propose_history(params, rows)


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


def score_local_pipeline_with_history(
    frame: pd.DataFrame | pl.DataFrame,
    artifact: LocalPipelineArtifact,
    *,
    history_state: dict[str, Any] | None = None,
    bootstrap_history: bool = False,
) -> LocalPrediction:
    """Score one supplied batch and return detached, caller-owned carry continuation.

    Without previous state, carry steps use their fitted training tails. Set
    ``bootstrap_history=True`` to rebuild from a complete initial observation
    snapshot instead; it cannot accompany ``history_state``. Each later request
    must follow the previous entity times. Late, replayed and tied times fail.

    Only configured carry steps continue across requests. Other group/window
    callbacks still require their complete groups/history in each supplied frame.
    Eligibility-excluded rows do not enter history. Empty input preserves it and
    returns the declared output schema using nullable pandas dtypes.

    Persist predictions and returned history atomically after success. Concurrent
    writers must compare-and-swap their previous state or serialize publication;
    this function neither stores state nor detects another caller's commit.
    """
    if not isinstance(frame, pd.DataFrame | pl.DataFrame):
        raise TypeError("Local prediction requires a pandas or Polars DataFrame.")
    with local_history_session(artifact, history_state, bootstrap=bootstrap_history) as session:
        if len(frame):
            result = score_local_pipeline(frame, artifact)
        else:
            validate_local_input(frame, artifact)
            _preserve_history(artifact)
            index = frame.index if isinstance(frame, pd.DataFrame) else pd.RangeIndex(0)
            result = pd.DataFrame(
                {
                    column.name: _typed_column(pd.Series(index=index), column.dtype)
                    for column in scoring_output_schema(artifact)
                },
                index=index,
            )
    return LocalPrediction(result, session.state if session is not None else None)


def scoring_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Separate successful estimates from explicit exclusions and total output records."""
    if "scoring_status" not in frame:
        return {}
    predicted = int(frame["scoring_status"].eq("predicted").sum())
    excluded = int(frame["scoring_status"].eq("excluded").sum())
    if predicted + excluded != len(frame):
        raise ValueError("Scoring status must describe every requested row.")
    return {"predicted_count": predicted, "excluded_count": excluded}
