"""Optional training-only CV using Core estimators and fold-local feature engineering."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ...data.dataset import SplitDataset
from ...modeling.base import BaseModelCalculator, StatefulEstimator
from ...modeling.cross_validation import _build_splitter
from ...preprocessing.fold_adapter import (
    SPLITTER_STEP_TYPES,
    AuditedFoldPreprocessor,
    FeatureEngineerFoldAdapter,
)
from ...registry import NodeRegistry

CV_FIELDS = {
    "cv_enabled": "enabled",
    "cv_folds": "folds",
    "cv_type": "method",
    "cv_shuffle": "shuffle",
    "cv_random_state": "random_state",
}


@dataclass(frozen=True, slots=True)
class LocalCVSpec:
    """Bound optional fixed-parameter evaluation without changing the final model."""

    enabled: bool = False
    folds: int = 5
    method: str = "k_fold"
    shuffle: bool = True
    random_state: int = 42

    def __post_init__(self) -> None:
        """Reject mistyped settings and splitter fallbacks before any source access."""
        if type(self.enabled) is not bool or type(self.shuffle) is not bool:
            raise ValueError("cv_enabled and cv_shuffle must be boolean.")
        if type(self.folds) is not int or not 2 <= self.folds <= 20:
            raise ValueError("cv_folds must be an integer from 2 to 20.")
        if type(self.random_state) is not int or not 0 <= self.random_state < 2**32:
            raise ValueError("cv_random_state must be an integer from 0 to 2**32 - 1.")
        if self.method not in ("k_fold", "stratified_k_fold", "time_series_split", "shuffle_split"):
            raise ValueError(
                "cv_type must be k_fold, stratified_k_fold, time_series_split or shuffle_split."
            )
        if self.method == "time_series_split" and self.shuffle:
            raise ValueError("Time-series CV requires cv_shuffle=false.")
        if self.method == "shuffle_split" and not self.shuffle:
            raise ValueError("Shuffle-split CV requires cv_shuffle=true.")

    @classmethod
    def from_workflow(cls, config: dict[str, Any]) -> "LocalCVSpec":
        """Read the same flat CV field names used by Canvas without leaking other config."""
        return cls(**{field: config[key] for key, field in CV_FIELDS.items() if key in config})

    def validate_pipeline(
        self, config: dict[str, Any], *, target_column: str, event_column: str | None = None
    ) -> None:
        """Validate the fixed estimator and fold chain before training or remote work."""
        if not self.enabled:
            return
        calculator = NodeRegistry.get_calculator(config["modeling"]["type"])()
        if not isinstance(calculator, BaseModelCalculator) or calculator.problem_type not in (
            "classification",
            "regression",
        ):
            raise ValueError("Bundle CV requires a registered classification or regression model.")
        if self.method == "stratified_k_fold" and calculator.problem_type != "classification":
            raise ValueError("Stratified CV requires a classification model.")
        if self.method == "time_series_split" and not event_column:
            raise ValueError(
                "Time-series CV requires window selection with an explicit event_column."
            )
        steps = config.get("preprocessing", [])
        if any(step.get("transformer") in SPLITTER_STEP_TYPES for step in steps):
            raise ValueError("Bundle CV owns the split; remove preprocessing splitter nodes.")
        FeatureEngineerFoldAdapter(steps, target_column)


def _validate_fold_membership(
    frame: pd.DataFrame | pl.DataFrame,
    spec: LocalCVSpec,
    target_column: str,
    problem_type: str,
    event_column: str | None,
) -> None:
    """Reject undersized folds and ambiguous time boundaries before fitting any model."""
    labels = frame[target_column].to_numpy()
    if spec.method == "stratified_k_fold":
        counts = pd.Series(labels).value_counts()
        if len(counts) < 2 or counts.min() < spec.folds:
            raise ValueError(
                "Stratified CV requires at least cv_folds rows per class and two classes."
            )
    times = None
    if spec.method == "time_series_split":
        if event_column is None or event_column not in frame.columns:
            raise ValueError("Time-series CV requires its event_column in the training payload.")
        values = frame[event_column]
        times = values.to_pandas() if isinstance(values, pl.Series) else values
        if not isinstance(times.dtype, pd.DatetimeTZDtype) or times.isna().any():
            raise ValueError(
                "CV event_column must contain normalized nonnull timezone-aware timestamps."
            )
        times = times.sort_values(kind="stable").reset_index(drop=True)
    splitter = _build_splitter(
        cv_type=spec.method,
        n_folds=spec.folds,
        problem_type=problem_type,
        shuffle=spec.shuffle,
        random_state=spec.random_state,
    )
    for train, validation in splitter.split(np.arange(len(frame)), labels):
        if len(train) < 2 or len(validation) < 2:
            raise ValueError("Each CV fold needs at least two training and validation rows.")
        if times is not None and times.iloc[train[-1]] >= times.iloc[validation[0]]:
            raise ValueError(
                "A shared timestamp crosses a CV fold boundary; use different folds or aggregate observations."
            )


def evaluate_training_cv(
    frame: pd.DataFrame | pl.DataFrame,
    config: dict[str, Any],
    spec: LocalCVSpec,
    *,
    target_column: str,
    event_column: str | None = None,
) -> dict[str, Any] | None:
    """Evaluate a bounded raw training partition; never fit or inspect a final holdout.

    Callers own source bounds and the outer split. Time metadata is passed only
    for time-series CV; Core sorts by it and removes it before fold preprocessing.
    The independent final pipeline fit does not reuse any fold's learned state.
    """
    if not spec.enabled:
        return None
    spec.validate_pipeline(config, target_column=target_column, event_column=event_column)
    model = config["modeling"]
    calculator = NodeRegistry.get_calculator(model["type"])()
    applier = NodeRegistry.get_applier(model["type"])()
    _validate_fold_membership(frame, spec, target_column, calculator.problem_type, event_column)
    adapter = AuditedFoldPreprocessor(
        FeatureEngineerFoldAdapter(config.get("preprocessing", []), target_column)
    )
    estimator = StatefulEstimator(calculator, applier, "bundle_cv")
    result = estimator.cross_validate(
        SplitDataset(train=frame, test=frame.head(0)),
        target_column,
        model,
        n_folds=spec.folds,
        cv_type=spec.method,
        shuffle=spec.shuffle,
        random_state=spec.random_state,
        time_column=event_column,
        preprocessing=adapter,
    )
    if not result["aggregated_metrics"]:
        raise ValueError("CV produced no finite aggregate metrics.")
    result["cv_config"]["time_column"] = event_column
    result["fold_refit"] = adapter.summary(train_rows=len(frame))
    return result
