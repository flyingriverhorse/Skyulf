"""Optional training-only CV using Core estimators and fold-local feature engineering."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from .....data.dataset import SplitDataset
from .....modeling._tuning.cv_policy import (
    effective_cv_type,
    policy_splitter,
    prepare_policy_data,
    take_rows,
)
from .....modeling._tuning.schemas import TuningConfig
from .....modeling._tuning.splitters import build_shuffle_split_cv
from .....modeling.base import BaseModelCalculator, StatefulEstimator
from .....preprocessing.fold_adapter import (
    SPLITTER_STEP_TYPES,
    AuditedFoldPreprocessor,
    FeatureEngineerFoldAdapter,
)
from .....registry import NodeRegistry
from ..thresholds.decision_thresholds import threshold_policy
from ..thresholds.threshold_cv import evaluate_threshold_cv
from .local_search import base_model_config, prepare_search_pipeline

CV_FIELDS = {
    "cv_enabled": "enabled",
    "cv_folds": "folds",
    "cv_type": "method",
    "cv_shuffle": "shuffle",
    "cv_random_state": "random_state",
    "cv_inner_folds": "inner_folds",
    "cv_nested_type": "nested_type",
    "cv_group_column": "group_column",
    "cv_gap": "gap",
    "cv_test_size": "test_size",
    "cv_max_train_size": "max_train_size",
}


@dataclass(frozen=True, slots=True)
class LocalCVSpec:
    """Bound shared training-only folds for fixed models and parameter search."""

    enabled: bool = False
    folds: int = 5
    method: str = "k_fold"
    shuffle: bool = True
    random_state: int = 42
    inner_folds: int | None = None
    nested_type: str = "auto"
    group_column: str | None = None
    gap: int = 0
    test_size: int | None = None
    max_train_size: int | None = None

    @property
    def temporal(self) -> bool:
        """Identify temporal policies at either search level."""
        return self.method == "time_series_split" or (
            self.method == "nested_cv" and self.nested_type == "time_series_split"
        )

    def __post_init__(self) -> None:
        """Reject mistyped settings and splitter fallbacks before any source access."""
        if type(self.enabled) is not bool or type(self.shuffle) is not bool:
            raise ValueError("cv_enabled and cv_shuffle must be boolean.")
        if type(self.folds) is not int or not 2 <= self.folds <= 20:
            raise ValueError("cv_folds must be an integer from 2 to 20.")
        if self.inner_folds is not None and (
            type(self.inner_folds) is not int or not 2 <= self.inner_folds <= 20
        ):
            raise ValueError("cv_inner_folds must be an integer from 2 to 20, or null.")
        if type(self.random_state) is not int or not 0 <= self.random_state < 2**32:
            raise ValueError("cv_random_state must be an integer from 0 to 2**32 - 1.")
        _validate_cv_method(self)
        _validate_policy_settings(self)

    @classmethod
    def from_workflow(cls, config: dict[str, Any]) -> LocalCVSpec:
        """Read the same flat CV field names used by Canvas without leaking other config."""
        return cls(**{field: config[key] for key, field in CV_FIELDS.items() if key in config})

    def validate_pipeline(
        self, config: dict[str, Any], *, target_column: str, event_column: str | None = None
    ) -> None:
        """Validate the selected model and fold chain before training or remote work."""
        prepare_search_pipeline(
            config, self, target_column=target_column, event_column=event_column
        )
        if not self.enabled:
            return
        _validate_cv_model(self, config, event_column)
        steps = config.get("preprocessing", [])
        if any(step.get("transformer") in SPLITTER_STEP_TYPES for step in steps):
            raise ValueError("Bundle CV owns the split; remove preprocessing splitter nodes.")
        FeatureEngineerFoldAdapter(steps, target_column)


def validate_fold_membership(
    frame: pd.DataFrame | pl.DataFrame,
    spec: LocalCVSpec,
    target_column: str,
    problem_type: str,
    event_column: str | None,
) -> None:
    """Reject undersized folds and ambiguous time boundaries before fitting any model."""
    settings = {key: getattr(spec, field) for key, field in CV_FIELDS.items()}
    settings["cv_time_column"] = event_column if spec.temporal else None
    config = TuningConfig(**settings)
    labels = frame[target_column]
    features = (
        frame.drop(target_column)
        if isinstance(frame, pl.DataFrame)
        else frame.drop(columns=[target_column])
    )
    features, labels, metadata = prepare_policy_data(features, labels, config, problem_type)
    _validate_stratified_counts(labels, config, problem_type)
    splitter = (
        build_shuffle_split_cv(config)
        if spec.method == "shuffle_split"
        else policy_splitter(config, problem_type, labels, metadata)
    )
    for train, validation in splitter.split(features, labels):
        _validate_fold_sizes(train, validation)
        if spec.method == "nested_cv":
            _validate_inner_membership(labels, metadata, config, problem_type, train)


def _validate_inner_membership(
    labels: Any, metadata: dict[str, Any], config: TuningConfig, problem_type: str, train: Any
) -> None:
    """Admit the exact inner policy on an isolated outer training partition."""
    folds = config.cv_folds
    count = config.cv_inner_folds or (min(3, folds - 1) if folds > 2 else 2)
    inner_config = replace(config, cv_folds=count)
    inner_labels = take_rows(labels, train)
    inner_metadata = {name: values[train] for name, values in metadata.items()}
    _validate_stratified_counts(inner_labels, inner_config, problem_type)
    inner = policy_splitter(inner_config, problem_type, inner_labels, inner_metadata)
    for inner_train, inner_test in inner.split(np.arange(len(train)), inner_labels):
        _validate_fold_sizes(inner_train, inner_test)


def _validate_stratified_counts(labels: Any, config: TuningConfig, problem_type: str) -> None:
    """Retain the Bundle's actionable per-class fold-count validation."""
    if effective_cv_type(config, problem_type) == "stratified_k_fold":
        counts = pd.Series(np.asarray(labels)).value_counts()
        if len(counts) < 2 or counts.min() < config.cv_folds:
            raise ValueError(
                "Stratified CV requires at least cv_folds rows per class and two classes."
            )


def _validate_fold_sizes(train: Any, validation: Any) -> None:
    """Keep every local fit and metric supported by at least two rows."""
    if len(train) < 2 or len(validation) < 2:
        raise ValueError("Each CV fold needs at least two training and validation rows.")


def evaluate_training_cv(
    frame: pd.DataFrame | pl.DataFrame,
    config: dict[str, Any],
    spec: LocalCVSpec,
    *,
    target_column: str,
    event_column: str | None = None,
    sample_weight: Any = None,
) -> dict[str, Any] | None:
    """Evaluate a bounded raw training partition; never fit or inspect a final holdout.

    Callers own source bounds and the outer split. Time metadata is passed only
    for time-series CV; Core sorts by it and removes it before fold preprocessing.
    The independent final pipeline fit does not reuse any fold's learned state.
    """
    if not spec.enabled:
        return None
    spec.validate_pipeline(config, target_column=target_column, event_column=event_column)
    if threshold_policy(config)["mode"] != "off":
        return evaluate_threshold_cv(
            frame,
            config,
            spec,
            target_column=target_column,
            event_column=event_column,
            sample_weight=sample_weight,
        )
    model = config["modeling"]
    calculator = NodeRegistry.get_calculator(model["type"])()
    applier = NodeRegistry.get_applier(model["type"])()
    validate_fold_membership(frame, spec, target_column, calculator.problem_type, event_column)
    adapter = AuditedFoldPreprocessor(
        FeatureEngineerFoldAdapter(config.get("preprocessing", []), target_column)
    )
    estimator = StatefulEstimator(calculator, applier, "bundle_cv")
    result = estimator.cross_validate(
        SplitDataset(train=frame, test=frame.head(0), train_sample_weight=sample_weight),
        target_column,
        model,
        n_folds=spec.folds,
        cv_type=spec.method,
        shuffle=spec.shuffle,
        random_state=spec.random_state,
        time_column=event_column,
        preprocessing=adapter,
        cv_nested_type=spec.nested_type,
        group_column=spec.group_column,
        gap=spec.gap,
        test_size=spec.test_size,
        max_train_size=spec.max_train_size,
        inner_folds=spec.inner_folds,
    )
    if not result["aggregated_metrics"]:
        raise ValueError("CV produced no finite aggregate metrics.")
    result["cv_config"]["time_column"] = event_column
    result["fold_refit"] = adapter.summary(train_rows=len(frame))
    return result


def _validate_cv_method(spec: LocalCVSpec) -> None:
    """Validate supported split methods and their shuffle requirements."""
    if spec.method not in (
        "k_fold",
        "stratified_k_fold",
        "time_series_split",
        "shuffle_split",
        "nested_cv",
        "group_k_fold",
        "stratified_group_k_fold",
    ):
        raise ValueError(
            "cv_type must be k_fold, stratified_k_fold, time_series_split, shuffle_split, "
            "group_k_fold, stratified_group_k_fold or nested_cv."
        )
    if spec.temporal and spec.shuffle:
        raise ValueError("Time-series CV requires cv_shuffle=false.")
    if spec.method == "shuffle_split" and not spec.shuffle:
        raise ValueError("Shuffle-split CV requires cv_shuffle=true.")


def _validate_policy_settings(spec: LocalCVSpec) -> None:
    """Reject missing metadata and inactive policy options before source access."""
    choices = {
        "auto",
        "k_fold",
        "stratified_k_fold",
        "time_series_split",
        "group_k_fold",
        "stratified_group_k_fold",
    }
    if spec.nested_type not in choices:
        raise ValueError("Invalid cv_nested_type.")
    if spec.method != "nested_cv" and spec.nested_type != "auto":
        raise ValueError("cv_nested_type requires nested_cv.")
    policy = spec.nested_type if spec.method == "nested_cv" else spec.method
    grouped = policy in {"group_k_fold", "stratified_group_k_fold"}
    if grouped and (not isinstance(spec.group_column, str) or not spec.group_column.strip()):
        raise ValueError("Group CV requires cv_group_column.")
    if not grouped and spec.group_column is not None:
        raise ValueError("cv_group_column requires a group CV policy.")
    _validate_time_sizes(spec)


def _validate_time_sizes(spec: LocalCVSpec) -> None:
    """Keep gap and window lengths explicit integer row counts."""
    if type(spec.gap) is not int or spec.gap < 0:
        raise ValueError("cv_gap must be a nonnegative integer row count.")
    for name in ("test_size", "max_train_size"):
        value = getattr(spec, name)
        if not _optional_positive_rows(value):
            raise ValueError(f"cv_{name} must be a positive integer row count or null.")
    if not spec.temporal and (
        spec.gap or spec.test_size is not None or spec.max_train_size is not None
    ):
        raise ValueError("CV gap and window settings require a temporal CV policy.")


def _optional_positive_rows(value: Any) -> bool:
    """Accept an omitted window or a strictly positive integer row count."""
    return value is None or (type(value) is int and value > 0)


def _validate_cv_model(spec: LocalCVSpec, config: dict[str, Any], event_column: str | None) -> None:
    """Require a compatible supervised model and temporal metadata for the selected CV."""
    calculator = NodeRegistry.get_calculator(base_model_config(config)["type"])()
    if not isinstance(calculator, BaseModelCalculator) or calculator.problem_type not in (
        "classification",
        "regression",
    ):
        raise ValueError("Bundle CV requires a registered classification or regression model.")
    if spec.method == "stratified_k_fold" and calculator.problem_type != "classification":
        raise ValueError("Stratified CV requires a classification model.")
    if spec.temporal and not event_column:
        raise ValueError("Time-series CV requires window selection with an explicit event_column.")
