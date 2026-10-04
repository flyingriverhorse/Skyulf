"""Fit an optional decision policy without exposing final holdout labels."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ...data.dataset import SplitDataset
from ...modeling._evaluation.thresholds import apply_thresholds, optimize_thresholds
from ...modeling._sample_weights import validate_sample_weight
from ...modeling._tuning.cv_policy import take_rows
from ...pipeline import SkyulfPipeline
from ...preprocessing._target_labels import encoded_label, original_labels
from ...preprocessing.split import DataSplitter
from .decision_thresholds import manual_thresholds, threshold_metric, threshold_policy


def calibration_partition(frame: Any, target: str, policy: dict, context: dict) -> tuple:
    """Respect chronological or whole-group separation before fitting preprocessing."""
    fraction = policy["validation_fraction"]
    group = context.get("group_column")
    event = context.get("event_column")
    if context.get("split_strategy") == "temporal":
        train, validation = _temporal_calibration(frame, event, fraction, context.get("gap", 0))
    elif group:
        splitter = DataSplitter(test_size=fraction, random_state=policy["random_state"])
        train, validation = splitter.split_indices(len(frame), groups=np.asarray(frame[group]))
    else:
        splitter = DataSplitter(test_size=fraction, random_state=policy["random_state"])
        train, validation = splitter.split_indices(len(frame), stratify=np.asarray(frame[target]))
    if min(len(train), len(validation)) < 2:
        raise ValueError(
            "Decision threshold calibration requires nonempty fitting and validation rows."
        )
    labels = np.asarray(frame[target])
    if set(labels[train]) != set(labels) or set(labels[validation]) != set(labels):
        raise ValueError("Decision threshold calibration requires every class in both partitions.")
    if group and set(np.asarray(frame[group])[train]) & set(np.asarray(frame[group])[validation]):
        raise ValueError("Decision threshold calibration must contain disjoint groups.")
    return train, validation


def _temporal_calibration(frame: Any, event: str | None, fraction: float, gap: int) -> tuple:
    """Choose a chronological tail without sharing boundary timestamps or gap rows."""
    if not event or event not in frame.columns:
        raise ValueError("Automatic decision thresholds require the training event column.")
    times = pd.to_datetime(np.asarray(frame[event]), utc=True, errors="raise")
    if np.any(times.isna()):
        raise ValueError("Decision threshold calibration requires valid event timestamps.")
    order = np.argsort(times, kind="stable")
    cut = int(len(order) * (1 - fraction))
    if not 0 < cut < len(order):
        raise ValueError("Decision threshold calibration requires fitting and validation rows.")
    boundary = times[order[cut]]
    train, validation = order[times[order] < boundary], order[times[order] >= boundary]
    return (train[:-gap] if gap else train), validation


def _model_frame(frame: Any, config: dict, target: str) -> Any:
    """Remove split-only metadata from fixed estimators after calibration membership is known."""
    columns = config.get("decision_threshold_context", {}).get("input_columns")
    if columns is None:
        return frame
    names = [*columns, target]
    if config["modeling"]["type"] == "hyperparameter_tuner":
        for key in ("cv_time_column", "cv_group_column"):
            column = config["modeling"].get(key)
            if column and column not in names:
                names.append(column)
    return frame.select(names) if isinstance(frame, pl.DataFrame) else frame.loc[:, names]


def _features(frame: Any, target: str) -> Any:
    """Keep native frames while separating the target from probability inputs."""
    return frame.drop(target) if isinstance(frame, pl.DataFrame) else frame.drop(columns=target)


def _select_thresholds(pipeline: Any, validation: Any, target: str, policy: dict) -> dict:
    """Optimize aligned calibration probabilities, keeping probability outputs unchanged."""
    features, labels = pipeline.feature_engineer.transform(
        (_features(validation, target), validation[target])
    )
    proba = np.asarray(pipeline._predict_proba_transformed(features))
    classes = np.asarray(pipeline.model_estimator._unwrap_tuned_model().classes_)
    positive = policy.get("positive_class")
    if positive is not None:
        if len(classes) != 2 or positive not in classes:
            raise ValueError(
                "Automatic decision threshold positive_class must match a binary class."
            )
        order = [
            int(np.flatnonzero(classes != positive)[0]),
            int(np.flatnonzero(classes == positive)[0]),
        ]
        classes, proba = classes[order], proba[:, order]
    if proba.shape != (len(labels), len(classes)) or not np.all(np.isfinite(proba)):
        raise ValueError("Decision threshold calibration probabilities must be finite and aligned.")
    if set(labels) != set(classes):
        raise ValueError(
            "Decision threshold calibration must retain every fitted class after preprocessing."
        )
    metric = threshold_metric(policy, classes.tolist())
    thresholds = optimize_thresholds(labels, proba, metric, classes=classes)
    baseline = np.asarray(classes)[np.argmax(proba, axis=1)]
    chosen = apply_thresholds(proba, thresholds, classes=classes, positive_class=positive)
    score = float(metric(labels, chosen))
    if not np.isfinite(score):
        raise ValueError("Decision threshold selection requires a finite metric.")
    pipeline._decision_threshold_evidence = {
        "mode": "auto",
        "selection": "training_calibration",
        "metric": policy["metric"],
        "calibration_rows": len(labels),
        "baseline_score": float(metric(labels, baseline)),
        "selected_score": score,
    }
    return thresholds


def fit_threshold_pipeline(
    config: dict[str, Any], data: SplitDataset, target: str
) -> SkyulfPipeline:
    """Fit normal/manual models or reserve a training-only calibration population."""
    policy = threshold_policy(config)
    copied = deepcopy(config)
    pipeline = SkyulfPipeline(copied)
    fitting, validation = _calibration_data(copied, data, target, policy)
    pipeline.fit(fitting, target_column=target)
    if policy["mode"] == "off":
        return pipeline
    assert pipeline.model_estimator is not None
    model = pipeline.model_estimator._unwrap_tuned_model()
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Decision thresholds require predict_proba probabilities.")
    classes = np.asarray(model.classes_).tolist()
    policy = _encoded_policy(pipeline, policy, classes)
    if policy["mode"] == "auto" and len(classes) == 2:
        policy.setdefault("positive_class", classes[-1])
    if policy["mode"] == "manual":
        thresholds = manual_thresholds(policy, classes)
        pipeline._decision_threshold_evidence = {"mode": "manual"}
    else:
        thresholds = _select_thresholds(pipeline, validation, target, policy)
    pipeline._tuned_thresholds = thresholds
    assert pipeline._decision_threshold_evidence is not None
    pipeline._decision_threshold_evidence.update(
        fitting_rows=len(fitting.train),
        thresholds=[
            {"class": raw, "value": float(thresholds[label])}
            for label, raw in zip(classes, original_labels(pipeline, classes).tolist(), strict=True)
        ],
        model_positive_class=policy.get("positive_class"),
        positive_class=(
            original_labels(pipeline, [policy["positive_class"]]).tolist()[0]
            if "positive_class" in policy
            else None
        ),
    )
    return pipeline


def _encoded_policy(pipeline: SkyulfPipeline, policy: dict, classes: list) -> dict:
    """Bind original user labels to the fitted codes without mutating the saved recipe."""
    result = deepcopy(policy)
    if "positive_class" in result:
        result["positive_class"] = encoded_label(pipeline, result["positive_class"], classes)
    if "thresholds" in result:
        for entry in result["thresholds"]:
            entry["class"] = encoded_label(pipeline, entry["class"], classes)
    return result


def _calibration_data(
    copied: dict, data: SplitDataset, target: str, policy: dict
) -> tuple[SplitDataset, Any]:
    """Separate calibration membership before projecting model-only columns."""
    fitting, validation = data, None
    if policy["mode"] == "manual":
        fitting = replace(
            data,
            train=_model_frame(data.train, copied, target),
            test=_model_frame(data.test, copied, target),
        )
    if policy["mode"] == "auto":
        frame = data.train
        if not isinstance(frame, pd.DataFrame | pl.DataFrame):
            raise ValueError("Decision threshold calibration requires a native training frame.")
        train, valid = calibration_partition(
            frame, target, policy, copied.get("decision_threshold_context", {})
        )
        weights = validate_sample_weight(data.train_sample_weight, len(frame))
        train_frame = _model_frame(take_rows(frame, train), copied, target)
        validation = _model_frame(take_rows(frame, valid), copied, target)
        fitting = SplitDataset(
            train=train_frame,
            test=train_frame.head(0),
            train_sample_weight=None if weights is None else weights[train],
        )
    return fitting, validation
