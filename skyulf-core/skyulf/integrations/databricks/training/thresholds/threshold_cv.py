"""Evaluate complete decision policies on independent outer validation folds."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from .....data.dataset import SplitDataset
from .....inference.local_evaluation import evaluate_local_holdout
from .....modeling._sample_weights import validate_sample_weight
from .....modeling._tuning.cv_policy import (
    effective_cv_type,
    policy_description,
    prepare_policy_data,
    take_rows,
)
from .....modeling._tuning.splitters import nested_inner_folds
from ...scoring.batch.local_batch import fit_local_workflow
from ...shared._local_frames import frame_bytes
from ..tuning.local_search import prepare_search_pipeline

if TYPE_CHECKING:
    from ..tuning.local_cv import LocalCVSpec


def _fold_config(
    config: dict, cv: LocalCVSpec, policy: Any, target: str, event: str | None, columns: list
) -> dict:
    """Keep model selection and threshold calibration inside each outer training fold."""
    config = deepcopy(config)
    inner = cv
    if cv.method == "nested_cv":
        inner = replace(
            cv,
            method=effective_cv_type(policy, "classification"),
            folds=nested_inner_folds(policy),
            nested_type="auto",
            inner_folds=None,
        )
        # The caller validated the outer contract. Rebind its generated wrapper
        # fields to the inner policy without weakening user configuration validation.
        config["modeling"] = {
            key: value for key, value in config["modeling"].items() if not key.startswith("cv_")
        }
    result = prepare_search_pipeline(config, inner, target_column=target, event_column=event)
    context = result.setdefault("decision_threshold_context", {})
    context.setdefault(
        "input_columns", [name for name in columns if name not in (target, event, cv.group_column)]
    )
    context.setdefault("group_column", cv.group_column)
    if cv.temporal:
        context.update(split_strategy="temporal", event_column=event, gap=cv.gap)
    return result


def _evaluate_fold(
    frame: Any, config: dict, train: Any, test: Any, weights: Any, target: str, directory: Path
) -> tuple[dict, dict]:
    """Reload the exact persisted decision model before scoring untouched outer rows."""
    training = take_rows(frame, train)
    artifact = fit_local_workflow(
        deepcopy(config),
        SplitDataset(
            train=training,
            test=training.head(0),
            train_sample_weight=None if weights is None else weights[train],
        ),
        target_column=target,
        artifact_path=directory,
        max_rows=len(frame),
        max_bytes=frame_bytes(frame) + 1,
    )
    metrics = evaluate_local_holdout(artifact, take_rows(frame, test), target_column=target)
    return {
        name.removeprefix("heldout_"): value for name, value in metrics.items()
    }, artifact.pipeline._decision_threshold_evidence or {}


def evaluate_threshold_cv(
    frame: Any,
    config: dict,
    cv: LocalCVSpec,
    *,
    target_column: str,
    event_column: str | None = None,
    sample_weight: Any = None,
) -> dict:
    """Fit preprocessing, search and calibration afresh for every shared outer split."""
    from ..competition.competition_evaluation import (  # noqa: PLC0415 - shared policy owner also dispatches threshold CV
        build_cv_policy,
        build_fold_plan,
        fold_membership_digest,
    )

    policy = build_cv_policy(cv, "balanced_accuracy", event_column)
    features = (
        frame.drop(target_column)
        if isinstance(frame, pl.DataFrame)
        else frame.drop(columns=target_column)
    )
    _, labels, metadata, positions = prepare_policy_data(
        features, frame[target_column], policy, "classification", return_positions=True
    )
    plan = build_fold_plan(policy, "classification", labels, metadata)
    weights = validate_sample_weight(sample_weight, len(frame))
    fold_config = _fold_config(config, cv, policy, target_column, event_column, list(frame.columns))
    scores, evidence = [], []
    with TemporaryDirectory(prefix="skyulf-threshold-cv-") as directory:
        for index, (train, test) in enumerate(plan.partitions):
            metrics, selected = _evaluate_fold(
                frame,
                fold_config,
                positions[train],
                positions[test],
                weights,
                target_column,
                Path(directory) / str(index),
            )
            scores.append(metrics)
            evidence.append(selected)
    return {
        "aggregated_metrics": _aggregate(scores),
        "fold_results": scores,
        "decision_threshold_folds": evidence,
        "cv_config": {"n_folds": cv.folds, "cv_type": cv.method, "time_column": event_column},
        "fold_membership_sha256": fold_membership_digest(frame, policy, plan),
        "split_policy": policy_description(policy, "classification"),
        "folds": [
            {"fold": index + 1, "metrics": score, "split": plan.evidence[index]}
            for index, score in enumerate(scores)
        ],
        "evaluation_mode": "threshold_pipeline_cv",
    }


def _aggregate(scores: list[dict]) -> dict:
    """Aggregate only common finite metrics without accepting partial fold success."""
    common = set.intersection(*(set(score) for score in scores))
    aggregated = {
        name: {
            "mean": float(np.mean([s[name] for s in scores])),
            "std": float(np.std([s[name] for s in scores])),
        }
        for name in sorted(common)
    }
    if not aggregated or not all(
        np.isfinite(value) for stats in aggregated.values() for value in stats.values()
    ):
        raise ValueError("Decision threshold CV requires finite metrics in every fold.")
    return aggregated
