"""
Threshold Tuning Service
------------------------
Service for previewing, saving, toggling, and clearing per-job tuned
decision thresholds. Tuning always operates on raw/undecoded target labels
(via ``EvaluationService._load_raw_evaluation_data``) so the resulting
threshold dict's keys match the live model's actual ``estimator.classes_``
values at predict time.
"""

from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import TrainingJob
from backend.ml_pipeline._services.evaluation_service import EvaluationService
from backend.ml_pipeline._services.prediction_utils import _to_int_like_array
from skyulf.modeling import optimize_thresholds


class ThresholdTuningError(ValueError):
    """Raised for invalid threshold-tuning requests (maps to HTTP 400)."""


_METRIC_SCORERS: dict[str, Any] = {
    "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted", zero_division=0),
    "precision": lambda y_true, y_pred: precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "recall": lambda y_true, y_pred: recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "balanced_accuracy": lambda y_true, y_pred: balanced_accuracy_score(y_true, y_pred),
    "roc_auc": lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
}


class ThresholdTuningService:
    """Preview, save, toggle, and clear tuned classification thresholds for a job."""

    @staticmethod
    def _select_split(evaluation_data: dict) -> tuple[str, dict]:
        """Prefer the ``validation`` split, falling back to ``test`` if validation is absent.

        This is a UI-hint-only policy: no backend validation error is raised
        when validation is missing, the frontend uses ``split_used`` to show
        a hint that a fresh holdout would be preferable.
        """
        splits = evaluation_data.get("splits")
        if not isinstance(splits, dict):
            raise ThresholdTuningError("Job has no evaluation splits to tune against.")

        validation = splits.get("validation")
        if isinstance(validation, dict) and "y_proba" in validation:
            return "validation", validation

        test = splits.get("test")
        if isinstance(test, dict) and "y_proba" in test:
            return "test", test

        raise ThresholdTuningError(
            "Job has no validation or test split with predicted probabilities to tune against."
        )

    @staticmethod
    def _coerce_classes_and_proba(y_proba: dict) -> tuple[np.ndarray, list]:
        """Reconcile stringified ``predict_proba`` classes against their underlying int dtype.

        ``y_proba["classes"]`` is stringified (e.g. ``["0", "1", "2"]``) even
        when the real model classes are integers, so it must be coerced back
        via ``_to_int_like_array`` to match ``y_true``'s dtype.
        """
        raw_classes = np.array(y_proba["classes"])
        coerced = _to_int_like_array(raw_classes)
        classes = coerced.tolist() if coerced is not None else raw_classes.tolist()
        values = np.asarray(y_proba["values"], dtype=float)
        return values, classes

    @staticmethod
    async def _get_job_or_raise(session: AsyncSession, job_id: str) -> TrainingJob:
        """Fetches a `TrainingJob` by id, raising `ThresholdTuningError` if not found."""
        job = (
            await session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        ).scalar_one_or_none()
        if job is None:
            raise ThresholdTuningError(f"Job not found: {job_id}")
        return job

    @staticmethod
    async def preview(session: AsyncSession, job_id: str, metric: str) -> dict:
        """Compute (without saving) tuned per-class thresholds for a job's evaluation data."""
        scorer = _METRIC_SCORERS.get(metric)
        if scorer is None:
            raise ThresholdTuningError(f"Unsupported metric: {metric}")

        await ThresholdTuningService._get_job_or_raise(session, job_id)

        evaluation_data, _artifact_store = await EvaluationService._load_raw_evaluation_data(
            session, job_id
        )
        split_used, split_data = ThresholdTuningService._select_split(evaluation_data)

        y_true = np.asarray(split_data["y_true"])
        y_proba_values, classes = ThresholdTuningService._coerce_classes_and_proba(
            split_data["y_proba"]
        )

        thresholds = optimize_thresholds(y_true, y_proba_values, metric=scorer, classes=classes)

        return {
            "thresholds": {str(k): v for k, v in thresholds.items()},
            "classes": classes,
            "metric": metric,
            "split_used": split_used,
        }

    @staticmethod
    async def save(
        session: AsyncSession,
        job_id: str,
        thresholds: dict[str, float],
        classes: list,
        metric: str,
        split_used: str,
    ) -> bool:
        """Persist a tuned threshold set on the job, enabling it by default."""
        job = await ThresholdTuningService._get_job_or_raise(session, job_id)

        job.tuned_thresholds = {
            "thresholds": thresholds,
            "classes": classes,
            "metric": metric,
            "split_used": split_used,
            "computed_at": datetime.now(UTC).isoformat(),
        }
        job.tuned_thresholds_enabled = True
        await session.commit()
        return True

    @staticmethod
    async def toggle(session: AsyncSession, job_id: str, enabled: bool) -> bool:
        """Enable or disable use of the job's saved tuned thresholds at predict time."""
        job = await ThresholdTuningService._get_job_or_raise(session, job_id)
        if job.tuned_thresholds is None:
            raise ThresholdTuningError("Job has no saved tuned thresholds to toggle.")

        job.tuned_thresholds_enabled = enabled
        await session.commit()
        return True

    @staticmethod
    async def clear(session: AsyncSession, job_id: str) -> bool:
        """Remove any saved tuned thresholds from the job."""
        job = await ThresholdTuningService._get_job_or_raise(session, job_id)

        job.tuned_thresholds = None
        job.tuned_thresholds_enabled = False
        await session.commit()
        return True
