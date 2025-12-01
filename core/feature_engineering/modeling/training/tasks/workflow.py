"""Celery workflow orchestration for training jobs."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, cast

from config import get_settings
from core.database.models import get_database_session
from core.feature_engineering.schemas import TrainingJobStatus

from ...shared import (
    celery_app,
    _build_metadata_update,
    _ensure_database_ready,
    _parse_cross_validation_config,
    _prepare_training_data,
    _resolve_training_inputs,
)
from ..jobs import get_training_job, update_job_status, update_job_progress_sync
from .execution import _train_and_save_model

logger = logging.getLogger(__name__)
_settings = get_settings()


async def _run_training_workflow(job_id: str) -> None:
    await _ensure_database_ready()

    async with get_database_session(expire_on_commit=False) as session:
        job = await get_training_job(session, job_id)
        if job is None:
            logger.warning("Training job %s not found; skipping", job_id)
            return

        try:
            job = await update_job_status(session, job, status=TrainingJobStatus.RUNNING)
            frame, resolved_node_config, dataset_meta, upstream_order = await _resolve_training_inputs(session, job)
            node_config: Dict[str, Any] = cast(Dict[str, Any], resolved_node_config)

            def _config_value(key: str) -> Any:
                return node_config[key] if key in node_config else None

            job_metadata: Dict[str, Any] = job.job_metadata if isinstance(job.job_metadata, dict) else {}

            target_column = (
                _config_value("target_column")
                or _config_value("targetColumn")
                or job_metadata.get("target_column")
            )
            if not target_column:
                raise ValueError("Training configuration missing target column")

            hyperparameters: Dict[str, Any] = {}
            hyperparameters_config = _config_value("hyperparameters")
            if isinstance(hyperparameters_config, dict):
                hyperparameters.update(hyperparameters_config)
            if isinstance(job.hyperparameters, dict):
                hyperparameters.update(job.hyperparameters)

            problem_type_hint = _config_value("problem_type") or _config_value("problemType") or "auto"

            model_type_value = str(job.model_type or "").strip()
            if not model_type_value:
                raise ValueError("Training job missing model_type")

            pipeline_id_value = str(job.pipeline_id or "").strip()
            if not pipeline_id_value:
                raise ValueError("Training job missing pipeline_id")

            job_id_value = str(job.id)
            version_value = int(job.version) if job.version is not None else None
            if version_value is None:
                raise ValueError("Training job missing version")

            (
                X_train,
                y_train,
                X_validation,
                y_validation,
                X_test,
                y_test,
                feature_columns,
                target_meta,
            ) = _prepare_training_data(frame, target_column)

            cv_config = _parse_cross_validation_config(node_config)

            artifact_uri, metrics, resolved_problem_type, training_warnings = _train_and_save_model(
                model_type=model_type_value,
                hyperparameters=hyperparameters or None,
                problem_type_hint=problem_type_hint,
                target_column=target_column,
                feature_columns=feature_columns,
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                X_test=X_test,
                y_test=y_test,
                artifact_root=_settings.TRAINING_ARTIFACT_DIR,
                pipeline_id=pipeline_id_value,
                job_id=job_id_value,
                version=version_value,
                cv_config=cv_config,
                upstream_node_order=upstream_order,
                progress_callback=lambda p, s: update_job_progress_sync(job.id, p, s),
            )

            metrics["problem_type"] = resolved_problem_type
            if dataset_meta:
                try:
                    metrics.setdefault("dataset", {}).update(dataset_meta)
                except AttributeError:
                    metrics["dataset"] = dataset_meta
            metadata_update = _build_metadata_update(
                resolved_problem_type=resolved_problem_type,
                target_column=target_column,
                feature_columns=feature_columns,
                cv_config=cv_config,
                dataset_meta=dataset_meta,
            )
            metadata_update["target_encoding"] = target_meta

            if training_warnings:
                metadata_update["warnings"] = training_warnings

            job = await update_job_status(
                session,
                job,
                status=TrainingJobStatus.SUCCEEDED,
                metrics=metrics,
                artifact_uri=artifact_uri,
                metadata=metadata_update,
            )

            logger.info(
                "Training job %s succeeded (model=%s, version=%s, pipeline=%s, node=%s)",
                job.id,
                model_type_value,
                job.version,
                job.pipeline_id,
                job.node_id,
            )
        except Exception as exc:  # pragma: no cover - defensive guard for worker runtime
            logger.exception("Training job %s failed", job_id)
            await update_job_status(
                session,
                job,
                status=TrainingJobStatus.FAILED,
                error_message=str(exc),
            )


@celery_app.task(name="core.feature_engineering.modeling.training.train_model")
def train_model(job_id: str) -> None:
    """Celery entrypoint for training jobs."""

    asyncio.run(_run_training_workflow(job_id))


def dispatch_training_job(job_id: str) -> None:
    """Queue a Celery task for the given job identifier."""

    train_model.apply_async(args=[job_id], task_id=job_id)


def cancel_training_task(job_id: str) -> None:
    """
    Send a revocation signal to the Celery worker for this job.
    
    Uses terminate=True to force kill the worker process if it's blocking.
    """
    celery_app.control.revoke(job_id, terminate=True)


__all__ = [
    "_run_training_workflow",
    "train_model",
    "dispatch_training_job",
    "cancel_training_task",
]
