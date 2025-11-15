"""Workflow orchestration for hyperparameter tuning tasks."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from core.database.models import get_database_session

from ...shared import (
    ModelingInputs,
    SearchConfiguration,
    _build_cv_splitter,
    _build_metadata_update,
    _build_search_configuration,
    _ensure_database_ready,
    _extract_problem_type_hint,
    _extract_target_column,
    _load_modeling_inputs,
    _persist_best_estimator,
    _resolve_cv_config,
    _resolve_model_spec_from_job,
    _resolve_problem_type_hint,
    _serialize_value,
    celery_app,
)
from ..jobs import get_tuning_job, update_tuning_job_status
from .data_bundle import TrainingDataBundle, _build_training_data_bundle
from .execution import _execute_search
from .searchers import _build_searcher, _prepare_search_parameters

from core.feature_engineering.schemas import HyperparameterTuningJobStatus

logger = logging.getLogger(__name__)


async def _run_hyperparameter_tuning_workflow(job_id: str) -> None:
    await _ensure_database_ready()

    async with get_database_session(expire_on_commit=False) as session:
        job = await get_tuning_job(session, job_id)
        if job is None:
            logger.warning("Tuning job %s not found; skipping", job_id)
            return

        try:
            job = await update_tuning_job_status(
                session,
                job,
                status=HyperparameterTuningJobStatus.RUNNING,
            )

            inputs: ModelingInputs = await _load_modeling_inputs(session, job)
            target_column = _extract_target_column(inputs.node_config, job)
            problem_type_hint = _extract_problem_type_hint(inputs.node_config)
            spec = _resolve_model_spec_from_job(job)
            training_data = _build_training_data_bundle(inputs.frame, target_column)
            search_config = _build_search_configuration(job, inputs.node_config)
            resolved_problem_type = _resolve_problem_type_hint(problem_type_hint, spec.problem_type)

            scoring_override = search_config.scoring
            if (
                resolved_problem_type == "classification"
                and isinstance(scoring_override, str)
                and scoring_override.strip()
            ):
                normalized_scoring = scoring_override.strip()
                n_classes = _infer_target_class_count(training_data)

                if n_classes is not None and n_classes > 2:
                    mapping = {
                        "f1": "f1_weighted",
                        "precision": "precision_weighted",
                        "recall": "recall_weighted",
                    }
                    lower = normalized_scoring.lower()
                    if lower in mapping and normalized_scoring != mapping[lower]:
                        logger.info(
                            "Adjusting scoring metric '%s' -> '%s' for multiclass target (%s classes)",
                            normalized_scoring,
                            mapping[lower],
                            n_classes,
                        )
                        search_config = SearchConfiguration(
                            strategy=search_config.strategy,
                            selected_strategy=search_config.selected_strategy,
                            search_space=search_config.search_space,
                            n_iterations=search_config.n_iterations,
                            scoring=mapping[lower],
                            random_state=search_config.random_state,
                            cross_validation=search_config.cross_validation,
                        )

            base_params, filtered_search_space = _prepare_search_parameters(
                job,
                inputs.node_config,
                spec,
                search_config,
            )

            cv_config = _resolve_cv_config(search_config)
            splitter = _build_cv_splitter(resolved_problem_type, cv_config, training_data.y_train)

            searcher = _build_searcher(
                spec,
                search_config,
                base_params,
                filtered_search_space,
                splitter,
            )

            search_result = _execute_search(
                searcher,
                training_data,
                spec.key,
                resolved_problem_type,
                target_column,
                search_config,
                cv_config,
            )

            artifact_uri = _persist_best_estimator(job, search_result.best_estimator)
            if artifact_uri:
                search_result.metrics["artifact_uri"] = artifact_uri

            metadata_update = _build_metadata_update(
                resolved_problem_type,
                target_column,
                training_data.feature_columns,
                cv_config,
                inputs.dataset_meta,
            )

            if search_result.warnings:
                metadata_update["warnings"] = search_result.warnings

            await update_tuning_job_status(
                session,
                job,
                status=HyperparameterTuningJobStatus.SUCCEEDED,
                metrics=search_result.metrics,
                results=search_result.summary,
                best_params={
                    key: _serialize_value(value)
                    for key, value in (search_result.searcher.best_params_ or {}).items()
                },
                best_score=float(search_result.searcher.best_score_),
                artifact_uri=artifact_uri,
                metadata=metadata_update,
            )
        except Exception as exc:  # pragma: no cover - defensive guard for worker runtime
            logger.exception("Tuning job %s failed", job_id)
            await update_tuning_job_status(
                session,
                job,
                status=HyperparameterTuningJobStatus.FAILED,
                error_message=str(exc),
            )


def _infer_target_class_count(training_data: TrainingDataBundle) -> int | None:
    n_classes = None
    try:
        if isinstance(training_data.target_meta, dict):
            cats = training_data.target_meta.get("categories")
            if isinstance(cats, (list, tuple)):
                n_classes = len(cats)
    except Exception:
        n_classes = None
    if n_classes is None:
        try:
            n_classes = int(len(training_data.y_train.astype(int).unique()))
        except Exception:
            n_classes = None
    return n_classes


@celery_app.task(name="core.feature_engineering.modeling.hyperparameter_tuning.run")
def run_hyperparameter_tuning(job_id: str) -> None:
    """Celery entrypoint for hyperparameter tuning jobs."""

    asyncio.run(_run_hyperparameter_tuning_workflow(job_id))


def dispatch_hyperparameter_tuning_job(job_id: str) -> None:
    """Queue a hyperparameter tuning job via Celery."""

    run_hyperparameter_tuning.delay(job_id)


__all__ = [
    "_run_hyperparameter_tuning_workflow",
    "run_hyperparameter_tuning",
    "dispatch_hyperparameter_tuning_job",
]
