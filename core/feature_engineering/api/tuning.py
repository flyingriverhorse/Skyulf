import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.engine import get_async_session
from core.database.models import HyperparameterTuningJob
from core.feature_engineering.modeling.hyperparameter_tuning.jobs.service import (
    create_tuning_job,
)
from core.feature_engineering.modeling.hyperparameter_tuning.jobs.repository import (
    get_tuning_job,
    list_tuning_jobs,
)
from core.feature_engineering.modeling.hyperparameter_tuning.jobs.status import (
    update_tuning_job_status,
)
from core.feature_engineering.modeling.hyperparameter_tuning.tasks import (
    dispatch_hyperparameter_tuning_job,
)
from core.feature_engineering.schemas import (
    HyperparameterTuningJobBatchResponse,
    HyperparameterTuningJobCreate,
    HyperparameterTuningJobResponse,
    HyperparameterTuningJobListResponse,
    HyperparameterTuningJobStatus,
    HyperparameterTuningJobSummary,
)
from core.feature_engineering.modeling.config.hyperparameters import (
    get_hyperparameters_for_model,
    get_default_hyperparameters,
)

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/hyperparameter-tuning-jobs",
    response_model=HyperparameterTuningJobBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def enqueue_tuning_job(
    payload: HyperparameterTuningJobCreate,
    session: AsyncSession = Depends(get_async_session),
) -> HyperparameterTuningJobBatchResponse:
    """Create a new hyperparameter tuning job."""
    created_jobs = []
    try:
        for model_type in payload.model_types:
            job = await create_tuning_job(session, payload, model_type_override=model_type)
            created_jobs.append(job)
            
            if payload.run_tuning:
                try:
                    dispatch_hyperparameter_tuning_job(str(job.id))
                except Exception as exc:
                    await update_tuning_job_status(
                        session,
                        job,
                        status=HyperparameterTuningJobStatus.FAILED,
                        error_message="Failed to enqueue tuning job",
                    )
                    raise HTTPException(
                        status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Failed to enqueue tuning job",
                    ) from exc

    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    response_jobs = [
        HyperparameterTuningJobResponse.model_validate(job, from_attributes=True)
        for job in created_jobs
    ]
    return HyperparameterTuningJobBatchResponse(jobs=response_jobs, total=len(response_jobs))

@router.get(
    "/hyperparameter-tuning-jobs/{job_id}",
    response_model=HyperparameterTuningJobResponse,
)
async def get_tuning_job_detail(
    job_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> HyperparameterTuningJobResponse:
    """Return a single tuning job."""
    job = await get_tuning_job(session, job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Tuning job not found")
    return HyperparameterTuningJobResponse.model_validate(job, from_attributes=True)

@router.get(
    "/hyperparameter-tuning-jobs",
    response_model=HyperparameterTuningJobListResponse,
)
async def list_tuning_job_records(
    pipeline_id: Optional[str] = Query(default=None),
    node_id: Optional[str] = Query(default=None),
    dataset_source_id: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
) -> HyperparameterTuningJobListResponse:
    """Return recent tuning jobs."""
    jobs = await list_tuning_jobs(
        session,
        dataset_source_id=dataset_source_id,
        pipeline_id=pipeline_id,
        node_id=node_id,
        limit=limit,
    )
    summaries = [HyperparameterTuningJobSummary.model_validate(job, from_attributes=True) for job in jobs]
    return HyperparameterTuningJobListResponse(jobs=summaries, total=len(summaries))

@router.get("/model-hyperparameters/{model_type}")
async def get_model_hyperparameters(
    model_type: str,
) -> Dict[str, Any]:
    """Return hyperparameter configuration for a specific model type."""
    try:
        fields = get_hyperparameters_for_model(model_type)
        defaults = get_default_hyperparameters(model_type)

        return {
            "model_type": model_type,
            "fields": fields,
            "defaults": defaults,
        }
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model type '{model_type}' not found",
        )


@router.get("/hyperparameter-tuning/best-params/{model_type}")
async def get_best_hyperparameters_for_model(
    model_type: str,
    pipeline_id: Optional[str] = Query(default=None),
    dataset_source_id: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    """
    Return the best hyperparameters from the most recent successful tuning job
    for a specific model type.

    This endpoint allows the Train Model node to check if there are tuned
    hyperparameters available for the currently selected model type and display
    an "Apply Best Params" button when applicable.
    """

    # Build query to find the most recent successful tuning job for this model type
    stmt = select(HyperparameterTuningJob).where(
        HyperparameterTuningJob.model_type == model_type,
        HyperparameterTuningJob.status == HyperparameterTuningJobStatus.SUCCEEDED.value,
    )

    # Apply optional filters
    if pipeline_id:
        stmt = stmt.where(HyperparameterTuningJob.pipeline_id == pipeline_id)
    if dataset_source_id:
        stmt = stmt.where(HyperparameterTuningJob.dataset_source_id == dataset_source_id)

    # Order by most recent and get the first result
    stmt = stmt.order_by(HyperparameterTuningJob.finished_at.desc()).limit(1)

    result = await session.execute(stmt)
    job = result.scalar_one_or_none()

    if not job:
        return {
            "available": False,
            "model_type": model_type,
            "message": f"No successful tuning results found for model type '{model_type}'",
        }

    return {
        "available": True,
        "model_type": model_type,
        "job_id": job.id,
        "pipeline_id": job.pipeline_id,
        "node_id": job.node_id,
        "run_number": job.run_number,
        "best_params": job.best_params or {},
        "best_score": job.best_score,
        "scoring": job.scoring,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "search_strategy": job.search_strategy,
        "n_iterations": job.n_iterations,
    }
