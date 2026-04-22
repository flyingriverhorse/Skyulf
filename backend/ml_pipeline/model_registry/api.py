import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

from backend.database.engine import get_async_session

from .schemas import ArtifactListResponse, ModelRegistryEntry, ModelVersion, RegistryStats
from .service import ModelRegistryService

router = APIRouter(prefix="/registry", tags=["Model Registry"])


@router.get("/stats", response_model=RegistryStats)
async def get_registry_stats(session: AsyncSession = Depends(get_async_session)):
    """
    Get statistics for the model registry.
    """
    return await ModelRegistryService.get_registry_stats(session)


@router.get("/models", response_model=List[ModelRegistryEntry])
async def list_models(
    skip: int = 0, limit: int = 10, session: AsyncSession = Depends(get_async_session)
):
    """
    List all models in the registry.
    """
    return await ModelRegistryService.list_models(session, skip=skip, limit=limit)


@router.get("/models/{model_type}/versions", response_model=List[ModelVersion])
async def get_model_versions(model_type: str, session: AsyncSession = Depends(get_async_session)):
    """
    Get all versions for a specific model type.
    """
    return await ModelRegistryService.get_model_versions(session, model_type)


@router.get("/artifacts/{job_id}", response_model=ArtifactListResponse)
async def list_job_artifacts(job_id: str, session: AsyncSession = Depends(get_async_session)):
    """
    List artifacts for a specific job.
    """
    try:
        return await ModelRegistryService.get_job_artifacts(session, job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to list artifacts for job %s", job_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve job artifacts")
