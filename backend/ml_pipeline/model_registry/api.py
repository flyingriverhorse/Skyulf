"""Read-only model-registry endpoints: stats, models, versions and job artifacts."""

import logging
import traceback

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.exceptions.core import SkyulfException
from backend.utils.logging_utils import redact_credentials, sanitize_for_log

logger = logging.getLogger(__name__)

from backend.database.engine import get_async_session

from .schemas import ArtifactListResponse, ModelRegistryEntry, ModelVersion, RegistryStats
from .service import ModelRegistryService

router = APIRouter(prefix="/registry", tags=["Model Registry"])


@router.get("/stats", response_model=RegistryStats)
async def get_registry_stats(session: AsyncSession = Depends(get_async_session)):
    """Get statistics for the model registry."""
    return await ModelRegistryService.get_registry_stats(session)


@router.get("/models", response_model=list[ModelRegistryEntry])
async def list_models(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=10, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
):
    """List all models in the registry."""
    return await ModelRegistryService.list_models(session, skip=skip, limit=limit)


@router.get("/models/{model_type}/versions", response_model=list[ModelVersion])
async def get_model_versions(model_type: str, session: AsyncSession = Depends(get_async_session)):
    """Get all versions for a specific model type."""
    return await ModelRegistryService.get_model_versions(session, model_type)


@router.get("/artifacts/{job_id}", response_model=ArtifactListResponse)
async def list_job_artifacts(job_id: str, session: AsyncSession = Depends(get_async_session)):
    """List artifacts for a specific job."""
    try:
        return await ModelRegistryService.get_job_artifacts(session, job_id)
    except ValueError as e:
        logger.warning(
            "Artifact lookup failed for job %s: %s",
            sanitize_for_log(redact_credentials(job_id)),
            sanitize_for_log(redact_credentials(e)),
        )
        raise HTTPException(status_code=404, detail="Job artifacts not found") from e
    except Exception as e:  # noqa: BLE001 - final artifact HTTP boundary, sanitized and redacted
        # Format and scrub the complete chain before any handler can render it.
        safe_traceback = sanitize_for_log(
            redact_credentials("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        )
        logger.error(
            "Failed to list artifacts for job %s: %s",
            sanitize_for_log(redact_credentials(job_id)),
            safe_traceback,
        )
        raise SkyulfException(message="Failed to retrieve job artifacts") from None
