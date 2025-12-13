from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from backend.database.engine import get_async_session
from .schemas import ModelRegistryEntry, RegistryStats, ModelVersion
from .service import ModelRegistryService

router = APIRouter(prefix="/registry", tags=["Model Registry"])

@router.get("/stats", response_model=RegistryStats)
async def get_registry_stats(
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get statistics for the model registry.
    """
    return await ModelRegistryService.get_registry_stats(session)

@router.get("/models", response_model=List[ModelRegistryEntry])
async def list_models(
    skip: int = 0,
    limit: int = 10,
    session: AsyncSession = Depends(get_async_session)
):
    """
    List all models in the registry.
    """
    return await ModelRegistryService.list_models(session, skip=skip, limit=limit)

@router.get("/models/{model_type}/versions", response_model=List[ModelVersion])
async def get_model_versions(
    model_type: str,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get all versions for a specific model type.
    """
    return await ModelRegistryService.get_model_versions(session, model_type)
