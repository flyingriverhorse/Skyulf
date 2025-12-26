from typing import AsyncGenerator, List, cast

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import get_database_session

from .schemas import DeploymentInfo, PredictionRequest, PredictionResponse
from .service import DeploymentService

router = APIRouter(prefix="/deployment", tags=["Deployment"])


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with get_database_session() as session:
        yield session


@router.post("/deploy/{job_id}", response_model=DeploymentInfo)
async def deploy_model(job_id: str, session: AsyncSession = Depends(get_async_session)):
    """
    Deploys a model from a completed job.
    """
    try:
        deployment = await DeploymentService.deploy_model(session, job_id)
        # Enrich with schema
        return await DeploymentService.get_deployment_details(session, deployment)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active", response_model=DeploymentInfo)
async def get_active_deployment(session: AsyncSession = Depends(get_async_session)):
    """
    Returns the currently active deployment.
    """
    deployment = await DeploymentService.get_active_deployment(session)
    if not deployment:
        raise HTTPException(status_code=404, detail="No active deployment found")
    
    # Enrich with schema
    return await DeploymentService.get_deployment_details(session, deployment)


@router.get("/history", response_model=List[DeploymentInfo])
async def list_deployments(
    limit: int = 50, skip: int = 0, session: AsyncSession = Depends(get_async_session)
):
    """
    Lists deployment history.
    """
    deployments = await DeploymentService.list_deployments(session, limit, skip)
    return deployments


@router.post("/deactivate")
async def deactivate_deployment(session: AsyncSession = Depends(get_async_session)):
    """
    Deactivates the currently active deployment.
    """
    await DeploymentService.deactivate_current_deployment(session)
    return {"status": "success", "message": "Deployment deactivated"}


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest, session: AsyncSession = Depends(get_async_session)
):
    """
    Makes predictions using the active model.
    """
    try:
        deployment = await DeploymentService.get_active_deployment(session)
        if not deployment:
            raise HTTPException(status_code=404, detail="No active deployment found")

        predictions = await DeploymentService.predict(session, request.data)

        return PredictionResponse(
            predictions=predictions, model_version=cast(str, deployment.job_id)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
