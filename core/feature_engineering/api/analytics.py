from typing import Optional, Dict, Any
from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from core.database.engine import get_async_session
from core.feature_engineering.execution.recommendations import (
    generate_binned_distribution_response,
    prepare_categorical_recommendation_context,
)
from core.feature_engineering.preprocessing.inspection import build_quick_profile_payload
from core.feature_engineering.execution.data import build_eda_service
from core.feature_engineering.schemas import BinnedDistributionResponse, QuickProfileResponse

router = APIRouter()

class BinnedDistributionRequest(BaseModel):
    dataset_source_id: str
    sample_size: Optional[int] = 500
    graph: Optional[Dict[str, Any]] = None
    target_node_id: Optional[str] = None

@router.get("/binned-distribution", response_model=BinnedDistributionResponse)
async def get_binned_distribution(
    dataset_source_id: str = Query(..., alias="dataset_source_id"),
    sample_size: int = Query(500, alias="sample_size"),
    graph: Optional[str] = Query(None, alias="graph"),
    target_node_id: Optional[str] = Query(None, alias="target_node_id"),
    db: AsyncSession = Depends(get_async_session),
) -> BinnedDistributionResponse:
    return await generate_binned_distribution_response(
        session=db,
        dataset_source_id=dataset_source_id,
        sample_size=sample_size,
        graph_input=graph,
        target_node_id=target_node_id,
    )

@router.post("/binned-distribution", response_model=BinnedDistributionResponse)
async def post_binned_distribution(
    payload: BinnedDistributionRequest,
    db: AsyncSession = Depends(get_async_session),
) -> BinnedDistributionResponse:
    return await generate_binned_distribution_response(
        session=db,
        dataset_source_id=payload.dataset_source_id,
        sample_size=payload.sample_size or 500,
        graph_input=payload.graph,
        target_node_id=payload.target_node_id,
    )

@router.get("/quick-profile", response_model=QuickProfileResponse)
async def get_quick_profile(
    dataset_source_id: str = Query(..., alias="dataset_source_id"),
    sample_size: int = Query(500, alias="sample_size"),
    graph: Optional[str] = Query(None, alias="graph"),
    target_node_id: Optional[str] = Query(None, alias="target_node_id"),
    db: AsyncSession = Depends(get_async_session),
) -> QuickProfileResponse:
    """
    Generate a lightweight profile of the dataset.
    """
    graph_payload = None
    if graph:
        import json
        try:
            graph_payload = json.loads(graph)
        except:
            pass

    normalized_sample_size = 0 if sample_size == 0 else max(50, sample_size)
    
    eda_service = build_eda_service(db, normalized_sample_size)
    frame, column_metadata, notes = await prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=dataset_source_id,
        sample_size=normalized_sample_size,
        graph=graph_payload,
        target_node_id=target_node_id,
    )
    
    if frame.empty:
         return QuickProfileResponse(
             dataset_source_id=dataset_source_id,
             row_count=0,
             column_count=0,
             columns=[],
             memory_usage_mb=0.0,
         )

    return build_quick_profile_payload(frame, dataset_source_id)
