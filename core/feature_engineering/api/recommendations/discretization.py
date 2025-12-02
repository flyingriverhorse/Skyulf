from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.engine import get_async_session
from core.feature_engineering.execution.data import build_eda_service
from core.feature_engineering.execution.recommendations import (
    prepare_categorical_recommendation_context,
)
from core.feature_engineering.schemas import BinningRecommendationsResponse
from core.feature_engineering.preprocessing.bucketing import _build_binning_recommendations

from .schemas import RecommendationRequest
from .utils import _get_target_column_from_graph

router = APIRouter()

@router.post("/binning", response_model=BinningRecommendationsResponse)
async def recommend_binning(
    request: RecommendationRequest = Body(...),
    db: AsyncSession = Depends(get_async_session),
) -> BinningRecommendationsResponse:
    """
    Return binning recommendations for numeric features.
    """
    eda_service = build_eda_service(db, request.sample_size)
    frame, column_metadata, notes = await prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=request.dataset_source_id,
        sample_size=request.sample_size,
        graph=request.graph,
        target_node_id=request.target_node_id,
    )

    recommendations, excluded = _build_binning_recommendations(frame)

    # Filter out target column if configured
    target_column = _get_target_column_from_graph(request.graph)
    if target_column:
        recommendations = [rec for rec in recommendations if rec.column != target_column]
        excluded = [ex for ex in excluded if ex.column != target_column]

    return BinningRecommendationsResponse(
        dataset_source_id=request.dataset_source_id,
        sample_size=int(frame.shape[0]),
        columns=recommendations,
        excluded_columns=excluded,
    )
