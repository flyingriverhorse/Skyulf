from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.engine import get_async_session
from core.feature_engineering.execution.data import build_eda_service
from core.feature_engineering.execution.recommendations import (
    prepare_categorical_recommendation_context,
    parse_skewness_transformations,
)
from core.feature_engineering.schemas import (
    OutlierRecommendationsResponse,
    ScalingRecommendationsResponse,
    SkewnessRecommendationsResponse,
)

from core.feature_engineering.preprocessing.statistics import (
    _build_outlier_recommendations,
    _build_scaling_recommendations,
    _build_skewness_recommendations,
    _outlier_method_details,
    _scaling_method_details,
    _skewness_method_details,
    SKEWNESS_THRESHOLD,
    OUTLIER_DEFAULT_METHOD,
)

from .schemas import RecommendationRequest, SkewnessRecommendationRequest
from .utils import _get_target_column_from_graph, _collect_applied_skewness_methods

router = APIRouter()

@router.post("/outliers", response_model=OutlierRecommendationsResponse)
async def recommend_outlier_detection(
    request: RecommendationRequest = Body(...),
    db: AsyncSession = Depends(get_async_session),
) -> OutlierRecommendationsResponse:
    """
    Recommend numeric columns for outlier detection.
    """
    eda_service = build_eda_service(db, request.sample_size)
    frame, column_metadata, notes = await prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=request.dataset_source_id,
        sample_size=request.sample_size,
        graph=request.graph,
        target_node_id=request.target_node_id,
    )

    if frame.empty:
        return OutlierRecommendationsResponse(
            dataset_source_id=request.dataset_source_id,
            sample_size=request.sample_size,
            default_method=OUTLIER_DEFAULT_METHOD,
            methods=[],
            columns=[],
        )
    
    columns = _build_outlier_recommendations(frame)

    # Filter out target column if configured
    target_column = _get_target_column_from_graph(request.graph)
    if target_column:
        columns = [col for col in columns if col != target_column]

    return OutlierRecommendationsResponse(
        dataset_source_id=request.dataset_source_id,
        sample_size=request.sample_size,
        default_method=OUTLIER_DEFAULT_METHOD,
        methods=_outlier_method_details(),
        columns=columns,
    )


@router.post("/scaling", response_model=ScalingRecommendationsResponse)
async def recommend_scaling(
    request: RecommendationRequest = Body(...),
    db: AsyncSession = Depends(get_async_session),
) -> ScalingRecommendationsResponse:
    """
    Recommend numeric columns for scaling.
    """
    eda_service = build_eda_service(db, request.sample_size)
    frame, column_metadata, notes = await prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=request.dataset_source_id,
        sample_size=request.sample_size,
        graph=request.graph,
        target_node_id=request.target_node_id,
    )

    recommendations = _build_scaling_recommendations(frame)

    # Filter out target column if configured
    target_column = _get_target_column_from_graph(request.graph)
    if target_column:
        recommendations = [rec for rec in recommendations if rec.column != target_column]

    return ScalingRecommendationsResponse(
        dataset_source_id=request.dataset_source_id,
        sample_size=int(frame.shape[0]),
        methods=_scaling_method_details(),
        columns=recommendations,
    )


@router.post("/skewness", response_model=SkewnessRecommendationsResponse)
async def recommend_skewness_transform(
    request: SkewnessRecommendationRequest = Body(...),
    db: AsyncSession = Depends(get_async_session),
) -> SkewnessRecommendationsResponse:
    """
    Recommend columns for skewness transformation.
    """
    eda_service = build_eda_service(db, request.sample_size)
    frame, column_metadata, notes = await prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=request.dataset_source_id,
        sample_size=request.sample_size,
        graph=request.graph,
        target_node_id=request.target_node_id,
    )

    # Note: routes.py has _apply_skewness_graph_context logic which we might need if we want full parity.
    # For now we use the frame as is.
    
    selected_methods = parse_skewness_transformations(request.transformations)
    
    # Collect applied methods from the graph history to prevent double application
    applied_methods = _collect_applied_skewness_methods(request.graph, request.target_node_id)
    
    recommendations = _build_skewness_recommendations(frame, selected_methods, applied_methods)

    # Filter out target column if configured
    target_column = _get_target_column_from_graph(request.graph)
    if target_column:
        recommendations = [rec for rec in recommendations if rec.column != target_column]

    return SkewnessRecommendationsResponse(
        dataset_source_id=request.dataset_source_id,
        sample_size=request.sample_size,
        skewness_threshold=SKEWNESS_THRESHOLD,
        methods=_skewness_method_details(),
        columns=recommendations,
    )
