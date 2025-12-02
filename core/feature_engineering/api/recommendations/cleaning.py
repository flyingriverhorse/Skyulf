from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.engine import get_async_session
from core.feature_engineering.execution.data import build_eda_service
from core.feature_engineering.execution.recommendations import (
    prepare_categorical_recommendation_context,
    DropColumnRecommendationBuilder,
)
from core.feature_engineering.schemas import DropColumnRecommendations

from .schemas import RecommendationRequest
from .utils import _get_target_column_from_graph

router = APIRouter()

@router.post("/drop-columns", response_model=DropColumnRecommendations)
async def recommend_drop_columns(
    request: RecommendationRequest = Body(...),
    db: AsyncSession = Depends(get_async_session),
) -> DropColumnRecommendations:
    """
    Analyze the dataset (after applying the graph up to target_node_id)
    and recommend columns to drop based on:
      - High missing values (> 95%)
      - Single unique value (constant columns)
      - High correlation (if we implement that check)
    """
    eda_service = build_eda_service(db, request.sample_size)

    # 1. Get the frame (with graph applied) to know which columns are currently available
    frame, _, _ = await prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=request.dataset_source_id,
        sample_size=request.sample_size,
        graph=request.graph,
        target_node_id=request.target_node_id,
    )

    if frame.empty:
        return DropColumnRecommendations(
            dataset_source_id=request.dataset_source_id,
            candidates=[],
            all_columns=[],
            available_filters=[],
            column_missing_map={},
            suggested_threshold=40.0,
        )

    # 2. Get quality report for the source dataset (contains missingness stats etc.)
    quality_payload = await eda_service.quality_report(request.dataset_source_id, sample_size=request.sample_size)
    quality_report = quality_payload.get("quality_report") or {}
    quality_metrics = quality_report.get("quality_metrics") or {}

    # 3. Build recommendations using the robust builder
    builder = DropColumnRecommendationBuilder()

    # Ingest missing summary
    missing_summary = quality_metrics.get("missing_value_summary") or []
    builder.ingest_missing_summary(missing_summary)

    # Ingest EDA recommendations
    eda_recommendations = quality_report.get("recommendations") or []
    builder.ingest_eda_recommendations(eda_recommendations)

    # Collect column details
    column_details = quality_metrics.get("column_details") or []
    builder.collect_column_details(column_details)

    # Collect sample preview columns
    builder.collect_sample_preview(quality_report)

    # 4. Generate payload and filter by currently available columns
    candidates = builder.build_candidate_payload()
    allowed_columns = set(frame.columns)

    # Filter out target column if configured
    target_column = _get_target_column_from_graph(request.graph)
    if target_column:
        if target_column in allowed_columns:
            allowed_columns.remove(target_column)
        # Handle both object and dict access for candidates
        candidates = [
            c for c in candidates 
            if (c.get('column') if isinstance(c, dict) else c.column) != target_column
        ]
    
    # Filter candidates to only include columns that exist in the current frame
    filtered_candidates = builder.filter_candidates(candidates, allowed_columns)
    
    # Sort candidates
    builder.sort_candidates(filtered_candidates)
    
    # Finalize all columns list (ensuring we only list columns in the frame)
    all_columns = builder.finalize_all_columns(filtered_candidates, allowed_columns)
    
    # Build filters based on the filtered candidates
    filters = builder.build_filters(filtered_candidates)
    
    # Build missing map for the response
    column_missing_map = builder.build_column_missing_map(all_columns)

    return DropColumnRecommendations(
        dataset_source_id=request.dataset_source_id,
        candidates=filtered_candidates,
        all_columns=all_columns,
        available_filters=[{"id": f.id, "label": f.label, "count": f.count, "description": f.description} for f in filters],
        column_missing_map=column_missing_map,
        suggested_threshold=builder.suggested_threshold(),
    )
