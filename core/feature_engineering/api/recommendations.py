from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.engine import get_async_session
from core.feature_engineering.execution.data import build_eda_service
from core.feature_engineering.execution.recommendations import (
    prepare_categorical_recommendation_context,
    generate_binned_distribution_response,
    parse_skewness_transformations,
    DropColumnRecommendationBuilder,
)
from core.feature_engineering.execution.engine import apply_recommendation_graph
from core.feature_engineering.execution.graph import extract_graph_payload, normalize_target_node

from core.feature_engineering.schemas import (
    BinnedDistributionResponse,
    DropColumnRecommendations,
    LabelEncodingRecommendationsResponse,
    OneHotEncodingRecommendationsResponse,
    OutlierRecommendationsResponse,
    ScalingRecommendationsResponse,
    SkewnessRecommendationsResponse,
    LabelEncodingColumnSuggestion,
    OneHotEncodingColumnSuggestion,
    BinningRecommendationsResponse,
    OrdinalEncodingRecommendationsResponse,
    OrdinalEncodingColumnSuggestion,
    HashEncodingRecommendationsResponse,
    HashEncodingColumnSuggestion,
    TargetEncodingRecommendationsResponse,
    TargetEncodingColumnSuggestion,
    DummyEncodingRecommendationsResponse,
    DummyEncodingColumnSuggestion,
)

from core.feature_engineering.preprocessing.bucketing import _build_binning_recommendations
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

from core.feature_engineering.recommendations import (
    build_label_encoding_suggestions,
    build_one_hot_encoding_suggestions,
    build_ordinal_encoding_suggestions,
    build_hash_encoding_suggestions,
    build_target_encoding_suggestions,
    build_dummy_encoding_suggestions,
)

from core.feature_engineering.preprocessing.encoding.hash_encoding import HASH_ENCODING_DEFAULT_MAX_CATEGORIES
from core.feature_engineering.preprocessing.encoding.ordinal_encoding import ORDINAL_ENCODING_DEFAULT_MAX_CATEGORIES
from core.feature_engineering.preprocessing.encoding.target_encoding import TARGET_ENCODING_DEFAULT_MAX_CATEGORIES
from core.feature_engineering.preprocessing.encoding.dummy_encoding import DUMMY_ENCODING_DEFAULT_MAX_CATEGORIES

router = APIRouter()


class RecommendationRequest(BaseModel):
    dataset_source_id: str
    target_node_id: Optional[str] = None
    sample_size: int = 10000
    graph: Optional[Dict[str, Any]] = None


class SkewnessRecommendationRequest(RecommendationRequest):
    transformations: Optional[str] = None


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


@router.post("/label-encoding", response_model=LabelEncodingRecommendationsResponse)
async def recommend_label_encoding(
    request: RecommendationRequest = Body(...),
    db: AsyncSession = Depends(get_async_session),
) -> LabelEncodingRecommendationsResponse:
    """
    Recommend columns for Label Encoding.
    """
    eda_service = build_eda_service(db, request.sample_size)
    frame, column_metadata, notes = await prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=request.dataset_source_id,
        sample_size=request.sample_size,
        graph=request.graph,
        target_node_id=request.target_node_id,
        skip_catalog_types={
            "target_encoding",
            "ordinal_encoding",
            "label_encoding",
            "one_hot_encoding",
            "dummy_encoding",
            "hash_encoding",
        },
    )

    suggestions = build_label_encoding_suggestions(frame, column_metadata)
    
    # Calculate summary stats
    total_text_columns = len(suggestions)
    recommended_count = sum(1 for item in suggestions if item.status == "recommended" and item.selectable)
    high_cardinality_columns = [
        item.column for item in suggestions if item.status in {"high_cardinality", "too_many_categories"}
    ]

    return LabelEncodingRecommendationsResponse(
        dataset_source_id=request.dataset_source_id,
        sample_size=int(frame.shape[0]),
        total_text_columns=total_text_columns,
        recommended_count=recommended_count,
        auto_detect_default=bool(recommended_count),
        high_cardinality_columns=high_cardinality_columns,
        notes=notes,
        columns=[
            LabelEncodingColumnSuggestion(**suggestion.to_payload())
            for suggestion in suggestions
        ],
    )


@router.post("/one-hot-encoding", response_model=OneHotEncodingRecommendationsResponse)
async def recommend_one_hot_encoding(
    request: RecommendationRequest = Body(...),
    db: AsyncSession = Depends(get_async_session),
) -> OneHotEncodingRecommendationsResponse:
    """
    Recommend columns for One-Hot Encoding.
    """
    eda_service = build_eda_service(db, request.sample_size)
    frame, column_metadata, notes = await prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=request.dataset_source_id,
        sample_size=request.sample_size,
        graph=request.graph,
        target_node_id=request.target_node_id,
        skip_catalog_types={
            "target_encoding",
            "one_hot_encoding",
            "dummy_encoding",
            "ordinal_encoding",
            "hash_encoding",
        },
    )

    suggestions = build_one_hot_encoding_suggestions(frame, column_metadata)

    total_text_columns = len(suggestions)
    recommended_count = sum(1 for item in suggestions if item.status == "recommended" and item.selectable)
    cautioned_count = sum(
        1 for item in suggestions if item.status in {"high_cardinality", "too_many_categories"}
    )
    high_cardinality_columns = [
        item.column for item in suggestions if item.status in {"high_cardinality", "too_many_categories"}
    ]

    return OneHotEncodingRecommendationsResponse(
        dataset_source_id=request.dataset_source_id,
        sample_size=int(frame.shape[0]),
        total_text_columns=total_text_columns,
        recommended_count=recommended_count,
        cautioned_count=cautioned_count,
        high_cardinality_columns=high_cardinality_columns,
        notes=notes,
        columns=[
            OneHotEncodingColumnSuggestion(**suggestion.to_payload())
            for suggestion in suggestions
        ],
    )


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
    
    # We don't have graph_selected_methods here easily without duplicating logic, 
    # passing empty dict for now.
    recommendations = _build_skewness_recommendations(frame, selected_methods, {})

    return SkewnessRecommendationsResponse(
        dataset_source_id=request.dataset_source_id,
        sample_size=request.sample_size,
        skewness_threshold=SKEWNESS_THRESHOLD,
        methods=_skewness_method_details(),
        columns=recommendations,
    )


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

    return BinningRecommendationsResponse(
        dataset_source_id=request.dataset_source_id,
        sample_size=int(frame.shape[0]),
        columns=recommendations,
        excluded_columns=excluded,
    )


@router.post("/ordinal-encoding", response_model=OrdinalEncodingRecommendationsResponse)
async def recommend_ordinal_encoding(
    request: RecommendationRequest = Body(...),
    db: AsyncSession = Depends(get_async_session),
) -> OrdinalEncodingRecommendationsResponse:
    """
    Recommend columns for Ordinal Encoding.
    """
    eda_service = build_eda_service(db, request.sample_size)
    frame, column_metadata, notes = await prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=request.dataset_source_id,
        sample_size=request.sample_size,
        graph=request.graph,
        target_node_id=request.target_node_id,
        skip_catalog_types={
            "target_encoding",
            "ordinal_encoding",
            "label_encoding",
            "one_hot_encoding",
            "dummy_encoding",
            "hash_encoding",
        },
    )

    suggestions = build_ordinal_encoding_suggestions(frame, column_metadata)

    total_text_columns = len(suggestions)
    recommended_count = sum(1 for item in suggestions if item.status == "recommended" and item.selectable)
    high_cardinality_columns = [
        item.column for item in suggestions if item.status in {"high_cardinality", "too_many_categories"}
    ]

    return OrdinalEncodingRecommendationsResponse(
        dataset_source_id=request.dataset_source_id,
        sample_size=int(frame.shape[0]),
        total_text_columns=total_text_columns,
        recommended_count=recommended_count,
        auto_detect_default=bool(recommended_count),
        high_cardinality_columns=high_cardinality_columns,
        notes=notes,
        columns=[
            OrdinalEncodingColumnSuggestion(**suggestion.to_payload())
            for suggestion in suggestions
        ],
    )


@router.post("/hash-encoding", response_model=HashEncodingRecommendationsResponse)
async def recommend_hash_encoding(
    request: RecommendationRequest = Body(...),
    db: AsyncSession = Depends(get_async_session),
) -> HashEncodingRecommendationsResponse:
    """
    Recommend columns for Hash Encoding.
    """
    eda_service = build_eda_service(db, request.sample_size)
    frame, column_metadata, notes = await prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=request.dataset_source_id,
        sample_size=request.sample_size,
        graph=request.graph,
        target_node_id=request.target_node_id,
        skip_catalog_types={
            "hash_encoding",
            "label_encoding",
            "target_encoding",
            "ordinal_encoding",
            "dummy_encoding",
            "one_hot_encoding",
        },
    )

    suggestions = build_hash_encoding_suggestions(frame, column_metadata)

    total_text_columns = len(suggestions)
    recommended_count = sum(1 for item in suggestions if item.status == "recommended" and item.selectable)
    high_cardinality_columns = [
        item.column
        for item in suggestions
        if (item.unique_count or 0) > HASH_ENCODING_DEFAULT_MAX_CATEGORIES
    ]

    return HashEncodingRecommendationsResponse(
        dataset_source_id=request.dataset_source_id,
        sample_size=int(frame.shape[0]),
        total_text_columns=total_text_columns,
        recommended_count=recommended_count,
        auto_detect_default=bool(recommended_count),
        high_cardinality_columns=high_cardinality_columns,
        notes=notes,
        columns=[
            HashEncodingColumnSuggestion(**suggestion.to_payload())
            for suggestion in suggestions
        ],
    )


@router.post("/target-encoding", response_model=TargetEncodingRecommendationsResponse)
async def recommend_target_encoding(
    request: RecommendationRequest = Body(...),
    db: AsyncSession = Depends(get_async_session),
) -> TargetEncodingRecommendationsResponse:
    """
    Recommend columns for Target Encoding.
    """
    eda_service = build_eda_service(db, request.sample_size)
    frame, column_metadata, notes = await prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=request.dataset_source_id,
        sample_size=request.sample_size,
        graph=request.graph,
        target_node_id=request.target_node_id,
        skip_catalog_types={
            "target_encoding",
            "ordinal_encoding",
            "label_encoding",
            "one_hot_encoding",
            "dummy_encoding",
            "hash_encoding",
        },
    )

    suggestions = build_target_encoding_suggestions(frame, column_metadata)

    total_text_columns = len(suggestions)
    recommended_count = sum(1 for item in suggestions if item.status == "recommended" and item.selectable)
    high_cardinality_columns = [
        item.column
        for item in suggestions
        if (item.unique_count or 0) > TARGET_ENCODING_DEFAULT_MAX_CATEGORIES
    ]

    return TargetEncodingRecommendationsResponse(
        dataset_source_id=request.dataset_source_id,
        sample_size=int(frame.shape[0]),
        total_text_columns=total_text_columns,
        recommended_count=recommended_count,
        auto_detect_default=bool(recommended_count),
        high_cardinality_columns=high_cardinality_columns,
        notes=notes,
        columns=[
            TargetEncodingColumnSuggestion(**suggestion.to_payload())
            for suggestion in suggestions
        ],
    )


@router.post("/dummy-encoding", response_model=DummyEncodingRecommendationsResponse)
async def recommend_dummy_encoding(
    request: RecommendationRequest = Body(...),
    db: AsyncSession = Depends(get_async_session),
) -> DummyEncodingRecommendationsResponse:
    """
    Recommend columns for Dummy Encoding.
    """
    eda_service = build_eda_service(db, request.sample_size)
    frame, column_metadata, notes = await prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=request.dataset_source_id,
        sample_size=request.sample_size,
        graph=request.graph,
        target_node_id=request.target_node_id,
        skip_catalog_types={
            "target_encoding",
            "dummy_encoding",
            "one_hot_encoding",
            "label_encoding",
            "ordinal_encoding",
            "hash_encoding",
        },
    )

    suggestions = build_dummy_encoding_suggestions(frame, column_metadata)

    total_text_columns = len(suggestions)
    recommended_count = sum(1 for item in suggestions if item.status == "recommended" and item.selectable)
    cautioned_count = sum(
        1 for item in suggestions if item.status in {"high_cardinality", "too_many_categories"}
    )
    high_cardinality_columns = [
        item.column for item in suggestions if item.status in {"high_cardinality", "too_many_categories"}
    ]

    return DummyEncodingRecommendationsResponse(
        dataset_source_id=request.dataset_source_id,
        sample_size=int(frame.shape[0]),
        total_text_columns=total_text_columns,
        recommended_count=recommended_count,
        cautioned_count=cautioned_count,
        high_cardinality_columns=high_cardinality_columns,
        notes=notes,
        columns=[
            DummyEncodingColumnSuggestion(**suggestion.to_payload())
            for suggestion in suggestions
        ],
    )

