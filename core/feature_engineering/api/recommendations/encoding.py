from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.engine import get_async_session
from core.feature_engineering.execution.data import build_eda_service
from core.feature_engineering.execution.recommendations import (
    prepare_categorical_recommendation_context,
)
from core.feature_engineering.schemas import (
    LabelEncodingRecommendationsResponse,
    OneHotEncodingRecommendationsResponse,
    LabelEncodingColumnSuggestion,
    OneHotEncodingColumnSuggestion,
    OrdinalEncodingRecommendationsResponse,
    OrdinalEncodingColumnSuggestion,
    HashEncodingRecommendationsResponse,
    HashEncodingColumnSuggestion,
    TargetEncodingRecommendationsResponse,
    TargetEncodingColumnSuggestion,
    DummyEncodingRecommendationsResponse,
    DummyEncodingColumnSuggestion,
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

from .schemas import RecommendationRequest
from .utils import _get_target_column_from_graph

router = APIRouter()

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
    
    # Handle target column specifically
    target_column = _get_target_column_from_graph(request.graph)
    if target_column:
        for suggestion in suggestions:
            if suggestion.column == target_column:
                suggestion.status = "recommended"
                suggestion.selectable = True
                suggestion.reason = "Target Column detected. You MUST select 'Replace Original' to use the encoded values for training."
                suggestion.score = 1.0  # Boost score to ensure visibility

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

    # Filter out target column if configured
    target_column = _get_target_column_from_graph(request.graph)
    if target_column:
        suggestions = [s for s in suggestions if s.column != target_column]

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

    # Filter out target column if configured
    target_column = _get_target_column_from_graph(request.graph)
    if target_column:
        suggestions = [s for s in suggestions if s.column != target_column]

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

    # Filter out target column if configured
    target_column = _get_target_column_from_graph(request.graph)
    if target_column:
        suggestions = [s for s in suggestions if s.column != target_column]

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

    # Filter out target column if configured
    target_column = _get_target_column_from_graph(request.graph)
    if target_column:
        suggestions = [s for s in suggestions if s.column != target_column]

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

    # Filter out target column if configured
    target_column = _get_target_column_from_graph(request.graph)
    if target_column:
        suggestions = [s for s in suggestions if s.column != target_column]

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
