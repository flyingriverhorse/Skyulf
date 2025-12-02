"""Discretization recommendation logic."""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from core.feature_engineering.execution.data import build_eda_service
from core.feature_engineering.execution.graph import resolve_catalog_type
from core.feature_engineering.execution.recommendations.utils import (
    prepare_categorical_recommendation_context,
)
from core.feature_engineering.schemas import (
    BinnedColumnDistribution,
    BinnedDistributionResponse,
)

logger = logging.getLogger(__name__)


def collect_binned_columns_from_graph(
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    target_node_id: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    columns: Dict[str, Dict[str, Any]] = {}

    for node_id in execution_order:
        if node_id == target_node_id:
            break

        node = node_map.get(node_id)
        if not node:
            continue

        catalog_type = resolve_catalog_type(node)
        if catalog_type == "binning_discretization":
            config = node.get("data", {}).get("config", {})
            cols = config.get("columns", [])
            method = config.get("method")
            bins = config.get("bins")
            
            for col in cols:
                columns[col] = {
                    "method": method,
                    "bins": bins,
                    "node_id": node_id
                }

    return columns


def build_candidate_binned_columns(
    frame: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    candidate_columns: Dict[str, Dict[str, Any]] = {}

    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
            
        # Skip low cardinality (likely categorical/ordinal)
        unique_count = frame[column].nunique()
        if unique_count < 10:
            continue
            
        col_meta = metadata.get(column, {})
        candidate_columns[column] = {
            "column": column,
            "type": str(frame[column].dtype),
            "unique_count": unique_count,
            "missing_count": frame[column].isna().sum(),
            "min": float(frame[column].min()) if not frame[column].empty else 0,
            "max": float(frame[column].max()) if not frame[column].empty else 0,
            **col_meta
        }

    return candidate_columns


def build_binned_distributions_list(
    frame: pd.DataFrame,
    candidate_columns: Dict[str, Dict[str, Any]],
) -> List[BinnedColumnDistribution]:
    distributions: List[BinnedColumnDistribution] = []

    for column_name, metadata in candidate_columns.items():
        try:
            # Simple equal-width binning for visualization
            # In a real app, we might want to use the configured method if available
            series = frame[column_name].dropna()
            if series.empty:
                continue
                
            # Create histogram
            hist_values, bin_edges = pd.cut(series, bins=20, retbins=True, duplicates='drop')
            counts = hist_values.value_counts().sort_index()
            
            dist_data = []
            for interval, count in counts.items():
                dist_data.append({
                    "bin_start": float(interval.left),
                    "bin_end": float(interval.right),
                    "count": int(count)
                })
                
            distributions.append(
                BinnedColumnDistribution(
                    column=column_name,
                    distribution=dist_data,
                    top_count=int(counts.max()) if not counts.empty else 0
                )
            )
        except Exception as e:
            logger.warning(f"Failed to build distribution for {column_name}: {e}")

    distributions.sort(
        key=lambda item: (
            -(item.top_count or 0),
            item.column.lower(),
        )
    )

    return distributions


async def generate_binned_distribution_response(
    session: AsyncSession,
    *,
    dataset_source_id: str,
    sample_size: int,
    graph_input: Any,
    target_node_id: Optional[str],
) -> BinnedDistributionResponse:
    eda_service = build_eda_service(session, sample_size)
    
    # Load data
    frame, column_metadata, notes = await prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=dataset_source_id,
        sample_size=sample_size,
        graph=graph_input,
        target_node_id=target_node_id,
    )
    
    if frame.empty:
        return BinnedDistributionResponse(
            dataset_source_id=dataset_source_id,
            distributions=[],
            notes=notes,
        )
        
    # Identify candidates
    candidates = build_candidate_binned_columns(frame, column_metadata)
    
    # Build distributions
    distributions = build_binned_distributions_list(frame, candidates)
    
    return BinnedDistributionResponse(
        dataset_source_id=dataset_source_id,
        distributions=distributions,
        notes=notes,
    )
