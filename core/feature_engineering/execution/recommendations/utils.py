"""Recommendation utilities."""

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from core.feature_engineering.eda_fast import FeatureEngineeringEDAService
from core.feature_engineering.execution.data import build_preview_frame
from core.feature_engineering.execution.engine import apply_recommendation_graph
from core.feature_engineering.execution.graph import (
    extract_graph_payload,
    normalize_target_node,
)

logger = logging.getLogger(__name__)


def build_recommendation_column_metadata(
    quality_payload: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    column_metadata: Dict[str, Dict[str, Any]] = {}
    notes: List[str] = []

    if not quality_payload.get("success"):
        notes.append("Could not retrieve data quality metrics.")
        return column_metadata, notes

    quality_report = quality_payload.get("quality_report") or {}
    quality_metrics = quality_report.get("quality_metrics") or {}
    column_details = quality_metrics.get("column_details") or []
    if isinstance(column_details, Iterable):
        for col in column_details:
            if isinstance(col, dict):
                name = col.get("column")
                if name:
                    column_metadata[name] = col

    text_summary = quality_report.get("text_analysis_summary") or {}
    categorical_columns = text_summary.get("categorical_text_columns") or []
    if isinstance(categorical_columns, Iterable):
        for col_name in categorical_columns:
            if col_name in column_metadata:
                column_metadata[col_name]["is_categorical_text"] = True

    return column_metadata, notes


async def prepare_categorical_recommendation_context(
    *,
    eda_service: FeatureEngineeringEDAService,
    dataset_source_id: str,
    sample_size: int,
    graph: Optional[Dict[str, Any]],
    target_node_id: Optional[str],
    skip_catalog_types: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], List[str]]:
    preview_payload = await eda_service.preview_source(dataset_source_id, sample_size=sample_size)

    if not preview_payload.get("success"):
        return pd.DataFrame(), {}, ["Could not load dataset preview."]

    frame, _ = build_preview_frame(preview_payload)

    graph_node_map, graph_edges = extract_graph_payload(graph)
    normalized_target_node = normalize_target_node(target_node_id)
    frame = apply_recommendation_graph(
        frame,
        graph_node_map,
        graph_edges,
        normalized_target_node,
        skip_catalog_types,
    )

    quality_payload = await eda_service.quality_report(dataset_source_id, sample_size=sample_size)
    column_metadata, notes = build_recommendation_column_metadata(quality_payload)

    return frame, column_metadata, notes
