"""Pipeline execution engine."""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from core.feature_engineering.execution.data import load_dataset_frame
from core.feature_engineering.execution.graph import (
    DATASET_NODE_ID,
    determine_node_split_type,
    ensure_dataset_node,
    execution_order,
    resolve_catalog_type,
    resolve_node_label,
)
from core.feature_engineering.execution.registry import (
    NODE_EXECUTION_SPECS,
    NODE_TRANSFORMS,
    NodeExecutionContext,
    PipelineNodeOutcome,
)
from core.feature_engineering.preprocessing.split import SPLIT_TYPE_COLUMN, apply_train_test_split
from core.feature_engineering.schemas import (
    FullExecutionSignal,
    PipelinePreviewSignals,
    TrainModelDraftReadinessSnapshot,
)
from core.feature_engineering.shared.utils import _is_node_pending
from core.feature_engineering.split_handler import detect_splits, log_split_processing, remove_split_column

logger = logging.getLogger(__name__)


def invoke_node_transform(
    catalog_type: str,
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str],
) -> Tuple[pd.DataFrame, bool]:
    handler = NODE_TRANSFORMS.get(catalog_type)
    if not handler:
        return frame, False

    transform_fn, requires_pipeline = handler
    if requires_pipeline:
        frame, _, _ = transform_fn(frame=frame, node=node, pipeline_id=pipeline_id)
    else:
        frame, _, _ = transform_fn(frame=frame, node=node)

    return frame, True


def should_skip_preprocessing_node(
    node: Dict[str, Any],
    catalog_type: Optional[str],
    skip_types: Set[str],
) -> bool:
    if _is_node_pending(node):
        return True
    if catalog_type in skip_types:
        return True
    if catalog_type in {"dataset", "dataset-source", "data_preview"}:
        return True
    return False


def apply_train_test_split_with_filter(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    target_node_id: Optional[str],
    edges: List[Dict[str, Any]],
    node_map: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    updated_frame, _, _ = apply_train_test_split(frame, node)

    if not target_node_id or SPLIT_TYPE_COLUMN not in updated_frame.columns:
        return updated_frame

    split_type_filter = determine_node_split_type(target_node_id, edges, node_map)
    if not split_type_filter:
        return updated_frame

    original_rows = len(updated_frame)
    filtered_frame = updated_frame[updated_frame[SPLIT_TYPE_COLUMN] == split_type_filter].copy()
    filtered_rows = len(filtered_frame)

    if filtered_rows < original_rows:
        logger.debug(
            f"Filtered dataset from {original_rows} to {filtered_rows} rows "
            f"using split type '{split_type_filter}'"
        )

    return filtered_frame


def apply_graph_transformations_before_node(
    frame: pd.DataFrame,
    node_map: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, Any]],
    target_node_id: Optional[str],
    skip_catalog_types: Optional[Set[str]] = None,
    pipeline_id: Optional[str] = None,
) -> pd.DataFrame:
    if frame.empty or not node_map:
        return frame

    exec_order = execution_order(node_map, edges, target_node_id)
    if not exec_order:
        return frame

    working_frame = frame.copy()
    skip_types = set(skip_catalog_types or [])

    for node_id in exec_order:
        if node_id == target_node_id:
            break

        node = node_map.get(node_id)
        if not node:
            continue

        catalog_type = resolve_catalog_type(node)
        
        if should_skip_preprocessing_node(node, catalog_type, skip_types):
            continue

        # Special handling for train/test split filtering
        if catalog_type == "train_test_split":
            working_frame = apply_train_test_split_with_filter(
                working_frame, node, target_node_id, edges, node_map
            )
            continue

        working_frame, _ = invoke_node_transform(catalog_type, working_frame, node, pipeline_id)

    # Remove internal split column before returning
    working_frame = remove_split_column(working_frame)

    return working_frame


def apply_graph_with_execution_order(
    frame: pd.DataFrame,
    node_map: Dict[str, Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    target_node: Optional[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Dict[str, Any]]]:
    if not (node_map or graph_edges or target_node):
        return frame, [], {}

    ensured_map = ensure_dataset_node(node_map)
    exec_order = execution_order(ensured_map, graph_edges, target_node)

    transformed_frame = frame
    try:
        transformed_frame, applied_steps, _, _ = run_pipeline_execution(
            frame,
            exec_order,
            ensured_map,
            collect_signals=False,
        )
    except Exception:
        logger.exception("Failed to apply graph transformations")
        return frame, [], {}

    return transformed_frame, applied_steps, ensured_map


def apply_recommendation_graph(
    frame: pd.DataFrame,
    node_map: Dict[str, Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    target_node: Optional[str],
    skip_catalog_types: Optional[Set[str]],
) -> pd.DataFrame:
    if not (node_map or graph_edges or target_node):
        return frame

    ensured_map = ensure_dataset_node(node_map)
    logger.debug(
        "Applying graph transformations",
        extra={
            "node_count": len(ensured_map),
            "edge_count": len(graph_edges),
            "target_node": target_node,
            "skip_types": list(skip_catalog_types or []),
        },
    )

    try:
        return apply_graph_transformations_before_node(
            frame,
            ensured_map,
            graph_edges,
            target_node,
            skip_catalog_types,
        )
    except Exception as exc:
        logger.warning(f"Failed to apply recommendation graph: {exc}")
        return frame


def _log_node_split_state(
    frame: pd.DataFrame,
    *,
    node_id: str,
    catalog_type: str,
) -> None:
    split_info = detect_splits(frame)
    log_split_processing(
        node_id=node_id,
        catalog_type=catalog_type,
        split_info=split_info,
        action="processing",
    )


def _special_catalog_summary(catalog_type: str, label: str) -> Optional[str]:
    if catalog_type in {"dataset", "dataset-source"}:
        return f"{label}: loaded dataset"
    if catalog_type == "data_preview":
        return f"{label}: data preview point"
    return None


def execute_pipeline_node(
    node_id: str,
    frame: pd.DataFrame,
    node_map: Dict[str, Dict[str, Any]],
    context: NodeExecutionContext,
    *,
    collect_signals: bool,
    signals: Optional[PipelinePreviewSignals],
) -> PipelineNodeOutcome:
    node = node_map.get(node_id)
    if not node:
        return PipelineNodeOutcome(frame, f"Node {node_id} not found")

    catalog_type = resolve_catalog_type(node)
    label = resolve_node_label(node)

    _log_node_split_state(frame, node_id=node_id, catalog_type=catalog_type)

    if _is_node_pending(node):
        return PipelineNodeOutcome(frame, f"{label}: pending configuration")

    special_summary = _special_catalog_summary(catalog_type, label)
    if special_summary:
        return PipelineNodeOutcome(frame, special_summary)

    spec = NODE_EXECUTION_SPECS.get(catalog_type)
    if not spec:
        return PipelineNodeOutcome(frame, f"{label}: no execution handler found")

    result = spec.handler(frame, node, context)

    if collect_signals and signals is not None and spec.signal_attr:
        current_signals = getattr(signals, spec.signal_attr)
        if current_signals is not None:
            if spec.signal_mode == "assign":
                setattr(signals, spec.signal_attr, result.signal)
            else:
                current_signals.append(result.signal)

    modeling_metadata = result.signal if spec.update_modeling_metadata else None

    return PipelineNodeOutcome(result.frame, result.summary, modeling_metadata)


def run_pipeline_execution(
    frame: pd.DataFrame,
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    *,
    pipeline_id: Optional[str] = None,
    collect_signals: bool = True,
    existing_signals: Optional[PipelinePreviewSignals] = None,
    preserve_split_column: bool = False,
) -> Tuple[pd.DataFrame, List[str], Optional[PipelinePreviewSignals], Optional[TrainModelDraftReadinessSnapshot]]:
    """Apply configured nodes against a frame, optionally collecting preview signals.

    Args:
        frame: Input DataFrame to transform
        execution_order: Ordered list of node IDs to execute
        node_map: Dictionary mapping node IDs to node configurations
        pipeline_id: Unique identifier for this pipeline instance (for transformer storage)
        collect_signals: Whether to collect detailed node execution metadata
        existing_signals: Optional pre-existing signals to extend

    Returns:
        Tuple of (transformed_frame, applied_steps, signals, modeling_metadata)
    """

    working_frame = frame.copy()
    applied_steps: List[str] = []
    modeling_metadata: Optional[TrainModelDraftReadinessSnapshot] = None
    signals: Optional[PipelinePreviewSignals]
    if collect_signals:
        signals = existing_signals or PipelinePreviewSignals()
    else:
        signals = None

    context = NodeExecutionContext(pipeline_id=pipeline_id, node_map=node_map)

    for node_id in execution_order:
        outcome = execute_pipeline_node(
            node_id,
            working_frame,
            node_map,
            context,
            collect_signals=collect_signals,
            signals=signals,
        )
        working_frame = outcome.frame
        if outcome.summary:
            applied_steps.append(outcome.summary)
        
        if outcome.modeling_metadata:
            modeling_metadata = outcome.modeling_metadata

    # Remove internal split column before returning results unless explicitly preserved
    if not preserve_split_column:
        working_frame = remove_split_column(working_frame)

    return working_frame, applied_steps, signals, modeling_metadata


def collect_pipeline_signals(
    frame: pd.DataFrame,
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    *,
    pipeline_id: Optional[str] = None,
    existing_signals: Optional[PipelinePreviewSignals] = None,
    preserve_split_column: bool = False,
) -> Tuple[pd.DataFrame, PipelinePreviewSignals, Optional[TrainModelDraftReadinessSnapshot]]:
    """Run the pipeline with signal collection enabled and return diagnostics only.

    This helper lets callers (e.g., export orchestrators) gather node-level
    signals without relying on the snapshot response pipeline.
    """

    frame_result, _, signals, modeling_metadata = run_pipeline_execution(
        frame,
        execution_order,
        node_map,
        pipeline_id=pipeline_id,
        collect_signals=True,
        existing_signals=existing_signals,
        preserve_split_column=preserve_split_column,
    )


async def run_full_dataset_execution(
    *,
    session: AsyncSession,
    dataset_source_id: str,
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    pipeline_id: str,
    applied_steps: List[str],
    preview_total_rows: int,
) -> Tuple[FullExecutionSignal, int]:
    
    # Load full dataset
    frame, meta = await load_dataset_frame(
        session,
        dataset_source_id,
        sample_size=0,
        execution_mode="full",
    )
    
    if frame.empty:
        raise ValueError(f"Could not load full dataset for {dataset_source_id}")
        
    total_rows = len(frame)
    
    # Run pipeline
    transformed_frame, steps, signals, _ = run_pipeline_execution(
        frame,
        execution_order,
        node_map,
        pipeline_id=pipeline_id,
        collect_signals=True,
    )
    
    # Build signal
    signal = FullExecutionSignal(
        status="succeeded",
        job_id=pipeline_id,
        reason="Full execution completed",
        total_rows=total_rows,
        processed_rows=len(transformed_frame),
        applied_steps=steps,
        dataset_source_id=dataset_source_id,
    )
    
    return signal, len(transformed_frame)

    return frame_result, signals, modeling_metadata
