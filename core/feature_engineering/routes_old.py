"""This is older file we are not using. but we can use to fix other modularize files if something happens."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from statistics import StatisticsError, median
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Set, Tuple, TypedDict, cast
from pathlib import Path

import pandas as pd

import joblib

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import core.database.engine as db_engine
from core.database.engine import get_async_session
from core.database.models import DataSource, FeatureEngineeringPipeline, HyperparameterTuningJob, TrainingJob
from core.utils.datetime import utcnow
from core.feature_engineering.eda_fast import FeatureEngineeringEDAService
from core.feature_engineering.eda_fast.service import DEFAULT_SAMPLE_CAP
from core.feature_engineering.full_capture import FullDatasetCaptureService

from .preprocessing.bucketing import (
    BINNING_DEFAULT_EQUAL_FREQUENCY_BINS,
    BINNING_DEFAULT_EQUAL_WIDTH_BINS,
    BINNING_DEFAULT_MISSING_LABEL,
    BINNING_DEFAULT_PRECISION,
    BINNING_DEFAULT_SUFFIX,
    _apply_binning_discretization,
    _build_binned_distribution,
    _build_binning_recommendations,
    _normalize_binning_config,
)
from .preprocessing.statistics import (
    DEFAULT_METHOD_PARAMETERS,
    OUTLIER_DEFAULT_METHOD,
    SCALING_DEFAULT_METHOD,
    SKEWNESS_METHODS,
    SKEWNESS_THRESHOLD,
    _apply_outlier_removal,
    _apply_scale_numeric_features,
    _apply_skewness_transformations,
    _build_outlier_recommendations,
    _build_scaling_recommendations,
    _build_skewness_recommendations,
    _outlier_method_details,
    _scaling_method_details,
    _skewness_method_details,
    apply_imputation_methods as _apply_imputation_methods,
)
from .preprocessing.feature_generation import apply_feature_math, apply_polynomial_features
from .preprocessing.feature_selection import apply_feature_selection
from .preprocessing.inspection import (
    apply_transformer_audit,
    build_data_snapshot_response,
    build_quick_profile_payload,
)
from .preprocessing.encoding.label_encoding import (
    LABEL_ENCODING_DEFAULT_MAX_UNIQUE,
    LABEL_ENCODING_DEFAULT_SUFFIX,
    apply_label_encoding,
)
from .preprocessing.encoding.hash_encoding import (
    HASH_ENCODING_DEFAULT_BUCKETS,
    HASH_ENCODING_DEFAULT_MAX_CATEGORIES,
    HASH_ENCODING_DEFAULT_SUFFIX,
    HASH_ENCODING_MAX_CARDINALITY_LIMIT,
    HASH_ENCODING_MAX_BUCKETS,
    apply_hash_encoding,
)
from .preprocessing.resampling import (
    OVERSAMPLING_DEFAULT_K_NEIGHBORS,
    OVERSAMPLING_DEFAULT_METHOD,
    OVERSAMPLING_DEFAULT_RANDOM_STATE,
    OVERSAMPLING_DEFAULT_REPLACEMENT,
    OVERSAMPLING_METHOD_LABELS,
    RESAMPLING_DEFAULT_METHOD as UNDERSAMPLING_DEFAULT_METHOD,
    RESAMPLING_DEFAULT_RANDOM_STATE as UNDERSAMPLING_DEFAULT_RANDOM_STATE,
    RESAMPLING_DEFAULT_REPLACEMENT as UNDERSAMPLING_DEFAULT_REPLACEMENT,
    RESAMPLING_METHOD_LABELS as UNDERSAMPLING_METHOD_LABELS,
    apply_oversampling,
    apply_resampling,
)
from .preprocessing.split import apply_feature_target_split
from .preprocessing.split import apply_train_test_split, SPLIT_TYPE_COLUMN
from .split_handler import detect_splits, log_split_processing, remove_split_column
from .preprocessing.encoding.ordinal_encoding import (
    ORDINAL_ENCODING_DEFAULT_MAX_CATEGORIES,
    ORDINAL_ENCODING_DEFAULT_SUFFIX,
    ORDINAL_ENCODING_MAX_CARDINALITY_LIMIT,
    ORDINAL_ENCODING_DEFAULT_UNKNOWN_VALUE,
    apply_ordinal_encoding,
)
from .modeling.hyperparameter_tuning.registry import (
    get_default_strategy_value,
    get_strategy_choices_for_ui,
)
from .preprocessing.encoding.target_encoding import (
    TARGET_ENCODING_DEFAULT_MAX_CATEGORIES,
    TARGET_ENCODING_DEFAULT_SUFFIX,
    TARGET_ENCODING_DEFAULT_SMOOTHING,
    TARGET_ENCODING_MAX_CARDINALITY_LIMIT,
    TARGET_ENCODING_DEFAULT_HANDLE_UNKNOWN,
    apply_target_encoding,
)
from .preprocessing.encoding.one_hot_encoding import (
    ONE_HOT_ENCODING_DEFAULT_MAX_CATEGORIES,
    ONE_HOT_ENCODING_DEFAULT_PREFIX_SEPARATOR,
    ONE_HOT_ENCODING_MAX_CARDINALITY_LIMIT,
    apply_one_hot_encoding,
)
from .preprocessing.encoding.dummy_encoding import (
    DUMMY_ENCODING_DEFAULT_MAX_CATEGORIES,
    DUMMY_ENCODING_DEFAULT_PREFIX_SEPARATOR,
    DUMMY_ENCODING_MAX_CARDINALITY_LIMIT,
    apply_dummy_encoding,
)
from .preprocessing.casting import _apply_cast_column_types
from .preprocessing.cleaning import (
    apply_normalize_text_case,
    apply_regex_cleanup,
    apply_remove_special_characters,
    apply_replace_aliases_typos,
    apply_replace_invalid_values,
    apply_standardize_date_formats,
    apply_trim_whitespace,
)
from .preprocessing.drop_and_missing import (
    apply_drop_missing_columns,
    apply_drop_missing_rows,
    apply_missing_value_flags,
    apply_remove_duplicates,
)
from core.feature_engineering.shared.utils import _is_node_pending
from .modeling.training.train_model_draft import apply_train_model_draft
from .modeling.training.evaluation import (
    apply_model_evaluation,
    build_classification_split_report,
    build_regression_split_report,
)
from .modeling.training.registry import list_registered_models

from .schemas import (
    BinnedColumnDistribution,
    BinnedDistributionRequest,
    BinnedDistributionResponse,
    DatasetSourceSummary,
    DropColumnCandidate,
    DropColumnRecommendationFilter,
    DropColumnRecommendations,
    OrdinalEncodingColumnSuggestion,
    OrdinalEncodingRecommendationsResponse,
    LabelEncodingColumnSuggestion,
    LabelEncodingRecommendationsResponse,
    OneHotEncodingColumnSuggestion,
    OneHotEncodingRecommendationsResponse,
    DummyEncodingColumnSuggestion,
    DummyEncodingRecommendationsResponse,
    TargetEncodingColumnSuggestion,
    TargetEncodingRecommendationsResponse,
    HashEncodingColumnSuggestion,
    HashEncodingRecommendationsResponse,
    FeaturePipelineCreate,
    FeaturePipelineResponse,
    SkewnessRecommendationsResponse,
    QuickProfileDatasetMetrics,
    QuickProfileResponse,
    PipelinePreviewRequest,
    PipelinePreviewMetrics,
    PipelinePreviewResponse,
    PipelinePreviewRowsResponse,
    PipelinePreviewSignals,
    FullExecutionSignal,
    ScalingRecommendationsResponse,
    BinningRecommendationsResponse,
    OutlierRecommendationsResponse,
    TrainModelDraftReadinessSnapshot,
    ModelEvaluationReport,
    ModelEvaluationRequest,
    TrainingJobCreate,
    TrainingJobBatchResponse,
    TrainingJobListResponse,
    TrainingJobResponse,
    TrainingJobStatus,
    TrainingJobSummary,
    HyperparameterTuningJobCreate,
    HyperparameterTuningJobBatchResponse,
    HyperparameterTuningJobListResponse,
    HyperparameterTuningJobResponse,
    HyperparameterTuningJobStatus,
    HyperparameterTuningJobSummary,
)

from core.feature_engineering.recommendations import (
    build_label_encoding_suggestions,
    build_one_hot_encoding_suggestions,
    build_dummy_encoding_suggestions,
    build_ordinal_encoding_suggestions,
    build_target_encoding_suggestions,
    build_hash_encoding_suggestions,
)

from .modeling.training.jobs import (
    create_training_job as create_training_job_record,
    get_training_job as fetch_training_job,
    list_training_jobs as fetch_training_jobs,
    update_job_status,
)
from .modeling.training.tasks import (
    dispatch_training_job,
    _prepare_training_data,
    _resolve_training_inputs,
)
from .modeling.hyperparameter_tuning.jobs import (
    create_tuning_job as create_hyperparameter_tuning_job_record,
    get_tuning_job as fetch_hyperparameter_tuning_job,
    list_tuning_jobs as fetch_hyperparameter_tuning_jobs,
    update_tuning_job_status,
)
from .modeling.hyperparameter_tuning.tasks import dispatch_hyperparameter_tuning_job


DROP_COLUMN_FILTER_LABELS: Dict[str, Dict[str, Optional[str]]] = {
    "missing_data": {
        "label": "High missingness",
        "description": "Columns exceeding the configured missingness threshold.",
    },
    "empty_column": {
        "label": "Empty columns",
        "description": "Columns with no observed values in the sampled data.",
    },
    "multicollinearity": {
        "label": "Multi-collinearity",
        "description": "Columns flagged as part of highly correlated feature pairs.",
    },
    "low_variance": {
        "label": "Low variance",
        "description": "Columns with near-constant values.",
    },
    "other": {
        "label": "Other reasons",
        "description": "Columns surfaced by broader EDA heuristics.",
    },
}


# Request body models for POST recommendation endpoints
class RecommendationRequest(BaseModel):
    """Base request model for recommendation endpoints."""
    dataset_source_id: str = Field(..., description="Identifier of the dataset source")
    sample_size: int = Field(
        default=500,
        ge=50,
        le=5000,
        description="Number of rows to sample for analysis"
    )
    graph: Optional[Dict[str, Any]] = Field(
        None,
        description="Pipeline graph structure with nodes and edges"
    )
    target_node_id: Optional[str] = Field(
        None,
        description="ID of the target node in the pipeline"
    )


class SkewnessRecommendationRequest(RecommendationRequest):
    """Request model for skewness recommendations with transformations."""
    transformations: Optional[str] = Field(
        None,
        description="JSON string of applied transformations"
    )


router = APIRouter(prefix="/ml-workflow", tags=["ml-workflow"])

DATASET_NODE_ID = "dataset-source"
FULL_DATASET_EXECUTION_ROW_LIMIT = 200_000
logger = logging.getLogger(__name__)


def _generate_pipeline_id(dataset_source_id: str, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> str:
    """Generate a stable pipeline ID from dataset and graph structure.

    Creates a unique identifier that combines the dataset source with a hash
    of the pipeline graph. Same graph = same ID, different graph = different ID.
    This ensures transformers are isolated across different pipeline versions.

    Args:
        dataset_source_id: The source dataset identifier
        nodes: List of node dictionaries from the graph
        edges: List of edge dictionaries from the graph

    Returns:
        Pipeline ID in format: {dataset_source_id}_{hash}
    """
    import hashlib

    # Create a stable representation of the graph for hashing
    # We want to capture: node IDs, types, configs, and connections
    graph_representation = {
        "nodes": [
            {
                "id": node.get("id"),
                "type": node.get("type"),
                "catalogType": node.get("data", {}).get("catalogType"),
                "config": node.get("data", {}).get("config"),
            }
            for node in sorted(nodes, key=lambda n: n.get("id", ""))
        ],
        "edges": [
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "sourceHandle": edge.get("sourceHandle"),
                "targetHandle": edge.get("targetHandle"),
            }
            for edge in sorted(edges, key=lambda e: (e.get("source", ""), e.get("target", "")))
        ],
    }

    # Serialize to JSON with sorted keys for consistency
    graph_json = json.dumps(graph_representation, sort_keys=True, separators=(',', ':'))

    # Hash the graph structure
    graph_hash = hashlib.sha256(graph_json.encode('utf-8')).hexdigest()[:8]

    # Combine dataset ID with graph hash
    pipeline_id = f"{dataset_source_id}_{graph_hash}"

    return pipeline_id


FullExecutionJobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


@dataclass
class FullExecutionJob:
    id: str
    dataset_source_id: str
    execution_order: List[str]
    node_map: Dict[str, Dict[str, Any]]
    target_node_id: Optional[str]
    total_rows_estimate: Optional[int]
    graph_nodes: List[Dict[str, Any]] = field(default_factory=list)
    graph_edges: List[Dict[str, Any]] = field(default_factory=list)
    signal_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    status: FullExecutionJobStatus = "queued"
    task: Optional[asyncio.Task] = None

    def to_signal(self) -> FullExecutionSignal:
        payload = dict(self.signal_data)
        payload.setdefault("status", "deferred")
        warnings = payload.get("warnings") or []
        if not isinstance(warnings, list):
            warnings = [warnings]
        payload["warnings"] = [
            str(item).strip() for item in warnings if isinstance(item, str) and item.strip()
        ]
        applied_steps = payload.get("applied_steps") or []
        if not isinstance(applied_steps, list):
            applied_steps = [applied_steps]
        payload["applied_steps"] = [
            str(item).strip() for item in applied_steps if isinstance(item, str) and item.strip()
        ]
        payload["job_id"] = self.id
        payload["job_status"] = self.status
        payload["dataset_source_id"] = self.dataset_source_id
        payload["last_updated"] = self.updated_at
        if payload.get("total_rows") is None:
            payload["total_rows"] = self.total_rows_estimate
        return FullExecutionSignal(**payload)


class FullExecutionJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, FullExecutionJob] = {}
        self._lock = asyncio.Lock()

    async def ensure_job(
        self,
        *,
        dataset_source_id: str,
        execution_order: List[str],
        node_map: Dict[str, Dict[str, Any]],
        target_node_id: Optional[str],
        total_rows_estimate: Optional[int],
        graph_nodes: Optional[List[Dict[str, Any]]] = None,
        graph_edges: Optional[List[Dict[str, Any]]] = None,
        defer_reason: Optional[str] = None,
    ) -> Tuple[FullExecutionJob, FullExecutionSignal, bool]:
        async with self._lock:
            active = next(
                (
                    job
                    for job in self._jobs.values()
                    if job.dataset_source_id == dataset_source_id
                    and job.status in {"queued", "running"}
                ),
                None,
            )
            if active:
                if total_rows_estimate and not active.signal_data.get("total_rows"):
                    active.signal_data["total_rows"] = total_rows_estimate
                if total_rows_estimate and not active.total_rows_estimate:
                    active.total_rows_estimate = total_rows_estimate
                active.signal_data.setdefault("warnings", ["dataset_too_large"])
                active.signal_data.setdefault("applied_steps", [])
                reason_parts: List[str] = []
                if defer_reason:
                    reason_parts.append(defer_reason.strip())
                reason_parts.append(
                    f"Background job {active.id} already queued for full execution."
                )
                active.signal_data["reason"] = " ".join(part for part in reason_parts if part)
                active.signal_data["poll_after_seconds"] = 5
                active.updated_at = utcnow()
                return active, active.to_signal(), False

            job_id = uuid.uuid4().hex
            job = FullExecutionJob(
                id=job_id,
                dataset_source_id=dataset_source_id,
                execution_order=list(execution_order),
                node_map=copy.deepcopy(node_map),
                target_node_id=target_node_id,
                total_rows_estimate=total_rows_estimate,
                graph_nodes=copy.deepcopy(graph_nodes) if graph_nodes else [],
                graph_edges=copy.deepcopy(graph_edges) if graph_edges else [],
            )
            queued_reason_parts: List[str] = []
            if defer_reason:
                queued_reason_parts.append(defer_reason.strip())
            queued_reason_parts.append(f"Background job {job_id} queued for full execution.")

            job.signal_data = {
                "status": "deferred",
                "reason": " ".join(part for part in queued_reason_parts if part),
                "total_rows": total_rows_estimate,
                "warnings": ["dataset_too_large"],
                "applied_steps": [],
                "poll_after_seconds": 5,
            }
            self._jobs[job_id] = job
            return job, job.to_signal(), True

    async def _update(
        self,
        job_id: str,
        *,
        status: Optional[FullExecutionJobStatus] = None,
        signal_updates: Optional[Dict[str, Any]] = None,
    ) -> Optional[FullExecutionJob]:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if status:
                job.status = status
            if signal_updates:
                job.signal_data.update(signal_updates)
            job.updated_at = utcnow()
            return job

    async def mark_running(
        self,
        job_id: str,
        *,
        reason: Optional[str] = None,
        poll_after_seconds: Optional[int] = None,
    ) -> Optional[FullExecutionSignal]:
        updates: Dict[str, Any] = {}
        if reason is not None:
            updates["reason"] = reason
        if poll_after_seconds is not None:
            updates["poll_after_seconds"] = poll_after_seconds
        job = await self._update(job_id, status="running", signal_updates=updates)
        return job.to_signal() if job else None

    async def mark_completed(
        self,
        job_id: str,
        *,
        status: FullExecutionJobStatus,
        signal_updates: Dict[str, Any],
    ) -> Optional[FullExecutionSignal]:
        job = await self._update(job_id, status=status, signal_updates=signal_updates)
        if not job:
            return None
        if signal_updates.get("total_rows") is not None:
            job.total_rows_estimate = signal_updates["total_rows"]
        job.execution_order.clear()
        job.node_map.clear()
        job.task = None
        return job.to_signal()

    async def get_signal(self, dataset_source_id: str, job_id: str) -> Optional[FullExecutionSignal]:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.dataset_source_id != dataset_source_id:
                return None
            return job.to_signal()


full_execution_job_store = FullExecutionJobStore()


def _append_unique_step(applied_steps: List[str], message: str) -> None:
    normalized = (message or "").strip()
    if not normalized:
        return
    if normalized not in applied_steps:
        applied_steps.append(normalized)


def _log_background_exception(task: asyncio.Task) -> None:  # pragma: no cover - background logging
    try:
        task.result()
    except Exception:
        logger.exception("Full dataset background execution task failed")


async def _run_full_execution_job(job: FullExecutionJob) -> None:
    await full_execution_job_store.mark_running(
        job.id,
        reason="Full dataset execution running in background.",
        poll_after_seconds=4,
    )
    try:
        session_factory = db_engine.async_session_factory
        if session_factory is None:
            raise RuntimeError("Async session factory is not initialized")

        async with session_factory() as job_session:
            full_frame, full_meta = await _load_dataset_frame(
                job_session,
                job.dataset_source_id,
                sample_size=0,
                execution_mode="full",
            )

        # Generate pipeline ID for transformer storage
        pipeline_id = _generate_pipeline_id(job.dataset_source_id, job.graph_nodes, job.graph_edges)

        _, applied_steps, _, _ = _run_pipeline_execution(
            full_frame,
            job.execution_order,
            job.node_map,
            pipeline_id=pipeline_id,
            collect_signals=False,
        )

        processed_rows = int(full_frame.shape[0])
        total_rows_full = _coerce_int(full_meta.get("total_rows"), processed_rows)

        await full_execution_job_store.mark_completed(
            job.id,
            status="succeeded",
            signal_updates={
                "status": "succeeded",
                "reason": "Full dataset execution completed in background.",
                "processed_rows": processed_rows,
                "total_rows": total_rows_full,
                "applied_steps": applied_steps,
                "warnings": [],
                "poll_after_seconds": None,
            },
        )
    except MemoryError:
        await full_execution_job_store.mark_completed(
            job.id,
            status="failed",
            signal_updates={
                "status": "failed",
                "reason": "Full dataset execution failed due to insufficient memory.",
                "warnings": ["memory_error"],
                "poll_after_seconds": None,
            },
        )
    except Exception as exc:
        logger.exception(
            "Full dataset background execution failed for dataset %s (job %s)",
            job.dataset_source_id,
            job.id,
        )
        await full_execution_job_store.mark_completed(
            job.id,
            status="failed",
            signal_updates={
                "status": "failed",
                "reason": f"Full dataset execution failed: {exc}",
                "warnings": ["background_failure"],
                "poll_after_seconds": None,
            },
        )


def _schedule_full_execution_job(job: FullExecutionJob) -> None:
    if job.task and not job.task.done():
        return
    task = asyncio.create_task(_run_full_execution_job(job))
    job.task = task
    task.add_done_callback(_log_background_exception)


def _resolve_catalog_type(node: Dict[str, Any]) -> str:
    data = node.get("data") or {}
    if isinstance(data, dict):
        candidate = data.get("catalogType") or data.get("type")
        if candidate:
            return str(candidate)
        if data.get("isDataset"):
            return "dataset"
    node_type = node.get("type")
    return str(node_type) if node_type else ""


def _resolve_node_label(node: Dict[str, Any]) -> str:
    data = node.get("data") or {}
    if isinstance(data, dict):
        label = data.get("label")
        if isinstance(label, str) and label.strip():
            return label.strip()
    label = node.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return str(node.get("id") or "node")


def _build_predecessor_map(edges: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    predecessors: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source and target:
            predecessors[str(target)].append(str(source))
    return predecessors


def _build_successor_map(edges: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    successors: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source and target:
            successors[str(source)].append(str(target))
    return successors


def _execution_order(
    nodes: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, Any]],
    target_node_id: Optional[str] = None,
) -> List[str]:
    node_ids = set(nodes.keys())
    predecessors = _build_predecessor_map(edges)
    successors = _build_successor_map(edges)

    visited: Dict[str, bool] = {}
    order: List[str] = []

    def dfs(node_id: str) -> None:
        if node_id in visited or node_id not in node_ids:
            return
        visited[node_id] = True
        for parent_id in predecessors.get(node_id, []):
            dfs(parent_id)
        order.append(node_id)

    if target_node_id:
        dfs(target_node_id)
    else:
        for node_id in node_ids:
            dfs(node_id)

    reachable: Set[str] = set()
    stack: List[str] = [DATASET_NODE_ID]
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        for child in successors.get(current, []):
            stack.append(child)

    ordered = [node_id for node_id in order if node_id in reachable]
    if target_node_id and target_node_id in ordered:
        cutoff = ordered.index(target_node_id) + 1
        ordered = ordered[:cutoff]
    return ordered


def _sanitize_graph_nodes(raw_nodes: Any) -> Dict[str, Dict[str, Any]]:
    node_map: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw_nodes, list):
        return node_map

    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if not node_id:
            continue
        data = node.get("data")
        safe_data: Dict[str, Any] = {}
        if isinstance(data, dict):
            safe_data = {
                key: value
                for key, value in data.items()
                if not callable(value)
            }

        node_map[str(node_id)] = {**node, "id": str(node_id), "data": safe_data}

    return node_map


def _sanitize_graph_edges(raw_edges: Any) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    if not isinstance(raw_edges, list):
        return edges

    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue

        # Preserve sourceHandle and targetHandle for split type tracking
        edge_data = {
            "source": str(source),
            "target": str(target),
        }

        if "sourceHandle" in edge:
            edge_data["sourceHandle"] = edge["sourceHandle"]
        if "targetHandle" in edge:
            edge_data["targetHandle"] = edge["targetHandle"]

        edges.append(edge_data)

    return edges


def _determine_node_split_type(
    node_id: str,
    edges: List[Dict[str, Any]],
    node_map: Dict[str, Dict[str, Any]],
    visited: Optional[Set[str]] = None,
) -> Optional[str]:
    """
    Determines which split type (train/test/validation) a node should process
    based on which output handle of a train_test_split node it's connected to.

    Returns: 'train', 'test', 'validation', or None
    """
    # Prevent infinite recursion by tracking visited nodes
    if visited is None:
        visited = set()

    if node_id in visited:
        logger.debug(f"Cycle detected at node '{node_id}', stopping recursion")
        return None

    visited.add(node_id)

    # Find incoming edges to this node
    incoming_edges = [e for e in edges if e.get("target") == node_id]

    for edge in incoming_edges:
        source_node_id = edge.get("source")
        source_handle = edge.get("sourceHandle", "")

        if not source_node_id:
            continue

        source_node = node_map.get(source_node_id)
        if not source_node:
            continue

        source_catalog_type = source_node.get("data", {}).get("catalogType", "")

        # Check if source is a train_test_split node
        if source_catalog_type == "train_test_split":
            # Extract split type from handle name
            # Handles are like: "node-123-train", "node-123-test", "node-123-validation"
            if source_handle.endswith("-train"):
                return "train"
            elif source_handle.endswith("-test"):
                return "test"
            elif source_handle.endswith("-validation"):
                return "validation"
        else:
            # Recursively check the source node's split type
            # This propagates split type downstream
            parent_split_type = _determine_node_split_type(source_node_id, edges, node_map, visited)
            if parent_split_type:
                return parent_split_type

    return None


def _ensure_dataset_node(node_map: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if DATASET_NODE_ID not in node_map:
        node_map[DATASET_NODE_ID] = {
            "id": DATASET_NODE_ID,
            "data": {
                "catalogType": "dataset",
                "label": "Dataset input",
                "isDataset": True,
            },
        }
    return node_map


NodeTransformSpec = Tuple[Callable[..., Tuple[pd.DataFrame, Any, Any]], bool]

_NODE_TRANSFORMS: Dict[str, NodeTransformSpec] = {
    "drop_missing_columns": (apply_drop_missing_columns, False),
    "drop_missing_rows": (apply_drop_missing_rows, False),
    "remove_duplicates": (apply_remove_duplicates, False),
    "missing_value_indicator": (apply_missing_value_flags, False),
    "cast_column_types": (_apply_cast_column_types, False),
    "trim_whitespace": (apply_trim_whitespace, False),
    "normalize_text_case": (apply_normalize_text_case, False),
    "replace_aliases_typos": (apply_replace_aliases_typos, False),
    "standardize_date_formats": (apply_standardize_date_formats, False),
    "remove_special_characters": (apply_remove_special_characters, False),
    "replace_invalid_values": (apply_replace_invalid_values, False),
    "regex_replace_fix": (apply_regex_cleanup, False),
    "feature_math": (apply_feature_math, True),
    "binning_discretization": (_apply_binning_discretization, True),
    "skewness_transform": (_apply_skewness_transformations, True),
    "outlier_removal": (_apply_outlier_removal, True),
    "scale_numeric_features": (_apply_scale_numeric_features, True),
    "feature_target_split": (apply_feature_target_split, False),
    "class_undersampling": (apply_resampling, False),
    "class_oversampling": (apply_oversampling, False),
    "label_encoding": (apply_label_encoding, True),
    "target_encoding": (apply_target_encoding, True),
    "hash_encoding": (apply_hash_encoding, True),
    "train_model_draft": (apply_train_model_draft, False),
    "ordinal_encoding": (apply_ordinal_encoding, True),
    "one_hot_encoding": (apply_one_hot_encoding, True),
    "dummy_encoding": (apply_dummy_encoding, True),
}

for key in ("imputation_methods", "advanced_imputer", "simple_imputer"):
    _NODE_TRANSFORMS[key] = (_apply_imputation_methods, True)


def _invoke_node_transform(
    catalog_type: str,
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str],
) -> Tuple[pd.DataFrame, bool]:
    handler = _NODE_TRANSFORMS.get(catalog_type)
    if not handler:
        return frame, False

    transform_fn, requires_pipeline = handler
    if requires_pipeline:
        updated_frame, _, _ = transform_fn(frame, node, pipeline_id=pipeline_id)
    else:
        updated_frame, _, _ = transform_fn(frame, node)

    return updated_frame, True


def _should_skip_preprocessing_node(
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


def _apply_train_test_split_with_filter(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    target_node_id: Optional[str],
    edges: List[Dict[str, Any]],
    node_map: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    updated_frame, _, _ = apply_train_test_split(frame, node)

    if not target_node_id or SPLIT_TYPE_COLUMN not in updated_frame.columns:
        return updated_frame

    split_type_filter = _determine_node_split_type(target_node_id, edges, node_map)
    if not split_type_filter:
        return updated_frame

    original_rows = len(updated_frame)
    filtered_frame = updated_frame[updated_frame[SPLIT_TYPE_COLUMN] == split_type_filter].copy()
    filtered_rows = len(filtered_frame)

    if filtered_rows < original_rows:
        logger.info(
            f"Filtered to '{split_type_filter}' split for node '{target_node_id}': "
            f"{filtered_rows:,} of {original_rows:,} rows",
            extra={
                "node_id": target_node_id,
                "split_type": split_type_filter,
                "original_rows": original_rows,
                "filtered_rows": filtered_rows,
            },
        )

    return filtered_frame


def _apply_graph_transformations_before_node(
    frame: pd.DataFrame,
    node_map: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, Any]],
    target_node_id: Optional[str],
    skip_catalog_types: Optional[Set[str]] = None,
    pipeline_id: Optional[str] = None,
) -> pd.DataFrame:
    if frame.empty or not node_map:
        return frame

    execution_order = _execution_order(node_map, edges, target_node_id)
    if not execution_order:
        return frame

    working_frame = frame.copy()
    skip_types = set(skip_catalog_types or [])

    for node_id in execution_order:
        if target_node_id and node_id == target_node_id:
            break

        node = node_map.get(node_id)
        if not node:
            continue

        catalog_type = _resolve_catalog_type(node)
        split_info = detect_splits(working_frame)
        log_split_processing(
            node_id=node_id,
            catalog_type=catalog_type,
            split_info=split_info,
            action="preprocessing",
        )

        if _should_skip_preprocessing_node(node, catalog_type, skip_types):
            continue

        working_frame, handled = _invoke_node_transform(
            catalog_type,
            working_frame,
            node,
            pipeline_id,
        )
        if handled:
            continue

        if catalog_type == "train_test_split":
            working_frame = _apply_train_test_split_with_filter(
                working_frame,
                node,
                target_node_id,
                edges,
                node_map,
            )

    # Remove internal split column before returning
    working_frame = remove_split_column(working_frame)

    return working_frame


def _coerce_int(value: Any, fallback: int) -> int:
    """Safely coerce a value to int, returning fallback on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return fallback


def _resolve_sample_cap(requested_size: Any) -> int:
    """Ensure the EDA service sample cap honors larger caller requests."""
    try:
        resolved = int(requested_size)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return DEFAULT_SAMPLE_CAP

    if resolved <= 0:
        return DEFAULT_SAMPLE_CAP

    return max(DEFAULT_SAMPLE_CAP, resolved)


def _build_eda_service(session: AsyncSession, requested_size: Any) -> FeatureEngineeringEDAService:
    return FeatureEngineeringEDAService(session, sample_cap=_resolve_sample_cap(requested_size))


async def _load_dataset_frame(
    session: AsyncSession,
    dataset_source_id: str,
    *,
    sample_size: int,
    mode: str = "head",
    execution_mode: Literal["auto", "sample", "full"] = "auto",
    allow_empty_sample: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Resolve a dataset frame using sampling or full capture as needed."""

    normalized_id = dataset_source_id.strip() if isinstance(dataset_source_id, str) else str(dataset_source_id)
    if not normalized_id:
        raise HTTPException(status_code=400, detail="dataset_source_id must not be empty")

    normalized_mode = (mode or "head").strip().lower() or "head"
    should_use_full_capture = (
        execution_mode == "full"
        or (execution_mode == "auto" and sample_size == 0 and not allow_empty_sample)
    )

    if should_use_full_capture:
        capture_service = FullDatasetCaptureService(session)
        try:
            frame, metadata = await capture_service.capture(normalized_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        total_rows = _coerce_int(metadata.get("total_rows"), frame.shape[0])
        return frame, {
            "total_rows": total_rows,
            "sample_size": int(frame.shape[0]),
            "columns": metadata.get("columns") or frame.columns.tolist(),
            "dtypes": metadata.get("dtypes") or {},
            "mode": "full_capture",
            "sampling_adjustments": [],
            "large_dataset": False,
        }

    effective_sample = sample_size if sample_size > 0 else DEFAULT_SAMPLE_CAP
    if allow_empty_sample and sample_size == 0:
        effective_sample = 0
    eda_service = _build_eda_service(session, effective_sample)
    preview_payload = await eda_service.preview_source(
        normalized_id,
        sample_size=effective_sample,
        mode=normalized_mode,
    )

    if not preview_payload.get("success"):
        detail = preview_payload.get("error") or preview_payload.get("message") or "Unable to preview dataset"
        raise HTTPException(status_code=400, detail=detail)

    preview = preview_payload.get("preview") or {}
    sample_rows = preview.get("sample_data") or []

    try:
        frame = pd.DataFrame(sample_rows)
    except Exception:  # pragma: no cover - defensive
        frame = pd.DataFrame()

    if frame.empty:
        columns = preview.get("columns") or []
        if columns:
            frame = pd.DataFrame(columns=columns)

    total_rows = _coerce_int(preview.get("total_rows"), frame.shape[0])
    sample_rows_used = _coerce_int(preview.get("sample_size"), frame.shape[0])

    return frame, {
        "total_rows": total_rows,
        "sample_size": sample_rows_used,
        "columns": preview.get("columns") or frame.columns.tolist(),
        "dtypes": preview.get("dtypes") or {},
        "mode": preview.get("mode") or normalized_mode,
        "sampling_adjustments": preview.get("sampling_adjustments") or [],
        "large_dataset": bool(preview.get("large_dataset", False)),
    }


def _normalize_target_node(target_node_id: Optional[str]) -> Optional[str]:
    if isinstance(target_node_id, str):
        stripped = target_node_id.strip()
        if stripped:
            return stripped
    return None


def _build_preview_frame(preview_payload: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    preview_data = preview_payload.get("preview") or {}
    sample_rows = preview_data.get("sample_data") or []

    try:
        frame = pd.DataFrame(sample_rows)
    except Exception:  # pragma: no cover - defensive fallback
        frame = pd.DataFrame()

    if frame.empty:
        columns = preview_data.get("columns") or []
        if columns:
            frame = pd.DataFrame(columns=columns)

    return frame, preview_data


def _require_dataset_source_id(raw_dataset_id: Any) -> str:
    normalized_id = str(raw_dataset_id).strip()
    if not normalized_id:
        raise HTTPException(status_code=400, detail="dataset_source_id must not be empty")
    return normalized_id


async def _load_preview_frame_for_recommendations(
    session: AsyncSession,
    dataset_source_id: str,
    sample_size: Any,
) -> pd.DataFrame:
    eda_service = _build_eda_service(session, sample_size)
    preview_payload = await eda_service.preview_source(dataset_source_id, sample_size=sample_size)

    if not preview_payload.get("success"):
        detail = preview_payload.get("error") or preview_payload.get("message") or "Unable to preview dataset"
        raise HTTPException(status_code=400, detail=detail)

    frame, _ = _build_preview_frame(preview_payload)
    return frame


def _ensure_dict_graph_payload(graph_input: Any) -> Optional[Dict[str, Any]]:
    if not graph_input:
        return None

    if isinstance(graph_input, dict):
        return graph_input

    raise HTTPException(status_code=400, detail="graph payload must be a JSON object")


def _apply_skewness_graph_context(
    frame: pd.DataFrame,
    graph_payload: Optional[Dict[str, Any]],
    target_node_id: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    node_map, graph_edges = _extract_graph_payload(graph_payload)
    normalized_target = _normalize_target_node(target_node_id)

    if not (node_map or graph_edges or normalized_target):
        return frame, {}

    ensured_map = _ensure_dataset_node(node_map)
    execution_order = _execution_order(ensured_map, graph_edges, normalized_target)

    transformed_frame = frame
    try:
        transformed_frame = _apply_graph_transformations_before_node(
            frame,
            ensured_map,
            graph_edges,
            normalized_target,
            skip_catalog_types=None,
            pipeline_id=None,
        )
    except Exception:  # pragma: no cover - defensive
        pass

    graph_selected_methods = _collect_skewness_transformations_from_graph(
        execution_order,
        ensured_map,
        normalized_target,
    )

    return transformed_frame, graph_selected_methods


def _resolve_binned_graph_payload(graph_input: Any) -> Optional[Dict[str, Any]]:
    if graph_input is None:
        return None

    if isinstance(graph_input, dict):
        return graph_input

    if isinstance(graph_input, str):
        graph_text = graph_input.strip()
        if not graph_text:
            return None

        try:
            parsed = json.loads(graph_text)
        except (TypeError, ValueError) as exc:  # pragma: no cover - bad payload
            logger.error(
                f"Failed to parse graph JSON in binned-distribution endpoint: {exc}",
                extra={
                    "endpoint": "/api/analytics/binned-distribution",
                    "graph_type": type(graph_input).__name__,
                    "graph_length": len(graph_text),
                    "graph_preview": graph_text[:200] if graph_text else None,
                },
            )
            raise HTTPException(status_code=400, detail="graph must be valid JSON")

        if not isinstance(parsed, dict):  # pragma: no cover - invalid type
            raise HTTPException(status_code=400, detail="graph payload must be a JSON object")

        return parsed

    raise HTTPException(status_code=400, detail="graph payload must be a JSON object")


def _apply_graph_with_execution_order(
    frame: pd.DataFrame,
    node_map: Dict[str, Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    target_node: Optional[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Dict[str, Any]]]:
    if not (node_map or graph_edges or target_node):
        return frame, [], node_map

    ensured_map = _ensure_dataset_node(node_map)
    execution_order = _execution_order(ensured_map, graph_edges, target_node)

    transformed_frame = frame
    try:
        transformed_frame = _apply_graph_transformations_before_node(
            frame,
            ensured_map,
            graph_edges,
            target_node,
            skip_catalog_types=None,
            pipeline_id=None,
        )
    except Exception:  # pragma: no cover - defensive
        pass

    return transformed_frame, execution_order, ensured_map


def _build_candidate_binned_columns(
    frame: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    candidate_columns: Dict[str, Dict[str, Any]] = {}

    for column in frame.columns:
        column_name = str(column)
        if not column_name:
            continue

        metadata_entry = metadata.get(column_name)
        if metadata_entry:
            candidate_columns[column_name] = metadata_entry
            continue

        if column_name.endswith(BINNING_DEFAULT_SUFFIX):
            source_candidate = column_name[: -len(BINNING_DEFAULT_SUFFIX)] or None
            candidate_columns.setdefault(
                column_name,
                {
                    "source_column": source_candidate,
                    "missing_label": BINNING_DEFAULT_MISSING_LABEL,
                },
            )

    return candidate_columns


def _build_binned_distributions_list(
    frame: pd.DataFrame,
    candidate_columns: Dict[str, Dict[str, Any]],
) -> List[BinnedColumnDistribution]:
    distributions: List[BinnedColumnDistribution] = []

    for column_name, metadata in candidate_columns.items():
        if column_name not in frame.columns:
            continue

        series = frame[column_name]
        distribution = _build_binned_distribution(
            column_name,
            series,
            source_column=metadata.get("source_column"),
            missing_label=metadata.get("missing_label"),
        )

        if distribution:
            distributions.append(distribution)

    distributions.sort(
        key=lambda item: (
            -(item.top_count or 0),
            item.column.lower(),
        )
    )

    return distributions


def _extract_graph_payload(graph: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    if not graph:
        return {}, []

    if not isinstance(graph, dict):  # pragma: no cover - invalid type
        raise HTTPException(status_code=400, detail="graph payload must be a JSON object")

    node_map = _sanitize_graph_nodes(graph.get("nodes"))
    graph_edges = _sanitize_graph_edges(graph.get("edges"))
    return node_map, graph_edges


def _apply_recommendation_graph(
    frame: pd.DataFrame,
    node_map: Dict[str, Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    target_node: Optional[str],
    skip_catalog_types: Optional[Set[str]],
) -> pd.DataFrame:
    if not (node_map or graph_edges or target_node):
        return frame

    ensured_map = _ensure_dataset_node(node_map)
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
        transformed_frame = _apply_graph_transformations_before_node(
            frame,
            ensured_map,
            graph_edges,
            target_node,
            skip_catalog_types=set(skip_catalog_types or []),
            pipeline_id=None,
        )
        logger.debug(
            "Graph transformations completed",
            extra={"result_shape": transformed_frame.shape, "result_columns": len(transformed_frame.columns)},
        )
        return transformed_frame
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to apply graph transformations in recommendations")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply graph transformations: {exc}",
        ) from exc


def _build_recommendation_column_metadata(
    quality_payload: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    column_metadata: Dict[str, Dict[str, Any]] = {}
    notes: List[str] = []

    if not quality_payload.get("success"):
        detail = quality_payload.get("error") or quality_payload.get("message")
        if detail:
            notes.append(str(detail))
        return column_metadata, notes

    quality_report = quality_payload.get("quality_report") or {}
    quality_metrics = quality_report.get("quality_metrics") or {}
    column_details = quality_metrics.get("column_details") or []
    if isinstance(column_details, Iterable):
        for detail in column_details:
            if not isinstance(detail, dict):
                continue
            raw_name = detail.get("name") or detail.get("column")
            column_name = str(raw_name or "").strip()
            if not column_name:
                continue
            column_metadata[column_name] = detail

    text_summary = quality_report.get("text_analysis_summary") or {}
    categorical_columns = text_summary.get("categorical_text_columns") or []
    if isinstance(categorical_columns, Iterable):
        for column in categorical_columns:
            column_name = str(column or "").strip()
            if not column_name:
                continue
            meta = column_metadata.setdefault(column_name, {})
            if not meta.get("text_category"):
                meta["text_category"] = "categorical"

    return column_metadata, notes


async def _prepare_categorical_recommendation_context(
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
        detail = preview_payload.get("error") or preview_payload.get("message") or "Unable to preview dataset"
        raise HTTPException(status_code=400, detail=detail)

    frame, _ = _build_preview_frame(preview_payload)

    graph_node_map, graph_edges = _extract_graph_payload(graph)
    normalized_target_node = _normalize_target_node(target_node_id)
    frame = _apply_recommendation_graph(
        frame,
        graph_node_map,
        graph_edges,
        normalized_target_node,
        skip_catalog_types,
    )

    quality_payload = await eda_service.quality_report(dataset_source_id, sample_size=sample_size)
    column_metadata, notes = _build_recommendation_column_metadata(quality_payload)

    return frame, column_metadata, notes


def _parse_skewness_transformations(raw_payload: Optional[str]) -> Dict[str, str]:
    if not raw_payload:
        return {}

    try:
        payload = json.loads(raw_payload)
    except (TypeError, ValueError):  # pragma: no cover - bad payload
        raise HTTPException(status_code=400, detail="transformations must be valid JSON")

    parsed: Dict[str, str] = {}

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            column = str(item.get("column") or "").strip()
            method = str(item.get("method") or "").strip().lower()
            if column and method in SKEWNESS_METHODS:
                parsed[column] = method
    elif isinstance(payload, dict):
        for column, method in payload.items():
            column_key = str(column or "").strip()
            method_key = str(method or "").strip().lower()
            if column_key and method_key in SKEWNESS_METHODS:
                parsed[column_key] = method_key
    else:  # pragma: no cover - unsupported payload type
        raise HTTPException(status_code=400, detail="transformations payload must be an object or list")

    return parsed


def _collect_skewness_transformations_from_graph(
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    target_node_id: Optional[str],
) -> Dict[str, str]:
    selections: Dict[str, str] = {}

    for node_id in execution_order:
        if target_node_id and node_id == target_node_id:
            break

        node = node_map.get(node_id)
        if not node:
            continue

        if _is_node_pending(node):
            continue

        if _resolve_catalog_type(node) != "skewness_transform":
            continue

        data = node.get("data") or {}
        config = data.get("config") or {}
        raw_transformations = config.get("transformations")

        if not isinstance(raw_transformations, list):
            continue

        for entry in raw_transformations:
            if not isinstance(entry, dict):
                continue
            column = str(entry.get("column") or "").strip()
            method = str(entry.get("method") or "").strip().lower()
            if column and method in SKEWNESS_METHODS:
                selections[column] = method

    return selections


def _collect_binned_columns_from_graph(
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    target_node_id: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    columns: Dict[str, Dict[str, Any]] = {}

    for node_id in execution_order:
        if target_node_id and node_id == target_node_id:
            break

        node = node_map.get(node_id)
        if not node:
            continue

        if _is_node_pending(node):
            continue

        if _resolve_catalog_type(node) != "binning_discretization":
            continue

        data = node.get("data") or {}
        config_payload = data.get("config") or {}
        config = _normalize_binning_config(config_payload)

        for raw_column in config.columns:
            source_column = str(raw_column or "").strip()
            if not source_column:
                continue
            suffix = config.output_suffix or BINNING_DEFAULT_SUFFIX
            new_column = f"{source_column}{suffix}"
            columns[new_column] = {
                "source_column": source_column,
                "missing_label": config.missing_label or BINNING_DEFAULT_MISSING_LABEL,
                "config": config,
            }

    return columns


@dataclass(frozen=True)
class NodeExecutionContext:
    pipeline_id: Optional[str]
    node_map: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class NodeExecutionResult:
    frame: pd.DataFrame
    summary: str
    signal: Any = None


@dataclass(frozen=True)
class NodeExecutionSpec:
    handler: Callable[[pd.DataFrame, Dict[str, Any], NodeExecutionContext], NodeExecutionResult]
    signal_attr: Optional[str] = None
    signal_mode: Literal["append", "assign"] = "append"
    update_modeling_metadata: bool = False


def _wrap_node_handler(
    func: Callable[..., Tuple[pd.DataFrame, str, Any]],
    *,
    requires_pipeline: bool = False,
    requires_node_map: bool = False,
) -> Callable[[pd.DataFrame, Dict[str, Any], NodeExecutionContext], NodeExecutionResult]:
    def handler(frame: pd.DataFrame, node: Dict[str, Any], context: NodeExecutionContext) -> NodeExecutionResult:
        kwargs: Dict[str, Any] = {}
        if requires_pipeline:
            kwargs["pipeline_id"] = context.pipeline_id
        if requires_node_map:
            kwargs["node_map"] = context.node_map
        new_frame, summary, signal = func(frame, node, **kwargs)
        return NodeExecutionResult(new_frame, summary, signal)

    return handler


def _model_registry_overview_handler(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    _: NodeExecutionContext,
) -> NodeExecutionResult:
    label = _resolve_node_label(node)
    summary = f"{label}: model registry view – no transformation applied"
    return NodeExecutionResult(frame, summary, None)


_NODE_EXECUTION_SPECS: Dict[str, NodeExecutionSpec] = {
    "drop_missing_columns": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_drop_missing_columns),
        signal_attr="drop_missing_columns",
    ),
    "drop_missing_rows": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_drop_missing_rows),
        signal_attr="drop_missing_rows",
    ),
    "remove_duplicates": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_remove_duplicates),
        signal_attr="remove_duplicates",
    ),
    "missing_value_indicator": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_missing_value_flags),
        signal_attr="missing_value_indicator",
    ),
    "cast_column_types": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_cast_column_types),
        signal_attr="cast_column_types",
    ),
    "trim_whitespace": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_trim_whitespace),
        signal_attr="trim_whitespace",
    ),
    "normalize_text_case": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_normalize_text_case),
        signal_attr="normalize_text_case",
    ),
    "replace_aliases_typos": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_replace_aliases_typos),
        signal_attr="replace_aliases",
    ),
    "standardize_date_formats": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_standardize_date_formats),
        signal_attr="standardize_dates",
    ),
    "remove_special_characters": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_remove_special_characters),
        signal_attr="remove_special_characters",
    ),
    "replace_invalid_values": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_replace_invalid_values),
        signal_attr="replace_invalid_values",
    ),
    "feature_math": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_feature_math, requires_pipeline=True),
        signal_attr="feature_math",
    ),
    "regex_replace_fix": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_regex_cleanup),
        signal_attr="regex_cleanup",
    ),
    "imputation_methods": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_imputation_methods, requires_pipeline=True),
        signal_attr="imputation",
    ),
    "advanced_imputer": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_imputation_methods, requires_pipeline=True),
        signal_attr="imputation",
    ),
    "simple_imputer": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_imputation_methods, requires_pipeline=True),
        signal_attr="imputation",
    ),
    "binning_discretization": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_binning_discretization, requires_pipeline=True),
        signal_attr="binning",
    ),
    "skewness_transform": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_skewness_transformations, requires_pipeline=True),
        signal_attr="skewness_transform",
    ),
    "scale_numeric_features": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_scale_numeric_features, requires_pipeline=True),
        signal_attr="scaling",
    ),
    "feature_selection": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_feature_selection, requires_pipeline=True),
        signal_attr="feature_selection",
    ),
    "polynomial_features": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_polynomial_features, requires_pipeline=True),
        signal_attr="polynomial_features",
    ),
    "outlier_removal": NodeExecutionSpec(
        handler=_wrap_node_handler(_apply_outlier_removal, requires_pipeline=True),
        signal_attr="outlier_removal",
    ),
    "feature_target_split": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_feature_target_split),
        signal_attr="feature_target_split",
    ),
    "train_test_split": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_train_test_split),
        signal_attr="train_test_split",
    ),
    "class_undersampling": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_resampling),
        signal_attr="class_undersampling",
    ),
    "class_oversampling": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_oversampling),
        signal_attr="class_oversampling",
    ),
    "label_encoding": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_label_encoding, requires_pipeline=True),
        signal_attr="label_encoding",
    ),
    "target_encoding": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_target_encoding, requires_pipeline=True),
        signal_attr="target_encoding",
    ),
    "hash_encoding": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_hash_encoding, requires_pipeline=True),
        signal_attr="hash_encoding",
    ),
    "train_model_draft": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_train_model_draft),
        signal_attr="modeling",
        signal_mode="assign",
        update_modeling_metadata=True,
    ),
    "model_evaluation": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_model_evaluation, requires_pipeline=True),
        signal_attr="model_evaluation",
    ),
    "model_registry_overview": NodeExecutionSpec(
        handler=_model_registry_overview_handler,
        signal_attr=None,
    ),
    "ordinal_encoding": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_ordinal_encoding, requires_pipeline=True),
        signal_attr="ordinal_encoding",
    ),
    "one_hot_encoding": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_one_hot_encoding, requires_pipeline=True),
        signal_attr="one_hot_encoding",
    ),
    "dummy_encoding": NodeExecutionSpec(
        handler=_wrap_node_handler(apply_dummy_encoding, requires_pipeline=True),
        signal_attr="dummy_encoding",
    ),
    "transformer_audit": NodeExecutionSpec(
        handler=_wrap_node_handler(
            apply_transformer_audit,
            requires_pipeline=True,
            requires_node_map=True,
        ),
        signal_attr="transformer_audit",
    ),
}


@dataclass
class PipelineNodeOutcome:
    frame: pd.DataFrame
    summary: Optional[str] = None
    modeling_metadata: Optional[TrainModelDraftReadinessSnapshot] = None


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
        return f"Dataset input '{label}'"
    if catalog_type == "data_preview":
        return f"Preview node '{label}': inspection only"
    return None


def _execute_pipeline_node(
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
        return PipelineNodeOutcome(frame)

    catalog_type = _resolve_catalog_type(node)
    label = _resolve_node_label(node)

    _log_node_split_state(frame, node_id=node_id, catalog_type=catalog_type)

    if _is_node_pending(node):
        return PipelineNodeOutcome(frame, f"{label}: pending configuration – skipped")

    special_summary = _special_catalog_summary(catalog_type, label)
    if special_summary:
        return PipelineNodeOutcome(frame, special_summary)

    spec = _NODE_EXECUTION_SPECS.get(catalog_type)
    if not spec:
        return PipelineNodeOutcome(frame, f"Node '{label}' ({catalog_type}) not executed in preview")

    result = spec.handler(frame, node, context)

    if collect_signals and signals is not None and spec.signal_attr:
        if spec.signal_mode == "assign":
            setattr(signals, spec.signal_attr, result.signal)
        else:
            getattr(signals, spec.signal_attr).append(result.signal)

    modeling_metadata = result.signal if spec.update_modeling_metadata else None

    return PipelineNodeOutcome(result.frame, result.summary, modeling_metadata)


def _run_pipeline_execution(
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
        outcome = _execute_pipeline_node(
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

        if outcome.modeling_metadata is not None:
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

    frame_result, _, signals, modeling_metadata = _run_pipeline_execution(
        frame,
        execution_order,
        node_map,
        pipeline_id=pipeline_id,
        collect_signals=True,
        existing_signals=existing_signals,
        preserve_split_column=preserve_split_column,
    )

    return frame_result, signals or PipelinePreviewSignals(), modeling_metadata


class FeatureNodeParameterOption(TypedDict, total=False):
    """Enumerated option for select-style parameters."""

    value: str
    label: str
    description: Optional[str]
    metadata: Dict[str, Any]


class FeatureNodeParameterSource(TypedDict, total=False):
    """External source definition for dynamic parameter options."""

    type: str
    endpoint: str
    value_key: str


class FeatureNodeParameter(TypedDict, total=False):
    """Schema describing configurable parameters for a node."""

    name: str
    label: str
    description: Optional[str]
    type: str
    default: Optional[Any]
    min: Optional[float]
    max: Optional[float]
    step: Optional[float]
    unit: Optional[str]
    placeholder: Optional[str]
    options: List[FeatureNodeParameterOption]
    source: FeatureNodeParameterSource


class FeatureNodeCatalogEntryBase(TypedDict):
    """Schema for node catalog entries."""

    type: str
    label: str
    description: str
    inputs: List[str]
    outputs: List[str]


class FeatureNodeCatalogEntry(FeatureNodeCatalogEntryBase, total=False):
    category: str
    tags: List[str]
    parameters: List[FeatureNodeParameter]
    default_config: Dict[str, Any]


def _make_option(
    value: str,
    label: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FeatureNodeParameterOption:
    option: FeatureNodeParameterOption = {"value": value, "label": label}
    if description:
        option["description"] = description
    if metadata:
        option["metadata"] = metadata
    return option


@router.get("/api/datasets", response_model=List[DatasetSourceSummary])
async def list_active_datasets(  # pragma: no cover - simple read endpoint
    limit: int = Query(8, ge=1, le=50),
    session: AsyncSession = Depends(get_async_session),
) -> List[DatasetSourceSummary]:
    """Return a trimmed list of active dataset sources for quick selection."""

    limit_value = max(1, min(limit, 50))

    result = await session.execute(
        select(DataSource)
        .where(DataSource.is_active.is_(True))
        .order_by(DataSource.updated_at.desc(), DataSource.id.desc())
        .limit(limit_value)
    )

    datasets = result.scalars().all()

    summaries: List[DatasetSourceSummary] = []
    for dataset in datasets:
        dataset_id = cast(Optional[int], dataset.id)
        if dataset_id is None:
            continue
        source_id_value = cast(Optional[str], dataset.source_id) or str(dataset_id)
        summaries.append(
            DatasetSourceSummary(
                id=int(dataset_id),
                source_id=source_id_value,
                name=cast(Optional[str], dataset.name),
                description=cast(Optional[str], dataset.description),
                created_at=cast(Optional[datetime], dataset.created_at),
            )
        )

    return summaries


@router.get("/api/node-catalog", response_model=List[FeatureNodeCatalogEntry])
async def get_node_catalog() -> List[FeatureNodeCatalogEntry]:
    """Return the prototype node catalog for the feature engineering canvas."""

    def _build_preprocessing_nodes() -> List[FeatureNodeCatalogEntry]:
        drop_missing_node: FeatureNodeCatalogEntry = {
            "type": "drop_missing_columns",
            "label": "Drop high-missing columns",
            "description": "Drop columns",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA PREPROCESSING",
            "tags": ["missing_data", "cleanup"],
            "parameters": [
                {
                    "name": "missing_threshold",
                    "label": "Missingness threshold (%)",
                    "description": "Columns at or above this missing percentage will be removed.",
                    "type": "number",
                    "default": 40.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "unit": "%",
                },
                {
                    "name": "columns",
                    "label": "Columns to drop (recommended)",
                    "description": (
                        "Pre-populated with EDA suggestions covering high-missing, empty, or constant columns."
                    ),
                    "type": "multi_select",
                    "source": {
                        "type": "drop_column_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/drop-columns",
                        "value_key": "candidates",
                    },
                },
            ],
            "default_config": {
                "missing_threshold": 40.0,
                "columns": [],
            },
        }

        drop_missing_rows_node: FeatureNodeCatalogEntry = {
            "type": "drop_missing_rows",
            "label": "Drop high-missing rows",
            "description": "Remove rows with excessive missing values.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA PREPROCESSING",
            "tags": ["missing_data", "cleanup"],
            "parameters": [
                {
                    "name": "missing_threshold",
                    "label": "Missingness threshold (%)",
                    "description": "Rows at or above this missing percentage will be removed.",
                    "type": "number",
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "unit": "%",
                },
                {
                    "name": "drop_if_any_missing",
                    "label": "Drop rows with any missing value",
                    "description": "Override the threshold and drop rows that contain any missing value.",
                    "type": "boolean",
                    "default": False,
                },
            ],
            "default_config": {
                "missing_threshold": 50.0,
                "drop_if_any_missing": False,
            },

        }

        outlier_removal_node: FeatureNodeCatalogEntry = {
            "type": "outlier_removal",
            "label": "Remove outliers",
            "description": "Identify and handle numeric outliers using z-score, IQR, winsorization, or manual bounds.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA PREPROCESSING",
            "tags": ["numeric", "cleanup", "quality"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to inspect",
                    "description": "Select numeric columns manually or rely on auto-detection.",
                    "type": "multi_select",
                },
                {
                    "name": "default_method",
                    "label": "Default outlier strategy",
                    "description": "Method applied when a column override is not configured.",
                    "type": "text",
                    "default": OUTLIER_DEFAULT_METHOD,
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect numeric columns",
                    "description": "Automatically include numeric columns that pass quality checks.",
                    "type": "boolean",
                    "default": True,
                },
            ],
            "default_config": {
                "columns": [],
                "default_method": OUTLIER_DEFAULT_METHOD,
                "column_methods": {},
                "auto_detect": True,
                "skipped_columns": [],
                "method_parameters": {
                    key: dict(value)
                    for key, value in DEFAULT_METHOD_PARAMETERS.items()
                },
                "column_parameters": {},
            },
        }

        missing_indicator_node: FeatureNodeCatalogEntry = {
            "type": "missing_value_indicator",
            "label": "Missing value indicator",
            "description": "Append binary *_was_missing columns that flag rows with missing data.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "FEATURE ENGINEERING",
            "tags": ["missing_data", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to flag",
                    "description": (
                        "Choose columns to generate missing-value indicator fields "
                        "(default: columns with missing data)."
                    ),
                    "type": "multi_select",
                },
                {
                    "name": "flag_suffix",
                    "label": "Indicator suffix",
                    "description": (
                        "Suffix appended to the original column name when "
                        "creating the flag column."
                    ),
                    "type": "text",
                    "default": "_was_missing",
                },
            ],
            "default_config": {
                "columns": [],
                "flag_suffix": "_was_missing",
            },
        }

        remove_duplicates_node: FeatureNodeCatalogEntry = {
            "type": "remove_duplicates",
            "label": "Remove duplicate rows",
            "description": "Drop duplicate rows based on all columns or a selected subset.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA PREPROCESSING",
            "tags": ["duplicates", "cleanup"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to compare",
                    "description": "Leave blank to compare all columns when detecting duplicates.",
                    "type": "multi_select",
                },
                {
                    "name": "keep",
                    "label": "Keep strategy",
                    "description": "Accepts 'first', 'last', or 'none' (drop all duplicates).",
                    "type": "text",
                    "default": "first",
                },
            ],
            "default_config": {
                "columns": [],
                "keep": "first",
            },
        }

        cast_column_types_node: FeatureNodeCatalogEntry = {
            "type": "cast_column_types",
            "label": "Cast column types",
            "description": "Convert selected columns to a target pandas dtype (e.g., float64, Int64, datetime).",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA PREPROCESSING",
            "tags": ["typing", "cleanup"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to cast",
                    "description": "Select columns to convert to the target dtype.",
                    "type": "multi_select",
                },
                {
                    "name": "target_dtype",
                    "label": "Target dtype",
                    "description": "Enter a pandas dtype such as float64, Int64, string, boolean, or datetime64[ns].",
                    "type": "text",
                },
                {
                    "name": "coerce_on_error",
                    "label": "Coerce invalid values",
                    "description": "Convert unparseable values to missing instead of raising errors.",
                    "type": "boolean",
                    "default": True,
                },
            ],
            "default_config": {
                "columns": [],
                "target_dtype": "float64",
                "coerce_on_error": True,
            },
        }

        imputation_methods_node: FeatureNodeCatalogEntry = {
            "type": "imputation_methods",
            "label": "Imputation methods",
            "description": "Configure statistical or model-driven fills (mean/median/mode, KNN, regression, MICE).",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA PREPROCESSING",
            "tags": ["missing_data", "imputation", "advanced"],
            "parameters": [
                {
                    "name": "strategies",
                    "label": "Imputation recipes",
                    "description": "Configure multivariate strategies like KNN, regression, or MICE per column group.",
                    "type": "text",
                }
            ],
            "default_config": {
                "strategies": [],
            },
        }

        trim_whitespace_node: FeatureNodeCatalogEntry = {
            "type": "trim_whitespace",
            "label": "Trim whitespace",
            "description": "Remove leading/trailing whitespace from text columns.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA CONSISTENCY",
            "tags": ["text", "cleanup", "standardization"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Target columns",
                    "description": "Leave blank to auto-detect text columns from the preview sample.",
                    "type": "multi_select",
                },
                {
                    "name": "mode",
                    "label": "Trim mode",
                    "description": "Choose whether to trim leading, trailing, or both sides of whitespace.",
                    "type": "select",
                    "default": "both",
                    "options": [
                        {"value": "both", "label": "Leading and trailing"},
                        {"value": "leading", "label": "Leading only"},
                        {"value": "trailing", "label": "Trailing only"},
                    ],
                },
            ],
            "default_config": {
                "columns": [],
                "mode": "both",
            },
        }

        normalize_text_case_node: FeatureNodeCatalogEntry = {
            "type": "normalize_text_case",
            "label": "Normalize text case",
            "description": "Convert text columns to a consistent case (lower, upper, title, sentence).",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA CONSISTENCY",
            "tags": ["text", "standardization"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Target columns",
                    "description": "Leave blank to auto-detect text columns from the preview sample.",
                    "type": "multi_select",
                },
                {
                    "name": "mode",
                    "label": "Case style",
                    "description": "Supported modes: lower, upper, title, sentence.",
                    "type": "select",
                    "default": "lower",
                    "options": [
                        {"value": "lower", "label": "Lowercase"},
                        {"value": "upper", "label": "Uppercase"},
                        {"value": "title", "label": "Title case"},
                        {"value": "sentence", "label": "Sentence case"},
                    ],
                },
            ],
            "default_config": {
                "columns": [],
                "mode": "lower",
            },
        }

        replace_aliases_node: FeatureNodeCatalogEntry = {
            "type": "replace_aliases_typos",
            "label": "Standardize aliases & typos",
            "description": "Normalize common aliases (countries, booleans) or apply custom replacements.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA CONSISTENCY",
            "tags": ["text", "standardization", "cleanup"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Target columns",
                    "description": "Leave blank to auto-detect text columns from the preview sample.",
                    "type": "multi_select",
                },
                {
                    "name": "mode",
                    "label": "Replacement strategy",
                    "description": "Use presets for common aliases or switch to custom mappings.",
                    "type": "select",
                    "default": "canonicalize_country_codes",
                    "options": [
                        {"value": "canonicalize_country_codes", "label": "Country aliases"},
                        {"value": "normalize_boolean", "label": "Boolean tokens"},
                        {"value": "punctuation", "label": "Strip punctuation"},
                        {"value": "custom", "label": "Custom mappings"},
                    ],
                },
                {
                    "name": "custom_pairs",
                    "label": "Custom pairs",
                    "description": (
                        "Provide alias => replacement pairs (one per line). Applies "
                        "when mode is set to Custom."
                    ),
                    "type": "textarea",
                    "placeholder": "st -> Street\nrd => Road",
                },
            ],
            "default_config": {
                "columns": [],
                "mode": "canonicalize_country_codes",
                "custom_pairs": "",
            },
        }

        standardize_dates_node: FeatureNodeCatalogEntry = {
            "type": "standardize_date_formats",
            "label": "Standardize date formats",
            "description": "Parse and rewrite date/time strings into a consistent format.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA CONSISTENCY",
            "tags": ["datetime", "cleanup", "standardization"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Target columns",
                    "description": "Leave blank to auto-detect datetime-like columns from the preview sample.",
                    "type": "multi_select",
                },
                {
                    "name": "mode",
                    "label": "Output format",
                    "description": "Select the desired date format for parsed values.",
                    "type": "select",
                    "default": "iso_date",
                    "options": [
                        {"value": "iso_date", "label": "ISO date (YYYY-MM-DD)"},
                        {"value": "iso_datetime", "label": "ISO datetime (YYYY-MM-DD HH:MM:SS)"},
                        {"value": "month_day_year", "label": "MM/DD/YYYY"},
                        {"value": "day_month_year", "label": "DD/MM/YYYY"},
                    ],
                },
            ],
            "default_config": {
                "columns": [],
                "mode": "iso_date",
            },
        }

        remove_special_chars_node: FeatureNodeCatalogEntry = {
            "type": "remove_special_characters",
            "label": "Remove special characters",
            "description": "Strip or replace non-alphanumeric characters in text columns.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA CONSISTENCY",
            "tags": ["text", "cleanup"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Target columns",
                    "description": "Leave blank to auto-detect text columns from the preview sample.",
                    "type": "multi_select",
                },
                {
                    "name": "mode",
                    "label": "Retention rule",
                    "description": "Choose which character classes should be preserved.",
                    "type": "select",
                    "default": "keep_alphanumeric",
                    "options": [
                        {"value": "keep_alphanumeric", "label": "Keep letters & digits"},
                        {"value": "keep_alphanumeric_space", "label": "Keep letters, digits & spaces"},
                        {"value": "letters_only", "label": "Letters only"},
                        {"value": "digits_only", "label": "Digits only"},
                    ],
                },
                {
                    "name": "replacement",
                    "label": "Replacement",
                    "description": "Text to insert when removing characters (default: remove).",
                    "type": "text",
                    "default": "",
                },
            ],
            "default_config": {
                "columns": [],
                "mode": "keep_alphanumeric",
                "replacement": "",
            },
        }

        replace_invalid_values_node: FeatureNodeCatalogEntry = {
            "type": "replace_invalid_values",
            "label": "Replace invalid numeric values",
            "description": "Convert out-of-range or placeholder numeric values to missing.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA CONSISTENCY",
            "tags": ["numeric", "cleanup", "quality"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Target columns",
                    "description": "Leave blank to auto-detect numeric columns from the preview sample.",
                    "type": "multi_select",
                },
                {
                    "name": "mode",
                    "label": "Rule",
                    "description": "Select how invalid values are detected and replaced.",
                    "type": "select",
                    "default": "negative_to_nan",
                    "options": [
                        {"value": "negative_to_nan", "label": "Negative to missing"},
                        {"value": "zero_to_nan", "label": "Zero to missing"},
                        {"value": "percentage_bounds", "label": "Percentage bounds (0-100%)"},
                        {"value": "age_bounds", "label": "Age bounds (0-120)"},
                        {"value": "custom_range", "label": "Custom numeric range"},
                    ],
                },
                {
                    "name": "min_value",
                    "label": "Minimum value",
                    "description": "Optional lower bound (applies to percentage, age, or custom modes).",
                    "type": "number",
                },
                {
                    "name": "max_value",
                    "label": "Maximum value",
                    "description": "Optional upper bound (applies to percentage, age, or custom modes).",
                    "type": "number",
                },
            ],
            "default_config": {
                "columns": [],
                "mode": "negative_to_nan",
                "min_value": None,
                "max_value": None,
            },
        }

        regex_replace_node: FeatureNodeCatalogEntry = {
            "type": "regex_replace_fix",
            "label": "Regex cleanup",
            "description": "Apply preset or custom regular expression replacements to text columns.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "DATA CONSISTENCY",
            "tags": ["text", "regex", "cleanup"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Target columns",
                    "description": "Leave blank to auto-detect text columns from the preview sample.",
                    "type": "multi_select",
                },
                {
                    "name": "mode",
                    "label": "Regex preset",
                    "description": "Use a preset cleanup or switch to custom pattern/replacement.",
                    "type": "select",
                    "default": "normalize_slash_dates",
                    "options": [
                        {"value": "normalize_slash_dates", "label": "Normalize slash dates"},
                        {"value": "collapse_whitespace", "label": "Collapse whitespace"},
                        {"value": "extract_digits", "label": "Extract digits"},
                        {"value": "custom", "label": "Custom pattern"},
                    ],
                },
                {
                    "name": "pattern",
                    "label": "Custom pattern",
                    "description": "Regular expression applied when mode is Custom.",
                    "type": "text",
                    "placeholder": r"(?i)acct#:?",
                },
                {
                    "name": "replacement",
                    "label": "Replacement text",
                    "description": "Replacement string for the regex substitution (supports backreferences).",
                    "type": "text",
                    "default": "",
                },
            ],
            "default_config": {
                "columns": [],
                "mode": "normalize_slash_dates",
                "pattern": "",
                "replacement": "",
            },
        }

        binning_node: FeatureNodeCatalogEntry = {
            "type": "binning_discretization",
            "label": "Bin numeric columns",
            "description": "Discretize numeric columns with equal-width, equal-frequency, or custom thresholds.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "FEATURE ENGINEERING",
            "tags": ["numeric", "discretization", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to bin",
                    "description": "Select numeric columns to convert into discrete bins.",
                    "type": "multi_select",
                }
            ],
            "default_config": {
                "strategy": "equal_width",
                "columns": [],
                "equal_width_bins": BINNING_DEFAULT_EQUAL_WIDTH_BINS,
                "equal_frequency_bins": BINNING_DEFAULT_EQUAL_FREQUENCY_BINS,
                "include_lowest": True,
                "precision": BINNING_DEFAULT_PRECISION,
                "duplicates": "raise",
                "output_suffix": BINNING_DEFAULT_SUFFIX,
                "drop_original": False,
                "label_format": "range",
                "missing_strategy": "keep",
                "missing_label": BINNING_DEFAULT_MISSING_LABEL,
                "custom_bins": {},
                "custom_labels": {},
            },
        }

        polynomial_features_node: FeatureNodeCatalogEntry = {
            "type": "polynomial_features",
            "label": "Polynomial features",
            "description": "Generate polynomial and interaction terms for numeric columns.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "FEATURE ENGINEERING",
            "tags": ["numeric", "feature_engineering", "transformation"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to expand",
                    "description": "Select numeric columns manually or leave blank to auto-detect.",
                    "type": "multi_select",
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect numeric columns",
                    "description": "Automatically include numeric columns detected at runtime.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "degree",
                    "label": "Maximum degree",
                    "description": "Upper polynomial degree to generate (higher degrees create more features).",
                    "type": "number",
                    "default": 2,
                    "min": 2.0,
                    "max": 5.0,
                    "step": 1.0,
                },
                {
                    "name": "include_bias",
                    "label": "Add bias column",
                    "description": "Include a constant bias column when enabled.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "interaction_only",
                    "label": "Interaction only",
                    "description": "Generate only interaction features without power terms.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "include_input_features",
                    "label": "Include original features",
                    "description": "Retain degree-1 terms alongside the generated features.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "output_prefix",
                    "label": "Feature prefix",
                    "description": "Prefix applied to generated feature columns.",
                    "type": "text",
                    "default": "poly",
                },
            ],
            "default_config": {
                "columns": [],
                "auto_detect": False,
                "degree": 2,
                "include_bias": False,
                "interaction_only": False,
                "include_input_features": False,
                "output_prefix": "poly",
            },
        }

        feature_selection_node: FeatureNodeCatalogEntry = {
            "type": "feature_selection",
            "label": "Feature selection",
            "description": (
                "Score features and retain the most informative columns using "
                "univariate tests or model-based selectors."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "FEATURE ENGINEERING",
            "tags": ["feature_selection", "numeric", "modeling"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Candidate columns",
                    "description": "Select columns to score or leave blank to rely on auto-detection.",
                    "type": "multi_select",
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect numeric columns",
                    "description": "Automatically include numeric columns detected at runtime.",
                    "type": "boolean",
                    "default": True,
                },
                {
                    "name": "target_column",
                    "label": "Target column",
                    "description": (
                        "Supervised selectors require a target column for scoring "
                        "(e.g., regression/classification target)."
                    ),
                    "type": "text",
                    "placeholder": "target",
                },
                {
                    "name": "method",
                    "label": "Selection method",
                    "description": "Choose the feature selection strategy to apply.",
                    "type": "select",
                    "default": "select_k_best",
                    "options": [
                        {"value": "select_k_best", "label": "Select K Best"},
                        {"value": "select_percentile", "label": "Select Percentile"},
                        {"value": "generic_univariate_select", "label": "Generic univariate (mode driven)"},
                        {"value": "select_fpr", "label": "Select FPR"},
                        {"value": "select_fdr", "label": "Select FDR"},
                        {"value": "select_fwe", "label": "Select FWE"},
                        {"value": "select_from_model", "label": "Select From Model"},
                        {"value": "variance_threshold", "label": "Variance Threshold"},
                        {"value": "rfe", "label": "Recursive Feature Elimination"},
                    ],
                },
                {
                    "name": "score_func",
                    "label": "Scoring function",
                    "description": "Statistical test used for univariate selectors.",
                    "type": "select",
                    "default": "f_classif",
                    "options": [
                        {"value": "f_classif", "label": "ANOVA F-value (classification)"},
                        {"value": "f_regression", "label": "F-value (regression)"},
                        {"value": "mutual_info_classif", "label": "Mutual information (classification)"},
                        {"value": "mutual_info_regression", "label": "Mutual information (regression)"},
                        {"value": "chi2", "label": "Chi-squared (non-negative features)"},
                        {"value": "r_regression", "label": "Pearson r"},
                    ],
                },
                {
                    "name": "problem_type",
                    "label": "Problem type",
                    "description": "Guides default scoring function and estimator selection.",
                    "type": "select",
                    "default": "auto",
                    "options": [
                        {"value": "auto", "label": "Auto detect"},
                        {"value": "classification", "label": "Classification"},
                        {"value": "regression", "label": "Regression"},
                    ],
                },
                {
                    "name": "k",
                    "label": "Top K features",
                    "description": "Number of features to keep when using K-based strategies.",
                    "type": "number",
                    "min": 1.0,
                    "step": 1.0,
                    "default": 10.0,
                },
                {
                    "name": "percentile",
                    "label": "Percentile",
                    "description": "Percentile of features to retain when using percentile strategies.",
                    "type": "number",
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "default": 10.0,
                },
                {
                    "name": "alpha",
                    "label": "Alpha",
                    "description": "Significance threshold for FPR/FDR/FWE modes.",
                    "type": "number",
                    "min": 0.0,
                    "step": 0.001,
                    "default": 0.05,
                },
                {
                    "name": "threshold",
                    "label": "Threshold",
                    "description": "Threshold for variance or model-based selectors (leave blank for defaults).",
                    "type": "number",
                },
                {
                    "name": "mode",
                    "label": "Generic mode",
                    "description": "Mode parameter for GenericUnivariateSelect (k_best, percentile, fpr, fdr, fwe).",
                    "type": "select",
                    "default": "k_best",
                    "options": [
                        {"value": "k_best", "label": "K Best"},
                        {"value": "percentile", "label": "Percentile"},
                        {"value": "fpr", "label": "FPR"},
                        {"value": "fdr", "label": "FDR"},
                        {"value": "fwe", "label": "FWE"},
                    ],
                },
                {
                    "name": "estimator",
                    "label": "Estimator",
                    "description": "Base estimator used for model-based selectors (SelectFromModel / RFE).",
                    "type": "select",
                    "default": "auto",
                    "options": [
                        {"value": "auto", "label": "Auto"},
                        {"value": "logistic_regression", "label": "Logistic regression"},
                        {"value": "random_forest", "label": "Random forest"},
                        {"value": "linear_regression", "label": "Linear regression"},
                    ],
                },
                {
                    "name": "step",
                    "label": "RFE step",
                    "description": "Number (or fraction) of features to remove at each RFE iteration.",
                    "type": "number",
                    "step": 0.1,
                    "default": 1.0,
                },
                {
                    "name": "min_features",
                    "label": "Minimum features",
                    "description": "Optional lower bound on features to keep (used by some estimators).",
                    "type": "number",
                },
                {
                    "name": "max_features",
                    "label": "Maximum features",
                    "description": "Optional upper bound on features to keep (used by some estimators).",
                    "type": "number",
                },
                {
                    "name": "drop_unselected",
                    "label": "Drop unselected columns",
                    "description": "Remove columns that fail the selection criteria from the dataset.",
                    "type": "boolean",
                    "default": True,
                },
            ],
            "default_config": {
                "columns": [],
                "auto_detect": True,
                "target_column": "",
                "method": "select_k_best",
                "score_func": "f_classif",
                "problem_type": "auto",
                "k": 10,
                "percentile": 10.0,
                "alpha": 0.05,
                "threshold": None,
                "mode": "k_best",
                "estimator": "auto",
                "step": 1.0,
                "min_features": None,
                "max_features": None,
                "drop_unselected": True,
            },
        }

        feature_math_node: FeatureNodeCatalogEntry = {
            "type": "feature_math",
            "label": "Feature math lab",
            "description": (
                "Combine columns using arithmetic, ratios, statistics, similarity "
                "scores, and datetime extraction steps."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "FEATURE ENGINEERING",
            "tags": ["numeric", "datetime", "text", "feature_engineering"],
            "parameters": [
                {
                    "name": "error_handling",
                    "label": "On operation error",
                    "description": (
                        "Choose whether the pipeline should continue or fail when "
                        "an operation encounters invalid inputs."
                    ),
                    "type": "select",
                    "default": "skip",
                    "options": [
                        {"value": "skip", "label": "Skip and continue"},
                        {"value": "fail", "label": "Fail immediately"},
                    ],
                },
                {
                    "name": "allow_overwrite",
                    "label": "Allow overwriting columns",
                    "description": (
                        "Permit operations to overwrite existing columns when the "
                        "chosen output already exists."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "default_timezone",
                    "label": "Default timezone",
                    "description": "Timezone applied when extracting datetime features (IANA name).",
                    "type": "text",
                    "default": "UTC",
                },
                {
                    "name": "epsilon",
                    "label": "Division epsilon",
                    "description": (
                        "Small constant added to denominators to avoid "
                        "divide-by-zero when computing ratios."
                    ),
                    "type": "number",
                    "default": 1e-9,
                    "min": 0.0,
                    "step": 1e-9,
                },
            ],
            "default_config": {
                "operations": [],
                "error_handling": "skip",
                "allow_overwrite": False,
                "default_timezone": "UTC",
                "epsilon": 1e-9,
            },
        }

        undersampling_method_options: List[FeatureNodeParameterOption] = [
            _make_option(method, label)
            for method, label in UNDERSAMPLING_METHOD_LABELS.items()
        ]

        oversampling_method_options: List[FeatureNodeParameterOption] = [
            _make_option(method, label)
            for method, label in OVERSAMPLING_METHOD_LABELS.items()
        ]

        undersampling_node: FeatureNodeCatalogEntry = {
            "type": "class_undersampling",
            "label": "Class undersampling",
            "description": (
                "Reduce majority-class rows with random under-sampling to "
                "improve balance."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "RESAMPLING DATASET",
            "tags": ["undersampling", "class_balance", "imbalanced"],
            "parameters": [
                {
                    "name": "method",
                    "label": "Resampling method",
                    "description": (
                        "Choose the sampling approach (under-sampling available now)."
                    ),
                    "type": "select",
                    "default": UNDERSAMPLING_DEFAULT_METHOD,
                    "options": undersampling_method_options,
                },
                {
                    "name": "target_column",
                    "label": "Target column",
                    "description": "Categorical target column used to guide sampling.",
                    "type": "text",
                    "placeholder": "target",
                },
                {
                    "name": "sampling_strategy",
                    "label": "Sampling ratio",
                    "description": "Minority-to-majority ratio (0 < ratio ≤ 1). Leave blank for auto.",
                    "type": "number",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                },
                {
                    "name": "random_state",
                    "label": "Random state",
                    "description": "Optional random seed for reproducible sampling.",
                    "type": "number",
                    "default": UNDERSAMPLING_DEFAULT_RANDOM_STATE,
                    "step": 1.0,
                },
                {
                    "name": "replacement",
                    "label": "Sample with replacement",
                    "description": "Allow sampling with replacement when reducing the majority class.",
                    "type": "boolean",
                    "default": UNDERSAMPLING_DEFAULT_REPLACEMENT,
                },
            ],
            "default_config": {
                "method": UNDERSAMPLING_DEFAULT_METHOD,
                "target_column": "",
                "sampling_strategy": None,
                "random_state": UNDERSAMPLING_DEFAULT_RANDOM_STATE,
                "replacement": UNDERSAMPLING_DEFAULT_REPLACEMENT,
            },
        }

        oversampling_node: FeatureNodeCatalogEntry = {
            "type": "class_oversampling",
            "label": "Class oversampling",
            "description": (
                "Boost minority-class representation with synthetic "
                "over-sampling techniques."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "RESAMPLING DATASET",
            "tags": ["oversampling", "class_balance", "imbalanced"],
            "parameters": [
                {
                    "name": "method",
                    "label": "Resampling method",
                    "description": (
                        "Choose the synthetic sampling strategy for balancing classes."
                    ),
                    "type": "select",
                    "default": OVERSAMPLING_DEFAULT_METHOD,
                    "options": oversampling_method_options,
                },
                {
                    "name": "target_column",
                    "label": "Target column",
                    "description": "Categorical target column used to guide sampling.",
                    "type": "text",
                    "placeholder": "target",
                },
                {
                    "name": "sampling_strategy",
                    "label": "Sampling ratio",
                    "description": "Minority-to-majority ratio (> 0). Leave blank for auto.",
                    "type": "number",
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                },
                {
                    "name": "k_neighbors",
                    "label": "K-neighbors",
                    "description": (
                        "Nearest neighbors considered when synthesising new "
                        "minority samples."
                    ),
                    "type": "number",
                    "default": OVERSAMPLING_DEFAULT_K_NEIGHBORS,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 1.0,
                },
                {
                    "name": "random_state",
                    "label": "Random state",
                    "description": "Optional random seed for reproducible sampling.",
                    "type": "number",
                    "default": OVERSAMPLING_DEFAULT_RANDOM_STATE,
                    "step": 1.0,
                },
            ],
            "default_config": {
                "method": OVERSAMPLING_DEFAULT_METHOD,
                "target_column": "",
                "sampling_strategy": None,
                "k_neighbors": OVERSAMPLING_DEFAULT_K_NEIGHBORS,
                "random_state": OVERSAMPLING_DEFAULT_RANDOM_STATE,
                "replacement": OVERSAMPLING_DEFAULT_REPLACEMENT,
            },
        }

        label_encoding_node: FeatureNodeCatalogEntry = {
            "type": "label_encoding",
            "label": "Label encode categories",
            "description": "Convert categorical columns into integer codes for model-ready features.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "Categorical Encoding",
            "tags": ["categorical", "encoding", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to encode",
                    "description": "Choose categorical columns or leave blank to rely on auto-detection.",
                    "type": "multi_select",
                    "source": {
                        "type": "label_encoding_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/label-encoding",
                        "value_key": "columns",
                    },
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect categorical columns",
                    "description": "Automatically include text columns that stay within the cardinality threshold.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "max_unique_values",
                    "label": "Max categories for auto-detect",
                    "description": "Upper bound on unique values when auto-detect is enabled (0 disables the cap).",
                    "type": "number",
                    "default": LABEL_ENCODING_DEFAULT_MAX_UNIQUE,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 1.0,
                },
                {
                    "name": "output_suffix",
                    "label": "Encoded suffix",
                    "description": "Suffix added to new encoded columns when originals are retained.",
                    "type": "text",
                    "default": LABEL_ENCODING_DEFAULT_SUFFIX,
                },
                {
                    "name": "drop_original",
                    "label": "Replace original column",
                    "description": "Overwrite the source column instead of creating a suffixed copy.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "missing_strategy",
                    "label": "Missing value strategy",
                    "description": "Choose whether to keep missing values as <NA> or assign a dedicated code.",
                    "type": "select",
                    "default": "keep_na",
                    "options": [
                        {"value": "keep_na", "label": "Keep as <NA>"},
                        {"value": "encode", "label": "Assign code"},
                    ],
                },
                {
                    "name": "missing_code",
                    "label": "Missing value code",
                    "description": "Integer code applied to missing values when the assign-code strategy is selected.",
                    "type": "number",
                    "default": -1,
                    "step": 1.0,
                },
            ],
            "default_config": {
                "columns": [],
                "auto_detect": False,
                "max_unique_values": LABEL_ENCODING_DEFAULT_MAX_UNIQUE,
                "output_suffix": LABEL_ENCODING_DEFAULT_SUFFIX,
                "drop_original": False,
                "missing_strategy": "keep_na",
                "missing_code": -1,
                "skipped_columns": [],
            },
        }

        target_encoding_node: FeatureNodeCatalogEntry = {
            "type": "target_encoding",
            "label": "Target encode categories",
            "description": (
                "Replace categorical values with smoothed averages of a "
                "numeric target column."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "Categorical Encoding",
            "tags": ["categorical", "encoding", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to encode",
                    "description": (
                        "Select categorical columns or rely on auto-detection "
                        "when safe."
                    ),
                    "type": "multi_select",
                    "source": {
                        "type": "target_encoding_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/target-encoding",
                        "value_key": "columns",
                    },
                },
                {
                    "name": "target_column",
                    "label": "Target column",
                    "description": (
                        "Numeric target column used to compute per-category "
                        "averages (e.g., regression value or 0/1 outcome)."
                    ),
                    "type": "text",
                    "placeholder": "target",
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect categorical columns",
                    "description": (
                        "Automatically include text columns that stay within "
                        "the category limit."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "max_categories",
                    "label": "Max categories",
                    "description": (
                        "Upper bound on unique values when auto-detect is "
                        "enabled (0 disables the cap)."
                    ),
                    "type": "number",
                    "default": TARGET_ENCODING_DEFAULT_MAX_CATEGORIES,
                    "min": 0.0,
                    "max": float(TARGET_ENCODING_MAX_CARDINALITY_LIMIT),
                    "step": 1.0,
                },
                {
                    "name": "output_suffix",
                    "label": "Encoded suffix",
                    "description": (
                        "Suffix added to new encoded columns when originals "
                        "are retained."
                    ),
                    "type": "text",
                    "default": TARGET_ENCODING_DEFAULT_SUFFIX,
                },
                {
                    "name": "drop_original",
                    "label": "Replace original column",
                    "description": (
                        "Overwrite the source column instead of creating a "
                        "suffixed copy."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "smoothing",
                    "label": "Smoothing strength",
                    "description": (
                        "Higher values pull category means toward the global "
                        "average to reduce overfitting."
                    ),
                    "type": "number",
                    "default": TARGET_ENCODING_DEFAULT_SMOOTHING,
                    "min": 0.0,
                    "step": 1.0,
                },
                {
                    "name": "encode_missing",
                    "label": "Encode missing values",
                    "description": (
                        "Assign the global mean to rows where the categorical "
                        "value is missing."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "handle_unknown",
                    "label": "Handle unseen categories",
                    "description": (
                        "Choose whether to assign the global mean or raise an "
                        "error for categories not observed during fitting."
                    ),
                    "type": "select",
                    "default": TARGET_ENCODING_DEFAULT_HANDLE_UNKNOWN,
                    "options": [
                        {"value": "global_mean", "label": "Use global mean"},
                        {"value": "error", "label": "Raise error"},
                    ],
                },
            ],
            "default_config": {
                "columns": [],
                "target_column": "",
                "auto_detect": False,
                "max_categories": TARGET_ENCODING_DEFAULT_MAX_CATEGORIES,
                "output_suffix": TARGET_ENCODING_DEFAULT_SUFFIX,
                "drop_original": False,
                "smoothing": TARGET_ENCODING_DEFAULT_SMOOTHING,
                "encode_missing": False,
                "handle_unknown": TARGET_ENCODING_DEFAULT_HANDLE_UNKNOWN,
                "skipped_columns": [],
            },
        }

        hash_encoding_node: FeatureNodeCatalogEntry = {
            "type": "hash_encoding",
            "label": "Hash encode categories",
            "description": (
                "Project categorical values into deterministic hash buckets "
                "to bound feature width."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "Categorical Encoding",
            "tags": ["categorical", "encoding", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to encode",
                    "description": (
                        "Select categorical columns or rely on auto-detection "
                        "to handle high-cardinality features."
                    ),
                    "type": "multi_select",
                    "source": {
                        "type": "hash_encoding_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/hash-encoding",
                        "value_key": "columns",
                    },
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect categorical columns",
                    "description": (
                        "Automatically include text columns that stay within "
                        "the category limit."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "max_categories",
                    "label": "Max categories",
                    "description": (
                        "Upper bound on unique values when auto-detect is "
                        "enabled (0 disables the cap)."
                    ),
                    "type": "number",
                    "default": HASH_ENCODING_DEFAULT_MAX_CATEGORIES,
                    "min": 0.0,
                    "max": float(HASH_ENCODING_MAX_CARDINALITY_LIMIT),
                    "step": 1.0,
                },
                {
                    "name": "n_buckets",
                    "label": "Hash buckets",
                    "description": (
                        "Number of hash buckets used to encode each column "
                        "(higher values reduce collisions)."
                    ),
                    "type": "number",
                    "default": HASH_ENCODING_DEFAULT_BUCKETS,
                    "min": float(2),
                    "max": float(HASH_ENCODING_MAX_BUCKETS),
                    "step": 1.0,
                },
                {
                    "name": "output_suffix",
                    "label": "Encoded suffix",
                    "description": (
                        "Suffix added to new hashed columns when originals "
                        "are retained."
                    ),
                    "type": "text",
                    "default": HASH_ENCODING_DEFAULT_SUFFIX,
                },
                {
                    "name": "drop_original",
                    "label": "Replace original column",
                    "description": (
                        "Overwrite the source column instead of creating a "
                        "suffixed copy."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "encode_missing",
                    "label": "Encode missing values",
                    "description": (
                        "Assign a dedicated bucket to missing values instead "
                        "of leaving them null."
                    ),
                    "type": "boolean",
                    "default": False,
                },
            ],
            "default_config": {
                "columns": [],
                "auto_detect": False,
                "max_categories": HASH_ENCODING_DEFAULT_MAX_CATEGORIES,
                "n_buckets": HASH_ENCODING_DEFAULT_BUCKETS,
                "output_suffix": HASH_ENCODING_DEFAULT_SUFFIX,
                "drop_original": False,
                "encode_missing": False,
                "skipped_columns": [],
            },
        }

        model_registry = list_registered_models()
        model_type_options: List[FeatureNodeParameterOption] = []
        for spec_key, spec in model_registry.items():
            label = spec_key.replace("_", " ").title()
            description = f"{spec.problem_type.title()} baseline"
            model_type_options.append(
                _make_option(
                    spec_key,
                    label,
                    description=description,
                    metadata={
                        "problem_type": spec.problem_type,
                        "default_params": dict(spec.default_params),
                    },
                )
            )

        model_type_options.sort(key=lambda option: option["label"])
        if not model_type_options:
            model_type_options.append(_make_option("logistic_regression", "Logistic Regression"))

        preferred_default_model_type = "logistic_regression"
        default_model_type = preferred_default_model_type
        preferred_exists = any(
            option["value"] == preferred_default_model_type for option in model_type_options
        )
        if not preferred_exists:
            default_model_type = model_type_options[0]["value"]

        default_model_spec = model_registry.get(default_model_type)
        default_hyperparameters_text = ""
        if default_model_spec and default_model_spec.default_params:
            try:
                default_hyperparameters_text = json.dumps(
                    default_model_spec.default_params,
                    indent=2,
                    sort_keys=True,
                )
            except TypeError:
                default_hyperparameters_text = json.dumps(default_model_spec.default_params, indent=2)

        feature_target_split_node: FeatureNodeCatalogEntry = {
            "type": "feature_target_split",
            "label": "Separate features & target",
            "description": (
                "Designate the supervised target column. All remaining "
                "columns are treated as features (X)."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "MODELING",
            "tags": ["modeling", "features", "target"],
            "parameters": [
                {
                    "name": "target_column",
                    "label": "Target column",
                    "description": (
                        "Column treated as the target (y) for downstream "
                        "modeling."
                    ),
                    "type": "text",
                    "placeholder": "target",
                },
            ],
            "default_config": {
                "target_column": "",
            },
        }

        train_test_split_node: FeatureNodeCatalogEntry = {
            "type": "train_test_split",
            "label": "Train/Test Split",
            "description": (
                "Split dataset into training, testing, and optionally "
                "validation sets. Supports stratification."
            ),
            "inputs": ["dataset"],
            "outputs": ["train", "test", "validation"],
            "category": "MODELING",
            "tags": ["modeling", "split", "train_test"],
            "parameters": [
                {
                    "name": "test_size",
                    "label": "Test size",
                    "description": (
                        "Proportion of the dataset to include in the test "
                        "split."
                    ),
                    "type": "number",
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                },
                {
                    "name": "validation_size",
                    "label": "Validation size",
                    "description": (
                        "Proportion of the dataset to include in the "
                        "validation split (optional)."
                    ),
                    "type": "number",
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                },
                {
                    "name": "random_state",
                    "label": "Random state",
                    "description": "Seed for reproducible splits. Leave empty for random.",
                    "type": "number",
                    "default": 42,
                    "min": 0,
                    "step": 1,
                },
                {
                    "name": "shuffle",
                    "label": "Shuffle",
                    "description": "Whether to shuffle the data before splitting.",
                    "type": "boolean",
                    "default": True,
                },
                {
                    "name": "stratify",
                    "label": "Stratify",
                    "description": "Preserve class distribution in splits using the target column.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "target_column",
                    "label": "Target column (for stratification)",
                    "description": "Column to use for stratified splitting (required if stratify is enabled).",
                    "type": "text",
                    "placeholder": "target",
                },
            ],
            "default_config": {
                "test_size": 0.2,
                "validation_size": 0.0,
                "random_state": 42,
                "shuffle": True,
                "stratify": False,
                "target_column": "",
            },
        }

        problem_type_options: List[FeatureNodeParameterOption] = [
            _make_option("classification", "Classification"),
            _make_option("regression", "Regression"),
        ]

        train_model_draft_node: FeatureNodeCatalogEntry = {
            "type": "train_model_draft",
            "label": "Train model",
            "description": (
                "Validate the pipeline output, launch background training "
                "jobs, and expose trained models downstream."
            ),
            "inputs": ["dataset"],
            "outputs": ["models"],
            "category": "MODELING",
            "tags": ["modeling", "validation", "preview"],
            "parameters": [
                {
                    "name": "target_column",
                    "label": "Target column",
                    "description": (
                        "Name of the response column to model (leave blank "
                        "to map via downstream configuration)."
                    ),
                    "type": "text",
                    "placeholder": "target",
                },
                {
                    "name": "problem_type",
                    "label": "Problem type",
                    "description": (
                        "Choose the expected modeling task for downstream "
                        "training."
                    ),
                    "type": "select",
                    "default": "classification",
                    "options": problem_type_options,
                },
                {
                    "name": "model_type",
                    "label": "Model template",
                    "description": (
                        "Select the registered estimator used when launching "
                        "background jobs."
                    ),
                    "type": "select",
                    "default": default_model_type,
                    "options": model_type_options,
                },
                {
                    "name": "hyperparameters",
                    "label": "Hyperparameters (JSON)",
                    "description": (
                        "Optional JSON object merged with the default "
                        "parameters for the selected model."
                    ),
                    "type": "textarea",
                    "placeholder": "{\n  \"n_estimators\": 200\n}",
                },
                {
                    "name": "cv_enabled",
                    "label": "Enable cross-validation",
                    "description": (
                        "Run k-fold cross-validation on the training split "
                        "before finalising the model."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "cv_strategy",
                    "label": "Cross-validation strategy",
                    "description": (
                        "Choose how folds are generated when cross-validation "
                        "is enabled."
                    ),
                    "type": "select",
                    "default": "auto",
                    "options": [
                        {"value": "auto", "label": "Auto"},
                        {"value": "kfold", "label": "K-Fold"},
                        {"value": "stratified_kfold", "label": "Stratified K-Fold"},
                    ],
                },
                {
                    "name": "cv_folds",
                    "label": "Number of folds",
                    "description": (
                        "How many folds to use when cross-validation is "
                        "enabled (minimum 2)."
                    ),
                    "type": "number",
                    "default": 5,
                    "min": 2.0,
                    "max": 20.0,
                    "step": 1.0,
                },
                {
                    "name": "cv_shuffle",
                    "label": "Shuffle before splitting",
                    "description": (
                        "Shuffle the training rows before generating "
                        "cross-validation folds."
                    ),
                    "type": "boolean",
                    "default": True,
                },
                {
                    "name": "cv_random_state",
                    "label": "Shuffle random state",
                    "description": (
                        "Optional random seed applied when shuffling folds "
                        "(leave blank for random)."
                    ),
                    "type": "number",
                    "default": 42,
                    "min": 0.0,
                    "step": 1.0,
                },
                {
                    "name": "cv_refit_strategy",
                    "label": "Refit using",
                    "description": (
                        "Choose whether the final model should be refit on "
                        "the training split only or on training+validation "
                        "after cross-validation."
                    ),
                    "type": "select",
                    "default": "train_plus_validation",
                    "options": [
                        {"value": "train_only", "label": "Train split only"},
                        {"value": "train_plus_validation", "label": "Train + validation"},
                    ],
                },
            ],
            "default_config": {
                "target_column": "",
                "problem_type": "classification",
                "model_type": default_model_type,
                "hyperparameters": default_hyperparameters_text,
                "cv_enabled": False,
                "cv_strategy": "auto",
                "cv_folds": 5,
                "cv_shuffle": True,
                "cv_random_state": 42,
                "cv_refit_strategy": "train_plus_validation",
            },
        }

        model_evaluation_node: FeatureNodeCatalogEntry = {
            "type": "model_evaluation",
            "label": "Model evaluation",
            "description": (
                "Review confusion matrices, ROC/PR curves, and residual "
                "diagnostics for trained models without leaving the canvas."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "MODELING",
            "tags": ["modeling", "evaluation", "diagnostics"],
            "parameters": [],
            "default_config": {
                "training_job_id": "",
                "splits": ["test"],
                "include_curves": True,
                "include_confusion": True,
                "include_residuals": True,
                "last_evaluated_at": None,
            },
        }

        model_registry_node: FeatureNodeCatalogEntry = {
            "type": "model_registry_overview",
            "label": "Model registry",
            "description": (
                "Review trained model versions, metrics, and artifacts from "
                "the connected training node."
            ),
            "inputs": ["models"],
            "outputs": ["models"],
            "category": "MODELING",
            "tags": ["modeling", "registry", "metrics"],
            "parameters": [
                {
                    "name": "default_problem_type",
                    "label": "Default problem tab",
                    "description": (
                        "Problem type tab that opens by default when viewing "
                        "the registry."
                    ),
                    "type": "select",
                    "default": "auto",
                    "options": [
                        {"value": "auto", "label": "Auto (latest model)"},
                        {"value": "classification", "label": "Classification"},
                        {"value": "regression", "label": "Regression"},
                    ],
                },
                {
                    "name": "default_method",
                    "label": "Default model method",
                    "description": (
                        "Optional model template to spotlight initially (e.g. "
                        "logistic_regression)."
                    ),
                    "type": "text",
                    "placeholder": "logistic_regression",
                },
                {
                    "name": "show_non_success",
                    "label": "Show non-successful runs",
                    "description": (
                        "Include queued, running, failed, and cancelled "
                        "versions in comparisons."
                    ),
                    "type": "boolean",
                    "default": True,
                },
            ],
            "default_config": {
                "default_problem_type": "auto",
                "default_method": "",
                "show_non_success": True,
            },
        }

        tuning_strategy_choices = [
            _make_option(
                choice.get("value", ""),
                choice.get("label", ""),
                description=choice.get("description"),
            )
            for choice in get_strategy_choices_for_ui()
        ]
        tuning_strategy_default = get_default_strategy_value()

        tuning_default_search_space = {
            "C": [0.1, 1.0, 10.0],
            "solver": ["lbfgs", "saga"],
        }
        default_search_space_text = json.dumps(tuning_default_search_space, indent=2)

        hyperparameter_tuning_node: FeatureNodeCatalogEntry = {
            "type": "hyperparameter_tuning",
            "label": "Hyperparameter tuning",
            "description": (
                "Search over candidate hyperparameter combinations using "
                "cross-validation and background workers."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "MODELING",
            "tags": ["modeling", "tuning", "optimization"],
            "parameters": [
                {
                    "name": "target_column",
                    "label": "Target column",
                    "description": (
                        "Name of the response column to optimize against."
                    ),
                    "type": "text",
                    "placeholder": "target",
                },
                {
                    "name": "problem_type",
                    "label": "Problem type",
                    "description": (
                        "Select the expected modeling task so metrics align."
                    ),
                    "type": "select",
                    "default": "classification",
                    "options": problem_type_options,
                },
                {
                    "name": "model_type",
                    "label": "Model template",
                    "description": (
                        "Choose which registered estimator to tune."
                    ),
                    "type": "select",
                    "default": default_model_type,
                    "options": model_type_options,
                },
                {
                    "name": "baseline_hyperparameters",
                    "label": "Baseline hyperparameters (JSON)",
                    "description": (
                        "Optional JSON object merged into the estimator "
                        "before tuning."
                    ),
                    "type": "textarea",
                    "placeholder": "{\n  \"max_iter\": 1000\n}",
                },
                {
                    "name": "search_strategy",
                    "label": "Search strategy",
                    "description": (
                        "Choose how to explore the search space. Available "
                        "strategies are configured in application settings."
                    ),
                    "type": "select",
                    "default": tuning_strategy_default,
                    "options": tuning_strategy_choices,
                },
                {
                    "name": "search_space",
                    "label": "Search space (JSON)",
                    "description": (
                        "JSON object mapping hyperparameter names to lists "
                        "of candidate values."
                    ),
                    "type": "textarea",
                    "placeholder": "{\n  \"C\": [0.1, 1.0, 10.0],\n  \"solver\": [\"lbfgs\", \"saga\"]\n}",
                },
                {
                    "name": "search_iterations",
                    "label": "Max iterations",
                    "description": (
                        "Maximum sampled combinations when random or Optuna "
                        "search is enabled (ignored for grid and halving)."
                    ),
                    "type": "number",
                    "default": 20,
                    "min": 1.0,
                    "max": 500.0,
                    "step": 1.0,
                },
                {
                    "name": "search_random_state",
                    "label": "Random state",
                    "description": (
                        "Optional seed controlling candidate sampling order "
                        "for random/Optuna search."
                    ),
                    "type": "number",
                    "default": 42,
                    "min": 0.0,
                    "step": 1.0,
                },
                {
                    "name": "scoring_metric",
                    "label": "Scoring metric",
                    "description": (
                        "Optional sklearn-compatible scoring string (leave "
                        "blank for model default)."
                    ),
                    "type": "text",
                    "placeholder": "accuracy",
                },
                {
                    "name": "cv_enabled",
                    "label": "Enable cross-validation",
                    "description": (
                        "Toggle K-fold cross-validation for evaluating each "
                        "candidate."
                    ),
                    "type": "boolean",
                    "default": True,
                },
                {
                    "name": "cv_strategy",
                    "label": "Cross-validation strategy",
                    "description": (
                        "Choose how folds are generated when tuning."
                    ),
                    "type": "select",
                    "default": "auto",
                    "options": [
                        {"value": "auto", "label": "Auto"},
                        {"value": "kfold", "label": "K-Fold"},
                        {"value": "stratified_kfold", "label": "Stratified K-Fold"},
                    ],
                },
                {
                    "name": "cv_folds",
                    "label": "Number of folds",
                    "description": (
                        "How many folds to use when tuning (minimum 2)."
                    ),
                    "type": "number",
                    "default": 5,
                    "min": 2.0,
                    "max": 20.0,
                    "step": 1.0,
                },
                {
                    "name": "cv_shuffle",
                    "label": "Shuffle before splitting",
                    "description": (
                        "Shuffle rows before generating folds."
                    ),
                    "type": "boolean",
                    "default": True,
                },
                {
                    "name": "cv_random_state",
                    "label": "Shuffle random state",
                    "description": (
                        "Optional random seed applied when shuffling folds."
                    ),
                    "type": "number",
                    "default": 42,
                    "min": 0.0,
                    "step": 1.0,
                },
            ],
            "default_config": {
                "target_column": "",
                "problem_type": "classification",
                "model_type": default_model_type,
                "baseline_hyperparameters": default_hyperparameters_text,
                "search_strategy": tuning_strategy_default,
                "search_space": default_search_space_text,
                "search_iterations": 20,
                "search_random_state": 42,
                "scoring_metric": "",
                "cv_enabled": True,
                "cv_strategy": "auto",
                "cv_folds": 5,
                "cv_shuffle": True,
                "cv_random_state": 42,
            },
        }

        ordinal_encoding_node: FeatureNodeCatalogEntry = {
            "type": "ordinal_encoding",
            "label": "Ordinal encode categories",
            "description": (
                "Map categorical columns to ordered codes with optional "
                "unknown fallbacks."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "Categorical Encoding",
            "tags": ["categorical", "encoding", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to encode",
                    "description": (
                        "Select categorical columns or rely on auto-detection "
                        "when safe."
                    ),
                    "type": "multi_select",
                    "source": {
                        "type": "ordinal_encoding_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/ordinal-encoding",
                        "value_key": "columns",
                    },
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect categorical columns",
                    "description": "Automatically include text columns that stay within the category limit.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "max_categories",
                    "label": "Max categories",
                    "description": (
                        "Upper bound on unique values when auto-detect is "
                        "enabled (0 disables the cap)."
                    ),
                    "type": "number",
                    "default": ORDINAL_ENCODING_DEFAULT_MAX_CATEGORIES,
                    "min": 0.0,
                    "max": float(ORDINAL_ENCODING_MAX_CARDINALITY_LIMIT),
                    "step": 1.0,
                },
                {
                    "name": "output_suffix",
                    "label": "Encoded suffix",
                    "description": (
                        "Suffix added to new encoded columns when originals "
                        "are retained."
                    ),
                    "type": "text",
                    "default": ORDINAL_ENCODING_DEFAULT_SUFFIX,
                },
                {
                    "name": "drop_original",
                    "label": "Replace original column",
                    "description": (
                        "Overwrite the source column instead of creating a "
                        "suffixed copy."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "encode_missing",
                    "label": "Encode missing values",
                    "description": (
                        "Assign the fallback code to missing values instead "
                        "of leaving them null."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "handle_unknown",
                    "label": "Handle unseen categories",
                    "description": (
                        "Choose whether to assign a fallback code or raise an "
                        "error for unknown categories."
                    ),
                    "type": "select",
                    "default": "use_encoded_value",
                    "options": [
                        {"value": "use_encoded_value", "label": "Assign fallback code"},
                        {"value": "error", "label": "Raise error"},
                    ],
                },
                {
                    "name": "unknown_value",
                    "label": "Fallback code",
                    "description": (
                        "Integer code applied to unseen categories or missing "
                        "values when enabled."
                    ),
                    "type": "number",
                    "default": ORDINAL_ENCODING_DEFAULT_UNKNOWN_VALUE,
                    "step": 1.0,
                },
            ],
            "default_config": {
                "columns": [],
                "auto_detect": False,
                "max_categories": ORDINAL_ENCODING_DEFAULT_MAX_CATEGORIES,
                "output_suffix": ORDINAL_ENCODING_DEFAULT_SUFFIX,
                "drop_original": False,
                "encode_missing": False,
                "handle_unknown": "use_encoded_value",
                "unknown_value": ORDINAL_ENCODING_DEFAULT_UNKNOWN_VALUE,
                "skipped_columns": [],
            },
        }

        dummy_encoding_node: FeatureNodeCatalogEntry = {
            "type": "dummy_encoding",
            "label": "Dummy encode categories",
            "description": (
                "Expand categorical columns while dropping a reference level "
                "per feature."
            ),
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "Categorical Encoding",
            "tags": ["categorical", "encoding", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to encode",
                    "description": (
                        "Select categorical columns or rely on auto-detection "
                        "to capture safe candidates."
                    ),
                    "type": "multi_select",
                    "source": {
                        "type": "dummy_encoding_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/dummy-encoding",
                        "value_key": "columns",
                    },
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect categorical columns",
                    "description": (
                        "Automatically include text columns that keep dummy "
                        "width manageable."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "max_categories",
                    "label": "Max categories",
                    "description": (
                        "Upper bound on unique values when auto-detect is "
                        "enabled (0 disables the cap)."
                    ),
                    "type": "number",
                    "default": DUMMY_ENCODING_DEFAULT_MAX_CATEGORIES,
                    "min": 0.0,
                    "max": float(DUMMY_ENCODING_MAX_CARDINALITY_LIMIT),
                    "step": 1.0,
                },
                {
                    "name": "drop_first",
                    "label": "Drop reference level",
                    "description": (
                        "Always drop the first dummy column per feature to "
                        "mitigate multicollinearity."
                    ),
                    "type": "boolean",
                    "default": True,
                },
                {
                    "name": "include_missing",
                    "label": "Encode missing values",
                    "description": (
                        "Include a dedicated dummy column for missing values."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "drop_original",
                    "label": "Replace original column",
                    "description": (
                        "Remove the source column after dummy expansion."
                    ),
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "prefix_separator",
                    "label": "Dummy prefix separator",
                    "description": (
                        "Separator between the column name and category when "
                        "naming dummy columns."
                    ),
                    "type": "text",
                    "default": DUMMY_ENCODING_DEFAULT_PREFIX_SEPARATOR,
                },
            ],
            "default_config": {
                "columns": [],
                "auto_detect": False,
                "max_categories": DUMMY_ENCODING_DEFAULT_MAX_CATEGORIES,
                "drop_first": True,
                "include_missing": False,
                "drop_original": False,
                "prefix_separator": DUMMY_ENCODING_DEFAULT_PREFIX_SEPARATOR,
                "skipped_columns": [],
            },
        }

        one_hot_encoding_node: FeatureNodeCatalogEntry = {
            "type": "one_hot_encoding",
            "label": "One-hot encode categories",
            "description": "Expand categorical columns into binary indicator features.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "Categorical Encoding",
            "tags": ["categorical", "encoding", "feature_engineering"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to encode",
                    "description": "Choose categorical columns or rely on auto-detection to keep configs light.",
                    "type": "multi_select",
                    "source": {
                        "type": "one_hot_encoding_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/one-hot-encoding",
                        "value_key": "columns",
                    },
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect categorical columns",
                    "description": "Automatically include text columns when their dummy expansion stays manageable.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "max_categories",
                    "label": "Max categories",
                    "description": "Upper bound on unique values when auto-detect is enabled (0 disables the cap).",
                    "type": "number",
                    "default": ONE_HOT_ENCODING_DEFAULT_MAX_CATEGORIES,
                    "min": 0.0,
                    "max": float(ONE_HOT_ENCODING_MAX_CARDINALITY_LIMIT),
                    "step": 1.0,
                },
                {
                    "name": "drop_first",
                    "label": "Drop first dummy",
                    "description": "Avoid multicollinearity by dropping the first dummy column per feature.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "include_missing",
                    "label": "Encode missing values",
                    "description": "Include a dedicated dummy column for missing values.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "drop_original",
                    "label": "Replace original column",
                    "description": "Remove the source column after dummy expansion.",
                    "type": "boolean",
                    "default": False,
                },
                {
                    "name": "prefix_separator",
                    "label": "Dummy prefix separator",
                    "description": "Separator between the column name and category when naming dummy columns.",
                    "type": "text",
                    "default": ONE_HOT_ENCODING_DEFAULT_PREFIX_SEPARATOR,
                },
            ],
            "default_config": {
                "columns": [],
                "auto_detect": False,
                "max_categories": ONE_HOT_ENCODING_DEFAULT_MAX_CATEGORIES,
                "drop_first": False,
                "include_missing": False,
                "drop_original": False,
                "prefix_separator": ONE_HOT_ENCODING_DEFAULT_PREFIX_SEPARATOR,
                "skipped_columns": [],
            },
        }

        scaling_node: FeatureNodeCatalogEntry = {
            "type": "scale_numeric_features",
            "label": "Scale numeric features",
            "description": "Standardize or normalize numeric columns with recommended scalers.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "FEATURE ENGINEERING",
            "tags": ["numeric", "scaling", "normalization"],
            "parameters": [
                {
                    "name": "columns",
                    "label": "Columns to scale",
                    "description": "Select numeric columns manually or leave blank to rely on auto-detection.",
                    "type": "multi_select",
                },
                {
                    "name": "default_method",
                    "label": "Default scaling method",
                    "description": "Fallback scaler applied whenever a column override is not configured.",
                    "type": "text",
                    "default": SCALING_DEFAULT_METHOD,
                },
                {
                    "name": "auto_detect",
                    "label": "Auto-detect numeric columns",
                    "description": "Automatically include continuous numeric columns that pass quality checks.",
                    "type": "boolean",
                    "default": True,
                },
            ],
            "default_config": {
                "columns": [],
                "default_method": SCALING_DEFAULT_METHOD,
                "column_methods": {},
                "auto_detect": True,
                "skipped_columns": [],
            },
        }

        skewness_transform_node: FeatureNodeCatalogEntry = {
            "type": "skewness_transform",
            "label": "Fix skewed columns",
            "description": "Surface skewness diagnostics and apply variance-stabilizing transforms.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "FEATURE ENGINEERING",
            "tags": ["skewness", "transformation", "numeric"],
            "parameters": [
                {
                    "name": "transformations",
                    "label": "Column transformations",
                    "description": "Pick recommended transforms per column to reduce skewness.",
                    "type": "text",
                    "source": {
                        "type": "skewness_recommendations",
                        "endpoint": "/ml-workflow/api/recommendations/skewness",
                        "value_key": "columns",
                    },
                }
            ],
            "default_config": {
                "transformations": [],
            },
        }

        binned_distribution_node: FeatureNodeCatalogEntry = {
            "type": "binned_distribution",
            "label": "Binned column distributions",
            "description": "Visualize category counts for columns generated by binning.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "INSPECTION",
            "tags": ["binning", "visualization"],
            "parameters": [],
            "default_config": {},
        }

        skewness_distribution_node: FeatureNodeCatalogEntry = {
            "type": "skewness_distribution",
            "label": "Skewness distributions",
            "description": "Visualize distributions for skewed numeric columns.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "INSPECTION",
            "tags": ["skewness", "visualization"],
            "parameters": [],
            "default_config": {},
        }

        data_preview_node: FeatureNodeCatalogEntry = {
            "type": "data_preview",
            "label": "Data snapshot",
            "description": "Inspect a sample of the dataset after upstream transforms.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "INSPECTION",
            "tags": ["preview", "validation"],
            "parameters": [],
            "default_config": {},
        }

        transformer_audit_node: FeatureNodeCatalogEntry = {
            "type": "transformer_audit",
            "label": "Transformer audit",
            "description": "Review transformer fit/transform activity across train, test, and validation splits.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "INSPECTION",
            "tags": ["monitoring", "transformers", "splits"],
            "parameters": [],
            "default_config": {},
        }

        dataset_profile_node: FeatureNodeCatalogEntry = {
            "type": "dataset_profile",
            "label": "Dataset profile",
            "description": "Generate a lightweight dataset profile with summary statistics.",
            "inputs": ["dataset"],
            "outputs": ["dataset"],
            "category": "INSPECTION",
            "tags": ["profiling", "eda", "quality"],
            "parameters": [],
            "default_config": {},
        }

        return [
            drop_missing_node,
            drop_missing_rows_node,
            missing_indicator_node,
            remove_duplicates_node,
            cast_column_types_node,
            imputation_methods_node,
            trim_whitespace_node,
            normalize_text_case_node,
            replace_aliases_node,
            standardize_dates_node,
            remove_special_chars_node,
            replace_invalid_values_node,
            regex_replace_node,
            binning_node,
            polynomial_features_node,
            feature_selection_node,
            feature_math_node,
            undersampling_node,
            oversampling_node,
            label_encoding_node,
            target_encoding_node,
            hash_encoding_node,
            feature_target_split_node,
            train_test_split_node,
            train_model_draft_node,
            model_evaluation_node,
            model_registry_node,
            hyperparameter_tuning_node,
            ordinal_encoding_node,
            dummy_encoding_node,
            one_hot_encoding_node,
            outlier_removal_node,
            scaling_node,
            skewness_transform_node,
            binned_distribution_node,
            skewness_distribution_node,
            data_preview_node,
            transformer_audit_node,
            dataset_profile_node,
        ]

    return _build_preprocessing_nodes()


@dataclass
class DropColumnCandidateEntry:
    name: str
    missing_percentage: Optional[float] = None
    priority: Optional[str] = None
    reasons: Set[str] = field(default_factory=set)
    signals: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)


@dataclass
class DropColumnRecommendationBuilder:
    candidates: Dict[str, DropColumnCandidateEntry] = field(default_factory=dict)
    column_missing_map: Dict[str, float] = field(default_factory=dict)
    missing_values: List[float] = field(default_factory=list)
    all_columns: Set[str] = field(default_factory=set)

    def _normalize_name(self, column: Any) -> str:
        if column is None:
            return ""
        return str(column).strip()

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    def record_missing_percentage(
        self,
        column: Any,
        value: Any,
        *,
        include_in_stats: bool = False,
    ) -> Optional[float]:
        name = self._normalize_name(column)
        if not name:
            return None

        numeric = self._safe_float(value)
        if numeric is None or numeric < 0:
            return None

        current = self.column_missing_map.get(name, 0.0)
        self.column_missing_map[name] = max(current, numeric)
        if include_in_stats:
            self.missing_values.append(numeric)

        self.all_columns.add(name)
        return numeric

    def register_candidate(
        self,
        column: Any,
        *,
        reason: str,
        priority: Optional[str] = None,
        missing_pct: Optional[float] = None,
        signals: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        name = self._normalize_name(column)
        if not name:
            return

        entry = self.candidates.setdefault(name, DropColumnCandidateEntry(name=name))

        normalized_reason = str(reason).strip()
        if normalized_reason:
            entry.reasons.add(normalized_reason)

        normalized_priority = str(priority).strip() if priority else None
        if normalized_priority and not entry.priority:
            entry.priority = normalized_priority

        if signals:
            entry.signals.update(
                {
                    signal
                    for signal in (str(item).strip() for item in signals)
                    if signal
                }
            )

        if tags:
            entry.tags.update(
                {
                    tag
                    for tag in (str(item).strip() for item in tags)
                    if tag
                }
            )

        if missing_pct is not None:
            numeric_pct = self.record_missing_percentage(name, missing_pct)
            if numeric_pct is not None:
                if entry.missing_percentage is None or numeric_pct > entry.missing_percentage:
                    entry.missing_percentage = numeric_pct

        self.all_columns.add(name)

    def ingest_missing_summary(self, missing_summary: Iterable[Any]) -> None:
        for record in missing_summary:
            column_name = record.get("column") if isinstance(record, dict) else None
            missing_pct = record.get("missing_percentage") if isinstance(record, dict) else None

            numeric_missing = self.record_missing_percentage(column_name, missing_pct, include_in_stats=True)
            if numeric_missing is None:
                continue

            if numeric_missing >= 30.0:
                priority = "critical" if numeric_missing >= 85 else "high" if numeric_missing >= 60 else "medium"
                signal_labels = ["missing_data"]
                if numeric_missing >= 99.5:
                    signal_labels.append("empty_column")
                self.register_candidate(
                    column_name,
                    reason="High missingness",
                    priority=priority,
                    missing_pct=numeric_missing,
                    signals=signal_labels,
                )

    def ingest_eda_recommendations(self, recommendations: Iterable[Any]) -> None:
        for recommendation in recommendations:
            if not isinstance(recommendation, dict):
                continue

            columns = recommendation.get("columns")
            if not columns:
                continue

            category = str(recommendation.get("category", "")).strip()
            if category and category not in {"data_quality", "missing_data", "feature_engineering"}:
                continue

            reason = recommendation.get("title") or recommendation.get("description") or "EDA recommendation"
            priority = recommendation.get("priority")
            signal_type = recommendation.get("signal_type")
            tags = recommendation.get("tags")

            for column in columns:
                self.register_candidate(
                    column,
                    reason=str(reason),
                    priority=str(priority) if priority else None,
                    signals=[signal_type] if signal_type else None,
                    tags=tags,
                )

    def collect_column_details(self, column_details: Iterable[Any]) -> None:
        for detail in column_details:
            if not isinstance(detail, dict):
                continue

            column_name = detail.get("name") or detail.get("column")
            normalized_name = self._normalize_name(column_name)
            if not normalized_name:
                continue

            detail_missing = (
                detail.get("missing_percentage")
                or detail.get("missing_percent")
                or detail.get("missing_pct")
                or detail.get("missing")
            )
            self.record_missing_percentage(normalized_name, detail_missing)

    def collect_sample_preview(self, quality_report: Dict[str, Any]) -> None:
        sample_preview = quality_report.get("sample_preview") or {}
        columns = sample_preview.get("columns") or []
        for column in columns:
            normalized = self._normalize_name(column)
            if normalized:
                self.all_columns.add(normalized)

    def build_candidate_payload(self) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for entry in self.candidates.values():
            payload.append(
                {
                    "name": entry.name,
                    "reason": ", ".join(sorted(entry.reasons)),
                    "missing_percentage": entry.missing_percentage,
                    "priority": entry.priority,
                    "signals": sorted(entry.signals),
                    "tags": sorted(entry.tags),
                }
            )
        return payload

    def filter_candidates(
        self,
        candidate_payload: List[Dict[str, Any]],
        allowed_columns: Optional[Set[str]],
    ) -> List[Dict[str, Any]]:
        if allowed_columns is None:
            return candidate_payload

        normalized_allowed = {
            self._normalize_name(column)
            for column in allowed_columns
            if self._normalize_name(column)
        }

        filtered = [
            candidate
            for candidate in candidate_payload
            if candidate.get("name") in normalized_allowed
        ]

        self.column_missing_map = {
            name: value
            for name, value in self.column_missing_map.items()
            if name in normalized_allowed
        }
        self.all_columns = {column for column in self.all_columns if column in normalized_allowed}
        self.all_columns.update(normalized_allowed)
        return filtered

    def finalize_all_columns(
        self,
        candidate_payload: Iterable[Dict[str, Any]],
        allowed_columns: Optional[Set[str]],
    ) -> List[str]:
        for candidate in candidate_payload:
            name = self._normalize_name(candidate.get("name"))
            if name:
                self.all_columns.add(name)

        if allowed_columns is not None:
            normalized_allowed = {
                self._normalize_name(column)
                for column in allowed_columns
                if self._normalize_name(column)
            }
            self.all_columns = {column for column in self.all_columns if column in normalized_allowed}
            self.all_columns.update(normalized_allowed)

        sorted_columns = sorted(self.all_columns)
        for column in sorted_columns:
            self.column_missing_map.setdefault(column, 0.0)

        return sorted_columns

    def build_column_missing_map(self, columns: Iterable[str]) -> Dict[str, float]:
        return {name: float(self.column_missing_map.get(name, 0.0)) for name in columns}

    def build_filters(self, candidate_payload: Iterable[Dict[str, Any]]) -> List[DropColumnRecommendationFilter]:
        available_filters_map: Dict[str, Dict[str, Any]] = {}
        orphan_candidates = 0

        for candidate in candidate_payload:
            signals = candidate.get("signals") or []
            if signals:
                for signal in signals:
                    if not signal:
                        continue
                    meta = available_filters_map.setdefault(
                        signal,
                        {
                            "id": signal,
                            **_build_drop_filter_meta(signal),
                            "count": 0,
                        },
                    )
                    meta["count"] += 1
            else:
                orphan_candidates += 1

        if orphan_candidates:
            other_meta = available_filters_map.setdefault(
                "other",
                {
                    "id": "other",
                    **_build_drop_filter_meta("other"),
                    "count": 0,
                },
            )
            other_meta["count"] += orphan_candidates

        filters = [
            DropColumnRecommendationFilter(**value)
            for value in available_filters_map.values()
            if value.get("count", 0) > 0
        ]
        filters.sort(key=lambda item: (-item.count, item.label))
        return filters

    def suggested_threshold(self) -> float:
        if not self.missing_values:
            return 40.0

        try:
            return float(max(20.0, min(95.0, median(self.missing_values))))
        except StatisticsError:  # pragma: no cover - defensive fallback
            return 40.0

    @staticmethod
    def sort_candidates(candidate_payload: List[Dict[str, Any]]) -> None:
        candidate_payload.sort(
            key=lambda item: (
                item.get("missing_percentage") is None,
                -(item.get("missing_percentage") or 0.0),
                item.get("name", ""),
            )
        )


def _build_drop_filter_meta(filter_id: str) -> Dict[str, Optional[str]]:
    base = DROP_COLUMN_FILTER_LABELS.get(filter_id, {})
    label = base.get("label") or filter_id.replace("_", " ").title() or "Other"
    description = base.get("description")
    return {"label": label, "description": description}


async def _resolve_drop_allowed_columns(
    session: AsyncSession,
    dataset_source_id: str,
    sample_size: int,
    node_map: Dict[str, Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    target_node_id: Optional[str],
) -> Optional[Set[str]]:
    if not (node_map or graph_edges or target_node_id):
        return None

    try:
        frame = await _load_preview_frame_for_recommendations(
            session,
            dataset_source_id,
            sample_size,
        )
    except HTTPException:
        return None

    try:
        frame = _apply_recommendation_graph(
            frame,
            node_map,
            graph_edges,
            target_node_id,
            skip_catalog_types=None,
        )
    except HTTPException:
        pass

    return {
        str(column).strip()
        for column in frame.columns
        if str(column).strip()
    }


@router.post(
    "/api/recommendations/drop-columns",
    response_model=DropColumnRecommendations,
)
async def get_drop_column_recommendations(
    request: RecommendationRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> DropColumnRecommendations:
    """Return column drop suggestions derived from EDA quality insights."""

    normalized_id = _require_dataset_source_id(request.dataset_source_id)

    graph_payload = _ensure_dict_graph_payload(request.graph)
    graph_node_map, graph_edges = _extract_graph_payload(graph_payload)
    normalized_target_node = _normalize_target_node(request.target_node_id)
    has_graph_context = bool(graph_node_map or graph_edges or normalized_target_node)

    if has_graph_context:
        graph_node_map = _ensure_dataset_node(graph_node_map)

    eda_service = _build_eda_service(session, request.sample_size)
    report = await eda_service.quality_report(normalized_id, sample_size=request.sample_size)

    if not report.get("success"):
        detail = report.get("error") or "Unable to generate quality report"
        raise HTTPException(status_code=400, detail=detail)

    quality_report = report.get("quality_report") or {}
    missing_summary = quality_report.get("missing_data_summary") or []
    eda_recommendations = quality_report.get("recommendations") or []
    column_details = (quality_report.get("quality_metrics") or {}).get("column_details") or []

    builder = DropColumnRecommendationBuilder()
    builder.ingest_missing_summary(missing_summary)
    builder.ingest_eda_recommendations(eda_recommendations)
    builder.collect_column_details(column_details)
    builder.collect_sample_preview(quality_report)

    allowed_columns: Optional[Set[str]] = None
    if has_graph_context:
        allowed_columns = await _resolve_drop_allowed_columns(
            session,
            dataset_source_id=normalized_id,
            sample_size=request.sample_size,
            node_map=graph_node_map,
            graph_edges=graph_edges,
            target_node_id=normalized_target_node,
        )

    candidate_payload = builder.build_candidate_payload()
    candidate_payload = builder.filter_candidates(candidate_payload, allowed_columns)
    available_filters = builder.build_filters(candidate_payload)

    all_columns = builder.finalize_all_columns(candidate_payload, allowed_columns)
    column_missing_map = builder.build_column_missing_map(all_columns)

    builder.sort_candidates(candidate_payload)

    candidates = [DropColumnCandidate.model_validate(candidate) for candidate in candidate_payload]

    return DropColumnRecommendations(
        dataset_source_id=normalized_id,
        suggested_threshold=builder.suggested_threshold(),
        candidates=candidates,
        available_filters=available_filters,
        all_columns=all_columns,
        column_missing_map=column_missing_map,
    )


@router.post(
    "/api/recommendations/label-encoding",
    response_model=LabelEncodingRecommendationsResponse,
)
async def get_label_encoding_recommendations(
    request: RecommendationRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> LabelEncodingRecommendationsResponse:
    normalized_id = _require_dataset_source_id(request.dataset_source_id)

    eda_service = _build_eda_service(session, request.sample_size)
    frame, column_metadata, notes = await _prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=normalized_id,
        sample_size=request.sample_size,
        graph=request.graph,
        target_node_id=request.target_node_id,
        skip_catalog_types={
            "label_encoding",
            "dummy_encoding",
            "ordinal_encoding",
            "target_encoding",
            "hash_encoding",
        },
    )

    notes = list(notes)
    suggestions = build_label_encoding_suggestions(frame, column_metadata)

    total_text_columns = len(suggestions)
    recommended_count = sum(1 for item in suggestions if item.status == "recommended")
    high_cardinality_columns = [item.column for item in suggestions if item.status == "high_cardinality"]

    if not suggestions:
        notes.append("No text columns detected in the sampled data.")
    if high_cardinality_columns:
        notes.append(
            "High-cardinality columns detected: " + ", ".join(sorted(high_cardinality_columns))
        )

    return LabelEncodingRecommendationsResponse(
        dataset_source_id=normalized_id,
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


@router.post(
    "/api/recommendations/target-encoding",
    response_model=TargetEncodingRecommendationsResponse,
)
async def get_target_encoding_recommendations(
    request: RecommendationRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> TargetEncodingRecommendationsResponse:
    normalized_id = _require_dataset_source_id(request.dataset_source_id)

    eda_service = _build_eda_service(session, request.sample_size)
    frame, column_metadata, notes = await _prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=normalized_id,
        sample_size=request.sample_size,
        graph=request.graph,
        target_node_id=request.target_node_id,
        skip_catalog_types={
            "target_encoding",
            "label_encoding",
            "ordinal_encoding",
            "dummy_encoding",
            "one_hot_encoding",
            "hash_encoding",
        },
    )

    notes = list(notes)
    suggestions = build_target_encoding_suggestions(frame, column_metadata)

    total_text_columns = len(suggestions)
    recommended_count = sum(1 for item in suggestions if item.status == "recommended" and item.selectable)
    enable_global_fallback_default = any(
        item.recommended_use_global_fallback for item in suggestions if item.selectable
    )
    high_cardinality_columns = [
        item.column for item in suggestions if item.status in {"high_cardinality", "too_many_categories"}
    ]

    if not suggestions:
        notes.append("No text columns detected in the sampled data.")
    if high_cardinality_columns:
        notes.append(
            "High-cardinality columns detected: " + ", ".join(sorted(high_cardinality_columns))
        )

    return TargetEncodingRecommendationsResponse(
        dataset_source_id=normalized_id,
        sample_size=int(frame.shape[0]),
        total_text_columns=total_text_columns,
        recommended_count=recommended_count,
        auto_detect_default=bool(recommended_count),
        enable_global_fallback_default=enable_global_fallback_default,
        high_cardinality_columns=high_cardinality_columns,
        notes=notes,
        columns=[
            TargetEncodingColumnSuggestion(**suggestion.to_payload())
            for suggestion in suggestions
        ],
    )


@router.post(
    "/api/recommendations/hash-encoding",
    response_model=HashEncodingRecommendationsResponse,
)
async def get_hash_encoding_recommendations(
    request: RecommendationRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> HashEncodingRecommendationsResponse:
    normalized_id = _require_dataset_source_id(request.dataset_source_id)

    eda_service = _build_eda_service(session, request.sample_size)
    frame, column_metadata, notes = await _prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=normalized_id,
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

    notes = list(notes)
    suggestions = build_hash_encoding_suggestions(frame, column_metadata)

    total_text_columns = len(suggestions)
    recommended_count = sum(1 for item in suggestions if item.status == "recommended" and item.selectable)
    high_cardinality_columns = [
        item.column
        for item in suggestions
        if (item.unique_count or 0) > HASH_ENCODING_DEFAULT_MAX_CATEGORIES
    ]

    if not suggestions:
        notes.append("No text columns detected in the sampled data.")
    if high_cardinality_columns:
        notes.append(
            "Hash encoding is well-suited for high-cardinality columns such as: "
            + ", ".join(sorted(high_cardinality_columns))
        )

    return HashEncodingRecommendationsResponse(
        dataset_source_id=normalized_id,
        sample_size=int(frame.shape[0]),
        total_text_columns=total_text_columns,
        recommended_count=recommended_count,
        auto_detect_default=bool(recommended_count),
        suggested_bucket_default=HASH_ENCODING_DEFAULT_BUCKETS,
        high_cardinality_columns=high_cardinality_columns,
        notes=notes,
        columns=[
            HashEncodingColumnSuggestion(**suggestion.to_payload())
            for suggestion in suggestions
        ],
    )


@router.post(
    "/api/recommendations/ordinal-encoding",
    response_model=OrdinalEncodingRecommendationsResponse,
)
async def get_ordinal_encoding_recommendations(
    request: RecommendationRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> OrdinalEncodingRecommendationsResponse:
    normalized_id = _require_dataset_source_id(request.dataset_source_id)

    eda_service = _build_eda_service(session, request.sample_size)
    frame, column_metadata, notes = await _prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=normalized_id,
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

    notes = list(notes)
    suggestions = build_ordinal_encoding_suggestions(frame, column_metadata)

    total_text_columns = len(suggestions)
    recommended_count = sum(1 for item in suggestions if item.status == "recommended" and item.selectable)
    enable_unknown_default = any(item.recommended_handle_unknown for item in suggestions if item.selectable)
    high_cardinality_columns = [
        item.column for item in suggestions if item.status in {"high_cardinality", "too_many_categories"}
    ]

    if not suggestions:
        notes.append("No text columns detected in the sampled data.")
    if high_cardinality_columns:
        notes.append(
            "High-cardinality columns detected: " + ", ".join(sorted(high_cardinality_columns))
        )

    return OrdinalEncodingRecommendationsResponse(
        dataset_source_id=normalized_id,
        sample_size=int(frame.shape[0]),
        total_text_columns=total_text_columns,
        recommended_count=recommended_count,
        auto_detect_default=bool(recommended_count),
        enable_unknown_default=enable_unknown_default,
        high_cardinality_columns=high_cardinality_columns,
        notes=notes,
        columns=[
            OrdinalEncodingColumnSuggestion(**suggestion.to_payload())
            for suggestion in suggestions
        ],
    )


@router.post(
    "/api/recommendations/one-hot-encoding",
    response_model=OneHotEncodingRecommendationsResponse,
)
async def get_one_hot_encoding_recommendations(
    request: RecommendationRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> OneHotEncodingRecommendationsResponse:
    logger.debug(
        "One-hot encoding recommendations request",
        extra={
            "dataset_source_id": request.dataset_source_id,
            "target_node_id": request.target_node_id,
            "has_graph": bool(request.graph),
        },
    )

    normalized_id = _require_dataset_source_id(request.dataset_source_id)

    logger.debug("Building EDA service", extra={"sample_size": request.sample_size})
    eda_service = _build_eda_service(session, request.sample_size)

    logger.debug("Calling _prepare_categorical_recommendation_context")
    try:
        frame, column_metadata, notes = await _prepare_categorical_recommendation_context(
            eda_service=eda_service,
            dataset_source_id=normalized_id,
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
        logger.debug(
            "Context preparation successful",
            extra={"frame_shape": frame.shape, "notes_count": len(notes)},
        )
    except Exception as exc:
        logger.exception("Failed in _prepare_categorical_recommendation_context: %s", exc)
        raise

    notes = list(notes)
    logger.debug("Building one-hot encoding suggestions", extra={"frame_shape": frame.shape})
    suggestions = build_one_hot_encoding_suggestions(frame, column_metadata)
    logger.debug("Suggestions built", extra={"total_suggestions": len(suggestions)})

    total_text_columns = len(suggestions)
    recommended_count = sum(1 for item in suggestions if item.status == "recommended" and item.selectable)
    cautioned_count = sum(
        1 for item in suggestions if item.status in {"high_cardinality", "too_many_categories"}
    )
    high_cardinality_columns = [
        item.column for item in suggestions if item.status in {"high_cardinality", "too_many_categories"}
    ]

    if not suggestions:
        notes.append("No text columns detected in the sampled data.")
    if high_cardinality_columns:
        notes.append(
            "Wide expansion risk from high-cardinality columns: " + ", ".join(sorted(high_cardinality_columns))
        )

    return OneHotEncodingRecommendationsResponse(
        dataset_source_id=normalized_id,
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


@router.post(
    "/api/recommendations/dummy-encoding",
    response_model=DummyEncodingRecommendationsResponse,
)
async def get_dummy_encoding_recommendations(
    request: RecommendationRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> DummyEncodingRecommendationsResponse:
    normalized_id = _require_dataset_source_id(request.dataset_source_id)

    eda_service = _build_eda_service(session, request.sample_size)
    frame, column_metadata, notes = await _prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=normalized_id,
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

    notes = list(notes)
    suggestions = build_dummy_encoding_suggestions(frame, column_metadata)

    total_text_columns = len(suggestions)
    recommended_count = sum(1 for item in suggestions if item.status == "recommended" and item.selectable)
    cautioned_count = sum(
        1 for item in suggestions if item.status in {"high_cardinality", "too_many_categories"}
    )
    high_cardinality_columns = [
        item.column for item in suggestions if item.status in {"high_cardinality", "too_many_categories"}
    ]

    if not suggestions:
        notes.append("No text columns detected in the sampled data.")
    if high_cardinality_columns:
        notes.append(
            "Wide expansion risk from high-cardinality columns: " + ", ".join(sorted(high_cardinality_columns))
        )

    return DummyEncodingRecommendationsResponse(
        dataset_source_id=normalized_id,
        sample_size=int(frame.shape[0]),
        total_text_columns=total_text_columns,
        recommended_count=recommended_count,
        cautioned_count=cautioned_count,
        high_cardinality_columns=high_cardinality_columns,
        notes=notes,
        auto_detect_default=bool(recommended_count),
        columns=[
            DummyEncodingColumnSuggestion(**suggestion.to_payload())
            for suggestion in suggestions
        ],
    )


@router.post(
    "/api/recommendations/scaling",
    response_model=ScalingRecommendationsResponse,
)
async def get_scaling_recommendations(
    request: RecommendationRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> ScalingRecommendationsResponse:
    """Return recommended scaling strategies for numeric columns."""

    normalized_id = _require_dataset_source_id(request.dataset_source_id)
    frame = await _load_preview_frame_for_recommendations(session, normalized_id, request.sample_size)

    graph_payload = _ensure_dict_graph_payload(request.graph)
    graph_node_map, graph_edges = _extract_graph_payload(graph_payload)
    normalized_target_node = _normalize_target_node(request.target_node_id)

    try:
        frame = _apply_recommendation_graph(
            frame,
            graph_node_map,
            graph_edges,
            normalized_target_node,
            skip_catalog_types=None,
        )
    except HTTPException:
        pass

    recommendations = _build_scaling_recommendations(frame)

    return ScalingRecommendationsResponse(
        dataset_source_id=normalized_id,
        sample_size=int(frame.shape[0]),
        methods=_scaling_method_details(),
        columns=recommendations,
    )


@router.post(
    "/api/recommendations/binning",
    response_model=BinningRecommendationsResponse,
)
async def get_binning_recommendations(
    request: RecommendationRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> BinningRecommendationsResponse:
    """Return binning recommendations for numeric features."""

    normalized_id = _require_dataset_source_id(request.dataset_source_id)
    frame = await _load_preview_frame_for_recommendations(session, normalized_id, request.sample_size)

    graph_payload = _ensure_dict_graph_payload(request.graph)
    graph_node_map, graph_edges = _extract_graph_payload(graph_payload)
    normalized_target_node = _normalize_target_node(request.target_node_id)

    try:
        frame = _apply_recommendation_graph(
            frame,
            graph_node_map,
            graph_edges,
            normalized_target_node,
            skip_catalog_types=None,
        )
    except HTTPException:
        pass

    recommendations, excluded = _build_binning_recommendations(frame)

    return BinningRecommendationsResponse(
        dataset_source_id=normalized_id,
        sample_size=int(frame.shape[0]),
        columns=recommendations,
        excluded_columns=excluded,
    )


@router.post(
    "/api/recommendations/outliers",
    response_model=OutlierRecommendationsResponse,
)
async def get_outlier_recommendations(
    request: RecommendationRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> OutlierRecommendationsResponse:
    normalized_id = _require_dataset_source_id(request.dataset_source_id)
    frame = await _load_preview_frame_for_recommendations(session, normalized_id, request.sample_size)

    graph_payload = _ensure_dict_graph_payload(request.graph)
    graph_node_map, graph_edges = _extract_graph_payload(graph_payload)
    normalized_target_node = _normalize_target_node(request.target_node_id)

    try:
        frame = _apply_recommendation_graph(
            frame,
            graph_node_map,
            graph_edges,
            normalized_target_node,
            skip_catalog_types=None,
        )
    except HTTPException:
        pass

    recommendations = _build_outlier_recommendations(frame)

    return OutlierRecommendationsResponse(
        dataset_source_id=normalized_id,
        sample_size=int(frame.shape[0]),
        default_method=OUTLIER_DEFAULT_METHOD,
        methods=_outlier_method_details(),
        columns=recommendations,
    )


@router.post(
    "/api/recommendations/skewness",
    response_model=SkewnessRecommendationsResponse,
)
async def get_skewness_recommendations(
    request: SkewnessRecommendationRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> SkewnessRecommendationsResponse:
    """Return transformation suggestions for skewed numeric columns."""

    normalized_id = _require_dataset_source_id(request.dataset_source_id)
    frame = await _load_preview_frame_for_recommendations(session, normalized_id, request.sample_size)

    graph_payload = _ensure_dict_graph_payload(request.graph)
    frame, graph_selected_methods = _apply_skewness_graph_context(
        frame,
        graph_payload,
        request.target_node_id,
    )

    selected_methods = _parse_skewness_transformations(request.transformations)

    recommendations = _build_skewness_recommendations(frame, selected_methods, graph_selected_methods)

    return SkewnessRecommendationsResponse(
        dataset_source_id=normalized_id,
        sample_size=request.sample_size,
        skewness_threshold=SKEWNESS_THRESHOLD,
        methods=_skewness_method_details(),
        columns=recommendations,
    )


async def _generate_binned_distribution_response(
    session: AsyncSession,
    *,
    dataset_source_id: str,
    sample_size: int,
    graph_input: Any,
    target_node_id: Optional[str],
) -> BinnedDistributionResponse:
    normalized_id = _require_dataset_source_id(dataset_source_id)
    normalized_sample_size = 0 if sample_size == 0 else max(50, sample_size)
    frame, _ = await _load_dataset_frame(
        session,
        normalized_id,
        sample_size=normalized_sample_size,
    )

    normalized_target_node = _normalize_target_node(target_node_id)
    graph_payload = _resolve_binned_graph_payload(graph_input)

    graph_node_map, graph_edges = _extract_graph_payload(graph_payload)
    frame, execution_order, ensured_map = _apply_graph_with_execution_order(
        frame,
        graph_node_map,
        graph_edges,
        normalized_target_node,
    )

    binned_column_metadata = (
        _collect_binned_columns_from_graph(execution_order, ensured_map, normalized_target_node)
        if ensured_map
        else {}
    )

    candidate_columns = _build_candidate_binned_columns(frame, binned_column_metadata)
    distributions = _build_binned_distributions_list(frame, candidate_columns)

    return BinnedDistributionResponse(
        dataset_source_id=normalized_id,
        sample_size=int(frame.shape[0]),
        columns=distributions,
    )


@router.get(
    "/api/analytics/binned-distribution",
    response_model=BinnedDistributionResponse,
)
async def get_binned_distribution(
    dataset_source_id: str = Query(..., description="Source identifier for retrieving binned column distributions."),
    sample_size: int = Query(
        500,
        ge=0,
        le=10000,
        description="Rows sampled when computing binned summaries. Use 0 to process the full dataset.",
    ),
    graph: Optional[str] = Query(
        None,
        description="JSON payload containing graph nodes and edges to simulate upstream pipeline steps.",
    ),
    target_node_id: Optional[str] = Query(
        None,
        description="Identifier of the node requesting binned column insights to determine upstream context.",
    ),
    session: AsyncSession = Depends(get_async_session),
) -> BinnedDistributionResponse:
    return await _generate_binned_distribution_response(
        session,
        dataset_source_id=dataset_source_id,
        sample_size=sample_size,
        graph_input=graph,
        target_node_id=target_node_id,
    )


@router.post(
    "/api/analytics/binned-distribution",
    response_model=BinnedDistributionResponse,
)
async def post_binned_distribution(
    payload: BinnedDistributionRequest,
    session: AsyncSession = Depends(get_async_session),
) -> BinnedDistributionResponse:
    sample_size = payload.sample_size if payload.sample_size is not None else 500
    return await _generate_binned_distribution_response(
        session,
        dataset_source_id=payload.dataset_source_id,
        sample_size=sample_size,
        graph_input=payload.graph,
        target_node_id=payload.target_node_id,
    )


@router.get(
    "/api/analytics/quick-profile",
    response_model=QuickProfileResponse,
)
async def get_quick_profile(
    dataset_source_id: str = Query(..., description="Source identifier for generating a lightweight profile."),
    sample_size: int = Query(
        500,
        ge=0,
        le=5000,
        description="Rows sampled when generating the profile. Use 0 to profile the full dataset (subject to limits).",
    ),
    graph: Optional[str] = Query(
        None,
        description="JSON payload containing graph nodes and edges to simulate upstream pipeline steps.",
    ),
    target_node_id: Optional[str] = Query(
        None,
        description="Identifier of the node requesting the profile for upstream context.",
    ),
    session: AsyncSession = Depends(get_async_session),
) -> QuickProfileResponse:
    normalized_id = dataset_source_id.strip()
    if not normalized_id:
        raise HTTPException(status_code=400, detail="dataset_source_id must not be empty")

    normalized_sample_size = 0 if sample_size == 0 else max(50, sample_size)

    frame, preview_meta = await _load_dataset_frame(
        session,
        normalized_id,
        sample_size=normalized_sample_size,
    )

    graph_node_map: Dict[str, Dict[str, Any]] = {}
    graph_edges: List[Dict[str, Any]] = []
    normalized_target_node = None
    if isinstance(target_node_id, str):
        stripped_target = target_node_id.strip()
        if stripped_target:
            normalized_target_node = stripped_target

    if graph:
        try:
            graph_payload = json.loads(graph)
        except (TypeError, ValueError) as e:
            logger.error(
                f"Failed to parse graph JSON in quick-profile endpoint: {e}",
                extra={
                    "endpoint": "/api/analytics/quick-profile",
                    "graph_type": type(graph).__name__,
                    "graph_length": len(graph),
                    "graph_preview": graph[:200] if graph else None,
                },
            )
            raise HTTPException(status_code=400, detail="graph must be valid JSON")

        if not isinstance(graph_payload, dict):
            raise HTTPException(status_code=400, detail="graph payload must be a JSON object")

        graph_node_map = _sanitize_graph_nodes(graph_payload.get("nodes"))
        graph_edges = _sanitize_graph_edges(graph_payload.get("edges"))

    if graph_node_map or graph_edges or normalized_target_node:
        graph_node_map = _ensure_dataset_node(graph_node_map)
        try:
            frame = _apply_graph_transformations_before_node(
                frame,
                graph_node_map,
                graph_edges,
                normalized_target_node,
                skip_catalog_types={"dataset_profile"},
                pipeline_id=None,
            )
        except Exception:
            pass

    if frame.empty:
        raise HTTPException(status_code=400, detail="No data available to profile. Ensure upstream steps produce rows.")

    payload = build_quick_profile_payload(frame, normalized_id)

    effective_sample_size = _coerce_int(preview_meta.get("sample_size"), frame.shape[0])

    return QuickProfileResponse(
        dataset_source_id=normalized_id,
        generated_at=utcnow(),
        sample_size=effective_sample_size,
        rows_analyzed=int(frame.shape[0]),
        columns_analyzed=int(frame.shape[1]),
        metrics=payload.get("metrics") or QuickProfileDatasetMetrics(),
        columns=payload.get("columns") or [],
        correlations=payload.get("correlations") or [],
        warnings=payload.get("warnings") or [],
    )


@dataclass(frozen=True)
class PreviewSamplingConfig:
    execution_order: List[str]
    target_catalog_type: str
    include_preview_rows: bool
    effective_sample_size: int
    metrics_requested_sample_size: int
    load_mode: Literal["auto", "sample", "full"]


def _build_preview_node_map(graph_nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    node_map: Dict[str, Dict[str, Any]] = {}
    for node in graph_nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        data = node.get("data")
        safe_data = {key: value for key, value in data.items() if not callable(value)} if isinstance(data, dict) else {}
        node_map[str(node_id)] = {**node, "data": safe_data}

    if DATASET_NODE_ID not in node_map:
        node_map[DATASET_NODE_ID] = {
            "id": DATASET_NODE_ID,
            "data": {"catalogType": "dataset", "label": "Dataset input", "isDataset": True},
        }

    return node_map


def _resolve_preview_sampling(
    payload: PipelinePreviewRequest,
    node_map: Dict[str, Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
) -> PreviewSamplingConfig:
    execution_order = _execution_order(node_map, graph_edges, payload.target_node_id)
    if not execution_order:
        execution_order = [DATASET_NODE_ID]

    target_node = node_map.get(payload.target_node_id) if payload.target_node_id else None
    target_catalog_type = _resolve_catalog_type(target_node) if target_node else ""

    include_preview_rows = bool(payload.include_preview_rows) or target_catalog_type == "data_preview"

    try:
        requested_sample_size = int(payload.sample_size)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        requested_sample_size = 1000
    if requested_sample_size < 0:
        requested_sample_size = 0

    effective_sample_size = requested_sample_size if include_preview_rows else 0
    if include_preview_rows and target_catalog_type == "data_preview" and effective_sample_size <= 0:
        effective_sample_size = DEFAULT_SAMPLE_CAP

    metrics_requested_sample_size = max(effective_sample_size, 0)
    if effective_sample_size > 0:
        load_mode: Literal["auto", "sample", "full"] = "sample"
    else:
        load_mode = "auto"

    return PreviewSamplingConfig(
        execution_order=execution_order,
        target_catalog_type=target_catalog_type,
        include_preview_rows=include_preview_rows,
        effective_sample_size=effective_sample_size,
        metrics_requested_sample_size=metrics_requested_sample_size,
        load_mode=load_mode,
    )


async def _load_preview_dataset(
    session: AsyncSession,
    dataset_source_id: str,
    sampling: PreviewSamplingConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any], int, int, int]:
    frame, preview_meta = await _load_dataset_frame(
        session,
        dataset_source_id,
        sample_size=sampling.effective_sample_size,
        execution_mode=sampling.load_mode,
        allow_empty_sample=not sampling.include_preview_rows,
    )

    preview_rows = int(frame.shape[0])
    preview_total_rows = _coerce_int(
        preview_meta.get("total_rows") if isinstance(preview_meta, dict) else None,
        preview_rows if sampling.include_preview_rows else 0,
    )
    metrics_requested_sample_size = _coerce_int(
        preview_meta.get("sample_size") if isinstance(preview_meta, dict) else None,
        sampling.metrics_requested_sample_size if sampling.metrics_requested_sample_size > 0 else preview_rows,
    )

    return frame, preview_meta, preview_rows, preview_total_rows, metrics_requested_sample_size


def _build_full_preview_signal(
    working_frame: pd.DataFrame,
    applied_steps: List[str],
    dataset_source_id: str,
    preview_total_rows: int,
) -> Tuple[FullExecutionSignal, int]:
    total_rows_actual = preview_total_rows if preview_total_rows > 0 else int(working_frame.shape[0])
    signal = FullExecutionSignal(
        status="succeeded",
        reason="Preview executed against full dataset.",
        total_rows=total_rows_actual,
        processed_rows=int(working_frame.shape[0]),
        applied_steps=list(applied_steps),
        dataset_source_id=dataset_source_id,
        last_updated=utcnow(),
    )
    return signal, total_rows_actual


def _should_defer_full_execution(total_rows_estimate: Optional[int]) -> bool:
    return bool(total_rows_estimate and total_rows_estimate > FULL_DATASET_EXECUTION_ROW_LIMIT)


async def _defer_full_execution_for_limit(
    *,
    applied_steps: List[str],
    dataset_source_id: str,
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    payload: PipelinePreviewRequest,
    graph_nodes: List[Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    total_rows_estimate: int,
    preview_total_rows: int,
) -> Tuple[FullExecutionSignal, int]:
    defer_reason = (
        f"Full dataset execution deferred – {total_rows_estimate:,} rows exceeds limit "
        f"{FULL_DATASET_EXECUTION_ROW_LIMIT:,}."
    )
    _append_unique_step(applied_steps, defer_reason)

    job, job_signal, _ = await full_execution_job_store.ensure_job(
        dataset_source_id=dataset_source_id,
        execution_order=execution_order,
        node_map=node_map,
        target_node_id=payload.target_node_id,
        total_rows_estimate=total_rows_estimate,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        defer_reason=defer_reason,
    )

    if job_signal.reason and job_signal.reason.strip() != defer_reason.strip():
        _append_unique_step(applied_steps, job_signal.reason)

    _schedule_full_execution_job(job)
    return job_signal, preview_total_rows


async def _run_full_dataset_execution(
    *,
    session: AsyncSession,
    dataset_source_id: str,
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    pipeline_id: str,
    applied_steps: List[str],
    preview_total_rows: int,
) -> Tuple[FullExecutionSignal, int]:
    full_frame, full_meta = await _load_dataset_frame(
        session,
        dataset_source_id,
        sample_size=0,
        execution_mode="full",
    )

    _, full_applied_steps, _, _ = _run_pipeline_execution(
        full_frame,
        execution_order,
        node_map,
        pipeline_id=pipeline_id,
        collect_signals=False,
    )

    processed_rows = int(full_frame.shape[0])
    total_rows_full = _coerce_int(full_meta.get("total_rows"), processed_rows)
    signal = FullExecutionSignal(
        status="succeeded",
        total_rows=total_rows_full,
        processed_rows=processed_rows,
        applied_steps=full_applied_steps,
        dataset_source_id=dataset_source_id,
        last_updated=utcnow(),
    )

    if total_rows_full:
        preview_total_rows = total_rows_full

    _append_unique_step(applied_steps, f"Full dataset run processed {processed_rows:,} row(s).")
    return signal, preview_total_rows


def _build_failed_full_execution_signal(
    *,
    status: FullExecutionJobStatus,
    reason: str,
    dataset_source_id: str,
    total_rows_estimate: Optional[int],
    warnings: Optional[List[str]] = None,
) -> FullExecutionSignal:
    if status == "succeeded":
        signal_status: Literal["succeeded", "deferred", "skipped", "failed"] = "succeeded"
    elif status in {"queued", "running"}:
        signal_status = "deferred"
    elif status == "cancelled":
        signal_status = "skipped"
    else:
        signal_status = "failed"

    return FullExecutionSignal(
        status=signal_status,
        reason=reason,
        total_rows=total_rows_estimate,
        warnings=warnings or [],
        dataset_source_id=dataset_source_id,
        last_updated=utcnow(),
    )


async def _maybe_collect_full_execution(
    session: AsyncSession,
    include_preview_rows: bool,
    effective_sample_size: int,
    preview_total_rows: int,
    working_frame: pd.DataFrame,
    applied_steps: List[str],
    dataset_source_id: str,
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    pipeline_id: str,
    payload: PipelinePreviewRequest,
    graph_nodes: List[Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
) -> Tuple[Optional[FullExecutionSignal], int]:
    if not include_preview_rows:
        return None, preview_total_rows

    dataset_already_full = effective_sample_size == 0
    if dataset_already_full:
        signal, updated_total_rows = _build_full_preview_signal(
            working_frame,
            applied_steps,
            dataset_source_id,
            preview_total_rows,
        )
        return signal, updated_total_rows

    total_rows_estimate = preview_total_rows if preview_total_rows > 0 else None
    if _should_defer_full_execution(total_rows_estimate):
        signal, updated_total_rows = await _defer_full_execution_for_limit(
            applied_steps=applied_steps,
            dataset_source_id=dataset_source_id,
            execution_order=execution_order,
            node_map=node_map,
            payload=payload,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            total_rows_estimate=total_rows_estimate or 0,
            preview_total_rows=preview_total_rows,
        )
        return signal, updated_total_rows

    try:
        signal, updated_total_rows = await _run_full_dataset_execution(
            session=session,
            dataset_source_id=dataset_source_id,
            execution_order=execution_order,
            node_map=node_map,
            pipeline_id=pipeline_id,
            applied_steps=applied_steps,
            preview_total_rows=preview_total_rows,
        )
        return signal, updated_total_rows
    except MemoryError:
        reason = "Full dataset execution failed due to insufficient memory."
        signal = _build_failed_full_execution_signal(
            status="failed",
            reason=reason,
            dataset_source_id=dataset_source_id,
            total_rows_estimate=total_rows_estimate,
            warnings=["memory_error"],
        )
        _append_unique_step(applied_steps, reason)
        return signal, preview_total_rows
    except Exception as exc:
        message = f"Full dataset execution failed: {exc}"
        signal = _build_failed_full_execution_signal(
            status="failed",
            reason=str(exc),
            dataset_source_id=dataset_source_id,
            total_rows_estimate=total_rows_estimate,
        )
        _append_unique_step(applied_steps, message)
        return signal, preview_total_rows


@router.post(
    "/api/pipelines/preview",
    response_model=PipelinePreviewResponse,
)
async def preview_pipeline(
    payload: PipelinePreviewRequest,
    session: AsyncSession = Depends(get_async_session),
) -> PipelinePreviewResponse:
    dataset_source_id = (payload.dataset_source_id or "").strip()
    if not dataset_source_id:
        raise HTTPException(status_code=400, detail="dataset_source_id must not be empty")

    graph_nodes = payload.graph.nodes or []
    graph_edges = payload.graph.edges or []

    node_map = _build_preview_node_map(graph_nodes)
    sampling_config = _resolve_preview_sampling(payload, node_map, graph_edges)

    frame, preview_meta, preview_rows, preview_total_rows, metrics_requested_sample_size = await _load_preview_dataset(
        session,
        dataset_source_id,
        sampling_config,
    )

    # Generate stable pipeline ID from dataset + graph structure
    pipeline_id = _generate_pipeline_id(dataset_source_id, graph_nodes, graph_edges)

    collect_signals = bool(payload.include_signals)

    working_frame, applied_steps, preview_signals, modeling_metadata = _run_pipeline_execution(
        frame,
        sampling_config.execution_order,
        node_map,
        pipeline_id=pipeline_id,
        collect_signals=collect_signals,
    )

    full_execution_signal, preview_total_rows = await _maybe_collect_full_execution(
        session=session,
        include_preview_rows=sampling_config.include_preview_rows,
        effective_sample_size=sampling_config.effective_sample_size,
        preview_total_rows=preview_total_rows,
        working_frame=working_frame,
        applied_steps=applied_steps,
        dataset_source_id=dataset_source_id,
        execution_order=sampling_config.execution_order,
        node_map=node_map,
        pipeline_id=pipeline_id,
        payload=payload,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
    )

    if full_execution_signal and collect_signals:
        if preview_signals is None:
            preview_signals = PipelinePreviewSignals()
        preview_signals.full_execution = full_execution_signal

    baseline_sample_rows = _coerce_int(
        preview_meta.get("sample_size") if isinstance(preview_meta, dict) else None,
        metrics_requested_sample_size if metrics_requested_sample_size > 0 else preview_rows,
    )

    # Filter DataFrame by split type if target node is connected to a specific train/test/validation output
    if sampling_config.include_preview_rows and payload.target_node_id and SPLIT_TYPE_COLUMN in working_frame.columns:
        split_type = _determine_node_split_type(payload.target_node_id, graph_edges, node_map)

        if split_type:
            # Filter to show only the requested split
            original_rows = len(working_frame)
            working_frame = working_frame[working_frame[SPLIT_TYPE_COLUMN] == split_type].copy()
            filtered_rows = len(working_frame)

            if filtered_rows < original_rows:
                filter_msg = f"Showing {split_type} split: {filtered_rows:,} of {original_rows:,} rows"
                _append_unique_step(applied_steps, filter_msg)

            # Update preview metrics
            preview_rows = filtered_rows
            if preview_total_rows > 0:
                # Estimate total rows for this split based on the ratio
                ratio = filtered_rows / original_rows if original_rows > 0 else 0
                preview_total_rows = int(preview_total_rows * ratio)

    if sampling_config.include_preview_rows:
        return build_data_snapshot_response(
            working_frame,
            target_node_id=payload.target_node_id,
            preview_rows=preview_rows,
            preview_total_rows=preview_total_rows,
            initial_sample_rows=baseline_sample_rows,
            applied_steps=applied_steps,
            metrics_requested_sample_size=metrics_requested_sample_size,
            modeling_signals=modeling_metadata,
            signals=preview_signals if collect_signals else None,
            include_signals=collect_signals,
        )

    # Preview rows were not requested; return structural metadata only.
    return PipelinePreviewResponse(
        node_id=payload.target_node_id,
        columns=list(working_frame.columns),
        sample_rows=[],
        metrics=PipelinePreviewMetrics(
            row_count=preview_total_rows,
            column_count=int(working_frame.shape[1]),
            duplicate_rows=0,
            missing_cells=0,
            preview_rows=0,
            total_rows=preview_total_rows,
            requested_sample_size=metrics_requested_sample_size,
        ),
        column_stats=[],
        applied_steps=applied_steps,
        row_missing_stats=[],
        schema_summary=None,
        modeling_signals=modeling_metadata,
        signals=preview_signals if collect_signals else None,
    )


@router.get(
    "/api/pipelines/{dataset_source_id}/preview/rows",
    response_model=PipelinePreviewRowsResponse,
)
async def preview_pipeline_rows(
    dataset_source_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    mode: str = Query("head"),
    session: AsyncSession = Depends(get_async_session),
) -> PipelinePreviewRowsResponse:
    dataset_source = (dataset_source_id or "").strip()
    if not dataset_source:
        raise HTTPException(status_code=400, detail="dataset_source_id must not be empty")

    eda_service = _build_eda_service(session, limit)
    window_payload = await eda_service.preview_rows_window(
        dataset_source,
        offset=offset,
        limit=limit,
        mode=mode,
    )

    if not window_payload.get("success"):
        detail = window_payload.get("error") or window_payload.get("message") or "Unable to load preview rows"
        raise HTTPException(status_code=400, detail=detail)

    preview = window_payload.get("preview") or {}

    rows = preview.get("rows") or []
    columns = preview.get("columns") or []

    preview_offset = _coerce_int(preview.get("offset"), offset)
    preview_limit = _coerce_int(preview.get("limit"), limit)
    returned_rows = _coerce_int(preview.get("returned_rows"), len(rows))

    raw_total_rows = preview.get("total_rows")
    total_rows_value: Optional[int] = None
    if isinstance(raw_total_rows, (int, float)):
        total_rows_value = _coerce_int(raw_total_rows, returned_rows)

    next_offset = preview.get("next_offset")
    if next_offset is not None:
        next_offset = _coerce_int(next_offset, preview_offset + returned_rows)

    return PipelinePreviewRowsResponse(
        columns=[str(column) for column in columns],
        rows=rows,
        offset=preview_offset,
        limit=preview_limit,
        returned_rows=returned_rows,
        total_rows=total_rows_value,
        next_offset=next_offset,
        has_more=bool(preview.get("has_more", False)),
        sampling_mode=str(preview.get("mode") or "window"),
        sampling_adjustments=preview.get("sampling_adjustments") or [],
        large_dataset=bool(preview.get("large_dataset", False)),
    )


@router.get(
    "/api/pipelines/{dataset_source_id}/full-execution/{job_id}",
    response_model=FullExecutionSignal,
)
async def get_full_execution_status(
    dataset_source_id: str,
    job_id: str,
) -> FullExecutionSignal:
    normalized_dataset = (dataset_source_id or "").strip()
    if not normalized_dataset:
        raise HTTPException(status_code=400, detail="dataset_source_id must not be empty")

    normalized_job_id = (job_id or "").strip()
    if not normalized_job_id:
        raise HTTPException(status_code=400, detail="job_id must not be empty")

    signal = await full_execution_job_store.get_signal(normalized_dataset, normalized_job_id)
    if not signal:
        raise HTTPException(status_code=404, detail="Full dataset execution job not found")
    return signal


def _build_pipeline_response(pipeline: FeatureEngineeringPipeline) -> FeaturePipelineResponse:
    """Coerce a SQLAlchemy pipeline row into a response model without validation failures."""

    graph_payload: Dict[str, Any]
    raw_graph = getattr(pipeline, "graph", None)
    if isinstance(raw_graph, dict):
        graph_payload = raw_graph
    else:
        graph_payload = {"nodes": [], "edges": []}

    metadata_payload = pipeline.pipeline_metadata if isinstance(pipeline.pipeline_metadata, dict) else None

    response_payload = {
        "id": pipeline.id,
        "dataset_source_id": pipeline.dataset_source_id,
        "name": pipeline.name,
        "description": pipeline.description,
        "graph": graph_payload,
        "metadata": metadata_payload,
        "is_active": pipeline.is_active,
        "created_at": pipeline.created_at,
        "updated_at": pipeline.updated_at,
    }

    return FeaturePipelineResponse.model_validate(response_payload)


@router.get(
    "/api/pipelines/{dataset_source_id}",
    response_model=Optional[FeaturePipelineResponse],
)
async def get_pipeline(
    dataset_source_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> Optional[FeaturePipelineResponse]:
    """Fetch the latest saved pipeline for a dataset, if one exists."""

    result = await session.execute(
        select(FeatureEngineeringPipeline)
        .where(FeatureEngineeringPipeline.dataset_source_id == dataset_source_id)
        .order_by(FeatureEngineeringPipeline.updated_at.desc())
    )
    pipeline = result.scalars().first()

    return _build_pipeline_response(pipeline) if pipeline else None


@router.get(
    "/api/pipelines/{dataset_source_id}/history",
    response_model=List[FeaturePipelineResponse],
)
async def get_pipeline_history(  # pragma: no cover - fastapi route
    dataset_source_id: str,
    limit: int = Query(10, ge=1, le=50),
    session: AsyncSession = Depends(get_async_session),
) -> List[FeaturePipelineResponse]:
    """Return the most recent pipeline revisions for a dataset."""

    limit_value = max(1, min(limit, 50))

    result = await session.execute(
        select(FeatureEngineeringPipeline)
        .where(FeatureEngineeringPipeline.dataset_source_id == dataset_source_id)
        .order_by(
            FeatureEngineeringPipeline.updated_at.desc(),
            FeatureEngineeringPipeline.id.desc(),
        )
        .limit(limit_value)
    )

    pipelines = result.scalars().all()

    return [_build_pipeline_response(item) for item in pipelines]


@router.post(
    "/api/pipelines/{dataset_source_id}",
    response_model=FeaturePipelineResponse,
    status_code=status.HTTP_200_OK,
)
async def upsert_pipeline(
    dataset_source_id: str,
    payload: FeaturePipelineCreate,
    session: AsyncSession = Depends(get_async_session),
) -> FeaturePipelineResponse:
    """Create or update the pipeline associated with a dataset."""

    result = await session.execute(
        select(FeatureEngineeringPipeline)
        .where(FeatureEngineeringPipeline.dataset_source_id == dataset_source_id)
        .order_by(FeatureEngineeringPipeline.updated_at.desc())
    )
    pipeline = result.scalars().first()

    graph_payload = payload.graph.model_dump()

    if pipeline:
        setattr(pipeline, "name", payload.name)
        setattr(pipeline, "description", payload.description)
        setattr(pipeline, "graph", graph_payload)
        setattr(pipeline, "pipeline_metadata", payload.metadata)
    else:
        pipeline = FeatureEngineeringPipeline(
            dataset_source_id=dataset_source_id,
            name=payload.name,
            description=payload.description,
            graph=graph_payload,
            pipeline_metadata=payload.metadata,
        )

    session.add(pipeline)
    await session.commit()
    await session.refresh(pipeline)

    return _build_pipeline_response(pipeline)


@router.post(
    "/api/training-jobs",
    response_model=TrainingJobBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def enqueue_training_job(
    payload: TrainingJobCreate,
    session: AsyncSession = Depends(get_async_session),
) -> TrainingJobBatchResponse:
    """Create one or more background training jobs and optionally dispatch them to Celery."""

    created_jobs: List[TrainingJob] = []

    try:
        for model_type in payload.model_types:
            scoped_payload = payload.model_copy(update={"model_types": [model_type]})
            job = await create_training_job_record(
                session,
                scoped_payload,
                user_id=None,
                model_type_override=model_type,
            )
            created_jobs.append(job)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if payload.run_training:
        for job in created_jobs:
            try:
                dispatch_training_job(str(job.id))
            except Exception as exc:  # pragma: no cover - Celery connection issues
                await update_job_status(
                    session,
                    job,
                    status=TrainingJobStatus.FAILED,
                    error_message="Failed to enqueue training job",
                )
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Failed to enqueue training job",
                ) from exc

    job_payloads = [TrainingJobResponse.model_validate(job, from_attributes=True) for job in created_jobs]
    return TrainingJobBatchResponse(jobs=job_payloads)


@router.get(
    "/api/training-jobs/{job_id}",
    response_model=TrainingJobResponse,
)
async def get_training_job_detail(
    job_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> TrainingJobResponse:
    """Return a single training job (no authentication required)."""

    job = await fetch_training_job(session, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training job not found")

    return TrainingJobResponse.model_validate(job, from_attributes=True)


@router.get(
    "/api/training-jobs",
    response_model=TrainingJobListResponse,
)
async def list_training_job_records(
    pipeline_id: Optional[str] = Query(default=None),
    node_id: Optional[str] = Query(default=None),
    dataset_source_id: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
) -> TrainingJobListResponse:
    """Return recent training jobs (no authentication required for viewing)."""

    logger.debug(
        "Listing training jobs (dataset_source_id=%s pipeline_id=%s node_id=%s limit=%s)",
        dataset_source_id,
        pipeline_id,
        node_id,
        limit,
    )

    # Return all jobs since authentication is not required
    jobs = await fetch_training_jobs(
        session,
        user_id=None,
        dataset_source_id=dataset_source_id,
        pipeline_id=pipeline_id,
        node_id=node_id,
        limit=limit,
    )

    summaries = [TrainingJobSummary.model_validate(job, from_attributes=True) for job in jobs]
    return TrainingJobListResponse(jobs=summaries, total=len(summaries))


@router.post(
    "/api/training-jobs/{job_id}/evaluate",
    response_model=ModelEvaluationReport,
    status_code=status.HTTP_200_OK,
)
async def evaluate_trained_model(
    job_id: str,
    payload: ModelEvaluationRequest = Body(default_factory=ModelEvaluationRequest),
    session: AsyncSession = Depends(get_async_session),
) -> ModelEvaluationReport:
    """Generate diagnostic plots and metrics for a completed training job."""

    job = await fetch_training_job(session, job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Training job not found")

    if not job.artifact_uri:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Training job does not have a stored artifact yet. Re-run the job first.",
        )

    artifact_path = Path(job.artifact_uri)
    if not artifact_path.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="Model artifact could not be located on disk.",
        )

    try:
        artifact_bundle = joblib.load(artifact_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to load model artifact for job %s", job_id)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load model artifact.",
        ) from exc

    dataset_frame, node_config, _, _ = await _resolve_training_inputs(session, job)

    node_config_map: Dict[str, Any] = node_config if isinstance(node_config, dict) else {}
    job_metadata: Dict[str, Any] = job.job_metadata if isinstance(job.job_metadata, dict) else {}

    def _node_value(key: str) -> Any:
        return node_config_map[key] if key in node_config_map else None

    target_column = _node_value("target_column") or _node_value("targetColumn") or job_metadata.get("target_column")
    if not target_column:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Evaluation node requires a configured target column.",
        )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
        target_meta,
    ) = _prepare_training_data(dataset_frame, target_column)

    problem_type = job_metadata.get("resolved_problem_type")
    if not isinstance(problem_type, str) or problem_type.lower() not in {"classification", "regression"}:
        artifact_problem_type = artifact_bundle.get("problem_type") if isinstance(artifact_bundle, dict) else None
        if isinstance(artifact_problem_type, str) and artifact_problem_type.lower() in {"classification", "regression"}:
            problem_type = artifact_problem_type.lower()
        else:
            problem_type = "classification"
    else:
        problem_type = problem_type.lower()

    raw_splits = payload.splits or ["test"]
    normalized_splits: List[str] = []
    seen: set[str] = set()
    for entry in raw_splits:
        if entry is None:
            continue
        normalized = str(entry).strip().lower()
        if normalized in {"train", "training"}:
            key = "train"
        elif normalized in {"validation", "valid", "val"}:
            key = "validation"
        elif normalized in {"test", "testing"}:
            key = "test"
        else:
            continue
        if key not in seen:
            seen.add(key)
            normalized_splits.append(key)
    if not normalized_splits:
        normalized_splits = ["test"]

    model = artifact_bundle.get("model") if isinstance(artifact_bundle, dict) else None
    if model is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Model artifact is missing trained estimator.",
        )

    include_confusion = bool(payload.include_confusion)
    include_curves = bool(payload.include_curves)
    include_residuals = bool(payload.include_residuals)
    max_curve_points = payload.max_curve_points or 500
    max_scatter_points = payload.max_scatter_points or 750

    splits_payload: Dict[str, Any] = {}
    label_names = None
    if isinstance(target_meta, dict) and target_meta.get("dtype") == "categorical":
        label_names = target_meta.get("categories")

    split_map = {
        "train": (X_train, y_train),
        "validation": (X_validation, y_validation),
        "test": (X_test, y_test),
    }

    for split_name in normalized_splits:
        features, target = split_map.get(split_name, (None, None))
        if problem_type == "classification":
            split_report = build_classification_split_report(
                model,
                split_name=split_name,
                features=features,
                target=target,
                label_names=label_names,
                include_confusion=include_confusion,
                include_curves=include_curves,
                max_curve_points=max_curve_points,
            )
        else:
            split_report = build_regression_split_report(
                model,
                split_name=split_name,
                features=features,
                target=target,
                include_residuals=include_residuals,
                max_scatter_points=max_scatter_points,
            )

        splits_payload[split_name] = split_report

    generated_at = utcnow()
    resolved_problem_type: Literal["classification", "regression"] = (
        "classification" if problem_type != "regression" else "regression"
    )
    problem_type = resolved_problem_type

    report = ModelEvaluationReport(
        job_id=str(job.id),
        pipeline_id=str(job.pipeline_id) if job.pipeline_id is not None else None,
        node_id=str(job.node_id) if job.node_id is not None else None,
        generated_at=generated_at,
        problem_type=resolved_problem_type,
        target_column=str(target_column),
        splits={name: payload for name, payload in splits_payload.items()},
    )

    evaluation_record = {
        "generated_at": generated_at.isoformat(),
        "problem_type": problem_type,
        "target_column": target_column,
        "splits": {name: payload.model_dump() for name, payload in splits_payload.items()},
    }

    metrics_payload: Dict[str, Any] = dict(job.metrics or {})
    metrics_payload["evaluation"] = evaluation_record
    setattr(job, "metrics", metrics_payload)

    metadata_payload: Dict[str, Any] = dict(job.job_metadata or {})
    metadata_payload.setdefault("evaluation", {})
    metadata_payload["evaluation"].update(
        {
            "last_evaluated_at": generated_at.isoformat(),
            "splits": normalized_splits,
        }
    )
    metadata_payload.setdefault("target_column", target_column)
    setattr(job, "job_metadata", metadata_payload)

    await session.commit()
    await session.refresh(job)

    return report


@router.post(
    "/api/hyperparameter-tuning-jobs",
    response_model=HyperparameterTuningJobBatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def enqueue_hyperparameter_tuning_job(
    payload: HyperparameterTuningJobCreate,
    session: AsyncSession = Depends(get_async_session),

) -> HyperparameterTuningJobBatchResponse:
    """Create one or more hyperparameter tuning jobs and optionally dispatch them."""
    created_jobs: List[HyperparameterTuningJob] = []

    try:
        for model_type in payload.model_types:
            scoped_payload = payload.model_copy(update={"model_types": [model_type]})
            job = await create_hyperparameter_tuning_job_record(
                session,
                scoped_payload,
                user_id=None,
                model_type_override=model_type,
            )
            created_jobs.append(job)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if payload.run_tuning:
        for job in created_jobs:
            try:
                dispatch_hyperparameter_tuning_job(str(job.id))
            except Exception as exc:  # pragma: no cover - Celery connection issues
                await update_tuning_job_status(
                    session,
                    job,
                    status=HyperparameterTuningJobStatus.FAILED,
                    error_message="Failed to enqueue tuning job",
                )
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Failed to enqueue tuning job",
                ) from exc

    job_payloads = [
        HyperparameterTuningJobResponse.model_validate(job, from_attributes=True) for job in created_jobs
    ]
    return HyperparameterTuningJobBatchResponse(jobs=job_payloads)


@router.get(
    "/api/hyperparameter-tuning-jobs/{job_id}",
    response_model=HyperparameterTuningJobResponse,
)
async def get_hyperparameter_tuning_job_detail(
    job_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> HyperparameterTuningJobResponse:
    """Return a single hyperparameter tuning job."""

    job = await fetch_hyperparameter_tuning_job(session, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tuning job not found")

    return HyperparameterTuningJobResponse.model_validate(job, from_attributes=True)


@router.get(
    "/api/hyperparameter-tuning-jobs",
    response_model=HyperparameterTuningJobListResponse,
)
async def list_hyperparameter_tuning_jobs(
    pipeline_id: Optional[str] = Query(default=None),
    node_id: Optional[str] = Query(default=None),
    dataset_source_id: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_async_session),
) -> HyperparameterTuningJobListResponse:
    """Return recent hyperparameter tuning jobs."""

    jobs = await fetch_hyperparameter_tuning_jobs(
        session,
        user_id=None,
        dataset_source_id=dataset_source_id,
        pipeline_id=pipeline_id,
        node_id=node_id,
        limit=limit,
    )

    summaries = [HyperparameterTuningJobSummary.model_validate(job, from_attributes=True) for job in jobs]
    return HyperparameterTuningJobListResponse(jobs=summaries, total=len(summaries))


@router.get("/api/model-hyperparameters/{model_type}")
async def get_model_hyperparameters(
    model_type: str,
) -> Dict[str, Any]:
    """Return hyperparameter configuration for a specific model type."""
    from core.feature_engineering.modeling.config.hyperparameters import (
        get_hyperparameters_for_model,
        get_default_hyperparameters,
    )

    try:
        fields = get_hyperparameters_for_model(model_type)
        defaults = get_default_hyperparameters(model_type)

        return {
            "model_type": model_type,
            "fields": fields,
            "defaults": defaults,
        }
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model type '{model_type}' not found",
        )


@router.get("/api/hyperparameter-tuning/best-params/{model_type}")
async def get_best_hyperparameters_for_model(
    model_type: str,
    pipeline_id: Optional[str] = Query(default=None),
    dataset_source_id: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, Any]:
    """
    Return the best hyperparameters from the most recent successful tuning job
    for a specific model type.

    This endpoint allows the Train Model node to check if there are tuned
    hyperparameters available for the currently selected model type and display
    an "Apply Best Params" button when applicable.
    """

    # Build query to find the most recent successful tuning job for this model type
    stmt = select(HyperparameterTuningJob).where(
        HyperparameterTuningJob.model_type == model_type,
        HyperparameterTuningJob.status == HyperparameterTuningJobStatus.SUCCEEDED.value,
    )

    # Apply optional filters
    if pipeline_id:
        stmt = stmt.where(HyperparameterTuningJob.pipeline_id == pipeline_id)
    if dataset_source_id:
        stmt = stmt.where(HyperparameterTuningJob.dataset_source_id == dataset_source_id)

    # Order by most recent and get the first result
    stmt = stmt.order_by(HyperparameterTuningJob.finished_at.desc()).limit(1)

    result = await session.execute(stmt)
    job = result.scalar_one_or_none()

    if not job:
        return {
            "available": False,
            "model_type": model_type,
            "message": f"No successful tuning results found for model type '{model_type}'",
        }

    return {
        "available": True,
        "model_type": model_type,
        "job_id": job.id,
        "pipeline_id": job.pipeline_id,
        "node_id": job.node_id,
        "run_number": job.run_number,
        "best_params": job.best_params or {},
        "best_score": job.best_score,
        "scoring": job.scoring,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "search_strategy": job.search_strategy,
        "n_iterations": job.n_iterations,
    }
