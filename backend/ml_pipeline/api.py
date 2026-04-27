import json
import logging
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional, Union, cast

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from skyulf.data.dataset import SplitDataset
from skyulf.modeling.hyperparameters import (
    get_default_search_space,
    get_hyperparameters,
)
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.data_ingestion.service import DataIngestionService
from backend.database.engine import get_async_session
import backend.database.engine as db_engine
from backend.database.models import (
    FeatureEngineeringPipeline,
    AdvancedTuningJob,
    BasicTrainingJob,
    DataSource,
    Deployment,
)
from backend.ml_pipeline.constants import StepType
from backend.ml_pipeline.tasks import run_pipeline_task
from backend.realtime.events import JobEvent, publish_job_event
from backend.utils.file_utils import extract_file_path_from_source
from backend.ml_pipeline.services.evaluation_service import EvaluationService
from backend.ml_pipeline.resolution import resolve_pipeline_nodes
from backend.data.catalog import create_catalog_from_options, FileSystemCatalog

from .artifacts.local import LocalArtifactStore
from .execution.engine import PipelineEngine

# from .data.profiler import DataProfiler
# from .recommendations.schemas import Recommendation, AnalysisProfile
from .execution.jobs import JobInfo, JobManager
from .execution.schemas import NodeConfig, PipelineConfig, coerce_step_type

# from .data.loader import DataLoader
from .node_definitions import NodeRegistry, RegistryItem

logger = logging.getLogger(__name__)


# Stubs for deleted modules


class Recommendation(BaseModel):
    type: str  # "imputation", "cleaning", "encoding", "outlier", "transformation"
    rule_id: Optional[str] = None
    target_columns: List[str]
    action: Optional[str] = None
    message: Optional[str] = None
    severity: Optional[str] = "info"
    suggestion: Optional[str] = None


class AnalysisProfile(BaseModel):
    row_count: int
    column_count: int
    duplicate_row_count: int
    columns: Dict[str, Any]


class DataProfiler:
    @staticmethod
    def generate_profile(df: pd.DataFrame) -> AnalysisProfile:
        # Return a minimal profile
        columns = {}
        for col in df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            columns[col] = {
                "name": col,
                "dtype": str(df[col].dtype),
                "column_type": "numeric" if is_numeric else "categorical",
                "missing_count": int(df[col].isnull().sum()),
                "missing_ratio": float(df[col].isnull().mean()),
                "unique_count": int(df[col].nunique()),
                "min_value": (
                    float(cast(Union[float, int], df[col].min())) if is_numeric else None
                ),
                "max_value": (
                    float(cast(Union[float, int], df[col].max())) if is_numeric else None
                ),
                "mean_value": (
                    float(cast(Union[float, int], df[col].mean())) if is_numeric else None
                ),
                "std_value": (
                    float(cast(Union[float, int], df[col].std())) if is_numeric else None
                ),
                "skewness": (
                    float(cast(Union[float, int], df[col].skew())) if is_numeric else None
                ),
            }
        return AnalysisProfile(
            row_count=len(df),
            column_count=len(df.columns),
            duplicate_row_count=int(df.duplicated().sum()),
            columns=columns,
        )


class AdvisorEngine:
    def analyze(self, profile: AnalysisProfile) -> List[Recommendation]:
        recs = []

        # 1. Imputation
        missing_cols = [col for col, stats in profile.columns.items() if stats["missing_count"] > 0]
        if missing_cols:
            recs.append(
                Recommendation(
                    type="imputation",
                    rule_id="imputation_mean",  # Default rule id
                    target_columns=missing_cols,
                    message=f"Found {len(missing_cols)} columns with missing values.",
                    suggestion="Consider using SimpleImputer or KNNImputer.",
                )
            )

        # 2. Cleaning (Duplicates & High Missing)
        if profile.duplicate_row_count > 0:
            recs.append(
                Recommendation(
                    type="cleaning",
                    rule_id="duplicate_rows_drop",
                    target_columns=[],
                    action="drop_duplicates",
                    message=f"Found {profile.duplicate_row_count} duplicate rows.",
                    suggestion="Add a DropDuplicates node.",
                )
            )

        high_missing_cols = [
            col for col, stats in profile.columns.items() if stats["missing_ratio"] > 0.5
        ]
        if high_missing_cols:
            recs.append(
                Recommendation(
                    type="cleaning",
                    rule_id="high_missing_drop",
                    target_columns=high_missing_cols,
                    action="drop_columns",
                    message=f"Found {len(high_missing_cols)} columns with >50% missing values.",
                    suggestion="Consider dropping these columns.",
                )
            )

        # 3. Encoding (Categorical columns)
        # Test expects "one_hot_encoding" for low cardinality
        cat_cols = [
            col
            for col, stats in profile.columns.items()
            if stats["column_type"] == "categorical"
            and stats["unique_count"] < 20  # Arbitrary threshold for OHE
        ]
        if cat_cols:
            recs.append(
                Recommendation(
                    type="encoding",
                    rule_id="one_hot_encoding",
                    target_columns=cat_cols,
                    message=f"Found {len(cat_cols)} categorical columns suitable for OneHotEncoding.",
                    suggestion="Consider OneHotEncoder.",
                )
            )

        # 4. Outliers (Simple Z-score check proxy)
        # If max is > mean + 3*std or min < mean - 3*std
        outlier_cols = []
        for col, stats in profile.columns.items():
            if stats["column_type"] == "numeric" and stats["std_value"] and stats["std_value"] > 0:
                mean = stats["mean_value"]
                std = stats["std_value"]
                if (stats["max_value"] > mean + 3 * std) or (stats["min_value"] < mean - 3 * std):
                    outlier_cols.append(col)

        if outlier_cols:
            recs.append(
                Recommendation(
                    type="outlier",
                    rule_id="outlier_removal_iqr",
                    target_columns=outlier_cols,
                    message=f"Found {len(outlier_cols)} columns with potential outliers.",
                    suggestion="Consider using IsolationForest or Z-score filtering.",
                )
            )

        # 5. Transformation (Skewness)
        # Test expects "power_transform_box_cox" or "power_transform_yeo_johnson"
        pos_skewed_cols = []
        neg_skewed_cols = []

        for col, stats in profile.columns.items():
            if (
                stats["column_type"] == "numeric"
                and stats["skewness"]
                and abs(stats["skewness"]) > 1.0
            ):
                if stats["min_value"] > 0:
                    pos_skewed_cols.append(col)
                else:
                    neg_skewed_cols.append(col)

        if pos_skewed_cols:
            recs.append(
                Recommendation(
                    type="transformation",
                    rule_id="power_transform_box_cox",
                    target_columns=pos_skewed_cols,
                    message=f"Found {len(pos_skewed_cols)} positively skewed columns (strictly positive).",
                    suggestion="Consider Box-Cox transformation.",
                )
            )

        if neg_skewed_cols:
            recs.append(
                Recommendation(
                    type="transformation",
                    rule_id="power_transform_yeo_johnson",
                    target_columns=neg_skewed_cols,
                    message=f"Found {len(neg_skewed_cols)} skewed columns (with non-positive values).",
                    suggestion="Consider Yeo-Johnson transformation.",
                )
            )

        return recs


# Remove prefix here to allow flexible mounting in main.py
router = APIRouter(tags=["ML Pipeline"])

# --- Pydantic Models for API ---
# We mirror the dataclasses but use Pydantic for validation


class NodeConfigModel(BaseModel):
    node_id: str
    step_type: str
    params: Dict[str, Any] = {}
    inputs: List[str] = []


class PipelineConfigModel(BaseModel):
    pipeline_id: str
    nodes: List[NodeConfigModel]
    metadata: Dict[str, Any] = {}
    target_node_id: Optional[str] = None
    job_type: Optional[str] = (
        StepType.BASIC_TRAINING
    )  # "basic_training", "advanced_tuning", or "preview"


class RunPipelineResponse(BaseModel):
    message: str
    pipeline_id: str
    job_id: str
    job_ids: List[str] = []  # All jobs when parallel branches are detected


@router.post("/run", response_model=RunPipelineResponse)
async def run_pipeline(  # noqa: C901
    config: PipelineConfigModel,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Submit a pipeline for asynchronous execution via Celery or BackgroundTasks.

    When the graph contains multiple training nodes or a training node with
    ``execution_mode == "parallel"``, the pipeline is automatically partitioned
    into independent sub-pipelines. Each sub-pipeline gets its own job and
    runs concurrently. The response includes all ``job_ids``.
    """
    from backend.ml_pipeline.execution.graph_utils import (
        partition_parallel_pipeline,
        _split_connected_components,
    )

    pipeline_id = config.pipeline_id

    # --- Path Resolution Logic ---
    ingestion_service = DataIngestionService(db)
    resolved_s3_options = await resolve_pipeline_nodes(config.nodes, ingestion_service)
    # -----------------------------

    # Convert API models to internal dataclasses for partitioning
    internal_nodes = [
        NodeConfig(
            node_id=n.node_id,
            step_type=coerce_step_type(n.step_type),
            params=n.params,
            inputs=n.inputs,
        )
        for n in config.nodes
    ]
    internal_config = PipelineConfig(
        pipeline_id=pipeline_id,
        nodes=internal_nodes,
        metadata=config.metadata,
    )

    # Split disconnected subgraphs into separate experiment groups first,
    # then partition each group for parallel branches.
    components = _split_connected_components(internal_config)
    sub_pipelines: list[PipelineConfig] = []
    for comp in components:
        sub_pipelines.extend(partition_parallel_pipeline(comp))

    # When a specific target node was requested and the partitioner split
    # by multiple terminals, only run the branch containing that node.
    # This ensures clicking Train on node A doesn't also execute node B.
    if config.target_node_id and len(sub_pipelines) > 1:
        filtered = [
            sub
            for sub in sub_pipelines
            if any(n.node_id == config.target_node_id for n in sub.nodes)
        ]
        if filtered:
            sub_pipelines = filtered

    # Detect dataset_id from the first DATA_LOADER node
    dataset_id = "unknown"
    for node in config.nodes:
        if node.step_type == StepType.DATA_LOADER:
            dataset_id = node.params.get("dataset_id", "unknown")
            break

    settings = get_settings()
    all_job_ids: List[str] = []
    task_payloads: List[tuple] = []

    for sub in sub_pipelines:
        # Identify the terminal node for this sub-pipeline
        target_node_id = config.target_node_id
        terminal_types = {StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING, "data_preview"}
        for n in reversed(sub.nodes):
            if n.step_type in terminal_types:
                target_node_id = n.node_id
                break
        if not target_node_id and sub.nodes:
            target_node_id = sub.nodes[-1].node_id

        # Determine model type and job type from the terminal node
        model_type = "unknown"
        job_type = config.job_type or StepType.BASIC_TRAINING
        for n in sub.nodes:
            if n.node_id == target_node_id:
                if n.step_type == StepType.BASIC_TRAINING:
                    model_type = n.params.get("model_type", n.params.get("algorithm", "unknown"))
                    job_type = StepType.BASIC_TRAINING
                elif n.step_type == StepType.ADVANCED_TUNING:
                    model_type = n.params.get("algorithm", n.params.get("model_type", "unknown"))
                    job_type = StepType.ADVANCED_TUNING
                elif n.step_type == "data_preview":
                    model_type = "preview"
                    job_type = "preview"
                break

        # Build the per-branch graph snapshot. We persist *this branch's*
        # nodes (with the terminal's `inputs` already rewritten by the
        # partitioner to point only at this branch's parent in parallel
        # mode) instead of the full original config — otherwise the
        # Experiments comparison view walks back from a shared terminal
        # whose `inputs` still list every branch's parent and ends up
        # showing both branches' preprocessing chains in every column
        # (reported as "Path A and Path B both show every Encoding").
        branch_graph: Dict[str, Any] = {
            "pipeline_id": sub.pipeline_id,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "step_type": n.step_type,
                    "params": n.params,
                    "inputs": n.inputs,
                }
                for n in sub.nodes
            ],
            "metadata": sub.metadata,
        }

        # Create Job in DB
        job_id = await JobManager.create_job(
            session=db,
            pipeline_id=sub.pipeline_id,
            node_id=target_node_id or "unknown",
            job_type=cast(Literal["basic_training", "advanced_tuning", "preview"], job_type),
            dataset_id=dataset_id,
            model_type=model_type,
            graph=branch_graph,
        )
        all_job_ids.append(job_id)
        publish_job_event(
            JobEvent(event="created", job_id=job_id, status="queued", progress=0)
        )

        # Reuse the same dict shape for the Celery payload (storage_options
        # is added below; we don't persist it into the DB graph snapshot).
        sub_payload: Dict[str, Any] = dict(branch_graph)
        if resolved_s3_options:
            sub_payload["storage_options"] = resolved_s3_options

        # Trigger task
        if settings.USE_CELERY:
            task = run_pipeline_task.delay(job_id, sub_payload)
            # Persist the celery task id so cancel_job can revoke it later.
            try:
                await JobManager.attach_celery_task_id(db, job_id, task.id)
            except Exception:
                # Best-effort: never let metadata persistence block job submit.
                logger.warning("Failed to attach celery task id for job %s", job_id)
        else:
            task_payloads.append((job_id, sub_payload))

    # Non-Celery: run branches concurrently via ThreadPoolExecutor
    if not settings.USE_CELERY and task_payloads:
        if len(task_payloads) == 1:
            background_tasks.add_task(run_pipeline_task, *task_payloads[0])
        else:

            def _run_branches_concurrently(payloads: List[tuple]) -> None:
                with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
                    futures = [pool.submit(run_pipeline_task, jid, pl) for jid, pl in payloads]
                    for f in futures:
                        f.result()  # propagate exceptions per-branch via logging

            background_tasks.add_task(_run_branches_concurrently, task_payloads)

    is_parallel = len(all_job_ids) > 1
    message = (
        f"Parallel execution started: {len(all_job_ids)} branches"
        if is_parallel
        else "Pipeline execution started"
    )

    return RunPipelineResponse(
        message=message,
        pipeline_id=pipeline_id,
        job_id=all_job_ids[0],
        job_ids=all_job_ids,
    )


class PreviewResponse(BaseModel):
    pipeline_id: str
    status: str
    node_results: Dict[str, Any]
    # We return the preview data for the last node (or specific nodes)
    preview_data: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    # True row counts for each split key in `preview_data` (the rows in
    # `preview_data` are capped at 50 for transport). Mirrors the dict keys
    # of `preview_data`; for single-frame previews uses the synthetic
    # `_total` key.
    preview_totals: Optional[Dict[str, int]] = None
    # When the pipeline has multiple branches (parallel execution), per-branch
    # preview keyed by branch label (e.g. "Path A · Random Forest").
    # Each value matches the shape of preview_data above.
    branch_previews: Optional[Dict[str, Any]] = None
    # Per-branch true row counts (mirrors `branch_previews` keys).
    branch_preview_totals: Optional[Dict[str, Dict[str, int]]] = None
    # Per-branch list of node IDs that ran in that branch. Used by the
    # frontend to show only the relevant "applied steps" pills per tab.
    branch_node_ids: Optional[Dict[str, List[str]]] = None
    recommendations: List[Recommendation] = []
    # Advisory messages from the engine about merge semantics applied during
    # this preview (e.g. sibling fan-in detected). Empty list when nothing
    # noteworthy happened.
    merge_warnings: List[Dict[str, Any]] = []


def _prettify_model_type(model_type: str) -> str:
    """Convert snake_case model id to readable name (mirror frontend useBranchColors)."""
    if not model_type:
        return ""
    cleaned = model_type
    for suffix in ("_classifier", "_regressor"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return " ".join(w.capitalize() for w in cleaned.split("_"))


def _branch_label(index: int, sub_config: PipelineConfig, dup_suffix: str = "") -> str:
    """Build a 'Path A · <suffix>' label for a branch sub-pipeline.

    Suffix priority (most specific first):
      1. Model name when a training/tuning node is present.
      2. The leaf node's ``_display_name`` (canvas label, sent by the
         frontend converter) so preview tabs read "Encoding" / "Scaling"
         rather than the raw step type "LabelEncoder" / "RobustScaler".
      3. Friendly version of the leaf node's step_type as last resort.

    ``dup_suffix`` is appended to disambiguate sibling branches that share
    the same model_type (e.g. two XGBoost training nodes get ``#1`` / ``#2``)
    so the tab labels match the canvas edge labels exactly.
    """
    letter = chr(ord("A") + index)
    model_type = ""
    leaf_step = ""
    leaf_display = ""
    for n in sub_config.nodes:
        if n.step_type in {StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING}:
            model_type = n.params.get("model_type") or n.params.get("algorithm") or ""
    if sub_config.nodes:
        leaf = sub_config.nodes[-1]
        if leaf.step_type not in {StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING}:
            leaf_step = str(leaf.step_type)
            leaf_display = str(leaf.params.get("_display_name") or "")
    pretty = _prettify_model_type(str(model_type))
    suffix_tail = f" {dup_suffix}" if dup_suffix else ""
    if pretty:
        return f"Path {letter} · {pretty}{suffix_tail}"
    if leaf_display:
        return f"Path {letter} · {leaf_display}{suffix_tail}"
    if leaf_step:
        # Convert PascalCase / snake_case step types to a readable suffix:
        # "StandardScaler" → "Standard Scaler", "data_loader" → "Data Loader".
        if "_" in leaf_step:
            friendly = " ".join(w.capitalize() for w in leaf_step.split("_"))
        else:
            import re

            friendly = re.sub(r"(?<!^)(?=[A-Z])", " ", leaf_step).strip()
        return f"Path {letter} · {friendly}{suffix_tail}"
    return f"Path {letter}"


class SavedPipelineModel(BaseModel):
    name: str
    description: Optional[str] = None
    graph: Dict[str, Any]


# --- Endpoints ---


@router.post("/save/{dataset_id}")
async def save_pipeline(
    dataset_id: str,
    payload: SavedPipelineModel,
    session: AsyncSession = Depends(get_async_session),
):
    """Saves the pipeline configuration (supports DB or JSON based on config)."""
    settings = get_settings()

    if settings.PIPELINE_STORAGE_TYPE == "json":
        storage_dir = settings.PIPELINE_STORAGE_PATH
        os.makedirs(storage_dir, exist_ok=True)
        file_path = os.path.join(storage_dir, f"{dataset_id}.json")
        try:
            with open(file_path, "w") as f:
                json.dump(payload.model_dump(), f, indent=2)
            return {"status": "success", "id": dataset_id, "storage": "json"}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to save pipeline to JSON: {str(e)}"
            )

    # Default: Database Storage
    try:
        # Check if pipeline exists for this dataset
        stmt = select(FeatureEngineeringPipeline).where(
            FeatureEngineeringPipeline.dataset_source_id == dataset_id,
            FeatureEngineeringPipeline.is_active,
        )
        result = await session.execute(stmt)
        existing_pipeline = result.scalar_one_or_none()

        if existing_pipeline:
            # Update existing
            cast(Any, existing_pipeline).graph = payload.graph
            cast(Any, existing_pipeline).name = payload.name
            if payload.description:
                cast(Any, existing_pipeline).description = payload.description
            # existing_pipeline.updated_at is handled by mixin
        else:
            # Create new
            new_pipeline = FeatureEngineeringPipeline(
                dataset_source_id=dataset_id,
                name=payload.name,
                description=payload.description,
                graph=payload.graph,
                is_active=True,
            )
            session.add(new_pipeline)

        await session.commit()
        return {"status": "success", "id": dataset_id, "storage": "database"}
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save pipeline: {str(e)}")


@router.get("/load/{dataset_id}")
async def load_pipeline(dataset_id: str, session: AsyncSession = Depends(get_async_session)):
    """Loads the pipeline configuration (supports DB or JSON based on config)."""
    settings = get_settings()

    if settings.PIPELINE_STORAGE_TYPE == "json":
        storage_dir = settings.PIPELINE_STORAGE_PATH
        file_path = os.path.join(storage_dir, f"{dataset_id}.json")
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load pipeline from JSON: {str(e)}"
            )

    # Default: Database Storage
    try:
        stmt = select(FeatureEngineeringPipeline).where(
            FeatureEngineeringPipeline.dataset_source_id == dataset_id,
            FeatureEngineeringPipeline.is_active,
        )
        result = await session.execute(stmt)
        pipeline = result.scalar_one_or_none()

        if not pipeline:
            return None

        return pipeline.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load pipeline: {str(e)}")


@router.post("/preview", response_model=PreviewResponse)
async def preview_pipeline(  # noqa: C901
    config: PipelineConfigModel, session: AsyncSession = Depends(get_async_session)
):
    """
    Runs the pipeline in Preview Mode:
    - Uses a temporary artifact store (cleaned up after request).
    - Resolves dataset paths from IDs.
    """

    # Resolve dataset paths

    # 1. Create Temporary Artifact Store
    temp_dir = tempfile.mkdtemp(prefix="skyulf_preview_")
    artifact_store = LocalArtifactStore(temp_dir)

    # Resolve paths and credentials for Preview (Async)
    ingestion_service = DataIngestionService(session)
    resolved_s3_options = await resolve_pipeline_nodes(config.nodes, ingestion_service)

    try:
        logger.debug(f"Preview request received with {len(config.nodes)} nodes")
        for n in config.nodes:
            logger.debug(f"Node {n.node_id} - Type: {n.step_type}")

        # 2. Adapt Config for Preview
        # Convert Pydantic to Dataclass
        nodes = []
        for node in config.nodes:
            params = node.params.copy()

            # Force sampling for Data Loader
            if node.step_type == StepType.DATA_LOADER:
                params["sample"] = True
                params["limit"] = 1000  # Default preview limit

            nodes.append(
                NodeConfig(
                    node_id=node.node_id,
                    step_type=coerce_step_type(node.step_type),
                    params=params,
                    inputs=node.inputs,
                )
            )

        pipeline_config = PipelineConfig(
            pipeline_id=config.pipeline_id, nodes=nodes, metadata=config.metadata
        )

        # 3. Run Engine

        # IMPORTANT: Pass session to create_catalog_from_options so SmartCatalog can resolve IDs
        # But session here is AsyncSession, SmartCatalog expects Sync Session (usually)
        # However, SmartCatalog uses self.session.query() which is sync-style ORM usage.
        # If we pass AsyncSession, query() won't work.
        # We need a sync session for SmartCatalog.

        if db_engine.sync_session_factory is None:
            raise HTTPException(status_code=500, detail="Database not initialized")

        sync_session = db_engine.sync_session_factory()

        # Partition the pipeline for preview. We want one preview tab per
        # logical experiment, mirroring "Run All Experiments". Strategy:
        #   1. Use partition_parallel_pipeline() — gives one sub per
        #      (training_node × input_branch) combination. This handles the
        #      case where one data node feeds multiple training nodes.
        #   2. If no training nodes exist, fall back to partition_for_preview()
        #      which splits by data leaves so parallel preprocessing chains
        #      still get separate tabs.
        # Either way, training/tuning nodes are stripped before execution
        # because preview never fits models.
        from backend.ml_pipeline.execution.graph_utils import (
            partition_for_preview,
            partition_parallel_pipeline,
        )

        training_types = {StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING}
        has_training = any(n.step_type in training_types for n in nodes)

        if has_training:
            training_subs = partition_parallel_pipeline(pipeline_config)

            def _strip(sub: PipelineConfig) -> PipelineConfig:
                stripped = [n for n in sub.nodes if n.step_type not in training_types]
                return PipelineConfig(
                    pipeline_id=sub.pipeline_id,
                    nodes=stripped,
                    metadata=sub.metadata,
                )

            paired_subs = [(orig, _strip(orig)) for orig in training_subs]

            # Also include preview-only branches (data leaves that don't feed
            # any training terminal). Without this, a canvas mixing a training
            # pipeline with one or more dangling preprocessing chains would
            # silently drop the dangling chains from Run Preview — users only
            # saw the training branch's data and assumed the others vanished.
            covered_node_ids: set[str] = set()
            for orig, _runnable in paired_subs:
                for n in orig.nodes:
                    covered_node_ids.add(n.node_id)
            preview_only_subs = partition_for_preview(pipeline_config)
            for sub in preview_only_subs:
                if not sub.nodes:
                    continue
                leaf = sub.nodes[-1]
                # Skip sub-pipelines whose leaf is already produced by a
                # training branch (avoids duplicate tabs for the same data).
                if leaf.node_id in covered_node_ids:
                    continue
                paired_subs.append((sub, sub))
        else:
            preview_subs = partition_for_preview(pipeline_config)
            # No training → no separate label source; reuse the runnable sub.
            paired_subs = [(sub, sub) for sub in preview_subs]

        try:
            catalog = create_catalog_from_options(
                resolved_s3_options, config.nodes, session=sync_session
            )
            engine = PipelineEngine(artifact_store, catalog=catalog)
            # Single shared artifact store across branches deduplicates work
            # for ancestor nodes that appear in multiple sub-pipelines.
            sub_results = [(orig, runnable, engine.run(runnable)) for orig, runnable in paired_subs]
        finally:
            sync_session.close()

        # ---- Preview extraction helpers ----

        def to_records(df):
            if isinstance(df, pd.DataFrame):
                return json.loads(df.head(50).to_json(orient="records"))
            if isinstance(df, pd.Series):
                return json.loads(df.to_frame().head(50).to_json(orient="records"))
            try:
                import polars as pl

                if isinstance(df, pl.DataFrame):
                    return json.loads(df.head(50).to_pandas().to_json(orient="records"))
                if isinstance(df, pl.Series):
                    return json.loads(df.head(50).to_pandas().to_frame().to_json(orient="records"))
            except ImportError:
                pass
            return []

        def count_rows(df) -> int:
            # True total row count for the underlying artifact, regardless of
            # the 50-row preview cap applied by `to_records`. Used so the UI
            # can show the real dataset size in tab badges.
            if df is None:
                return 0
            if isinstance(df, pd.DataFrame):
                return int(df.shape[0])
            if isinstance(df, pd.Series):
                return int(df.shape[0])
            try:
                import polars as pl

                if isinstance(df, (pl.DataFrame, pl.Series)):
                    return int(df.shape[0])
            except ImportError:
                pass
            try:
                return int(len(df))
            except Exception:
                return 0

        def process_xy(xy_tuple, prefix):
            X, y = xy_tuple
            return {f"{prefix}_X": to_records(X), f"{prefix}_y": to_records(y)}

        def process_xy_totals(xy_tuple, prefix):
            X, y = xy_tuple
            return {f"{prefix}_X": count_rows(X), f"{prefix}_y": count_rows(y)}

        def to_pandas_safe(df):
            if isinstance(df, pd.DataFrame):
                return df
            try:
                import polars as pl

                if isinstance(df, pl.DataFrame):
                    return df.to_pandas()
            except ImportError:
                pass
            return None

        def pick_target_node_id(node_list) -> Optional[str]:
            """Pick the last data-bearing node to preview.

            Training/tuning nodes are stripped before execution, so the last
            node in a runnable sub-pipeline is already the right preview
            target. Falls back to the input of a training node for callers
            that pass an unstripped node list.
            """
            if not node_list:
                return None
            target = node_list[-1]
            target_id = target.node_id
            if target.step_type in [
                StepType.BASIC_TRAINING,
                StepType.ADVANCED_TUNING,
            ]:
                if target.inputs:
                    target_id = target.inputs[0]
            return target_id

        def extract_preview(target_node_id: Optional[str]):
            """Return (preview_data, totals, df_for_analysis) for a single artifact.

            `totals` mirrors the shape of `preview_data`:
            - dict of split-name → row count when `preview_data` is a dict;
            - a single int wrapped under the `_total` key when `preview_data`
              is a list (single-frame case).
            """
            preview_data: Any = {}
            totals: Dict[str, int] = {}
            df_for_analysis = None
            if not target_node_id or not artifact_store.exists(target_node_id):
                return preview_data, totals, df_for_analysis

            artifact = artifact_store.load(target_node_id)
            logger.debug(f"Loaded artifact for node {target_node_id}. Type: {type(artifact)}")

            is_polars = False
            try:
                import polars as pl

                if isinstance(artifact, pl.DataFrame):
                    is_polars = True
            except ImportError:
                pass

            if is_polars:
                preview_data = to_records(artifact)
                totals = {"_total": count_rows(artifact)}
                df_for_analysis = to_pandas_safe(artifact)
            elif isinstance(artifact, pd.DataFrame):
                preview_data = json.loads(artifact.head(50).to_json(orient="records"))
                totals = {"_total": count_rows(artifact)}
                df_for_analysis = artifact
            elif isinstance(artifact, SplitDataset):
                preview_data = {}
                if isinstance(artifact.train, tuple):
                    preview_data.update(process_xy(artifact.train, "train"))
                    totals.update(process_xy_totals(artifact.train, "train"))
                    df_for_analysis = to_pandas_safe(artifact.train[0])
                else:
                    preview_data["train"] = to_records(artifact.train)
                    totals["train"] = count_rows(artifact.train)
                    df_for_analysis = to_pandas_safe(artifact.train)
                if isinstance(artifact.test, tuple):
                    preview_data.update(process_xy(artifact.test, "test"))
                    totals.update(process_xy_totals(artifact.test, "test"))
                else:
                    preview_data["test"] = to_records(artifact.test)
                    totals["test"] = count_rows(artifact.test)
                if artifact.validation is not None:
                    if isinstance(artifact.validation, tuple):
                        preview_data.update(process_xy(artifact.validation, "validation"))
                        totals.update(process_xy_totals(artifact.validation, "validation"))
                    else:
                        preview_data["validation"] = to_records(artifact.validation)
                        totals["validation"] = count_rows(artifact.validation)
            elif isinstance(artifact, tuple) and len(artifact) == 2:
                X, y = artifact
                preview_data = {"X": to_records(X), "y": to_records(y)}
                totals = {"X": count_rows(X), "y": count_rows(y)}
                df_for_analysis = to_pandas_safe(X)
            elif (
                isinstance(artifact, dict)
                and "train" in artifact
                and isinstance(artifact["train"], tuple)
            ):
                preview_data = {}
                preview_data.update(process_xy(artifact["train"], "train"))
                totals.update(process_xy_totals(artifact["train"], "train"))
                df_for_analysis = to_pandas_safe(artifact["train"][0])
                if "test" in artifact:
                    preview_data.update(process_xy(artifact["test"], "test"))
                    totals.update(process_xy_totals(artifact["test"], "test"))
                if "validation" in artifact:
                    preview_data.update(process_xy(artifact["validation"], "validation"))
                    totals.update(process_xy_totals(artifact["validation"], "validation"))

            return preview_data, totals, df_for_analysis

        # 4. Aggregate per-branch previews
        preview_data: Any = {}
        preview_totals: Dict[str, int] = {}
        branch_previews: Optional[Dict[str, Any]] = None
        branch_preview_totals: Optional[Dict[str, Dict[str, int]]] = None
        branch_node_ids: Optional[Dict[str, List[str]]] = None
        recommendations: List[Recommendation] = []
        combined_node_results: Dict[str, Any] = {}
        # Aggregate status: failed > running > success
        agg_status = "success"
        first_pdf = None

        is_multi = len(sub_results) > 1
        if is_multi:
            branch_previews = {}
            branch_preview_totals = {}
            branch_node_ids = {}

        # Pre-compute "#N" disambiguators for training terminals that share
        # the same model_type (mirrors the canvas edge label scheme in
        # useBranchColors). The suffix is keyed on the *terminal node_id*,
        # not the branch index — so two branches feeding the same XGBoost
        # terminal both render as "#1", and a second XGBoost terminal (with
        # any number of branches) becomes "#2". Without this, parallel
        # branches of the same terminal incorrectly got "#1" / "#2" / "#3"
        # despite originating from the same training node on the canvas.
        # Preview-only branches (no model_type) skip this and stay unsuffixed.
        terminals_by_model: Dict[str, List[str]] = {}
        terminal_id_per_branch: Dict[int, str] = {}
        for i, (orig_sub, _runnable, _res) in enumerate(sub_results):
            mt = ""
            term_id = ""
            for n in orig_sub.nodes:
                if n.step_type in {StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING}:
                    mt = str(n.params.get("model_type") or n.params.get("algorithm") or "")
                    term_id = n.node_id
                    break
            if mt and term_id:
                terminal_id_per_branch[i] = term_id
                ids = terminals_by_model.setdefault(mt, [])
                if term_id not in ids:
                    ids.append(term_id)
        terminal_suffix: Dict[str, str] = {}
        for _mt, ids in terminals_by_model.items():
            if len(ids) < 2:
                continue
            for n_, term_id in enumerate(ids, start=1):
                terminal_suffix[term_id] = f"#{n_}"
        dup_suffix_by_branch: Dict[int, str] = {
            branch_idx: terminal_suffix[term_id]
            for branch_idx, term_id in terminal_id_per_branch.items()
            if term_id in terminal_suffix
        }

        for idx, (orig_sub, runnable_sub, sub_result) in enumerate(sub_results):
            for k, v in sub_result.node_results.items():
                combined_node_results[k] = v.__dict__
            if sub_result.status != "success":
                agg_status = sub_result.status
            if sub_result.status == "success" and runnable_sub.nodes:
                target_id = pick_target_node_id(runnable_sub.nodes)
                pdata, ptotals, pdf = extract_preview(target_id)
                if idx == 0:
                    preview_data = pdata
                    preview_totals = ptotals
                    first_pdf = pdf
                if (
                    is_multi
                    and branch_previews is not None
                    and branch_node_ids is not None
                    and branch_preview_totals is not None
                ):
                    # Label uses the ORIGINAL sub (training node included) so
                    # the model name shows up in the tab.
                    label = _branch_label(idx, orig_sub, dup_suffix_by_branch.get(idx, ""))
                    branch_previews[label] = pdata
                    branch_preview_totals[label] = ptotals
                    branch_node_ids[label] = [n.node_id for n in runnable_sub.nodes]

        # 5. Recommendations from first branch's data only (avoid heavy work).
        if first_pdf is not None:
            try:
                profile = DataProfiler.generate_profile(first_pdf)
                advisor = AdvisorEngine()
                recommendations = advisor.analyze(profile)
            except Exception as e:
                logger.warning("Error generating recommendations: %s", e)

        # Dedup merge warnings across branches: when N parallel sub-pipelines
        # run independently (Path A / Path B), each engine emits the same
        # advisory for the shared upstream merge node. Collapse duplicates so
        # the UI shows one row per logically distinct merge.
        all_warnings = [
            w for _orig, _runnable, res in sub_results for w in getattr(res, "merge_warnings", [])
        ]
        seen_warning_keys: set = set()
        deduped_warnings: List[Dict[str, Any]] = []
        for w in all_warnings:
            key = (
                w.get("node_id"),
                w.get("kind"),
                tuple(sorted(w.get("inputs") or ())),
                tuple(w.get("overlap_columns") or ()),
                tuple(w.get("dropped_columns") or ()),
                w.get("part"),
            )
            if key in seen_warning_keys:
                continue
            seen_warning_keys.add(key)
            deduped_warnings.append(w)

        return PreviewResponse(
            pipeline_id=pipeline_config.pipeline_id,
            status=agg_status,
            node_results=combined_node_results,
            preview_data=preview_data,
            preview_totals=preview_totals,
            branch_previews=branch_previews,
            branch_preview_totals=branch_preview_totals,
            branch_node_ids=branch_node_ids,
            recommendations=recommendations,
            merge_warnings=deduped_warnings,
        )

    except Exception:
        logger.exception("Pipeline preview failed")
        raise HTTPException(status_code=500, detail="Pipeline preview failed")
    finally:
        # 5. Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job_status(job_id: str, session: AsyncSession = Depends(get_async_session)):
    """
    Returns the status of a background job.
    """
    job = await JobManager.get_job(session, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, session: AsyncSession = Depends(get_async_session)):
    """
    Cancels a running or queued job.
    """
    success = await JobManager.cancel_job(session, job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job could not be cancelled (maybe it's already finished or doesn't exist)",
        )
    publish_job_event(JobEvent(event="status", job_id=job_id, status="cancelled"))
    return {"message": "Job cancelled successfully"}


@router.post("/jobs/{job_id}/promote")
async def promote_job(job_id: str, session: AsyncSession = Depends(get_async_session)):
    """Marks a completed job as the promoted winner."""
    success = await JobManager.promote_job(session, job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job could not be promoted (must be completed and exist)",
        )
    return {"message": "Job promoted successfully"}


@router.delete("/jobs/{job_id}/promote")
async def unpromote_job(job_id: str, session: AsyncSession = Depends(get_async_session)):
    """Removes promotion from a job."""
    success = await JobManager.unpromote_job(session, job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job unPromoted successfully"}


@router.get("/jobs/{job_id}/evaluation")
async def get_job_evaluation(  # noqa: C901
    job_id: str, session: AsyncSession = Depends(get_async_session)
):
    """Retrieves the raw evaluation data (y_true, y_pred) for a job."""
    try:
        return await EvaluationService.get_job_evaluation(session, job_id)
    except ValueError as e:
        # Map ValueError to 404 or 400 depending on message, or just 404/400 generic
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("Failed to retrieve evaluation for job %s", job_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve evaluation data")


@router.get("/jobs", response_model=List[JobInfo])
async def list_jobs(
    limit: int = 50,
    skip: int = 0,
    job_type: Optional[Literal["training", "tuning"]] = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Lists recent jobs.
    """
    return await JobManager.list_jobs(session, limit, skip, job_type)


@router.get("/jobs/tuning/latest/{node_id}", response_model=Optional[JobInfo])
async def get_latest_tuning_job(node_id: str, session: AsyncSession = Depends(get_async_session)):
    """
    Retrieves the latest completed tuning job for a specific node.
    """
    return await JobManager.get_latest_tuning_job_for_node(session, node_id)


@router.get("/jobs/tuning/best/{model_type}", response_model=Optional[JobInfo])
async def get_best_tuning_job_model(
    model_type: str, session: AsyncSession = Depends(get_async_session)
):
    """
    Retrieves the latest completed tuning job for a specific model type.
    """
    return await JobManager.get_best_tuning_job_for_model(session, model_type)


@router.get("/jobs/tuning/history/{model_type}", response_model=List[JobInfo])
async def get_tuning_jobs_history(
    model_type: str, session: AsyncSession = Depends(get_async_session)
):
    """
    Retrieves a history of completed tuning jobs for a specific model type.
    """
    return await JobManager.get_tuning_jobs_for_model(session, model_type)


@router.get("/stats", response_model=Dict[str, int])
async def get_system_stats(session: AsyncSession = Depends(get_async_session)):
    """
    Returns high-level system statistics for the dashboard.
    """

    # Execute queries in parallel or sequence
    # 1. Total Jobs (Training + Tuning)
    training_count = await session.scalar(select(func.count(BasicTrainingJob.id)))
    tuning_count = await session.scalar(select(func.count(AdvancedTuningJob.id)))

    # 2. Active Deployments
    deployment_count = await session.scalar(
        select(func.count(Deployment.id)).where(Deployment.is_active)
    )

    # 3. Data Sources (Only successful ones)
    datasource_count = await session.scalar(
        select(func.count(DataSource.id)).where(DataSource.test_status == "success")
    )

    return {
        "total_jobs": (training_count or 0) + (tuning_count or 0),
        "active_deployments": deployment_count or 0,
        "data_sources": datasource_count or 0,
        "training_jobs": training_count or 0,
        "tuning_jobs": tuning_count or 0,
    }


@router.get("/registry", response_model=List[RegistryItem])
def get_node_registry():
    """
    Returns the list of available pipeline nodes (transformers, models, etc.).
    """
    return NodeRegistry.get_all_nodes()


@router.get("/datasets/{dataset_id}/schema", response_model=AnalysisProfile)
async def get_dataset_schema(dataset_id: int, session: AsyncSession = Depends(get_async_session)):
    """
    Returns the schema (columns, types, stats) of a dataset.
    Uses the DataProfiler.
    """
    ingestion_service = DataIngestionService(session)
    ds = await ingestion_service.get_source(dataset_id)

    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    # Check if we have a cached profile in metadata
    if ds.source_metadata and "profile" in ds.source_metadata:
        try:
            cached_profile = ds.source_metadata["profile"]
            columns = {}
            for col_name, stats in cached_profile.get("columns", {}).items():
                # Map stats to ColumnProfile
                dtype = str(stats.get("type", "unknown"))
                col_type = "unknown"
                if any(x in dtype for x in ["Int", "Float", "Decimal"]):
                    col_type = "numeric"
                elif any(x in dtype for x in ["Utf8", "String", "Categorical", "Object"]):
                    col_type = "categorical"
                elif "Date" in dtype or "Time" in dtype:
                    col_type = "datetime"
                elif "Bool" in dtype:
                    col_type = "boolean"

                columns[col_name] = {
                    "name": col_name,
                    "dtype": dtype,
                    "column_type": col_type,
                    "missing_count": stats.get("null_count", 0),
                    "missing_ratio": stats.get("null_percentage", 0) / 100.0,
                    "unique_count": stats.get("unique_count", 0),
                    "min_value": stats.get("min"),
                    "max_value": stats.get("max"),
                    "mean_value": stats.get("mean"),
                    "std_value": stats.get("std"),
                }

            return {
                "row_count": cached_profile.get("row_count", 0),
                "column_count": cached_profile.get("column_count", 0),
                "duplicate_row_count": cached_profile.get("duplicate_rows", 0),
                "columns": columns,
            }
        except Exception as e:
            logger.warning(f"Failed to parse cached profile for {dataset_id}: {e}")
            # Fallback to loading file

    try:
        # Resolve path
        ds_dict = {
            "connection_info": ds.config,
            "file_path": ds.config.get("file_path") if ds.config else None,
        }
        path = extract_file_path_from_source(ds_dict)

        if not path:
            raise HTTPException(
                status_code=400,
                detail=f"Could not resolve path for dataset {dataset_id}",
            )

        # Load sample
        catalog = FileSystemCatalog()
        df = catalog.load(str(path), limit=1000)

        # Profile
        profile = DataProfiler.generate_profile(df)
        return profile

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to profile dataset: {str(e)}")


@router.get("/hyperparameters/{model_type}")
def get_model_hyperparameters(model_type: str):
    """
    Returns the list of tunable hyperparameters for a specific model type.
    """
    return get_hyperparameters(model_type)


@router.get("/hyperparameters/{model_type}/defaults")
def get_model_default_search_space(model_type: str):
    """
    Returns the default search space for a specific model type.
    """
    return get_default_search_space(model_type)


@router.get("/datasets/list", response_model=List[Dict[str, Any]])
async def list_datasets(session: AsyncSession = Depends(get_async_session)):
    """
    Returns a simple list of available datasets for filtering.
    """
    stmt = select(DataSource.source_id, DataSource.name).where(DataSource.is_active)
    result = await session.execute(stmt)
    return [{"id": row.source_id, "name": row.name} for row in result.all()]
