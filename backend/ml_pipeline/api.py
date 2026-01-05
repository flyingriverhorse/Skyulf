import json
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Literal, Optional, Union, cast

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from skyulf.data.dataset import SplitDataset
from skyulf.modeling.hyperparameters import (
    get_default_search_space,
    get_hyperparameters,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.data_ingestion.service import DataIngestionService
from backend.database.engine import get_async_session
from backend.database.models import (
    FeatureEngineeringPipeline,
    AdvancedTuningJob,
    BasicTrainingJob,
)
from backend.ml_pipeline.constants import StepType
from backend.ml_pipeline.tasks import run_pipeline_task
from backend.utils.file_utils import extract_file_path_from_source
from backend.ml_pipeline.services.evaluation_service import EvaluationService

from .artifacts.local import LocalArtifactStore
from .execution.engine import PipelineEngine

# from .data.profiler import DataProfiler
# from .recommendations.schemas import Recommendation, AnalysisProfile
from .execution.jobs import JobInfo, JobManager
from .execution.schemas import NodeConfig, PipelineConfig

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
                    float(cast(Union[float, int], df[col].min()))
                    if is_numeric
                    else None
                ),
                "max_value": (
                    float(cast(Union[float, int], df[col].max()))
                    if is_numeric
                    else None
                ),
                "mean_value": (
                    float(cast(Union[float, int], df[col].mean()))
                    if is_numeric
                    else None
                ),
                "std_value": (
                    float(cast(Union[float, int], df[col].std()))
                    if is_numeric
                    else None
                ),
                "skewness": (
                    float(cast(Union[float, int], df[col].skew()))
                    if is_numeric
                    else None
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
        missing_cols = [
            col for col, stats in profile.columns.items() if stats["missing_count"] > 0
        ]
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
            col
            for col, stats in profile.columns.items()
            if stats["missing_ratio"] > 0.5
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
            if (
                stats["column_type"] == "numeric"
                and stats["std_value"]
                and stats["std_value"] > 0
            ):
                mean = stats["mean_value"]
                std = stats["std_value"]
                if (stats["max_value"] > mean + 3 * std) or (
                    stats["min_value"] < mean - 3 * std
                ):
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
    job_type: Optional[str] = "basic_training"  # "basic_training", "advanced_tuning", or "preview"


class RunPipelineResponse(BaseModel):
    message: str
    pipeline_id: str
    job_id: str


@router.post("/run", response_model=RunPipelineResponse)
async def run_pipeline(  # noqa: C901
    config: PipelineConfigModel,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Submit a pipeline for asynchronous execution via Celery or BackgroundTasks.
    """
    # Extract details for Job creation
    pipeline_id = config.pipeline_id
    target_node_id = config.target_node_id

    # Find target node to get model type if possible
    model_type = "unknown"
    dataset_id = "unknown"

    # Simple heuristic: find the last node if target not specified
    if not target_node_id and config.nodes:
        target_node_id = config.nodes[-1].node_id

    # Try to find model type from target node params
    for node in config.nodes:
        if node.node_id == target_node_id:
            if node.step_type == "trainer" or node.step_type == "basic_training":
                model_type = node.params.get("model_type", "unknown")
            elif node.step_type == "advanced_tuning":
                # For tuning nodes, model_type might be in params directly or inside tuning_config?
                # Usually it's 'algorithm' or 'model_type' in params
                model_type = node.params.get("algorithm") or node.params.get(
                    "model_type", "unknown"
                )
            elif node.step_type == "data_preview":
                model_type = "preview"

        # Try to find dataset_id from data_loader node
        if node.step_type == "data_loader":
            dataset_id = node.params.get("dataset_id", "unknown")

    # --- Path Resolution Logic ---
    from backend.ml_pipeline.resolution import resolve_pipeline_nodes
    
    ingestion_service = DataIngestionService(db)
    resolved_s3_options = await resolve_pipeline_nodes(config.nodes, ingestion_service)
    # -----------------------------

    # Create Job in DB
    job_id = await JobManager.create_job(
        session=db,
        pipeline_id=pipeline_id,
        node_id=target_node_id or "unknown",
        job_type=cast(Literal["basic_training", "advanced_tuning", "preview"], config.job_type),
        dataset_id=dataset_id,
        model_type=model_type,
        graph=config.model_dump(),
    )

    # Trigger Task
    settings = get_settings()
    
    # Pass resolved options to the task so it can init S3Catalog if needed
    job_payload = config.model_dump()
    if resolved_s3_options:
        job_payload["storage_options"] = resolved_s3_options

    if settings.USE_CELERY:
        run_pipeline_task.delay(job_id, job_payload)
    else:
        # Run in background thread if Celery is disabled
        background_tasks.add_task(run_pipeline_task, job_id, job_payload)

    return RunPipelineResponse(
        message="Pipeline execution started", pipeline_id=pipeline_id, job_id=job_id
    )


class PreviewResponse(BaseModel):
    pipeline_id: str
    status: str
    node_results: Dict[str, Any]
    # We return the preview data for the last node (or specific nodes)
    preview_data: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    recommendations: List[Recommendation] = []


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
                json.dump(payload.dict(), f, indent=2)
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
        raise HTTPException(
            status_code=500, detail=f"Failed to save pipeline: {str(e)}"
        )


@router.get("/load/{dataset_id}")
async def load_pipeline(
    dataset_id: str, session: AsyncSession = Depends(get_async_session)
):
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
        raise HTTPException(
            status_code=500, detail=f"Failed to load pipeline: {str(e)}"
        )


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
    from backend.ml_pipeline.resolution import resolve_pipeline_nodes
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
            if node.step_type == "data_loader":
                params["sample"] = True
                params["limit"] = 1000  # Default preview limit

            nodes.append(
                NodeConfig(
                    node_id=node.node_id,
                    step_type=cast(
                        Literal[
                            "data_loader",
                            "feature_engineering",
                            "basic_training",
                            "advanced_tuning",
                        ],
                        node.step_type,
                    ),
                    params=params,
                    inputs=node.inputs,
                )
            )

        pipeline_config = PipelineConfig(
            pipeline_id=config.pipeline_id, nodes=nodes, metadata=config.metadata
        )

        # 3. Run Engine
        from backend.data.catalog import create_catalog_from_options
        
        # IMPORTANT: Pass session to create_catalog_from_options so SmartCatalog can resolve IDs
        # But session here is AsyncSession, SmartCatalog expects Sync Session (usually)
        # However, SmartCatalog uses self.session.query() which is sync-style ORM usage.
        # If we pass AsyncSession, query() won't work.
        # We need a sync session for SmartCatalog.
        
        from backend.database.engine import sync_session_factory
        
        if sync_session_factory is None:
            raise HTTPException(status_code=500, detail="Database not initialized")
            
        sync_session = sync_session_factory()
        
        try:
            catalog = create_catalog_from_options(resolved_s3_options, config.nodes, session=sync_session)
            engine = PipelineEngine(artifact_store, catalog=catalog)
            result = engine.run(pipeline_config)
        finally:
            sync_session.close()

        # 4. Extract Preview Data & Generate Recommendations
        preview_data = {}
        recommendations = []

        if result.status == "success" and config.nodes:
            # Determine which node's output to preview
            target_node = config.nodes[-1]
            target_node_id = target_node.node_id

            # If the last node is a modeling node, we can't preview the model object as data.
            # Instead, we preview the input data that went into the model.
            if target_node.step_type in [StepType.BASIC_TRAINING, StepType.ADVANCED_TUNING]:
                if target_node.inputs:
                    # Use the input node's output
                    target_node_id = target_node.inputs[0]
                    logger.info(
                        f"Last node is {target_node.step_type}, previewing input node {target_node_id} instead"
                    )

            if artifact_store.exists(target_node_id):
                artifact = artifact_store.load(target_node_id)

                # Debug logging
                logger.debug(
                    f"Loaded artifact for node {target_node_id}. Type: {type(artifact)}"
                )
                if isinstance(artifact, SplitDataset):
                    logger.debug(f"SplitDataset Train Type: {type(artifact.train)}")

                df_for_analysis = None

                # Helper to convert DataFrame/Series (Pandas or Polars) to records list
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

                # Helper to process (X, y) tuple
                def process_xy(xy_tuple, prefix):
                    X, y = xy_tuple
                    return {
                        f"{prefix}_X": to_records(X),
                        f"{prefix}_y": to_records(y),
                    }
                
                # Helper to ensure df_for_analysis is Pandas
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

                # Handle different artifact types
                # Check for Polars DataFrame
                is_polars = False
                try:
                    import polars as pl
                    if isinstance(artifact, pl.DataFrame):
                        is_polars = True
                except ImportError:
                    pass

                if is_polars:
                    logger.debug("Handling Polars DataFrame artifact")
                    preview_data = to_records(artifact)
                    df_for_analysis = to_pandas_safe(artifact)

                elif isinstance(artifact, pd.DataFrame):
                    logger.debug("Handling DataFrame artifact")
                    preview_data = json.loads(
                        artifact.head(50).to_json(orient="records")
                    )
                    df_for_analysis = artifact
                elif isinstance(artifact, SplitDataset):
                    logger.debug("Handling SplitDataset artifact")
                    preview_data = {}

                    # Handle Train
                    if isinstance(artifact.train, tuple):
                        logger.debug("Train is tuple")
                        preview_data.update(process_xy(artifact.train, "train"))
                        df_for_analysis = to_pandas_safe(artifact.train[0])
                    else:
                        logger.debug("Train is DataFrame")
                        preview_data["train"] = to_records(artifact.train)
                        df_for_analysis = to_pandas_safe(artifact.train)

                    # Handle Test
                    if isinstance(artifact.test, tuple):
                        preview_data.update(process_xy(artifact.test, "test"))
                    else:
                        preview_data["test"] = to_records(artifact.test)

                    # Handle Validation
                    if artifact.validation is not None:
                        if isinstance(artifact.validation, tuple):
                            preview_data.update(
                                process_xy(artifact.validation, "validation")
                            )
                        else:
                            preview_data["validation"] = to_records(artifact.validation)

                elif isinstance(artifact, tuple) and len(artifact) == 2:
                    logger.debug("Handling Tuple artifact")
                    # Assume (X, y) from FeatureTargetSplitter
                    X, y = artifact
                    preview_data = {}

                    preview_data["X"] = to_records(X)
                    preview_data["y"] = to_records(y)

                    df_for_analysis = to_pandas_safe(X)
                elif (
                    isinstance(artifact, dict)
                    and "train" in artifact
                    and isinstance(artifact["train"], tuple)
                ):
                    # Handle FeatureTargetSplitter result on SplitDataset OR TrainTestSplitter result on (X, y)
                    # Both result in {'train': (X, y), 'test': (X, y)}
                    preview_data = {}

                    if "train" in artifact:
                        preview_data.update(process_xy(artifact["train"], "train"))
                        df_for_analysis = to_pandas_safe(artifact["train"][0])  # X_train

                    if "test" in artifact:
                        preview_data.update(process_xy(artifact["test"], "test"))

                    if "validation" in artifact:
                        preview_data.update(
                            process_xy(artifact["validation"], "validation")
                        )

                # Generate Recommendations
                if df_for_analysis is not None:
                    try:
                        profile = DataProfiler.generate_profile(df_for_analysis)
                        advisor = AdvisorEngine()
                        recommendations = advisor.analyze(profile)
                    except Exception as e:
                        print(f"Error generating recommendations: {e}")

        return PreviewResponse(
            pipeline_id=result.pipeline_id,
            status=result.status,
            node_results={k: v.__dict__ for k, v in result.node_results.items()},
            preview_data=preview_data,
            recommendations=recommendations,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 5. Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job_status(
    job_id: str, session: AsyncSession = Depends(get_async_session)
):
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
    return {"message": "Job cancelled successfully"}


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
async def get_latest_tuning_job(
    node_id: str, session: AsyncSession = Depends(get_async_session)
):
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
    from sqlalchemy import func, select

    from backend.database.models import (
        DataSource,
        Deployment,
        AdvancedTuningJob,
        BasicTrainingJob,
    )

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
async def get_dataset_schema(
    dataset_id: int, session: AsyncSession = Depends(get_async_session)
):
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
                elif any(
                    x in dtype for x in ["Utf8", "String", "Categorical", "Object"]
                ):
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
            pass

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
        from backend.data.catalog import FileSystemCatalog

        catalog = FileSystemCatalog()
        df = catalog.load(str(path), limit=1000)

        # Profile
        from .data.profiler import DataProfiler

        profile = DataProfiler.generate_profile(df)
        return profile

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to profile dataset: {str(e)}"
        )


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
    from backend.database.models import DataSource

    stmt = select(DataSource.source_id, DataSource.name).where(DataSource.is_active)
    result = await session.execute(stmt)
    return [{"id": row.source_id, "name": row.name} for row in result.all()]
