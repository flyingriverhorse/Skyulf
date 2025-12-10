from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel
from datetime import datetime
import tempfile
import shutil
import os
import json
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from core.config import get_settings
from core.database.engine import get_async_session, get_db
from core.database.models import FeatureEngineeringPipeline, TrainingJob, HyperparameterTuningJob
from core.data_ingestion.service import DataIngestionService
from core.utils.file_utils import extract_file_path_from_source
from core.ml_pipeline.tasks import run_pipeline_task

from .execution.engine import PipelineEngine
from .execution.schemas import PipelineConfig, NodeConfig, PipelineExecutionResult
from .artifacts.local import LocalArtifactStore
from .data.container import SplitDataset
from .data.loader import DataLoader
from .registry import NodeRegistry, RegistryItem
from .data.profiler import DataProfiler
from .recommendations.engine import AdvisorEngine
from .recommendations.schemas import Recommendation, AnalysisProfile
from .execution.jobs import JobManager, JobStatus, JobInfo
from .modeling.hyperparameters import get_hyperparameters, get_default_search_space
import logging

logger = logging.getLogger(__name__)

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
    job_type: Optional[str] = "training" # "training" or "tuning"

class RunPipelineResponse(BaseModel):
    message: str
    pipeline_id: str
    job_id: str

@router.post("/run", response_model=RunPipelineResponse)
async def run_pipeline(
    config: PipelineConfigModel,
    db: AsyncSession = Depends(get_async_session)
):
    """
    Submit a pipeline for asynchronous execution via Celery.
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
            if node.step_type == "trainer" or node.step_type == "model_training":
                model_type = node.params.get("model_type", "unknown")
            elif node.step_type == "model_tuning":
                # For tuning nodes, model_type might be in params directly or inside tuning_config?
                # Usually it's 'algorithm' or 'model_type' in params
                model_type = node.params.get("algorithm") or node.params.get("model_type", "unknown")
                
        # Try to find dataset_id from data_loader node
        if node.step_type == "data_loader":
            dataset_id = node.params.get("dataset_id", "unknown")

    # --- Path Resolution Logic ---
    ingestion_service = DataIngestionService(db)
    for node in config.nodes:
        if node.step_type == "data_loader" and "dataset_id" in node.params:
            try:
                ds_id = int(node.params["dataset_id"])
                ds = await ingestion_service.get_source(ds_id)
                if ds:
                    # Use the UUID source_id for consistency with the database join
                    if ds.source_id:
                        dataset_id = ds.source_id # Update dataset_id for job record
                    
                    ds_dict = {
                        "connection_info": ds.config,
                        "file_path": ds.config.get("file_path") if ds.config else None
                    }
                    path = extract_file_path_from_source(ds_dict)
                    if path:
                        node.params["path"] = str(path)
                    else:
                        raise HTTPException(status_code=400, detail=f"Could not resolve path for dataset {ds_id}")
                else:
                    raise HTTPException(status_code=404, detail=f"Dataset {ds_id} not found")
            except ValueError:
                 # If it's not an int, assume it's already a UUID or string ID
                 pass
    # -----------------------------

    # Create Job in DB
    job_id = await JobManager.create_job(
        session=db,
        pipeline_id=pipeline_id,
        node_id=target_node_id or "unknown",
        job_type=config.job_type,
        dataset_id=dataset_id,
        model_type=model_type,
        graph=config.dict()
    )
    
    # Trigger Celery Task
    run_pipeline_task.delay(job_id, config.dict())
    
    return RunPipelineResponse(
        message="Pipeline execution started",
        pipeline_id=pipeline_id,
        job_id=job_id
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
    session: AsyncSession = Depends(get_async_session)
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
            raise HTTPException(status_code=500, detail=f"Failed to save pipeline to JSON: {str(e)}")

    # Default: Database Storage
    try:
        # Check if pipeline exists for this dataset
        stmt = select(FeatureEngineeringPipeline).where(
            FeatureEngineeringPipeline.dataset_source_id == dataset_id,
            FeatureEngineeringPipeline.is_active == True
        )
        result = await session.execute(stmt)
        existing_pipeline = result.scalar_one_or_none()

        if existing_pipeline:
            # Update existing
            existing_pipeline.graph = payload.graph
            existing_pipeline.name = payload.name
            if payload.description:
                existing_pipeline.description = payload.description
            # existing_pipeline.updated_at is handled by mixin
        else:
            # Create new
            new_pipeline = FeatureEngineeringPipeline(
                dataset_source_id=dataset_id,
                name=payload.name,
                description=payload.description,
                graph=payload.graph,
                is_active=True
            )
            session.add(new_pipeline)
        
        await session.commit()
        return {"status": "success", "id": dataset_id, "storage": "database"}
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save pipeline: {str(e)}")

@router.get("/load/{dataset_id}")
async def load_pipeline(
    dataset_id: str,
    session: AsyncSession = Depends(get_async_session)
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
            raise HTTPException(status_code=500, detail=f"Failed to load pipeline from JSON: {str(e)}")

    # Default: Database Storage
    try:
        stmt = select(FeatureEngineeringPipeline).where(
            FeatureEngineeringPipeline.dataset_source_id == dataset_id,
            FeatureEngineeringPipeline.is_active == True
        )
        result = await session.execute(stmt)
        pipeline = result.scalar_one_or_none()
        
        if not pipeline:
            return None
            
        return pipeline.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load pipeline: {str(e)}")

@router.post("/preview", response_model=PreviewResponse)
async def preview_pipeline(
    config: PipelineConfigModel,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Runs the pipeline in Preview Mode:
    - Uses a temporary artifact store (cleaned up after request).
    - Resolves dataset paths from IDs.
    """
    
    # Resolve dataset paths
    ingestion_service = DataIngestionService(session)
    for node in config.nodes:
        if node.step_type == "data_loader" and "dataset_id" in node.params:
            try:
                ds_id = int(node.params["dataset_id"])
                ds = await ingestion_service.get_data_source_by_id(ds_id)
                if ds:
                    # Convert SQLAlchemy model to dict for utility
                    # Note: DataSource model stores connection info in 'config' column
                    ds_dict = {
                        "connection_info": ds.config,
                        "file_path": ds.config.get("file_path") if ds.config else None
                    }
                    path = extract_file_path_from_source(ds_dict)
                    if path:
                        node.params["path"] = str(path)
                    else:
                        raise HTTPException(status_code=400, detail=f"Could not resolve path for dataset {ds_id}")
                else:
                    raise HTTPException(status_code=404, detail=f"Dataset {ds_id} not found")
            except ValueError:
                 raise HTTPException(status_code=400, detail=f"Invalid dataset ID: {node.params['dataset_id']}")

    # 1. Create Temporary Artifact Store
    temp_dir = tempfile.mkdtemp(prefix="skyulf_preview_")
    artifact_store = LocalArtifactStore(temp_dir)
    
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
                params["limit"] = 1000 # Default preview limit
            
            nodes.append(NodeConfig(
                node_id=node.node_id,
                step_type=node.step_type,
                params=params,
                inputs=node.inputs
            ))
            
        pipeline_config = PipelineConfig(
            pipeline_id=config.pipeline_id,
            nodes=nodes,
            metadata=config.metadata
        )
        
        # 3. Run Engine
        engine = PipelineEngine(artifact_store)
        result = engine.run(pipeline_config)
        
        # 4. Extract Preview Data & Generate Recommendations
        preview_data = {}
        recommendations = []
        
        if result.status == "success" and config.nodes:
            # Determine which node's output to preview
            target_node = config.nodes[-1]
            target_node_id = target_node.node_id
            
            # If the last node is a modeling node, we can't preview the model object as data.
            # Instead, we preview the input data that went into the model.
            if target_node.step_type in ["model_training", "model_tuning"]:
                if target_node.inputs:
                    # Use the input node's output
                    target_node_id = target_node.inputs[0]
                    logger.info(f"Last node is {target_node.step_type}, previewing input node {target_node_id} instead")

            if artifact_store.exists(target_node_id):
                artifact = artifact_store.load(target_node_id)
                
                # Debug logging
                logger.debug(f"Loaded artifact for node {target_node_id}. Type: {type(artifact)}")
                if isinstance(artifact, SplitDataset):
                    logger.debug(f"SplitDataset Train Type: {type(artifact.train)}")
                
                df_for_analysis = None
                
                # Helper to process (X, y) tuple
                def process_xy(xy_tuple, prefix):
                    X, y = xy_tuple
                    if isinstance(X, pd.Series): X = X.to_frame()
                    if isinstance(y, pd.Series): y = y.to_frame()
                    return {
                        f"{prefix}_X": json.loads(X.head(50).to_json(orient="records")),
                        f"{prefix}_y": json.loads(y.head(50).to_json(orient="records"))
                    }

                # Handle different artifact types
                if isinstance(artifact, pd.DataFrame):
                    logger.debug("Handling DataFrame artifact")
                    preview_data = json.loads(artifact.head(50).to_json(orient="records"))
                    df_for_analysis = artifact
                elif isinstance(artifact, SplitDataset):
                    logger.debug("Handling SplitDataset artifact")
                    preview_data = {}
                    
                    # Handle Train
                    if isinstance(artifact.train, tuple):
                        logger.debug("Train is tuple")
                        preview_data.update(process_xy(artifact.train, "train"))
                        df_for_analysis = artifact.train[0]
                    else:
                        logger.debug("Train is DataFrame")
                        preview_data["train"] = json.loads(artifact.train.head(50).to_json(orient="records"))
                        df_for_analysis = artifact.train

                    # Handle Test
                    if isinstance(artifact.test, tuple):
                        preview_data.update(process_xy(artifact.test, "test"))
                    else:
                        preview_data["test"] = json.loads(artifact.test.head(50).to_json(orient="records"))

                    # Handle Validation
                    if artifact.validation is not None:
                        if isinstance(artifact.validation, tuple):
                            preview_data.update(process_xy(artifact.validation, "validation"))
                        else:
                            preview_data["validation"] = json.loads(artifact.validation.head(50).to_json(orient="records"))

                elif isinstance(artifact, tuple) and len(artifact) == 2:
                    logger.debug("Handling Tuple artifact")
                    # Assume (X, y) from FeatureTargetSplitter
                    X, y = artifact
                    preview_data = {}
                    
                    if isinstance(X, (pd.DataFrame, pd.Series)):
                        if isinstance(X, pd.Series): X = X.to_frame()
                        preview_data["X"] = json.loads(X.head(50).to_json(orient="records"))
                        
                    if isinstance(y, (pd.DataFrame, pd.Series)):
                        if isinstance(y, pd.Series): y = y.to_frame()
                        preview_data["y"] = json.loads(y.head(50).to_json(orient="records"))
                        
                    df_for_analysis = X if isinstance(X, pd.DataFrame) else None
                elif isinstance(artifact, dict) and "train" in artifact and isinstance(artifact["train"], tuple):
                    # Handle FeatureTargetSplitter result on SplitDataset OR TrainTestSplitter result on (X, y)
                    # Both result in {'train': (X, y), 'test': (X, y)}
                    preview_data = {}
                    
                    if "train" in artifact:
                        preview_data.update(process_xy(artifact["train"], "train"))
                        df_for_analysis = artifact["train"][0] # X_train
                        
                    if "test" in artifact:
                        preview_data.update(process_xy(artifact["test"], "test"))
                        
                    if "validation" in artifact:
                        preview_data.update(process_xy(artifact["validation"], "validation"))
                
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
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 5. Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

@router.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job_status(
    job_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Returns the status of a background job.
    """
    job = await JobManager.get_job(session, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Cancels a running or queued job.
    """
    success = await JobManager.cancel_job(session, job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job could not be cancelled (maybe it's already finished or doesn't exist)")
    return {"message": "Job cancelled successfully"}

@router.get("/jobs/{job_id}/evaluation")
async def get_job_evaluation(
    job_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """Retrieves the raw evaluation data (y_true, y_pred) for a job."""
    
    # 1. Get Job Info
    stmt = select(TrainingJob).where(TrainingJob.id == job_id)
    result = await session.execute(stmt)
    job = result.scalar_one_or_none()
    
    if not job:
        # Try Tuning Job
        stmt = select(HyperparameterTuningJob).where(HyperparameterTuningJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()
        
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # 2. Determine Artifact Path
    # Matches the path used in run_pipeline (tasks.py)
    settings = get_settings()
    persistent_path = os.path.join(settings.TRAINING_ARTIFACT_DIR, job_id)
    
    artifact_store = LocalArtifactStore(persistent_path)
    
    # 3. Load Evaluation Artifact
    # Key format: {node_id}_evaluation_data
    # Or {job_id}_evaluation_data if saved with job_id (which we do now)
    
    # Try job_id key first (preferred)
    key = f"{job_id}_evaluation_data"
    if not artifact_store.exists(key):
        # Fallback to node_id key
        key = f"{job.node_id}_evaluation_data"
    
    if not artifact_store.exists(key):
        # Fallback: Check if the job failed or is still running
        if job.status != "completed" and job.status != "succeeded":
             raise HTTPException(status_code=400, detail=f"Job is {job.status}, evaluation data not available yet.")
        
        # Debug info
        path = artifact_store._get_path(key)
        raise HTTPException(status_code=404, detail=f"Evaluation data artifact not found. Key: {key}, Path: {path}")
        
    try:
        data = artifact_store.load(key)
        # Verify it belongs to this job (since we share the folder)
        if data.get("job_id") != job_id:
             # This confirms the overwrite issue mentioned earlier.
             # If the ID doesn't match, it means a newer job overwrote it.
             # We return it anyway with a warning in the logs, or we accept it as "latest for this pipeline".
             # For now, let's return it but log a warning.
             logger.warning(f"Evaluation data job_id mismatch. Requested: {job_id}, Found: {data.get('job_id')}")
             
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load evaluation data: {str(e)}")


@router.get("/jobs", response_model=List[JobInfo])
async def list_jobs(
    limit: int = 50,
    skip: int = 0,
    job_type: Optional[Literal["training", "tuning"]] = None,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Lists recent jobs.
    """
    return await JobManager.list_jobs(session, limit, skip, job_type)

@router.get("/jobs/tuning/latest/{node_id}", response_model=Optional[JobInfo])
async def get_latest_tuning_job(
    node_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Retrieves the latest completed tuning job for a specific node.
    """
    return await JobManager.get_latest_tuning_job_for_node(session, node_id)

@router.get("/jobs/tuning/best/{model_type}", response_model=Optional[JobInfo])
async def get_best_tuning_job_model(
    model_type: str,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Retrieves the latest completed tuning job for a specific model type.
    """
    return await JobManager.get_best_tuning_job_for_model(session, model_type)

@router.get("/jobs/tuning/history/{model_type}", response_model=List[JobInfo])
async def get_tuning_jobs_history(
    model_type: str,
    session: AsyncSession = Depends(get_async_session)
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
    from core.database.models import TrainingJob, HyperparameterTuningJob, Deployment, DataSource

    # Execute queries in parallel or sequence
    # 1. Total Jobs (Training + Tuning)
    training_count = await session.scalar(select(func.count(TrainingJob.id)))
    tuning_count = await session.scalar(select(func.count(HyperparameterTuningJob.id)))
    
    # 2. Active Deployments
    deployment_count = await session.scalar(select(func.count(Deployment.id)).where(Deployment.is_active == True))
    
    # 3. Data Sources (Only successful ones)
    datasource_count = await session.scalar(select(func.count(DataSource.id)).where(DataSource.test_status == 'success'))

    return {
        "total_jobs": (training_count or 0) + (tuning_count or 0),
        "active_deployments": deployment_count or 0,
        "data_sources": datasource_count or 0,
        "training_jobs": training_count or 0,
        "tuning_jobs": tuning_count or 0
    }

@router.get("/registry", response_model=List[RegistryItem])
def get_node_registry():
    """
    Returns the list of available pipeline nodes (transformers, models, etc.).
    """
    return NodeRegistry.get_all_nodes()

@router.get("/datasets/{dataset_id}/schema", response_model=AnalysisProfile)
async def get_dataset_schema(
    dataset_id: int,
    session: AsyncSession = Depends(get_async_session)
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
    if ds.source_metadata and 'profile' in ds.source_metadata:
        try:
            cached_profile = ds.source_metadata['profile']
            columns = {}
            for col_name, stats in cached_profile.get('columns', {}).items():
                # Map stats to ColumnProfile
                dtype = str(stats.get('type', 'unknown'))
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
                    "missing_count": stats.get('null_count', 0),
                    "missing_ratio": stats.get('null_percentage', 0) / 100.0,
                    "unique_count": stats.get('unique_count', 0),
                    "min_value": stats.get('min'),
                    "max_value": stats.get('max'),
                    "mean_value": stats.get('mean'),
                    "std_value": stats.get('std'),
                }
            
            return {
                "row_count": cached_profile.get('row_count', 0),
                "column_count": cached_profile.get('column_count', 0),
                "duplicate_row_count": cached_profile.get('duplicate_rows', 0),
                "columns": columns
            }
        except Exception as e:
            logger.warning(f"Failed to parse cached profile for {dataset_id}: {e}")
            # Fallback to loading file
            pass
        
    try:
        # Resolve path
        ds_dict = {
            "connection_info": ds.config,
            "file_path": ds.config.get("file_path") if ds.config else None
        }
        path = extract_file_path_from_source(ds_dict)
        
        if not path:
             raise HTTPException(status_code=400, detail=f"Could not resolve path for dataset {dataset_id}")
             
        # Load sample
        loader = DataLoader()
        df = loader.load_sample(str(path), n=1000)
        
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
async def list_datasets(
    session: AsyncSession = Depends(get_async_session)
):
    """
    Returns a simple list of available datasets for filtering.
    """
    from core.database.models import DataSource
    stmt = select(DataSource.source_id, DataSource.name).where(DataSource.is_active == True)
    result = await session.execute(stmt)
    return [{"id": row.source_id, "name": row.name} for row in result.all()]
