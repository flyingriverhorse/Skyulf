from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
from datetime import datetime
import tempfile
import shutil
import os
import json
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from config import get_settings
from core.database.engine import get_async_session, get_db
from core.database.models import FeatureEngineeringPipeline, TrainingJob, HyperparameterTuningJob
from core.data_ingestion.service import DataIngestionService
from core.utils.file_utils import extract_file_path_from_source

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

router = APIRouter()

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

@router.post("/run")
async def run_pipeline(
    config: PipelineConfigModel, 
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Runs the pipeline in Full Mode:
    - Uses the persistent artifact store.
    - Uses full dataset.
    - Runs asynchronously (simulated via BackgroundTasks for now).
    """
    # 1. Determine Job Details
    target_node = config.target_node_id or (config.nodes[-1].node_id if config.nodes else "unknown")
    job_type = config.job_type or "training"
    
    # Resolve dataset paths (Same logic as preview)
    ingestion_service = DataIngestionService(session)
    dataset_source_id = "unknown"

    for node in config.nodes:
        if node.step_type == "data_loader" and "dataset_id" in node.params:
            # Default to whatever is passed
            dataset_source_id = str(node.params["dataset_id"])
            try:
                ds_id = int(node.params["dataset_id"])
                ds = await ingestion_service.get_data_source_by_id(ds_id)
                if ds:
                    # Use the UUID source_id for consistency with the database join
                    if ds.source_id:
                        dataset_source_id = ds.source_id
                    
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

    # 2. Create Job (Async, Persistent)
    
    # Extract model_type from target node if available
    model_type = "unknown"
    for node in config.nodes:
        if node.node_id == target_node:
            # Check params for model_type or algorithm
            if "model_type" in node.params:
                model_type = node.params["model_type"]
            elif "algorithm" in node.params:
                model_type = node.params["algorithm"]
            # Also check config inside params if nested
            elif "config" in node.params and "model_type" in node.params["config"]:
                model_type = node.params["config"]["model_type"]
            break

    job_id = await JobManager.create_job(
        session, 
        config.pipeline_id, 
        target_node, 
        job_type,
        dataset_id=dataset_source_id,
        model_type=model_type,
        graph=config.dict()
    )
    
    # 3. Setup Persistent Store
    # In production, this path should come from env vars or config
    persistent_path = os.path.join(os.getcwd(), "exports", "models", config.pipeline_id)
    artifact_store = LocalArtifactStore(persistent_path)
    
    # 4. Adapt Config
    nodes = [
        NodeConfig(
            node_id=n.node_id,
            step_type=n.step_type,
            params=n.params, # No sampling injection
            inputs=n.inputs
        ) for n in config.nodes
    ]
    
    pipeline_config = PipelineConfig(
        pipeline_id=config.pipeline_id,
        nodes=nodes,
        metadata=config.metadata
    )
    
    # 5. Run in Background
    background_tasks.add_task(_run_engine_task, artifact_store, pipeline_config, job_id, target_node)
    
    return {"message": "Pipeline execution started", "pipeline_id": config.pipeline_id, "job_id": job_id}

def _run_engine_task(store, config, job_id, target_node_id=None):
    """Helper to run engine in background."""
    # Use a fresh sync session for the background task
    db_gen = get_db()
    session = next(db_gen)
    
    def log_callback(msg):
        try:
            ts_msg = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
            JobManager.update_status_sync(session, job_id, logs=[ts_msg])
        except Exception as e:
            logger.error(f"Failed to write log to DB: {e}")
    
    try:
        JobManager.update_status_sync(session, job_id, JobStatus.RUNNING)
        
        engine = PipelineEngine(store, log_callback=log_callback)
        result = engine.run(config)
        
        if result.status == "success":
            # Extract results from target node if available
            job_result = {"pipeline_id": result.pipeline_id}
            
            if target_node_id and target_node_id in result.node_results:
                node_res = result.node_results[target_node_id]
                # For tuning jobs, metrics contains best_params and best_score
                # For training jobs, metrics contains accuracy, f1, etc.
                if node_res.metrics:
                    job_result.update(node_res.metrics)
                
                # Add artifact_uri
                job_result["artifact_uri"] = target_node_id # Using node_id as the URI key for LocalArtifactStore

                # Extract hyperparameters from the config for the target node
                # We need to find the node config in the pipeline config
                target_node_config = next((n for n in config.nodes if n.node_id == target_node_id), None)
                if target_node_config:
                    # Check for hyperparameters in params
                    # Logic should match engine._run_model_training extraction
                    params = target_node_config.params
                    hyperparameters = params.get("hyperparameters")
                    
                    # If not explicitly in "hyperparameters" key, maybe it's flat in params?
                    # But engine._run_model_training looks for params.get("hyperparameters", {})
                    # So we should stick to that.
                    if hyperparameters:
                        job_result["hyperparameters"] = hyperparameters
            
            JobManager.update_status_sync(
                session, 
                job_id, 
                JobStatus.COMPLETED, 
                result=job_result
            )
        else:
            # Find the error
            error_msg = "Unknown error"
            for node_res in result.node_results.values():
                if node_res.status == "failed":
                    error_msg = f"Node {node_res.node_id} failed: {node_res.error}"
                    break
            JobManager.update_status_sync(session, job_id, JobStatus.FAILED, error=error_msg)
            
    except Exception as e:
        JobManager.update_status_sync(session, job_id, JobStatus.FAILED, error=str(e))
    finally:
        # Close the session
        try:
            next(db_gen)
        except StopIteration:
            pass
        except Exception:
            pass

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
    # Matches the path used in run_pipeline
    persistent_path = os.path.join(os.getcwd(), "exports", "models", job.pipeline_id)
    
    artifact_store = LocalArtifactStore(persistent_path)
    
    # 3. Load Evaluation Artifact
    # Key format: {node_id}_evaluation_data
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
    session: AsyncSession = Depends(get_async_session)
):
    """
    Lists recent jobs.
    """
    return await JobManager.list_jobs(session, limit)

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
    Uses the V2 DataProfiler.
    """
    ingestion_service = DataIngestionService(session)
    ds = await ingestion_service.get_data_source_by_id(dataset_id)
    
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
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
