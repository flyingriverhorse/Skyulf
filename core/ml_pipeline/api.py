from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
import tempfile
import shutil
import os
import json
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from config import get_settings
from core.database.engine import get_async_session
from core.database.models import FeatureEngineeringPipeline
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
            # Get the last node's output
            last_node_id = config.nodes[-1].node_id
            if artifact_store.exists(last_node_id):
                artifact = artifact_store.load(last_node_id)
                
                df_for_analysis = None
                
                # Helper to process (X, y) tuple
                def process_xy(xy_tuple, prefix):
                    X, y = xy_tuple
                    if isinstance(X, pd.Series): X = X.to_frame()
                    if isinstance(y, pd.Series): y = y.to_frame()
                    return {
                        f"{prefix}_X": json.loads(X.head(20).to_json(orient="records")),
                        f"{prefix}_y": json.loads(y.head(20).to_json(orient="records"))
                    }

                # Handle different artifact types
                if isinstance(artifact, pd.DataFrame):
                    preview_data = json.loads(artifact.head(20).to_json(orient="records"))
                    df_for_analysis = artifact
                elif isinstance(artifact, SplitDataset):
                    preview_data = {}
                    
                    # Handle Train
                    if isinstance(artifact.train, tuple):
                        preview_data.update(process_xy(artifact.train, "train"))
                        df_for_analysis = artifact.train[0]
                    else:
                        preview_data["train"] = json.loads(artifact.train.head(20).to_json(orient="records"))
                        df_for_analysis = artifact.train

                    # Handle Test
                    if isinstance(artifact.test, tuple):
                        preview_data.update(process_xy(artifact.test, "test"))
                    else:
                        preview_data["test"] = json.loads(artifact.test.head(20).to_json(orient="records"))

                    # Handle Validation
                    if artifact.validation is not None:
                        if isinstance(artifact.validation, tuple):
                            preview_data.update(process_xy(artifact.validation, "val"))
                        else:
                            preview_data["validation"] = json.loads(artifact.validation.head(20).to_json(orient="records"))

                elif isinstance(artifact, tuple) and len(artifact) == 2:
                    # Assume (X, y) from FeatureTargetSplitter
                    X, y = artifact
                    preview_data = {}
                    
                    if isinstance(X, (pd.DataFrame, pd.Series)):
                        if isinstance(X, pd.Series): X = X.to_frame()
                        preview_data["X"] = json.loads(X.head(20).to_json(orient="records"))
                        
                    if isinstance(y, (pd.DataFrame, pd.Series)):
                        if isinstance(y, pd.Series): y = y.to_frame()
                        preview_data["y"] = json.loads(y.head(20).to_json(orient="records"))
                        
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
                        preview_data.update(process_xy(artifact["validation"], "val"))
                
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
def run_pipeline(config: PipelineConfigModel, background_tasks: BackgroundTasks):
    """
    Runs the pipeline in Full Mode:
    - Uses the persistent artifact store.
    - Uses full dataset.
    - Runs asynchronously (simulated via BackgroundTasks for now).
    """
    # 1. Create Job
    job_id = JobManager.create_job(config.pipeline_id)
    
    # 2. Setup Persistent Store
    # In production, this path should come from env vars or config
    persistent_path = os.path.join(os.getcwd(), "exports", "models", config.pipeline_id)
    artifact_store = LocalArtifactStore(persistent_path)
    
    # 3. Adapt Config
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
    
    # 4. Run in Background
    background_tasks.add_task(_run_engine_task, artifact_store, pipeline_config, job_id)
    
    return {"message": "Pipeline execution started", "pipeline_id": config.pipeline_id, "job_id": job_id}

def _run_engine_task(store, config, job_id):
    """Helper to run engine in background."""
    JobManager.update_status(job_id, JobStatus.RUNNING)
    try:
        engine = PipelineEngine(store)
        result = engine.run(config)
        
        if result.status == "success":
            JobManager.update_status(job_id, JobStatus.COMPLETED, result={"pipeline_id": result.pipeline_id})
        else:
            # Find the error
            error_msg = "Unknown error"
            for node_res in result.node_results.values():
                if node_res.status == "failed":
                    error_msg = f"Node {node_res.node_id} failed: {node_res.error}"
                    break
            JobManager.update_status(job_id, JobStatus.FAILED, error=error_msg)
            
    except Exception as e:
        JobManager.update_status(job_id, JobStatus.FAILED, error=str(e))

@router.get("/jobs/{job_id}", response_model=JobInfo)
def get_job_status(job_id: str):
    """
    Returns the status of a background job.
    """
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

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
