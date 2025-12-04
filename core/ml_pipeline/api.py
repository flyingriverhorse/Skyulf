from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
import tempfile
import shutil
import os
import json
import pandas as pd

from .execution.engine import PipelineEngine
from .execution.schemas import PipelineConfig, NodeConfig, PipelineExecutionResult
from .artifacts.local import LocalArtifactStore
from .data.container import SplitDataset
from .registry import NodeRegistry, RegistryItem
from .data.profiler import DataProfiler
from .recommendations.engine import AdvisorEngine
from .recommendations.schemas import Recommendation
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

# --- Endpoints ---

@router.post("/preview", response_model=PreviewResponse)
def preview_pipeline(config: PipelineConfigModel):
    """
    Runs the pipeline in Preview Mode:
    - Uses a temporary artifact store (cleaned up after request).
    - Forces 'sample=True' on Data Loader nodes.
    - Returns execution results + data preview of the final node.
    """
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
                
                # Handle different artifact types
                if isinstance(artifact, pd.DataFrame):
                    preview_data = json.loads(artifact.head(20).to_json(orient="records"))
                    df_for_analysis = artifact
                elif isinstance(artifact, SplitDataset):
                    preview_data = {
                        "train": json.loads(artifact.train.head(20).to_json(orient="records")),
                        "test": json.loads(artifact.test.head(20).to_json(orient="records")),
                    }
                    if artifact.validation is not None:
                        preview_data["validation"] = json.loads(artifact.validation.head(20).to_json(orient="records"))
                    df_for_analysis = artifact.train
                
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
