from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class ModelVersion(BaseModel):
    job_id: str
    pipeline_id: str
    node_id: str
    model_type: str
    version: Union[int, str]  # version for training, run_number for tuning
    source: str  # "training" or "tuning"
    status: str
    metrics: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    artifact_uri: Optional[str] = None
    is_deployed: bool = False
    deployment_id: Optional[int] = None


class ModelRegistryEntry(BaseModel):
    model_type: str
    dataset_id: str
    dataset_name: str
    latest_version: Optional[ModelVersion] = None
    versions: List[ModelVersion] = []
    deployment_count: int = 0


class RegistryStats(BaseModel):
    total_models: int
    total_versions: int
    active_deployments: int
