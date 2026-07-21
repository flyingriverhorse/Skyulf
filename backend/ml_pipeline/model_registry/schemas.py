from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ModelVersion(BaseModel):
    job_id: str
    pipeline_id: str
    node_id: str
    model_type: str
    version: int | str  # shared version sequence for both "fixed" and "tuned" run_modes
    source: str  # "training" or "tuning"
    status: str
    metrics: dict[str, Any] | None = None
    hyperparameters: dict[str, Any] | None = None
    created_at: datetime | None = None
    artifact_uri: str | None = None
    is_deployed: bool = False
    deployment_id: int | None = None


class ModelRegistryEntry(BaseModel):
    model_type: str
    dataset_id: str
    dataset_name: str
    dataset_type: str | None = "unknown"
    latest_version: ModelVersion | None = None
    versions: list[ModelVersion] = []
    deployment_count: int = 0


class RegistryStats(BaseModel):
    total_models: int
    total_versions: int
    active_deployments: int


class ArtifactListResponse(BaseModel):
    storage_type: str
    base_uri: str
    files: list[str]
