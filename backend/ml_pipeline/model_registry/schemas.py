"""Response schemas for the model registry endpoints."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ModelVersion(BaseModel):
    """One registry version, built from a single completed training or tuning job."""

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
    """One model_type/dataset pair: its versions newest-first and how many are deployed."""

    model_type: str
    dataset_id: str
    dataset_name: str
    dataset_type: str | None = "unknown"
    latest_version: ModelVersion | None = None
    versions: list[ModelVersion] = []
    deployment_count: int = 0


class RegistryStats(BaseModel):
    """Registry-wide counts returned by ``GET /registry/stats``."""

    total_models: int
    total_versions: int
    active_deployments: int


class ArtifactListResponse(BaseModel):
    """Where one job's artifacts live, and the keys its store lists there."""

    storage_type: str
    base_uri: str
    files: list[str]
