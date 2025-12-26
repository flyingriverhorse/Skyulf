from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DeploymentCreate(BaseModel):
    job_id: str


class DeploymentInfo(BaseModel):
    id: int
    job_id: str
    model_type: str
    artifact_uri: str
    is_active: bool
    deployed_by: Optional[int]
    created_at: datetime
    updated_at: datetime
    input_schema: Optional[List[Dict[str, Any]]] = None  # List of column definitions
    output_schema: Optional[Dict[str, Any]] = None
    target_column: Optional[str] = None


class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]  # List of records (rows)


class PredictionResponse(BaseModel):
    predictions: List[Any]
    model_version: str  # job_id
