from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

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

class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]] # List of records (rows)

class PredictionResponse(BaseModel):
    predictions: List[Any]
    model_version: str # job_id
