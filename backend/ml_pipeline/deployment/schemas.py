from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DeploymentCreate(BaseModel):
    job_id: str


class DeploymentInfo(BaseModel):
    id: int
    job_id: str
    model_type: str
    artifact_uri: str
    is_active: bool
    deployed_by: int | None
    created_at: datetime
    updated_at: datetime
    input_schema: list[dict[str, Any]] | None = None  # List of column definitions
    output_schema: dict[str, Any] | None = None
    target_column: str | None = None


class PredictionRequest(BaseModel):
    data: list[dict[str, Any]]  # List of records (rows). Row-count is capped
    # dynamically in the /predict route via Settings.MAX_PREDICT_REQUEST_ROWS.
    # Ad-hoc per-class decision thresholds applied to THIS request only,
    # overriding any saved/enabled tuned thresholds on the deployed job.
    override_thresholds: dict[str, float] | None = None


class PredictionResponse(BaseModel):
    predictions: list[Any]
    model_version: str  # job_id
    # The per-class thresholds actually applied (override or saved+enabled),
    # or None when the model's default decision rule (argmax/0.5) was used.
    thresholds_applied: dict[str, float] | None = None
