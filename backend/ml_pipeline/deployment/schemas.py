"""Request and response schemas for the deployment and prediction endpoints."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DeploymentCreate(BaseModel):
    """Job id to deploy, as a request body.

    Currently unused: ``POST /deployment/deploy/{job_id}`` takes the job id as a
    path parameter instead.
    """

    job_id: str


class DeploymentInfo(BaseModel):
    """One deployment as the API returns it: identity, artifact URI and lineage.

    ``input_schema`` is filled from the deployed artifact by the endpoints that
    load it and stays null on the history list; ``output_schema`` is never
    populated by any endpoint today.
    """

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
    # Lineage back to the Registry: the training job's dataset and the shared
    # version sequence, so Deployments can render the same model-version
    # identity Registry uses instead of a bare, unlinked job id.
    dataset_id: str | None = None
    version: int | str | None = None
    # The deployment this one replaced, so History can show an unbroken
    # replacement chain instead of isolated rows.
    previous_deployment_id: int | None = None


class PredictionRequest(BaseModel):
    """Rows to score with the active deployment, plus optional threshold overrides."""

    data: list[dict[str, Any]]  # List of records (rows). Row-count is capped
    # dynamically in the /predict route via Settings.MAX_PREDICT_REQUEST_ROWS.
    # Ad-hoc per-class decision thresholds applied to THIS request only,
    # overriding any saved/enabled tuned thresholds on the deployed job.
    override_thresholds: dict[str, float] | None = None


class PredictionResponse(BaseModel):
    """Scored rows, the deployed job id served as ``model_version``, and the thresholds used."""

    predictions: list[Any]
    model_version: str  # job_id
    # The per-class thresholds actually applied (override or saved+enabled),
    # or None when the model's default decision rule (argmax/0.5) was used.
    thresholds_applied: dict[str, float] | None = None
