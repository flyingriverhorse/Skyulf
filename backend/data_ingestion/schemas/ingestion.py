from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Keys inside `config` that should never be sent to clients.
_SENSITIVE_CONFIG_KEYS = frozenset(
    {
        "storage_options",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "password",
        "secret",
        "private_key",
        "token",
        "api_key",
    }
)


def _redact_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *config* with sensitive keys removed."""
    return {k: v for k, v in config.items() if k not in _SENSITIVE_CONFIG_KEYS}


class DataSourceCreate(BaseModel):
    name: str
    type: str = "file"
    config: dict[str, Any] = {}
    description: str | None = None


class IngestionJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    file_id: str | None = None


class IngestionStatus(BaseModel):
    status: str  # pending, processing, completed, failed
    progress: float
    error: str | None = None
    details: dict[str, Any] | None = None
    updated_at: datetime


class DataSourceRead(BaseModel):
    id: int
    source_id: str | None
    name: str
    type: str
    config: dict[str, Any]
    is_active: bool
    test_status: str
    created_at: datetime
    updated_at: datetime
    source_metadata: dict[str, Any] | None = Field(None, alias="source_metadata")

    rows: int | None = None
    columns: int | None = None
    size_bytes: int | None = None

    @field_validator("config", mode="before")
    @classmethod
    def redact_sensitive_config(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return _redact_config(v)
        return v

    @model_validator(mode="after")
    def extract_metadata_fields(self):
        if self.source_metadata:
            self.rows = self.source_metadata.get("row_count")
            self.columns = self.source_metadata.get("column_count")
            self.size_bytes = self.source_metadata.get("file_size")
        return self

    model_config = ConfigDict(from_attributes=True)


class DataSourceListResponse(BaseModel):
    sources: list[DataSourceRead]


class DataSourceResponse(BaseModel):
    source: DataSourceRead


class DataSourceSampleResponse(BaseModel):
    data: list[dict[str, Any]]
