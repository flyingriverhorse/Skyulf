from datetime import datetime
from typing import Any, Dict, Optional

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


def _redact_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *config* with sensitive keys removed."""
    return {k: v for k, v in config.items() if k not in _SENSITIVE_CONFIG_KEYS}


class DataSourceCreate(BaseModel):
    name: str
    type: str = "file"
    config: Dict[str, Any] = {}
    description: Optional[str] = None


class IngestionJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    file_id: Optional[str] = None


class IngestionStatus(BaseModel):
    status: str  # pending, processing, completed, failed
    progress: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    updated_at: datetime


class DataSourceRead(BaseModel):
    id: int
    source_id: Optional[str]
    name: str
    type: str
    config: Dict[str, Any]
    is_active: bool
    test_status: str
    created_at: datetime
    updated_at: datetime
    source_metadata: Optional[Dict[str, Any]] = Field(None, alias="source_metadata")

    rows: Optional[int] = None
    columns: Optional[int] = None
    size_bytes: Optional[int] = None

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
    data: list[Dict[str, Any]]
