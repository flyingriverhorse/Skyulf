"""Request and response models for the data ingestion API.

The response models are where sensitive material is stripped before it reaches a
client: ``DataSourceRead`` removes credential-bearing keys from ``config`` and
lifts the row/column/byte counts out of ``source_metadata`` into top-level
fields, so the frontend never has to know the storage layout.
"""

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
    """Request body for registering a source from an inline config, no upload.

    Consumed by ``POST /api/ingestion/database``. Only the ``s3`` type is
    accepted there (``DataIngestionService._INLINE_SOURCE_TYPES``); ``file``
    sources go through the upload endpoint instead. ``config`` carries the
    source-specific connection details.
    """

    name: str
    type: str = "file"
    config: dict[str, Any] = {}
    description: str | None = None


class IngestionJobResponse(BaseModel):
    """Acknowledgement returned once an ingestion job has been queued."""

    job_id: str
    status: str
    message: str
    file_id: str | None = None


class IngestionStatus(BaseModel):
    """Progress payload returned by the ingestion status endpoint for polling."""

    status: str  # pending, processing, completed, failed
    progress: float
    error: str | None = None
    details: dict[str, Any] | None = None
    updated_at: datetime


class DataSourceRead(BaseModel):
    """Client-safe view of a ``DataSource`` row.

    ``from_attributes`` is enabled so instances are validated straight off the
    ORM object. The two validators below strip credentials from ``config`` and
    flatten the profiling counts up to top-level fields.
    """

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
        """Drop the credential-bearing keys listed in ``_SENSITIVE_CONFIG_KEYS``.

        Runs ``mode="before"``, so redaction cannot be bypassed by a caller
        constructing the model directly. Non-dict input passes through to normal
        field validation.
        """
        if isinstance(v, dict):
            return _redact_config(v)
        return v

    @model_validator(mode="after")
    def extract_metadata_fields(self):
        """Lift ``row_count``/``column_count``/``file_size`` up to top-level fields.

        A source that has not been profiled yet has no ``source_metadata``, so
        ``rows``/``columns``/``size_bytes`` keep their ``None`` defaults and the
        row still serializes.
        """
        if self.source_metadata:
            self.rows = self.source_metadata.get("row_count")
            self.columns = self.source_metadata.get("column_count")
            self.size_bytes = self.source_metadata.get("file_size")
        return self

    model_config = ConfigDict(from_attributes=True)


class DataSourceListResponse(BaseModel):
    """Wrap a page of sources under a ``sources`` key for the list endpoints."""

    sources: list[DataSourceRead]


class DataSourceResponse(BaseModel):
    """Wrap one source under a ``source`` key for the single-get endpoint."""

    source: DataSourceRead


class DataSourceSampleResponse(BaseModel):
    """Wrap sampled rows under a ``data`` key, one JSON object per row."""

    data: list[dict[str, Any]]
