"""Pydantic models for data ingestion API endpoints."""

from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum


class SourceType(str, Enum):
    """Supported data source types."""
    FILE = "file"
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    TXT = "txt"
    PARQUET = "parquet"


class DataSourceStatus(str, Enum):
    """Data source status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PROCESSING = "processing"


class DataSourceBase(BaseModel):
    """Base model for data source."""
    name: str = Field(..., min_length=1, max_length=255)
    source_type: SourceType
    category: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()


class DataSourceCreate(DataSourceBase):
    """Model for creating a new data source."""
    connection_info: Dict[str, Any]


class DataSourceUpdate(BaseModel):
    """Model for updating a data source."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    metadata: Optional[Dict[str, Any]] = None
    category: Optional[str] = None
    status: Optional[DataSourceStatus] = None
    connection_info: Optional[Dict[str, Any]] = None


class DataSource(DataSourceBase):
    """Full data source model with ID and timestamps."""
    id: str
    connection_info: Dict[str, Any]
    created_at: Optional[datetime] = None
    status: DataSourceStatus = DataSourceStatus.ACTIVE
    created_by: Optional[str] = None
    last_accessed: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class DataSourceResponse(DataSourceBase):
    """Response model for data source."""
    source_id: str
    status: DataSourceStatus = DataSourceStatus.ACTIVE
    connection_info: Dict[str, Any]
    created_at: datetime
    last_accessed: Optional[datetime] = None
    created_by: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class FileUploadRequest(BaseModel):
    """File upload metadata."""
    custom_name: Optional[str] = None


class FileUploadResponse(BaseModel):
    """File upload response."""
    success: bool
    source_id: Optional[str] = None
    file_info: Dict[str, Any]
    message: str
    is_duplicate: bool = False


class APITestRequest(BaseModel):
    """API connection test request."""
    url: str = Field(..., pattern=r'^https?://')
    method: str = Field(default="GET", pattern=r'^(GET|POST|PUT|PATCH|DELETE)$')
    headers: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    pagination: Optional[Dict[str, Any]] = None
    max_pages: int = Field(default=1, ge=1, le=100)
    preview_rows: int = Field(default=5, ge=1, le=100)


class APIConnectRequest(APITestRequest):
    """API connection request."""
    name: str = Field(..., min_length=1, max_length=255)


class APIResponse(BaseModel):
    """API response model."""
    success: bool
    message: Optional[str] = None
    source_id: Optional[str] = None
    preview: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class QualityMetrics(BaseModel):
    """Data quality metrics."""
    overall_completeness: float = Field(..., ge=0, le=100)
    columns_with_missing: int = Field(..., ge=0)
    sample_size: int = Field(..., ge=0)
    estimated_rows: Optional[int] = None
    duplicate_rows: Optional[int] = None
    unique_values_per_column: Optional[Dict[str, int]] = None


class DataSourceList(BaseModel):
    """List of data sources."""
    sources: List[DataSourceResponse]
    total: int
    offset: int = 0
    limit: int = 100


class DatabaseConnectionRequest(BaseModel):
    """Database connection request."""
    name: str = Field(..., min_length=1, max_length=255)
    db_type: str = Field(..., pattern=r'^(sqlite|postgresql|mysql|sqlserver|oracle)$')
    host: Optional[str] = None
    port: Optional[int] = Field(None, ge=1, le=65535)
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    db_schema: Optional[str] = Field(None, alias="schema")
    table: str
    query: Optional[str] = None
    ssl_mode: Optional[str] = None


class BulkImportRequest(BaseModel):
    """Bulk data import request."""
    sources: List[str] = Field(..., min_length=1, max_length=50)
    target_format: str = Field(default="csv", pattern=r'^(csv|json|parquet)$')
    merge_strategy: Optional[str] = Field(None, pattern=r'^(concat|merge|join)$')
    output_name: Optional[str] = None


class DataExportRequest(BaseModel):
    """Data export request."""
    source_id: str
    format: str = Field(default="csv", pattern=r'^(csv|json|excel|parquet)$')
    filters: Optional[Dict[str, Any]] = None
    columns: Optional[List[str]] = None
    limit: Optional[int] = Field(None, ge=1, le=100000)


class SyncConfigRequest(BaseModel):
    """Sync configuration request."""
    source_id: str
    schedule: str = Field(..., pattern=r'^(hourly|daily|weekly|monthly|manual)$')
    auto_refresh: bool = True
    notification_email: Optional[str] = Field(None, pattern=r'^[^@]+@[^@]+\.[^@]+$')
    webhook_url: Optional[str] = Field(None, pattern=r'^https?://')
