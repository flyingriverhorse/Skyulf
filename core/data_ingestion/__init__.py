"""Data ingestion core module."""

from .models import (
    DataSourceCreate,
    DataSourceResponse,
    DataSourceUpdate,
    SourceType,
    DataSourceStatus,
    FileUploadResponse,
    APITestRequest,
    APIConnectRequest,
    DatabaseConnectionRequest,
    QualityMetrics
)
from .exceptions import (
    DataIngestionException,
    DataSourceNotFoundError,
    FileUploadError,
    APIConnectionError,
    DatabaseConnectionError,
    PermissionError
)
from .service import DataIngestionService, get_data_ingestion_service
from .dependencies import (
    get_data_service,
    require_data_access,
    require_data_admin,
    handle_data_ingestion_exception
)

__all__ = [
    # Models
    "DataSourceCreate",
    "DataSourceResponse",
    "DataSourceUpdate",
    "SourceType",
    "DataSourceStatus",
    "FileUploadResponse",
    "APITestRequest",
    "APIConnectRequest",
    "DatabaseConnectionRequest",
    "QualityMetrics",
    # Exceptions
    "DataIngestionException",
    "DataSourceNotFoundError",
    "FileUploadError",
    "APIConnectionError",
    "DatabaseConnectionError",
    "PermissionError",
    # Service
    "DataIngestionService",
    "get_data_ingestion_service",
    # Dependencies
    "get_data_service",
    "require_data_access",
    "require_data_admin",
    "handle_data_ingestion_exception",
]