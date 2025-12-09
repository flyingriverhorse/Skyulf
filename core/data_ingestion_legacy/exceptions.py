"""Data ingestion exceptions for FastAPI."""

from typing import Any, Dict, Optional


class DataIngestionException(Exception):
    """Base exception for data ingestion operations."""

    def __init__(
        self,
        message: str,
        detail: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        super().__init__(message)
        self.message = message
        self.detail = detail or {}
        self.status_code = status_code


# Alias for backward compatibility
DataIngestionError = DataIngestionException


class DataSourceNotFoundError(DataIngestionException):
    """Raised when a data source is not found."""

    def __init__(self, source_id: str):
        super().__init__(
            f"Data source '{source_id}' not found",
            {"source_id": source_id},
            404
        )


class DataSourceValidationError(DataIngestionException):
    """Raised when data source validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        detail = {"field": field} if field else {}
        super().__init__(message, detail, 422)


class FileUploadError(DataIngestionException):
    """Raised when file upload fails."""

    def __init__(self, message: str, filename: Optional[str] = None):
        detail = {"filename": filename} if filename else {}
        super().__init__(message, detail, 400)


class APIConnectionError(DataIngestionException):
    """Raised when API connection fails."""

    def __init__(self, message: str, url: Optional[str] = None):
        detail = {"url": url} if url else {}
        super().__init__(message, detail, 400)


class DatabaseConnectionError(DataIngestionException):
    """Raised when database connection fails."""

    def __init__(self, message: str, db_type: Optional[str] = None):
        detail = {"db_type": db_type} if db_type else {}
        super().__init__(message, detail, 400)


class WebScrapingError(DataIngestionException):
    """Raised when web scraping fails."""

    def __init__(self, message: str, url: Optional[str] = None):
        detail = {"url": url} if url else {}
        super().__init__(message, detail, 400)


class DataQualityError(DataIngestionException):
    """Raised when data quality issues are detected."""

    def __init__(self, message: str, metrics: Optional[Dict[str, Any]] = None):
        super().__init__(message, {"quality_metrics": metrics}, 422)


class PermissionError(DataIngestionException):
    """Raised when user lacks required permissions."""

    def __init__(self, action: str, required_permission: str):
        super().__init__(
            f"Permission denied for action '{action}'. Required: {required_permission}",
            {"action": action, "required_permission": required_permission},
            403
        )


class DuplicateDataSourceError(DataIngestionException):
    """Raised when attempting to create duplicate data source."""

    def __init__(self, identifier: str, existing_id: Optional[str] = None):
        detail = {"identifier": identifier}
        if existing_id:
            detail["existing_id"] = existing_id
        super().__init__(
            f"Data source with identifier '{identifier}' already exists",
            detail,
            409
        )


class DataProcessingError(DataIngestionException):
    """Raised when data processing fails."""

    def __init__(self, message: str, source_id: Optional[str] = None):
        detail = {"source_id": source_id} if source_id else {}
        super().__init__(message, detail, 422)


class ExportError(DataIngestionException):
    """Raised when data export fails."""

    def __init__(self, message: str, format: Optional[str] = None):
        detail = {"format": format} if format else {}
        super().__init__(message, detail, 400)


class SyncConfigError(DataIngestionException):
    """Raised when sync configuration fails."""

    def __init__(self, message: str, source_id: Optional[str] = None):
        detail = {"source_id": source_id} if source_id else {}
        super().__init__(message, detail, 400)
