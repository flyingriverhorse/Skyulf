from typing import Any, Dict, Optional


class SkyulfException(Exception):
    """Base exception for all Skyulf application errors."""

    status_code: int = 500
    error_code: str = "INTERNAL_SERVER_ERROR"

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ResourceNotFoundException(SkyulfException):
    """Raised when a requested resource (dataset, pipeline, etc.) is not found."""

    status_code = 404
    error_code = "RESOURCE_NOT_FOUND"


class InvalidRequestException(SkyulfException):
    """Raised when a user request is invalid or malformed."""

    status_code = 400
    error_code = "BAD_REQUEST"


class PipelineExecutionException(SkyulfException):
    """Raised when a pipeline execution fails during running/training."""

    status_code = 422
    error_code = "PIPELINE_EXECUTION_ERROR"


class DataIngestionException(SkyulfException):
    """Raised when ingesting data from a source fails."""

    status_code = 400
    error_code = "DATA_INGESTION_ERROR"


class UnauthorizedException(SkyulfException):
    """Raised when the user is unauthorized to access or perform an action."""

    status_code = 401
    error_code = "UNAUTHORIZED"


class ForbiddenException(SkyulfException):
    """Raised when the user's action is forbidden."""

    status_code = 403
    error_code = "FORBIDDEN"
