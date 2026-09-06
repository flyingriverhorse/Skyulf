"""The ``SkyulfException`` hierarchy the API raises and renders as JSON.

Every class here carries two class attributes — the HTTP ``status_code`` it maps
to and a stable machine-readable ``error_code`` — so raising one is enough for
``backend.exceptions.handlers`` to build the response without the call site
knowing anything about HTTP. A new application error belongs here as a subclass
overriding both, rather than as a bare ``HTTPException`` at the raise site.
"""

from typing import Any


class SkyulfException(Exception):
    """Base exception for all Skyulf application errors."""

    status_code: int = 500
    error_code: str = "INTERNAL_SERVER_ERROR"

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """Store the caller-facing ``message`` and optional structured ``details``.

        Args:
            message: Human-readable explanation. Also passed to ``Exception``, so
                ``str(exc)`` and ``exc.message`` are the same text and a log line
                that interpolates the exception shows what the client will see.
            details: Structured context to return alongside the message. ``None``
                is normalized to an empty dict, so handlers serialize it without a
                None check.
        """
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
