"""
Custom Exception Handlers for API Responses

Provides exception handlers that return JSON responses for API errors.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def not_found_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle 404 Not Found errors."""
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Not Found",
            "message": getattr(exc, "detail", "The requested resource was not found"),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def unauthorized_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle 401 Unauthorized and 403 Forbidden errors."""
    status_code = getattr(exc, "status_code", 401)
    error_msg = (
        "Could not validate credentials" if status_code == 401 else "Access forbidden"
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": "Unauthorized" if status_code == 401 else "Forbidden",
            "message": getattr(exc, "detail", error_msg),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def validation_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle 422 Validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Unprocessable Entity",
            "message": getattr(exc, "detail", "Validation error"),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def method_not_allowed_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle 405 Method Not Allowed errors."""
    return JSONResponse(
        status_code=405,
        content={
            "success": False,
            "error": "Method Not Allowed",
            "message": f"Method {request.method} not allowed",
            "status_code": 405,
        },
    )


async def generic_http_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Generic handler for other HTTP exceptions."""
    status_code = getattr(exc, "status_code", 500)

    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": f"HTTP {status_code}",
            "message": getattr(exc, "detail", f"HTTP {status_code} error"),
            "status_code": status_code,
        },
    )
