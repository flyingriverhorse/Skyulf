"""
Error Handling Middleware

Centralizes error handling and logging for the FastAPI application.
"""

import logging
import traceback
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle uncaught exceptions and standardize error responses.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and handle any uncaught exceptions.

        Args:
            request: The HTTP request
            call_next: The next middleware or endpoint handler

        Returns:
            Response: The HTTP response
        """
        # Generate a unique request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        try:
            # Process the request
            response = await call_next(request)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as exc:
            # Log the error with full traceback
            logger.error(
                f"Unhandled exception in request {request_id}: {exc}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "client": request.client.host if request.client else "unknown",
                    "traceback": traceback.format_exc()
                },
                exc_info=True
            )

            # Return standardized error response
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Internal server error",
                    "error": "An unexpected error occurred. Please try again later.",
                    "request_id": request_id
                },
                headers={"X-Request-ID": request_id}
            )
