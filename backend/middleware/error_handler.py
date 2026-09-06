"""Error Handling Middleware.

Centralizes error handling and logging for the FastAPI application.
"""

import logging
import traceback
import uuid
from collections.abc import Callable
from typing import cast

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware to handle uncaught exceptions and standardize error responses."""

    def __init__(self, app: ASGIApp):
        """Wrap the downstream ASGI ``app`` so its requests pass through ``dispatch``.

        A pass-through to ``BaseHTTPMiddleware``'s own constructor: the class keeps
        no state of its own. The ``request_id`` every response and error log carries
        is minted per call inside ``dispatch`` and stamped onto ``request.state``,
        which is where the logging middleware and the exception handlers read it
        from — this middleware is added outside those, so it runs first.
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and handle any uncaught exceptions.

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

            return cast(Response, response)

        except Exception as exc:
            # Log the error with full traceback
            logger.error(
                f"Unhandled exception in request {request_id}: {exc}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "client": request.client.host if request.client else "unknown",
                    "traceback": traceback.format_exc(),
                },
                exc_info=True,
            )

            # Return standardized error response
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Internal server error",
                    "error": "An unexpected error occurred. Please try again later.",
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )
