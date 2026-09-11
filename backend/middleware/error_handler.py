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

from backend.utils.logging_utils import redact_credentials, sanitize_for_log

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Return standard errors and log credential-redacted exception chains."""

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

        except Exception as exc:  # noqa: BLE001 - final HTTP error boundary, redacted and logged
            # Redact the complete chain before it reaches any log formatter.
            # Raw exc_info would let handlers render the original secrets again.
            safe_traceback = redact_credentials(
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            )
            logger.error(
                "Unhandled exception in request %s: %s\n%s",
                request_id,
                sanitize_for_log(redact_credentials(exc)),
                safe_traceback,
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": sanitize_for_log(redact_credentials(request.url)),
                    "client": request.client.host if request.client else "unknown",
                    "traceback": safe_traceback,
                },
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
