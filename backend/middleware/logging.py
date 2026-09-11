"""Logging Middleware.

Provides request/response logging for monitoring and debugging.
"""

import logging
import time
from collections.abc import Callable
from typing import cast

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.utils.logging_utils import redact_credentials, sanitize_for_log

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log HTTP requests and responses with credential-redacted metadata."""

    def __init__(self, app: ASGIApp):
        """Wrap the downstream ASGI ``app`` so its requests pass through ``dispatch``.

        A pass-through to ``BaseHTTPMiddleware``'s own constructor; the class keeps
        no state. It mints no request id either — ``dispatch`` reads the one
        ``ErrorHandlerMiddleware`` stamped onto ``request.state`` and falls back to
        ``"unknown"``. That ordering holds because the error handler is added after
        this middleware, so it wraps it and runs first.
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request details and response metrics.

        Args:
            request: The HTTP request
            call_next: The next middleware or endpoint handler

        Returns:
            Response: The HTTP response
        """
        # Record start time
        start_time = time.time()

        # Extract request information
        method = request.method
        url = sanitize_for_log(redact_credentials(request.url))
        client_ip = request.client.host if request.client else "unknown"
        user_agent = sanitize_for_log(
            redact_credentials(request.headers.get("user-agent", "unknown"))
        )
        request_id = getattr(request.state, "request_id", "unknown")

        # Log incoming request (DEBUG only to reduce noise)
        logger.debug(
            f"Request started: {method} {url}",
            extra={
                "request_id": request_id,
                "method": method,
                "url": url,
                "client_ip": client_ip,
                "user_agent": user_agent,
                "event_type": "request_start",
            },
        )

        try:
            # Process the request
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response with color-coded status if using rich (handled by level)
            log_level = logging.INFO
            if response.status_code >= 400:
                log_level = logging.WARNING
            if response.status_code >= 500:
                log_level = logging.ERROR

            logger.log(
                log_level,
                f"{method} {url} - {response.status_code} ({process_time:.3f}s)",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "url": url,
                    "status_code": response.status_code,
                    "process_time": process_time,
                    "client_ip": client_ip,
                    "event_type": "request_complete",
                },
            )

            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)

            return cast(Response, response)

        except Exception as exc:
            # Calculate processing time for failed requests
            process_time = time.time() - start_time

            # Log error
            safe_error = sanitize_for_log(redact_credentials(exc))
            logger.error(
                f"Request failed: {method} {url} in {process_time:.3f}s - {safe_error}",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "url": url,
                    "process_time": process_time,
                    "client_ip": client_ip,
                    "error": safe_error,
                    "event_type": "request_error",
                },
            )

            # Re-raise the exception to be handled by error middleware
            raise
