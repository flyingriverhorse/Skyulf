"""
Custom Exception Handlers for API Responses

Provides exception handlers that return JSON responses for API errors.
"""

import logging
import traceback as tb_module

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def _record_error(
    route: str,
    error_type: str,
    message: str,
    traceback: str,
    status_code: int,
    job_id: str = "",
) -> None:
    """Persist an unhandled error to the error_events table. Best-effort — never raises."""
    try:
        from backend.database.engine import async_session_factory
        from backend.database.models import ErrorEvent

        if not async_session_factory:
            return
        async with async_session_factory() as session:
            event = ErrorEvent(
                route=route,
                error_type=error_type,
                message=message[:2000],
                traceback=traceback[:8000],
                job_id=job_id or None,
                status_code=status_code,
            )
            session.add(event)
            await session.commit()
    except Exception as persist_err:
        logger.debug("ErrorEvent persist failed: %s", persist_err)


def record_pipeline_error(job_id: str, message: str, traceback: str) -> None:
    """Sync helper for recording pipeline errors from Celery tasks. Best-effort."""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from backend.config import get_settings
        from backend.database.models import ErrorEvent

        settings = get_settings()
        db_url = settings.DATABASE_URL
        if db_url.startswith("sqlite+aiosqlite://"):
            db_url = db_url.replace("sqlite+aiosqlite://", "sqlite://")
        else:
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")

        engine = create_engine(db_url)
        session = sessionmaker(bind=engine)()
        try:
            event = ErrorEvent(
                route="celery/pipeline",
                error_type="PipelineExecutionException",
                message=message[:2000],
                traceback=traceback[:8000],
                job_id=job_id or None,
                status_code=500,
            )
            session.add(event)
            session.commit()
        finally:
            session.close()
    except Exception as persist_err:
        logger.debug("ErrorEvent (sync) persist failed: %s", persist_err)


async def skyulf_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle custom Skyulf app exceptions."""
    from backend.exceptions.core import SkyulfException

    if not isinstance(exc, SkyulfException):
        raise exc

    if exc.status_code >= 500:
        await _record_error(
            route=str(request.url.path),
            error_type=type(exc).__name__,
            message=exc.message,
            traceback=tb_module.format_exc(),
            status_code=exc.status_code,
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


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


async def unauthorized_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle 401 Unauthorized and 403 Forbidden errors."""
    status_code = getattr(exc, "status_code", 401)
    error_msg = "Could not validate credentials" if status_code == 401 else "Access forbidden"

    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": "Unauthorized" if status_code == 401 else "Forbidden",
            "message": getattr(exc, "detail", error_msg),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
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


async def method_not_allowed_exception_handler(request: Request, exc: Exception) -> JSONResponse:
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

    if status_code >= 500:
        await _record_error(
            route=str(request.url.path),
            error_type=type(exc).__name__,
            message=str(getattr(exc, "detail", exc)),
            traceback=tb_module.format_exc(),
            status_code=status_code,
        )

    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": f"HTTP {status_code}",
            "message": getattr(exc, "detail", f"HTTP {status_code} error"),
            "status_code": status_code,
        },
    )
