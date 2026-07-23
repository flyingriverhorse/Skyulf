"""
Health Check Endpoints

Basic health and status endpoints for monitoring and load balancer checks.
"""

import logging
import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.config import Settings
from backend.dependencies import get_config

router = APIRouter()

# Initialize start time for uptime calculation
START_TIME = time.time()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime
    version: str
    environment: str
    uptime_seconds: float


class DetailedHealthResponse(HealthResponse):
    """Detailed health check with additional information.

    This endpoint has no authentication, so the response intentionally stays
    coarse (aggregate booleans only) rather than naming specific backends,
    integrations, or connection details — avoiding disclosure of internal
    architecture/topology to anonymous callers.
    """

    dependencies_healthy: bool


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_config)):
    """
    Basic health check endpoint.
    Returns simple status information for load balancers.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC),
        version=settings.APP_VERSION,
        environment=settings.environment_name,
        uptime_seconds=time.time() - START_TIME,
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(settings: Settings = Depends(get_config)):
    """
    Detailed health check endpoint.

    Reports a single aggregate `dependencies_healthy` boolean rather than
    naming individual backends/integrations, since this endpoint has no
    authentication (see sast/missingauth-results.md).
    """
    dependencies_healthy = True

    # Check database connectivity
    try:
        from backend.database.engine import health_check as db_health_check

        if not await db_health_check():
            dependencies_healthy = False
    except Exception:
        logging.getLogger(__name__).debug("Database health check failed", exc_info=True)
        dependencies_healthy = False

    # Check cache connectivity
    if settings.USE_CELERY:
        try:
            import redis  # ty: ignore[unresolved-import]

            r = redis.from_url(
                settings.CELERY_BROKER_URL,
                socket_connect_timeout=settings.REDIS_HEALTHCHECK_TIMEOUT_SECONDS,
            )
            r.ping()
        except Exception:
            logging.getLogger(__name__).debug("Cache health check failed", exc_info=True)
            dependencies_healthy = False

    return DetailedHealthResponse(
        status="healthy" if dependencies_healthy else "degraded",
        timestamp=datetime.now(UTC),
        version=settings.APP_VERSION,
        environment=settings.environment_name,
        uptime_seconds=time.time() - START_TIME,
        dependencies_healthy=dependencies_healthy,
    )


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness probe: fits a tiny sklearn pipeline end-to-end.

    Catches dependency breakage (sklearn/polars upgrade, broken install)
    that the basic /health check misses. Returns 503 if the fit fails so
    load balancers can pull the instance.
    """
    import numpy as np
    from fastapi.responses import JSONResponse
    from sklearn.preprocessing import StandardScaler

    t0 = time.monotonic()
    try:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        scaler.fit_transform(X)
        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
        return {"status": "ready", "fit_ms": elapsed_ms}
    except Exception as exc:
        logging.getLogger(__name__).error("Readiness probe failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": str(exc)},
        )


@router.get("/ping")
async def ping():
    """Simple ping endpoint."""
    return {"message": "pong"}
