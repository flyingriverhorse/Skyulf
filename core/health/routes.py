"""
Health Check Endpoints

Basic health and status endpoints for monitoring and load balancer checks.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
import time
from datetime import datetime, timezone

from core.dependencies import get_config
from core.config import Settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    environment: str
    uptime_seconds: float


class DetailedHealthResponse(HealthResponse):
    """Detailed health check with additional information."""
    database_status: str
    cache_status: str
    external_services: dict


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_config)):
    """
    Basic health check endpoint.
    Returns simple status information for load balancers.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version="2.0.0",
        environment="development" if settings.DEBUG else "production",
        uptime_seconds=time.time() - getattr(health_check, "_start_time", time.time())
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(settings: Settings = Depends(get_config)):
    """
    Detailed health check endpoint.
    Includes status of various system components.
    """
    # Check database connectivity
    try:
        from core.database.engine import health_check as db_health_check
        database_status = "healthy" if await db_health_check() else "unhealthy"
    except Exception:
        database_status = "error"

    # Check cache connectivity
    cache_status = "healthy"  # TODO: Implement cache health check if using Redis

    # Check external services
    external_services = {
        "snowflake": "not_configured" if not settings.SNOWFLAKE_ACCOUNT else "healthy",
    }

    return DetailedHealthResponse(
        status="healthy" if database_status == "healthy" else "degraded",
        timestamp=datetime.now(timezone.utc),
        version="2.0.0",
        environment="development" if settings.DEBUG else "production",
        uptime_seconds=time.time() - getattr(health_check, "_start_time", time.time()),
        database_status=database_status,
        cache_status=cache_status,
        external_services=external_services
    )


@router.get("/ping")
async def ping():
    """Simple ping endpoint."""
    return {"message": "pong"}


# Initialize start time for uptime calculation
health_check._start_time = time.time()
