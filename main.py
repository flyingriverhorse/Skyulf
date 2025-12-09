"""
FastAPI MLops Application Entry Point

© 2025 Murat Unsal — Skyulf Project

This module creates and configures the FastAPI application instance.
It provides better concurrency support compared to the Flask implementation.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
import logging
import time
from typing import AsyncGenerator
from pathlib import Path

# Use absolute imports to fix import issues
from config import get_settings

from core.health.routes import router as health_router
# from core.feature_engineering.routes import router as feature_engineering_router
from core.ml_pipeline.api import router as ml_pipeline_router
from core.ml_pipeline.deployment.api import router as deployment_router
from core.data_ingestion.router import router as data_ingestion_router, sources_router as data_sources_router
from middleware.error_handler import ErrorHandlerMiddleware
from middleware.logging import LoggingMiddleware
from core.database.engine import init_db, close_db, create_tables
from core.exceptions.handlers import (
    not_found_exception_handler,
    validation_exception_handler,
    method_not_allowed_exception_handler,
    generic_http_exception_handler
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
settings = get_settings()

OPENAPI_DESCRIPTION = (
    f"{settings.APP_DESCRIPTION}\n\n"
    "### Highlights\n"
    "- **Data Management**: Upload datasets, explore sources, and clean ML assets.\n"
    "- **Feature Engineering**: Launch automated feature pipelines and inspect results.\n"
    "- **Monitoring**: Track system health via lightweight readiness and detailed diagnostics.\n"
)

OPENAPI_TAGS = [
    {"name": "Data Management", "description": "Dataset ingestion, lifecycle, and catalog management APIs."},
    {"name": "ml-workflow", "description": "Feature engineering and ML workflow orchestration endpoints."},
    {"name": "health", "description": "Health and readiness probes consumed by monitoring systems."},
]


def _build_swagger_ui_parameters() -> dict:
    """Create swagger UI configuration from settings."""
    params = {
        "persistAuthorization": settings.API_DOCS_PERSIST_AUTH,
        "displayRequestDuration": settings.API_DOCS_DISPLAY_REQUEST_DURATION,
        "defaultModelsExpandDepth": settings.API_DOCS_DEFAULT_MODELS_EXPAND_DEPTH,
        "docExpansion": settings.API_DOCS_DEFAULT_DOC_EXPANSION,
        "filter": settings.API_DOCS_ENABLE_FILTER,
        "tryItOutEnabled": settings.API_DOCS_ENABLE_TRY_IT_OUT,
    }
    return {key: value for key, value in params.items() if value is not None}


def _configure_openapi(app: FastAPI) -> None:
    """Attach a custom OpenAPI schema builder with enriched metadata."""
    servers = settings.API_DOCS_SERVERS or [f"http://{settings.HOST}:{settings.PORT}"]

    def custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=settings.APP_NAME,
            version=settings.APP_VERSION,
            description=OPENAPI_DESCRIPTION,
            routes=app.routes,
        )
        openapi_schema["info"]["summary"] = settings.APP_SUMMARY

        if settings.API_TOS_URL:
            openapi_schema["info"]["termsOfService"] = settings.API_TOS_URL

        contact = {}
        if settings.API_CONTACT_NAME:
            contact["name"] = settings.API_CONTACT_NAME
        if settings.API_CONTACT_EMAIL:
            contact["email"] = settings.API_CONTACT_EMAIL
        if settings.API_CONTACT_URL:
            contact["url"] = settings.API_CONTACT_URL
        if contact:
            openapi_schema["info"]["contact"] = contact

        if settings.API_LICENSE_NAME:
            license_info = {"name": settings.API_LICENSE_NAME}
            if settings.API_LICENSE_URL:
                license_info["url"] = settings.API_LICENSE_URL
            openapi_schema["info"]["license"] = license_info

        if settings.API_LOGO_URL:
            openapi_schema["info"]["x-logo"] = {"url": settings.API_LOGO_URL}

        openapi_schema["tags"] = OPENAPI_TAGS
        openapi_schema["servers"] = [{"url": url} for url in servers]

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting FastAPI MLops Application")
    start_time = time.time()

    # Initialize database connections
    await init_db()
    logger.info("✅ Database initialized")

    # Create database tables if they don't exist
    await create_tables()
    logger.info("✅ Database tables created/verified")

    # Initialize other services here as needed
    # await init_cache()
    # await init_background_tasks()

    startup_time = time.time() - start_time
    logger.info(f"🎉 Application started in {startup_time:.2f} seconds")

    yield

    # Shutdown
    logger.info("🛑 Shutting down FastAPI MLops Application")
    await close_db()
    logger.info("✅ Application shutdown complete")


def create_app() -> FastAPI:
    """
    FastAPI application factory.

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    docs_enabled = settings.API_DOCS_ENABLED
    if docs_enabled is None:
        docs_enabled = settings.DEBUG

    app = FastAPI(
        title=settings.APP_NAME,
        description=OPENAPI_DESCRIPTION,
        version=settings.APP_VERSION,
        docs_url=settings.API_DOCS_URL if docs_enabled else None,
        redoc_url=settings.API_REDOC_URL if docs_enabled else None,
        openapi_url=settings.API_OPENAPI_URL if docs_enabled else None,
        lifespan=lifespan,
        swagger_ui_parameters=_build_swagger_ui_parameters(),
    )

    # Configure templates and static files
    _setup_templates_and_static(app)

    # Add middleware (order matters!)
    _add_middleware(app, settings)

    # Include routers
    _include_routers(app)

    # Add global exception handlers
    _add_exception_handlers(app)

    # Enrich OpenAPI schema once routers are registered
    _configure_openapi(app)

    return app


def _setup_templates_and_static(app: FastAPI) -> None:
    """Setup templates and static files for the FastAPI application."""

    # Define paths
    base_dir = Path(__file__).parent
    static_dir = base_dir / "static"

    # Verify static directory exists
    if not static_dir.exists():
        logger.warning(f"Static directory not found: {static_dir}")
        static_dir.mkdir(parents=True, exist_ok=True)

    # Setup static files
    try:
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info(f"✅ Static files mounted from: {static_dir}")
    except Exception as e:
        logger.error(f"❌ Failed to mount static files: {e}")


def _add_middleware(app: FastAPI, settings) -> None:
    """Add middleware to the FastAPI application."""

    # Security middleware
    if settings.ALLOWED_HOSTS:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS
        )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)


def _include_routers(app: FastAPI) -> None:
    """Include all API routers."""
    # Include data ingestion router first so its API routes (e.g. /data/api/sources/search)
    # are registered before the template-level routes which may define overlapping paths.
    app.include_router(data_ingestion_router)
    app.include_router(data_sources_router)

    # Health check (no prefix, available at /health)
    app.include_router(health_router, tags=["health"])

    # Feature engineering routes (prototype API)
    try:
        app.include_router(feature_engineering_router)
    except Exception:
        logger.debug("feature_engineering_router not available to include")

    # ML Pipeline
    # Note: ml_pipeline_router already has prefix="/api/pipeline" defined in its file
    # But we double check to avoid double prefixing if we change it later.
    # For now, let's remove the prefix here since we added it to the router definition.
    app.include_router(ml_pipeline_router)
    app.include_router(deployment_router, prefix="/api", tags=["Deployment"])

    # Root redirect
    from fastapi.responses import RedirectResponse
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")



def _add_exception_handlers(app: FastAPI) -> None:
    """Add global exception handlers for both HTML and JSON responses."""
    from fastapi import HTTPException
    from starlette.exceptions import HTTPException as StarletteHTTPException

    # Handle 404 Not Found
    app.add_exception_handler(404, not_found_exception_handler)

    # Handle 405 Method Not Allowed
    app.add_exception_handler(405, method_not_allowed_exception_handler)

    # Handle 422 Validation Error
    app.add_exception_handler(422, validation_exception_handler)

    # Handle 500 Internal Server Error (and other 5xx server errors)
    app.add_exception_handler(500, generic_http_exception_handler)

    # Generic HTTP exception handler for other status codes
    app.add_exception_handler(HTTPException, generic_http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, generic_http_exception_handler)

    # Add general exception handler for unhandled Python exceptions
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected Python exceptions and convert them to 500 errors."""
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        # Convert to HTTPException for consistent handling
        http_exc = HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(exc)}"
        )
        return await generic_http_exception_handler(request, http_exc)


# Create the application instance
app = create_app()


# For development server
if __name__ == "__main__":
    import uvicorn
    settings = get_settings()

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning"
    )
