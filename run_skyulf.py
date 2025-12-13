"""
FastAPI Application Startup Script

© 2025 Murat Unsal — Skyulf Project

This script starts the FastAPI application with proper configuration.
It replaces the Flask run.py with modern async server capabilities.
"""

import logging
import sys

from backend.config import get_settings, setup_universal_logging
from backend.main import app


def setup_logging():
    """Setup logging configuration."""
    settings = get_settings()
    # Use the unified logging setup which supports time-based rotation
    setup_universal_logging(
        log_file=settings.LOG_FILE,
        log_level=settings.LOG_LEVEL,
        rotation_type=settings.LOG_ROTATION_TYPE,
        rotation_when=settings.LOG_ROTATION_WHEN,
        rotation_interval=settings.LOG_ROTATION_INTERVAL,
        max_bytes=settings.LOG_MAX_SIZE,
        backup_count=settings.LOG_BACKUP_COUNT,
    )


def main():
    """Main entry point for the FastAPI application."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Get settings
    settings = get_settings()

    # Log startup information
    logger.info(f"[START] Starting {settings.APP_NAME}")
    logger.info(f"[ENV] Environment: {'Development' if settings.DEBUG else 'Production'}")
    logger.info(f"[HOST] Host: {settings.HOST}:{settings.PORT}")

    # Import uvicorn here to avoid import errors if not installed
    try:
        import uvicorn
    except ImportError:
        logger.error("[ERROR] uvicorn not installed. Please run: pip install uvicorn[standard]")
        sys.exit(1)

    # Development vs Production server configuration
    if settings.DEBUG:
        # Development server with auto-reload
        logger.info("🔧 Running in development mode with auto-reload")
        uvicorn.run(
            "backend.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=True,
            reload_dirs=["."],
            log_level="info",
            access_log=True,
            use_colors=True
        )
    else:
        # Production server
        logger.info("🏭 Running in production mode")
        uvicorn.run(
            app,
            host=settings.HOST,
            port=settings.PORT,
            workers=settings.WORKERS,
            log_level="warning",
            access_log=False,
            server_header=False,
            date_header=False
        )


if __name__ == "__main__":
    main()
