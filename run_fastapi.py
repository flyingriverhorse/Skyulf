"""
FastAPI Application Startup Script

© 2025 Murat Unsal — Skyulf Project

This script starts the FastAPI application with proper configuration.
It replaces the Flask run.py with modern async server capabilities.
"""

import logging
import os
import subprocess
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


def start_celery_worker():
    """Start the Celery worker in a separate process."""
    settings = get_settings()
    if not settings.USE_CELERY:
        print("🚫 Celery worker disabled by configuration.")
        return None

    if sys.platform == "win32":
        pool_arg = "--pool=solo"
    else:
        pool_arg = "--pool=prefork"

    cmd = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "celery_worker.celery_app",
        "worker",
        pool_arg,
        "--loglevel=info",
        "--queues",
        "mlops-training",
    ]

    print(f"👷 Starting Celery worker: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=os.getcwd())


def check_redis_availability():
    """Check if Redis is running and accessible."""
    settings = get_settings()
    try:
        import redis

        # Use a short timeout to avoid hanging
        client = redis.from_url(settings.CELERY_BROKER_URL, socket_connect_timeout=1)
        client.ping()
        return True
    except Exception:
        return False


def main():
    """Main entry point for the FastAPI application."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Get settings
    settings = get_settings()

    # Log startup information
    logger.info(f"[START] Starting {settings.APP_NAME}")
    logger.info(
        f"[ENV] Environment: {'Development' if settings.DEBUG else 'Production'}"
    )
    logger.info(f"[HOST] Host: {settings.HOST}:{settings.PORT}")

    # Import uvicorn here to avoid import errors if not installed
    try:
        import uvicorn
    except ImportError:
        logger.error(
            "[ERROR] uvicorn not installed. Please run: pip install uvicorn[standard]"
        )
        sys.exit(1)

    celery_process = None

    # Development vs Production server configuration
    if settings.DEBUG:
        # Start Celery worker in development mode if Redis is available
        if check_redis_availability():
            try:
                celery_process = start_celery_worker()
            except Exception as e:
                logger.error(f"[ERROR] Failed to start Celery worker: {e}")
        else:
            logger.warning("[WARN] Redis not found. Celery worker will NOT be started.")
            logger.warning(
                "   To enable background tasks, start Redis: docker-compose up -d redis"
            )

        # Development server with auto-reload
        logger.info("[DEV] Running in development mode with auto-reload")
        try:
            uvicorn.run(
                "core.main:app",
                host=settings.HOST,
                port=settings.PORT,
                reload=True,
                reload_dirs=["."],
                log_level="info",
                access_log=True,
                use_colors=True,
            )
        finally:
            if celery_process:
                logger.info("🛑 Stopping Celery worker...")
                celery_process.terminate()
                celery_process.wait()
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
            date_header=False,
        )


if __name__ == "__main__":
    main()
