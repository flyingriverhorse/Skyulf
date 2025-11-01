"""
FastAPI Application Startup Script

This script starts the FastAPI application with proper configuration.
It replaces the Flask run.py with modern async server capabilities.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

# Import after path setup
from config import get_settings
from main import app


def setup_logging():
    """Setup logging configuration."""
    settings = get_settings()
    
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=settings.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(settings.LOG_FILE) if settings.LOG_FILE else logging.NullHandler()
        ]
    )


def main():
    """Main entry point for the FastAPI application."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Get settings
    settings = get_settings()
    
    # Log startup information
    logger.info(f"🚀 Starting {settings.APP_NAME}")
    logger.info(f"🌍 Environment: {'Development' if settings.DEBUG else 'Production'}")
    logger.info(f"🏠 Host: {settings.HOST}:{settings.PORT}")
    
    # Import uvicorn here to avoid import errors if not installed
    try:
        import uvicorn
    except ImportError:
        logger.error("❌ uvicorn not installed. Please run: pip install uvicorn[standard]")
        sys.exit(1)
    
    # Development vs Production server configuration
    if settings.DEBUG:
        # Development server with auto-reload
        logger.info("🔧 Running in development mode with auto-reload")
        uvicorn.run(
            "main:app",
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