"""
FastAPI Admin Module

Modern async admin interface for system management.
Provides data ingestion management, user administration, and system monitoring.
"""

from .services import AsyncAdminDataService, AsyncAdminUserService, AsyncAdminSystemService
from .routes import admin_router

__all__ = [
    "AsyncAdminDataService",
    "AsyncAdminUserService",
    "AsyncAdminSystemService",
    "admin_router"
]
