"""
Page routes for FastAPI application.

This module contains HTML template routes that were previously in main.py.
Separating these routes helps keep main.py clean and focused on application setup.
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import logging

logger = logging.getLogger(__name__)


def add_page_routes(app: FastAPI) -> None:
    """Add HTML page routes to the FastAPI application."""

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Main index page with configurable authentication"""
        from core.auth.page_security import create_page_response

        # Use the centralized page response helper
        return create_page_response(
            request=request,
            template_name="index.html",
            extra_context={
                "page_title": "ML Automation Platform"
            }
        )

    @app.get("/data/preview/{source_id}", response_class=HTMLResponse)
    async def data_preview_page(request: Request, source_id: str):
        """Data preview page for uploaded files with configurable authentication"""
        from core.auth.page_security import create_page_response

        # Use the centralized page response helper
        return create_page_response(
            request=request,
            template_name="data_preview.html",
            extra_context={
                "source_id": source_id,
                "page_title": f"Data Preview - {source_id}",
                # Legacy routes for compatibility
                "routes": {
                    'home': '/',
                    'data_ingestion': '/data-ingestion',
                    'admin.dashboard': '/admin',
                },
                # Legacy user context for compatibility - will be properly set by create_page_response
                "user": None,  # Triggers override to current_user
                "permissions": [],
                "is_admin": False  # Triggers override to user_permissions.is_admin
            }
        )

    @app.get("/data-ingestion", response_class=HTMLResponse)
    async def data_ingestion_page(request: Request):
        """Data ingestion page with configurable authentication"""
        from core.auth.page_security import create_page_response

        # Use the centralized page response helper
        return create_page_response(
            request=request,
            template_name="data_ingestion.html",
            extra_context={
                "page_title": "Data Ingestion"
            }
        )

    @app.get("/ml-workflow", response_class=HTMLResponse)
    async def ml_workflow_page(request: Request):
        """ML workflow canvas shell page."""
        from core.auth.page_security import create_page_response

        # Check if we should serve the new V2 canvas
        # For now, we'll default to V2, but we could use a feature flag
        return create_page_response(
            request=request,
            template_name="ml_canvas.html",
            extra_context={
                "page_title": "ML Workflow Canvas V2"
            }
        )

    @app.get("/ml-workflow-v1", response_class=HTMLResponse)
    async def ml_workflow_v1_page(request: Request):
        """Legacy ML workflow canvas page."""
        from core.auth.page_security import create_page_response

        return create_page_response(
            request=request,
            template_name="feature_canvas.html",
            extra_context={
                "page_title": "ML Workflow Canvas (Legacy)"
            }
        )
