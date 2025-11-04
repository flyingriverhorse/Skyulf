"""
FastAPI MLops Application Entry Point

© 2025 Murat Unsal — Skyulf Project

This module creates and configures the FastAPI application instance.
It provides better concurrency support compared to the Flask implementation.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging
import time
from typing import AsyncGenerator
from pathlib import Path

# Use absolute imports to fix import issues
from config import get_settings
from core.auth import auth_router, auth_template_router
from core.admin.routes import admin_router
from core.health.routes import router as health_router
from core.data_ingestion.routers.data_routes import data_router
from core.user_management.routes import user_router, current_user_router, public_user_router
from core.llm.routes import llm_router
from core.feature_engineering.routes import router as feature_engineering_router
from core.templates import setup_templates
from core.pages import add_page_routes
from middleware.error_handler import ErrorHandlerMiddleware
from middleware.logging import LoggingMiddleware
from core.database.engine import init_db, close_db, create_tables
from core.exceptions.handlers import (
    not_found_exception_handler,
    unauthorized_exception_handler,
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    settings = get_settings()
    
    # Startup
    logger.info("🚀 Starting FastAPI MLops Application")
    start_time = time.time()
    
    # Initialize database connections
    await init_db()
    logger.info("✅ Database initialized")
    
    # Create database tables if they don't exist
    await create_tables()
    logger.info("✅ Database tables created/verified")
    
    # Run startup maintenance if enabled
    try:
        from core.utils.maintenance import run_startup_maintenance
        await run_startup_maintenance()
    except Exception as e:
        logger.warning(f"⚠️ Startup maintenance failed: {e}")
    
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
    settings = get_settings()
    
    # Create FastAPI instance
    app = FastAPI(
        title="MLops Platform API",
        description="Modern MLops platform with data ingestion, model management, and LLM capabilities",
        version="2.0.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
        # Enable OpenAPI schema generation
        openapi_url="/openapi.json" if settings.DEBUG else None,
    )
    
    # Configure templates and static files
    _setup_templates_and_static(app)
    
    # Add middleware (order matters!)
    _add_middleware(app, settings)
    
    # Include routers
    _include_routers(app)
    
    # Add global exception handlers
    _add_exception_handlers(app)
    
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
    
    # Setup templates using the utility function
    try:
        templates = setup_templates(base_dir)
        
        # Store templates in app state for use in routes
        app.state.templates = templates
        logger.info(f"✅ Templates initialized from: {base_dir / 'templates'}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize templates: {e}")
        raise


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
    app.include_router(data_router)

    # Add template routes next
    add_page_routes(app)
    
    # Health check (no prefix, available at /health)
    app.include_router(health_router, tags=["health"])
    
    # === CORE DOMAIN ROUTERS (Business Logic Layer) ===
    # Authentication routes (from core/auth)
    app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
    app.include_router(auth_template_router, tags=["auth-pages"])  # Template endpoints
    
    # Admin routes (from core/admin) - admin_router already has /admin prefix
    app.include_router(admin_router, tags=["admin"])
    
    # User Management routes (from core/user_management)
    app.include_router(user_router, prefix="/api", tags=["user-management-admin"])
    app.include_router(current_user_router, prefix="/api", tags=["user-profile"])
    app.include_router(public_user_router, prefix="/api", tags=["user-public"])
    
    # Data ingestion routes (from core/data_ingestion/routers)
    # The router's prefix is '/data' so include it without additional prefix to keep paths like '/data/api/sources/*'
    app.include_router(data_router)

    # LLM routes (from core/llm)
    try:
        app.include_router(llm_router)
    except Exception:
        # If router import failed or router not available, skip gracefully
        logger.debug("llm_router not available to include")

    # Feature engineering routes (prototype API)
    try:
        app.include_router(feature_engineering_router)
    except Exception:
        logger.debug("feature_engineering_router not available to include")


def _add_exception_handlers(app: FastAPI) -> None:
    """Add global exception handlers for both HTML and JSON responses."""
    from fastapi import HTTPException
    from starlette.exceptions import HTTPException as StarletteHTTPException
    
    # Handle 404 Not Found
    app.add_exception_handler(404, not_found_exception_handler)
    
    # Handle 401 Unauthorized and 403 Forbidden
    app.add_exception_handler(401, unauthorized_exception_handler)
    app.add_exception_handler(403, unauthorized_exception_handler)
    
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
    
# === TEMPORARY TEST ROUTES FOR ERROR PAGES ===
    @app.get("/test-errors")
    async def test_errors_index():
        """Index page with links to preview the minimalist error templates."""
        html_content = """
        <!DOCTYPE html>
        <html lang=\"en\">
        <head>
            <meta charset=\"utf-8\">
            <title>Error Page Playground</title>
        </head>
        <body>
            <h1>Error page playground</h1>
            <p>Use these links to preview the lightweight error templates. Keep the tab open while tweaking copy or wording.</p>
            <ul>
                <li><a href=\"/test-errors/404\">404 — not found</a></li>
                <li><a href=\"/test-errors/unauthorized\">401 — unauthorized</a></li>
                <li><a href=\"/test-errors/forbidden\">403 — forbidden</a></li>
                <li><a href=\"/test-errors/validation\">422 — validation error</a></li>
                <li><a href=\"/test-errors/method-not-allowed\">405 — method not allowed</a></li>
                <li><a href=\"/test-errors/server-error\">500 — server error</a></li>
            </ul>
            <h2>HTTP method quick checks</h2>
            <div>
                <button onclick=\"triggerMethod('POST')\">POST → 405</button>
                <button onclick=\"triggerMethod('PUT')\">PUT → 405</button>
                <button onclick=\"triggerMethod('DELETE')\">DELETE → 405</button>
                <button onclick=\"inspectHeaders()\">Inspect headers</button>
            </div>
            <p>The buttons call /test-errors/method-not-allowed with the selected verb. You should see the 405 page if everything is wired up.</p>
            <script>
                async function triggerMethod(method) {
                    try {
                        const response = await fetch('/test-errors/method-not-allowed', { method });
                        const text = await response.text();
                        document.open();
                        document.write(text);
                        document.close();
                    } catch (error) {
                        alert('Request failed: ' + error.message);
                    }
                }
                async function inspectHeaders() {
                    try {
                        const res = await fetch('/test-errors/debug-headers');
                        const data = await res.json();
                        alert(`UA: ${data.user_agent}\nAccept: ${data.accept}\nBrowser detected: ${data.is_browser_request}`);
                    } catch (error) {
                        alert('Could not read headers: ' + error.message);
                    }
                }
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    @app.get("/test-errors/404")
    async def test_not_found():
        """Test route to trigger 404 Not Found error page."""
        raise HTTPException(status_code=404, detail="We looked everywhere but that page isn't on this server.")

    @app.get("/test-errors/unauthorized")
    async def test_unauthorized():
        """Test route to trigger 401 Unauthorized error page."""
        raise HTTPException(status_code=401, detail="Please sign in again before continuing.")

    @app.get("/test-errors/forbidden")
    async def test_forbidden():
        """Test route to trigger 403 Forbidden error page."""
        raise HTTPException(status_code=403, detail="Your account doesn't have access to that area yet.")

    @app.get("/test-errors/debug-headers")
    async def debug_headers(request: Request):
        """Debug route to check request headers and browser detection."""
        from core.exceptions.handlers import is_browser_request

        headers_info = dict(request.headers)
        return {
            "is_browser_request": is_browser_request(request),
            "headers": headers_info,
            "method": request.method,
            "url": str(request.url),
            "accept": headers_info.get("accept", "Not provided"),
            "user_agent": headers_info.get("user-agent", "Not provided")[:100]
        }

    @app.get("/test-errors/server-error")
    async def test_server_error():
        """Test route to trigger 500 Server Error page."""
        raise HTTPException(status_code=500, detail="An unexpected error bubbled up in the backend. We're on it.")

    @app.get("/test-errors/validation")
    async def test_validation():
        """Test route to trigger 422 Validation Error page."""
        raise HTTPException(status_code=422, detail="Something in the payload didn't pass validation.")

    @app.get("/test-errors/method-not-allowed")
    async def test_method_not_allowed_get():
        """Test route that only allows GET method - other methods will trigger 405."""
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html lang=\"en\">
        <head>
            <meta charset=\"utf-8\">
            <title>Method playground</title>
        </head>
        <body>
            <h1>Method tester</h1>
            <p>This endpoint only speaks GET. Try a POST, PUT, or DELETE to confirm the 405 template.</p>
            <div>
                <button onclick=\"invoke('POST')\">POST</button>
                <button onclick=\"invoke('PUT')\">PUT</button>
                <button onclick=\"invoke('DELETE')\">DELETE</button>
            </div>
            <p><a href=\"/test-errors\">Back to previews</a></p>
            <script>
                async function invoke(method) {
                    try {
                        const res = await fetch('/test-errors/method-not-allowed', { method });
                        const html = await res.text();
                        document.open();
                        document.write(html);
                        document.close();
                    } catch (error) {
                        alert('Request failed: ' + error.message);
                    }
                }
            </script>
        </body>
        </html>
        """)

    # This route will properly trigger 405 for non-GET methods
    # FastAPI automatically handles method validation


    # Keep existing 500 handler but improve it
    @app.exception_handler(500)
    async def internal_server_error_handler(request: Request, exc: Exception):
        logger.error(f"Internal server error: {exc}", exc_info=True)
        
        # Check if it's a browser request
        from core.exceptions.handlers import is_browser_request, get_templates_from_app
        
        if is_browser_request(request):
            templates = get_templates_from_app(request)
            settings = get_settings()
            context = {
                "request": request,
                "error_code": "500",
                "error_message": "The server hit an unexpected snag. We're already on it.",
                "request_id": getattr(request.state, "request_id", None),
                "config": {"DEBUG": bool(getattr(settings, "DEBUG", False))},
                "error_details": str(exc)
            }
            return templates.TemplateResponse("errors/500.html", context, status_code=500)
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Internal Server Error",
                    "message": str(exc),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )


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