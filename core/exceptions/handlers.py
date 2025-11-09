"""
Custom Exception Handlers for HTML Error Pages

Provides exception handlers that return HTML error pages instead of JSON
when requests come from browsers (based on Accept header).
"""

import logging
from typing import Union

from fastapi import Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import get_settings

logger = logging.getLogger(__name__)


def _get_template_debug_context() -> dict:
    """Return a lightweight config-like object for templates."""
    try:
        settings = get_settings()
        return {"DEBUG": bool(getattr(settings, "DEBUG", False))}
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Falling back to safe debug context: %s", exc)
        return {"DEBUG": False}


def get_templates_from_app(request: Request):
    """Get templates instance from the FastAPI app state"""
    try:
        if hasattr(request.app.state, 'templates'):
            return request.app.state.templates
    except Exception as e:
        logger.warning("Could not access app state templates: %s", e)

    # Fallback: create templates instance
    try:
        from fastapi.templating import Jinja2Templates
        from pathlib import Path
        base_dir = Path(__file__).parent.parent.parent
        templates_dir = base_dir / "templates"
        logger.info("Creating fallback templates instance from: %s", templates_dir)
        return Jinja2Templates(directory=str(templates_dir))
    except Exception as e:
        logger.error("Failed to create templates instance: %s", e)
        raise


def is_browser_request(request: Request) -> bool:
    """
    Determine if the request comes from a browser based on Accept header.

    Args:
        request: FastAPI request object

    Returns:
        bool: True if request appears to be from a browser
    """
    accept = request.headers.get("accept", "").lower()
    user_agent = request.headers.get("user-agent", "").lower()

    # Check if request accepts HTML - be more lenient for fetch requests
    accepts_html = "text/html" in accept or "*/*" in accept

    # Check if it's likely from a browser based on user agent
    browser_indicators = ["mozilla", "chrome", "safari", "edge", "firefox"]
    is_browser = any(indicator in user_agent for indicator in browser_indicators)

    # API clients typically don't accept HTML or have specific user agents
    api_indicators = ["curl", "postman", "insomnia", "httpie", "python-requests", "axios"]
    is_api_client = any(indicator in user_agent for indicator in api_indicators)

    # For fetch requests from browsers, prioritize browser detection over accept headers
    # since fetch() doesn't automatically send text/html in accept header
    if is_browser and not is_api_client:
        logger.debug(
            "Browser request detected: UA=%s, Accept=%s",
            user_agent[:50],
            accept[:50],
        )
        return True

    # If explicitly requesting JSON as first priority, treat as API request
    if accept.startswith("application/json"):
        logger.debug("API request detected: Accept starts with application/json")
        return False

    logger.debug(
        "Request type: accepts_html=%s, is_browser=%s, is_api_client=%s",
        accepts_html,
        is_browser,
        is_api_client,
    )
    return accepts_html and is_browser and not is_api_client


async def not_found_exception_handler(request: Request, exc: HTTPException) -> Union[HTMLResponse, JSONResponse]:
    """
    Handle 404 Not Found errors.

    Args:
        request: FastAPI request object
        exc: HTTPException with status 404

    Returns:
        HTMLResponse or JSONResponse based on request type
    """
    try:
        if is_browser_request(request):
            # Return HTML page for browser requests
            try:
                templates = get_templates_from_app(request)
                default_message = (
                    "The page you're looking for doesn't exist. "
                    "It might have been moved or deleted."
                )
                context = {
                    "request": request,
                    "error_message": getattr(exc, "detail", default_message),
                    "request_id": getattr(request.state, "request_id", None),
                }
                return templates.TemplateResponse("errors/404.html", context, status_code=404)
            except Exception as template_error:
                logger.error("Failed to render 404 template: %s", template_error)
                # Return simple HTML fallback
                return HTMLResponse(
                    content=(
                        "<html><body><h1>404 - Page Not Found</h1>"
                        "<p>The page you're looking for doesn't exist.</p>"
                        "</body></html>"
                    ),
                    status_code=404
                )
        else:
            # Return JSON for API requests
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": "Not Found",
                    "message": getattr(exc, 'detail', 'The requested resource was not found'),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
    except Exception as e:
        logger.error("Error in 404 handler: %s", e, exc_info=True)
        # Fallback to simple JSON response
        return JSONResponse(
            status_code=404,
            content={"error": "Not Found", "message": "Resource not found"}
        )


async def unauthorized_exception_handler(request: Request, exc: HTTPException) -> Union[HTMLResponse, JSONResponse]:
    """
    Handle 401 Unauthorized and 403 Forbidden errors.

    Args:
        request: FastAPI request object
        exc: HTTPException with status 401 or 403

    Returns:
        HTMLResponse or JSONResponse based on request type
    """
    try:
        status_code = exc.status_code

        if is_browser_request(request):
            # Return appropriate HTML page for browser requests
            try:
                templates = get_templates_from_app(request)
                template_name = "errors/401.html" if status_code == 401 else "errors/403.html"
                if status_code == 401:
                    default_msg = "You need to sign in to access this page."
                else:
                    default_msg = "You don't have permission to access this resource."
                context = {
                    "request": request,
                    "error_code": str(status_code),
                    "error_message": getattr(exc, "detail", default_msg),
                    "request_id": getattr(request.state, "request_id", None),
                }
                return templates.TemplateResponse(template_name, context, status_code=status_code)
            except Exception as template_error:
                logger.error("Failed to render %s template: %s", status_code, template_error)
                # Return simple HTML fallback
                title = "Unauthorized" if status_code == 401 else "Forbidden"
                return HTMLResponse(
                    content=(
                        f"<html><body><h1>{status_code} - {title}</h1>"
                        f"<p>{getattr(exc, 'detail', 'Access denied')}</p>"
                        "</body></html>"
                    ),
                    status_code=status_code
                )
        else:
            # Return JSON for API requests
            error_msg = "Could not validate credentials" if status_code == 401 else "Access forbidden"
            return JSONResponse(
                status_code=status_code,
                content={
                    "success": False,
                    "error": "Unauthorized" if status_code == 401 else "Forbidden",
                    "message": getattr(exc, "detail", error_msg),
                    "request_id": getattr(request.state, "request_id", None),
                }
            )
    except Exception as e:
        logger.error("Error in unauthorized handler: %s", e, exc_info=True)
        # Fallback to simple JSON response
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": getattr(exc, "detail", "Access denied")}
        )


async def validation_exception_handler(request: Request, exc: HTTPException) -> Union[HTMLResponse, JSONResponse]:
    """
    Handle 422 Validation errors.

    Args:
        request: FastAPI request object
        exc: HTTPException with status 422

    Returns:
        HTMLResponse or JSONResponse based on request type
    """
    try:
        if is_browser_request(request):
            # Return HTML error page for browser requests
            try:
                templates = get_templates_from_app(request)
                default_msg = (
                    "The data you submitted couldn't be processed. "
                    "Please check your input and try again."
                )
                context = {
                    "request": request,
                    "error_code": "422",
                    "error_message": getattr(exc, "detail", default_msg),
                    "request_id": getattr(request.state, "request_id", None),
                    "error_details": str(getattr(exc, "detail", "")),
                }
                return templates.TemplateResponse("errors/422.html", context, status_code=422)
            except Exception as template_error:
                logger.error("Failed to render 422 template: %s", template_error)
                # Return simple HTML fallback
                return HTMLResponse(
                    content=(
                        "<html><body><h1>422 - Validation Error</h1>"
                        "<p>The data you submitted couldn't be processed.</p>"
                        "</body></html>"
                    ),
                    status_code=422
                )
        else:
            # Return JSON for API requests
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": "Unprocessable Entity",
                    "message": getattr(exc, "detail", "Validation error"),
                    "request_id": getattr(request.state, "request_id", None),
                }
            )
    except Exception as e:
        logger.error("Error in 422 handler: %s", e, exc_info=True)
        # Fallback to simple JSON response
        return JSONResponse(
            status_code=422,
            content={"error": "Unprocessable Entity", "message": "Validation error"}
        )


async def method_not_allowed_exception_handler(
    request: Request,
    exc: HTTPException,
) -> Union[HTMLResponse, JSONResponse]:
    """
    Handle 405 Method Not Allowed errors.

    Args:
        request: FastAPI request object
        exc: HTTPException with status 405

    Returns:
        HTMLResponse or JSONResponse based on request type
    """
    try:
        if is_browser_request(request):
            # Get allowed methods from headers or default list
            allowed_methods = request.headers.get("Allow", "GET, POST")

            # Return HTML error page for browser requests
            try:
                templates = get_templates_from_app(request)
                default_msg = (
                    f"The HTTP method {request.method} is not allowed for this "
                    "endpoint."
                )
                context = {
                    "request": request,
                    "error_message": getattr(exc, "detail", default_msg),
                    "request_method": request.method,
                    "allowed_methods": allowed_methods,
                    "request_id": getattr(request.state, "request_id", None),
                    "request_url": str(request.url),
                }
                return templates.TemplateResponse("errors/405.html", context, status_code=405)
            except Exception as template_error:
                logger.error("Failed to render 405 template: %s", template_error)
                # Return simple HTML fallback
                return HTMLResponse(
                    content=(
                        "<html><body><h1>405 - Method Not Allowed</h1>"
                        f"<p>The HTTP method {request.method} is not allowed.</p>"
                        "</body></html>"
                    ),
                    status_code=405
                )
        else:
            # Return JSON for API requests
            return JSONResponse(
                status_code=405,
                content={
                    "success": False,
                    "error": "Method Not Allowed",
                    "message": f"Method {request.method} not allowed",
                    "status_code": 405
                }
            )
    except Exception as e:
        logger.error("Error in method not allowed handler: %s", e, exc_info=True)
        # Fallback to simple JSON response
        return JSONResponse(
            status_code=405,
            content={"detail": "Method not allowed"}
        )


async def generic_http_exception_handler(
    request: Request,
    exc: Union[HTTPException, StarletteHTTPException],
) -> Union[HTMLResponse, JSONResponse]:
    """
    Generic handler for other HTTP exceptions.

    Args:
        request: FastAPI request object
        exc: HTTPException or StarletteHTTPException

    Returns:
        HTMLResponse or JSONResponse based on request type
    """
    try:
        status_code = exc.status_code

        if is_browser_request(request):
            # Determine which template to use based on status code
            try:
                templates = get_templates_from_app(request)
                if status_code == 404:
                    template_name = "errors/404.html"
                elif status_code == 401:
                    template_name = "errors/401.html"
                elif status_code == 403:
                    template_name = "errors/403.html"
                elif status_code == 405:
                    template_name = "errors/405.html"
                elif status_code == 422:
                    template_name = "errors/422.html"
                elif status_code in [500, 502, 503, 504]:
                    # Use the 500 error page for server errors
                    template_name = "errors/500.html"
                else:
                    # Fallback to the 500 error page for unknown status codes
                    template_name = "errors/500.html"

                error_detail = getattr(exc, "detail", f"HTTP {status_code} error occurred")
                context = {
                    "request": request,
                    "status_code": str(status_code),
                    "error_code": str(status_code),
                    "error": error_detail,
                    "error_message": error_detail,
                    "request_url": str(request.url),
                    "request_method": request.method,
                    "allowed_methods": request.headers.get("Allow", "GET, POST"),
                    "error_details": str(getattr(exc, "detail", "")),
                    "request_id": getattr(request.state, "request_id", None),
                }
                context["config"] = _get_template_debug_context()
                return templates.TemplateResponse(template_name, context, status_code=status_code)
            except Exception as template_error:
                logger.error("Failed to render %s template: %s", status_code, template_error)
                # Return simple HTML fallback
                return HTMLResponse(
                    content=(
                        f"<html><body><h1>{status_code} Error</h1>"
                        f"<p>{getattr(exc, 'detail', 'An error occurred')}</p>"
                        "</body></html>"
                    ),
                    status_code=status_code
                )
        else:
            # Return JSON for API requests
            return JSONResponse(
                status_code=status_code,
                content={
                    "success": False,
                    "error": f"HTTP {status_code}",
                    "message": getattr(exc, "detail", f"HTTP {status_code} error"),
                    "status_code": status_code
                }
            )
    except Exception as e:
        logger.error("Error in generic HTTP exception handler: %s", e, exc_info=True)
        # Fallback to simple JSON response
        return JSONResponse(
            status_code=getattr(exc, 'status_code', 500),
            content={"detail": getattr(exc, 'detail', 'An error occurred')}
        )
