"""
Page-level security dependencies for FastAPI template routes.

This module provides configurable authentication and authorization checking
for HTML template pages based on configuration settings.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, TYPE_CHECKING
from fnmatch import fnmatch
from fastapi import Request, HTTPException, status
from fastapi.responses import RedirectResponse

from config import get_settings
from .auth_core import verify_token, get_user_permissions, USERS_DB

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .auth_core import UserInDB

logger = logging.getLogger(__name__)


def _load_user_for_page(username: str) -> Optional["UserInDB"]:
    """Fetch a user from the database for template guards."""
    try:
        from .service import get_user_by_username
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Page security user hydration import failed: %s", exc)
        return None

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(get_user_by_username(username))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Page security user hydration failed for %s: %s", username, exc)
        return None
    finally:
        asyncio.set_event_loop(None)
        loop.close()


class PageSecurityChecker:
    """Handles page-level security checking based on configuration."""

    def __init__(self):
        self.settings = get_settings()

    def is_public_page(self, path: str) -> bool:
        """Check if a page is configured as public (no authentication required)."""
        public_pages = self.settings.public_pages_list

        # Check exact matches first
        if path in public_pages:
            return True

        # Check wildcard patterns
        for public_path in public_pages:
            if public_path.endswith("*"):
                # Remove the * and check if path starts with the pattern
                pattern = public_path[:-1]
                if path.startswith(pattern):
                    return True
            elif fnmatch(path, public_path):
                return True

        return False

    def requires_admin(self, path: str) -> bool:
        """Check if a page requires admin privileges."""
        admin_pages = self.settings.admin_required_pages_list

        # Check exact matches first
        if path in admin_pages:
            return True

        # Check wildcard patterns
        for admin_path in admin_pages:
            if admin_path.endswith("*"):
                # Remove the * and check if path starts with the pattern
                pattern = admin_path[:-1]
                if path.startswith(pattern):
                    return True
            elif fnmatch(path, admin_path):
                return True

        return False

    def check_page_access(self, request: Request) -> Dict[str, Any]:
        """
        Check if current request should have access to the page with enhanced security.

        Returns:
            Dict containing user info and access decision:
            - current_user: User info if authenticated, None otherwise
            - user_permissions: Permission flags
            - should_redirect: True if should redirect to login
            - redirect_url: URL to redirect to if authentication required
            - clear_cookie: True if invalid token cookie should be cleared
        """
        path = request.url.path

        # First check if this page requires admin privileges
        requires_admin = self.requires_admin(path)

        # Check if this page is public (unless it requires admin)
        is_public = self.is_public_page(path) and not requires_admin

        # If REQUIRE_LOGIN_BY_DEFAULT is False, flip the logic
        if not self.settings.REQUIRE_LOGIN_BY_DEFAULT:
            # In this mode, only pages in PUBLIC_PAGES require auth (inverted logic)
            requires_auth = path in self.settings.public_pages_list or requires_admin
        else:
            # Default mode: require auth unless explicitly public
            requires_auth = not is_public or requires_admin

        # Initialize result
        result = {
            "current_user": None,
            "user_permissions": {
                "can_user": False,
                "is_admin": False
            },
            "should_redirect": False,
            "redirect_url": self.settings.LOGIN_REDIRECT_URL,
            "requires_admin": requires_admin,
            "is_public": is_public,
            "clear_cookie": False,
            "token_status": "none"
        }

        # Get token from cookies
        access_token = request.cookies.get("access_token")

        if access_token:
            try:
                # Remove "Bearer " prefix if present
                token = access_token.replace("Bearer ", "") if access_token.startswith("Bearer ") else access_token

                # Verify token with enhanced checking
                payload = verify_token(token, self.settings.SECRET_KEY, "access")

                if payload:
                    # Token is valid
                    result["token_status"] = "valid"
                    current_user = {
                        "username": payload.get("sub"),
                        "display_name": payload.get("display_name"),
                        "email": payload.get("email"),
                        "is_admin": payload.get("is_admin", False)
                    }

                    # Get user permissions
                    users_db = USERS_DB
                    user_obj = users_db.get(current_user["username"])

                    if not user_obj:
                        user_obj = _load_user_for_page(current_user["username"])
                        if user_obj:
                            users_db[current_user["username"]] = user_obj

                    if user_obj and not user_obj.is_active:
                        logger.warning(f"Inactive user attempted access: {current_user['username']}")
                        result["token_status"] = "inactive_user"
                        result["clear_cookie"] = True
                    elif user_obj:
                        permissions_list = get_user_permissions(user_obj)

                        user_permissions = {
                            "can_user": len(permissions_list) > 0,
                            "is_admin": current_user["is_admin"] or "admin:system" in permissions_list
                        }

                        result["current_user"] = current_user
                        result["user_permissions"] = user_permissions

                    if not user_obj:
                        # User not found in database
                        logger.warning(f"Token valid but user not found: {current_user['username']}")
                        result["token_status"] = "user_not_found"
                        result["clear_cookie"] = True
                else:
                    # Token verification failed (expired, invalid, etc.)
                    logger.info(f"Token verification failed for {path} - clearing cookie")
                    result["token_status"] = "invalid"
                    result["clear_cookie"] = True

            except Exception as e:
                logger.warning(f"Token processing error for {path}: {e}")
                result["token_status"] = "error"
                result["clear_cookie"] = True

        # Check if user should be redirected
        if requires_auth and not result["current_user"]:
            result["should_redirect"] = True
        elif requires_admin and not result["user_permissions"]["is_admin"]:
            result["should_redirect"] = True
            result["redirect_url"] = "/"  # Redirect to home for insufficient privileges

        return result


# Global instance
_page_security_checker = PageSecurityChecker()


def check_page_authentication(request: Request) -> Dict[str, Any]:
    """
    FastAPI dependency function to check page-level authentication.

    This can be used as a dependency in route handlers to get user info
    and determine if the user should have access.

    Returns:
        Dict with user info and access control information
    """
    return _page_security_checker.check_page_access(request)


def require_page_authentication(request: Request) -> Dict[str, Any]:
    """
    FastAPI dependency that enforces page authentication.

    This will raise an HTTPException or return a redirect response
    if the user doesn't have appropriate access.

    Returns:
        Dict with user info if access is granted
    """
    result = _page_security_checker.check_page_access(request)

    if result["should_redirect"]:
        # For API requests, raise an exception
        if request.headers.get("accept", "").startswith("application/"):
            if result["requires_admin"] and result["current_user"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin privileges required"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )

        # For HTML requests, redirect
        raise HTTPException(
            status_code=status.HTTP_302_FOUND,
            detail="Redirect required",
            headers={"Location": result["redirect_url"]}
        )

    return result


def create_page_response(
    request: Request,
    template_name: str,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create a page response with authentication handling, redirects, and cookie clearing.

    This centralizes all the authentication logic including:
    - Authentication checking
    - Redirect handling with cookie clearing
    - Template response with cookie clearing for invalid tokens

    Args:
        request: FastAPI request object
        template_name: Template file name to render
        extra_context: Additional context to add to template

    Returns:
        Either a RedirectResponse or TemplateResponse with proper cookie handling
    """
    # Get authentication context
    auth_info = check_page_authentication(request)

    # Handle redirect if needed
    if auth_info["should_redirect"]:
        response = RedirectResponse(
            url=auth_info["redirect_url"],
            status_code=status.HTTP_302_FOUND
        )

        # Clear invalid cookie if needed
        if auth_info.get("clear_cookie"):
            response.delete_cookie(
                "access_token",
                path="/",
                domain=None,
                secure=False,
                httponly=True,
                samesite="lax"
            )
            logger.info(
                "Cleared invalid cookie during redirect - token status: %s",
                auth_info.get('token_status', 'unknown'),
            )

        return response

    # Build template context
    context = {
        "request": request,
        "current_user": auth_info["current_user"],
        "user_permissions": auth_info["user_permissions"],
        "requires_admin": auth_info["requires_admin"],
        "is_public": auth_info["is_public"]
    }

    # Add extra context if provided
    if extra_context:
        context.update(extra_context)

    # Handle legacy context overrides for backward compatibility
    if "user" in context and context["user"] is None:
        context["user"] = auth_info["current_user"]
    if "is_admin" in context and context["is_admin"] is False:
        context["is_admin"] = auth_info["user_permissions"]["is_admin"]

    # Get templates from app state
    templates = request.app.state.templates
    response = templates.TemplateResponse(template_name, context)

    # Clear invalid cookie if needed (for authenticated access with invalid token)
    if auth_info.get("clear_cookie"):
        response.delete_cookie(
            "access_token",
            path="/",
            domain=None,
            secure=False,
            httponly=True,
            samesite="lax"
        )
        logger.info(
            "Cleared invalid cookie in template response - token status: %s",
            auth_info.get('token_status', 'unknown'),
        )

    return response


def create_page_context(request: Request) -> Dict[str, Any]:
    """
    Create template context with authentication info and handle cookie clearing.

    This is a helper function that can be used in template routes
    to get consistent user context. For new code, prefer using create_page_response().
    """
    auth_info = check_page_authentication(request)

    context = {
        "request": request,
        "current_user": auth_info["current_user"],
        "user_permissions": auth_info["user_permissions"],
        "requires_admin": auth_info["requires_admin"],
        "is_public": auth_info["is_public"]
    }

    # Add redirect handling if needed
    if auth_info["should_redirect"]:
        context["should_redirect"] = True
        context["redirect_url"] = auth_info["redirect_url"]

    # Add cookie clearing info if needed
    if auth_info.get("clear_cookie"):
        context["clear_cookie"] = True
        context["token_status"] = auth_info.get("token_status", "unknown")

    return context
