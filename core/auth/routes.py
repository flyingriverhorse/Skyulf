"""
FastAPI Authentication API Routes

Provides JWT-based authentication endpoints for login, logout, token refresh, etc.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, Form
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse

from .auth_core import (
    UserLogin, 
    Token, 
    User, 
    UserInDB,
    authenticate_user,
    create_tokens,
    verify_token,
    create_access_token,
    USERS_DB
)
from .dependencies import (
    get_current_user, 
    get_current_active_user,
    get_optional_user,
    log_auth_event
)
from .service import get_user_by_username

# Import config from the fastapi_app level
try:
    from config import get_settings
except ImportError:
    # Fallback for when running as standalone module
    import sys
    from pathlib import Path
    # Add fastapi_app directory to path
    fastapi_app_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(fastapi_app_dir))
    from config import get_settings

logger = logging.getLogger(__name__)

# Create auth API router
auth_router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
    responses={
        401: {"description": "Authentication failed"},
        403: {"description": "Access forbidden"}
    }
)

# Create auth template router (no prefix for direct access)
auth_template_router = APIRouter(
    tags=["auth-pages"]
)


@auth_router.post("/token", response_model=Token)
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Token:
    """
    OAuth2 compatible token login endpoint.
    
    This endpoint accepts form data with username and password,
    and returns access and refresh tokens.
    
    Args:
        request: FastAPI request object
        form_data: OAuth2 password form data
        
    Returns:
        Token response with access and refresh tokens
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Authenticate user
        user = await authenticate_user(
            username=form_data.username,
            password=form_data.password,
            users_db=USERS_DB
        )
        
        if not user:
            await log_auth_event(
                request=request,
                event_type="LOGIN_FAILURE",
                success=False,
                username=form_data.username,
                details="Invalid credentials"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create tokens
        settings = get_settings()
        tokens = create_tokens(user, settings.SECRET_KEY)
        
        # Log successful login
        await log_auth_event(
            request=request,
            event_type="LOGIN_SUCCESS",
            success=True,
            username=user.username,
            details=f"Access granted with {len(user.ad_groups)} groups"
        )
        
        logger.info(f"User '{user.username}' logged in successfully")
        return tokens
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        await log_auth_event(
            request=request,
            event_type="LOGIN_ERROR",
            success=False,
            username=getattr(form_data, 'username', 'unknown'),
            details=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during login"
        )


@auth_router.post("/login", response_model=Token)
async def login_json(
    request: Request,
    user_login: UserLogin
) -> Token:
    """
    JSON login endpoint (alternative to OAuth2 form).
    
    Args:
        request: FastAPI request object
        user_login: User login credentials
        
    Returns:
        Token response with access and refresh tokens
    """
    try:
        # Authenticate user
        user = await authenticate_user(
            username=user_login.username,
            password=user_login.password,
            users_db=USERS_DB
        )
        
        if not user:
            await log_auth_event(
                request=request,
                event_type="LOGIN_FAILURE",
                success=False,
                username=user_login.username,
                details="Invalid credentials (JSON)"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        # Create tokens
        settings = get_settings()
        tokens = create_tokens(user, settings.SECRET_KEY)
        
        # Log successful login
        await log_auth_event(
            request=request,
            event_type="LOGIN_SUCCESS",
            success=True,
            username=user.username,
            details="JSON login successful"
        )
        
        return tokens
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"JSON login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during login"
        )


@auth_router.post("/refresh", response_model=Token)
async def refresh_access_token(
    request: Request,
    refresh_token: str
) -> Token:
    """
    Refresh access token using refresh token.
    
    Args:
        request: FastAPI request object
        refresh_token: Valid refresh token
        
    Returns:
        New token response
    """
    try:
        settings = get_settings()
        
        # Verify refresh token
        payload = verify_token(refresh_token, settings.SECRET_KEY, "refresh")
        if payload is None:
            await log_auth_event(
                request=request,
                event_type="REFRESH_FAILURE",
                success=False,
                details="Invalid refresh token"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        username = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # Get user from cache or hydrate from database
        user = USERS_DB.get(username)
        if not user:
            user = await get_user_by_username(username)
            if user:
                USERS_DB[username] = user

        if not user or not user.is_active:
            await log_auth_event(
                request=request,
                event_type="REFRESH_FAILURE", 
                success=False,
                username=username,
                details="User not found or inactive"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new tokens
        tokens = create_tokens(user, settings.SECRET_KEY)
        
        await log_auth_event(
            request=request,
            event_type="TOKEN_REFRESH",
            success=True,
            username=username,
            details="Access token refreshed"
        )
        
        return tokens
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during token refresh"
        )


@auth_router.post("/logout")
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Logout endpoint.
    
    Note: JWT tokens cannot be truly invalidated without a token blacklist.
    This endpoint logs the logout event but tokens remain valid until expiration.
    
    Args:
        request: FastAPI request object
        current_user: Currently authenticated user
        
    Returns:
        Logout confirmation message
    """
    await log_auth_event(
        request=request,
        event_type="LOGOUT",
        success=True,
        username=current_user.username,
        details="User logged out"
    )
    
    logger.info(f"User '{current_user.username}' logged out")
    
    return {"message": "Successfully logged out"}


@auth_router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)) -> User:
    """
    Get current user profile information.
    
    Args:
        current_user: Currently authenticated user
        
    Returns:
        Current user profile
    """
    return current_user


@auth_router.get("/permissions")
async def get_my_permissions(current_user: User = Depends(get_current_active_user)) -> Dict[str, Any]:
    """
    Get current user permissions.
    
    Args:
        current_user: Currently authenticated user
        
    Returns:
        User permissions and scopes
    """
    from .auth_core import get_user_permissions
    
    # Get user from database to access full user data
    user_in_db = USERS_DB.get(current_user.username)
    if not user_in_db:
        user_in_db = await get_user_by_username(current_user.username)
        if user_in_db:
            USERS_DB[current_user.username] = user_in_db

    if not user_in_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    permissions = get_user_permissions(user_in_db)
    
    return {
        "username": current_user.username,
        "permissions": permissions,
        "ad_groups": current_user.ad_groups,
        "is_admin": current_user.is_admin
    }


@auth_router.get("/validate")
async def validate_token(current_user: User = Depends(get_current_active_user)) -> Dict[str, Any]:
    """
    Validate current token and return user info.
    
    This endpoint can be used to check if a token is still valid.
    
    Args:
        current_user: Currently authenticated user
        
    Returns:
        Token validation result with user info
    """
    return {
        "valid": True,
        "user": current_user.dict(),
        "message": "Token is valid"
    }


@auth_router.get("/status")
async def auth_status(user: User = Depends(get_optional_user)) -> Dict[str, Any]:
    """
    Get authentication status (works with or without authentication).
    
    Args:
        user: Optional current user
        
    Returns:
        Authentication status information
    """
    if user:
        return {
            "authenticated": True,
            "user": user.username,
            "display_name": user.display_name,
            "groups": user.ad_groups
        }
    else:
        return {
            "authenticated": False,
            "message": "No valid authentication found"
        }


# Health check endpoint for auth service
@auth_router.get("/health")
async def auth_health() -> Dict[str, Any]:
    """
    Authentication service health check.
    
    Returns:
        Health status of authentication service
    """
    return {
        "status": "healthy",
        "service": "authentication",
        "timestamp": datetime.utcnow().isoformat(),
        "users_loaded": len(USERS_DB),
        "token_algorithm": "HS256"
    }


# =============================================================================
# HTML Template Endpoints (Login/Logout Pages)
# =============================================================================

@auth_template_router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page template"""
    templates = request.app.state.templates
    context = {
        "request": request,
        "page_title": "Login - ML Platform"
    }
    return templates.TemplateResponse("login.html", context)


@auth_template_router.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    """Process login form submission"""
    try:
        # Authenticate user
        user = await authenticate_user(username=username, password=password, users_db=USERS_DB)
        
        if not user:
            await log_auth_event(
                request=request,
                event_type="LOGIN_FORM_FAILURE",
                success=False,
                username=username,
                details="Invalid credentials via form"
            )
            # Redirect back to login with error
            templates = request.app.state.templates
            context = {
                "request": request,
                "page_title": "Login - ML Platform",
                "error": "Invalid username or password"
            }
            return templates.TemplateResponse("login.html", context)
        
        # Create full tokens (access + refresh) so token contains user claims
        settings = get_settings()
        tokens = create_tokens(user, settings.SECRET_KEY)
        access_token = tokens.access_token
        
        # Log successful login
        await log_auth_event(
            request=request,
            event_type="LOGIN_FORM_SUCCESS",
            success=True,
            username=user.username,
            details="Successful form-based login"
        )
        
        # Redirect to home with cookie
        response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
        # Set access token cookie (HTTP only)
        response.set_cookie(
            key="access_token",
            value=f"Bearer {access_token}",
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax"
        )
        # Also set refresh token cookie for completeness (optional)
        try:
            response.set_cookie(
                key="refresh_token",
                value=tokens.refresh_token,
                httponly=True,
                secure=False,
                samesite="lax"
            )
        except Exception:
            # If refresh_token is not available for any reason, ignore
            pass
        
        logger.info(f"User '{user.username}' logged in successfully via form")
        return response
        
    except Exception as e:
        logger.error(f"Login form error: {e}")
        await log_auth_event(
            request=request,
            event_type="LOGIN_FORM_ERROR",
            success=False,
            username=username,
            details=str(e)
        )
        
        # Return error page
        templates = request.app.state.templates
        context = {
            "request": request,
            "page_title": "Login - ML Platform",
            "error": "Login failed. Please try again."
        }
        return templates.TemplateResponse("login.html", context)


@auth_template_router.get("/logout")
async def logout():
    """Logout user by clearing the cookie"""
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response