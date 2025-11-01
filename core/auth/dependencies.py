"""
FastAPI Authentication Dependencies

Provides dependency injection for authentication and authorization.
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.engine import get_async_session
from core.database.repository import get_user_repository, UserRepository  
from .auth_core import (
    verify_token, 
    UserInDB, 
    User, 
    TokenData, 
    create_dummy_users, 
    get_user_permissions,
    AD_GROUPS
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

# OAuth2 scheme for FastAPI documentation
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="auth/token",
    scopes={
        "read:data": "Read data sources and content",
        "create:data_source": "Create new data sources", 
        "update:own_data_source": "Update own data sources",
        "delete:own_data_source": "Delete own data sources",
        "read:all": "Read all content (admin)",
        "create:all": "Create all content (admin)",
        "update:all": "Update all content (admin)",
        "delete:all": "Delete all content (admin)",
        "admin:users": "Manage users (admin)",
        "admin:system": "System administration (admin)"
    }
    ,
    auto_error=False
)


# HTTP Bearer for direct token verification
http_bearer = HTTPBearer(auto_error=False)

# Global users database (replace with real database in production)
USERS_DB: Dict[str, UserInDB] = create_dummy_users()


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer),
    token: str = Depends(oauth2_scheme),
) -> User:
    """
    Get current authenticated user from JWT token.
    
    This dependency can be used in any FastAPI endpoint to require authentication.
    
    Args:
        credentials: HTTP Bearer credentials
        token: OAuth2 token from authorization header
        
    Returns:
        User object for authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Use token from credentials or oauth2_scheme or cookie
    auth_token = None
    token_source = None
    if credentials and credentials.credentials:
        auth_token = credentials.credentials
        token_source = 'authorization_header'
    elif token:
        auth_token = token
        token_source = 'oauth2_scheme'
    else:
        # Fallback: check cookies for access_token (frontend may store token in cookie)
        try:
            cookie_token = request.cookies.get('access_token')
            if cookie_token:
                # cookie may be set as 'Bearer <token>' or just token
                if cookie_token.lower().startswith('bearer '):
                    auth_token = cookie_token.split(' ', 1)[1]
                else:
                    auth_token = cookie_token
                token_source = 'cookie'
        except Exception:
            logger.debug('No access_token cookie found')
    
    if not auth_token:
        logger.warning("No authentication token provided (header/okta cookie missing). Token source: %s", token_source)
        raise credentials_exception
    else:
        logger.debug("Authentication token found from %s", token_source)
    
    try:
        settings = get_settings()
        payload = verify_token(auth_token, settings.SECRET_KEY, "access")
        
        if payload is None:
            logger.warning("Token verification failed")
            raise credentials_exception
        
        username: str = payload.get("sub")
        if username is None:
            logger.warning("Token missing username (sub)")
            raise credentials_exception
            
        token_data = TokenData(
            username=username,
            scopes=payload.get("scopes", [])
        )
        
    except Exception as e:
        logger.error(f"Token processing error: {e}")
        raise credentials_exception
    
    # Get user from database (in production, use async database)
    user = USERS_DB.get(token_data.username)
    if user is None:
        try:
            user = await get_user_by_username(token_data.username)
        except Exception as exc:  # pragma: no cover - telemetry only
            logger.error("Failed to hydrate user %s from database: %s", token_data.username, exc)
            user = None

        if user is None:
            logger.warning(f"User not found in database: {token_data.username}")
            raise credentials_exception

        USERS_DB[token_data.username] = user
    
    # Check if user is still active
    if not user.is_active:
        logger.warning(f"Inactive user attempted access: {token_data.username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    # Convert to User model (no password)
    return User(
        id=user.id,  # Include the ID field
        username=user.username,
        email=user.email,
        display_name=user.display_name,
        ad_groups=user.ad_groups,
        is_active=user.is_active,
        is_admin=user.is_admin,
        created_date=user.created_date,
        last_login=user.last_login
    )


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current active user (additional check for active status).
    
    Args:
        current_user: Current user from get_current_user dependency
        
    Returns:
        Active user object
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current user and verify admin privileges.
    
    Args:
        current_user: Current user from get_current_user dependency
        
    Returns:
        Admin user object
        
    Raises:
        HTTPException: If user is not admin
    """
    if not current_user.is_admin and AD_GROUPS["Admin"] not in current_user.ad_groups:
        logger.warning(f"Non-admin user attempted admin access: {current_user.username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_permissions(required_permissions: List[str]):
    """
    Create a dependency that requires specific permissions.
    
    Args:
        required_permissions: List of required permission strings
        
    Returns:
        FastAPI dependency function
    """
    async def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        """Check if user has required permissions."""
        # Get user permissions
        user_in_db = USERS_DB.get(current_user.username)
        if not user_in_db:
            user_in_db = await get_user_by_username(current_user.username)
            if not user_in_db:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            USERS_DB[current_user.username] = user_in_db
        
        user_permissions = get_user_permissions(user_in_db)
        
        # Check if user has all required permissions
        missing_permissions = []
        for permission in required_permissions:
            if permission not in user_permissions:
                missing_permissions.append(permission)
        
        if missing_permissions:
            logger.warning(
                f"User {current_user.username} missing permissions: {missing_permissions}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permissions: {', '.join(missing_permissions)}"
            )
        
        return current_user
    
    return permission_checker


def require_scope(required_scope: str):
    """
    Create a dependency that requires a specific OAuth2 scope.
    
    Args:
        required_scope: Required OAuth2 scope
        
    Returns:
        FastAPI dependency function
    """
    return require_permissions([required_scope])


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer)
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.
    
    This dependency can be used for endpoints that have optional authentication.
    
    Args:
        credentials: HTTP Bearer credentials (optional)
        
    Returns:
        User object if authenticated, None otherwise
    """
    if not credentials or not credentials.credentials:
        return None
    
    try:
        settings = get_settings()
        payload = verify_token(credentials.credentials, settings.SECRET_KEY, "access")
        
        if payload is None:
            return None
        
        username: str = payload.get("sub")
        if username is None:
            return None
            
        # Get user from database
        user = USERS_DB.get(username)
        if user is None:
            user = await get_user_by_username(username)
            if user:
                USERS_DB[username] = user

        if user is None or not user.is_active:
            return None
        
        # Convert to User model (no password)
        return User(
            username=user.username,
            email=user.email,
            display_name=user.display_name,
            ad_groups=user.ad_groups,
            is_active=user.is_active,
            is_admin=user.is_admin,
            created_date=user.created_date,
            last_login=user.last_login
        )
        
    except Exception as e:
        logger.debug(f"Optional authentication failed: {e}")
        return None


async def log_auth_event(
    request: Request,
    event_type: str,
    success: bool = True,
    username: str = None,
    details: str = None,
    **kwargs
):
    """
    Log authentication events for security monitoring.
    
    Args:
        request: FastAPI request object
        event_type: Type of authentication event
        success: Whether the event succeeded
        username: Username involved (if any)
        details: Additional details
        **kwargs: Additional data to log
    """
    try:
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")[:200]
        
        # Build log message
        status = "SUCCESS" if success else "FAILED"
        username_str = username or "unknown"
        
        log_msg = (
            f"AUTH_EVENT [{status}] {event_type} | "
            f"user:{username_str} | IP:{client_ip}"
        )
        if details:
            log_msg += f" | {details}"
        
        # Log at appropriate level
        if event_type in ["LOGIN_FAILURE", "UNAUTHORIZED_ACCESS", "TOKEN_INVALID"]:
            if success:
                logger.warning(log_msg)
            else:
                logger.error(log_msg)
        else:
            if success:
                logger.info(log_msg)
            else:
                logger.warning(log_msg)
                
    except Exception as e:
        logger.error(f"Failed to log auth event: {e}")


# For backward compatibility with async database operations
async def get_user_from_db(
    username: str,
    session: AsyncSession = Depends(get_async_session)
) -> Optional[UserInDB]:
    """
    Get user from database asynchronously.
    
    This is a placeholder for when we implement proper database user storage.
    
    Args:
        username: Username to lookup
        session: Database session
        
    Returns:
        User object if found, None otherwise
    """
    # TODO: Implement actual database lookup
    # For now, use dummy users
    return USERS_DB.get(username)