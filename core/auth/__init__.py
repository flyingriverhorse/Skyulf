"""
FastAPI Authentication Module

Modern JWT-based authentication system migrated from Flask.
Provides secure authentication with async support and dependency injection.
"""

from .auth_core import (
    User,
    UserInDB, 
    UserCreate,
    UserLogin,
    Token,
    TokenData,
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
    authenticate_user,
    get_user_permissions,
    create_tokens,
    create_dummy_users,
    AD_GROUPS,
    get_jwt_config,
    get_security_config
)

from .dependencies import (
    get_current_user,
    get_current_active_user,
    get_current_admin_user,
    get_optional_user,
    require_permissions,
    require_scope,
    log_auth_event,
    oauth2_scheme,
    http_bearer
)

from .page_security import (
    PageSecurityChecker,
    check_page_authentication,
    require_page_authentication,
    create_page_context
)

from .routes import auth_router, auth_template_router

__all__ = [
    # Core models and functions
    "User",
    "UserInDB",
    "UserCreate", 
    "UserLogin",
    "Token",
    "TokenData",
    "get_password_hash",
    "verify_password",
    "create_access_token",
    "create_refresh_token", 
    "verify_token",
    "authenticate_user",
    "get_user_permissions",
    "create_tokens",
    "create_dummy_users",
    "AD_GROUPS",
    "get_jwt_config",
    "get_security_config",
    
    # Dependencies
    "get_current_user",
    "get_current_active_user",
    "get_current_admin_user",
    "get_optional_user",
    "require_permissions",
    "require_scope", 
    "log_auth_event",
    "oauth2_scheme",
    "http_bearer",
    
    # Page-level security
    "PageSecurityChecker",
    "check_page_authentication",
    "require_page_authentication", 
    "create_page_context",
    
    # Router
    "auth_router",
    "auth_template_router"
]