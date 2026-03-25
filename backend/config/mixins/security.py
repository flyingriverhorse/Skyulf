"""Security, authentication, session, and CORS settings."""

import secrets
from typing import List


class SecurityMixin:
    """Auth, JWT, session, CORS, and API docs configuration."""

    # Secret key
    SECRET_KEY: str = secrets.token_urlsafe(32)

    # JWT
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # 8 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"

    # Login throttling
    MAX_LOGIN_ATTEMPTS: int = 5
    ACCOUNT_LOCKOUT_DURATION_MINUTES: int = 30

    # Developer fallback auth (disable in production)
    AUTH_FALLBACK_ENABLED: bool = True
    AUTH_FALLBACK_USERNAME: str = "admin"
    AUTH_FALLBACK_PASSWORD: str = "admin123"
    AUTH_FALLBACK_DISPLAY_NAME: str = "Fallback Admin"
    AUTH_FALLBACK_IS_ADMIN: bool = True

    # Session
    PERMANENT_SESSION_LIFETIME: int = 28800  # 8 hours
    SESSION_COOKIE_SECURE: bool = False
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"

    # Page-level auth redirects
    LOGIN_REDIRECT_URL: str = "/login"
    LOGIN_SUCCESS_REDIRECT_URL: str = "/"
    ALLOW_USER_REGISTRATION: bool = True

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]

    # API Docs
    API_DOCS_ENABLED: bool | None = None
    API_DOCS_URL: str = "/docs"
    API_REDOC_URL: str = "/redoc"
    API_OPENAPI_URL: str = "/openapi.json"
    API_DOCS_DEFAULT_MODELS_EXPAND_DEPTH: int | None = -1
    API_DOCS_DEFAULT_DOC_EXPANSION: str | None = "list"
    API_DOCS_PERSIST_AUTH: bool = True
    API_DOCS_DISPLAY_REQUEST_DURATION: bool = True
    API_DOCS_ENABLE_FILTER: bool = True
    API_DOCS_ENABLE_TRY_IT_OUT: bool = True
    API_DOCS_SERVERS: List[str] = []
    API_CONTACT_NAME: str | None = "Skyulf Support"
    API_CONTACT_EMAIL: str | None = None
    API_CONTACT_URL: str | None = None
    API_TOS_URL: str | None = None
    API_LICENSE_NAME: str | None = "Apache 2.0"
    API_LICENSE_URL: str | None = "https://www.apache.org/licenses/LICENSE-2.0.html"
    API_LOGO_URL: str | None = None
