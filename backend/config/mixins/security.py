"""Security, authentication, session, and CORS settings."""

import secrets
from typing import Annotated

from pydantic_settings import NoDecode


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

    # Baseline rate limit applied to every endpoint that doesn't declare its own
    # stricter @limiter.limit(...). Provides a safety net so a route the developers
    # forgot to decorate isn't left completely unprotected. Endpoint-specific limits
    # (e.g. "10/minute" on ingestion routes) still take precedence when declared.
    RATE_LIMIT_DEFAULT: str = "200/minute"

    # Maximum number of rows accepted in a single POST /deployment/predict
    # request body. Prevents an unbounded JSON payload from exhausting worker
    # memory in one synchronous request (this endpoint is meant for ad-hoc/
    # interactive inference, e.g. the frontend's Inference page — bulk dataset
    # scoring should go through pipeline execution instead, not this endpoint).
    # Configurable via MAX_PREDICT_REQUEST_ROWS env var.
    MAX_PREDICT_REQUEST_ROWS: int = 10_000

    # Developer fallback auth — off by default; set AUTH_FALLBACK_ENABLED=true in .env to enable.
    # AUTH_FALLBACK_USERNAME and AUTH_FALLBACK_PASSWORD must be explicitly set when enabled.
    AUTH_FALLBACK_ENABLED: bool = False
    AUTH_FALLBACK_USERNAME: str = "admin"
    AUTH_FALLBACK_PASSWORD: str = ""
    AUTH_FALLBACK_DISPLAY_NAME: str = "Fallback Admin"
    AUTH_FALLBACK_IS_ADMIN: bool = True

    # Session
    # NOTE: These are reserved for future SessionMiddleware integration.
    # Starlette's SessionMiddleware is not currently wired — do not rely on
    # these values for security decisions.
    PERMANENT_SESSION_LIFETIME: int = 28800  # 8 hours

    # Page-level auth redirects
    LOGIN_REDIRECT_URL: str = "/login"
    LOGIN_SUCCESS_REDIRECT_URL: str = "/"
    ALLOW_USER_REGISTRATION: bool = True

    # CORS
    # NoDecode: pydantic-settings must not JSON-decode these before the field_validator
    # runs in base.py. Without it, comma-separated env values (e.g. localhost,127.0.0.1)
    # raise a JSONDecodeError before our split() validator ever fires.
    CORS_ORIGINS: Annotated[list[str], NoDecode] = [
        "http://localhost:3000",
        "http://localhost:8080",
    ]
    ALLOWED_HOSTS: Annotated[list[str], NoDecode] = ["localhost", "127.0.0.1"]

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
    API_DOCS_SERVERS: list[str] = []
    API_CONTACT_NAME: str | None = "Skyulf Support"
    API_CONTACT_EMAIL: str | None = None
    API_CONTACT_URL: str | None = None
    API_TOS_URL: str | None = None
    API_LICENSE_NAME: str | None = "Apache 2.0"
    API_LICENSE_URL: str | None = "https://www.apache.org/licenses/LICENSE-2.0.html"
    API_LOGO_URL: str | None = None
