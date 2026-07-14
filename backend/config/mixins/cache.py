"""Cache configuration settings."""


class CacheMixin:
    """Filesystem/Redis cache settings."""

    CACHE_TYPE: str = "filesystem"
    CACHE_DEFAULT_TIMEOUT: int = 3600
    REDIS_URL: str | None = None
    CACHE_TTL: int = 300

    # Socket connect timeout (seconds) used for the lightweight Redis PING in
    # /health. Kept short by default since health checks should fail fast, but
    # a 1s default can false-positive as "unhealthy" against a remote/slower
    # Redis instance under load — raise via env var if you see flaky health
    # check failures against a real (non-local) Redis deployment.
    REDIS_HEALTHCHECK_TIMEOUT_SECONDS: float = 1.0
