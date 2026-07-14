"""Cache configuration settings."""


class CacheMixin:
    """Filesystem/Redis cache settings."""

    CACHE_TYPE: str = "filesystem"
    CACHE_DEFAULT_TIMEOUT: int = 3600
    REDIS_URL: str | None = None
    CACHE_TTL: int = 300
