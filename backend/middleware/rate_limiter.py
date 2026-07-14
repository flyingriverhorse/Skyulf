"""Rate limiting infrastructure using slowapi (key by client IP)."""

from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.config import get_settings

_settings = get_settings()

# Single shared Limiter instance; imported by routers and registered on the app.
# `default_limits` provides a baseline safety net for any endpoint that isn't
# individually decorated with @limiter.limit(...) — without it, endpoints the
# developers forgot to decorate would have zero rate limiting. Endpoints that
# declare their own @limiter.limit(...) still use that stricter, route-specific
# limit instead of this default.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[_settings.RATE_LIMIT_DEFAULT],
)
