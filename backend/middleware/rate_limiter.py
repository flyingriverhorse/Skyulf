"""Rate limiting infrastructure using slowapi (key by client IP)."""

from slowapi import Limiter
from slowapi.util import get_remote_address

# Single shared Limiter instance; imported by routers and registered on the app.
limiter = Limiter(key_func=get_remote_address, default_limits=[])
