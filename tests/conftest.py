"""Pytest configuration helpers for Skyulf test suite."""

import sys
from pathlib import Path

# During tests we want the FastAPI TestClient which uses the host
# "testserver" to be accepted by TrustedHostMiddleware. Add it to
# the runtime settings ALLOWED_HOSTS here so tests that call the app
# via TestClient don't get rejected with "Invalid host header".
try:
    # Import lazily so tests that don't import config aren't impacted
    from config import get_settings

    settings = get_settings()
    if isinstance(settings, object):
        allowed = getattr(settings, "ALLOWED_HOSTS", None)
        if isinstance(allowed, list) and "testserver" not in allowed:
            allowed.append("testserver")
except Exception:
    # If anything goes wrong here we still want tests to run; they may
    # patch settings themselves. Silence errors to avoid hiding real
    # failures in test collection.
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
