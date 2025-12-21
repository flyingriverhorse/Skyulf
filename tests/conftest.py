"""Pytest configuration helpers for Skyulf test suite."""

import os
import sys
import pytest
from pathlib import Path

# Force Celery to use in-memory broker/backend for tests to avoid Redis connection issues
# and the lingering _connection_worker_thread.
os.environ["CELERY_BROKER_URL"] = "memory://"
os.environ["CELERY_RESULT_BACKEND"] = "rpc://"
os.environ["USE_CELERY"] = "False"

# During tests we want the FastAPI TestClient which uses the host
# "testserver" to be accepted by TrustedHostMiddleware. Add it to
# the runtime settings ALLOWED_HOSTS here so tests that call the app
# via TestClient don't get rejected with "Invalid host header".
try:
    # Import lazily so tests that don't import config aren't impacted
    from backend.config import get_settings

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


@pytest.fixture(scope="session", autouse=True)
def cleanup_celery_connections():
    """
    Explicitly close Celery connections at the end of the test session.
    This prevents the '_connection_worker_thread' (from redis-py) from hanging the process.
    """
    yield
    try:
        # 1. Close the current default app (which might be implicitly created)
        from celery import current_app
        if current_app:
            print(f"\n[pytest] Closing Celery current_app: {current_app}")
            current_app.close()

        # 2. Close the explicit app if it was loaded
        if "backend.celery_app" in sys.modules:
            from backend.celery_app import celery_app
            print("\n[pytest] Closing backend.celery_app...")
            celery_app.close()
    except Exception as e:
        print(f"\n[pytest] Error closing Celery app: {e}")

