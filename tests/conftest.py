"""Pytest configuration helpers for Skyulf test suite."""

import os
import sys
import pytest
from pathlib import Path

# Force Celery to use in-memory broker/backend for tests to avoid Redis connection issues
# and the lingering _connection_worker_thread.
# NOTE: We prefer using a real Redis service in CI, but for local tests or if CI service fails,
# this fallback prevents hangs. However, if a real Redis is available (like in CI with services),
# we might want to use it. For now, we force memory to be safe and avoid the hang.
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
def cleanup_resources():
    """
    Explicitly close connections (Celery, DB) at the end of the test session.
    This prevents hanging threads from keeping the process alive.
    """
    yield

    import threading
    import asyncio

    # --- Celery Cleanup ---
    try:
        from celery import current_app
        if current_app:
            print(f"\n[pytest] Closing Celery current_app: {current_app}", file=sys.stderr)
            current_app.close()

        try:
            from backend.celery_app import celery_app
            print("[pytest] Closing backend.celery_app...", file=sys.stderr)
            celery_app.close()
        except ImportError:
            pass
    except Exception as e:
        print(f"[pytest] Error closing Celery app: {e}", file=sys.stderr)

    # --- Database Cleanup ---
    try:
        from backend.database.engine import async_engine
        if async_engine:
            print("[pytest] Disposing async_engine...", file=sys.stderr)
            try:
                # Create a new loop to run the dispose coroutine
                asyncio.run(async_engine.dispose())
            except Exception as e:
                print(f"[pytest] Could not dispose async_engine: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[pytest] Error accessing async_engine: {e}", file=sys.stderr)

    # --- Thread Debugging ---
    active_threads = threading.enumerate()
    print(f"[pytest] Active threads at exit: {[t.name for t in active_threads]}", file=sys.stderr)
