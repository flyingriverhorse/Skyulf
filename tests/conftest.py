"""Pytest configuration helpers for Skyulf test suite."""

import sys
from pathlib import Path

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


def pytest_sessionfinish(session, exitstatus):
    """Debug hook: print active threads and asyncio tasks at session end.

    Add this temporarily to help identify lingering background work
    that prevents the process from exiting. Remove once root cause
    is found.
    """
    try:
        import threading
        import asyncio

        threads = threading.enumerate()
        print("\n[pytest debug] active threads:")
        for t in threads:
            print(f" - {t.name} (daemon={t.daemon})")

        try:
            loop = asyncio.get_event_loop()
            tasks = asyncio.all_tasks(loop)
            print("[pytest debug] asyncio tasks:")
            for task in tasks:
                print(f" - {task!r}")
        except Exception as _:
            print("[pytest debug] could not enumerate asyncio tasks")
    except Exception as e:
        print(f"[pytest debug] sessionfinish hook error: {e}")
