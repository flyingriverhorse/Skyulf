"""Regression tests for backend infra fixes:

- rate limiter now has a conservative app-wide default (safety net for
  undecorated endpoints) instead of ``default_limits=[]``.
- health endpoints report an actual environment name instead of deriving
  it from the ``DEBUG`` flag.
- exception handlers use the traceback attached to the handled exception
  object (``exc.__traceback__``) instead of the ambient
  ``traceback.format_exc()``, which can describe an unrelated exception.
"""

import traceback

import pytest
from fastapi.testclient import TestClient

from backend.config.base import Settings
from backend.exceptions.core import SkyulfException
from backend.exceptions.handlers import generic_http_exception_handler, skyulf_exception_handler
from backend.main import app
from backend.middleware.rate_limiter import limiter


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient reusing the real app instance."""
    with TestClient(app, base_url="http://localhost") as c:
        yield c


# ── Rate limiting default ────────────────────────────────────────────────


def test_limiter_has_nonempty_default_limits():
    """The shared limiter must define a baseline default limit, not an empty list.

    An empty ``default_limits`` means any endpoint without an explicit
    ``@limiter.limit(...)`` decorator has zero rate limiting.
    """
    assert limiter._default_limits, "limiter default_limits must not be empty"


def test_limiter_default_limit_matches_settings():
    """The limiter's default limit should come from Settings.RATE_LIMIT_DEFAULT."""
    settings = Settings()
    assert settings.RATE_LIMIT_DEFAULT

    requests_count = settings.RATE_LIMIT_DEFAULT.split("/")[0]
    limit_strs = [
        str(limit.limit) for limit_group in limiter._default_limits for limit in limit_group
    ]
    assert any(requests_count in s and "minute" in s for s in limit_strs)


def test_undecorated_endpoint_still_works_under_default_limit(client):
    """A handful of requests to an undecorated endpoint should stay well under
    the generous default limit and keep succeeding (no false-positive throttling).
    """
    for _ in range(5):
        response = client.get("/ping")
        assert response.status_code == 200


# ── Health environment field ─────────────────────────────────────────────


def test_environment_name_falls_back_to_debug_when_unset():
    """With ENVIRONMENT unset, environment_name derives from DEBUG (legacy behavior)."""
    settings = Settings(ENVIRONMENT=None, DEBUG=True)
    assert settings.environment_name == "development"

    settings = Settings(ENVIRONMENT=None, DEBUG=False)
    assert settings.environment_name == "production"


def test_environment_name_uses_explicit_setting_over_debug():
    """A staging deployment with DEBUG=False must report 'staging', not 'production'."""
    settings = Settings(ENVIRONMENT="staging", DEBUG=False)
    assert settings.environment_name == "staging"


def test_health_endpoint_reports_environment_name(client, monkeypatch):
    """The /health endpoint should surface settings.environment_name directly."""
    from backend import dependencies

    original_settings = dependencies.get_config()

    class _StagingSettings:
        def __getattr__(self, item):
            return getattr(original_settings, item)

        @property
        def environment_name(self):
            return "staging"

    app.dependency_overrides[dependencies.get_config] = lambda: _StagingSettings()
    try:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["environment"] == "staging"
    finally:
        app.dependency_overrides.pop(dependencies.get_config, None)


# ── Exception handler traceback source ───────────────────────────────────


async def _run_skyulf_handler_with_poisoned_ambient_exc_info():
    """Poison sys.exc_info() with an unrelated exception, then invoke the handler
    with a *different*, fresh exception and verify the recorded traceback matches
    the fresh exception, not the poisoned ambient one.
    """
    # Step 1: poison ambient exception state with an unrelated exception.
    try:
        raise ValueError("unrelated ambient exception - should NOT appear in result")
    except ValueError:
        pass  # caught and handled; sys.exc_info() may still reflect this on some pythons

    # Step 2: raise+catch a fresh, distinct exception to hand to the handler.
    try:
        raise SkyulfException(message="fresh exception - SHOULD appear in traceback")
    except SkyulfException as fresh_exc:
        captured = {}

        async def fake_record_error(**kwargs):
            captured.update(kwargs)

        import backend.exceptions.handlers as handlers_module

        original_record_error = handlers_module._record_error
        handlers_module._record_error = fake_record_error
        try:
            request = _make_fake_request()
            await skyulf_exception_handler(request, fresh_exc)
        finally:
            handlers_module._record_error = original_record_error

        return captured, fresh_exc


class _FakeURL:
    path = "/fake/route"


class _FakeState:
    request_id = "test-request-id"


class _FakeRequest:
    url = _FakeURL()
    state = _FakeState()


def _make_fake_request():
    return _FakeRequest()


@pytest.mark.asyncio
async def test_skyulf_handler_uses_fresh_exception_traceback_not_ambient():
    """Regression test for the ambient sys.exc_info() traceback bug."""
    captured, fresh_exc = await _run_skyulf_handler_with_poisoned_ambient_exc_info()

    assert "message" in captured
    assert captured["message"] == "fresh exception - SHOULD appear in traceback"
    assert "traceback" in captured
    assert "unrelated ambient exception" not in captured["traceback"]
    # The recorded traceback must correspond to the fresh exception's own traceback.
    expected_tb = "".join(
        traceback.format_exception(type(fresh_exc), fresh_exc, fresh_exc.__traceback__)
    )
    assert captured["traceback"] == expected_tb
    assert "SkyulfException" in captured["traceback"]


@pytest.mark.asyncio
async def test_generic_http_handler_uses_fresh_exception_traceback_not_ambient():
    """Same regression check for generic_http_exception_handler."""
    captured = {}

    async def fake_record_error(**kwargs):
        captured.update(kwargs)

    import backend.exceptions.handlers as handlers_module

    original_record_error = handlers_module._record_error
    handlers_module._record_error = fake_record_error
    try:
        # Poison ambient exception state first.
        try:
            raise ValueError("unrelated ambient exception - should NOT appear in result")
        except ValueError:
            pass

        class _FakeHTTPExc(Exception):
            status_code = 500
            detail = "fresh http exception - SHOULD appear in traceback"

        try:
            raise _FakeHTTPExc("fresh http exception - SHOULD appear in traceback")
        except _FakeHTTPExc as fresh_exc:
            request = _make_fake_request()
            await generic_http_exception_handler(request, fresh_exc)

            expected_tb = "".join(
                traceback.format_exception(type(fresh_exc), fresh_exc, fresh_exc.__traceback__)
            )
            assert captured["traceback"] == expected_tb
            assert "unrelated ambient exception" not in captured["traceback"]
            assert "_FakeHTTPExc" in captured["traceback"]
    finally:
        handlers_module._record_error = original_record_error
