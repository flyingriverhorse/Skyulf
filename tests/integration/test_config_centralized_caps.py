"""Regression tests for config-centralization of previously hardcoded caps:

- Redis health check ping timeout now reads ``Settings.REDIS_HEALTHCHECK_TIMEOUT_SECONDS``
  instead of a hardcoded ``1``.
- ``/api/monitoring/errors/timeline`` clamps ``hours`` using
  ``Settings.MONITORING_MAX_TIMELINE_HOURS`` instead of a hardcoded ``168``.
- ``/api/monitoring/slow-nodes`` clamps ``days`` using
  ``Settings.MONITORING_MAX_SLOWNODES_DAYS`` instead of a hardcoded ``90``, and
  clamps ``limit`` using ``Settings.MAX_PAGE_SIZE`` instead of a hardcoded ``50``.
"""

import pytest
from fastapi.testclient import TestClient

from backend.config import get_settings
from backend.main import app


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient reusing the real app instance."""
    with TestClient(app, base_url="http://localhost") as c:
        yield c


def _patched_settings(monkeypatch, **overrides):
    """Return the real Settings object patched with the given attribute overrides,
    and point ``backend.config.get_settings`` at it so inline call sites pick it up.
    """
    settings = get_settings()
    for key, value in overrides.items():
        monkeypatch.setattr(settings, key, value, raising=False)
    monkeypatch.setattr("backend.config.get_settings", lambda: settings)
    return settings


# ── Redis health check timeout ───────────────────────────────────────────


def test_redis_healthcheck_uses_settings_timeout(monkeypatch):
    """The Redis ping in /health/detailed should pass the configured
    ``REDIS_HEALTHCHECK_TIMEOUT_SECONDS`` as ``socket_connect_timeout``, not a
    hardcoded value.
    """
    import redis

    from backend import dependencies

    original_settings = dependencies.get_config()
    captured: dict = {}

    class _CeleryEnabledSettings:
        def __getattr__(self, item):
            return getattr(original_settings, item)

        @property
        def USE_CELERY(self):
            return True

        @property
        def REDIS_HEALTHCHECK_TIMEOUT_SECONDS(self):
            return 4.25

    class _FakeRedisClient:
        def ping(self):
            return True

    def _fake_from_url(url, **kwargs):
        captured.update(kwargs)
        return _FakeRedisClient()

    monkeypatch.setattr(redis, "from_url", _fake_from_url)
    app.dependency_overrides[dependencies.get_config] = lambda: _CeleryEnabledSettings()
    try:
        with TestClient(app, base_url="http://localhost") as c:
            response = c.get("/health/detailed")
        assert response.status_code == 200
        assert captured.get("socket_connect_timeout") == 4.25
    finally:
        app.dependency_overrides.pop(dependencies.get_config, None)


# ── Monitoring: errors/timeline hours cap ────────────────────────────────


def test_error_timeline_hours_clamped_to_default_setting(client):
    """Default settings should still cap at 168 hours (unchanged behavior)."""
    response = client.get("/api/monitoring/errors/timeline", params={"hours": 9999})
    assert response.status_code == 200
    assert len(response.json()) == 168


def test_detailed_health_check_returns_only_aggregate_status(client):
    """/health/detailed must not disclose per-backend/integration names or details
    to anonymous callers — only a coarse ``dependencies_healthy`` boolean.
    """
    response = client.get("/health/detailed")
    assert response.status_code == 200
    body = response.json()
    assert "dependencies_healthy" in body
    assert isinstance(body["dependencies_healthy"], bool)
    # These previously-leaked per-backend/integration fields must be gone.
    assert "database_status" not in body
    assert "cache_status" not in body
    assert "external_services" not in body


def test_error_timeline_hours_clamped_to_configured_setting(client, monkeypatch):
    """Lowering MONITORING_MAX_TIMELINE_HOURS should tighten the clamp."""
    _patched_settings(monkeypatch, MONITORING_MAX_TIMELINE_HOURS=5)
    response = client.get("/api/monitoring/errors/timeline", params={"hours": 9999})
    assert response.status_code == 200
    assert len(response.json()) == 5


# ── Monitoring: slow-nodes days/limit caps ───────────────────────────────


def test_slow_nodes_days_clamped_to_default_setting(client):
    """Default settings should still cap at 90 days (unchanged behavior)."""
    response = client.get("/api/monitoring/slow-nodes", params={"days": 9999, "limit": 1})
    assert response.status_code == 200
    assert response.json()["days"] == 90


def test_slow_nodes_days_clamped_to_configured_setting(client, monkeypatch):
    """Lowering MONITORING_MAX_SLOWNODES_DAYS should tighten the clamp."""
    _patched_settings(monkeypatch, MONITORING_MAX_SLOWNODES_DAYS=3)
    response = client.get("/api/monitoring/slow-nodes", params={"days": 9999, "limit": 1})
    assert response.status_code == 200
    assert response.json()["days"] == 3


def test_slow_nodes_limit_clamped_to_configured_setting(client, monkeypatch):
    """Lowering MAX_PAGE_SIZE should tighten the `limit` clamp (previously a
    hardcoded 50 ceiling; now driven by the general MAX_PAGE_SIZE setting).
    """
    _patched_settings(monkeypatch, MAX_PAGE_SIZE=1)
    response = client.get("/api/monitoring/slow-nodes", params={"days": 90, "limit": 9999})
    assert response.status_code == 200
    assert len(response.json()["aggregates"]) <= 1
