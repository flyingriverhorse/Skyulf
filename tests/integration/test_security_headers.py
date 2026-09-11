"""OC-184: configured production security headers reach real HTTP responses."""

import json

import pytest
from fastapi import FastAPI, HTTPException, Response, WebSocket
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.testclient import TestClient

from backend.config import environments as profiles
from backend.config.base import Settings
from backend.config.environments import ProductionSettings
from backend.main import _add_exception_handlers, _add_middleware


@pytest.fixture
def settings_factory(monkeypatch):
    """Read isolated profile defaults without changing process logging handlers."""
    for name in (*Settings.model_fields, "SECURITY_HEADERS", "S3_BUCKET_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Settings, "setup_logging", lambda self: None)

    def make(profile=ProductionSettings, **kwargs):
        """Construct the real settings profile with a test host and explicit secret."""
        return profile(
            _env_file=None,
            SECRET_KEY="oc184-test-key-with-at-least-32-characters",
            ALLOWED_HOSTS=["testserver"],
            CORS_ORIGINS=["https://canvas.test"],
            **kwargs,
        )

    return make


def _app(settings, tmp_path):
    """Exercise real middleware order with normal, error, file and streaming responses."""
    app = FastAPI()
    asset = tmp_path / "style.css"
    asset.write_text("body { color: black; }", encoding="utf-8")

    @app.get("/ok")
    async def ok():
        """Return an ordinary API response."""
        return {"ok": True}

    @app.get("/cookies")
    async def cookies():
        """Return multiple cookies and a conflicting endpoint security header."""
        response = Response("cookies", headers={"X-Frame-Options": "SAMEORIGIN"})
        response.set_cookie("first", "one")
        response.set_cookie("second", "two")
        return response

    @app.get("/invalid")
    async def invalid():
        """Return a handled client error."""
        raise HTTPException(status_code=422, detail="invalid column")

    @app.get("/failed")
    async def failed():
        """Trigger the custom error middleware's generated response."""
        raise RuntimeError("oc184 expected error")

    @app.get("/redirect")
    async def redirect():
        """Preserve redirects when response headers are added."""
        return RedirectResponse("/ok")

    @app.get("/asset")
    async def file():
        """Return a static asset with a real content type."""
        return FileResponse(asset)

    @app.get("/stream")
    async def stream():
        """Ensure the middleware does not consume or replace streamed bodies."""
        return StreamingResponse(iter(["first\n", "second\n"]), media_type="text/plain")

    @app.websocket("/ws")
    async def echo(websocket: WebSocket):
        """Keep non-HTTP scopes usable for live job updates."""
        await websocket.accept()
        await websocket.send_text(await websocket.receive_text())
        await websocket.close()

    _add_exception_handlers(app)
    _add_middleware(app, settings)
    return app


@pytest.mark.parametrize(
    ("path", "status"),
    [
        ("/ok", 200),
        ("/invalid", 422),
        ("/failed", 500),
        ("/redirect", 307),
        ("/asset", 200),
        ("/stream", 200),
        ("/missing", 404),
    ],
)
def test_production_headers_cover_success_and_error_responses(
    settings_factory, tmp_path, path, status
):
    """Security headers and CORS must survive every normal HTTP response path."""
    settings = settings_factory()
    response = TestClient(_app(settings, tmp_path), base_url="https://testserver").get(
        path, headers={"Origin": "https://canvas.test"}, follow_redirects=False
    )
    assert response.status_code == status
    for name, value in settings.SECURITY_HEADERS.items():
        assert response.headers[name] == value
    assert response.headers["Access-Control-Allow-Origin"] == "https://canvas.test"
    assert response.headers["X-Request-ID"]
    if path == "/stream":
        assert response.text == "first\nsecond\n"
    elif path == "/asset":
        assert response.headers["content-type"].startswith("text/css")
    elif path == "/redirect":
        assert response.headers["location"] == "/ok"


@pytest.mark.parametrize("profile", [profiles.DevelopmentSettings, profiles.TestingSettings])
def test_other_profiles_do_not_enable_production_headers(settings_factory, tmp_path, profile):
    """Local development and tests must retain their existing browser behavior."""
    response = TestClient(_app(settings_factory(profile), tmp_path)).get("/ok")
    assert response.status_code == 200
    assert "content-security-policy" not in response.headers
    assert "strict-transport-security" not in response.headers


@pytest.mark.parametrize("source", ["constructor", "environment"])
def test_custom_security_headers_are_used(settings_factory, tmp_path, monkeypatch, source):
    """Operators must be able to replace the production policy through Settings."""
    policy = {"Content-Security-Policy": "default-src 'none'", "Referrer-Policy": "no-referrer"}
    if source == "environment":
        monkeypatch.setenv("SECURITY_HEADERS", json.dumps(policy))
        settings = settings_factory()
    else:
        settings = settings_factory(SECURITY_HEADERS=policy)
    response = TestClient(_app(settings, tmp_path)).get("/ok")
    for name, value in policy.items():
        assert response.headers[name] == value
    assert "x-frame-options" not in response.headers


def test_policy_overrides_endpoint_header_and_preserves_cookies(settings_factory, tmp_path):
    """Applying the server policy must preserve unrelated repeated response headers."""
    response = TestClient(_app(settings_factory(), tmp_path)).get("/cookies")
    assert response.headers["X-Frame-Options"] == "DENY"
    assert len(response.headers.get_list("Set-Cookie")) == 2
    assert response.cookies["first"] == "one"
    assert response.cookies["second"] == "two"


def test_explicit_empty_policy_disables_injection(settings_factory, tmp_path):
    """An operator-managed proxy can take responsibility for these headers."""
    response = TestClient(_app(settings_factory(SECURITY_HEADERS={}), tmp_path)).get("/ok")
    assert response.status_code == 200
    assert "content-security-policy" not in response.headers


def test_cors_preflight_remains_outermost(settings_factory, tmp_path):
    """A CORS preflight must still be answered before the inner request middleware."""
    response = TestClient(_app(settings_factory(), tmp_path)).options(
        "/ok", headers={"Origin": "https://canvas.test", "Access-Control-Request-Method": "GET"}
    )
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "https://canvas.test"
    assert "GET" in response.headers["Access-Control-Allow-Methods"]


def test_websocket_messages_are_unchanged(settings_factory, tmp_path):
    """HTTP header injection must not affect the WebSocket job-event transport."""
    with TestClient(_app(settings_factory(), tmp_path)).websocket_connect("/ws") as websocket:
        websocket.send_text("job update")
        assert websocket.receive_text() == "job update"
