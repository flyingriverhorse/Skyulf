"""Default throttling must reach real routes without changing response delivery."""

import httpx
import pytest
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.util import get_remote_address

import backend.main as main
from backend.database.engine import get_async_session


@pytest.fixture
def limited_app(monkeypatch):
    """Use the production app factory with isolated counters and observable handlers."""
    limiter = Limiter(
        key_func=get_remote_address, default_limits=["3/minute"], headers_enabled=True
    )
    monkeypatch.setattr(main, "limiter", limiter)
    isolated_settings = main.settings.model_copy(
        update={
            "CORS_ORIGINS": ["https://canvas.test"],
            "SECURITY_HEADERS": {"X-Frame-Options": "DENY"},
        }
    )
    monkeypatch.setattr(main, "settings", isolated_settings)
    router = APIRouter()
    calls = []

    @router.post("/__limits/write")
    async def write():
        """Expose whether a rejected request reaches application side effects."""
        calls.append("write")
        return {"saved": True}

    @router.get("/__limits/explicit")
    @limiter.limit("5/minute")
    async def explicit(request: Request):
        """A route override must replace the smaller default, not consume it too."""
        return JSONResponse({"explicit": True})

    @router.get("/__limits/dynamic")
    @limiter.limit(lambda: "5/minute")
    async def dynamic(request: Request):
        """Runtime limit providers retain the same override behavior as static rules."""
        return JSONResponse({"dynamic": True})

    @router.get("/__limits/exempt")
    @limiter.exempt
    async def exempt(request: Request):
        """Explicit exemptions must not inherit the default limit."""
        return {"exempt": True}

    @router.get("/__limits/stream")
    async def stream():
        """Multiple response chunks and duplicate headers must survive throttling."""
        response = StreamingResponse(iter(["first\n", "second\n"]))
        response.set_cookie("first", "one")
        response.set_cookie("second", "two")
        return response

    original_include = main._include_routers

    def include_routes(app):
        """Keep probe routes ahead of the real SPA catch-all, as API routes are."""
        app.include_router(router)
        original_include(app)

    monkeypatch.setattr(main, "_include_routers", include_routes)
    app = main.create_app()

    async def unused_session():
        """Unseeded schema preview does not query storage; never reach the user's DB."""
        yield None

    app.dependency_overrides[get_async_session] = unused_session
    yield app, limiter, calls
    limiter.reset()


async def test_default_rejects_before_mutation_and_keeps_cors_headers(limited_app):
    """An undecorated mutation must stop at its budget with a browser-readable 429."""
    app, _, calls = limited_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        responses = [
            await client.post("/__limits/write", headers={"Origin": "https://canvas.test"})
            for _ in range(4)
        ]
    assert [response.status_code for response in responses] == [200, 200, 200, 429]
    assert calls == ["write"] * 3
    assert responses[-1].headers["access-control-allow-origin"] == "https://canvas.test"
    assert responses[-1].headers["x-frame-options"] == "DENY"
    assert responses[-1].headers["retry-after"]


async def test_real_schema_preview_uses_default_and_separates_clients(limited_app):
    """The actual app's undecorated API must throttle each address independently."""
    app, _, _ = limited_app
    payload = {
        "pipeline_id": "rate-preview",
        "nodes": [{"node_id": "loader", "step_type": "data_loader", "params": {}, "inputs": []}],
        "metadata": {},
    }
    results = []
    for address, count in [("192.0.2.1", 4), ("192.0.2.2", 1)]:
        transport = httpx.ASGITransport(app=app, client=(address, 1234))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            results.extend(
                [
                    await client.post("/api/pipeline/schema-preview", json=payload)
                    for _ in range(count)
                ]
            )
    assert [response.status_code for response in results] == [200, 200, 200, 429, 200]
    assert results[-1].json()["predicted_schemas"] == {"loader": None}


@pytest.mark.parametrize("path", ["explicit", "dynamic"])
async def test_route_limit_overrides_default_despite_spa_catch_all(limited_app, path):
    """Route lookup must use the first match and charge decorated requests exactly once."""
    app, _, _ = limited_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        responses = [await client.get(f"/__limits/{path}") for _ in range(6)]
    assert [response.status_code for response in responses] == [200] * 5 + [429]


async def test_exemption_disable_and_preflight_preserve_request_budgets(limited_app):
    """Intentional exemptions, disabled limiting and browser preflight retain their contracts."""
    app, limiter, calls = limited_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        for _ in range(5):
            exempt = await client.get("/__limits/exempt")
            assert exempt.status_code == 200
            preflight = await client.options(
                "/__limits/write",
                headers={"Origin": "https://canvas.test", "Access-Control-Request-Method": "POST"},
            )
            assert preflight.status_code == 200
        limited = [await client.post("/__limits/write") for _ in range(4)]
        limiter.enabled = False
        unlimited = [await client.post("/__limits/write") for _ in range(4)]
    assert [response.status_code for response in limited] == [200, 200, 200, 429]
    assert [response.status_code for response in unlimited] == [200] * 4
    assert len(calls) == 7


async def test_streaming_headers_and_static_mount_survive_default_limiting(limited_app):
    """The middleware must forward one response start and keep every body chunk and cookie."""
    app, _, _ = limited_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/__limits/stream")
        static_responses = [await client.get("/static/nonexistent-rate-test.css") for _ in range(5)]
    assert response.status_code == 200
    assert response.text == "first\nsecond\n"
    assert len(response.headers.get_list("set-cookie")) == 2
    assert response.headers["x-ratelimit-limit"] == "3"
    assert [response.status_code for response in static_responses] == [404] * 5


def test_jobs_websocket_bypasses_http_default_limits(limited_app):
    """HTTP throttling must not interfere with the live job event socket."""
    app, _, _ = limited_app
    client = TestClient(app)
    for _ in range(5):
        with client.websocket_connect("/ws/jobs") as socket:
            socket.send_text("connection-check")
    assert not main.connection_manager._clients
