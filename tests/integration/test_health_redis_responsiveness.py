"""Exercise health-route Redis I/O against an owned local RESP server."""

import asyncio
import socketserver
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from backend.database import engine
from backend.dependencies import get_config
from backend.health.routes import router


@pytest.fixture
def redis_server():
    """Serve minimal Redis commands and release all owned sockets at teardown."""
    stop = threading.Event()
    disconnected = threading.Event()
    ping_received = threading.Event()
    state = SimpleNamespace(delay=0.0)

    class Handler(socketserver.StreamRequestHandler):
        """Reply to Redis initialization and optionally stall PING."""

        def handle(self):
            """Parse RESP arrays so a real Redis client reaches its PING."""
            self.request.settimeout(3)
            try:
                while line := self.rfile.readline():
                    command = []
                    for _ in range(int(line[1:])):
                        length = int(self.rfile.readline()[1:])
                        command.append(self.rfile.read(length))
                        self.rfile.read(2)
                    if command[0].upper() == b"PING":
                        ping_received.set()
                        stop.wait(state.delay)
                        self.wfile.write(b"+PONG\r\n")
                    else:
                        self.wfile.write(b"+OK\r\n")
            except OSError:
                pass
            finally:
                disconnected.set()

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        state.url = f"redis://127.0.0.1:{server.server_address[1]}/0"
        state.disconnected = disconnected
        state.ping_received = ping_received
        try:
            yield state
        finally:
            stop.set()
            server.shutdown()
            thread.join(timeout=3)


@pytest.mark.asyncio
@pytest.mark.parametrize("delay", [0.0, 0.8])
async def test_detailed_health_redis_is_bounded_and_responsive(monkeypatch, redis_server, delay):
    """Slow Redis must degrade health without holding up other event-loop work."""
    redis_server.delay = delay

    async def healthy_database():
        """Isolate the unrelated database probe from the Redis regression."""
        return True

    monkeypatch.setattr(engine, "health_check", healthy_database)
    settings = SimpleNamespace(
        USE_CELERY=True,
        CELERY_BROKER_URL=redis_server.url,
        REDIS_HEALTHCHECK_TIMEOUT_SECONDS=0.1,
        APP_VERSION="test",
        environment_name="test",
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_config] = lambda: settings
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        started = time.monotonic()
        request = asyncio.create_task(client.get("/health/detailed"))
        await asyncio.sleep(0.05)
        tick_elapsed = time.monotonic() - started
        response = await request
        elapsed = time.monotonic() - started

    disconnected = await asyncio.to_thread(redis_server.disconnected.wait, 2)
    print(f"Redis delay={delay}: concurrent tick={tick_elapsed:.3f}s, route={elapsed:.3f}s")
    assert redis_server.ping_received.is_set()
    assert tick_elapsed < 0.4
    assert elapsed < 0.6
    assert response.status_code == 200
    assert response.json()["status"] == ("degraded" if delay else "healthy")
    assert response.json()["dependencies_healthy"] is (not bool(delay))
    assert disconnected
