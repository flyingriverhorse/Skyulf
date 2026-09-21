"""Measure publisher reuse with real redis-py and an owned RESP socket server."""

import json
import socketserver
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import redis

from backend.realtime import events


@pytest.fixture
def publisher_server(monkeypatch):
    """Record actual TCP connections without requiring a running Redis service."""
    state = SimpleNamespace(connections=0, active=0, messages=[], reject=False)
    condition = threading.Condition()

    class Handler(socketserver.StreamRequestHandler):
        """Implement the RESP commands used by redis-py's publisher."""

        def handle(self):
            """Count connections and return integer PUBLISH replies."""
            with condition:
                state.connections += 1
                state.active += 1
            self.request.settimeout(3)
            try:
                while line := self.rfile.readline():
                    command = []
                    for _ in range(int(line[1:])):
                        length = int(self.rfile.readline()[1:])
                        command.append(self.rfile.read(length))
                        self.rfile.read(2)
                    if command[0].upper() == b"PUBLISH":
                        if state.reject:
                            return
                        state.messages.append(json.loads(command[2]))
                        self.wfile.write(b":1\r\n")
                    else:
                        self.wfile.write(b"+OK\r\n")
            except OSError:
                pass
            finally:
                with condition:
                    state.active -= 1
                    condition.notify_all()

    clients = []
    pools = []
    original_factory = redis.Redis.from_url

    def create_client(*args, **kwargs):
        """Count allocations without extending client or pool lifetimes."""
        client = original_factory(*args, **kwargs)
        clients.append(weakref.ref(client))
        pools.append(weakref.ref(client.connection_pool))
        return client

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        settings = SimpleNamespace(
            USE_CELERY=True,
            CELERY_BROKER_URL=f"redis://127.0.0.1:{server.server_address[1]}/0",
        )
        monkeypatch.setattr(events, "get_settings", lambda: settings)
        monkeypatch.setattr(redis.Redis, "from_url", create_client)
        state.clients = clients
        state.pools = pools
        state.condition = condition
        try:
            yield state
        finally:
            for reference in clients:
                if client := reference():
                    client.close()
            events.close_job_event_publisher()
            server.shutdown()
            thread.join(timeout=3)


def test_many_events_reuse_one_client_pool_and_connection(publisher_server):
    """High-frequency progress must not allocate one TCP connection per event."""
    state = publisher_server
    for progress in range(40):
        events.publish_job_event(events.JobEvent(event="progress", job_id="job", progress=progress))
    print(
        f"events=40 clients={len(state.clients)} pools={len(state.pools)} "
        f"TCP connections={state.connections} "
        f"live clients={sum(ref() is not None for ref in state.clients)}"
    )
    assert [message["progress"] for message in state.messages] == list(range(40))
    assert len(state.clients) == len(state.pools) == state.connections == 1


def test_disconnect_does_not_raise_and_next_event_reconnects(publisher_server):
    """A broken transport must not stop training or prevent subsequent delivery."""
    state = publisher_server
    state.reject = True
    events.publish_job_event(events.JobEvent(event="progress", job_id="dropped"))
    state.reject = False
    events.publish_job_event(events.JobEvent(event="status", job_id="recovered"))
    assert [message["job_id"] for message in state.messages] == ["recovered"]
    assert len(state.clients) == 1
    assert state.connections == 2


def test_shutdown_closes_socket_and_allows_new_lifecycle(publisher_server):
    """Application shutdown must release the publisher and permit a fresh startup."""
    state = publisher_server
    events.publish_job_event(events.JobEvent(event="status", job_id="first"))
    events.close_job_event_publisher()
    with state.condition:
        disconnected = state.condition.wait_for(lambda: state.active == 0, timeout=2)
    events.close_job_event_publisher()
    events.publish_job_event(events.JobEvent(event="status", job_id="second"))
    assert disconnected
    assert len(state.clients) == state.connections == 2
    assert [message["job_id"] for message in state.messages] == ["first", "second"]


def test_threads_share_one_publisher(publisher_server):
    """Concurrent API worker threads must share the client and its connection pool."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(
            executor.map(
                events.publish_job_event,
                [events.JobEvent(event="progress", job_id=str(index)) for index in range(20)],
            )
        )
    assert len(publisher_server.clients) == len(publisher_server.pools) == 1
    assert {message["job_id"] for message in publisher_server.messages} == {
        str(index) for index in range(20)
    }


def test_child_reset_discards_inherited_client_and_locked_mutex(publisher_server):
    """The fork callback must permit fresh ownership even if another thread held the lock."""
    events.publish_job_event(events.JobEvent(event="status", job_id="parent"))
    inherited_client = publisher_server.clients[0]()
    inherited_lock = events._publisher_lock
    with inherited_lock:
        events._reset_publisher_after_fork()
        events.publish_job_event(events.JobEvent(event="status", job_id="child"))
    assert inherited_client is not publisher_server.clients[1]()
    assert events._publisher_lock is not inherited_lock
    assert publisher_server.connections == 2


@pytest.mark.parametrize("signal_name", ["worker_process_shutdown", "worker_shutdown"])
def test_celery_shutdown_releases_publisher(publisher_server, signal_name):
    """Prefork children and solo workers must both invoke publisher cleanup."""
    import celery.signals

    import celery_worker  # noqa: F401 - registers worker lifecycle signal handlers

    events.publish_job_event(events.JobEvent(event="status", job_id="worker"))
    getattr(celery.signals, signal_name).send(sender=None)
    with publisher_server.condition:
        disconnected = publisher_server.condition.wait_for(
            lambda: publisher_server.active == 0, timeout=2
        )
    assert disconnected
    assert events._publisher_client is None


@pytest.mark.asyncio
async def test_api_shutdown_releases_publisher(publisher_server, monkeypatch):
    """The API owns a publisher too and must close it on lifespan teardown."""
    from fastapi import FastAPI

    import backend.main as main

    async def noop():
        """Isolate unrelated startup services from the real publisher shutdown."""
        return None

    for name in ("init_db", "create_tables", "close_db"):
        monkeypatch.setattr(main, name, noop)
    monkeypatch.setattr(main, "_reset_stale_jobs", lambda: None)
    monkeypatch.setattr(main.connection_manager, "start", noop)
    monkeypatch.setattr(main.connection_manager, "stop", noop)
    async with main.lifespan(FastAPI()):
        events.publish_job_event(events.JobEvent(event="status", job_id="api"))
    with publisher_server.condition:
        disconnected = publisher_server.condition.wait_for(
            lambda: publisher_server.active == 0, timeout=2
        )
    assert disconnected
    assert events._publisher_client is None


def test_failed_initialization_retries_on_next_event(publisher_server, monkeypatch):
    """A transient construction failure must not poison the cached publisher."""
    original_factory = redis.Redis.from_url

    def fail_creation(*args, **kwargs):
        """Model a failing publisher setup before a client exists."""
        raise redis.ConnectionError("setup failed")

    monkeypatch.setattr(redis.Redis, "from_url", fail_creation)
    events.publish_job_event(events.JobEvent(event="status", job_id="dropped"))
    monkeypatch.setattr(redis.Redis, "from_url", original_factory)
    events.publish_job_event(events.JobEvent(event="status", job_id="recovered"))
    assert [message["job_id"] for message in publisher_server.messages] == ["recovered"]


def test_cleanup_failure_does_not_prevent_new_publisher(publisher_server, monkeypatch):
    """Failed cleanup must remain nonfatal and detach the old cached client."""
    events.publish_job_event(events.JobEvent(event="status", job_id="first"))
    client = publisher_server.clients[0]()

    def fail_close():
        """Close the real socket before simulating a teardown error."""
        client.connection_pool.disconnect()
        raise redis.ConnectionError("close failed")

    with monkeypatch.context() as patch:
        patch.setattr(client, "close", fail_close)
        events.close_job_event_publisher()
    events.publish_job_event(events.JobEvent(event="status", job_id="second"))
    assert len(publisher_server.clients) == 2
    assert [message["job_id"] for message in publisher_server.messages] == ["first", "second"]
