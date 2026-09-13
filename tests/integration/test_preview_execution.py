"""Preview executes real data without starving requests or accepting invalid graphs."""

import asyncio
import threading
from time import perf_counter
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.database.models import Base, DataSource, ErrorEvent
from backend.exceptions.core import SkyulfException
from backend.exceptions.handlers import skyulf_exception_handler
from backend.ml_pipeline._internal._routers import preview as preview_mod
from tests.integration.test_node_inspection import _node


@pytest.fixture(params=["pandas", "polars"])
async def preview_execution(monkeypatch, tmp_path, request):
    """Keep HTTP, SQL, file loading, transforms, and error persistence real."""
    database_path = tmp_path / "preview.sqlite"
    async_engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
    sync_engine = create_engine(f"sqlite:///{database_path}")
    sessions = async_sessionmaker(async_engine, expire_on_commit=False)
    async with async_engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    frame = pd.DataFrame({f"value_{index}": np.arange(1100) + index for index in range(16)})
    csv_path = tmp_path / "input.csv"
    frame.to_csv(csv_path, index=False)
    async with sessions() as session:
        session.add(
            DataSource(id=1, name="Preview input", type="file", config={"file_path": str(csv_path)})
        )
        await session.commit()
    directories = []
    session_threads = []

    class PreviewSession(Session):
        """Measure the thread that creates and closes each real ORM session."""

        def __init__(self):
            """Record resource ownership before the catalog can query it."""
            super().__init__(bind=sync_engine)
            session_threads.append(("create", threading.get_ident()))

        def close(self):
            """Close the real connection before recording the lifecycle boundary."""
            super().close()
            session_threads.append(("close", threading.get_ident()))

    @event.listens_for(sync_engine, "before_cursor_execute")
    def record_query(connection, cursor, statement, parameters, context, executemany):
        """Record the catalog's actual SQL work to catch cross-thread session use."""
        session_threads.append(("query", threading.get_ident()))

    def make_temp_dir(**kwargs):
        """Expose real artifact directories for success and failure cleanup checks."""
        directory = tmp_path / str(uuid4())
        directory.mkdir()
        directories.append(directory)
        return str(directory)

    async def request_session():
        """Give dataset resolution its own asynchronous database session."""
        async with sessions() as session:
            yield session

    async def heartbeat():
        """Represent another HTTP request served on the preview's event loop."""
        return {"alive": True}

    monkeypatch.setattr(preview_mod, "tempfile", SimpleNamespace(mkdtemp=make_temp_dir))
    monkeypatch.setattr(preview_mod.db_engine, "sync_session_factory", PreviewSession)
    monkeypatch.setattr(preview_mod.db_engine, "async_session_factory", sessions)
    monkeypatch.setattr(get_settings(), "AWS_BUCKET_NAME", None)
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", request.param)
    app = FastAPI()
    app.include_router(preview_mod.router, prefix="/pipeline")
    app.add_api_route("/heartbeat", heartbeat)
    app.add_exception_handler(SkyulfException, skyulf_exception_handler)
    app.dependency_overrides[preview_mod.get_async_session] = request_session
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield SimpleNamespace(
            client=client,
            sessions=sessions,
            directories=directories,
            session_threads=session_threads,
            source=_node("source", "data_loader", params={"path": "1"}),
        )
    sync_engine.dispose()
    await async_engine.dispose()


async def test_preview_data_sink_executes_upstream_like_toolbar(preview_execution):
    """A direct API sink must expose the same sampled data as the toolbar graph."""
    harness = preview_execution
    nodes = [harness.source]
    normal = await harness.client.post(
        "/pipeline/preview", json={"pipeline_id": "normal", "nodes": nodes}
    )
    direct = await harness.client.post(
        "/pipeline/preview?inspect_all=true",
        json={"pipeline_id": "sink", "nodes": [*nodes, _node("sink", "data_preview", ["source"])]},
    )
    assert normal.status_code == direct.status_code == 200
    response = direct.json()
    assert response["status"] == "success"
    assert set(response["node_results"]) == {"source"}
    assert response["preview_data"] == normal.json()["preview_data"]
    assert response["preview_totals"] == {"_total": 1000}
    assert len(response["preview_data"]) == 50
    assert response["preview_data"][0]["value_0"] == 0
    receipts = {entry["node_id"]: entry for entry in response["node_inspections"]}
    assert receipts["source"]["output"]["tables"][0]["row_count"] == 1000
    assert receipts["sink"]["output"]["status"] == "unavailable"
    assert all(not directory.exists() for directory in harness.directories)


async def test_preview_resolves_dataset_before_worker_execution(preview_execution):
    """Resolved database paths must reach the sampled worker graph after adaptation."""
    harness = preview_execution
    response = await harness.client.post(
        "/pipeline/preview",
        json={
            "pipeline_id": "dataset",
            "nodes": [_node("source", "data_loader", params={"dataset_id": "1"})],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["preview_totals"] == {"_total": 1000}
    assert payload["preview_data"][0]["value_0"] == 0
    assert payload["preview_data"][-1]["value_0"] == 49
    assert all(not directory.exists() for directory in harness.directories)


@pytest.mark.parametrize("with_training", [False, True])
async def test_preview_sink_preserves_branch_data_and_provenance(preview_execution, with_training):
    """Shared sinks must not discard data branches or change their inspection identities."""
    harness = preview_execution
    nodes = [
        harness.source,
        _node("left", inputs=["source"]),
        _node("right", "MinMaxScaler", ["source"]),
    ]
    if with_training:
        nodes.extend(
            _node(f"model_{branch}", "training", [branch], {"model_type": "logistic_regression"})
            for branch in ("left", "right")
        )
    responses = []
    for graph in [nodes, [*nodes, _node("sink", "data_preview", ["left", "right"])]]:
        response = await harness.client.post(
            "/pipeline/preview?inspect_all=true", json={"pipeline_id": "branches", "nodes": graph}
        )
        assert response.status_code == 200
        responses.append(response.json())
    normal, direct = responses
    assert normal["status"] == direct["status"] == "success"
    for field in (
        "preview_data",
        "preview_totals",
        "branch_previews",
        "branch_preview_totals",
        "branch_node_ids",
    ):
        assert direct[field] == normal[field]
    assert len(direct["branch_previews"]) == 2
    assert set(direct["node_results"]) == {"source", "left", "right"}
    normal_paths = {
        (entry["node_id"], entry["branch_id"]): entry["path_id"]
        for entry in normal["node_inspections"]
    }
    direct_paths = {
        (entry["node_id"], entry["branch_id"]): entry["path_id"]
        for entry in direct["node_inspections"]
        if entry["node_id"] != "sink"
    }
    assert direct_paths == normal_paths
    assert all(not directory.exists() for directory in harness.directories)


@pytest.mark.parametrize("sink_cycle", [False, True])
async def test_cyclic_preview_is_client_error_without_error_event(preview_execution, sink_cycle):
    """Malformed client graphs must never create internal critical incidents."""
    harness = preview_execution
    response = await harness.client.post(
        "/pipeline/preview",
        json={
            "pipeline_id": "cycle",
            "nodes": [
                _node("left", inputs=["right"]),
                _node("right", "data_preview" if sink_cycle else "StandardScaler", ["left"]),
            ],
        },
    )
    async with harness.sessions() as session:
        error_events = list((await session.scalars(select(ErrorEvent))).all())
    assert response.status_code == 400, (
        response.text,
        [(row.status_code, row.error_type) for row in error_events],
    )
    assert "cycle" in response.json()["detail"].lower()
    assert error_events == []
    assert harness.session_threads == []
    assert all(not directory.exists() for directory in harness.directories)


async def test_cancelled_preview_releases_resources_after_worker_finishes(
    preview_execution, monkeypatch
):
    """A cancelled request must not remove artifacts still needed by its worker."""
    harness = preview_execution
    entered = asyncio.Event()
    finished = threading.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    saved = preview_mod.LocalArtifactStore.save
    removed = preview_mod.shutil.rmtree
    completed_artifacts = []
    scaled_values = []

    def pause_after_source(store, key, data):
        """Keep real artifact writes while pausing execution at a known boundary."""
        saved(store, key, data)
        completed_artifacts.append(key)
        if key == "scale":
            scaled_values.append(float(store.load(key)["value_0"][0]))
        if key == "source":
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(timeout=10), "Cancelled preview worker was never released"

    def remove_artifacts(path, **kwargs):
        """Signal only after the actual worker-owned directory has been removed."""
        removed(path, **kwargs)
        finished.set()

    monkeypatch.setattr(preview_mod.LocalArtifactStore, "save", pause_after_source)
    monkeypatch.setattr(preview_mod, "shutil", SimpleNamespace(rmtree=remove_artifacts))
    task = asyncio.create_task(
        harness.client.post(
            "/pipeline/preview",
            json={
                "pipeline_id": "cancel",
                "nodes": [harness.source, _node("scale", inputs=["source"])],
            },
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert harness.directories and all(path.exists() for path in harness.directories)
        assert "close" not in [action for action, _ in harness.session_threads]
    finally:
        release.set()
        async with asyncio.timeout(10):
            while not finished.is_set():
                await asyncio.sleep(0.005)
    assert completed_artifacts[0] == "source" and completed_artifacts[-1] == "scale"
    assert scaled_values == pytest.approx([-1.7303196219])
    assert [action for action, _ in harness.session_threads][-1] == "close"
    assert len({thread for _, thread in harness.session_threads}) == 1
    assert all(not path.exists() for path in harness.directories)


async def test_real_preview_keeps_http_heartbeat_responsive(preview_execution):
    """Sixty actual scalers must allow other HTTP requests throughout their execution."""
    harness = preview_execution
    warmup = await harness.client.post(
        "/pipeline/preview",
        json={"pipeline_id": "warmup", "nodes": [harness.source, _node("warm", inputs=["source"])]},
    )
    assert warmup.status_code == 200 and warmup.json()["status"] == "success"
    harness.session_threads.clear()
    nodes = [harness.source]
    for index in range(60):
        nodes.append(_node(f"scale_{index}", inputs=[nodes[-1]["node_id"]]))
    ticks = [perf_counter()]
    finished = asyncio.Event()

    async def post_preview():
        """Signal completion even if the measured endpoint unexpectedly raises."""
        try:
            return await harness.client.post(
                "/pipeline/preview", json={"pipeline_id": "concurrency", "nodes": nodes}
            )
        finally:
            finished.set()

    async def poll_heartbeat():
        """Measure scheduling gaps with real HTTP traffic on the same ASGI loop."""
        while not finished.is_set():
            response = await harness.client.get("/heartbeat")
            assert response.status_code == 200 and response.json() == {"alive": True}
            ticks.append(perf_counter())
            await asyncio.sleep(0.005)
        ticks.append(perf_counter())

    response, _ = await asyncio.gather(post_preview(), poll_heartbeat())
    duration = ticks[-1] - ticks[0]
    largest_gap = max(later - earlier for earlier, later in zip(ticks, ticks[1:], strict=False))
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert len(payload["node_results"]) == 61
    assert payload["preview_totals"] == {"_total": 1000}
    assert payload["preview_data"][0]["value_0"] == pytest.approx(-1.7303196219)
    assert largest_gap < duration * 0.6, {
        "duration": duration,
        "largest_gap": largest_gap,
        "ticks": len(ticks),
    }
    assert len(ticks) >= 5
    assert [action for action, _ in harness.session_threads][0] == "create"
    assert [action for action, _ in harness.session_threads][-1] == "close"
    assert any(action == "query" for action, _ in harness.session_threads)
    assert len({thread for _, thread in harness.session_threads}) == 1
    assert harness.session_threads[0][1] != threading.get_ident()
    assert all(not directory.exists() for directory in harness.directories)
