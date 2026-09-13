"""Artifact lookup must not turn user identifiers or storage errors into forged logs."""

import logging
from urllib.parse import quote

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.database.engine import Base, get_async_session
from backend.database.models import TrainingJob
from backend.ml_pipeline.artifacts.factory import ArtifactFactory
from backend.ml_pipeline.model_registry.api import router


@pytest_asyncio.fixture
async def registry_client():
    """Use real registry queries against an isolated database and the mounted HTTP route."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    app = FastAPI()
    app.include_router(router)

    async def session_dependency():
        """Supply the same isolated database to each production route query."""
        async with sessions() as session:
            yield session

    app.dependency_overrides[get_async_session] = session_dependency
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as client:
            yield client, sessions
    finally:
        await engine.dispose()


def _registry_logs(caplog):
    """Inspect formatter inputs too, so a deferred raw traceback cannot bypass sanitization."""
    records = [r for r in caplog.records if r.name.startswith("backend.ml_pipeline.model_registry")]
    assert records
    for record in records:
        message = logging.Formatter("%(levelname)s %(message)s").format(record)
        assert "\n" not in message and "\r" not in message and "\x1b" not in message
        assert "oc289-password" not in message
        assert "oc289-secret" not in message
        assert record.exc_info is None
    return records


@pytest.mark.asyncio
async def test_missing_job_identifier_cannot_forge_a_log_record(registry_client, caplog):
    """A percent-encoded line break in a real missing-job request must remain visible but escaped."""
    client, _ = registry_client
    job_id = "missing\nforged\r\x1b[31m password=oc289-password"
    with caplog.at_level(logging.WARNING, logger="backend.ml_pipeline.model_registry"):
        response = await client.get(f"/registry/artifacts/{quote(job_id, safe='')}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Job artifacts not found"
    records = _registry_logs(caplog)
    assert len(records) == 1
    assert "missing\\x0aforged\\x0d\\x1b" in records[0].getMessage()
    assert "[REDACTED]" in records[0].getMessage()


def _storage_failure(kind):
    """Represent external storage failures while keeping service and HTTP error paths real."""
    if kind == "value":
        raise ValueError("invalid\nlocation password=oc289-password")
    if kind == "notes":
        error = RuntimeError("storage diagnostic")
        error.add_note("forged\nsecret=oc289-secret")
        raise error
    if kind == "group":
        raise ExceptionGroup("storage diagnostic", [RuntimeError("forged\nsecret=oc289-secret")])
    try:
        raise RuntimeError("forged\nsecret=oc289-secret")
    except RuntimeError as cause:
        if kind == "context":
            raise ConnectionError("storage diagnostic")  # noqa: B904 - exercise implicit chaining
        raise ConnectionError("storage diagnostic") from cause


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["value", "cause", "context", "notes", "group"])
async def test_storage_failures_keep_every_registry_log_safe(
    registry_client, monkeypatch, caplog, kind
):
    """Service and route logs must sanitize errors without leaking secrets through exception chains."""
    client, sessions = registry_client
    job_id = "saved\nforged"
    async with sessions() as session:
        session.add(
            TrainingJob(
                id=job_id,
                pipeline_id="test",
                node_id="model",
                dataset_source_id="data",
                model_type="ridge",
                run_mode="fixed",
                graph={},
                artifact_uri="s3://bucket/artifacts",
            )
        )
        await session.commit()

    def unavailable_store(uri):
        """Fail only the external artifact boundary after the real database lookup."""
        _storage_failure(kind)

    monkeypatch.setattr(ArtifactFactory, "get_artifact_store", unavailable_store)
    with caplog.at_level(logging.WARNING, logger="backend.ml_pipeline.model_registry"):
        response = await client.get(f"/registry/artifacts/{quote(job_id, safe='')}")

    assert response.status_code == (404 if kind == "value" else 500)
    records = _registry_logs(caplog)
    assert len(records) == 2
    assert "[REDACTED]" in records[-1].getMessage()
    if kind != "value":
        assert "_storage_failure" in records[-1].getMessage()
