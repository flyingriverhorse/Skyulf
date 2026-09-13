"""Regression coverage for upload and pagination resource limits."""

import io
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import polars as pl
import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException, UploadFile
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.config import get_settings
from backend.data_ingestion.connectors.file import LocalFileConnector
from backend.data_ingestion.dependencies import get_data_service
from backend.data_ingestion.router import sources_router
from backend.data_ingestion.service import DataIngestionService
from backend.database.engine import get_async_session
from backend.database.models import Base, DataSource, TrainingJob
from backend.dependencies import get_db
from backend.ml_pipeline._execution.jobs import JobManager
from backend.ml_pipeline._internal._routers.jobs import router as jobs_router
from backend.monitoring.router import _load_current_dataframe
from backend.monitoring.router import router as monitoring_router


@pytest_asyncio.fixture
async def intake_client(tmp_path):
    """Exercise actual routes and SQL against isolated source and job records."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        source_file = tmp_path / "source.json"
        pl.DataFrame({"value": list(range(25))}).write_json(source_file)
        session.add(DataSource(name="source", type="file", config={"file_path": str(source_file)}))
        for index in range(4):
            session.add(
                TrainingJob(
                    id=f"job-{index}",
                    pipeline_id="pipeline",
                    node_id=f"node-{index}",
                    dataset_source_id="source",
                    status="completed",
                    model_type="classifier",
                    graph={},
                    run_mode="fixed" if index % 2 == 0 else "tuned",
                    version=1,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index),
                    metrics={"summary": f"Job {index}", "test_accuracy": index / 4},
                )
            )
        await session.commit()
        statements = []

        def record_sql(conn, cursor, statement, parameters, context, executemany):
            """Record SQL so rejection tests catch validation after database reads."""
            statements.append(statement)

        event.listen(engine.sync_engine, "before_cursor_execute", record_sql)
        service = DataIngestionService(session, upload_dir=str(tmp_path))
        app = FastAPI()
        app.include_router(jobs_router, prefix="/pipeline")
        app.include_router(sources_router)
        app.include_router(monitoring_router)
        app.dependency_overrides[get_async_session] = lambda: session
        app.dependency_overrides[get_db] = lambda: session
        app.dependency_overrides[get_data_service] = lambda: service
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            yield client, statements, service
    await engine.dispose()


@pytest.mark.parametrize("limit", [-1, 0, 50_001])
@pytest.mark.parametrize("extension", ["csv", "parquet", "json"])
async def test_connector_rejects_invalid_sample_before_reading(
    tmp_path, monkeypatch, limit, extension
):
    """Invalid limits must never materialize a source or use negative head semantics."""
    source_file = tmp_path / f"source.{extension}"
    frame = pl.DataFrame({"value": list(range(25))})
    getattr(frame, f"write_{extension}")(source_file)
    connector = LocalFileConnector(str(source_file))
    scan = MagicMock(wraps=connector._scan)
    load = AsyncMock(wraps=connector._load_data)
    monkeypatch.setattr(connector, "_scan", scan)
    monkeypatch.setattr(connector, "_load_data", load)
    with pytest.raises(ValueError, match="limit"):
        await connector.fetch_data(limit=limit)
    scan.assert_not_called()
    load.assert_not_called()
    assert connector._df is None


@pytest.mark.parametrize("extension", ["csv", "parquet", "json"])
async def test_connector_keeps_full_reads_and_positive_samples(tmp_path, extension):
    """Training keeps full-file reads while valid previews return the requested head."""
    source_file = tmp_path / f"source.{extension}"
    frame = pl.DataFrame({"value": list(range(25))})
    getattr(frame, f"write_{extension}")(source_file)
    connector = LocalFileConnector(str(source_file))
    assert (await connector.fetch_data(limit=3)).to_dicts() == [
        {"value": 0},
        {"value": 1},
        {"value": 2},
    ]
    assert (await connector.fetch_data()).equals(frame)


@pytest.mark.parametrize("params", [{"limit": -1}, {"limit": 0}, {"limit": 501}, {"skip": -1}])
@pytest.mark.parametrize("job_type", [None, "training", "tuning", "ingestion"])
async def test_job_lists_reject_invalid_bounds_before_sql(intake_client, params, job_type):
    """Bad pages must be rejected before selecting jobs and serializing their metrics."""
    client, statements, _ = intake_client
    endpoint = "/data/api/sources" if job_type == "ingestion" else "/pipeline/jobs"
    query = dict(params)
    if job_type in {"training", "tuning"}:
        query["job_type"] = job_type
    response = await client.get(endpoint, params=query)
    assert response.status_code == 422
    assert statements == []


@pytest.mark.parametrize("limit", [-1, 0, 501])
async def test_node_summaries_reject_invalid_limits_before_sql(intake_client, limit):
    """Node cards must not bypass the job-list page cap while loading metrics."""
    client, statements, _ = intake_client
    response = await client.get("/pipeline/jobs/node-summaries", params={"limit": limit})
    assert response.status_code == 422
    assert statements == []


@pytest.mark.parametrize("limit", [-1, 0, 50_001])
async def test_source_sample_rejects_invalid_limits_before_sql(intake_client, limit):
    """Sample endpoints must reject bad bounds before source lookup and file reads."""
    client, statements, _ = intake_client
    response = await client.get("/data/api/sources/1/sample", params={"limit": limit})
    assert response.status_code == 422
    assert statements == []


async def test_source_sample_preserves_defaults_and_export_cap(intake_client):
    """The five-row preview default and existing 50,000-row export request stay valid."""
    client, _, _ = intake_client
    preview = await client.get("/data/api/sources/1/sample")
    exported = await client.get("/data/api/sources/1/export", params={"limit": 50_000})
    assert preview.status_code == 200
    assert preview.json()["data"] == [{"value": index} for index in range(5)]
    assert exported.status_code == 200
    assert pl.read_csv(io.BytesIO(exported.content))["value"].to_list() == list(range(25))


@pytest.mark.parametrize("limit", [-1, 0, 50_001])
async def test_direct_source_sample_rejects_limits_before_sql(intake_client, limit):
    """Non-HTTP sampling callers must receive validation before source lookup too."""
    _, statements, service = intake_client
    with pytest.raises(ValueError, match="limit"):
        await service.get_sample(1, limit=limit)
    assert statements == []


@pytest.mark.parametrize("params", [{"limit": -1}, {"limit": 0}, {"limit": 501}, {"skip": -1}])
async def test_direct_job_lists_reject_bounds_before_sql(intake_client, params):
    """The manager must validate direct calls before merging or serializing job rows."""
    _, statements, service = intake_client
    with pytest.raises(ValueError, match="limit|skip"):
        await JobManager.list_jobs(service.session, **params)
    assert statements == []


async def test_job_list_keeps_order_and_offset(intake_client):
    """A valid combined page must remain newest-first after merging training and tuning."""
    client, _, _ = intake_client
    response = await client.get("/pipeline/jobs", params={"limit": 2, "skip": 1})
    assert response.status_code == 200
    assert [job["job_id"] for job in response.json()] == ["job-2", "job-1"]


async def test_job_list_uses_current_configured_cap(intake_client, monkeypatch):
    """Changing the configured cap must constrain requests without reimporting routers."""
    client, statements, _ = intake_client
    monkeypatch.setattr(get_settings(), "MAX_PAGE_SIZE", 2)
    response = await client.get("/pipeline/jobs", params={"limit": 3})
    assert response.status_code == 422
    assert statements == []


async def test_monitoring_upload_preserves_413_over_http(intake_client, monkeypatch):
    """A multipart upload one byte over the total policy must remain HTTP 413."""
    client, _, _ = intake_client
    monkeypatch.setattr(get_settings(), "MAX_UPLOAD_SIZE", 16)
    with (
        patch("backend.monitoring.router.ArtifactFactory"),
        patch("backend.monitoring.router._find_reference_key", return_value="reference"),
        patch(
            "backend.monitoring.router._find_deployment_context",
            AsyncMock(return_value=(None, None)),
        ),
    ):
        response = await client.post(
            "/monitoring/drift/calculate",
            data={"job_id": "job-0"},
            files={"file": ("current.csv", b"x\n12345678901234\n", "text/csv")},
        )
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


async def test_monitoring_upload_bounds_reads_under_large_total_policy(monkeypatch):
    """A 10 GiB total policy must not turn a tiny upload into a 10 GiB read request."""
    monkeypatch.setattr(get_settings(), "MAX_UPLOAD_SIZE", 10 * 1024**3)
    upload = UploadFile(file=io.BytesIO(b"x\n1\n2\n"), filename="current.csv")
    sizes = []
    original_read = upload.read

    async def tracked_read(size=-1):
        """Instrument the real small source without allocating the requested size."""
        sizes.append(size)
        return await original_read(size)

    monkeypatch.setattr(upload, "read", tracked_read)
    frame = await _load_current_dataframe(upload)
    assert frame.to_dicts() == [{"x": 1}, {"x": 2}]
    assert sizes and all(0 < size <= 1024**2 for size in sizes)


@pytest.mark.parametrize("extension", ["csv", "parquet"])
async def test_monitoring_upload_accepts_exact_total_limit(monkeypatch, extension):
    """A valid upload exactly at the configured byte limit must still parse."""
    frame = pl.DataFrame({"value": [1, 2]})
    content = io.BytesIO()
    getattr(frame, f"write_{extension}")(content)
    monkeypatch.setattr(get_settings(), "MAX_UPLOAD_SIZE", len(content.getvalue()))
    content.seek(0)
    upload = UploadFile(file=content, filename=f"current.{extension}")
    result = await _load_current_dataframe(upload)
    assert result.equals(frame)


async def test_monitoring_upload_enforces_total_across_short_reads(monkeypatch):
    """Short reads must accumulate toward the cap and stop after the first excess byte."""
    monkeypatch.setattr(get_settings(), "MAX_UPLOAD_SIZE", 16)
    upload = UploadFile(file=io.BytesIO(b"x\n12345678901234\n"), filename="current.csv")
    original_read = upload.read

    async def short_read(size=-1):
        """Simulate a source returning less than the requested chunk without padding it."""
        return await original_read(min(size, 5))

    monkeypatch.setattr(upload, "read", short_read)
    with pytest.raises(HTTPException) as error:
        await _load_current_dataframe(upload)
    assert error.value.status_code == 413
    assert upload.file.tell() == 17
