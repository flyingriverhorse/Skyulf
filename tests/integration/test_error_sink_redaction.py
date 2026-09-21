"""QW-60/87: error responses and durable diagnostics must not expose credentials."""

import json
import logging
import traceback
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from sqlalchemy import create_engine, select
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

from backend.data_ingestion import tasks
from backend.data_ingestion.schemas.ingestion import (
    DataSourceCreate,
    DataSourceRead,
    IngestionStatus,
)
from backend.data_ingestion.service import DataIngestionService
from backend.data_ingestion.tasks import _handle_ingestion_failure
from backend.database.models import Base, DataSource, ErrorEvent
from backend.exceptions import handlers
from backend.exceptions.core import ForbiddenException, SkyulfException
from backend.monitoring.router import ErrorEventResponse

ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
SECRET = "qw60-synthetic-secret"
TOKEN = "qw60-synthetic-token"


def _assert_redacted(value):
    """Check text and structured payloads without relying on replacement formatting."""
    text = json.dumps(value, default=str)
    for credential in (ACCESS_KEY, SECRET, TOKEN):
        assert credential not in text


def _db_failure():
    """Keep realistic SQLAlchemy parameter formatting without contacting a source."""
    return StatementError(
        "synthetic database unavailable",
        "INSERT INTO data_sources (config) VALUES (:config)",
        {"config": {"aws_access_key_id": ACCESS_KEY, "aws_secret_access_key": SECRET}},
        OSError("synthetic disk failure"),
    )


@pytest.fixture
async def error_sessions(monkeypatch):
    """Use a fresh in-memory database so no application error history is touched."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr("backend.database.engine.async_session_factory", factory)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_kind", ["inline", "upload"])
async def test_ingestion_db_failure_redacts_http_persistence_and_logs(
    source_kind, error_sessions, tmp_path, caplog
):
    """An actual ingestion service wrapper must scrub every downstream diagnostic sink."""
    failed_session = Mock()
    failed_session.commit = AsyncMock(side_effect=_db_failure())
    service = DataIngestionService(failed_session, upload_dir=str(tmp_path))
    app = FastAPI()
    app.add_exception_handler(SkyulfException, handlers.skyulf_exception_handler)

    @app.post("/probe")
    async def fail():
        """Drive the real source registration or upload persistence failure path."""
        if source_kind == "inline":
            payload = DataSourceCreate(
                name="probe", type="s3", config={"path": "s3://probe/data.csv"}
            )
            return await service.handle_create_source(payload, user_id=1, background_tasks=None)
        path = tmp_path / "upload.csv"
        path.write_text("x\n1\n", encoding="utf-8")
        return await service._create_file_source_and_ingest(
            file_path=path,
            raw_name="upload.csv",
            file_id="probe",
            user_id=1,
            background_tasks=None,
        )

    with caplog.at_level(logging.ERROR, logger="backend"):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/probe")
    assert response.status_code == 500
    _assert_redacted(response.json())
    async with error_sessions() as session:
        event = (await session.execute(select(ErrorEvent))).scalar_one()
        _assert_redacted({"message": event.message, "traceback": event.traceback})
        assert "StatementError" in event.traceback
    records = [r for r in caplog.records if r.name == "backend.data_ingestion.service"]
    assert records
    for record in records:
        _assert_redacted(logging.Formatter().format(record))
        assert record.exc_info is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 500])
async def test_custom_error_details_keep_shape_and_hide_nested_credentials(
    status_code, error_sessions
):
    """Structured details must be copied and redacted even on non-persisted client errors."""
    details = {
        "storage_options": {"aws_session_token": TOKEN, "aws_secret_access_key": SECRET},
        "failures": [{"reason": f"password={SECRET}", "count": 2}],
        "available": False,
        "missing": None,
    }
    original = deepcopy(details)
    exc = SkyulfException(f"Database error: secret={SECRET}", details)
    exc.status_code = status_code
    response = await handlers.skyulf_exception_handler(
        Request({"type": "http", "path": "/probe", "headers": []}), exc
    )
    body = json.loads(response.body)
    _assert_redacted(body)
    assert body["details"]["failures"][0]["count"] == 2
    assert body["details"]["available"] is False
    assert body["details"]["missing"] is None
    assert details == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler",
    [
        handlers.generic_http_exception_handler,
        handlers.validation_exception_handler,
        handlers.not_found_exception_handler,
        handlers.unauthorized_exception_handler,
    ],
)
async def test_http_details_cannot_bypass_redaction(handler, error_sessions):
    """Handled HTTP errors must apply the same structured diagnostic boundary."""
    response = await handler(
        Request({"type": "http", "path": "/probe", "headers": []}),
        HTTPException(status_code=422, detail={"password": SECRET, "rows": 3}),
    )
    body = json.loads(response.body)
    _assert_redacted(body)
    assert body["message"]["rows"] == 3


@pytest.mark.asyncio
async def test_async_error_recorder_scrubs_before_truncation(error_sessions):
    """Persistence must enforce redaction even when the caller passes raw text."""
    message = "diagnostic " + "x" * 1970 + f" secret={SECRET}"
    await handlers._record_error("/probe", "ProbeError", message, f"password={SECRET}", 500)
    async with error_sessions() as session:
        row = (await session.execute(select(ErrorEvent))).scalar_one()
        assert "qw60" not in row.message
        _assert_redacted(row.traceback)
        assert row.message.startswith("diagnostic ")


def test_sync_error_recorder_scrubs_durable_values(tmp_path, monkeypatch):
    """Celery's independent persistence entry point must not bypass the async fix."""
    url = f"sqlite:///{(tmp_path / 'errors.db').as_posix()}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    monkeypatch.setattr("backend.config.get_settings", lambda: SimpleNamespace(DATABASE_URL=url))
    try:
        handlers.record_pipeline_error("probe", f"secret={SECRET}", f"password={SECRET}")
        with Session(engine) as session:
            row = session.execute(select(ErrorEvent)).scalar_one()
            _assert_redacted({"message": row.message, "traceback": row.traceback})
            assert row.job_id == "probe"
    finally:
        engine.dispose()


def test_ingestion_failure_scrubs_metadata_and_logs(tmp_path, caplog):
    """A worker failure must persist safe status while preserving unrelated profile data."""
    engine = create_engine(f"sqlite:///{(tmp_path / 'source.db').as_posix()}")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session:
            row = DataSource(
                name="probe",
                type="s3",
                config={},
                source_metadata={"row_count": 12},
                is_active=True,
                test_status="pending",
            )
            session.add(row)
            session.commit()
            source_id = row.id
            with caplog.at_level(logging.ERROR, logger="backend.data_ingestion.tasks"):
                _handle_ingestion_failure(session, source_id, _db_failure())
        with Session(engine) as session:
            saved = session.get(DataSource, source_id)
            _assert_redacted(saved.source_metadata)
            assert saved.source_metadata["row_count"] == 12
            assert saved.source_metadata["ingestion_status"]["status"] == "failed"
        assert caplog.records
        _assert_redacted(caplog.text)
    finally:
        engine.dispose()


def test_saved_source_diagnostics_are_redacted_on_read_without_mutating_profile():
    """Old unsafe status must be safe through both source and status response models."""
    now = datetime.now(UTC)
    metadata = {
        "row_count": 12,
        "profile": {"sample_data": [{"password": "ordinary dataset value"}]},
        "ingestion_status": {
            "status": "failed",
            "progress": 0,
            "updated_at": now,
            "error": f"secret={SECRET}",
            "details": {"aws_session_token": TOKEN},
        },
    }
    original = deepcopy(metadata)
    response = DataSourceRead(
        id=1,
        source_id="probe",
        name="probe",
        type="s3",
        config={},
        is_active=True,
        test_status="failed",
        created_at=now,
        updated_at=now,
        source_metadata=metadata,
    )
    _assert_redacted(response.model_dump())
    _assert_redacted(IngestionStatus(**metadata["ingestion_status"]).model_dump())
    assert response.source_metadata["profile"] == original["profile"]
    assert metadata == original


def test_legacy_error_response_hides_stored_credentials():
    """Read-time redaction protects existing error history without rewriting the database."""
    response = ErrorEventResponse(
        id=1,
        route="/probe",
        error_type="StatementError",
        message=f"secret={SECRET}",
        traceback=f"password={SECRET}",
        status_code=500,
        severity="error",
    )
    _assert_redacted(response.model_dump())
    assert response.error_type == "StatementError"


def test_worker_recovery_failure_exposes_only_redacted_exception(monkeypatch, caplog):
    """A second DB failure must remain a failed task without leaking its exception chain."""
    session = Mock()
    session.query.return_value.filter.return_value.first.return_value = DataSource(
        name="probe", type="s3", config={}, source_metadata={}
    )
    session.commit.side_effect = _db_failure()
    monkeypatch.setattr(tasks, "get_db_session", lambda: session)
    with pytest.raises(RuntimeError) as captured:
        tasks.ingest_data_task.run(1)
    _assert_redacted("".join(traceback.format_exception(captured.value)))
    _assert_redacted(caplog.text)
    session.close.assert_called_once()
    assert "persist ingestion failure" in str(captured.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("log_exc_info", [False, True])
@pytest.mark.parametrize("typed", [False, True])
async def test_s3_sample_logs_redact_provider_exception_chain(
    monkeypatch, caplog, typed, log_exc_info
):
    """Sample failures must not attach raw provider exceptions to log records."""

    async def fail():
        """Simulate a provider cause beneath a classified connector error."""
        try:
            raise ValueError(f"secret={SECRET}")
        except ValueError as error:
            error_type = ForbiddenException if typed else RuntimeError
            raise error_type(f"password={SECRET}") from error

    connector = Mock()
    connector.connect = fail
    monkeypatch.setattr(
        "backend.data_ingestion.connectors.s3.S3Connector", lambda *a, **kw: connector
    )
    with pytest.raises((HTTPException, SkyulfException)):
        await DataIngestionService._fetch_s3_sample(
            "s3://probe/data.csv", {}, 10, log_exc_info=log_exc_info
        )
    records = [r for r in caplog.records if r.name == "backend.data_ingestion.service"]
    assert records
    for record in records:
        _assert_redacted(logging.Formatter().format(record))
        assert record.exc_info is None
