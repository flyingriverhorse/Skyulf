"""Focused unit tests for `backend.data_ingestion.service.DataIngestionService`
helper methods that were extracted during a pure-refactor (extract-method
complexity cleanup) and lost direct test coverage attribution in the process:

- `_fetch_s3_sample` (S3 sample fetch happy path + generic-exception mapping)
- `_sample_local_parquet` / `_sample_s3_or_parquet` (local parquet branch)
- `_trigger_ingestion` (Celery / BackgroundTasks / thread-fallback branches)
- The file-upload chain: `_check_declared_upload_size`,
  `_validate_upload_filename`, `_save_uploaded_file`,
  `_create_file_source_and_ingest`, and the `handle_file_upload` orchestrator.

These call the (previously-private, still-accessible) helper methods directly
so each branch is exercised in isolation, independent of the router/HTTP
integration tests elsewhere in the suite.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.config import get_settings
from backend.data_ingestion.service import DataIngestionService
from backend.exceptions.core import SkyulfException

settings = get_settings()
DATABASE_URL = f"sqlite+aiosqlite:///{settings.DB_PATH}"
engine = create_async_engine(DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
async def db_session():
    async with TestingSessionLocal() as session:
        yield session


class _FakeUploadFile:
    """Minimal stand-in for fastapi.UploadFile, driven by an in-memory buffer."""

    def __init__(self, filename: str, content: bytes, content_length: str | None = None):
        self.filename = filename
        self._content = content
        self._offset = 0
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = content_length

    async def read(self, size: int) -> bytes:
        chunk = self._content[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


# ── _fetch_s3_sample ─────────────────────────────────────────────────────


class _FakeS3DataFrame:
    def to_dicts(self):
        return [{"a": 1, "b": 2}]


class _FakeS3Connector:
    def __init__(self, *_args, **_kwargs):
        pass

    async def connect(self):
        return None

    async def fetch_data(self, limit):
        return _FakeS3DataFrame()


@pytest.mark.asyncio
async def test_fetch_s3_sample_happy_path_returns_dicts():
    """Successful S3 fetch returns the connector's rows as dicts."""
    with patch("backend.data_ingestion.connectors.s3.S3Connector", _FakeS3Connector):
        rows = await DataIngestionService._fetch_s3_sample(
            "s3://bucket/key.csv", {"key": "abc", "secret": None}, limit=10
        )
    assert rows == [{"a": 1, "b": 2}]


@pytest.mark.asyncio
async def test_fetch_s3_sample_generic_exception_maps_to_skyulf_exception():
    """A non-typed exception from the connector maps to a generic SkyulfException,
    not a raw/unclassified error."""
    with (
        patch("backend.data_ingestion.connectors.s3.S3Connector.connect", new=AsyncMock()),
        patch(
            "backend.data_ingestion.connectors.s3.S3Connector.fetch_data",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
        pytest.raises(SkyulfException),
    ):
        await DataIngestionService._fetch_s3_sample("s3://bucket/key.csv", {}, limit=5)


# ── _sample_local_parquet / _sample_s3_or_parquet ───────────────────────


@pytest.mark.asyncio
async def test_sample_local_parquet_reads_local_file(tmp_path):
    """A local (non-s3://) parquet path is read via LocalFileConnector."""
    import polars as pl

    parquet_path = tmp_path / "data.parquet"
    pl.DataFrame({"a": [1, 2], "b": [3, 4]}).write_parquet(parquet_path)

    rows = await DataIngestionService._sample_local_parquet(str(parquet_path), limit=10)
    assert rows == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]


@pytest.mark.asyncio
async def test_sample_local_parquet_missing_file_raises_skyulf_exception(tmp_path):
    """A nonexistent local parquet path surfaces as a SkyulfException, not a raw error."""
    missing = tmp_path / "does_not_exist.parquet"
    with pytest.raises(SkyulfException):
        await DataIngestionService._sample_local_parquet(str(missing), limit=10)


@pytest.mark.asyncio
async def test_sample_s3_or_parquet_missing_path_raises_400(db_session):
    service = DataIngestionService(session=db_session)
    with pytest.raises(HTTPException) as exc_info:
        await service._sample_s3_or_parquet(None, {}, limit=10)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_sample_s3_or_parquet_local_path_delegates_to_local_parquet(db_session, tmp_path):
    """A non-s3:// path routes to `_sample_local_parquet` rather than S3Connector."""
    import polars as pl

    parquet_path = tmp_path / "local.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(parquet_path)

    service = DataIngestionService(session=db_session)
    rows = await service._sample_s3_or_parquet(str(parquet_path), {}, limit=10)
    assert rows == [{"x": 1}]


# ── _trigger_ingestion ───────────────────────────────────────────────────


class _SettingsStub:
    def __init__(self, use_celery: bool):
        self.USE_CELERY = use_celery


@pytest.mark.asyncio
async def test_trigger_ingestion_uses_celery_delay_when_enabled(db_session):
    service = DataIngestionService(session=db_session)
    with patch("backend.data_ingestion.service.ingest_data_task") as mock_task:
        service._trigger_ingestion(_SettingsStub(use_celery=True), 42, background_tasks=None)
    mock_task.delay.assert_called_once_with(42)


@pytest.mark.asyncio
async def test_trigger_ingestion_uses_background_tasks_when_provided(db_session):
    service = DataIngestionService(session=db_session)
    background_tasks = BackgroundTasks()
    with patch("backend.data_ingestion.service.ingest_data_task") as mock_task:
        service._trigger_ingestion(
            _SettingsStub(use_celery=False), 42, background_tasks=background_tasks
        )
    assert len(background_tasks.tasks) == 1
    # Run the queued task to confirm it targets ingest_data_task(42).
    await background_tasks()
    mock_task.assert_called_once_with(42)


@pytest.mark.asyncio
async def test_trigger_ingestion_falls_back_to_thread_when_no_background_tasks(db_session):
    """Without Celery or BackgroundTasks, ingestion runs via a fire-and-forget thread."""
    service = DataIngestionService(session=db_session)
    with patch("backend.data_ingestion.service.ingest_data_task") as mock_task:
        service._trigger_ingestion(_SettingsStub(use_celery=False), 42, background_tasks=None)
        # Give the created asyncio task a chance to run to completion.
        import asyncio

        await asyncio.sleep(0.2)
    mock_task.assert_called_once_with(42)


# ── File-upload chain ────────────────────────────────────────────────────


def test_check_declared_upload_size_no_header_is_noop():
    file = _FakeUploadFile("a.csv", b"data")
    # No content-length header -> should return without raising.
    DataIngestionService._check_declared_upload_size(file, settings)


def test_check_declared_upload_size_malformed_header_is_noop():
    file = _FakeUploadFile("a.csv", b"data", content_length="not-a-number")
    # Malformed header -> falls through, doesn't raise here (streaming check catches it).
    DataIngestionService._check_declared_upload_size(file, settings)


def test_check_declared_upload_size_over_limit_raises_413():
    file = _FakeUploadFile("a.csv", b"data", content_length=str(settings.MAX_UPLOAD_SIZE + 1))
    with pytest.raises(HTTPException) as exc_info:
        DataIngestionService._check_declared_upload_size(file, settings)
    assert exc_info.value.status_code == 413


def test_validate_upload_filename_accepts_valid_csv():
    file = _FakeUploadFile("data.csv", b"a,b\n1,2\n")
    raw_name, file_ext = DataIngestionService._validate_upload_filename(file, settings)
    assert raw_name == "data.csv"
    assert file_ext == ".csv"


def test_validate_upload_filename_rejects_path_traversal():
    file = _FakeUploadFile("../../etc/passwd", b"x")
    with pytest.raises(HTTPException) as exc_info:
        DataIngestionService._validate_upload_filename(file, settings)
    assert exc_info.value.status_code == 400


def test_validate_upload_filename_rejects_unsupported_extension():
    file = _FakeUploadFile("malware.exe", b"x")
    with pytest.raises(HTTPException) as exc_info:
        DataIngestionService._validate_upload_filename(file, settings)
    assert exc_info.value.status_code == 415


@pytest.mark.asyncio
async def test_save_uploaded_file_writes_full_content(tmp_path):
    file = _FakeUploadFile("data.csv", b"a,b\n1,2\n")
    dest = tmp_path / "saved.csv"
    await DataIngestionService._save_uploaded_file(file, dest, settings)
    assert dest.read_bytes() == b"a,b\n1,2\n"


@pytest.mark.asyncio
async def test_save_uploaded_file_over_limit_raises_413_and_cleans_up(tmp_path):
    file = _FakeUploadFile("data.csv", b"x" * 100)
    dest = tmp_path / "saved.csv"

    class _TinyLimit:
        MAX_UPLOAD_SIZE = 10

    with pytest.raises(HTTPException) as exc_info:
        await DataIngestionService._save_uploaded_file(file, dest, _TinyLimit())
    assert exc_info.value.status_code == 413
    assert not dest.exists()


@pytest.mark.asyncio
async def test_create_file_source_and_ingest_happy_path(db_session, tmp_path):
    file_path = tmp_path / "uploaded.csv"
    file_path.write_text("a,b\n1,2\n")

    service = DataIngestionService(session=db_session)
    with patch("backend.data_ingestion.service.ingest_data_task") as mock_task:
        response = await service._create_file_source_and_ingest(
            file_id=str(uuid.uuid4()),
            raw_name="uploaded.csv",
            file_path=file_path,
            user_id=1,
            background_tasks=None,
        )
        # No BackgroundTasks and USE_CELERY is False in test settings -> the
        # thread-fallback branch schedules an asyncio task; give it a beat.
        import asyncio

        await asyncio.sleep(0.2)
    assert response.status == "pending"
    assert response.job_id
    mock_task.assert_called_once()


@pytest.mark.asyncio
async def test_create_file_source_and_ingest_db_failure_cleans_up_file(tmp_path):
    """A DB failure during row creation must delete the just-saved file and
    surface a SkyulfException instead of leaving an orphaned upload."""
    file_path = tmp_path / "uploaded.csv"
    file_path.write_text("a,b\n1,2\n")

    bad_session = MagicMock()
    bad_session.add = MagicMock()
    bad_session.commit = AsyncMock(side_effect=RuntimeError("db down"))

    service = DataIngestionService(session=bad_session)
    with pytest.raises(SkyulfException):
        await service._create_file_source_and_ingest(
            file_id=str(uuid.uuid4()),
            raw_name="uploaded.csv",
            file_path=file_path,
            user_id=1,
            background_tasks=None,
        )
    assert not file_path.exists()


@pytest.mark.asyncio
async def test_handle_file_upload_end_to_end(db_session, tmp_path):
    """Full orchestration: declared-size check, filename validation, streamed
    save, DataSource creation, and ingestion trigger all succeed together."""
    service = DataIngestionService(session=db_session, upload_dir=str(tmp_path / "uploads"))
    file = _FakeUploadFile("dataset.csv", b"a,b\n1,2\n3,4\n")

    with patch("backend.data_ingestion.service.ingest_data_task") as mock_task:
        response = await service.handle_file_upload(file, user_id=7, background_tasks=None)
        # Thread-fallback branch (no BackgroundTasks, USE_CELERY False) -
        # give the scheduled asyncio task a beat to run.
        import asyncio

        await asyncio.sleep(0.2)

    assert response.status == "pending"
    assert response.file_id
    mock_task.assert_called_once()

    source = await service.get_source(int(response.job_id))
    assert source is not None
    assert source.type == "file"
    assert source.config["file_path"].endswith(".csv")


@pytest.mark.asyncio
async def test_handle_file_upload_rejects_unsupported_extension(db_session, tmp_path):
    service = DataIngestionService(session=db_session, upload_dir=str(tmp_path / "uploads"))
    file = _FakeUploadFile("malware.exe", b"x")

    with pytest.raises(HTTPException) as exc_info:
        await service.handle_file_upload(file, user_id=7, background_tasks=None)
    assert exc_info.value.status_code == 415
