"""Regression tests for backend/data_ingestion security/robustness fixes (round 5):

1. `DataIngestionService.get_sample()` — defense-in-depth path containment
   guard for the "file"/"csv"/"txt" branch, which previously read
   `source.config["file_path"]` directly via `DataService.get_sample`
   without going through `LocalFileConnector`'s containment check at all.
2. `DataIngestionService.delete_source()` — a failed on-disk file removal
   must not block deleting the DB row, and must log a clear, discoverable
   "orphaned file" error instead of being silently swallowed.
3. `DataIngestionService.get_sample()` S3 branch — error classification now
   uses typed exceptions (`ForbiddenException`/`ResourceNotFoundException`)
   raised by `S3Connector` instead of substring-matching `str(exc)` for
   "403"/"404".
4. `backend.config.mixins.files.FilesMixin.ALLOWED_EXTENSIONS` no longer
   allows `.pkl`/`.pickle`/`.h5`/`.hdf5` uploads (latent deserialization RCE
   risk with no legitimate reader for these formats in the ingestion path).
"""

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.config import get_settings
from backend.data_ingestion.connectors.file import LocalFileConnector
from backend.data_ingestion.service import DataIngestionService
from backend.database.models import DataSource
from backend.exceptions.core import ForbiddenException, ResourceNotFoundException

settings = get_settings()
DATABASE_URL = f"sqlite+aiosqlite:///{settings.DB_PATH}"
engine = create_async_engine(DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
async def db_session():
    async with TestingSessionLocal() as session:
        yield session


async def _make_file_source(db_session, file_path: str) -> DataSource:
    source = DataSource(
        source_id=f"test-src-{uuid.uuid4()}",
        name="test source",
        type="file",
        config={"file_path": file_path},
        created_by=None,
        is_active=True,
        test_status="success",
    )
    db_session.add(source)
    await db_session.commit()
    await db_session.refresh(source)
    return source


# ── 1. get_sample() path containment (defense-in-depth) ────────────────


class TestLocalFileConnectorResolveSafePath:
    """Unit tests for the shared containment helper (testing=False forces
    the real containment check, bypassing the TESTING-mode skip)."""

    def test_contained_path_resolves(self, tmp_path):
        base = tmp_path / "uploads"
        base.mkdir()
        inside = base / "data.csv"
        inside.write_text("a,b\n1,2\n")

        resolved = LocalFileConnector.resolve_safe_path(str(inside), base_path=base, testing=False)
        assert resolved == inside.resolve()

    def test_traversal_outside_base_rejected(self, tmp_path):
        base = tmp_path / "uploads"
        base.mkdir()
        outside = tmp_path / "secret.csv"
        outside.write_text("a,b\n1,2\n")

        with pytest.raises(PermissionError):
            LocalFileConnector.resolve_safe_path(str(outside), base_path=base, testing=False)

    def test_relative_traversal_rejected(self, tmp_path):
        base = tmp_path / "uploads"
        base.mkdir()

        with pytest.raises(PermissionError):
            LocalFileConnector.resolve_safe_path(
                "../../../../etc/passwd", base_path=base, testing=False
            )


@pytest.mark.asyncio
async def test_get_sample_file_branch_rejects_path_outside_upload_dir(db_session):
    """`get_sample()`'s file/csv/txt branch must reject a `file_path` that
    escapes the upload directory instead of reading it directly."""
    service = DataIngestionService(session=db_session)
    source = await _make_file_source(db_session, "/etc/passwd")

    with (
        patch.object(
            LocalFileConnector,
            "resolve_safe_path",
            side_effect=PermissionError(
                "File path resolves outside the configured upload directory"
            ),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await service.get_sample(source.id)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_sample_file_branch_allows_contained_path(db_session, tmp_path):
    """A well-behaved, contained file path still works normally."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n")

    service = DataIngestionService(session=db_session)
    source = await _make_file_source(db_session, str(csv_path))

    # resolve_safe_path runs in TESTING mode (skips containment) like the
    # rest of the local-file-connector test suite, but must still return a
    # usable path so the sample is read successfully.
    rows = await service.get_sample(source.id, limit=5)
    assert rows == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


# ── 2. delete_source() orphaned-file handling ───────────────────────────


@pytest.mark.asyncio
async def test_delete_source_logs_orphaned_file_and_still_deletes_record(db_session, caplog):
    """If on-disk file removal fails, the DB row must still be deleted and
    a discoverable ERROR must be logged referencing the orphaned file."""
    service = DataIngestionService(session=db_session)
    source = await _make_file_source(db_session, "/nonexistent/does-not-matter.csv")

    with patch("backend.data_ingestion.service.Path") as mock_path_cls:
        mock_path = mock_path_cls.return_value
        mock_path.exists.return_value = True
        mock_path.unlink.side_effect = PermissionError("Permission denied")

        with caplog.at_level("ERROR"):
            result = await service.delete_source(source.id)

    assert result is True
    assert any("Orphaned file" in rec.message for rec in caplog.records)

    # The DB row is gone despite the failed file removal.
    refetched = await service.get_source(source.id)
    assert refetched is None


@pytest.mark.asyncio
async def test_delete_source_removes_file_and_record_on_success(db_session, tmp_path):
    """Happy path: file removed, DB row removed, no error logged."""
    file_path = tmp_path / "to_delete.csv"
    file_path.write_text("a,b\n1,2\n")

    service = DataIngestionService(session=db_session)
    source = await _make_file_source(db_session, str(file_path))

    result = await service.delete_source(source.id)

    assert result is True
    assert not file_path.exists()
    assert await service.get_source(source.id) is None


# ── 3. get_sample() S3 typed-exception error classification ─────────────


@pytest.mark.asyncio
async def test_get_sample_s3_forbidden_maps_to_400(db_session):
    """A `ForbiddenException` from S3Connector must map to HTTP 400 via an
    isinstance check, not by substring-matching the error message."""
    service = DataIngestionService(session=db_session)
    source = DataSource(
        source_id=f"test-s3-forbidden-{uuid.uuid4()}",
        name="s3 source",
        type="s3",
        config={"path": "s3://bucket/key.csv"},
        created_by=None,
        is_active=True,
        test_status="success",
    )
    db_session.add(source)
    await db_session.commit()
    await db_session.refresh(source)

    with (
        patch("backend.data_ingestion.connectors.s3.S3Connector.connect", new=AsyncMock()),
        patch(
            "backend.data_ingestion.connectors.s3.S3Connector.fetch_data",
            new=AsyncMock(side_effect=ForbiddenException(message="Access denied")),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await service.get_sample(source.id)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_s3_connector_get_schema_classifies_403_as_forbidden_exception():
    """S3Connector.get_schema classifies a provider 403 error into a typed
    `ForbiddenException` instead of leaving callers to string-match."""
    from backend.data_ingestion.connectors.s3 import S3Connector

    connector = S3Connector("s3://bucket/private.bin", storage_options={})

    with (
        patch("polars.scan_csv", side_effect=Exception("Access Denied: status 403")),
        patch("polars.scan_parquet", side_effect=Exception("Access Denied: status 403")),
        pytest.raises(ForbiddenException),
    ):
        await connector.get_schema()


@pytest.mark.asyncio
async def test_s3_connector_get_schema_classifies_404_as_not_found_exception():
    from backend.data_ingestion.connectors.s3 import S3Connector

    connector = S3Connector("s3://bucket/missing.parquet", storage_options={})

    with (
        patch("polars.scan_csv", side_effect=Exception("Not Found: status 404")),
        patch("polars.scan_parquet", side_effect=Exception("Not Found: status 404")),
        pytest.raises(ResourceNotFoundException),
    ):
        await connector.get_schema()


# ── 4. Upload extension allow-list no longer includes pkl/h5 ───────────


def test_allowed_extensions_excludes_unsafe_deserialization_formats():
    unsafe = {".pkl", ".pickle", ".h5", ".hdf5"}
    allowed = {e.lower() for e in settings.ALLOWED_EXTENSIONS}
    assert unsafe.isdisjoint(allowed)


def test_allowed_extensions_still_includes_safe_dataset_formats():
    expected = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".txt", ".feather"}
    allowed = {e.lower() for e in settings.ALLOWED_EXTENSIONS}
    assert expected == allowed
