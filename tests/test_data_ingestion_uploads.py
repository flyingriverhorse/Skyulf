import io
from pathlib import Path
from types import SimpleNamespace

# pyright: reportMissingImports=false
import pytest
import pytest_asyncio
from starlette.datastructures import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.data_ingestion.exceptions import FileUploadError
from core.data_ingestion.service import DataIngestionService
from core.database.models import Base, DataSource


@pytest_asyncio.fixture()
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    async with session_maker() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_handle_file_upload_creates_source_and_file(db_session, tmp_path):
    service = DataIngestionService(db_session, upload_dir=str(tmp_path))
    current_user = SimpleNamespace(id=None, username="tester")

    content = io.BytesIO(b"col\n1\n2\n")
    upload = UploadFile(filename="dataset.csv", file=content)

    result = await service.handle_file_upload(upload, current_user)

    assert result["success"] is True
    saved_path = Path(result["file_info"]["file_path"])
    assert saved_path.exists()

    sources = (await db_session.execute(select(DataSource))).scalars().all()
    assert len(sources) == 1
    assert sources[0].config.get("file_hash")


@pytest.mark.asyncio
async def test_handle_file_upload_detects_duplicates(db_session, tmp_path):
    service = DataIngestionService(db_session, upload_dir=str(tmp_path))
    current_user = SimpleNamespace(id=None, username="tester")

    first_upload = UploadFile(
        filename="duplicate.csv",
        file=io.BytesIO(b"col\n1\n")
    )
    await service.handle_file_upload(first_upload, current_user)

    second_upload = UploadFile(
        filename="duplicate.csv",
        file=io.BytesIO(b"col\n1\n")
    )
    second_result = await service.handle_file_upload(second_upload, current_user)

    assert second_result["is_duplicate"] is True
    assert second_result["duplicate_of"]

    sources = (await db_session.execute(select(DataSource))).scalars().all()
    assert len(sources) == 1


@pytest.mark.asyncio
async def test_handle_file_upload_rejects_disallowed_extension(db_session, tmp_path):
    service = DataIngestionService(db_session, upload_dir=str(tmp_path))
    service.allowed_extensions = {".csv"}
    current_user = SimpleNamespace(id=None, username="tester")

    upload = UploadFile(
        filename="payload.exe",
        file=io.BytesIO(b"binary")
    )

    with pytest.raises(FileUploadError):
        await service.handle_file_upload(upload, current_user)
