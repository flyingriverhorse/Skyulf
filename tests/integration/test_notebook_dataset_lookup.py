"""Keep notebook dataset resolution aligned with Canvas data-source identifiers."""

from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.database.models import DataSource
from backend.ml_pipeline._internal._routers.notebook_export import (
    _lookup_dataset_file_path,
)


@pytest_asyncio.fixture
async def dataset_session(tmp_path):
    """Use an isolated database so uploaded dataset probes cannot touch user data."""
    (tmp_path / "actual-upload.csv").write_text("value,label\n1,0\n", encoding="utf-8")
    (tmp_path / "wrong-upload.csv").write_text("value,label\n2,1\n", encoding="utf-8")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(DataSource.__table__.create)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as session:
            session.add_all(
                [
                    DataSource(
                        id=1456,
                        source_id="uploaded-csv-uuid",
                        name="Uploaded CSV",
                        type="file",
                        config={"file_path": str(tmp_path / "actual-upload.csv")},
                    ),
                    DataSource(
                        id=1457,
                        source_id="1456",
                        name="Numeric UUID collision",
                        type="file",
                        config={"file_path": str(tmp_path / "wrong-upload.csv")},
                    ),
                    DataSource(
                        id=1458,
                        source_id="no-file",
                        name="Remote source",
                        type="postgres",
                        config={},
                    ),
                ]
            )
            await session.commit()
            yield session
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("dataset_id", ["1456", "01456", "uploaded-csv-uuid"])
async def test_lookup_resolves_canvas_id_or_source_uuid(dataset_session, tmp_path, dataset_id):
    """Canvas numeric IDs must export the uploaded file, even with a colliding source UUID."""
    result = await _lookup_dataset_file_path(dataset_id, dataset_session)

    assert result == Path(tmp_path / "actual-upload.csv").as_posix()


@pytest.mark.asyncio
@pytest.mark.parametrize("dataset_id", ["99999", "unknown-source", "no-file"])
async def test_lookup_retains_missing_file_fallback(dataset_session, dataset_id):
    """Missing datasets and non-file sources must retain the notebook loader fallback."""
    result = await _lookup_dataset_file_path(dataset_id, dataset_session)

    assert result is None
