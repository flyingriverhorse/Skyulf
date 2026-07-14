"""Regression tests for backend/data_ingestion + backend/ml_pipeline router bugs.

- POST /api/ingestion/database (DataIngestionService.handle_create_source):
  the frontend called this endpoint but the backend never defined it, so
  every "Add Data Source" (S3) attempt 404'd.
- GET /api/pipeline/datasets/{id}/schema: an intentional 400 HTTPException
  was swallowed by a broad except and re-raised as a 500.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.config import get_settings
from backend.database.models import DataSource
from backend.main import app

settings = get_settings()
DATABASE_URL = f"sqlite+aiosqlite:///{settings.DB_PATH}"
engine = create_async_engine(DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
async def db_session():
    async with TestingSessionLocal() as session:
        yield session


@pytest.fixture
def client():
    with TestClient(app, base_url="http://localhost") as c:
        yield c


@pytest.mark.asyncio
async def test_create_s3_source_returns_pending_job(client, db_session):
    """A valid S3 config creates a DataSource row and returns a pending job."""
    # Avoid actually running ingestion against a real S3 bucket in the
    # background task triggered by this request.
    with patch("backend.data_ingestion.service.ingest_data_task") as mock_task:
        response = client.post(
            "/api/ingestion/database",
            json={
                "name": "Test S3 Source",
                "type": "s3",
                "config": {"path": "s3://bucket/key.csv"},
            },
        )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "pending"
    assert data["job_id"]
    mock_task.assert_called_once()  # USE_CELERY is False -> BackgroundTasks calls it directly (not .delay())

    source = await db_session.get(DataSource, int(data["job_id"]))
    assert source is not None
    assert source.type == "s3"
    assert source.config == {"path": "s3://bucket/key.csv"}


def test_create_source_rejects_unsupported_type(client):
    response = client.post(
        "/api/ingestion/database",
        json={"name": "Bad Source", "type": "postgres", "config": {}},
    )
    assert response.status_code == 400
    assert "Unsupported source type" in response.json()["message"]


def test_create_s3_source_requires_path(client):
    response = client.post(
        "/api/ingestion/database",
        json={"name": "No Path", "type": "s3", "config": {}},
    )
    assert response.status_code == 400
    assert "path" in response.json()["message"].lower()


def test_create_s3_source_rejects_local_path(client):
    """A type="s3" source whose path isn't an s3:// URI would otherwise be
    treated as an arbitrary local filesystem path downstream (see
    `get_source_sample`), with no traversal/extension checks — reject it here
    instead of letting it slip through as a path-traversal-style read."""
    response = client.post(
        "/api/ingestion/database",
        json={"name": "Sneaky Local Path", "type": "s3", "config": {"path": "/etc/passwd"}},
    )
    assert response.status_code == 400
    assert "s3://" in response.json()["message"]


@pytest.mark.asyncio
async def test_dataset_schema_unresolvable_path_returns_400(client, db_session):
    """GET /api/pipeline/datasets/{id}/schema for a dataset with no resolvable
    file path used to raise an HTTPException(400) inside a try/except that
    caught it and re-raised as a generic 500 SkyulfException, masking the
    real, actionable 400 error."""
    import uuid

    ds = DataSource(
        name="No Path Source",
        type="file",
        source_id=f"no-path-source-{uuid.uuid4()}",
        test_status="untested",
        config={},  # no file_path -> extract_file_path_from_source() returns None
    )
    db_session.add(ds)
    await db_session.commit()
    await db_session.refresh(ds)

    response = client.get(f"/api/pipeline/datasets/{ds.id}/schema")
    assert response.status_code == 400
    assert "could not resolve path" in response.json()["message"].lower()
