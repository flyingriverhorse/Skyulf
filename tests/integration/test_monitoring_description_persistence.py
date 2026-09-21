"""Check description updates persist beyond the request's SQLAlchemy session."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from backend.database.models import Base, TrainingJob
from backend.dependencies import get_db
from backend.monitoring.router import router


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param({"branch_index": 2, "nested": {"keep": True}}, id="populated"),
        pytest.param({}, id="empty"),
        pytest.param(None, id="null"),
    ],
)
async def test_description_add_change_clear_persists(tmp_path, metadata):
    """A successful PATCH must survive a new session and preserve unrelated metadata."""
    database = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'descriptions.db'}")
    try:
        async with database.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with AsyncSession(database) as session:
            session.add(
                TrainingJob(
                    id="description-job",
                    pipeline_id="pipeline",
                    node_id="model",
                    dataset_source_id="source",
                    model_type="linear_regression",
                    run_mode="fixed",
                    graph={},
                    job_metadata=metadata,
                )
            )
            await session.commit()

        async def request_session():
            """Give each request its own real database session."""
            async with AsyncSession(database) as session:
                yield session

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_db] = request_session
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            for description in ("Initial description", "Changed description", ""):
                response = await client.patch(
                    "/monitoring/jobs/description-job/description",
                    json={"description": description},
                )
                assert response.status_code == 200
                assert response.json() == {"status": "ok"}
                async with AsyncSession(database) as reader:
                    saved = await reader.get(TrainingJob, "description-job")
                    assert saved is not None
                    assert saved.job_metadata == {**(metadata or {}), "description": description}
    finally:
        await database.dispose()
