"""Drift threshold form validation must reject unusable overrides before evaluation."""

from types import SimpleNamespace

import polars as pl
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from backend.database.engine import Base
from backend.dependencies import get_db
from backend.ml_pipeline.artifacts.factory import ArtifactFactory
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.monitoring.router import router


@pytest_asyncio.fixture
async def drift_client(tmp_path, monkeypatch):
    """Exercise real upload, calculation, and persistence with isolated artifacts and SQLite."""
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    store.save("reference_data_ds_job-1", pl.DataFrame({"value": [0.0, 1.0, 2.0]}))
    discovery = SimpleNamespace(get_store_for_job=lambda job_id: store)
    monkeypatch.setattr(ArtifactFactory, "get_discovery", lambda: discovery)
    database = create_async_engine("sqlite+aiosqlite:///:memory:")
    statements = []

    def record_statement(connection, cursor, statement, parameters, context, executemany):
        """Observe actual database work to catch invalid requests reaching the route body."""
        statements.append(statement)

    async def database_session():
        """Provide a fresh session for each HTTP request."""
        async with AsyncSession(database, expire_on_commit=False) as session:
            yield session

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = database_session
    try:
        async with database.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        event.listen(database.sync_engine, "before_cursor_execute", record_statement)
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://testserver",
        ) as client:
            yield client, statements
    finally:
        await database.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        (field, value)
        for field in ("psi", "ks", "wasserstein", "kl")
        for value in ("-0.1", "nan", "inf", "-inf")
    ]
    + [("ks", "1.001")],
)
async def test_invalid_drift_thresholds_fail_before_evaluation(drift_client, field, value):
    """Negative, nonfinite, and impossible KS overrides must never become stored evaluations."""
    client, statements = drift_client
    response = await client.post(
        "/monitoring/drift/calculate",
        data={"job_id": "job-1", "dataset_name": "ds", f"threshold_{field}": value},
        files={"file": ("current.csv", b"value\n0\n1\n2\n", "text/csv")},
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["loc"] == ["body", f"threshold_{field}"]
    assert statements == []


@pytest.mark.asyncio
@pytest.mark.parametrize("psi,ks,wasserstein,kl", [(0.0, 0.0, 0.0, 0.0), (2.5, 1.0, 3.0, 4.0)])
async def test_valid_threshold_boundaries_survive_persistence(
    drift_client, psi, ks, wasserstein, kl
):
    """Zero cutoffs and unbounded non-KS weights must remain usable and retain their actual values."""
    client, _ = drift_client
    thresholds = {
        "threshold_psi": psi,
        "threshold_ks": ks,
        "threshold_wasserstein": wasserstein,
        "threshold_kl": kl,
    }
    response = await client.post(
        "/monitoring/drift/calculate",
        data={"job_id": "job-1", "dataset_name": "ds", **thresholds},
        files={"file": ("current.csv", b"value\n0\n1\n2\n", "text/csv")},
    )
    assert response.status_code == 200, response.text
    history = await client.get("/monitoring/drift/history/job-1")
    assert history.status_code == 200
    row = history.json()[0]
    assert {field: row[field] for field in thresholds} == thresholds
