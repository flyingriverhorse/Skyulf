"""Categorical PSI must survive drift calculation, persistence, and history export."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from backend.database.engine import Base
from backend.dependencies import get_db
from backend.ml_pipeline.artifacts.factory import ArtifactFactory
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.monitoring.router import _build_drift_column_summary, router
from skyulf.profiling.drift import ColumnDrift, DriftMetric, DriftReport


@pytest.mark.parametrize("metric_name", ["psi", "psi_categorical"])
@pytest.mark.parametrize("value", [0.0, 0.48])
def test_summary_preserves_numeric_and_categorical_psi(metric_name, value):
    """History must retain both shifted and stable PSI values under its shared PSI field."""
    report = DriftReport(
        reference_rows=100,
        current_rows=100,
        drifted_columns_count=int(value > 0.2),
        column_drifts={
            "feature": ColumnDrift(
                column="feature",
                drift_detected=value > 0.2,
                metrics=[
                    DriftMetric(
                        metric=metric_name, value=value, has_drift=value > 0.2, threshold=0.2
                    )
                ],
            )
        },
    )

    assert _build_drift_column_summary(report)["feature"]["psi"] == value


@pytest.mark.asyncio
async def test_calculated_psi_survives_persisted_history_and_alert_detail(tmp_path, monkeypatch):
    """A real categorical drift signal must not disappear between the report and stored history."""
    reference = pl.DataFrame({"amount": list(range(100)), "category": ["a"] * 80 + ["b"] * 20})
    current = pl.DataFrame({"amount": list(range(100)), "category": ["a"] * 50 + ["b"] * 50})
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    store.save("reference_data_ds_job-1", reference)
    discovery = SimpleNamespace(get_store_for_job=lambda job_id: store)
    monkeypatch.setattr(ArtifactFactory, "get_discovery", lambda: discovery)

    database = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'drift.db'}")

    async def database_session():
        """Open a fresh session per HTTP request to exercise committed persistence."""
        async with AsyncSession(database, expire_on_commit=False) as session:
            yield session

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = database_session
    try:
        async with database.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/monitoring/drift/calculate",
                data={"job_id": "job-1", "dataset_name": "ds"},
                files={"file": ("current.csv", current.write_csv().encode(), "text/csv")},
            )
            assert response.status_code == 200, response.text
            report = response.json()
            category_metrics = report["column_drifts"]["category"]["metrics"]
            assert category_metrics[0]["metric"] == "psi_categorical"
            category_psi = category_metrics[0]["value"]
            # (0.8 - 0.5) ln(0.8 / 0.5) + (0.2 - 0.5) ln(0.2 / 0.5).
            assert category_psi == pytest.approx(0.3 * math.log(4))
            assert report["column_drifts"]["category"]["drift_detected"] is True
            assert report["column_drifts"]["amount"]["drift_detected"] is False

            history_response = await client.get("/monitoring/drift/history/job-1")
            assert history_response.status_code == 200
            history = history_response.json()
            assert len(history) == 1
            detail_response = await client.get(f"/monitoring/drift/alerts/{report['alert_id']}")
            assert detail_response.status_code == 200
            detail = detail_response.json()
    finally:
        await database.dispose()

    assert history[0]["summary"]["category"]["psi"] == category_psi
    assert history[0]["summary"]["amount"]["psi"] == 0.0
    assert detail["summary"] == history[0]["summary"]
    assert detail["column_drifts"] == report["column_drifts"]

    # Keep the browser fixture anchored to the real producer and persisted
    # summary. Histograms and timestamps are irrelevant to this PSI contract.
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "frontend/ml-canvas/e2e/fixtures/drift-mixed-report.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    for drift in report["column_drifts"].values():
        drift.pop("distribution", None)
    assert fixture["report"]["column_drifts"] == report["column_drifts"]
    assert fixture["history"][0]["summary"] == history[0]["summary"]
