"""Non-finite drift inputs must persist failed evaluations without invalid metric JSON."""

import io
import json
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from starlette.requests import Request

from backend.database.engine import Base
from backend.database.models import DriftCheckResult
from backend.exceptions.core import SkyulfException
from backend.ml_pipeline.artifacts.factory import ArtifactFactory
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.monitoring.router import calculate_drift, get_drift_alert


@pytest.mark.parametrize(
    "current_values",
    [[0.0, float("inf")], ["0", "1e400"]],
    ids=["numeric-infinity", "overflowing-numeric-text"],
)
@pytest.mark.asyncio
async def test_nonfinite_input_records_a_durable_failed_drift_evaluation(
    current_values, tmp_path, monkeypatch
):
    """Actual upload parsing and Core validation must prevent completed infinite-metric alerts."""
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    store.save("reference_data_ds_job-1", pl.DataFrame({"feature": [0, 1]}))
    discovery = SimpleNamespace(get_store_for_job=lambda job_id: store)
    monkeypatch.setattr(ArtifactFactory, "get_discovery", lambda: discovery)

    # Parquet preserves the two distinct ingestion cases: a numeric infinity
    # and text that overflows only when Core parses it as a numeric value.
    upload_bytes = io.BytesIO()
    pl.DataFrame({"feature": current_values}).write_parquet(upload_bytes)
    upload_bytes.seek(0)
    upload = UploadFile(file=upload_bytes, filename="current.parquet")
    database = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'drift.db'}")
    try:
        async with database.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with AsyncSession(database, expire_on_commit=False) as session:
            with pytest.raises(SkyulfException, match="Drift calculation failed") as failure:
                await calculate_drift(
                    request=Request(
                        {
                            "type": "http",
                            "path": "/monitoring/drift/calculate",
                            "client": ("127.0.0.1", 1234),
                        }
                    ),
                    job_id="job-1",
                    file=upload,
                    dataset_name="ds",
                    threshold_psi=None,
                    threshold_ks=None,
                    threshold_wasserstein=None,
                    threshold_kl=None,
                    db=session,
                )
            assert failure.value.status_code == 500
            assert failure.value.error_code == "INTERNAL_SERVER_ERROR"

        # A fresh session proves the route committed the failure to durable
        # storage, rather than merely adding an object to its session.
        async with AsyncSession(database) as session:
            alerts = (await session.scalars(select(DriftCheckResult))).all()
            assert len(alerts) == 1
            alert = alerts[0]
            assert alert.evaluation_status == "failed"
            assert alert.job_id == "job-1" and alert.dataset_name == "ds"
            assert alert.summary is None and alert.column_drifts is None
            detail = await get_drift_alert(alert.id, db=session)
            payload = json.loads(json.dumps(detail.model_dump(mode="json"), allow_nan=False))
    finally:
        await upload.close()
        await database.dispose()

    assert payload["evaluation_status"] == "failed"
    assert payload["column_drifts"] is None
    assert "Drift column 'feature' contains infinite values" in payload["error_message"]
    assert "finite or missing values before comparison" in payload["error_message"]
