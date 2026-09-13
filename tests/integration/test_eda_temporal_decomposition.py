"""Temporal decomposition drill-down survives real file loading and HTTP serialization."""

from datetime import UTC, date, datetime, time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.dependencies import get_db
from backend.eda.router import router


@pytest.mark.parametrize(
    "values",
    [
        [date(2026, 1, 1), date(2026, 1, 2)],
        [datetime(2026, 1, 1, 9, 30, 0, 123456), datetime(2026, 1, 1, 16, 45)],
        [
            datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
            datetime(2026, 1, 1, 16, 45, tzinfo=UTC),
        ],
        [time(9, 30), time(16, 45)],
    ],
    ids=["date", "datetime", "timezone", "time"],
)
def test_temporal_drill_down_via_http_and_native_parquet(monkeypatch, tmp_path, values):
    """Actual temporal file dtypes must accept the JSON filter emitted by a prior response."""
    frame = pl.DataFrame({"when": values, "detail": ["morning", "evening"], "amount": [2, 5]})
    path = tmp_path / "temporal.parquet"
    frame.write_parquet(path)
    source = SimpleNamespace(
        id=288, config={"file_path": str(path)}, source_metadata={}, source_id="temporal.parquet"
    )
    session = SimpleNamespace(get=AsyncMock(return_value=source))
    monkeypatch.setattr("backend.eda.router.limiter.enabled", False)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = lambda: session

    with TestClient(app) as client:
        response = client.post(
            "/eda/288/decomposition",
            json={"measure_col": "amount", "measure_agg": "sum", "split_col": "when"},
        )
        assert response.status_code == 200
        buckets = response.json()
        assert len(buckets) == 2
        for bucket in buckets:
            response = client.post(
                "/eda/288/decomposition",
                json={
                    "measure_col": "amount",
                    "measure_agg": "sum",
                    "split_col": "detail",
                    "filters": [
                        {"column": "when", "operator": "==", "value": bucket["filter_value"]}
                    ],
                },
            )
            assert response.status_code == 200
            assert response.json() == [
                {
                    "name": "morning" if bucket["value"] == 2 else "evening",
                    "filter_value": "morning" if bucket["value"] == 2 else "evening",
                    "value": bucket["value"],
                    "ratio": 1.0,
                }
            ]
