"""Metric selections use the same aggregate contract through the HTTP route and Core."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import polars as pl
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.dependencies import get_db
from backend.eda.router import router


def test_decomposition_metric_selection_round_trip(monkeypatch) -> None:
    """Count, revenue sum, and average must return 2, 30, and 15 for two input rows."""
    monkeypatch.setattr(
        "backend.eda.router._prepare_decomposition_dataframe",
        AsyncMock(return_value=pl.DataFrame({"revenue": [10, 20]})),
    )
    monkeypatch.setattr("backend.eda.router.limiter.enabled", False)
    session = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(id=6161)))
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = lambda: session
    with TestClient(app) as client:
        for column, aggregation, expected in [
            (None, "count", 2),
            ("revenue", "sum", 30),
            ("revenue", "mean", 15),
            (None, "count", 2),
        ]:
            response = client.post(
                "/eda/6161/decomposition",
                json={
                    "measure_col": column,
                    "measure_agg": aggregation,
                    "split_col": "",
                    "filters": [],
                },
            )
            assert response.status_code == 200
            assert response.json() == [{"name": "Total", "value": expected, "ratio": 1.0}]
