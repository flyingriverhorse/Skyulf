"""HTTP decomposition round trips preserve missing and literal category identities."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.dependencies import get_db
from backend.eda.router import router


@pytest.mark.parametrize("split_col", ["group", "value", "ratio"])
def test_decomposition_http_drill_down_keeps_missing_rows_separate(
    monkeypatch, split_col: str
) -> None:
    """Request validation and JSON serialization must retain null-valued bucket filters."""
    frame = pl.DataFrame(
        {
            split_col: ["a", None, "Unknown", None],
            "detail": ["normal", "missing_a", "literal", "missing_b"],
            "amount": [1, 2, 3, 4],
        }
    )
    monkeypatch.setattr(
        "backend.eda.router._prepare_decomposition_dataframe", AsyncMock(return_value=frame)
    )
    monkeypatch.setattr("backend.eda.router.limiter.enabled", False)
    session = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(id=192)))
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = lambda: session
    body = {"measure_col": "amount", "measure_agg": "sum", "split_col": split_col, "filters": []}

    with TestClient(app) as client:
        response = client.post("/eda/192/decomposition", json=body)
        assert response.status_code == 200
        rows = response.json()
        assert {row["filter_value"]: row["value"] for row in rows} == {
            "a": 1,
            None: 6,
            "Unknown": 3,
        }
        for selected, expected in [
            (None, {"missing_a": 2, "missing_b": 4}),
            ("Unknown", {"literal": 3}),
        ]:
            bucket = next(row for row in rows if row["filter_value"] == selected)
            response = client.post(
                "/eda/192/decomposition",
                json={
                    **body,
                    "split_col": "detail",
                    "filters": [
                        {"column": split_col, "operator": "==", "value": bucket["filter_value"]}
                    ],
                },
            )
            assert response.status_code == 200
            assert {row["name"]: row["value"] for row in response.json()} == expected
