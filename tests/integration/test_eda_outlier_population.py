"""Outlier population metadata survives the backend task and saved-report HTTP boundary."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.database.models import EDAReport
from backend.dependencies import get_db
from backend.eda.router import router
from backend.eda.tasks import _run_eda_analyzer


@pytest.mark.parametrize("legacy", [False, True])
def test_saved_outlier_report_preserves_sample_population(legacy: bool) -> None:
    """Both report endpoints must retain measured sample counts without upgrading legacy metadata."""
    frame = pl.DataFrame(
        {"row_id": np.arange(200007), "measurement": np.random.default_rng(302).normal(size=200007)}
    )
    config = {
        "filters": [{"column": "row_id", "operator": ">=", "value": 7}],
        "exclude_cols": ["row_id"],
    }
    data = _run_eda_analyzer(frame, config).model_dump(mode="json")
    if legacy:
        data["outliers"].pop("analyzed_rows")
        data["outliers"].pop("total_rows")
    report = EDAReport(
        id=302, data_source_id=302, status="COMPLETED", config=config, profile_data=data
    )
    result = MagicMock()
    result.scalar_one_or_none.return_value = report
    session = SimpleNamespace(
        get=AsyncMock(return_value=report), execute=AsyncMock(return_value=result)
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = lambda: session

    with TestClient(app) as client:
        for path in ("/eda/302/latest", "/eda/reports/302"):
            response = client.get(path)
            assert response.status_code == 200
            payload = response.json()["profile_data"]
            assert payload["row_count"] == 200000
            assert payload["outliers"] == data["outliers"]
            if legacy:
                assert "analyzed_rows" not in payload["outliers"]
                assert "total_rows" not in payload["outliers"]
            else:
                assert payload["outliers"]["analyzed_rows"] == 50000
                assert payload["outliers"]["total_rows"] == 200000
                assert payload["outliers"]["outlier_percentage"] == pytest.approx(
                    100 * payload["outliers"]["total_outliers"] / 50000
                )
