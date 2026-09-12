"""Unstable correlation matrices must retain VIF warnings in saved EDA reports."""

import json

import numpy as np
import polars as pl

from backend.eda.tasks import _run_eda_analyzer


def test_eda_report_serializes_near_collinearity_warnings() -> None:
    """API consumers must receive finite high VIF and warnings for almost duplicated features."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=100)
    frame = pl.DataFrame(
        {"a": a, "b": 2 * a + 1e-9 * rng.normal(size=100), "independent": rng.normal(size=100)}
    )

    payload = _run_eda_analyzer(frame, None).model_dump(mode="json")
    result = json.loads(json.dumps(payload, allow_nan=False))

    assert result["vif"]["a"] > 10.0
    assert result["vif"]["b"] > 10.0
    assert result["vif"]["independent"] < 5.0
    assert len([alert for alert in result["alerts"] if alert["type"] == "Multicollinearity"]) == 2
