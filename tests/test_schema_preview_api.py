"""Integration test for `POST /api/pipeline/schema-preview` (C7 Phase C)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://localhost") as c:
        yield c


def test_schema_preview_unseeded_returns_none_chain(client):
    """Unseeded loaders -> all predictions None, no broken refs surfaced."""
    payload = {
        "pipeline_id": "p_test",
        "nodes": [
            {"node_id": "loader", "step_type": "data_loader", "params": {}, "inputs": []},
            {
                "node_id": "drop",
                "step_type": "DropMissingColumns",
                "params": {"columns": ["ghost"]},
                "inputs": ["loader"],
            },
        ],
        "metadata": {},
    }

    response = client.post("/api/pipeline/schema-preview", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["pipeline_id"] == "p_test"
    assert body["predicted_schemas"]["loader"] is None
    assert body["predicted_schemas"]["drop"] is None
    # Without an upstream schema, the validator can't flag anything.
    assert body["broken_references"] == []
