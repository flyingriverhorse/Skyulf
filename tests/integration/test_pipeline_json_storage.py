"""JSON pipeline-storage backend (`PIPELINE_STORAGE_TYPE=json`).

The default backend is the database, so the on-disk branch of `/save` and
`/load` never executes anywhere else in the suite — yet it is the branch
CodeQL flags for path injection and the one `_pipeline_json_path` guards.
These tests drive it through the real HTTP surface so both the `aiofiles`
read/write and the rejection path are exercised.
"""

import json

import pytest
from fastapi.testclient import TestClient

from backend.config import get_settings
from backend.main import app

DATASET = "json_storage_dataset-01"
GRAPH = {"nodes": [{"id": "n1"}, {"id": "n2"}], "edges": [{"source": "n1", "target": "n2"}]}
PAYLOAD = {"name": "json-backed pipeline", "graph": GRAPH}


@pytest.fixture
def json_storage(tmp_path, monkeypatch):
    """Point the pipeline routes' JSON backend at an isolated `tmp_path`."""
    settings = get_settings()
    monkeypatch.setattr(settings, "PIPELINE_STORAGE_TYPE", "json", raising=False)
    monkeypatch.setattr(settings, "PIPELINE_STORAGE_PATH", str(tmp_path), raising=False)


@pytest.fixture
def client(json_storage):
    with TestClient(app, base_url="http://localhost") as c:
        yield c


def test_save_writes_json_file(client: TestClient, tmp_path) -> None:
    response = client.post(f"/api/pipeline/save/{DATASET}", json=PAYLOAD)

    assert response.status_code == 200, response.text
    assert response.json() == {"status": "success", "id": DATASET, "storage": "json"}

    written = tmp_path / f"{DATASET}.json"
    assert written.is_file()
    persisted = json.loads(written.read_text(encoding="utf-8"))
    assert persisted["name"] == PAYLOAD["name"]
    assert persisted["graph"] == GRAPH


def test_load_round_trips_the_saved_graph(client: TestClient) -> None:
    assert client.post(f"/api/pipeline/save/{DATASET}", json=PAYLOAD).status_code == 200

    response = client.get(f"/api/pipeline/load/{DATASET}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["name"] == PAYLOAD["name"]
    assert body["graph"] == GRAPH


def test_save_overwrites_an_existing_file(client: TestClient, tmp_path) -> None:
    client.post(f"/api/pipeline/save/{DATASET}", json=PAYLOAD)
    edited = {"name": "renamed", "graph": {"nodes": [{"id": "only"}], "edges": []}}

    response = client.post(f"/api/pipeline/save/{DATASET}", json=edited)

    assert response.status_code == 200, response.text
    assert sorted(f.name for f in tmp_path.iterdir()) == [f"{DATASET}.json"]
    assert client.get(f"/api/pipeline/load/{DATASET}").json()["name"] == "renamed"


def test_load_unknown_dataset_returns_null(client: TestClient) -> None:
    response = client.get("/api/pipeline/load/never-saved-dataset")

    assert response.status_code == 200, response.text
    assert response.json() is None


def test_save_rejects_a_dataset_id_outside_the_allowlist(client: TestClient, tmp_path) -> None:
    """Routed through HTTP, not just the helper: an id with a space fails
    `_SAFE_DATASET_ID_RE` and must 400 without touching the filesystem."""
    response = client.post("/api/pipeline/save/bad%20id", json=PAYLOAD)

    assert response.status_code == 400, response.text
    assert list(tmp_path.iterdir()) == []


def test_load_rejected_dataset_id_returns_null(client: TestClient) -> None:
    response = client.get("/api/pipeline/load/bad%20id")

    assert response.status_code == 200, response.text
    assert response.json() is None
