"""L7 — Pipeline versioning API tests.

Smoke tests covering the new /pipeline/versions/{dataset_id} routes
and the auto-snapshot side-effect on /pipeline/save.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://localhost") as c:
        yield c


# Use a dataset id unlikely to clash with other tests.
DATASET = "test_l7_versions_dataset"


def _save(client: TestClient, name: str, graph: dict, note: str | None = None) -> None:
    payload: dict = {"name": name, "graph": graph}
    if note is not None:
        payload["note"] = note
    response = client.post(f"/api/pipeline/save/{DATASET}", json=payload)
    assert response.status_code == 200, response.text


def test_save_creates_auto_version(client: TestClient) -> None:
    """Every successful Save should append a version row."""
    # Clear any pre-existing versions from prior test runs.
    listing = client.get(f"/api/pipeline/versions/{DATASET}")
    if listing.status_code == 200:
        for v in listing.json():
            client.delete(f"/api/pipeline/versions/{DATASET}/{v['id']}")

    graph = {"nodes": [{"id": "n1"}, {"id": "n2"}], "edges": [{"source": "n1", "target": "n2"}]}
    _save(client, "v1", graph, note="first save")

    response = client.get(f"/api/pipeline/versions/{DATASET}")
    assert response.status_code == 200
    versions = response.json()
    assert len(versions) >= 1
    latest = versions[0]
    assert latest["name"] == "v1"
    assert latest["kind"] == "manual"
    assert latest["node_count"] == 2
    assert latest["edge_count"] == 1
    assert latest["version_int"] >= 1


def test_versions_increment_per_save(client: TestClient) -> None:
    """version_int increments monotonically per dataset."""
    before = client.get(f"/api/pipeline/versions/{DATASET}").json()
    last_int = before[0]["version_int"] if before else 0

    _save(client, "v2", {"nodes": [{"id": "x"}], "edges": []})
    after = client.get(f"/api/pipeline/versions/{DATASET}").json()
    # Newest first (after pinned), so [0] should be the new one
    # unless something is pinned. Find by name to be robust.
    matching = [v for v in after if v["name"] == "v2"]
    assert matching, "expected a v2 row"
    assert matching[0]["version_int"] == last_int + 1


def test_pin_rename_and_delete(client: TestClient) -> None:
    """PATCH toggles pin/rename, DELETE removes a row."""
    _save(client, "to_modify", {"nodes": [], "edges": []})
    versions = client.get(f"/api/pipeline/versions/{DATASET}").json()
    target = next(v for v in versions if v["name"] == "to_modify")

    # Pin + rename + note
    patch = client.patch(
        f"/api/pipeline/versions/{DATASET}/{target['id']}",
        json={"pinned": True, "name": "renamed", "note": "important"},
    )
    assert patch.status_code == 200
    body = patch.json()
    assert body["pinned"] is True
    assert body["name"] == "renamed"
    assert body["note"] == "important"

    # Pinned should float to the top
    listing = client.get(f"/api/pipeline/versions/{DATASET}").json()
    assert listing[0]["id"] == target["id"]

    # Delete
    delete = client.delete(f"/api/pipeline/versions/{DATASET}/{target['id']}")
    assert delete.status_code == 200
    after = client.get(f"/api/pipeline/versions/{DATASET}").json()
    assert all(v["id"] != target["id"] for v in after)


def test_delete_unknown_returns_404(client: TestClient) -> None:
    response = client.delete(f"/api/pipeline/versions/{DATASET}/999999")
    assert response.status_code == 404
