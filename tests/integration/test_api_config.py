"""Integration tests for the GET /api/config upload-limits endpoint."""

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://localhost") as c:
        yield c


class TestApiConfig:
    def test_returns_200_and_json(self, client: TestClient):
        response = client.get("/api/config")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_matches_settings(self, client: TestClient):
        settings = Settings()
        payload = client.get("/api/config").json()
        assert payload["max_upload_size_bytes"] == settings.MAX_UPLOAD_SIZE
        assert payload["allowed_extensions"] == list(settings.ALLOWED_EXTENSIONS)

    def test_payload_shape(self, client: TestClient):
        payload = client.get("/api/config").json()
        assert set(payload.keys()) == {"max_upload_size_bytes", "allowed_extensions"}
        assert isinstance(payload["max_upload_size_bytes"], int)
        assert all(ext.startswith(".") for ext in payload["allowed_extensions"])
