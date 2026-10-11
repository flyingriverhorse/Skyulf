"""Public API documentation identifies the Kayra product and its Skyulf company."""

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.config import Settings


@pytest.fixture
def client(monkeypatch):
    """Exercise the real app factory with documentation enabled and default branding."""
    for name in ("APP_NAME", "APP_SUMMARY", "APP_DESCRIPTION", "API_CONTACT_NAME"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None, API_DOCS_ENABLED=True, ALLOWED_HOSTS=["testserver"])
    monkeypatch.setattr(main, "settings", settings)
    return TestClient(main.create_app())


def test_openapi_identifies_kayra_and_preserves_skyulf_support(client):
    """API consumers must receive the product name while retaining the company contact."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    info = response.json()["info"]
    assert info["title"] == "Kayra"
    assert info["summary"].startswith("Kayra ")
    assert "Kayra" in info["description"]
    assert "by Skyulf" in info["description"]
    assert info["contact"]["name"] == "Skyulf Support"


@pytest.mark.parametrize("path", ["/docs", "/redoc"])
def test_documentation_browser_title_uses_kayra(client, path):
    """Both supported documentation UIs must display the same product name as the API."""
    response = client.get(path)
    assert response.status_code == 200
    assert "<title>Kayra - " in response.text
