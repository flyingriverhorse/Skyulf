from fastapi.testclient import TestClient

from main import app


def test_health_endpoint_returns_healthy():
    # TestClient will run lifespan to initialize DB (SQLite by default)
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data
