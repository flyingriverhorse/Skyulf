from fastapi.testclient import TestClient

from main import app


def test_health_endpoint_returns_healthy():
    # TestClient will run lifespan to initialize DB (SQLite by default)
    with TestClient(app) as client:
        # Provide an explicit Host header that matches the allowed hosts in
        # application settings to avoid TrustedHostMiddleware rejecting the
        # TestClient default host in some environments.
        res = client.get("/health", headers={"host": "localhost"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data
