"""Pin per-client drift throttling without letting one test exhaust the next test's budget."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

from backend.monitoring.router import calculate_drift


async def _missing_reference_request(client):
    """Reach the real decorated route without requiring a trained model or uploaded data."""
    return await calculate_drift(
        request=Request(
            {
                "type": "http",
                "path": "/monitoring/drift/calculate",
                "client": (client, 1234),
            }
        ),
        job_id="rate-limit-test",
        dataset_name="missing-reference",
        db=AsyncMock(),
    )


@pytest.mark.parametrize("repetition", [1, 2])
async def test_drift_limit_stays_active_with_a_fresh_budget_per_test(repetition):
    """Each case gets 20 requests; a 21st is blocked while another address keeps its budget."""
    with (
        patch("backend.monitoring.router.ArtifactFactory.get_discovery"),
        patch("backend.monitoring.router._find_reference_key", return_value=None),
        patch(
            "backend.monitoring.router._find_deployment_context",
            new=AsyncMock(return_value=(None, None)),
        ),
        patch("backend.monitoring.router._save_drift_alert", new=AsyncMock()) as saved,
    ):
        for _ in range(20):
            with pytest.raises(HTTPException) as missing:
                await _missing_reference_request("127.0.0.1")
            assert missing.value.status_code == 404, f"case {repetition} lost its request budget"

        with pytest.raises(RateLimitExceeded):
            await _missing_reference_request("127.0.0.1")
        assert saved.await_count == 20

        with pytest.raises(HTTPException) as other_client:
            await _missing_reference_request("192.0.2.17")
        assert other_client.value.status_code == 404
        assert saved.await_count == 21
