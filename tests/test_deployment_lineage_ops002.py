"""Tests for OPS-002: model-version/deployment lineage on the Deployment record.

Covers the fields Deployments/Registry need to render a traceable
model-to-deployment decision chain: the deployed version's dataset/version
identity, and the replacement chain between successive deployments.
"""

from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base
from backend.ml_pipeline.deployment.service import DeploymentService

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def async_session():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session_maker() as session:
        yield session

    await engine.dispose()


async def _insert_training_job(
    session, job_id: str, pipeline_id: str, dataset_id: str, version: int
):
    await session.execute(
        text(
            """
        INSERT INTO training_jobs (id, pipeline_id, node_id, dataset_source_id, user_id, status, run_mode, version, model_type, graph, artifact_uri, error_message, progress, current_step, started_at, finished_at, created_at, updated_at)
        VALUES (:id, :pipeline_id, :node_id, :ds_id, :user_id, :status, :run_mode, :version, :model_type, :graph, :artifact_uri, :error_message, :progress, :current_step, :started_at, :finished_at, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        ),
        {
            "id": job_id,
            "pipeline_id": pipeline_id,
            "node_id": f"node_{job_id}",
            "ds_id": dataset_id,
            "user_id": None,
            "status": "completed",
            "run_mode": "fixed",
            "version": version,
            "model_type": "random_forest",
            "graph": "{}",
            "artifact_uri": job_id,
            "error_message": None,
            "progress": 0,
            "current_step": None,
            "started_at": None,
            "finished_at": None,
        },
    )
    await session.commit()


@pytest.mark.asyncio
async def test_deploy_model_captures_previous_deployment(async_session, tmp_path):
    """A second deploy must record which deployment it replaced."""
    await _insert_training_job(async_session, "job_v1", "pipe_lineage", "ds_lineage", 1)
    await _insert_training_job(async_session, "job_v2", "pipe_lineage", "ds_lineage", 2)

    with patch("os.getcwd", return_value=str(tmp_path)):
        first = await DeploymentService.deploy_model(async_session, "job_v1")
        assert first.previous_deployment_id is None
        assert first.is_active is True

        second = await DeploymentService.deploy_model(async_session, "job_v2")
        assert second.previous_deployment_id == first.id
        assert second.is_active is True

    await async_session.refresh(first)
    assert first.is_active is False


@pytest.mark.asyncio
async def test_get_deployment_details_includes_dataset_and_version(async_session, tmp_path):
    """Deployment details must expose the dataset/version identity of the deployed job."""
    await _insert_training_job(async_session, "job_lineage_a", "pipe_a", "dataset_a", 3)

    with patch("os.getcwd", return_value=str(tmp_path)):
        deployment = await DeploymentService.deploy_model(async_session, "job_lineage_a")
        details = await DeploymentService.get_deployment_details(async_session, deployment)

    assert details["dataset_id"] == "dataset_a"
    assert details["version"] == 3
    assert details["previous_deployment_id"] is None


@pytest.mark.asyncio
async def test_list_deployments_history_includes_lineage_for_each_entry(async_session, tmp_path):
    """History listing must enrich every row, not only the active deployment."""
    await _insert_training_job(async_session, "job_hist_1", "pipe_hist", "dataset_hist", 1)
    await _insert_training_job(async_session, "job_hist_2", "pipe_hist", "dataset_hist", 2)

    with patch("os.getcwd", return_value=str(tmp_path)):
        first = await DeploymentService.deploy_model(async_session, "job_hist_1")
        await DeploymentService.deploy_model(async_session, "job_hist_2")

        history = await DeploymentService.list_deployments(async_session, limit=10, skip=0)
        enriched = [
            await DeploymentService.get_deployment_details(async_session, d) for d in history
        ]

    by_job = {entry["job_id"]: entry for entry in enriched}
    assert by_job["job_hist_1"]["version"] == 1
    assert by_job["job_hist_2"]["version"] == 2
    assert by_job["job_hist_2"]["previous_deployment_id"] == first.id
