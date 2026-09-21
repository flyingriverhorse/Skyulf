"""Preserve drift deployment context when concurrent promotions leave active rows."""

import asyncio
import os
from datetime import datetime, timedelta
from uuid import uuid4

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.sql.dml import Update

from backend.config import get_settings
from backend.database.models import Base, Deployment, TrainingJob
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.deployment.service import DeploymentService
from backend.monitoring.router import _find_deployment_context


def _job(artifact_uri="unused"):
    """Create a versioned candidate for independent deployment-context assertions."""
    return TrainingJob(
        id="context-job",
        pipeline_id="pipeline",
        node_id="model",
        dataset_source_id="source",
        status="completed",
        run_mode="fixed",
        model_type="linear_regression",
        graph={},
        version=7,
        artifact_uri=artifact_uri,
    )


@pytest.mark.asyncio
async def test_concurrent_postgres_promotions_keep_drift_context(tmp_path, monkeypatch):
    """Real overlapping deploy transactions must not erase drift links or model versions."""
    url = os.environ.get("SKYULF_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set SKYULF_TEST_POSTGRES_URL to an isolated PostgreSQL test database")
    schema = f"qw114_{uuid4().hex}"
    admin = create_async_engine(url)
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        root = tmp_path / "models"
        monkeypatch.setattr(get_settings(), "TRAINING_ARTIFACT_DIR", str(root))
        monkeypatch.setattr(get_settings(), "TESTING", False)
        store = LocalArtifactStore(str(root / "pipeline"))
        model = LinearRegression().fit(pd.DataFrame({"value": [1.0, 2.0]}), [2.0, 4.0])
        store.save("context-job", model)
        async with AsyncSession(engine) as session:
            session.add(_job(store.get_artifact_uri("context-job")))
            await session.commit()

        barrier = asyncio.Barrier(2)

        class ConcurrentSession(AsyncSession):
            """Pause after real deactivation statements to force overlapping transactions."""

            async def execute(self, statement, *args, **kwargs):
                """Retain real SQL execution while controlling the scheduling boundary."""
                result = await super().execute(statement, *args, **kwargs)
                if isinstance(statement, Update) and statement.table.name == "deployments":
                    await asyncio.wait_for(barrier.wait(), timeout=10)
                return result

        async def deploy():
            """Use the normal production promotion path with a distinct DB session."""
            async with ConcurrentSession(engine, expire_on_commit=False) as session:
                return await DeploymentService.deploy_model(session, "context-job")

        deployments = await asyncio.wait_for(asyncio.gather(deploy(), deploy()), timeout=25)
        async with AsyncSession(engine) as reader:
            active = (await reader.scalars(select(Deployment).where(Deployment.is_active))).all()
            assert len(active) == 2
            newest = max(deployments, key=lambda row: (row.created_at, row.id))
            assert await _find_deployment_context(reader, "context-job") == (newest.id, "v7")
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["missing", "undeployed", "single", "multiple", "tied"])
async def test_deployment_context_selection(tmp_path, case):
    """Newest active deployment selection preserves the independently stored model version."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'context.db'}")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with AsyncSession(engine) as session:
            if case != "missing":
                session.add(_job())
                await session.flush()
            expected = None
            if case in {"single", "multiple", "tied"}:
                now = datetime(2026, 1, 1)
                rows = [
                    Deployment(
                        id=1,
                        job_id="context-job",
                        model_type="linear_regression",
                        artifact_uri="unused",
                        is_active=True,
                        created_at=now,
                    )
                ]
                if case != "single":
                    rows.append(
                        Deployment(
                            id=2,
                            job_id="context-job",
                            model_type="linear_regression",
                            artifact_uri="unused",
                            is_active=True,
                            created_at=now if case == "tied" else now + timedelta(days=1),
                        )
                    )
                expected = rows[-1].id
                rows.append(
                    Deployment(
                        id=3,
                        job_id="context-job",
                        model_type="linear_regression",
                        artifact_uri="unused",
                        is_active=False,
                        created_at=now + timedelta(days=2),
                    )
                )
                session.add_all(rows)
            await session.commit()
            assert await _find_deployment_context(session, "context-job") == (
                expected,
                None if case == "missing" else "v7",
            )
    finally:
        await engine.dispose()
