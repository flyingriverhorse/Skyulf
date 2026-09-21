"""Prediction must leave the async server responsive during synchronous model work."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from backend.database.models import Base, Deployment
from backend.ml_pipeline.artifacts.factory import ArtifactFactory
from backend.ml_pipeline.deployment.service import DeploymentService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bundled", "slow_stage"),
    [(False, "load"), (False, "predict"), (True, "load"), (True, "transform"), (True, "predict")],
)
async def test_prediction_allows_concurrent_coroutine(monkeypatch, bundled, slow_stage):
    """A waiting model stage must allow another coroutine to release it; DB stays on loop."""
    entered = threading.Event()
    release = threading.Event()
    observations = []
    loop_thread = threading.get_ident()
    query_threads = []

    def pause(stage):
        """Wait for the concurrent coroutine, recording whether it ran in time."""
        if stage == slow_stage:
            entered.set()
            observations.append(release.wait(timeout=1))

    class Model:
        """Return a deterministic prediction after a controlled blocking stage."""

        def predict(self, frame):
            """Double inputs so model output is independently checkable."""
            pause("predict")
            return (frame["value"] * 2).tolist()

    class Engineer:
        """Preserve input rows while allowing a controlled slow transform."""

        def transform(self, frame):
            """Exercise the production preprocessing path."""
            pause("transform")
            return frame

    artifact = {"model": Model(), "feature_engineer": Engineer()} if bundled else Model()

    def load(key):
        """Delay real artifact loader's storage boundary without replacing the service."""
        pause("load")
        return artifact

    monkeypatch.setattr(
        ArtifactFactory, "get_artifact_store", lambda uri: SimpleNamespace(load=load)
    )
    database = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with database.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with AsyncSession(database, expire_on_commit=False) as session:
            session.add(
                Deployment(
                    job_id="model-job",
                    model_type="test",
                    artifact_uri="pipeline/model",
                    is_active=True,
                )
            )
            await session.commit()

            def record_query(*args):
                """Check every prediction database query stays on the event loop thread."""
                query_threads.append(threading.get_ident())

            event.listen(database.sync_engine, "before_cursor_execute", record_query)

            async def lightweight_coroutine():
                """Release model work only after observing its synchronous stage start."""
                while not entered.is_set():
                    await asyncio.sleep(0.001)
                release.set()

            prediction, _ = await asyncio.wait_for(
                asyncio.gather(
                    DeploymentService.predict(session, [{"value": 3.0}]),
                    lightweight_coroutine(),
                ),
                timeout=5,
            )
    finally:
        release.set()
        await database.dispose()

    assert prediction == ([6.0], None)
    assert query_threads and set(query_threads) == {loop_thread}
    assert len(query_threads) == (2 if bundled else 1)
    assert observations == [True], "Synchronous model work blocked the concurrent coroutine"
