"""Tests for the node-inspector endpoints (`GET /monitoring/jobs/{job_id}/nodes/{node_id}`
and `GET /monitoring/pipeline-runs/{pipeline_id}/nodes/{node_id}`).

A node investigated from an operational surface (Error Log, Slow Nodes, Jobs)
must be inspectable purely from the job's stored `graph`/`metrics` columns —
never from live canvas state — so these tests exercise the three honest
outcomes: node found, node absent from the graph, and job/pipeline-run gone.
"""

from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, PipelineRunLog, TrainingJob
from backend.monitoring.router import (
    _build_node_inspector_response,
    _humanize_step_type,
    _is_synthetic_pipeline,
    get_job_node,
    get_pipeline_run_node,
)

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def async_session():
    """A fresh in-memory SQLite `AsyncSession` with the full ORM schema created."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session_maker() as session:
        yield session

    await engine.dispose()


_SAMPLE_GRAPH = {
    "pipeline_id": "preview_abc123",
    "nodes": [
        {
            "node_id": "loader-1",
            "step_type": "data_loader",
            "params": {"dataset_id": "ds-1"},
            "inputs": [],
        },
        {
            "node_id": "impute-1",
            "step_type": "simple_imputer",
            "params": {"strategy": "mean"},
            "inputs": ["loader-1"],
        },
        {
            "node_id": "train-1",
            "step_type": "training",
            "params": {"algorithm": "RandomForest", "target_column": "y"},
            "inputs": ["impute-1"],
        },
    ],
    "metadata": {},
}


async def _seed_job(
    session: AsyncSession,
    *,
    job_id: str = "job-1",
    pipeline_id: str = "preview_abc123",
    node_id: str = "train-1",
    graph: dict | None = None,
    metrics: dict | None = None,
    job_metadata: dict | None = None,
) -> TrainingJob:
    row = TrainingJob(
        id=job_id,
        pipeline_id=pipeline_id,
        node_id=node_id,
        dataset_source_id="ds-1",
        status="completed",
        model_type="RandomForest",
        run_mode="fixed",
        version=1,
        graph=graph if graph is not None else dict(_SAMPLE_GRAPH),
        metrics=metrics,
        job_metadata=job_metadata,
        started_at=datetime(2026, 8, 7, 10, 0),
        finished_at=datetime(2026, 8, 7, 10, 1),
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


class TestHumanizeStepType:
    def test_snake_case_becomes_title_case(self) -> None:
        assert _humanize_step_type("train_test_split") == "Train Test Split"

    def test_falls_back_to_original_when_blank(self) -> None:
        assert _humanize_step_type("") == ""


class TestIsSyntheticPipeline:
    def test_preview_id_is_synthetic(self) -> None:
        assert _is_synthetic_pipeline("preview_abc123") is True

    def test_branch_of_preview_is_synthetic(self) -> None:
        assert _is_synthetic_pipeline("preview_abc123__branch_0") is True

    def test_saved_pipeline_id_is_not_synthetic(self) -> None:
        assert _is_synthetic_pipeline("dataset-42-v3") is False


class TestBuildNodeInspectorResponse:
    """Direct tests of the shared builder — the meat of the graph-walk logic."""

    @pytest.mark.asyncio
    async def test_found_node_reports_neighbours_and_provenance(
        self, async_session: AsyncSession
    ) -> None:
        job = await _seed_job(
            async_session,
            metrics={
                "node_timings": [{"node_id": "train-1", "execution_time": 4.5, "status": "success"}]
            },
        )
        response = await _build_node_inspector_response(async_session, job, "train-1")

        assert response.node_found is True
        assert response.node is not None
        assert response.node.label == "Training"
        assert response.node.execution_seconds == 4.5
        assert response.node.execution_status == "success"
        assert [n.node_id for n in response.node.upstream] == ["impute-1"]
        assert response.node.downstream == []
        assert response.is_synthetic_pipeline is True
        assert response.can_open_in_canvas is False

    @pytest.mark.asyncio
    async def test_upstream_node_reports_downstream_neighbour(
        self, async_session: AsyncSession
    ) -> None:
        job = await _seed_job(async_session)
        response = await _build_node_inspector_response(async_session, job, "impute-1")

        assert response.node_found is True
        assert response.node is not None
        assert [n.node_id for n in response.node.upstream] == ["loader-1"]
        assert [n.node_id for n in response.node.downstream] == ["train-1"]

    @pytest.mark.asyncio
    async def test_node_absent_from_graph_reports_plainly(
        self, async_session: AsyncSession
    ) -> None:
        job = await _seed_job(async_session)
        response = await _build_node_inspector_response(async_session, job, "does-not-exist")

        assert response.node_found is False
        assert response.node is None
        # Job-level context still comes back so the caller isn't left with nothing.
        assert response.job_id == "job-1"
        assert response.pipeline_id == "preview_abc123"

    @pytest.mark.asyncio
    async def test_saved_pipeline_allows_canvas_link(self, async_session: AsyncSession) -> None:
        job = await _seed_job(async_session, pipeline_id="dataset-42-v3")
        response = await _build_node_inspector_response(async_session, job, "train-1")

        assert response.is_synthetic_pipeline is False
        assert response.can_open_in_canvas is True

    @pytest.mark.asyncio
    async def test_recent_logs_scoped_to_pipeline_and_node(
        self, async_session: AsyncSession
    ) -> None:
        job = await _seed_job(async_session)
        async_session.add(
            PipelineRunLog(
                pipeline_id="preview_abc123",
                node_id="train-1",
                node_type="training",
                level="error",
                message="boom",
                run_at=datetime(2026, 8, 7, 10, 0, 30),
            )
        )
        async_session.add(
            PipelineRunLog(
                pipeline_id="preview_abc123",
                node_id="impute-1",
                node_type="simple_imputer",
                level="warning",
                message="unrelated node",
                run_at=datetime(2026, 8, 7, 10, 0, 20),
            )
        )
        await async_session.commit()

        response = await _build_node_inspector_response(async_session, job, "train-1")

        assert len(response.recent_logs) == 1
        assert response.recent_logs[0].message == "boom"

    @pytest.mark.asyncio
    async def test_branch_index_surfaced_from_job_metadata(
        self, async_session: AsyncSession
    ) -> None:
        job = await _seed_job(async_session, job_metadata={"branch_index": 2})
        response = await _build_node_inspector_response(async_session, job, "train-1")

        assert response.branch_index == 2


class TestGetJobNode:
    @pytest.mark.asyncio
    async def test_returns_full_detail_when_node_found(self, async_session: AsyncSession) -> None:
        await _seed_job(async_session)
        response = await get_job_node("job-1", "train-1", db=async_session)
        assert response.node_found is True
        assert response.node is not None
        assert response.node.node_id == "train-1"

    @pytest.mark.asyncio
    async def test_node_absent_from_graph_still_returns_job_context(
        self, async_session: AsyncSession
    ) -> None:
        await _seed_job(async_session)
        response = await get_job_node("job-1", "ghost-node", db=async_session)
        assert response.node_found is False
        assert response.job_id == "job-1"

    @pytest.mark.asyncio
    async def test_missing_job_raises_404(self, async_session: AsyncSession) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await get_job_node("does-not-exist", "train-1", db=async_session)
        assert exc_info.value.status_code == 404


class TestGetPipelineRunNode:
    @pytest.mark.asyncio
    async def test_resolves_latest_job_for_pipeline_id(self, async_session: AsyncSession) -> None:
        await _seed_job(async_session, job_id="job-1", pipeline_id="preview_xyz")
        response = await get_pipeline_run_node("preview_xyz", "train-1", db=async_session)
        assert response.node_found is True
        assert response.job_id == "job-1"

    @pytest.mark.asyncio
    async def test_missing_pipeline_run_raises_404(self, async_session: AsyncSession) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await get_pipeline_run_node("never-ran", "train-1", db=async_session)
        assert exc_info.value.status_code == 404
