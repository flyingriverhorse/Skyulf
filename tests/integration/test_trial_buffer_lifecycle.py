"""OC-151: finished jobs release live chart buffers without losing saved metrics."""

from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, TrainingJob
from backend.ml_pipeline import tasks
from backend.ml_pipeline._execution.jobs import JobManager
from backend.ml_pipeline._execution.schemas import NodeExecutionResult, PipelineExecutionResult
from backend.ml_pipeline._services import pipeline_execution_service as execution
from backend.realtime import trial_buffer


@pytest.fixture
def job_database(tmp_path, monkeypatch):
    """Keep persistence real while isolating model fitting and artifact storage."""
    database_path = tmp_path / "jobs.db"
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(tasks, "get_db_session", sessions)
    monkeypatch.setattr(execution, "publish_job_event", lambda event: None)
    monkeypatch.setattr(
        execution, "_create_artifact_store", lambda *args: (MagicMock(), "memory://job", "test")
    )
    yield sessions, database_path
    engine.dispose()


@pytest.fixture
def chart_jobs():
    """Clean only this test's buffers, including deliberately retained active jobs."""
    job_ids = ["oc151-first", "oc151-second", "oc151-active"]
    yield job_ids
    for job_id in job_ids:
        trial_buffer.clear_trials(job_id)
        trial_buffer.clear_iterations(job_id)


def _add_job(session, job_id, run_mode, status="queued"):
    """Create a complete job row for real persistence and cancellation paths."""
    session.add(
        TrainingJob(
            id=job_id,
            pipeline_id="pipeline",
            node_id="model",
            dataset_source_id="dataset",
            model_type="logistic_regression",
            run_mode=run_mode,
            status=status,
            graph={"nodes": []},
        )
    )


def _record_charts(job_id):
    """Populate both independent live chart series for a job."""
    trial_buffer.record_trial(job_id, 1, 1, 0.8, "accuracy")
    trial_buffer.record_iteration(job_id, 1, 1, 0.4, "logloss", "minimize")


@pytest.mark.parametrize("run_mode", ["fixed", "tuned"])
@pytest.mark.parametrize("entry_point", ["single", "batch"])
@pytest.mark.parametrize("outcome", ["success", "failed", "exception", "cancelled"])
def test_execution_releases_finished_job_charts(
    job_database, chart_jobs, monkeypatch, run_mode, entry_point, outcome
):
    """Every execution exit frees only its job buffers and keeps committed chart history."""
    sessions, _ = job_database
    job_ids = chart_jobs[:2] if entry_point == "batch" else chart_jobs[:1]
    active_job_id = chart_jobs[-1]
    _record_charts(active_job_id)
    with sessions() as session:
        for job_id in job_ids:
            _add_job(session, job_id, run_mode)
        session.commit()

    committed_with_backfill = []

    def observe_commit(session):
        """Successful results must reach durable storage before backfill is dropped."""
        committed_with_backfill.extend(
            (
                job.id,
                len(trial_buffer.get_trials(job.id)),
                len(trial_buffer.get_iterations(job.id)),
            )
            for job in session.identity_map.values()
            if isinstance(job, TrainingJob) and job.status == "completed"
        )

    event.listen(sessions, "after_commit", observe_commit)

    def run_engine(config, *, job_id, dataset_name):
        """Emit real chart points, then emulate one model execution outcome."""
        _record_charts(job_id)
        if outcome == "exception":
            raise RuntimeError("model fit failed")
        if outcome == "cancelled":
            with sessions() as cancellation_session:
                job = cancellation_session.get(TrainingJob, job_id)
                assert job is not None
                job.status = "cancelled"
                cancellation_session.commit()
            # A fitting thread may emit more points after the API cancels it.
            _record_charts(job_id)
        return PipelineExecutionResult(
            pipeline_id=config.pipeline_id,
            status="failed" if outcome == "failed" else "success",
            node_results={
                "model": NodeExecutionResult(
                    node_id="model",
                    status="failed" if outcome == "failed" else "success",
                    error="model fit failed" if outcome == "failed" else None,
                    metrics={
                        "trials": trial_buffer.get_trials(job_id),
                        "iterations": trial_buffer.get_iterations(job_id),
                    },
                )
            },
        )

    model_engine = MagicMock()
    model_engine.run.side_effect = run_engine
    monkeypatch.setattr(execution, "PipelineEngine", lambda *args, **kwargs: model_engine)
    config = {"pipeline_id": "pipeline", "nodes": []}
    if entry_point == "batch":
        tasks.run_pipeline_batch_task([(job_id, config) for job_id in job_ids])
    else:
        tasks.run_pipeline_task(job_ids[0], config)

    with sessions() as session:
        for job_id in job_ids:
            saved = session.get(TrainingJob, job_id)
            assert saved is not None
            expected_status = {"success": "completed", "exception": "failed"}.get(outcome, outcome)
            assert saved.status == expected_status
            if outcome == "success":
                assert saved.metrics["trials"][0]["score"] == 0.8
                assert saved.metrics["iterations"][0]["score"] == 0.4
                assert (job_id, 1, 1) in committed_with_backfill
            assert trial_buffer.get_trials(job_id) == []
            assert trial_buffer.get_iterations(job_id) == []
    assert len(trial_buffer.get_trials(active_job_id)) == 1
    assert len(trial_buffer.get_iterations(active_job_id)) == 1


@pytest.mark.parametrize("run_mode", ["fixed", "tuned"])
@pytest.mark.parametrize("commit_fails", [False, True])
async def test_cancellation_clears_charts_only_after_commit(
    job_database, chart_jobs, run_mode, commit_fails, monkeypatch
):
    """Cancellation frees local history only when its database update succeeds."""
    sessions, database_path = job_database
    job_id = chart_jobs[0]
    with sessions() as session:
        _add_job(session, job_id, run_mode, status="running")
        session.commit()
    _record_charts(job_id)
    async_engine = create_async_engine(f"sqlite+aiosqlite:///{database_path.as_posix()}")
    try:
        async with async_sessionmaker(async_engine)() as session:
            if commit_fails:

                async def fail_commit():
                    """Leave the actual database unchanged on a failed cancellation."""
                    raise RuntimeError("commit failed")

                monkeypatch.setattr(session, "commit", fail_commit)
                with pytest.raises(RuntimeError, match="commit failed"):
                    await JobManager.cancel_job(session, job_id)
            else:
                assert await JobManager.cancel_job(session, job_id)
    finally:
        await async_engine.dispose()

    with sessions() as session:
        saved = session.get(TrainingJob, job_id)
        assert saved is not None
        assert saved.status == ("running" if commit_fails else "cancelled")
    expected_count = 1 if commit_fails else 0
    assert len(trial_buffer.get_trials(job_id)) == expected_count
    assert len(trial_buffer.get_iterations(job_id)) == expected_count
