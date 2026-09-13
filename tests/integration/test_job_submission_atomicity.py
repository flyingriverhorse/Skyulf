"""Submission reservations stay atomic across processes and failed requests."""

import asyncio
import importlib
import multiprocessing
import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy import event, insert, select, text
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from backend.database.engine import get_async_session
from backend.database.models import Base, ModelVersionCounter, TrainingJob
from backend.ml_pipeline._execution.jobs import JobManager
from backend.ml_pipeline._execution.schemas import PipelineConfig
from backend.ml_pipeline._execution.submission import _advisory_key
from backend.ml_pipeline.model_registry.service import ModelRegistryService

submit = importlib.import_module("backend.ml_pipeline._internal._routers.run_pipeline")


@pytest.fixture(autouse=True)
def isolated_lock_registry(monkeypatch):
    """One failure's leaked entries must not hide another regression's cause."""
    monkeypatch.setattr(submit, "_submit_locks", {})
    monkeypatch.setattr(submit, "_submit_lock_users", {}, raising=False)


async def _submit(session, *, node="node", branch=0, job_type="training", graph=None):
    """Exercise the production reservation shared by run and retry requests."""
    return await submit._submit_or_dedupe_branch_job(
        session,
        "dataset",
        node,
        branch,
        PipelineConfig("pipeline", []),
        job_type,
        "random_forest",
        graph or {},
    )


@pytest_asyncio.fixture
async def database(tmp_path):
    """Use separate file connections and the application's SQLite autocommit mode."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}",
        connect_args={"isolation_level": None, "timeout": 15},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


def _submission_process(
    url, index, first_checked, second_attempted, result_queue, scenario, advisory_key
):
    """Force a competing submit between the first absence check and creation."""

    async def run():
        """Use a separate interpreter, connection, and actual job manager."""
        engine = create_async_engine(url, connect_args={"isolation_level": None, "timeout": 15})
        original_find = JobManager.find_active_job
        original_seed = ModelRegistryService._compute_seed_version
        seed_calls = 0

        async def conflicting_seed(session, dataset_id, model_type):
            """Create a real primary-key conflict once to exercise allocator rollback."""
            nonlocal seed_calls
            seed_calls += 1
            if seed_calls == 1:
                await session.execute(
                    insert(ModelVersionCounter).values(
                        dataset_source_id=dataset_id, model_type=model_type, current_version=0
                    )
                )
                return 1
            return await original_seed(session, dataset_id, model_type)

        @event.listens_for(engine.sync_engine, "before_cursor_execute")
        def observe_reservation(conn, cursor, statement, parameters, context, executemany):
            """A blocked second write reservation lets the first process finish."""
            if index == 1 and statement == "BEGIN IMMEDIATE":
                second_attempted.set()

        async def controlled_find(*args, **kwargs):
            """Without a reservation both processes deterministically read absence."""
            found = await original_find(*args, **kwargs)
            if index == 0:
                first_checked.set()
                if not await asyncio.to_thread(second_attempted.wait, 20):
                    raise AssertionError("Second process never attempted its submission")
            else:
                second_attempted.set()
            return found

        JobManager.find_active_job = staticmethod(controlled_find)
        if index == 0 and scenario == "version_retry":
            ModelRegistryService._compute_seed_version = staticmethod(conflicting_seed)
        try:
            assert _advisory_key(("dataset", "node", 0)) == advisory_key
            if index == 1 and not await asyncio.to_thread(first_checked.wait, 20):
                raise AssertionError("First process never checked for an existing job")
            async with AsyncSession(engine, expire_on_commit=False) as session:
                graph = (
                    {"invalid_json": object()}
                    if index == 0 and scenario == "insert_failure"
                    else {}
                )
                result_queue.put(("ok", await _submit(session, graph=graph)))
        except Exception as exc:  # noqa: BLE001 - relay child failures to the parent test
            result_queue.put(("error", repr(exc)))
        finally:
            JobManager.find_active_job = staticmethod(original_find)
            ModelRegistryService._compute_seed_version = staticmethod(original_seed)
            await engine.dispose()

    asyncio.run(run())


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["success", "insert_failure", "version_retry"])
async def test_two_processes_reserve_one_job(database, scenario):
    """An absent job must still have one owner across independent API processes."""
    context = multiprocessing.get_context("spawn")
    first_checked, second_attempted = context.Event(), context.Event()
    results = context.Queue()
    workers = [
        context.Process(
            target=_submission_process,
            args=(
                str(database.url),
                index,
                first_checked,
                second_attempted,
                results,
                scenario,
                _advisory_key(("dataset", "node", 0)),
            ),
        )
        for index in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        outcomes = await asyncio.to_thread(lambda: [results.get(timeout=40) for _ in workers])
        for worker in workers:
            await asyncio.to_thread(worker.join, 10)
        assert [worker.exitcode for worker in workers] == [0, 0]
        errors = [value for status, value in outcomes if status == "error"]
        if scenario == "insert_failure":
            assert len(errors) == 1 and "JSON serializable" in errors[0]
        else:
            assert errors == [], outcomes
        submitted = [value for status, value in outcomes if status == "ok"]
        assert len({job_id for job_id, _ in submitted}) == 1, submitted
        assert sorted(existing for _, existing in submitted) == (
            [False] if scenario == "insert_failure" else [False, True]
        )
        async with AsyncSession(database) as session:
            rows = (await session.execute(select(TrainingJob))).scalars().all()
            assert [(row.id, row.version) for row in rows] == [(submitted[0][0], 1)]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(10)
        results.close()
        results.join_thread()


@pytest.mark.asyncio
async def test_waiting_submission_keeps_the_same_lock():
    """A later arrival must queue behind callers already waiting for this key."""
    key = "oc277-waiters"
    owner = await submit._get_submit_lock(key)
    await owner.acquire()
    waiter = await submit._get_submit_lock(key)
    waiting = asyncio.Event()

    async def wait_for_owner():
        """Queue a real coroutine before the registry owner releases its reference."""
        waiting.set()
        await waiter.acquire()

    waiting_task = asyncio.create_task(wait_for_owner())
    await waiting.wait()
    owner.release()
    await submit._release_submit_lock(key)
    newcomer = await submit._get_submit_lock(key)
    try:
        assert newcomer is waiter
    finally:
        await waiting_task
        waiter.release()
        await submit._release_submit_lock(key)
        await submit._release_submit_lock(key)
    assert key not in submit._submit_locks


@pytest.mark.asyncio
async def test_submission_reuses_request_connection_with_capacity_one(database):
    """Loader reads must not exhaust the pool while waiting for another connection."""
    engine = create_async_engine(
        database.url,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.1,
        connect_args={"isolation_level": None},
    )
    try:
        async with AsyncSession(engine) as session:
            await session.execute(text("SELECT 1"))
            job_id, existing = await _submit(session)
            assert job_id and not existing
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_orm", [False, True])
async def test_submission_rejects_unrelated_writes_without_losing_them(database, pending_orm):
    """Reservation preconditions must preserve writes owned by a different caller."""
    async with AsyncSession(database) as session:
        counter = ModelVersionCounter(
            dataset_source_id="unrelated", model_type="other", current_version=7
        )
        if pending_orm:
            session.add(counter)
        else:
            await session.execute(text("BEGIN"))
            await session.execute(
                insert(ModelVersionCounter).values(
                    dataset_source_id="unrelated", model_type="other", current_version=7
                )
            )
        with pytest.raises(ValueError, match="Submission requires"):
            await _submit(session)
        assert not submit._submit_locks
        if pending_orm:
            assert counter in session.new
        else:
            assert (
                await session.execute(select(ModelVersionCounter.current_version))
            ).scalar() == 7
        await session.rollback()
        job_id, existing = await _submit(session)
        assert job_id and not existing


@pytest.mark.asyncio
async def test_failed_creation_releases_lock_and_version(database):
    """A failed insert must leave no lock or committed version reservation."""
    async with AsyncSession(database, expire_on_commit=False) as session:
        with pytest.raises(StatementError, match="JSON serializable"):
            await _submit(session, graph={"invalid_json": object()})
        assert not submit._submit_locks
        assert (await session.execute(select(ModelVersionCounter))).all() == []
        job_id, existing = await _submit(session)
        assert not existing
        job = await session.get(TrainingJob, job_id)
        assert job is not None and job.version == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["lookup", "allocated_version"])
async def test_cancelled_owner_and_waiter_release_locks(database, monkeypatch, stage):
    """Cancellation must free both database ownership and waiter references."""
    checked, waiter_registered = asyncio.Event(), asyncio.Event()
    never_finish = asyncio.Event()
    original_find, original_get = JobManager.find_active_job, submit._get_submit_lock
    original_create = JobManager.create_job
    calls = 0

    async def hold_check(*args, **kwargs):
        """Suspend the owner inside the database reservation until cancellation."""
        checked.set()
        await never_finish.wait()
        return await original_find(*args, **kwargs)

    async def hold_after_version(**kwargs):
        """Suspend after the allocator commits internally but before the job exists."""
        await ModelRegistryService.get_next_version(
            kwargs["session"], kwargs["dataset_id"], kwargs["model_type"], "training"
        )
        checked.set()
        await never_finish.wait()
        return await original_create(**kwargs)

    async def observe_get(key):
        """Signal after the waiter owns a reference, before it acquires the lock."""
        nonlocal calls
        lock = await original_get(key)
        calls += 1
        if calls == 2:
            waiter_registered.set()
        return lock

    if stage == "lookup":
        monkeypatch.setattr(JobManager, "find_active_job", hold_check)
    else:
        monkeypatch.setattr(JobManager, "create_job", hold_after_version)
    monkeypatch.setattr(submit, "_get_submit_lock", observe_get)
    async with AsyncSession(database) as first, AsyncSession(database) as second:
        owner = asyncio.create_task(_submit(first))
        await asyncio.wait_for(checked.wait(), 5)
        waiter = asyncio.create_task(_submit(second))
        await asyncio.wait_for(waiter_registered.wait(), 5)
        waiter.cancel()
        owner.cancel()
        results = await asyncio.gather(owner, waiter, return_exceptions=True)
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
        assert not submit._submit_locks
    monkeypatch.setattr(JobManager, "find_active_job", original_find)
    monkeypatch.setattr(JobManager, "create_job", original_create)
    async with AsyncSession(database) as session:
        assert (await session.execute(select(ModelVersionCounter))).all() == []
        job_id, existing = await _submit(session)
        assert job_id and not existing


@pytest.mark.asyncio
async def test_invalid_job_types_return_400_without_locks(database, monkeypatch):
    """Malformed requests must fail before reserving or dispatching a job."""
    app = FastAPI()
    app.include_router(submit.router)

    async def session_dependency():
        """Supply the same isolated database as the other reservation tests."""
        async with AsyncSession(database) as session:
            yield session

    app.dependency_overrides[get_async_session] = session_dependency
    monkeypatch.setattr(submit, "resolve_pipeline_nodes", AsyncMock(return_value={}))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as client:
        responses = [
            await client.post(
                "/run",
                json={
                    "pipeline_id": f"invalid-{index}",
                    "job_type": f"unknown-{index}",
                    "nodes": [{"node_id": f"node-{index}", "step_type": "data_loader"}],
                },
            )
            for index in range(4)
        ]
    assert [response.status_code for response in responses] == [400] * 4
    assert not submit._submit_locks
    async with AsyncSession(database) as session:
        assert (await session.execute(select(TrainingJob))).all() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("job_type", ["training", "tuning", "preview"])
async def test_run_commits_before_dispatch_and_dispatches_only_once(
    database, monkeypatch, job_type
):
    """Workers must see the committed job, and deduplicated HTTP calls must not dispatch again."""
    app = FastAPI()
    app.include_router(submit.router)
    visible_jobs, dispatched = [], []

    async def session_dependency():
        """Keep request sessions independent, as production dependency injection does."""
        async with AsyncSession(database) as session:
            yield session

    def observe_created(event):
        """Read through an independent DB connection at the first dispatch-side event."""
        with sqlite3.connect(database.url.database) as connection:
            visible_jobs.append(
                connection.execute(
                    "SELECT id, status FROM training_jobs WHERE id = ?", (event.job_id,)
                ).fetchone()
            )

    def record_dispatch(job_id, payload):
        """Record the actual BackgroundTasks invocation without running machine learning."""
        dispatched.append(job_id)

    app.dependency_overrides[get_async_session] = session_dependency
    monkeypatch.setattr(submit, "resolve_pipeline_nodes", AsyncMock(return_value={}))
    monkeypatch.setattr(submit, "publish_job_event", observe_created)
    monkeypatch.setattr(submit, "run_pipeline_task", record_dispatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        payload = {
            "pipeline_id": "dispatch",
            "job_type": job_type,
            "nodes": [{"node_id": "loader", "step_type": "data_loader"}],
        }
        responses = [await client.post("/run", json=payload) for _ in range(2)]
    assert [response.status_code for response in responses] == [200, 200]
    job_id = responses[0].json()["job_id"]
    assert responses[1].json()["job_id"] == job_id
    assert visible_jobs == [(job_id, "queued")]
    assert dispatched == [job_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "expired"])
async def test_terminal_and_expired_jobs_preserve_history(database, status):
    """Idempotency must allow legitimate later runs without rewriting old jobs."""
    async with AsyncSession(database, expire_on_commit=False) as session:
        job_id, _ = await _submit(session)
        job = await session.get(TrainingJob, job_id)
        assert job is not None
        job.status = "queued" if status == "expired" else status
        if status == "expired":
            job.created_at = datetime.now(UTC) - timedelta(days=1)
        await session.commit()
        new_id, existing = await _submit(session)
        assert not existing and new_id != job_id
        rows = (
            (await session.execute(select(TrainingJob).order_by(TrainingJob.version)))
            .scalars()
            .all()
        )
        assert [(row.id, row.version) for row in rows] == [(job_id, 1), (new_id, 2)]


@pytest.mark.asyncio
async def test_parallel_branches_and_job_modes_dedupe_independently(database):
    """Many branches must remain distinct while every mode shares one branch key."""
    async with AsyncSession(database) as session:
        jobs = [await _submit(session, branch=index) for index in range(23)]
        for mode in ("training", "tuning", "preview"):
            assert await _submit(session, branch=22, job_type=mode) == (jobs[22][0], True)
        assert len({job_id for job_id, _ in jobs}) == 23
