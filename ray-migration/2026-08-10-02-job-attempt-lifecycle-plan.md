# Job Attempt Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add durable per-execution attempt records, a `cancel_requested → cancelled` two-phase cancellation, `submission_failed` and `retrying` job states, and per-attempt failure classification — all working with the existing local and Celery adapters, and reflected in the frontend job status surfaces.

**Architecture:** A new `execution_attempts` table (separate from `training_jobs`, so `Base.metadata.create_all` adds it without altering the existing job table) records every physical execution of a logical job: its backend, external execution id, status, error classification, and artifact URI. The logical `TrainingJob` row stays the frontend's source of truth; attempts give it retry lineage and a place to store the external id instead of stuffing everything into `job_metadata`. Cancellation becomes two-phase; retries append an attempt to the same logical job.

**Tech Stack:** Python 3.12, SQLAlchemy 2.0 (async + sync sessions), FastAPI, Pydantic v2, pytest, React + TypeScript (frontend status surfaces), Vitest.

## Global Constraints

- Preserve the DB-as-truth model, the cancellation late-write guard, retry
  endpoint guard rules (only failed/cancelled training/tuning jobs with a
  stored graph; 400 otherwise; concurrent-retry dedupe), the WebSocket
  invalidator pattern, the local fallback, and the Celery rollback path.
  Manual retry intentionally changes identity semantics: it returns the same
  logical job ID and appends a new attempt, while preserving the endpoint path
  and response shape.
- Do **not** add Ray or any runtime dependency in this plan (Ray arrives in plan 03).
- Prefer a **new table** (`execution_attempts`) over widening `training_jobs`; `create_all` adds new tables but never `ALTER`s existing ones. Include an idempotent backfill for pre-existing jobs.
- New job statuses `cancel_requested`, `submission_failed`, and `retrying` are exposed through `JobInfo.status`, which **changes the API status enum shape** the frontend consumes. Per the backend↔frontend sync rule, the exact frontend files below are updated in Task 7, and the frontend gate (`npm run lint`, `npx tsc --project tsconfig.json --noEmit`, `npm run build`) is run.
- Target Python 3.12 idioms and full typing; avoid `Any` where a concrete type exists. Every new function/method has a 1–2 line docstring.
- Every implementation task follows TDD and ends with a focused commit whose message includes:
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`
- After Python changes run, in order:
  - `ruff check .`
  - `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
  - `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
- Depends on plan 01 (execution backend registry, `dispatch_branches`, `EXECUTION_BACKEND`).

---

## File Structure

Create:

- `backend/ml_pipeline/_execution/attempts.py` — `AttemptStatus` enum, the `ExecutionAttemptRepository` (async + sync methods), and the `backfill_missing_attempts` helper.
- `backend/ml_pipeline/_execution/retry_policy.py` — `FailureClass` enum, `RetryDecision`, and `classify_failure`.
- `tests/test_execution_attempts_model.py`, `tests/test_execution_attempts_repository.py`, `tests/test_retry_policy.py`, `tests/test_cancellation_two_phase.py`, `tests/test_attempt_backfill.py`.

Modify:

- `backend/database/models.py` — add the `ExecutionAttempt` model after `TrainingJob`.
- `backend/ml_pipeline/_execution/schemas.py:11-17` — add `CANCEL_REQUESTED`, `SUBMISSION_FAILED`, `RETRYING` to `JobStatus`.
- `backend/config/mixins/execution.py` — add `MAX_AUTO_RETRIES`.
- `backend/ml_pipeline/_execution/jobs.py` — create attempt #1 in `create_job`; add `submission_failed` handling.
- `backend/ml_pipeline/_execution/backends/dispatch.py` — resolve the latest attempt per job, submit with its id, record the external id on it, and mark `submission_failed` on submit exceptions.
- `backend/ml_pipeline/_services/pipeline_execution_service.py` — mark the attempt running/terminal; finalize `cancel_requested` as `cancelled`.
- `backend/ml_pipeline/_execution/job_manager_base.py` — two-phase cancel + extended guard.
- `backend/ml_pipeline/_internal/_routers/run_pipeline.py` — `resubmit_job_as_new_attempt`.
- `backend/ml_pipeline/_internal/_routers/jobs.py` — retry endpoint returns the same logical `job_id`.
- `backend/main.py` — call `backfill_missing_attempts` once at startup.
- Frontend: `frontend/ml-canvas/src/core/api/jobs.ts`, `frontend/ml-canvas/src/components/shared/StatusBadge.tsx`, `frontend/ml-canvas/src/core/hooks/useJobPolling.ts` (+ their `.test` files).

---

### Task 1: ExecutionAttempt model + status enums

**Files:**
- Modify: `backend/database/models.py` (add model after `TrainingJob`, line ~353)
- Modify: `backend/ml_pipeline/_execution/schemas.py:11-17`
- Create: `backend/ml_pipeline/_execution/attempts.py` (enum only for this task; repository in Task 2)
- Test: `tests/test_execution_attempts_model.py`

**Interfaces:**
- Produces:
  - `class ExecutionAttempt(Base, TimestampMixin)` — table `execution_attempts` with columns: `id: str` (PK), `job_id: str` (FK `training_jobs.id`, indexed), `attempt_number: int`, `backend: str`, `external_execution_id: str | None`, `status: str`, `error_class: str | None`, `error_message: str | None`, `artifact_uri: str | None`, `is_final: bool`, `started_at: datetime | None`, `finished_at: datetime | None`.
  - `class AttemptStatus(StrEnum)` — `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCEL_REQUESTED`, `CANCELLED`, `SUBMISSION_FAILED`.
  - `JobStatus` gains `CANCEL_REQUESTED = "cancel_requested"`, `SUBMISSION_FAILED = "submission_failed"`, `RETRYING = "retrying"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_execution_attempts_model.py`:

```python
"""Schema tests for the execution_attempts table and status enums."""

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, ExecutionAttempt, TrainingJob
from backend.ml_pipeline._execution.attempts import AttemptStatus
from backend.ml_pipeline._execution.schemas import JobStatus


def test_job_status_has_new_lifecycle_members():
    """The logical-job status enum exposes the new lifecycle states."""
    assert JobStatus.CANCEL_REQUESTED.value == "cancel_requested"
    assert JobStatus.SUBMISSION_FAILED.value == "submission_failed"
    assert JobStatus.RETRYING.value == "retrying"


def test_attempt_status_enum_values():
    """Attempt statuses cover queued through the terminal/cancel states."""
    assert {s.value for s in AttemptStatus} == {
        "queued", "running", "succeeded", "failed",
        "cancel_requested", "cancelled", "submission_failed",
    }


@pytest.mark.asyncio
async def test_execution_attempts_table_is_created_alongside_training_jobs():
    """create_all adds execution_attempts without altering training_jobs."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        tables = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
    assert "execution_attempts" in tables
    assert "training_jobs" in tables

    maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        session.add(TrainingJob(id="j1", pipeline_id="p", node_id="n", dataset_source_id="d",
                                status="queued", run_mode="fixed", model_type="rf", graph={}))
        await session.commit()
        session.add(ExecutionAttempt(id="a1", job_id="j1", attempt_number=1, backend="local",
                                     status=AttemptStatus.QUEUED.value, is_final=False))
        await session.commit()
        got = await session.get(ExecutionAttempt, "a1")
        assert got is not None and got.job_id == "j1" and got.is_final is False
    await engine.dispose()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_execution_attempts_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'ExecutionAttempt'` (and `AttemptStatus`).

- [ ] **Step 3: Extend `JobStatus`**

In `backend/ml_pipeline/_execution/schemas.py`, extend the enum:

```python
class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # Lifecycle states (plan 02): cancellation requested but not yet confirmed,
    # a submission that never reached a worker, and the brief retry transition.
    CANCEL_REQUESTED = "cancel_requested"
    SUBMISSION_FAILED = "submission_failed"
    RETRYING = "retrying"
```

- [ ] **Step 4: Add the `AttemptStatus` enum**

Create `backend/ml_pipeline/_execution/attempts.py`:

```python
"""Execution attempt status enum, repository, and backfill.

A logical ``TrainingJob`` can be executed multiple times (initial run plus
retries). Each physical execution is one ``ExecutionAttempt`` row so a retry
never overwrites the failed execution record and the external execution id
lives on the attempt rather than in ``job_metadata``.
"""

from enum import StrEnum


class AttemptStatus(StrEnum):
    """Lifecycle state of a single physical execution attempt."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    SUBMISSION_FAILED = "submission_failed"
```

- [ ] **Step 5: Add the `ExecutionAttempt` model**

In `backend/database/models.py`, add after the `TrainingJob` class (before `ModelVersionCounter`):

```python
class ExecutionAttempt(Base, TimestampMixin):
    """One physical execution of a logical TrainingJob.

    Kept in its own table so ``create_all`` can add it without altering
    ``training_jobs``. Only a successful attempt is marked ``is_final`` and
    promoted as the job's artifact.
    """

    __tablename__ = "execution_attempts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("training_jobs.id"), nullable=False, index=True
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    backend: Mapped[str] = mapped_column(String(20), nullable=False, default="local")
    external_execution_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued", index=True)
    error_class: Mapped[str | None] = mapped_column(String(40), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_final: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="0")
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        """Serialize the attempt for admin/reconciliation views."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "attempt_number": self.attempt_number,
            "backend": self.backend,
            "external_execution_id": self.external_execution_id,
            "status": self.status,
            "error_class": self.error_class,
            "error_message": self.error_message,
            "artifact_uri": self.artifact_uri,
            "is_final": self.is_final,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_execution_attempts_model.py -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Static checks**

Run: `ruff check backend/database/models.py backend/ml_pipeline/_execution/attempts.py backend/ml_pipeline/_execution/schemas.py tests/test_execution_attempts_model.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add backend/database/models.py backend/ml_pipeline/_execution/attempts.py backend/ml_pipeline/_execution/schemas.py tests/test_execution_attempts_model.py
git commit -m "feat(db): add execution_attempts table and lifecycle status enums

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: ExecutionAttemptRepository (async + sync)

**Files:**
- Modify: `backend/ml_pipeline/_execution/attempts.py` (add the repository)
- Test: `tests/test_execution_attempts_repository.py`

**Interfaces:**
- Consumes: `ExecutionAttempt` model, `AttemptStatus` (Task 1).
- Produces `class ExecutionAttemptRepository` with:
  - `async def create_initial_attempt(session: AsyncSession, job_id: str, backend: str) -> str`
  - `async def create_retry_attempt(session: AsyncSession, job_id: str, backend: str) -> str`
  - `async def next_attempt_number(session: AsyncSession, job_id: str) -> int`
  - `async def latest_attempt(session: AsyncSession, job_id: str) -> ExecutionAttempt | None`
  - `async def record_external_id(session: AsyncSession, attempt_id: str, external_execution_id: str) -> None`
  - `async def has_active_attempt(session: AsyncSession, job_id: str) -> bool`
  - `def latest_attempt_sync(session: Session, job_id: str) -> ExecutionAttempt | None`
  - `def mark_running_sync(session: Session, attempt_id: str) -> None`
  - `def mark_terminal_sync(session: Session, attempt_id: str, status: AttemptStatus, *, error_class: str | None = None, error_message: str | None = None, artifact_uri: str | None = None, is_final: bool = False) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_execution_attempts_repository.py`:

```python
"""Repository tests for creating and transitioning execution attempts."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.database.models import Base, ExecutionAttempt, TrainingJob
from backend.ml_pipeline._execution.attempts import (
    AttemptStatus,
    ExecutionAttemptRepository as Repo,
)


@pytest.mark.asyncio
async def test_create_initial_then_retry_increments_number():
    """The first attempt is #1; a retry attempt is #2."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        session.add(TrainingJob(id="j1", pipeline_id="p", node_id="n", dataset_source_id="d",
                                status="queued", run_mode="fixed", model_type="rf", graph={}))
        await session.commit()
        a1 = await Repo.create_initial_attempt(session, "j1", "local")
        assert (await session.get(ExecutionAttempt, a1)).attempt_number == 1
        assert await Repo.has_active_attempt(session, "j1") is True
        a2 = await Repo.create_retry_attempt(session, "j1", "local")
        assert (await session.get(ExecutionAttempt, a2)).attempt_number == 2
        latest = await Repo.latest_attempt(session, "j1")
        assert latest is not None and latest.id == a2
        await Repo.record_external_id(session, a2, "celery-9")
        assert (await session.get(ExecutionAttempt, a2)).external_execution_id == "celery-9"
    await engine.dispose()


def test_sync_running_and_terminal_transitions():
    """Sync driver-path transitions move an attempt to running then terminal."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add(TrainingJob(id="j2", pipeline_id="p", node_id="n", dataset_source_id="d",
                            status="running", run_mode="fixed", model_type="rf", graph={}))
    session.add(ExecutionAttempt(id="a3", job_id="j2", attempt_number=1, backend="local",
                                 status=AttemptStatus.QUEUED.value, is_final=False))
    session.commit()
    Repo.mark_running_sync(session, "a3")
    assert session.get(ExecutionAttempt, "a3").status == "running"
    Repo.mark_terminal_sync(session, "a3", AttemptStatus.SUCCEEDED,
                            artifact_uri="s3://b/j2/a3", is_final=True)
    got = session.get(ExecutionAttempt, "a3")
    assert got.status == "succeeded" and got.is_final is True and got.finished_at is not None
    session.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_execution_attempts_repository.py -v`
Expected: FAIL — `ImportError: cannot import name 'ExecutionAttemptRepository'`.

- [ ] **Step 3: Implement the repository (append to `attempts.py`)**

```python
import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from backend.database.models import ExecutionAttempt

_ACTIVE_ATTEMPT_STATUSES = {
    AttemptStatus.QUEUED.value,
    AttemptStatus.RUNNING.value,
    AttemptStatus.CANCEL_REQUESTED.value,
}


class ExecutionAttemptRepository:
    """CRUD + transitions for ExecutionAttempt rows (async submit, sync driver)."""

    @staticmethod
    async def next_attempt_number(session: AsyncSession, job_id: str) -> int:
        """Return the next 1-based attempt number for a logical job."""
        stmt = select(func.max(ExecutionAttempt.attempt_number)).where(
            ExecutionAttempt.job_id == job_id
        )
        current = (await session.execute(stmt)).scalar()
        return int(current or 0) + 1

    @staticmethod
    async def _create(session: AsyncSession, job_id: str, backend: str, number: int) -> str:
        """Insert a queued attempt row and return its id."""
        attempt_id = str(uuid.uuid4())
        session.add(
            ExecutionAttempt(
                id=attempt_id,
                job_id=job_id,
                attempt_number=number,
                backend=backend,
                status=AttemptStatus.QUEUED.value,
                is_final=False,
            )
        )
        await session.commit()
        return attempt_id

    @staticmethod
    async def create_initial_attempt(session: AsyncSession, job_id: str, backend: str) -> str:
        """Create attempt #1 for a freshly created job."""
        return await ExecutionAttemptRepository._create(session, job_id, backend, 1)

    @staticmethod
    async def create_retry_attempt(session: AsyncSession, job_id: str, backend: str) -> str:
        """Create the next attempt for a job being retried."""
        number = await ExecutionAttemptRepository.next_attempt_number(session, job_id)
        return await ExecutionAttemptRepository._create(session, job_id, backend, number)

    @staticmethod
    async def latest_attempt(session: AsyncSession, job_id: str) -> ExecutionAttempt | None:
        """Return the highest-numbered attempt for a job, or None."""
        stmt = (
            select(ExecutionAttempt)
            .where(ExecutionAttempt.job_id == job_id)
            .order_by(ExecutionAttempt.attempt_number.desc())
            .limit(1)
        )
        return (await session.execute(stmt)).scalar_one_or_none()

    @staticmethod
    async def record_external_id(
        session: AsyncSession, attempt_id: str, external_execution_id: str
    ) -> None:
        """Store the backend-assigned external id on the attempt."""
        attempt = await session.get(ExecutionAttempt, attempt_id)
        if attempt is None:
            return
        attempt.external_execution_id = external_execution_id
        await session.commit()

    @staticmethod
    async def has_active_attempt(session: AsyncSession, job_id: str) -> bool:
        """Return True when the job has a queued/running/cancel-requested attempt."""
        stmt = select(ExecutionAttempt.id).where(
            ExecutionAttempt.job_id == job_id,
            ExecutionAttempt.status.in_(_ACTIVE_ATTEMPT_STATUSES),
        )
        return (await session.execute(stmt)).first() is not None

    @staticmethod
    def latest_attempt_sync(session: Session, job_id: str) -> ExecutionAttempt | None:
        """Sync variant of ``latest_attempt`` for the driver/worker path."""
        return (
            session.query(ExecutionAttempt)
            .filter(ExecutionAttempt.job_id == job_id)
            .order_by(ExecutionAttempt.attempt_number.desc())
            .first()
        )

    @staticmethod
    def mark_running_sync(session: Session, attempt_id: str) -> None:
        """Transition an attempt to running and stamp ``started_at``."""
        attempt = session.get(ExecutionAttempt, attempt_id)
        if attempt is None:
            return
        attempt.status = AttemptStatus.RUNNING.value
        attempt.started_at = datetime.now(UTC)
        session.commit()

    @staticmethod
    def mark_terminal_sync(
        session: Session,
        attempt_id: str,
        status: AttemptStatus,
        *,
        error_class: str | None = None,
        error_message: str | None = None,
        artifact_uri: str | None = None,
        is_final: bool = False,
    ) -> None:
        """Write a terminal state onto an attempt with optional error/artifact fields."""
        attempt = session.get(ExecutionAttempt, attempt_id)
        if attempt is None:
            return
        attempt.status = status.value
        attempt.error_class = error_class
        attempt.error_message = error_message
        if artifact_uri is not None:
            attempt.artifact_uri = artifact_uri
        attempt.is_final = is_final
        attempt.finished_at = datetime.now(UTC)
        session.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_execution_attempts_repository.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/attempts.py tests/test_execution_attempts_repository.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add backend/ml_pipeline/_execution/attempts.py tests/test_execution_attempts_repository.py
git commit -m "feat(execution): add ExecutionAttemptRepository (async submit + sync driver)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Wire attempts into submit → dispatch → execute

**Files:**
- Modify: `backend/ml_pipeline/_execution/jobs.py:29-77` (`create_job`)
- Modify: `backend/ml_pipeline/_execution/backends/dispatch.py`
- Modify: `backend/ml_pipeline/_services/pipeline_execution_service.py:169-229` (`execute_pipeline`, `_write_pipeline_result`, `_handle_execution_exception`)
- Test: `tests/test_pipeline_task.py` (extend), `tests/test_execution_backend_dispatch.py` (extend)

**Interfaces:**
- Consumes: `ExecutionAttemptRepository` (Task 2), `AttemptStatus`, `classify_failure` (Task 5 — imported lazily; until then failures record `error_class=None`), `get_settings().EXECUTION_BACKEND`.
- Produces:
  - `JobManager.create_job(...)` also creates attempt #1 (backend = `settings.EXECUTION_BACKEND`) and returns the same `job_id`.
  - `dispatch_branches(...)` resolves the latest attempt per job, submits with its `attempt_id`, records the external id on it, and on submit exception marks the attempt `submission_failed` and the job `submission_failed`.
  - `execute_pipeline(...)` marks the attempt running at start, terminal on success/failure, and finalizes `cancel_requested` jobs as `cancelled`.

- [ ] **Step 1: Write the failing tests (extend `tests/test_execution_backend_dispatch.py`)**

```python
@pytest.mark.asyncio
async def test_dispatch_records_external_id_on_attempt():
    """The dispatcher records the external id on the job's latest attempt."""
    from backend.ml_pipeline._execution.backends.dispatch import dispatch_branches

    fake_backend = MagicMock()
    fake_backend.name = "celery"
    fake_backend.submit.return_value.external_execution_id = "celery-777"
    fake_attempt = MagicMock()
    fake_attempt.id = "att-1"
    with (
        patch(
            "backend.ml_pipeline._execution.backends.dispatch.get_execution_backend",
            return_value=fake_backend,
        ),
        patch(
            "backend.ml_pipeline._execution.backends.dispatch.ExecutionAttemptRepository.latest_attempt",
            new=AsyncMock(return_value=fake_attempt),
        ),
        patch(
            "backend.ml_pipeline._execution.backends.dispatch.ExecutionAttemptRepository.record_external_id",
            new=AsyncMock(),
        ) as mock_record,
    ):
        await dispatch_branches([("j1", {"nodes": []})], settings=_settings(EXECUTION_BACKEND="celery"), db=AsyncMock())
    mock_record.assert_awaited_once()
    assert mock_record.await_args.args[1:] == ("att-1", "celery-777")


@pytest.mark.asyncio
async def test_dispatch_marks_submission_failed_on_submit_error():
    """A submit exception marks the job submission_failed instead of raising."""
    from backend.ml_pipeline._execution.backends.dispatch import dispatch_branches

    fake_backend = MagicMock()
    fake_backend.name = "celery"
    fake_backend.submit.side_effect = RuntimeError("broker down")
    with (
        patch(
            "backend.ml_pipeline._execution.backends.dispatch.get_execution_backend",
            return_value=fake_backend,
        ),
        patch(
            "backend.ml_pipeline._execution.backends.dispatch.ExecutionAttemptRepository.latest_attempt",
            new=AsyncMock(return_value=MagicMock(id="att-1")),
        ),
        patch(
            "backend.ml_pipeline._execution.backends.dispatch._mark_submission_failed",
            new=AsyncMock(),
        ) as mock_fail,
    ):
        await dispatch_branches([("j1", {})], settings=_settings(EXECUTION_BACKEND="celery"), db=AsyncMock())
    mock_fail.assert_awaited_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_execution_backend_dispatch.py -v -k "records_external_id or submission_failed"`
Expected: FAIL — dispatcher does not yet touch attempts or `_mark_submission_failed`.

- [ ] **Step 3: Create attempt #1 in `create_job`**

In `backend/ml_pipeline/_execution/jobs.py`, wrap the existing branch logic so that after the manager creates the job row, an initial attempt is created. Replace the three manager return points in `create_job` with a shared tail:

```python
    @staticmethod
    async def create_job(
        session: AsyncSession,
        pipeline_id: str,
        node_id: str,
        job_type: Literal["training", "tuning", "preview"],
        dataset_id: str = "unknown",
        user_id: int | None = None,
        model_type: str = "unknown",
        graph: dict[str, Any] | None = None,
        branch_index: int = 0,
    ) -> str:
        """Create a job row and its initial execution attempt (Async)."""
        if job_type == "training":
            job_id = await BasicTrainingManager.create_training_job(
                session, pipeline_id, node_id, dataset_id, user_id, model_type, graph,
                branch_index=branch_index,
            )
        elif job_type == "tuning":
            job_id = await AdvancedTuningManager.create_tuning_job(
                session, pipeline_id, node_id, dataset_id, user_id, model_type, graph,
                branch_index=branch_index,
            )
        elif job_type == "preview":
            job_id = await BasicTrainingManager.create_training_job(
                session, pipeline_id, node_id, dataset_id, user_id, model_type, graph,
                is_preview=True, branch_index=branch_index,
            )
        else:
            raise ValueError(f"Unknown job_type: {job_type}")

        from backend.ml_pipeline._execution.attempts import ExecutionAttemptRepository

        await ExecutionAttemptRepository.create_initial_attempt(
            session, job_id, get_settings().EXECUTION_BACKEND
        )
        return job_id
```

- [ ] **Step 4: Update the dispatcher to record external ids and mark submission failures**

Replace `backend/ml_pipeline/_execution/backends/dispatch.py` body with:

```python
"""Branch dispatcher — the single fan-out point for pipeline submissions.

Each branch payload becomes one execution submitted through the configured
backend. The returned external id is recorded on the job's latest attempt (and
kept on ``job_metadata.celery_task_id`` for the existing cancel path). A submit
exception marks the job/attempt ``submission_failed`` — production must never
silently drop a submission.
"""

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings
from backend.database.models import ExecutionAttempt, TrainingJob
from backend.ml_pipeline._execution.attempts import AttemptStatus, ExecutionAttemptRepository
from backend.ml_pipeline._execution.backends.base import ExecutionRequest
from backend.ml_pipeline._execution.backends.registry import get_execution_backend
from backend.ml_pipeline._execution.jobs import JobManager
from backend.ml_pipeline._execution.schemas import JobStatus
from backend.realtime.events import JobEvent, publish_job_event

logger = logging.getLogger(__name__)


async def _mark_submission_failed(db: AsyncSession, job_id: str, attempt_id: str | None) -> None:
    """Flip the job and its attempt to submission_failed and emit a status event."""
    if attempt_id is not None:
        attempt = await db.get(ExecutionAttempt, attempt_id)
        if attempt is not None:
            attempt.status = AttemptStatus.SUBMISSION_FAILED.value
    job = (await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))).scalar_one_or_none()
    if job is not None:
        job.status = JobStatus.SUBMISSION_FAILED.value
        job.error_message = "Submission to the execution backend failed."
    await db.commit()
    publish_job_event(JobEvent(event="status", job_id=job_id, status=JobStatus.SUBMISSION_FAILED.value))


async def dispatch_branches(
    task_payloads: list[tuple[str, dict[str, Any]]],
    *,
    settings: Settings,
    db: AsyncSession,
) -> None:
    """Submit each ``(job_id, payload)`` branch through the configured backend."""
    if not task_payloads:
        return

    backend = get_execution_backend(settings)
    for job_id, payload in task_payloads:
        attempt = await ExecutionAttemptRepository.latest_attempt(db, job_id)
        attempt_id = attempt.id if attempt is not None else job_id
        try:
            handle = backend.submit(
                ExecutionRequest(job_id=job_id, attempt_id=attempt_id, payload=payload)
            )
        except Exception:
            logger.exception("Submission failed for job %s", job_id)
            await _mark_submission_failed(db, job_id, attempt.id if attempt else None)
            continue

        if attempt is not None:
            await ExecutionAttemptRepository.record_external_id(db, attempt.id, handle.external_execution_id)
        if backend.name == "celery":
            try:
                await JobManager.attach_celery_task_id(db, job_id, handle.external_execution_id)
            except Exception:
                logger.warning("Failed to attach external id for job %s", job_id)
```

- [ ] **Step 5: Mark the attempt running/terminal in `execute_pipeline`**

In `backend/ml_pipeline/_services/pipeline_execution_service.py`, add the attempt import and transitions.

At module top add:

```python
from backend.ml_pipeline._execution.attempts import AttemptStatus, ExecutionAttemptRepository
```

In `execute_pipeline`, right after the job transitions to running (`session.commit()` following `job.status = "running"`), resolve and mark the attempt:

```python
        attempt = ExecutionAttemptRepository.latest_attempt_sync(session, job_id)
        if attempt is not None:
            ExecutionAttemptRepository.mark_running_sync(session, attempt.id)
```

Replace the mid-run cancellation check so `cancel_requested` finalizes as `cancelled`:

```python
        job.logs = job_logs
        session.refresh(job, ["status"])
        if job.status in (JobStatus.CANCELLED.value, JobStatus.CANCEL_REQUESTED.value):
            logger.info("Job %s cancelled during run; finalizing as cancelled", job_id)
            job.status = JobStatus.CANCELLED.value
            job.finished_at = datetime.now(UTC)
            if attempt is not None:
                ExecutionAttemptRepository.mark_terminal_sync(
                    session, attempt.id, AttemptStatus.CANCELLED
                )
            session.commit()
            publish_job_event(
                JobEvent(event="status", job_id=job_id, status=JobStatus.CANCELLED.value)
            )
            return

        _write_pipeline_result(session, job, strategy, job_id, result, base_artifact_uri)
        if attempt is not None:
            ExecutionAttemptRepository.mark_terminal_sync(
                session,
                attempt.id,
                AttemptStatus.SUCCEEDED if result.status == "success" else AttemptStatus.FAILED,
                artifact_uri=base_artifact_uri if result.status == "success" else None,
                is_final=result.status == "success",
            )
```

Add `JobStatus` to the imports in this module (it currently imports schemas types but not `JobStatus`):

```python
from backend.ml_pipeline._execution.schemas import (
    JobStatus,
    NodeConfig,
    PipelineConfig,
    PipelineExecutionResult,
)
```

In `_handle_execution_exception`, after recording the job failure, record the attempt failure with a classification (import lazily to avoid a cycle):

```python
        job, strategy = JobStrategyFactory.find_job(session, job_id)
        if job and strategy:
            if job.status != "cancelled":
                strategy.handle_failure(job, str(exc))
            session.commit()
            from backend.ml_pipeline._execution.attempts import (  # noqa: PLC0415
                AttemptStatus,
                ExecutionAttemptRepository,
            )
            from backend.ml_pipeline._execution.retry_policy import classify_failure  # noqa: PLC0415

            attempt = ExecutionAttemptRepository.latest_attempt_sync(session, job_id)
            if attempt is not None and attempt.status not in ("cancelled", "succeeded"):
                decision = classify_failure(exc)
                ExecutionAttemptRepository.mark_terminal_sync(
                    session, attempt.id, AttemptStatus.FAILED,
                    error_class=decision.failure_class.value, error_message=str(exc),
                )
            publish_job_event(JobEvent(event="status", job_id=job_id, status=job.status))
```

(Task 5 creates `retry_policy.classify_failure`; sequence Task 5 before running the failure test, or stub it. This plan orders Task 5 after Task 3 but the `_handle_execution_exception` change only executes on failure — the extend-tests in Step 1 do not exercise it, so Task 3's tests pass without `retry_policy`. The full failure test lives in Task 5.)

- [ ] **Step 6: Extend the pipeline task test to assert attempt transitions**

Add to `tests/test_pipeline_task.py`:

```python
def test_run_pipeline_task_marks_attempt_succeeded(mock_get_db_session, mock_engine_class):
    """A successful run marks the job's latest attempt succeeded and final."""
    from unittest.mock import MagicMock

    from backend.database.models import ExecutionAttempt

    session = MagicMock()
    mock_get_db_session.return_value = session
    job = TrainingJob(id=MOCK_JOB_ID, status="queued", run_mode="fixed")
    attempt = ExecutionAttempt(id="att-1", job_id=MOCK_JOB_ID, attempt_number=1,
                               backend="local", status="queued", is_final=False)
    session.query.return_value.filter.return_value.first.return_value = job
    session.query.return_value.filter.return_value.order_by.return_value.first.return_value = attempt
    session.get.return_value = attempt

    engine_instance = mock_engine_class.return_value
    engine_instance.run.return_value = PipelineExecutionResult(
        pipeline_id="test_pipeline", status="success",
        node_results={"node_1": NodeExecutionResult(node_id="node_1", status="success", metrics={"accuracy": 0.9})},
    )
    run_pipeline_task(MOCK_JOB_ID, MOCK_PIPELINE_CONFIG)
    assert attempt.status == "succeeded"
    assert attempt.is_final is True
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_execution_backend_dispatch.py tests/test_pipeline_task.py -q`
Expected: PASS.

- [ ] **Step 8: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/jobs.py backend/ml_pipeline/_execution/backends/dispatch.py backend/ml_pipeline/_services/pipeline_execution_service.py tests/test_execution_backend_dispatch.py tests/test_pipeline_task.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add backend/ml_pipeline/_execution/jobs.py backend/ml_pipeline/_execution/backends/dispatch.py backend/ml_pipeline/_services/pipeline_execution_service.py tests/test_execution_backend_dispatch.py tests/test_pipeline_task.py
git commit -m "feat(execution): thread execution attempts through submit and run

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Two-phase cancellation (cancel_requested → cancelled)

**Files:**
- Modify: `backend/ml_pipeline/_execution/job_manager_base.py:77-148` (`_cancel_job`, `_update_status_sync`, `_handle_cancelled_status_update`)
- Modify: `backend/ml_pipeline/_internal/_routers/jobs.py:114-124` (cancel endpoint status event)
- Test: `tests/test_cancellation_two_phase.py`

**Interfaces:**
- Consumes: `AttemptStatus`, `ExecutionAttemptRepository`, `JobStatus` (Tasks 1–2).
- Produces:
  - `_cancel_job(...)` sets `CANCEL_REQUESTED`, cancels the external execution, and confirms `CANCELLED` immediately only when the job was still `QUEUED`; a `RUNNING` job stays `CANCEL_REQUESTED` until the driver/reconciliation confirms.
  - `_update_status_sync` guard allows `CANCEL_REQUESTED → CANCELLED` but blocks any revival to `RUNNING`/`COMPLETED`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cancellation_two_phase.py`:

```python
"""Two-phase cancellation: queued cancels immediately; running requests then confirms."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.database.models import Base, ExecutionAttempt, TrainingJob
from backend.ml_pipeline._execution.basic_training_manager import BasicTrainingManager
from backend.ml_pipeline._execution.schemas import JobStatus


async def _seed(session, status):
    """Insert a job + queued attempt in the given status."""
    session.add(TrainingJob(id="j1", pipeline_id="p", node_id="n", dataset_source_id="d",
                            status=status, run_mode="fixed", model_type="rf", graph={},
                            job_metadata={"celery_task_id": "celery-1"}))
    session.add(ExecutionAttempt(id="a1", job_id="j1", attempt_number=1, backend="celery",
                                 status="running", is_final=False, external_execution_id="celery-1"))
    await session.commit()


@pytest.mark.asyncio
async def test_queued_job_cancels_immediately(monkeypatch):
    """A queued job is confirmed cancelled at once (nothing is executing)."""
    monkeypatch.setattr(
        "backend.ml_pipeline._execution.job_manager_base.get_execution_backend",
        lambda *_a, **_k: type("B", (), {"cancel": staticmethod(lambda _x: None)})(),
    )
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        await _seed(session, JobStatus.QUEUED.value)
        assert await BasicTrainingManager.cancel_training_job(session, "j1") is True
        job = await session.get(TrainingJob, "j1")
        assert job.status == JobStatus.CANCELLED.value
    await engine.dispose()


@pytest.mark.asyncio
async def test_running_job_enters_cancel_requested(monkeypatch):
    """A running job moves to cancel_requested and waits for confirmation."""
    monkeypatch.setattr(
        "backend.ml_pipeline._execution.job_manager_base.get_execution_backend",
        lambda *_a, **_k: type("B", (), {"cancel": staticmethod(lambda _x: None)})(),
    )
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        await _seed(session, JobStatus.RUNNING.value)
        assert await BasicTrainingManager.cancel_training_job(session, "j1") is True
        job = await session.get(TrainingJob, "j1")
        assert job.status == JobStatus.CANCEL_REQUESTED.value
    await engine.dispose()


def test_guard_allows_cancel_requested_to_cancelled_but_blocks_completed():
    """The sync guard confirms cancellation but refuses revival to completed."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add(TrainingJob(id="j2", pipeline_id="p", node_id="n", dataset_source_id="d",
                            status=JobStatus.CANCEL_REQUESTED.value, run_mode="fixed",
                            model_type="rf", graph={}))
    session.commit()
    # Revival attempt is blocked.
    BasicTrainingManager.update_status_sync(session, "j2", status=JobStatus.COMPLETED)
    assert session.get(TrainingJob, "j2").status == JobStatus.CANCEL_REQUESTED.value
    # Confirmation is allowed.
    BasicTrainingManager.update_status_sync(session, "j2", status=JobStatus.CANCELLED)
    assert session.get(TrainingJob, "j2").status == JobStatus.CANCELLED.value
    session.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cancellation_two_phase.py -v`
Expected: FAIL — current `_cancel_job` sets `CANCELLED` unconditionally and the guard blocks all writes to a cancelled/cancel_requested row.

- [ ] **Step 3: Rewrite `_cancel_job` for two-phase behavior**

In `backend/ml_pipeline/_execution/job_manager_base.py`, replace `_cancel_job`:

```python
    @staticmethod
    async def _cancel_job(
        session: AsyncSession, model: type[Any], job_id: str, run_mode: str | None = None
    ) -> bool:
        """Two-phase cancel: request cancellation, then confirm when safe.

        A queued job is confirmed CANCELLED immediately (nothing is executing).
        A running job moves to CANCEL_REQUESTED; the driver's terminal write or
        reconciliation confirms CANCELLED once the worker actually stops. The
        external execution is cancelled best-effort either way, and the
        late-write guard keeps the row from being revived to COMPLETED.
        """
        stmt = select(model).where(model.id == job_id)
        if run_mode is not None:
            stmt = stmt.where(model.run_mode == run_mode)
        job = (await session.execute(stmt)).scalar_one_or_none()
        if not job or job.status not in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
            return False

        was_queued = job.status == JobStatus.QUEUED.value
        job.status = JobStatus.CANCEL_REQUESTED.value
        job.error_message = "Job cancelled by user."
        from backend.ml_pipeline._execution.attempts import (  # noqa: PLC0415
            AttemptStatus,
            ExecutionAttemptRepository,
        )
        from backend.ml_pipeline._execution.backends.registry import (  # noqa: PLC0415
            get_execution_backend_by_name,
        )

        attempt = await ExecutionAttemptRepository.latest_attempt(session, job_id)
        if attempt is not None and attempt.external_execution_id:
            try:
                backend = get_execution_backend_by_name(attempt.backend)
                backend.cancel(attempt.external_execution_id)
            except Exception:
                logger.warning(
                    "Failed to cancel %s execution %s for job %s",
                    attempt.backend,
                    attempt.external_execution_id,
                    job_id,
                    exc_info=True,
                )
        if was_queued:
            job.status = JobStatus.CANCELLED.value
            job.finished_at = datetime.now(UTC)
            if attempt is not None:
                attempt.status = AttemptStatus.CANCELLED.value
                attempt.finished_at = datetime.now(UTC)
        elif attempt is not None:
            attempt.status = AttemptStatus.CANCEL_REQUESTED.value
        await session.commit()
        return True
```

- [ ] **Step 4: Extend the sync guard to confirm cancellation**

Replace `_update_status_sync`'s guard block:

```python
        if job.status in (JobStatus.CANCELLED.value, JobStatus.CANCEL_REQUESTED.value):
            # A worker finishing a cancelled run may confirm CANCELLED, but must
            # never revive the job to RUNNING/COMPLETED.
            if job.status == JobStatus.CANCEL_REQUESTED.value and status == JobStatus.CANCELLED:
                apply_fields_fn(job, status, error, logs, result)
                session.commit()
                return True
            return TrainingJobManagerBase._handle_cancelled_status_update(session, job, logs)
```

- [ ] **Step 5: Update the cancel endpoint's emitted status**

In `backend/ml_pipeline/_internal/_routers/jobs.py`, the cancel handler should reflect that cancellation may still be in progress. Replace the fixed `status="cancelled"` event with a fetch of the job's actual post-cancel status:

```python
@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, session: AsyncSession = Depends(get_async_session)):
    """Request cancellation of a running or queued job."""
    success = await JobManager.cancel_job(session, job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job could not be cancelled (maybe it's already finished or doesn't exist)",
        )
    job = await JobManager.get_job(session, job_id)
    status = job.status.value if job else "cancel_requested"
    publish_job_event(JobEvent(event="status", job_id=job_id, status=status))
    return {"message": "Job cancellation requested", "status": status}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_cancellation_two_phase.py tests/test_job_manager_base.py tests/test_ml_pipeline_backend_fixes.py -q`
Expected: PASS.

- [ ] **Step 7: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/job_manager_base.py backend/ml_pipeline/_internal/_routers/jobs.py tests/test_cancellation_two_phase.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add backend/ml_pipeline/_execution/job_manager_base.py backend/ml_pipeline/_internal/_routers/jobs.py tests/test_cancellation_two_phase.py
git commit -m "feat(cancel): model cancel_requested then cancelled two-phase lifecycle

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Retry classification + retry-as-new-attempt

**Files:**
- Create: `backend/ml_pipeline/_execution/retry_policy.py`
- Modify: `backend/config/mixins/execution.py` (add `MAX_AUTO_RETRIES`)
- Modify: `backend/ml_pipeline/_internal/_routers/run_pipeline.py` (add `resubmit_job_as_new_attempt`, keep `resubmit_job_from_graph` as a thin wrapper)
- Modify: `backend/ml_pipeline/_internal/_routers/jobs.py:134-170` (retry endpoint returns the same job id)
- Test: `tests/test_retry_policy.py`, `tests/test_jobs_router_retry.py` (update)

**Interfaces:**
- Produces:
  - `class FailureClass(StrEnum)` — `WORKER_OR_NODE_LOSS`, `TRANSIENT_IO`, `INVALID_CONFIG`, `OUT_OF_MEMORY`, `USER_CANCELLATION`, `UNKNOWN`.
  - `@dataclass(frozen=True) class RetryDecision` — `failure_class: FailureClass`, `retriable: bool`, `reason: str`.
  - `def classify_failure(error: BaseException | str) -> RetryDecision`.
  - `async def resubmit_job_as_new_attempt(db: AsyncSession, job: JobInfo, background_tasks: BackgroundTasks) -> str` — returns the same logical `job_id`.
- Consumes: `ExecutionAttemptRepository`, `dispatch_branches`, `JobStatus`, `Settings.MAX_AUTO_RETRIES`.

- [ ] **Step 1: Write the failing test for classification**

Create `tests/test_retry_policy.py`:

```python
"""Failure classification drives whether an automatic retry is eligible."""

from backend.ml_pipeline._execution.retry_policy import (
    FailureClass,
    classify_failure,
)


def test_worker_loss_is_retriable():
    """A worker/node loss is eligible for a bounded automatic retry."""
    d = classify_failure("WorkerLostError: worker exited prematurely")
    assert d.failure_class is FailureClass.WORKER_OR_NODE_LOSS
    assert d.retriable is True


def test_transient_io_is_retriable():
    """A transient connection reset is retriable."""
    d = classify_failure(ConnectionResetError("connection reset by peer"))
    assert d.failure_class is FailureClass.TRANSIENT_IO
    assert d.retriable is True


def test_oom_is_not_retriable():
    """An out-of-memory failure is not retried with unchanged resources."""
    d = classify_failure(MemoryError("Unable to allocate array"))
    assert d.failure_class is FailureClass.OUT_OF_MEMORY
    assert d.retriable is False


def test_invalid_config_is_not_retriable():
    """A deterministic ValueError is a config error and not retriable."""
    d = classify_failure(ValueError("Algorithm 'kmeans' is a clustering model"))
    assert d.failure_class is FailureClass.INVALID_CONFIG
    assert d.retriable is False


def test_user_cancellation_is_not_retriable():
    """User cancellation is never auto-retried."""
    d = classify_failure("Job cancelled by user.")
    assert d.failure_class is FailureClass.USER_CANCELLATION
    assert d.retriable is False


def test_unknown_defaults_to_not_retriable():
    """An unrecognized error is conservatively non-retriable."""
    d = classify_failure(RuntimeError("something odd"))
    assert d.failure_class is FailureClass.UNKNOWN
    assert d.retriable is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retry_policy.py -v`
Expected: FAIL — `ModuleNotFoundError: ...retry_policy`.

- [ ] **Step 3: Implement classification and the setting**

Create `backend/ml_pipeline/_execution/retry_policy.py`:

```python
"""Failure classification for automatic retry eligibility.

Manual retry (the ``/jobs/{id}/retry`` endpoint) is always allowed for
terminal failed/cancelled jobs. Automatic retry is bounded and only applies to
transient causes — this module decides the class of a failure so the driver
and reconciliation (plan 05) can schedule at most ``MAX_AUTO_RETRIES`` retries.
"""

from dataclasses import dataclass
from enum import StrEnum


class FailureClass(StrEnum):
    """Category of an execution failure for retry decisions."""

    WORKER_OR_NODE_LOSS = "worker_or_node_loss"
    TRANSIENT_IO = "transient_io"
    INVALID_CONFIG = "invalid_config"
    OUT_OF_MEMORY = "out_of_memory"
    USER_CANCELLATION = "user_cancellation"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """Whether a failure is eligible for automatic retry and why."""

    failure_class: FailureClass
    retriable: bool
    reason: str


_RETRIABLE = {FailureClass.WORKER_OR_NODE_LOSS, FailureClass.TRANSIENT_IO}


def _classify_text(text: str) -> FailureClass:
    """Map a lowercased error string to a failure class."""
    if "cancelled by user" in text or "job cancelled" in text:
        return FailureClass.USER_CANCELLATION
    if "workerlost" in text or "node" in text and "lost" in text or "worker exited" in text:
        return FailureClass.WORKER_OR_NODE_LOSS
    if "memoryerror" in text or "unable to allocate" in text or "out of memory" in text:
        return FailureClass.OUT_OF_MEMORY
    if any(tok in text for tok in ("connection reset", "timed out", "timeout", "temporarily unavailable", "broken pipe")):
        return FailureClass.TRANSIENT_IO
    if "valueerror" in text or "invalid" in text or "not supported" in text:
        return FailureClass.INVALID_CONFIG
    return FailureClass.UNKNOWN


def classify_failure(error: BaseException | str) -> RetryDecision:
    """Classify an execution failure and decide automatic-retry eligibility."""
    if isinstance(error, MemoryError):
        cls = FailureClass.OUT_OF_MEMORY
    elif isinstance(error, (ConnectionError, TimeoutError)):
        cls = FailureClass.TRANSIENT_IO
    elif isinstance(error, ValueError):
        text = f"valueerror {error}".lower()
        cls = FailureClass.USER_CANCELLATION if "cancelled by user" in text else FailureClass.INVALID_CONFIG
    else:
        cls = _classify_text(str(error).lower())
    retriable = cls in _RETRIABLE
    return RetryDecision(failure_class=cls, retriable=retriable, reason=cls.value)
```

Add to `backend/config/mixins/execution.py`:

```python
    # Upper bound on automatic retries per logical job (transient failures only).
    # Manual retries via the API are not bounded by this value.
    MAX_AUTO_RETRIES: int = 1
```

- [ ] **Step 4: Run classification test to verify it passes**

Run: `pytest tests/test_retry_policy.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Write the failing retry-as-attempt test (update `tests/test_jobs_router_retry.py`)**

Replace `test_retry_succeeds_for_failed_training_job` and `test_retry_succeeds_for_cancelled_tuning_job` so they expect the **same** logical job id plus a second attempt:

```python
@pytest.mark.asyncio
async def test_retry_reuses_job_and_adds_attempt(async_session, client):
    """Retry re-runs the same logical job and appends attempt #2."""
    from sqlalchemy import text as _text

    await _insert_job(async_session, "job-1", status="failed")

    with patch(
        "backend.ml_pipeline._execution.backends.dispatch.get_execution_backend"
    ) as mock_get_backend:
        backend = mock_get_backend.return_value
        backend.name = "local"
        backend.submit.return_value.external_execution_id = "local:x"
        response = client.post(f"{BASE}/jobs/job-1/retry")

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == "job-1"  # same logical job
    count = (
        await async_session.execute(
            _text("SELECT COUNT(*) FROM execution_attempts WHERE job_id = 'job-1'")
        )
    ).scalar_one()
    assert count == 2
```

Keep the 404/400 guard tests unchanged (`test_retry_returns_404_for_missing_job`, `test_retry_returns_400_for_non_terminal_job`, `test_retry_returns_400_when_graph_missing`). Update `test_concurrent_retries_create_only_one_job` to assert a single new active attempt (attempt #2) rather than a second job row:

```python
@pytest.mark.asyncio
async def test_concurrent_retries_add_only_one_attempt(async_session):
    """Two near-simultaneous retries of the same job append exactly one attempt."""
    import asyncio

    from sqlalchemy import text as _text

    from backend.ml_pipeline._execution.jobs import JobManager
    from backend.ml_pipeline._internal._routers.run_pipeline import resubmit_job_as_new_attempt

    await _insert_job(async_session, "job-race", status="failed")
    job = await JobManager.get_job(async_session, "job-race")
    assert job is not None

    from fastapi import BackgroundTasks

    with patch("backend.ml_pipeline._execution.backends.dispatch.get_execution_backend") as m:
        m.return_value.name = "local"
        m.return_value.submit.return_value.external_execution_id = "local:x"
        id_a, id_b = await asyncio.gather(
            resubmit_job_as_new_attempt(async_session, job, BackgroundTasks()),
            resubmit_job_as_new_attempt(async_session, job, BackgroundTasks()),
        )
    assert id_a == id_b == "job-race"
    count = (
        await async_session.execute(
            _text("SELECT COUNT(*) FROM execution_attempts WHERE job_id = 'job-race'")
        )
    ).scalar_one()
    assert count == 2  # one initial-equivalent + one retry (dedupe prevented a third)
```

Note: `_insert_job` inserts only the job row, so the retry adds attempt #2 only when an attempt #1 exists. Update `_insert_job` to also insert attempt #1 via raw SQL so numbering starts at 1:

```python
    await session.execute(
        text(
            "INSERT INTO execution_attempts (id, job_id, attempt_number, backend, status, "
            "is_final, created_at, updated_at) VALUES (:id, :job_id, 1, 'local', 'failed', 0, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {"id": f"{job_id}-a1", "job_id": job_id},
    )
    await session.commit()
```

- [ ] **Step 6: Implement `resubmit_job_as_new_attempt` and rewire the endpoint**

In `backend/ml_pipeline/_internal/_routers/run_pipeline.py`, add:

```python
async def resubmit_job_as_new_attempt(
    db: AsyncSession,
    job: JobInfo,
    background_tasks: BackgroundTasks,
) -> str:
    """Retry a terminal job by appending a new attempt to the same logical job.

    Rebuilds the branch payload from the job's stored single-branch graph,
    resets the job to queued via the retrying transition, creates the next
    attempt, and dispatches it through the configured backend. Serialized per
    job so two concurrent retries append only one attempt. Returns the job id.

    Raises:
        ValueError: if the job has no usable stored graph to resubmit.
    """
    from backend.ml_pipeline._execution.attempts import ExecutionAttemptRepository
    from backend.ml_pipeline._execution.backends.dispatch import dispatch_branches

    graph = job.graph or {}
    nodes_raw = graph.get("nodes") or []
    if not nodes_raw:
        raise ValueError("Job has no stored pipeline graph to retry")

    lock = await _get_submit_lock(f"retry:{job.job_id}")
    async with lock:
        if await ExecutionAttemptRepository.has_active_attempt(db, job.job_id):
            await _release_submit_lock(f"retry:{job.job_id}")
            return job.job_id

        internal_nodes = [
            NodeConfig(
                node_id=n["node_id"],
                step_type=coerce_step_type(n["step_type"]),
                params=n.get("params", {}),
                inputs=n.get("inputs", []),
            )
            for n in nodes_raw
        ]
        sub = PipelineConfig(
            pipeline_id=graph.get("pipeline_id", job.pipeline_id),
            nodes=internal_nodes,
            metadata=graph.get("metadata", {}),
        )
        branch_graph = _build_branch_graph(sub)

        settings = get_settings()
        await ExecutionAttemptRepository.create_retry_attempt(
            db, job.job_id, settings.EXECUTION_BACKEND
        )
        await JobManager.reset_job_for_retry(db, job.job_id)

    await _release_submit_lock(f"retry:{job.job_id}")

    publish_job_event(JobEvent(event="status", job_id=job.job_id, status="queued", progress=0))
    await dispatch_branches([(job.job_id, dict(branch_graph))], settings=get_settings(), db=db)
    return job.job_id


async def resubmit_job_from_graph(
    db: AsyncSession,
    job: JobInfo,
    background_tasks: BackgroundTasks,
) -> str:
    """Backward-compatible alias retained for external callers/tests."""
    return await resubmit_job_as_new_attempt(db, job, background_tasks)
```

Add `JobManager.reset_job_for_retry` to `backend/ml_pipeline/_execution/jobs.py`:

```python
    @staticmethod
    async def reset_job_for_retry(session: AsyncSession, job_id: str) -> None:
        """Transition a terminal job through retrying back to queued for a new attempt."""
        job = (await session.execute(select(TrainingJob).where(TrainingJob.id == job_id))).scalar_one_or_none()
        if job is None:
            return
        job.status = JobStatus.RETRYING.value
        await session.commit()
        job.status = JobStatus.QUEUED.value
        job.error_message = None
        job.progress = 0
        job.started_at = datetime.now(UTC)
        job.finished_at = None
        await session.commit()
```

In `backend/ml_pipeline/_internal/_routers/jobs.py`, update the retry endpoint to call the new function and return the same id:

```python
    try:
        result_job_id = await resubmit_job_as_new_attempt(session, job, background_tasks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return RetryJobResponse(job_id=result_job_id, message="Retry submitted")
```

Update the import at the top of `jobs.py`:

```python
from backend.ml_pipeline._internal._routers.run_pipeline import resubmit_job_as_new_attempt
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_retry_policy.py tests/test_jobs_router_retry.py -q`
Expected: PASS.

- [ ] **Step 8: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/retry_policy.py backend/config/mixins/execution.py backend/ml_pipeline/_internal/_routers/run_pipeline.py backend/ml_pipeline/_internal/_routers/jobs.py backend/ml_pipeline/_execution/jobs.py tests/test_retry_policy.py tests/test_jobs_router_retry.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add backend/ml_pipeline/_execution/retry_policy.py backend/config/mixins/execution.py backend/ml_pipeline/_internal/_routers/run_pipeline.py backend/ml_pipeline/_internal/_routers/jobs.py backend/ml_pipeline/_execution/jobs.py tests/test_retry_policy.py tests/test_jobs_router_retry.py
git commit -m "feat(retry): classify failures and retry as a new attempt on the same job

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: Idempotent attempt backfill at startup

**Files:**
- Modify: `backend/ml_pipeline/_execution/attempts.py` (add `backfill_missing_attempts`)
- Modify: `backend/main.py:244-246` (call the backfill after stale-job reset)
- Test: `tests/test_attempt_backfill.py`

**Interfaces:**
- Produces: `def backfill_missing_attempts(session: Session) -> int` — creates a single attempt #1 for every `training_jobs` row that has no attempt; idempotent; returns the number created.
- Consumes: `ExecutionAttempt`, `AttemptStatus`, `TrainingJob`, existing sync session (`backend.database.sync_session.get_sync_session`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_attempt_backfill.py`:

```python
"""The backfill creates one attempt for pre-existing jobs and is idempotent."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.database.models import Base, ExecutionAttempt, TrainingJob
from backend.ml_pipeline._execution.attempts import backfill_missing_attempts


def _job(session, job_id, status, celery_task_id=None):
    """Insert a training job row with optional stored external id."""
    meta = {"celery_task_id": celery_task_id} if celery_task_id else None
    session.add(TrainingJob(id=job_id, pipeline_id="p", node_id="n", dataset_source_id="d",
                            status=status, run_mode="fixed", model_type="rf", graph={},
                            job_metadata=meta))


def test_backfill_creates_one_attempt_per_job_and_is_idempotent():
    """Legacy jobs get attempt #1 mirroring their status; re-runs add nothing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    _job(session, "j-done", "completed")
    _job(session, "j-run", "running", celery_task_id="celery-5")
    session.commit()

    created = backfill_missing_attempts(session)
    assert created == 2

    done = session.query(ExecutionAttempt).filter_by(job_id="j-done").one()
    assert done.attempt_number == 1 and done.status == "succeeded" and done.is_final is True
    run = session.query(ExecutionAttempt).filter_by(job_id="j-run").one()
    assert run.status == "running" and run.external_execution_id == "celery-5"

    assert backfill_missing_attempts(session) == 0  # idempotent
    session.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_attempt_backfill.py -v`
Expected: FAIL — `ImportError: cannot import name 'backfill_missing_attempts'`.

- [ ] **Step 3: Implement the backfill (append to `attempts.py`)**

```python
from backend.database.models import TrainingJob

_JOB_TO_ATTEMPT_STATUS: dict[str, AttemptStatus] = {
    "completed": AttemptStatus.SUCCEEDED,
    "succeeded": AttemptStatus.SUCCEEDED,
    "failed": AttemptStatus.FAILED,
    "cancelled": AttemptStatus.CANCELLED,
    "cancel_requested": AttemptStatus.CANCEL_REQUESTED,
    "submission_failed": AttemptStatus.SUBMISSION_FAILED,
    "running": AttemptStatus.RUNNING,
    "queued": AttemptStatus.QUEUED,
}


def backfill_missing_attempts(session: Session) -> int:
    """Create attempt #1 for every job lacking any attempt; return count created.

    Mirrors the job's current status onto the synthesized attempt and carries
    over any stored ``celery_task_id`` as the external execution id. Idempotent:
    jobs that already have an attempt are skipped.
    """
    existing_job_ids = {row[0] for row in session.query(ExecutionAttempt.job_id).distinct().all()}
    created = 0
    for job in session.query(TrainingJob).all():
        if job.id in existing_job_ids:
            continue
        status = _JOB_TO_ATTEMPT_STATUS.get(str(job.status), AttemptStatus.QUEUED)
        meta = job.job_metadata if isinstance(job.job_metadata, dict) else {}
        session.add(
            ExecutionAttempt(
                id=str(uuid.uuid4()),
                job_id=job.id,
                attempt_number=1,
                backend=str(meta.get("backend", "celery")),
                external_execution_id=meta.get("celery_task_id"),
                status=status.value,
                is_final=status is AttemptStatus.SUCCEEDED,
                started_at=job.started_at,
                finished_at=job.finished_at,
            )
        )
        created += 1
    if created:
        session.commit()
    return created
```

- [ ] **Step 4: Call the backfill at startup**

In `backend/main.py`, inside `lifespan`, after `_reset_stale_jobs()`:

```python
    # Backfill execution attempts for jobs created before the attempt table
    # existed. Idempotent and best-effort — never block startup.
    try:
        from backend.database.sync_session import get_sync_session
        from backend.ml_pipeline._execution.attempts import backfill_missing_attempts

        _session = get_sync_session()
        try:
            created = backfill_missing_attempts(_session)
            if created:
                logger.info("Backfilled %d execution attempt(s)", created)
        finally:
            _session.close()
    except Exception as exc:
        logger.warning("Attempt backfill skipped: %s", exc)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_attempt_backfill.py tests/test_main_stale_job_cutoff.py -q`
Expected: PASS.

- [ ] **Step 6: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/attempts.py backend/main.py tests/test_attempt_backfill.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add backend/ml_pipeline/_execution/attempts.py backend/main.py tests/test_attempt_backfill.py
git commit -m "feat(execution): backfill attempts for pre-existing jobs on startup

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: Frontend status sync (backend↔frontend rule)

**Files:**
- Modify: `frontend/ml-canvas/src/core/api/jobs.ts:4` (`JobStatus` union)
- Modify: `frontend/ml-canvas/src/components/shared/StatusBadge.tsx:4-17,36-45` (union + `RESOLVE` map)
- Modify: `frontend/ml-canvas/src/core/hooks/useJobPolling.ts:15-20` (`TERMINAL_STATUSES`)
- Test: `frontend/ml-canvas/src/components/shared/StatusBadge.test.tsx`, `frontend/ml-canvas/src/core/hooks/useJobPolling.test.ts`

**Interfaces:**
- The backend now returns `cancel_requested`, `submission_failed`, and `retrying` in `JobInfo.status`. The frontend must render them, treat `submission_failed` as terminal, and keep polling on `cancel_requested`/`retrying`.

- [ ] **Step 1: Write the failing frontend tests**

In `frontend/ml-canvas/src/components/shared/StatusBadge.test.tsx`, add cases to the first `it` block:

```tsx
      ['cancel_requested', 'Cancelling'],
      ['submission_failed', 'Submit Failed'],
      ['retrying', 'Retrying'],
```

In `frontend/ml-canvas/src/core/hooks/useJobPolling.test.ts`, extend the terminal test:

```ts
  it('treats submission_failed as terminal but not cancel_requested/retrying', () => {
    expect(isTerminalStatus('submission_failed')).toBe(true);
    expect(isTerminalStatus('cancel_requested')).toBe(false);
    expect(isTerminalStatus('retrying')).toBe(false);
  });
```

- [ ] **Step 2: Run the frontend tests to verify they fail**

Run (from `frontend/ml-canvas/`): `npm run test -- StatusBadge useJobPolling`
Expected: FAIL — `Cancelling`/`Submit Failed`/`Retrying` not rendered; `submission_failed` not terminal.

- [ ] **Step 3: Extend the `JobStatus` union in `jobs.ts`**

```ts
export type JobStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | 'pending'
  | 'cancel_requested'
  | 'submission_failed'
  | 'retrying';
```

- [ ] **Step 4: Extend `StatusBadge.tsx`**

Add the three members to its local `JobStatus` union (after `'CANCELLED'`):

```tsx
  | 'cancel_requested'
  | 'submission_failed'
  | 'retrying'
```

Add to the `RESOLVE` map (reuse existing lucide icons `Loader2`, `Ban`, `XCircle`):

```tsx
  cancel_requested:  { label: 'Cancelling',   classes: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400 border-amber-200 dark:border-amber-800',  Icon: Loader2, spin: true },
  submission_failed: { label: 'Submit Failed', classes: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400 border-red-200 dark:border-red-800',              Icon: XCircle },
  retrying:          { label: 'Retrying',      classes: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 border-blue-200 dark:border-blue-800',        Icon: Loader2, spin: true },
```

- [ ] **Step 5: Extend `TERMINAL_STATUSES` in `useJobPolling.ts`**

```ts
const TERMINAL_STATUSES: ReadonlySet<JobStatus> = new Set<JobStatus>([
  'completed',
  'succeeded',
  'failed',
  'cancelled',
  'submission_failed',
]);
```

Do not add `cancel_requested` or `retrying` — the UI must keep polling until they resolve to a terminal state.

- [ ] **Step 6: Run the frontend tests to verify they pass**

Run (from `frontend/ml-canvas/`): `npm run test -- StatusBadge useJobPolling`
Expected: PASS.

- [ ] **Step 7: Frontend static gate**

Run (from `frontend/ml-canvas/`):
```bash
npm run lint
npx tsc --project tsconfig.json --noEmit
npm run build
```
Expected: no lint errors (`--max-warnings 0`), no type errors, successful build.

- [ ] **Step 8: Commit**

```bash
git add frontend/ml-canvas/src/core/api/jobs.ts frontend/ml-canvas/src/components/shared/StatusBadge.tsx frontend/ml-canvas/src/components/shared/StatusBadge.test.tsx frontend/ml-canvas/src/core/hooks/useJobPolling.ts frontend/ml-canvas/src/core/hooks/useJobPolling.test.ts
git commit -m "feat(frontend): render cancel_requested/submission_failed/retrying statuses

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 8: Lifecycle gate — regression + docs

**Files:**
- Modify: `backend/ml_pipeline/METRICS_AND_ARTIFACTS.md` (document the attempt lifecycle) — optional doc; no test.

- [ ] **Step 1: Document the attempt model**

Append a short "Execution attempts" section to `backend/ml_pipeline/METRICS_AND_ARTIFACTS.md` describing: one `execution_attempts` row per physical run, `is_final` marks the promoted attempt, external id lives on the attempt, and the `cancel_requested → cancelled` / `submission_failed` / `retrying` statuses.

- [ ] **Step 2: Run the lifecycle regression subset**

Run:
```bash
pytest tests/test_execution_attempts_model.py tests/test_execution_attempts_repository.py \
  tests/test_retry_policy.py tests/test_cancellation_two_phase.py tests/test_attempt_backfill.py \
  tests/test_jobs_router_retry.py tests/test_pipeline_task.py tests/test_execution_backend_dispatch.py \
  tests/test_job_manager_base.py tests/test_ml_pipeline_backend_fixes.py -q
```
Expected: PASS (all).

- [ ] **Step 3: Full static gate**

Run: `ruff check .`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add backend/ml_pipeline/METRICS_AND_ARTIFACTS.md
git commit -m "docs(ml): document execution attempt lifecycle

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Definition of Done (Lifecycle Gate)

- `execution_attempts` exists as a separate table added by `create_all`; `training_jobs` is unchanged; pre-existing jobs are backfilled idempotently at startup.
- Every job creates attempt #1 on submit; the external execution id is stored on the attempt; submit failures set `submission_failed` (no silent drop).
- Cancellation is two-phase: queued jobs confirm `cancelled` immediately, running jobs enter `cancel_requested` and are confirmed by the driver; the late-write guard allows only `cancel_requested → cancelled`, never revival to `completed`.
- Retry appends a new attempt to the same logical job, returns the same `job_id`, dedupes concurrent retries, and preserves the endpoint's 404/400 guard rules.
- Failures are classified per attempt (`error_class`); automatic retry eligibility is defined and bounded by `MAX_AUTO_RETRIES` (scheduling wired in plan 05).
- The frontend renders the three new statuses, treats `submission_failed` as terminal, and keeps polling on `cancel_requested`/`retrying`; frontend lint/tsc/build are green.
- Full backend `ruff` / `ruff format --check` / `ty` gate is green.
