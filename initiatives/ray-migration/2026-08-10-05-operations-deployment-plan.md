# Operations and Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Ray execution operable: reconcile durable DB attempts against the Ray control plane, replace Celery Beat with a standalone scheduler that also runs reconciliation (guarded by PostgreSQL advisory locks), keep health checks topology-agnostic, ship a Docker Compose Ray stack with a private dashboard, correlate Ray Jobs to Skyulf ids for observability, and prove worker/head-failure recovery with no silent fallback.

**Architecture:** A synchronous reconciliation engine walks active attempts,
resolves the adapter recorded in each attempt's `backend` column, asks that
adapter for `status`/`logs`, and applies the design's control-plane-state →
DB-state table (running, succeeded/inconsistent, failed, stopped,
missing/orphaned) — auto-retrying only bounded, transient failures. A small
scheduler process runs periodic maintenance (error-log cleanup,
reconciliation) instead of Celery Beat, using `pg_try_advisory_lock` so
replicated schedulers never double-run. The Docker Compose Ray profile wires
PostgreSQL, MinIO (S3), Redis (event bus), a Ray head + worker, the API, and
the scheduler, with the Ray dashboard bound to localhost only.

**Tech Stack:** Python 3.12, SQLAlchemy sync sessions, PostgreSQL advisory locks, Ray Jobs (via the plan-03 adapter), Docker Compose, MinIO, pytest with mocked backend/status.

## Global Constraints

- Reconciliation and the scheduler run on **synchronous** DB sessions (`backend.database.sync_session`) so they can run outside the FastAPI event loop.
- Reconciliation acts only on attempts with a durable external control plane (Ray/Celery); it is a no-op for the local backend (startup `_reset_stale_jobs` already handles local orphans). Cluster/head loss must never leave a job indefinitely running.
- Automatic retry is bounded by `MAX_AUTO_RETRIES` and applies only to transient failure classes (worker/node loss, transient I/O). Invalid config, OOM, and user cancellation are never auto-retried.
- Health endpoints stay **topology-agnostic**: `/health/detailed` reports a single aggregate boolean and never names backends, addresses, or ports (the endpoint is unauthenticated).
- The Ray dashboard and Jobs API are never public endpoints; Compose binds the dashboard to `127.0.0.1` only. The API and Ray workers use the same versioned image. Runtime package installation stays disabled.
- Production Ray never silently falls back to in-process execution; a submission failure surfaces as `submission_failed`.
- Preserve the plan 01–04 contracts, the DB-as-truth model, the cancellation late-write guard, the local fallback, and the Celery rollback path (the scheduler is additive; Celery Beat is removed only in plan 06).
- No public API path/response-model changes and no config/response shape exposed to the frontend changes in this plan, so no frontend files are touched.
- Target Python 3.12 idioms and full typing; avoid `Any` where a concrete type exists. Every new function/method has a 1–2 line docstring.
- Every implementation task follows TDD and ends with a focused commit whose message includes:
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`
- After Python changes run, in order:
  - `ruff check .`
  - `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
  - `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
- Depends on plans 01–04.

---

## File Structure

Create:

- `backend/ml_pipeline/_execution/reconciliation.py` — `ReconcileReport`, `reconcile_attempts`, and the sync auto-retry helper.
- `backend/monitoring/maintenance.py` — neutral `delete_error_events_older_than` extracted from the Celery task.
- `backend/scheduler/__init__.py`, `backend/scheduler/locks.py`, `backend/scheduler/runner.py` — advisory-locked periodic scheduler.
- `backend/health/execution.py` — `check_execution_backend_healthy`.
- `run_scheduler.py` — scheduler process entrypoint (repo root, mirrors `celery_worker.py`).
- `docker-compose.ray.yml` — Ray deployment stack.
- `tests/test_reconciliation.py`, `tests/test_scheduler_locks.py`, `tests/test_scheduler_runner.py`, `tests/test_health_execution_backend.py`, `tests/test_ray_failure_modes.py`, `tests/test_monitoring_maintenance.py`.

Modify:

- `backend/ml_pipeline/_execution/attempts.py` — add sync repository helpers used by reconciliation.
- `backend/monitoring/tasks.py` — delegate to `maintenance.delete_error_events_older_than`.
- `backend/health/routes.py:60-102` — fold execution-backend readiness into `dependencies_healthy`.
- `backend/ml_pipeline/_execution/backends/ray.py` — enrich submit metadata with `pipeline_id`.
- `backend/config/mixins/execution.py` — add scheduler interval settings.

---

### Task 1: Reconciliation engine

**Files:**
- Create: `backend/ml_pipeline/_execution/reconciliation.py`
- Modify: `backend/ml_pipeline/_execution/attempts.py` (add sync helpers)
- Test: `tests/test_reconciliation.py`

**Interfaces:**
- Consumes: `ExecutionBackend.status`/`logs` (plan 01), `ExecutionState` (plan 01), `ExecutionAttempt`/`TrainingJob`, `AttemptStatus`/`JobStatus`, `classify_failure`/`FailureClass` (plan 02), `Settings.MAX_AUTO_RETRIES`.
- Produces:
  - `@dataclass class ReconcileReport` — counters: `checked`, `updated_running`, `marked_failed`, `marked_cancelled`, `inconsistent`, `orphaned`, `auto_retried`.
  - `def reconcile_attempts(session: Session, settings, backend_resolver: Callable[[str], ExecutionBackend] = get_execution_backend_by_name) -> ReconcileReport`.
  - Sync repo helpers: `next_attempt_number_sync`, `create_retry_attempt_sync`, `record_external_id_sync`, `active_attempts_with_external_id_sync`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_reconciliation.py`:

```python
"""Reconciliation maps backend state to durable DB state (design §9)."""

from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.config.base import Settings
from backend.database.models import Base, ExecutionAttempt, TrainingJob
from backend.ml_pipeline._execution.backends.base import ExecutionState
from backend.ml_pipeline._execution.reconciliation import reconcile_attempts


def _settings(**env: object) -> Settings:
    """Settings with a valid secret and overrides."""
    return Settings(SECRET_KEY="x" * 32, MAX_AUTO_RETRIES=1, **env)


def _seed(
    session,
    *,
    job_status,
    attempt_status,
    external_id="ray-1",
    number=1,
    graph=None,
    backend_name="ray",
):
    """Insert one job + attempt in the given states."""
    session.add(TrainingJob(id="j1", pipeline_id="p", node_id="n", dataset_source_id="d",
                            status=job_status, run_mode="fixed", model_type="rf",
                            graph=graph or {"pipeline_id": "p", "nodes": [
                                {"node_id": "n", "step_type": "training", "params": {}, "inputs": []}],
                                "metadata": {}}))
    session.add(ExecutionAttempt(id="a1", job_id="j1", attempt_number=number, backend=backend_name,
                                 external_execution_id=external_id, status=attempt_status, is_final=False))
    session.commit()


def _session():
    """Build a sync in-memory session with tables created."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_running_ray_job_updates_queued_to_running():
    """A RUNNING Ray Job flips a queued attempt/job to running."""
    session = _session()
    _seed(session, job_status="queued", attempt_status="queued")
    backend = MagicMock(name="ray")
    backend.name = "ray"
    backend.status.return_value = ExecutionState.RUNNING
    report = reconcile_attempts(session, _settings(), lambda _name: backend)
    assert report.updated_running == 1
    assert session.get(TrainingJob, "j1").status == "running"
    session.close()


def test_stopped_ray_job_confirms_cancelled():
    """A STOPPED Ray Job confirms a cancel_requested job as cancelled."""
    session = _session()
    _seed(session, job_status="cancel_requested", attempt_status="cancel_requested")
    backend = MagicMock(name="ray")
    backend.name = "ray"
    backend.status.return_value = ExecutionState.STOPPED
    report = reconcile_attempts(session, _settings(), lambda _name: backend)
    assert report.marked_cancelled == 1
    assert session.get(TrainingJob, "j1").status == "cancelled"
    session.close()


def test_missing_ray_job_orphans_and_auto_retries_worker_loss():
    """A MISSING Ray Job for a running attempt orphans it and auto-retries once."""
    session = _session()
    _seed(session, job_status="running", attempt_status="running")
    backend = MagicMock(name="ray")
    backend.name = "ray"
    backend.status.return_value = ExecutionState.MISSING
    backend.submit.return_value.external_execution_id = "ray-2"
    report = reconcile_attempts(session, _settings(), lambda _name: backend)
    assert report.orphaned == 1
    assert report.auto_retried == 1
    # A new attempt #2 was created and submitted.
    assert session.query(ExecutionAttempt).filter_by(job_id="j1").count() == 2
    backend.submit.assert_called_once()
    session.close()


def test_failed_invalid_config_is_not_retried():
    """A FAILED Ray Job with a config error marks failed and does not retry."""
    session = _session()
    _seed(session, job_status="running", attempt_status="running")
    backend = MagicMock(name="ray")
    backend.name = "ray"
    backend.status.return_value = ExecutionState.FAILED
    backend.logs.return_value = "ValueError: invalid hyperparameter"
    report = reconcile_attempts(session, _settings(), lambda _name: backend)
    assert report.marked_failed == 1
    assert report.auto_retried == 0
    assert session.get(TrainingJob, "j1").status == "failed"
    session.close()


def test_local_backend_is_noop():
    """Reconciliation does nothing for the local backend (no external truth)."""
    session = _session()
    _seed(
        session,
        job_status="running",
        attempt_status="running",
        backend_name="local",
    )
    backend = MagicMock()
    backend.name = "local"
    report = reconcile_attempts(session, _settings(), lambda _name: backend)
    assert report.checked == 0
    session.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reconciliation.py -v`
Expected: FAIL — `ModuleNotFoundError: ...reconciliation`.

- [ ] **Step 3: Add sync repository helpers (append to `attempts.py`)**

```python
    @staticmethod
    def active_attempts_with_external_id_sync(session: Session) -> list[ExecutionAttempt]:
        """Return active attempts that carry an external execution id (sync)."""
        return (
            session.query(ExecutionAttempt)
            .filter(
                ExecutionAttempt.status.in_(list(_ACTIVE_ATTEMPT_STATUSES)),
                ExecutionAttempt.external_execution_id.isnot(None),
            )
            .all()
        )

    @staticmethod
    def next_attempt_number_sync(session: Session, job_id: str) -> int:
        """Return the next 1-based attempt number for a job (sync)."""
        current = (
            session.query(func.max(ExecutionAttempt.attempt_number))
            .filter(ExecutionAttempt.job_id == job_id)
            .scalar()
        )
        return int(current or 0) + 1

    @staticmethod
    def create_retry_attempt_sync(session: Session, job_id: str, backend: str) -> str:
        """Create the next queued attempt for a job (sync) and return its id."""
        attempt_id = str(uuid.uuid4())
        number = ExecutionAttemptRepository.next_attempt_number_sync(session, job_id)
        session.add(
            ExecutionAttempt(
                id=attempt_id, job_id=job_id, attempt_number=number, backend=backend,
                status=AttemptStatus.QUEUED.value, is_final=False,
            )
        )
        session.commit()
        return attempt_id

    @staticmethod
    def record_external_id_sync(session: Session, attempt_id: str, external_execution_id: str) -> None:
        """Store the external id on an attempt (sync)."""
        attempt = session.get(ExecutionAttempt, attempt_id)
        if attempt is not None:
            attempt.external_execution_id = external_execution_id
            session.commit()
```

- [ ] **Step 4: Implement the reconciliation engine**

Create `backend/ml_pipeline/_execution/reconciliation.py`:

```python
"""Reconcile durable DB attempts against the execution backend (design §9).

Runs on a sync session in the scheduler. For each active attempt with an
external id, asks the backend for its state and applies the Ray-state → DB-state
table. Only transient failures/orphans are auto-retried, bounded by
``MAX_AUTO_RETRIES``. Cluster/head loss can never leave a job running forever:
a MISSING external job orphans the attempt and either retries or fails it.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from backend.database.models import ExecutionAttempt, TrainingJob
from backend.ml_pipeline._execution.attempts import AttemptStatus, ExecutionAttemptRepository
from backend.ml_pipeline._execution.backends.base import (
    ExecutionBackend,
    ExecutionRequest,
    ExecutionState,
)
from backend.ml_pipeline._execution.retry_policy import classify_failure
from backend.ml_pipeline._execution.schemas import JobStatus

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ReconcileReport:
    """Counters describing one reconciliation pass."""

    checked: int = 0
    updated_running: int = 0
    marked_failed: int = 0
    marked_cancelled: int = 0
    inconsistent: int = 0
    orphaned: int = 0
    auto_retried: int = 0


def _payload_from_job(job: TrainingJob) -> dict:
    """Rebuild the submission payload from the job's stored graph."""
    graph = job.graph if isinstance(job.graph, dict) else {}
    return {
        "pipeline_id": graph.get("pipeline_id", job.pipeline_id),
        "nodes": graph.get("nodes", []),
        "metadata": graph.get("metadata", {}),
        "job_type": "tuning" if job.run_mode == "tuned" else "training",
    }


def _auto_retry(
    session: Session, backend: ExecutionBackend, job: TrainingJob, attempt: ExecutionAttempt, settings
) -> bool:
    """Submit a bounded retry for a transient failure/orphan; return whether it retried."""
    if attempt.attempt_number > settings.MAX_AUTO_RETRIES:
        return False
    new_attempt_id = ExecutionAttemptRepository.create_retry_attempt_sync(session, job.id, backend.name)
    job.status = JobStatus.QUEUED.value
    job.error_message = None
    job.finished_at = None
    session.commit()
    handle = backend.submit(
        ExecutionRequest(job_id=job.id, attempt_id=new_attempt_id, payload=_payload_from_job(job))
    )
    ExecutionAttemptRepository.record_external_id_sync(session, new_attempt_id, handle.external_execution_id)
    return True


def _mark_failed(session: Session, job: TrainingJob, attempt: ExecutionAttempt, error_message: str) -> str:
    """Mark the attempt+job failed and return the failure class."""
    decision = classify_failure(error_message)
    attempt.status = AttemptStatus.FAILED.value
    attempt.error_class = decision.failure_class.value
    attempt.error_message = error_message[:2000]
    attempt.finished_at = datetime.now(UTC)
    job.status = JobStatus.FAILED.value
    job.error_message = error_message[:2000]
    job.finished_at = datetime.now(UTC)
    session.commit()
    return decision.failure_class.value


def reconcile_attempts(
    session: Session,
    settings,
    backend_resolver: Callable[[str], ExecutionBackend] = get_execution_backend_by_name,
) -> ReconcileReport:
    """Compare active attempts to their persisted backend and converge the DB."""
    report = ReconcileReport()

    for attempt in ExecutionAttemptRepository.active_attempts_with_external_id_sync(session):
        if attempt.backend == "local":
            continue  # no durable external control plane to reconcile against
        backend = backend_resolver(attempt.backend)
        job = session.get(TrainingJob, attempt.job_id)
        if job is None:
            continue
        report.checked += 1
        state = backend.status(attempt.external_execution_id or "")

        if state is ExecutionState.RUNNING and attempt.status == AttemptStatus.QUEUED.value:
            attempt.status = AttemptStatus.RUNNING.value
            attempt.started_at = attempt.started_at or datetime.now(UTC)
            if job.status == JobStatus.QUEUED.value:
                job.status = JobStatus.RUNNING.value
            session.commit()
            report.updated_running += 1

        elif state is ExecutionState.SUCCEEDED and job.status != JobStatus.COMPLETED.value:
            # Backend says done but the driver never finalized the DB — do not
            # silently complete; flag as inconsistent for operator follow-up.
            logger.warning("Reconcile: job %s SUCCEEDED on backend but not finalized", job.id)
            report.inconsistent += 1

        elif state is ExecutionState.FAILED:
            error_text = backend.logs(attempt.external_execution_id or "") or "Execution failed"
            failure_class = _mark_failed(session, job, attempt, error_text)
            report.marked_failed += 1
            if classify_failure(error_text).retriable and _auto_retry(session, backend, job, attempt, settings):
                report.auto_retried += 1

        elif state is ExecutionState.STOPPED and job.status in (
            JobStatus.CANCEL_REQUESTED.value, JobStatus.CANCELLED.value,
        ):
            attempt.status = AttemptStatus.CANCELLED.value
            attempt.finished_at = datetime.now(UTC)
            job.status = JobStatus.CANCELLED.value
            job.finished_at = datetime.now(UTC)
            session.commit()
            report.marked_cancelled += 1

        elif state is ExecutionState.MISSING:
            report.orphaned += 1
            marker = "Execution disappeared from the cluster (worker/node loss)."
            _mark_failed(session, job, attempt, marker)
            if _auto_retry(session, backend, job, attempt, settings):
                report.auto_retried += 1

    return report
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_reconciliation.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/reconciliation.py backend/ml_pipeline/_execution/attempts.py tests/test_reconciliation.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add backend/ml_pipeline/_execution/reconciliation.py backend/ml_pipeline/_execution/attempts.py tests/test_reconciliation.py
git commit -m "feat(ops): reconcile durable attempts against the execution backend

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Advisory-locked scheduler (Celery Beat replacement)

**Files:**
- Create: `backend/monitoring/maintenance.py`
- Modify: `backend/monitoring/tasks.py:14-41` (delegate to the neutral function)
- Create: `backend/scheduler/__init__.py`, `backend/scheduler/locks.py`, `backend/scheduler/runner.py`
- Create: `run_scheduler.py`
- Modify: `backend/config/mixins/execution.py` (scheduler intervals)
- Test: `tests/test_monitoring_maintenance.py`, `tests/test_scheduler_locks.py`, `tests/test_scheduler_runner.py`

**Interfaces:**
- Produces:
  - `def delete_error_events_older_than(session: Session, cutoff: datetime) -> int`.
  - `def advisory_lock(session: Session, key: int) -> AbstractContextManager[bool]` (Postgres `pg_try_advisory_lock`; always-true on SQLite).
  - `@dataclass class ScheduledJob` — `name: str`, `interval_seconds: int`, `run: Callable[[Session], None]`, `lock_key: int`.
  - `class SchedulerRunner` — `run_due(now: float) -> list[str]` (returns names of jobs run) and `run_forever()`.
  - Settings `SCHEDULER_RECONCILE_INTERVAL_SECONDS`, `SCHEDULER_CLEANUP_INTERVAL_SECONDS`.

- [ ] **Step 1: Write the failing test for the extracted maintenance function**

Create `tests/test_monitoring_maintenance.py`:

```python
"""The neutral error-event cleanup deletes rows older than the cutoff."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.database.models import Base, ErrorEvent
from backend.monitoring.maintenance import delete_error_events_older_than


def test_delete_error_events_older_than_cutoff():
    """Only rows created before the cutoff are deleted."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    old = ErrorEvent(exception_type="E", created_at=datetime.now(UTC) - timedelta(days=40))
    new = ErrorEvent(exception_type="E", created_at=datetime.now(UTC))
    session.add_all([old, new])
    session.commit()

    deleted = delete_error_events_older_than(session, datetime.now(UTC) - timedelta(days=30))
    assert deleted == 1
    assert session.query(ErrorEvent).count() == 1
    session.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitoring_maintenance.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.monitoring.maintenance`.

(If `ErrorEvent` requires additional non-null columns, inspect `backend/database/models.py` `ErrorEvent` and populate them in the test; the neutral function itself only depends on `created_at`.)

- [ ] **Step 3: Extract the neutral maintenance function**

Create `backend/monitoring/maintenance.py`:

```python
"""Neutral maintenance operations usable by both Celery and the scheduler.

Extracted from ``backend.monitoring.tasks`` so the standalone scheduler can run
the same logic the Celery Beat task used to run, without importing Celery.
"""

from datetime import datetime

from sqlalchemy import delete
from sqlalchemy.orm import Session

from backend.database.models import ErrorEvent


def delete_error_events_older_than(session: Session, cutoff: datetime) -> int:
    """Delete error_events created before ``cutoff``; return the number removed."""
    result = session.execute(delete(ErrorEvent).where(ErrorEvent.created_at < cutoff))
    session.commit()
    return int(result.rowcount or 0)
```

Update `backend/monitoring/tasks.py`'s `cleanup_error_events` to delegate (keeping the Celery task for rollback):

```python
    async def _run() -> int:
        from backend.monitoring.maintenance import delete_error_events_older_than

        if not async_session_factory:
            raise RuntimeError("Database not initialized")
        # The neutral helper is sync; run it against a fresh sync session.
        from backend.database.sync_session import get_sync_session

        sync_session = get_sync_session()
        try:
            return delete_error_events_older_than(sync_session, cutoff)
        finally:
            sync_session.close()
```

- [ ] **Step 4: Write the failing advisory-lock and runner tests**

Create `tests/test_scheduler_locks.py`:

```python
"""Advisory lock is a no-op-true on SQLite and delegates to pg_try_advisory_lock on Postgres."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.scheduler.locks import advisory_lock


def test_sqlite_lock_always_acquires():
    """On SQLite (single process) the lock always acquires."""
    engine = create_engine("sqlite:///:memory:")
    session = Session(engine)
    with advisory_lock(session, 12345) as acquired:
        assert acquired is True
    session.close()
```

Create `tests/test_scheduler_runner.py`:

```python
"""The scheduler runs due jobs, honoring interval and advisory-lock gating."""

from unittest.mock import MagicMock

from backend.scheduler.runner import ScheduledJob, SchedulerRunner


def test_runs_job_when_interval_elapsed():
    """A job whose interval elapsed runs; a not-yet-due job does not."""
    ran = []
    jobs = [
        ScheduledJob(name="a", interval_seconds=10, run=lambda _s: ran.append("a"), lock_key=1),
        ScheduledJob(name="b", interval_seconds=1000, run=lambda _s: ran.append("b"), lock_key=2),
    ]
    session_factory = MagicMock(return_value=MagicMock())
    runner = SchedulerRunner(jobs, session_factory=session_factory, clock=lambda: 0.0)
    # First tick at t=0 runs both (never run before).
    assert set(runner.run_due(0.0)) == {"a", "b"}
    # At t=5, only "a" (interval 10) is not yet due again; nothing runs.
    assert runner.run_due(5.0) == []
    # At t=11, "a" is due again.
    assert runner.run_due(11.0) == ["a"]
```

- [ ] **Step 5: Run the failing tests**

Run: `pytest tests/test_scheduler_locks.py tests/test_scheduler_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.scheduler...`.

- [ ] **Step 6: Implement the scheduler package**

Create `backend/scheduler/__init__.py`:

```python
"""Standalone periodic scheduler (replaces Celery Beat for Ray deployments)."""
```

Create `backend/scheduler/locks.py`:

```python
"""Cross-replica advisory locking for scheduled jobs.

On PostgreSQL a session-level ``pg_try_advisory_lock`` ensures only one
scheduler replica runs a given job at a time. On SQLite (single-process dev)
the lock always acquires.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.orm import Session


@contextmanager
def advisory_lock(session: Session, key: int) -> Iterator[bool]:
    """Acquire a best-effort advisory lock for ``key``; yield whether acquired."""
    dialect = session.bind.dialect.name if session.bind is not None else "sqlite"
    if dialect != "postgresql":
        yield True
        return
    acquired = bool(session.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key}).scalar())
    try:
        yield acquired
    finally:
        if acquired:
            session.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
            session.commit()
```

Create `backend/scheduler/runner.py`:

```python
"""Interval scheduler that runs maintenance and reconciliation jobs.

Each job carries an interval and an advisory-lock key so replicated schedulers
never double-run it. Jobs run on fresh sync sessions.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from backend.scheduler.locks import advisory_lock

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScheduledJob:
    """A periodic job with its interval and advisory-lock key."""

    name: str
    interval_seconds: int
    run: Callable[[Session], None]
    lock_key: int
    _last_run: float = field(default=-1.0, init=False)


class SchedulerRunner:
    """Run due scheduled jobs on sync sessions, guarded by advisory locks."""

    def __init__(
        self,
        jobs: list[ScheduledJob],
        session_factory: Callable[[], Session],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Store the jobs, a sync session factory, and a monotonic clock."""
        self._jobs = jobs
        self._session_factory = session_factory
        self._clock = clock

    def run_due(self, now: float) -> list[str]:
        """Run every job whose interval has elapsed; return the names that ran."""
        ran: list[str] = []
        for job in self._jobs:
            if job._last_run >= 0 and (now - job._last_run) < job.interval_seconds:
                continue
            session = self._session_factory()
            try:
                with advisory_lock(session, job.lock_key) as acquired:
                    if not acquired:
                        continue
                    job.run(session)
                    ran.append(job.name)
            except Exception:
                logger.exception("Scheduled job %s failed", job.name)
            finally:
                session.close()
            job._last_run = now
        return ran

    def run_forever(self, tick_seconds: float = 5.0) -> None:  # pragma: no cover - loop
        """Continuously run due jobs every ``tick_seconds``."""
        while True:
            self.run_due(self._clock())
            time.sleep(tick_seconds)
```

Add scheduler intervals to `backend/config/mixins/execution.py`:

```python
    # ── Scheduler (plan 05) ──────────────────────────────────────────────────
    # How often the standalone scheduler runs reconciliation and error-log cleanup.
    SCHEDULER_RECONCILE_INTERVAL_SECONDS: int = 30
    SCHEDULER_CLEANUP_INTERVAL_SECONDS: int = 86400
```

- [ ] **Step 7: Implement the scheduler entrypoint**

Create `run_scheduler.py` (repo root):

```python
"""Standalone scheduler process entrypoint.

Replaces Celery Beat for Ray deployments: runs reconciliation and error-log
cleanup on intervals, guarded by advisory locks so replicas don't double-run.
"""

import logging
from datetime import UTC, datetime, timedelta

from backend.config import get_settings
from backend.database.sync_session import get_sync_session, get_sync_session_factory
from backend.ml_pipeline._execution.backends.registry import get_execution_backend
from backend.ml_pipeline._execution.reconciliation import reconcile_attempts
from backend.monitoring.maintenance import delete_error_events_older_than
from backend.scheduler.runner import ScheduledJob, SchedulerRunner
from backend.utils.logging_utils import setup_universal_logging

logger = logging.getLogger(__name__)


def _reconcile(session) -> None:
    """Scheduler job: reconcile active attempts against the execution backend."""
    settings = get_settings()
    report = reconcile_attempts(session, settings)
    logger.info("Reconcile pass: %s", report)


def _cleanup(session) -> None:
    """Scheduler job: delete error events past the retention window."""
    settings = get_settings()
    cutoff = datetime.now(UTC) - timedelta(days=settings.ERROR_LOG_RETENTION_DAYS)
    delete_error_events_older_than(session, cutoff)


def build_runner() -> SchedulerRunner:
    """Construct the SchedulerRunner with the reconcile + cleanup jobs."""
    settings = get_settings()
    factory = get_sync_session_factory()
    jobs = [
        ScheduledJob("reconcile", settings.SCHEDULER_RECONCILE_INTERVAL_SECONDS, _reconcile, lock_key=1001),
        ScheduledJob("cleanup", settings.SCHEDULER_CLEANUP_INTERVAL_SECONDS, _cleanup, lock_key=1002),
    ]
    return SchedulerRunner(jobs, session_factory=lambda: factory())


def main() -> None:  # pragma: no cover - process entrypoint
    """Run the scheduler loop forever."""
    setup_universal_logging(log_file="logs/scheduler.log", log_level="INFO", console_log_level="INFO")
    _ = get_sync_session  # ensure module import side effects are exercised
    build_runner().run_forever()


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_monitoring_maintenance.py tests/test_scheduler_locks.py tests/test_scheduler_runner.py -q`
Expected: PASS.

- [ ] **Step 9: Static checks**

Run: `ruff check backend/monitoring/maintenance.py backend/monitoring/tasks.py backend/scheduler/ run_scheduler.py backend/config/mixins/execution.py tests/test_monitoring_maintenance.py tests/test_scheduler_locks.py tests/test_scheduler_runner.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors. (Note: `run_scheduler.py` is intentionally not in the `ruff format --check`/`ty` file lists, which target `run_skyulf.py`/`celery_worker.py`; still run `ruff check .` in the gate to cover it.)

- [ ] **Step 10: Commit**

```bash
git add backend/monitoring/maintenance.py backend/monitoring/tasks.py backend/scheduler/ run_scheduler.py backend/config/mixins/execution.py tests/test_monitoring_maintenance.py tests/test_scheduler_locks.py tests/test_scheduler_runner.py
git commit -m "feat(ops): add advisory-locked scheduler replacing Celery Beat

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Topology-agnostic execution-backend health check

**Files:**
- Create: `backend/health/execution.py`
- Modify: `backend/health/routes.py:60-102`
- Test: `tests/test_health_execution_backend.py`

**Interfaces:**
- Produces: `def check_execution_backend_healthy(settings) -> bool` — for Ray, pings the cluster through the adapter (returns bool); for local/celery, returns `True`. Never raises; never returns names/addresses.
- Consumes: `RayJobClient` (plan 03), `Settings.EXECUTION_BACKEND`, `Settings.RAY_ADDRESS`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_health_execution_backend.py`:

```python
"""Execution-backend health is a bare boolean and never leaks topology."""

import sys
import types
from unittest.mock import MagicMock

from backend.config.base import Settings
from backend.health.execution import check_execution_backend_healthy


def _settings(**env: object) -> Settings:
    """Settings with a valid secret and overrides."""
    return Settings(SECRET_KEY="x" * 32, **env)


def test_local_backend_is_healthy():
    """The local backend is always healthy (in-process)."""
    assert check_execution_backend_healthy(_settings(EXECUTION_BACKEND="local")) is True


def test_ray_unreachable_returns_false(monkeypatch):
    """An unreachable Ray cluster yields False, not an exception."""
    ray_pkg = types.ModuleType("ray")
    js = types.ModuleType("ray.job_submission")
    js.JobSubmissionClient = MagicMock(side_effect=ConnectionError("no route"))
    ray_pkg.job_submission = js
    monkeypatch.setitem(sys.modules, "ray", ray_pkg)
    monkeypatch.setitem(sys.modules, "ray.job_submission", js)
    assert check_execution_backend_healthy(
        _settings(EXECUTION_BACKEND="ray", RAY_ADDRESS="http://ray-head:8265")
    ) is False


def test_ray_reachable_returns_true(monkeypatch):
    """A reachable Ray cluster (list_jobs succeeds) yields True."""
    fake_client = MagicMock()
    fake_client.list_jobs.return_value = []
    ray_pkg = types.ModuleType("ray")
    js = types.ModuleType("ray.job_submission")
    js.JobSubmissionClient = MagicMock(return_value=fake_client)
    ray_pkg.job_submission = js
    monkeypatch.setitem(sys.modules, "ray", ray_pkg)
    monkeypatch.setitem(sys.modules, "ray.job_submission", js)
    assert check_execution_backend_healthy(
        _settings(EXECUTION_BACKEND="ray", RAY_ADDRESS="http://ray-head:8265")
    ) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_health_execution_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.health.execution`.

- [ ] **Step 3: Implement the check**

Create `backend/health/execution.py`:

```python
"""Topology-agnostic execution-backend readiness check.

Returns a bare boolean so the unauthenticated ``/health/detailed`` endpoint can
fold it into its aggregate without revealing which backend is configured or how
to reach it. Never raises.
"""

import logging

logger = logging.getLogger(__name__)


def check_execution_backend_healthy(settings) -> bool:
    """Return whether the configured execution backend is reachable (bool only)."""
    backend = settings.EXECUTION_BACKEND
    if backend != "ray":
        return True
    if not settings.RAY_ADDRESS:
        return False
    try:
        from ray.job_submission import JobSubmissionClient  # noqa: PLC0415

        client = JobSubmissionClient(settings.RAY_ADDRESS)
        client.list_jobs()
        return True
    except Exception:
        logger.debug("Execution backend health check failed", exc_info=True)
        return False
```

- [ ] **Step 4: Fold the check into `/health/detailed`**

In `backend/health/routes.py`, inside `detailed_health_check`, after the cache check, add:

```python
    # Execution-backend readiness (aggregate only — no backend/address disclosed).
    try:
        from backend.health.execution import check_execution_backend_healthy

        if not check_execution_backend_healthy(settings):
            dependencies_healthy = False
    except Exception:
        logging.getLogger(__name__).debug("Execution backend health check failed", exc_info=True)
        dependencies_healthy = False
```

The response model is unchanged (`dependencies_healthy: bool`) — no new field, no topology leaked.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_health_execution_backend.py -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Static checks**

Run: `ruff check backend/health/execution.py backend/health/routes.py tests/test_health_execution_backend.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add backend/health/execution.py backend/health/routes.py tests/test_health_execution_backend.py
git commit -m "feat(ops): add topology-agnostic execution-backend health check

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Observability — link Ray Jobs to Skyulf ids

**Files:**
- Modify: `backend/ml_pipeline/_execution/backends/ray.py` (submit metadata)
- Modify: `backend/ml_pipeline/_execution/backends/dispatch.py` (pass `pipeline_id` in payload)
- Test: `tests/test_ray_execution_backend.py` (extend)

**Interfaces:**
- Produces: Ray submission metadata gains `skyulf_pipeline_id` (from the payload) alongside `skyulf_job_id`/`skyulf_attempt_id`, so the operator (Ray Dashboard / Prometheus labels) can trace a Ray Job back to a Skyulf logical job/pipeline.

- [ ] **Step 1: Write the failing test (extend `tests/test_ray_execution_backend.py`)**

```python
def test_submit_metadata_includes_pipeline_id():
    """Submission metadata links the Ray Job to the Skyulf pipeline id."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.submit_job.return_value = "skyulf-j1-a1"
    backend = RayExecutionBackend(client=client, entrypoint_python="python")
    backend.submit(
        ExecutionRequest(job_id="j1", attempt_id="a1", payload={"pipeline_id": "pipe-42"})
    )
    metadata = client.submit_job.call_args.kwargs["metadata"]
    assert metadata["skyulf_pipeline_id"] == "pipe-42"
    assert metadata["skyulf_job_id"] == "j1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ray_execution_backend.py -v -k pipeline_id`
Expected: FAIL — metadata lacks `skyulf_pipeline_id`.

- [ ] **Step 3: Enrich the metadata**

In `backend/ml_pipeline/_execution/backends/ray.py`, build metadata including the pipeline id from the payload:

```python
        metadata = {
            "skyulf_job_id": request.job_id,
            "skyulf_attempt_id": request.attempt_id,
            "skyulf_pipeline_id": str(request.payload.get("pipeline_id", "")),
        }
        returned = self._client.submit_job(
            entrypoint,
            submission_id=submission_id,
            metadata=metadata,
            runtime_env={"env_vars": {"SKYULF_NUM_CPUS": str(spec.num_cpus)}},
            entrypoint_num_cpus=spec.num_cpus,
            entrypoint_num_gpus=spec.num_gpus or None,
            entrypoint_memory=spec.memory_mb,
        )
```

Ensure `dispatch_branches` includes `pipeline_id` in the payload it submits (the branch graph already carries `pipeline_id`; when the payload is the branch graph this is present — assert it is retained when adding `job_type`):

```python
        payload = {**payload, "job_type": job_type}
        payload.setdefault("pipeline_id", job_row.pipeline_id if job_row is not None else "")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ray_execution_backend.py -q`
Expected: PASS.

- [ ] **Step 5: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/backends/ray.py backend/ml_pipeline/_execution/backends/dispatch.py tests/test_ray_execution_backend.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add backend/ml_pipeline/_execution/backends/ray.py backend/ml_pipeline/_execution/backends/dispatch.py tests/test_ray_execution_backend.py
git commit -m "feat(obs): link Ray Jobs to Skyulf pipeline id via submission metadata

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Docker Compose Ray stack + private dashboard

**Files:**
- Create: `docker-compose.ray.yml`
- Test: `docker compose config` validation (no unit test)

**Interfaces:** none (deployment config). Services: `postgres`, `minio`, `redis`, `ray-head`, `ray-worker`, `api`, `scheduler`. Dashboard bound to `127.0.0.1:8265` only.

- [ ] **Step 1: Write `docker-compose.ray.yml`**

```yaml
# © 2025 Murat Unsal — Skyulf Project
# Ray execution stack: API + Ray head/worker + PostgreSQL + MinIO (S3) + Redis
# event bus + standalone scheduler. Ray dashboard is bound to localhost only.

name: skyulf-ray

x-app-image: &app-image
  build: .
  image: skyulf-app:local

x-ray-env: &ray-env
  FASTAPI_ENV: production
  EXECUTION_BACKEND: ray
  EVENT_BUS: redis
  EVENT_BUS_URL: redis://redis:6379/0
  RAY_ADDRESS: http://ray-head:8265
  DB_TYPE: postgres
  DATABASE_URL: postgresql+asyncpg://skyulf:skyulf@postgres:5432/skyulf
  S3_ARTIFACT_BUCKET: skyulf-artifacts
  AWS_ENDPOINT_URL: http://minio:9000
  AWS_ACCESS_KEY_ID: minioadmin
  AWS_SECRET_ACCESS_KEY: minioadmin
  AWS_DEFAULT_REGION: us-east-1

services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: skyulf
      POSTGRES_PASSWORD: skyulf
      POSTGRES_DB: skyulf
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: unless-stopped

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "127.0.0.1:9001:9001"   # console on localhost only
    volumes:
      - miniodata:/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  ray-head:
    <<: *app-image
    command: >-
      ray start --head --port=6379 --dashboard-host=0.0.0.0
      --dashboard-port=8265 --num-cpus=0 --block
    environment:
      <<: *ray-env
    ports:
      - "127.0.0.1:8265:8265"   # Ray dashboard/Jobs API — localhost only, never public
    depends_on:
      - postgres
      - minio
      - redis
    restart: unless-stopped

  ray-worker:
    <<: *app-image
    command: ray start --address=ray-head:6379 --num-cpus=4 --block
    environment:
      <<: *ray-env
    depends_on:
      - ray-head
    deploy:
      replicas: 2
    restart: unless-stopped

  api:
    <<: *app-image
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000
    environment:
      <<: *ray-env
    ports:
      - "8000:8000"
    depends_on:
      - ray-head
      - postgres
    restart: unless-stopped

  scheduler:
    <<: *app-image
    command: python run_scheduler.py
    environment:
      <<: *ray-env
    depends_on:
      - postgres
      - ray-head
    restart: unless-stopped

volumes:
  pgdata:
  miniodata:

networks:
  default:
    name: skyulf-ray-net
```

- [ ] **Step 2: Validate the compose file**

Run: `docker compose -f docker-compose.ray.yml config >/dev/null && echo OK`
Expected: prints `OK` (compose file is syntactically valid and interpolates). If Docker is unavailable in the execution environment, validate YAML syntax instead:
`python -c "import yaml,sys; yaml.safe_load(open('docker-compose.ray.yml')); print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Confirm the dashboard is not publicly exposed**

Run: `grep -n "8265" docker-compose.ray.yml`
Expected: the only published mapping is `127.0.0.1:8265:8265` (localhost-bound); no `0.0.0.0:8265` or bare `8265:8265` host mapping exists.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.ray.yml
git commit -m "feat(deploy): add Ray Compose stack with localhost-only dashboard

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: Failure-mode tests (worker crash, head loss, no silent fallback)

**Files:**
- Test: `tests/test_ray_failure_modes.py`

**Interfaces:** none new — asserts the reconciliation + submission behavior under failures.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ray_failure_modes.py`:

```python
"""Worker/head failure recovery and the no-silent-fallback guarantee."""

from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.config.base import Settings
from backend.database.models import Base, ExecutionAttempt, TrainingJob
from backend.ml_pipeline._execution.backends.base import ExecutionState
from backend.ml_pipeline._execution.reconciliation import reconcile_attempts


def _settings(**env: object) -> Settings:
    """Settings with a valid secret + one auto-retry allowed."""
    return Settings(SECRET_KEY="x" * 32, MAX_AUTO_RETRIES=1, **env)


def _session_with_running_attempt(number=1):
    """A running job whose single attempt is #`number`."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add(TrainingJob(id="j1", pipeline_id="p", node_id="n", dataset_source_id="d",
                            status="running", run_mode="fixed", model_type="rf",
                            graph={"pipeline_id": "p", "nodes": [], "metadata": {}}))
    session.add(ExecutionAttempt(id="a1", job_id="j1", attempt_number=number, backend="ray",
                                 external_execution_id="ray-1", status="running", is_final=False))
    session.commit()
    return session


def test_worker_crash_missing_job_is_recovered():
    """A crashed worker (MISSING Ray Job) never leaves the job running forever."""
    session = _session_with_running_attempt()
    backend = MagicMock(); backend.name = "ray"
    backend.status.return_value = ExecutionState.MISSING
    backend.submit.return_value.external_execution_id = "ray-2"
    report = reconcile_attempts(session, _settings(), lambda _name: backend)
    assert report.orphaned == 1
    job = session.get(TrainingJob, "j1")
    # Auto-retried (attempt #1 within cap) -> job requeued, not stuck running.
    assert job.status == "queued"
    session.close()


def test_head_loss_exhausted_retries_marks_failed():
    """Once retries are exhausted, an orphaned attempt terminates as failed."""
    session = _session_with_running_attempt(number=2)  # already the retry attempt
    backend = MagicMock(); backend.name = "ray"
    backend.status.return_value = ExecutionState.MISSING
    report = reconcile_attempts(session, _settings(MAX_AUTO_RETRIES=1), lambda _name: backend)
    assert report.orphaned == 1
    assert report.auto_retried == 0
    assert session.get(TrainingJob, "j1").status == "failed"
    session.close()


def test_no_silent_fallback_on_submission_failure():
    """A Ray submission failure surfaces as submission_failed, never local execution."""
    import asyncio

    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from unittest.mock import patch

    from backend.ml_pipeline._execution.attempts import ExecutionAttemptRepository
    from backend.ml_pipeline._execution.backends.dispatch import dispatch_branches
    from backend.ml_pipeline._execution.backends.ray import RayExecutionBackend

    async def _run():
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with maker() as session:
            session.add(TrainingJob(id="j9", pipeline_id="p", node_id="n", dataset_source_id="d",
                                    status="queued", run_mode="fixed", model_type="rf", graph={}))
            await session.commit()
            await ExecutionAttemptRepository.create_initial_attempt(session, "j9", "ray")
            client = MagicMock(); client.submit_job.side_effect = ConnectionError("head down")
            with patch(
                "backend.ml_pipeline._execution.backends.dispatch.get_execution_backend",
                return_value=RayExecutionBackend(client=client, entrypoint_python="python"),
            ):
                await dispatch_branches(
                    [("j9", {})],
                    settings=_settings(EXECUTION_BACKEND="ray", RAY_ADDRESS="http://h:8265"),
                    db=session,
                )
            job = await session.get(TrainingJob, "j9")
            assert job.status == "submission_failed"
        await engine.dispose()

    asyncio.run(_run())
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_ray_failure_modes.py -v`
Expected: PASS (3 passed). (These exercise plan-02 `dispatch_branches` submission-failure handling and plan-05 reconciliation; no new production code is required — if a test fails, fix the corresponding production path, not the test.)

- [ ] **Step 3: Static checks**

Run: `ruff check tests/test_ray_failure_modes.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ray_failure_modes.py
git commit -m "test(ops): cover worker/head failure recovery and no silent fallback

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: Operations gate — docs + regression

**Files:**
- Modify: `.env.example` (scheduler intervals)
- Modify: `docs/` operational note (optional — only if a deployment doc exists; otherwise skip)

- [ ] **Step 1: Document the scheduler env keys**

Append to `.env.example`:

```bash
# --- Scheduler (plan 05) ---
# The standalone scheduler (run_scheduler.py) replaces Celery Beat for Ray.
# SCHEDULER_RECONCILE_INTERVAL_SECONDS=30
# SCHEDULER_CLEANUP_INTERVAL_SECONDS=86400
```

- [ ] **Step 2: Run the operations regression subset**

Run:
```bash
pytest tests/test_reconciliation.py tests/test_scheduler_locks.py tests/test_scheduler_runner.py \
  tests/test_monitoring_maintenance.py tests/test_health_execution_backend.py \
  tests/test_ray_failure_modes.py tests/test_ray_execution_backend.py -q
```
Expected: PASS (all).

- [ ] **Step 3: Full static gate**

Run: `ruff check .`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add .env.example
git commit -m "docs(ops): document scheduler interval env keys

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Definition of Done (Operations Gate)

- Reconciliation maps every design-§9 Ray/DB state pair, auto-retries only bounded transient failures, and is a no-op for the local backend; cluster/head loss cannot leave a job running indefinitely.
- A standalone scheduler runs reconciliation + error-log cleanup on intervals, guarded by PostgreSQL advisory locks so replicas never double-run; the error-cleanup logic is shared with (not duplicated from) the still-present Celery task.
- `/health/detailed` folds execution-backend readiness into its single aggregate boolean — no backend, address, or port is disclosed.
- Ray submissions carry `skyulf_job_id`/`skyulf_attempt_id`/`skyulf_pipeline_id` metadata for operator correlation.
- `docker-compose.ray.yml` wires PostgreSQL, MinIO, Redis, Ray head/worker, API, and scheduler on one image, with the Ray dashboard bound to `127.0.0.1` only.
- Worker-crash, head-loss, and no-silent-fallback behaviors are covered by tests.
- Full backend `ruff` / `ruff format --check` / `ty` gate is green.
