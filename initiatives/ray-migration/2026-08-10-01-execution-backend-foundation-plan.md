# Execution Backend Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a backend-neutral execution contract that wraps the existing local (in-process) and Celery dispatch paths, split `EXECUTION_BACKEND` from `EVENT_BUS` in configuration, and extract a neutral synchronous database session module — all without adding a Ray dependency or changing any public API.

**Architecture:** FastAPI stays the control plane and PostgreSQL/SQLite stays the job-state source of truth. A new `backend/ml_pipeline/_execution/backends/` package defines a typed `ExecutionBackend` protocol (`submit`/`cancel`/`status`/`logs`) keyed on an external execution id. Two adapters — `LocalExecutionBackend` and `CeleryExecutionBackend` — encapsulate today's behavior so routes and job managers stop calling Celery directly. A neutral `backend/database/sync_session.py` provides the synchronous session that Celery workers, the future Ray driver, and local execution all share.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.0 (async + sync), Pydantic v2 / pydantic-settings, Celery + Redis (wrapped, not removed), pytest.

## Global Constraints

- Preserve existing public pipeline, job, retry, cancellation, and WebSocket API behavior — no route path, request model, or response model changes in this plan.
- Do **not** add Ray or any new runtime dependency in this plan; Ray arrives in plan 03 as `ray[default]>=2.40,<3.0`, installed only in Ray images.
- Production never silently falls back from a selected backend to in-process execution (enforced fully in plan 03; this plan only lays the selection seam).
- Keep the Celery rollback path intact: `celery_app.py`, `celery_worker.py`, and both `@shared_task` entrypoints remain callable and unchanged in behavior.
- Preserve the DB-as-truth model, the cancellation late-write guard in `TrainingJobManagerBase._update_status_sync`, retry endpoint semantics, the WebSocket invalidator event pattern, and the local fallback.
- Target Python 3.12 idioms and full typing; avoid `Any` where a concrete type exists. Every new function/method has a 1–2 line docstring.
- Every implementation task follows TDD and ends with a focused commit whose message includes:
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`
- After Python changes run, in order:
  - `ruff check .`
  - `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
  - `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
- This plan does not change any config key or API response shape exposed to the frontend, so no frontend files are touched. (Plan 02 introduces new status values and lists the exact frontend files to update.)

---

## File Structure

Create:

- `backend/database/sync_session.py` — process-wide synchronous SQLAlchemy engine/session factory, extracted from `backend/ml_pipeline/tasks.py`. One responsibility: hand out sync `Session` objects for any backend.
- `backend/config/mixins/execution.py` — new `ExecutionMixin` owning `EXECUTION_BACKEND`, `EVENT_BUS`, and `EVENT_BUS_URL`.
- `backend/ml_pipeline/_execution/backends/__init__.py` — package marker + public re-exports.
- `backend/ml_pipeline/_execution/backends/base.py` — `ExecutionState` enum, `ExecutionRequest`, `ExecutionHandle`, and the `ExecutionBackend` protocol.
- `backend/ml_pipeline/_execution/backends/local.py` — `LocalExecutionBackend`.
- `backend/ml_pipeline/_execution/backends/celery.py` — `CeleryExecutionBackend`.
- `backend/ml_pipeline/_execution/backends/registry.py` —
  `get_execution_backend(settings)` and
  `get_execution_backend_by_name(name, settings)` factories + external-id
  helpers. The named factory is required for cancelling and reconciling
  attempts created before the global backend setting changes.
- `backend/ml_pipeline/_execution/backends/dispatch.py` — `dispatch_branches(...)`, the single place that fans branch payloads out to the selected backend, preserving today's Celery-batch / local-pool behavior.
- `tests/test_sync_session_module.py`, `tests/test_config_execution_backend.py`, `tests/test_execution_backend_contract.py`, `tests/test_execution_backend_dispatch.py`, `tests/test_event_bus_split.py` — new test modules.

Modify:

- `backend/ml_pipeline/tasks.py` — `get_db_session()` becomes a thin delegate to `backend.database.sync_session` (keeps the existing patch point used by tests).
- `backend/data_ingestion/tasks.py` — `get_db_session()` delegates to the same neutral module (removes the duplicated engine cache).
- `backend/config/base.py` — add `ExecutionMixin` to the `Settings` MRO and add an alias validator reconciling legacy `USE_CELERY` with `EXECUTION_BACKEND`/`EVENT_BUS`.
- `backend/ml_pipeline/_internal/_routers/run_pipeline.py` — `_dispatch_branch_tasks` and `resubmit_job_from_graph` delegate to `dispatch_branches`.
- `backend/ml_pipeline/_execution/job_manager_base.py` — `_revoke_celery_task` becomes `_cancel_external_execution` delegating to the backend adapter.
- `backend/realtime/events.py` — route `publish_job_event` on `EVENT_BUS` (not `USE_CELERY`).
- `backend/realtime/manager.py` — `ConnectionManager.start()` picks the transport from `EVENT_BUS`.

---

### Task 1: Neutral synchronous DB session module

**Files:**
- Create: `backend/database/sync_session.py`
- Modify: `backend/ml_pipeline/tasks.py:35-60` (module-level engine cache + `get_db_session`)
- Modify: `backend/data_ingestion/tasks.py:19-41` (duplicate engine cache + `get_db_session`)
- Test: `tests/test_sync_session_module.py`

**Interfaces:**
- Produces:
  - `def _sync_database_url(async_url: str) -> str`
  - `def get_sync_session_factory() -> sessionmaker[Session]`
  - `def get_sync_session() -> Session`
- Consumes: `backend.config.get_settings` (existing), `settings.DATABASE_URL` (existing).

- [ ] **Step 1: Write the failing test**

Create `tests/test_sync_session_module.py`:

```python
"""Unit tests for the neutral synchronous session module."""

from sqlalchemy.orm import Session, sessionmaker

from backend.database import sync_session


def test_sqlite_async_url_is_converted_to_sync():
    """The aiosqlite async URL maps to the plain sqlite sync driver."""
    assert sync_session._sync_database_url("sqlite+aiosqlite:///./x.db") == "sqlite:///./x.db"


def test_postgres_async_url_is_converted_to_sync():
    """The asyncpg URL maps to the psycopg2 sync driver."""
    got = sync_session._sync_database_url("postgresql+asyncpg://u:p@h:5432/db")
    assert got == "postgresql+psycopg2://u:p@h:5432/db"


def test_factory_is_cached_and_returns_sessions(monkeypatch):
    """The factory is built once and yields real Session objects."""
    sync_session._sync_engine = None
    sync_session._sync_session_factory = None
    factory = sync_session.get_sync_session_factory()
    assert isinstance(factory, sessionmaker)
    assert sync_session.get_sync_session_factory() is factory  # cached
    session = sync_session.get_sync_session()
    assert isinstance(session, Session)
    session.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sync_session_module.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.database.sync_session'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/database/sync_session.py`:

```python
"""Neutral synchronous SQLAlchemy session factory.

Extracted from ``backend.ml_pipeline.tasks`` so every execution backend
(Celery worker, the future Ray driver, and local in-process runs) can obtain
a synchronous ``Session`` without importing Celery-specific modules. The
engine is built once per process behind a double-checked lock — building a
SQLAlchemy engine on every call is expensive and long-lived worker processes
only need one.
"""

import threading

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.config import get_settings

_sync_engine: Engine | None = None
_sync_session_factory: sessionmaker[Session] | None = None
_engine_init_lock = threading.Lock()


def _sync_database_url(async_url: str) -> str:
    """Convert an async SQLAlchemy URL to its synchronous driver equivalent."""
    if async_url.startswith("sqlite+aiosqlite://"):
        return async_url.replace("sqlite+aiosqlite://", "sqlite://")
    return async_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")


def get_sync_session_factory() -> sessionmaker[Session]:
    """Return the process-wide sync session factory, building the engine once."""
    global _sync_engine, _sync_session_factory
    if _sync_session_factory is None:
        with _engine_init_lock:
            # Double-checked locking: another thread may have finished
            # initializing while we were waiting for the lock.
            if _sync_session_factory is None:
                settings = get_settings()
                _sync_engine = create_engine(
                    _sync_database_url(settings.DATABASE_URL), pool_pre_ping=True
                )
                _sync_session_factory = sessionmaker(
                    autocommit=False, autoflush=False, bind=_sync_engine
                )
    return _sync_session_factory


def get_sync_session() -> Session:
    """Return a new synchronous ``Session`` from the shared factory."""
    return get_sync_session_factory()()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sync_session_module.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Delegate the two task-module `get_db_session` helpers to the neutral module**

In `backend/ml_pipeline/tasks.py`, delete the module-level `_sync_engine`, `_sync_session_factory`, `_engine_init_lock`, and the body of `get_db_session`, replacing lines 35–60 with:

```python
from backend.database.sync_session import get_sync_session


def get_db_session():
    """Return a synchronous DB session (kept as a stable patch point for tests)."""
    return get_sync_session()
```

In `backend/data_ingestion/tasks.py`, delete its duplicate `_sync_engine`/`_sync_session_factory`/`_engine_init_lock` block and `get_db_session` body (lines 19–41), replacing with:

```python
from backend.database.sync_session import get_sync_session


def get_db_session():
    """Return a synchronous DB session (kept as a stable patch point for tests)."""
    return get_sync_session()
```

Remove now-unused imports (`threading`, `create_engine`, `sessionmaker`, `get_settings`) from both files only if no other code in the file uses them (in `data_ingestion/tasks.py`, `get_settings` is still used elsewhere — keep it; `create_engine`/`sessionmaker`/`threading` become unused — remove them).

- [ ] **Step 6: Run the affected existing tests to verify no regression**

Run: `pytest tests/test_pipeline_task.py tests/test_data_ingestion.py -q`
Expected: PASS (existing tests still patch `backend.ml_pipeline.tasks.get_db_session`, which still exists).

- [ ] **Step 7: Static checks**

Run: `ruff check backend/database/sync_session.py backend/ml_pipeline/tasks.py backend/data_ingestion/tasks.py tests/test_sync_session_module.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors in the touched files.

- [ ] **Step 8: Commit**

```bash
git add backend/database/sync_session.py backend/ml_pipeline/tasks.py backend/data_ingestion/tasks.py tests/test_sync_session_module.py
git commit -m "refactor(db): extract neutral synchronous session module

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Config split — EXECUTION_BACKEND / EVENT_BUS

**Files:**
- Create: `backend/config/mixins/execution.py`
- Modify: `backend/config/base.py:20-56` (mixin imports + MRO) and add one `@model_validator`
- Test: `tests/test_config_execution_backend.py`

**Interfaces:**
- Produces on `Settings`:
  - `EXECUTION_BACKEND: Literal["local", "celery", "ray"]` (default `"local"`)
  - `EVENT_BUS: Literal["local", "redis"]` (default `"local"`)
  - `EVENT_BUS_URL: str | None` (default `None`)
  - Validator `sync_execution_backend_aliases` that reconciles legacy `USE_CELERY`.
- Consumes: existing `Settings.is_field_set` (defined in `backend/config/base.py:204`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_execution_backend.py`:

```python
"""Tests for the EXECUTION_BACKEND / EVENT_BUS config split and USE_CELERY aliasing."""

from backend.config.base import Settings

_SECRET = "x" * 32


def _settings(**env: object) -> Settings:
    """Build a Settings instance with a valid SECRET_KEY and given overrides."""
    return Settings(SECRET_KEY=_SECRET, **env)


def test_defaults_are_local():
    """With nothing set, execution runs in-process and events stay in-process."""
    s = _settings()
    assert s.EXECUTION_BACKEND == "local"
    assert s.EVENT_BUS == "local"


def test_legacy_use_celery_true_derives_celery_and_redis():
    """A deployment that only sets USE_CELERY=true keeps working via aliasing."""
    s = _settings(USE_CELERY=True)
    assert s.EXECUTION_BACKEND == "celery"
    assert s.EVENT_BUS == "redis"


def test_explicit_execution_backend_sets_use_celery_alias():
    """Setting EXECUTION_BACKEND=celery keeps the legacy USE_CELERY flag in sync."""
    s = _settings(EXECUTION_BACKEND="celery")
    assert s.USE_CELERY is True


def test_explicit_values_are_not_overridden_by_alias():
    """Explicit EVENT_BUS wins even when USE_CELERY disagrees."""
    s = _settings(USE_CELERY=True, EVENT_BUS="local")
    assert s.EVENT_BUS == "local"
    assert s.EXECUTION_BACKEND == "celery"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_execution_backend.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'EXECUTION_BACKEND'`.

- [ ] **Step 3: Write the mixin**

Create `backend/config/mixins/execution.py`:

```python
"""Execution backend and realtime event transport settings.

Separates *where pipeline work runs* (``EXECUTION_BACKEND``) from *how job
events reach the WebSocket layer* (``EVENT_BUS``). Removing Celery as the
executor must not force removing Redis as the event transport, so these are
independent knobs. The legacy ``USE_CELERY`` boolean (see ``CeleryMixin``) is
reconciled with these fields by a validator in ``Settings``.
"""

from typing import Literal

ExecutionBackendName = Literal["local", "celery", "ray"]
EventBusName = Literal["local", "redis"]


class ExecutionMixin:
    """Execution backend selection and realtime event transport."""

    # Where pipeline jobs execute. ``local`` runs in-process; ``celery`` uses
    # the existing Celery worker; ``ray`` (plan 03) submits Ray Jobs.
    EXECUTION_BACKEND: ExecutionBackendName = "local"

    # How WebSocket invalidator events are delivered. ``local`` uses the
    # in-process bus; ``redis`` uses pub/sub across processes.
    EVENT_BUS: EventBusName = "local"

    # Redis URL for the event bus when ``EVENT_BUS == "redis"``. When unset,
    # the publisher falls back to ``REDIS_URL`` then ``CELERY_BROKER_URL`` so
    # existing deployments keep working unchanged.
    EVENT_BUS_URL: str | None = None
```

- [ ] **Step 4: Wire the mixin and alias validator into `Settings`**

In `backend/config/base.py`, add the import next to the other mixin imports:

```python
from backend.config.mixins.execution import ExecutionMixin
```

Add `ExecutionMixin` to the `Settings` base list (place it right after `CeleryMixin`):

```python
class Settings(
    CoreMixin,
    AWSMixin,
    SecurityMixin,
    DatabaseMixin,
    CeleryMixin,
    ExecutionMixin,
    FilesMixin,
    SnowflakeMixin,
    LoggingMixin,
    CacheMixin,
    LLMMixin,
    BaseSettings,
):
```

Add this validator method to `Settings` (after `validate_fallback_auth`):

```python
@model_validator(mode="after")
def sync_execution_backend_aliases(self) -> "Settings":
    """Reconcile the legacy ``USE_CELERY`` flag with ``EXECUTION_BACKEND``/``EVENT_BUS``.

    Explicitly-set fields always win; only unset counterparts are derived so
    old deployments that set only ``USE_CELERY`` and new deployments that set
    only ``EXECUTION_BACKEND`` both behave correctly.
    """
    exec_set = self.is_field_set("EXECUTION_BACKEND")
    bus_set = self.is_field_set("EVENT_BUS")
    celery_set = self.is_field_set("USE_CELERY")

    if celery_set and not exec_set:
        object.__setattr__(self, "EXECUTION_BACKEND", "celery" if self.USE_CELERY else "local")
    if celery_set and not bus_set:
        object.__setattr__(self, "EVENT_BUS", "redis" if self.USE_CELERY else "local")
    if exec_set and not celery_set:
        object.__setattr__(self, "USE_CELERY", self.EXECUTION_BACKEND == "celery")
    return self
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config_execution_backend.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Guard against regressions in existing config tests**

Run: `pytest tests/test_config_centralized_caps.py tests/test_jobs_settings_driven_constants.py -q`
Expected: PASS.

- [ ] **Step 7: Static checks**

Run: `ruff check backend/config/mixins/execution.py backend/config/base.py tests/test_config_execution_backend.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add backend/config/mixins/execution.py backend/config/base.py tests/test_config_execution_backend.py
git commit -m "feat(config): split EXECUTION_BACKEND from EVENT_BUS with USE_CELERY aliasing

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Execution backend protocol and value types

**Files:**
- Create: `backend/ml_pipeline/_execution/backends/__init__.py`
- Create: `backend/ml_pipeline/_execution/backends/base.py`
- Test: `tests/test_execution_backend_contract.py`

**Interfaces:**
- Produces:
  - `class ExecutionState(StrEnum)` with members `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `STOPPED`, `MISSING`.
  - `@dataclass(frozen=True, slots=True) class ExecutionRequest` with `job_id: str`, `attempt_id: str`, `payload: dict[str, Any]`.
  - `@dataclass(frozen=True, slots=True) class ExecutionHandle` with `external_execution_id: str`.
  - `class ExecutionBackend(Protocol)` with `name: str` and methods `submit(request: ExecutionRequest) -> ExecutionHandle`, `cancel(external_execution_id: str) -> None`, `status(external_execution_id: str) -> ExecutionState`, `logs(external_execution_id: str) -> str`.
  - `def get_execution_backend_by_name(name: str, settings: Settings | None = None) -> ExecutionBackend` resolves the adapter recorded on an execution attempt.
- Consumes: nothing backend-specific (keeps this module import-cheap so routes can depend on it).

- [ ] **Step 1: Write the failing test**

Create `tests/test_execution_backend_contract.py`:

```python
"""Contract tests for the execution backend protocol and value types."""

from backend.ml_pipeline._execution.backends.base import (
    ExecutionBackend,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionState,
)


class _Stub:
    """Minimal in-memory backend used only to prove the protocol is satisfiable."""

    name = "stub"

    def submit(self, request: ExecutionRequest) -> ExecutionHandle:
        """Record and echo the request as an external id."""
        return ExecutionHandle(external_execution_id=f"{request.job_id}:{request.attempt_id}")

    def cancel(self, external_execution_id: str) -> None:
        """No-op cancel."""

    def status(self, external_execution_id: str) -> ExecutionState:
        """Always running."""
        return ExecutionState.RUNNING

    def logs(self, external_execution_id: str) -> str:
        """Empty logs."""
        return ""


def test_request_is_frozen():
    """ExecutionRequest is immutable so a submitted request can't drift."""
    req = ExecutionRequest(job_id="j1", attempt_id="a1", payload={"k": 1})
    try:
        req.job_id = "j2"  # type: ignore[misc]
    except Exception as exc:  # FrozenInstanceError
        assert "cannot assign" in str(exc).lower() or "frozen" in str(exc).lower()
    else:
        raise AssertionError("ExecutionRequest should be frozen")


def test_stub_satisfies_protocol_and_roundtrips():
    """A structural backend implements the protocol and returns a handle."""
    backend: ExecutionBackend = _Stub()
    handle = backend.submit(ExecutionRequest(job_id="j1", attempt_id="a1"))
    assert isinstance(handle, ExecutionHandle)
    assert handle.external_execution_id == "j1:a1"
    assert backend.status(handle.external_execution_id) is ExecutionState.RUNNING
    assert isinstance(backend, ExecutionBackend)  # runtime_checkable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_execution_backend_contract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.ml_pipeline._execution.backends'`.

- [ ] **Step 3: Write the package and base module**

Create `backend/ml_pipeline/_execution/backends/__init__.py`:

```python
"""Backend-neutral execution package.

Re-exports the execution contract so callers depend on a single import path.
Concrete adapters and the registry are imported lazily by the registry to
avoid pulling Celery/Ray into modules that only need the types.
"""

from backend.ml_pipeline._execution.backends.base import (
    ExecutionBackend,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionState,
)

__all__ = [
    "ExecutionBackend",
    "ExecutionHandle",
    "ExecutionRequest",
    "ExecutionState",
]
```

Create `backend/ml_pipeline/_execution/backends/base.py`:

```python
"""Typed contract every execution backend implements.

No backend-specific imports live here so routes and job services can depend on
the protocol without importing Celery or Ray. The external execution id is the
only handle the control plane keeps for cancel/status/logs.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class ExecutionState(StrEnum):
    """Backend-reported lifecycle state of a submitted execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """A single unit of pipeline work handed to an execution backend."""

    job_id: str
    attempt_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionHandle:
    """Identifies a submitted execution by its backend-assigned external id."""

    external_execution_id: str


@runtime_checkable
class ExecutionBackend(Protocol):
    """Submit and control pipeline executions independently of the transport."""

    name: str

    def submit(self, request: ExecutionRequest) -> ExecutionHandle:
        """Submit one execution and return its external execution id handle."""
        ...

    def cancel(self, external_execution_id: str) -> None:
        """Request termination of the execution with the given external id."""
        ...

    def status(self, external_execution_id: str) -> ExecutionState:
        """Return the backend's view of the execution's lifecycle state."""
        ...

    def logs(self, external_execution_id: str) -> str:
        """Return sanitized backend logs for the execution, or an empty string."""
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_execution_backend_contract.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/backends/ tests/test_execution_backend_contract.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add backend/ml_pipeline/_execution/backends/__init__.py backend/ml_pipeline/_execution/backends/base.py tests/test_execution_backend_contract.py
git commit -m "feat(execution): add typed backend-neutral execution contract

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Local and Celery adapters + registry

**Files:**
- Create: `backend/ml_pipeline/_execution/backends/local.py`
- Create: `backend/ml_pipeline/_execution/backends/celery.py`
- Create: `backend/ml_pipeline/_execution/backends/registry.py`
- Test: `tests/test_execution_backend_contract.py` (extend)

**Interfaces:**
- Consumes: `ExecutionBackend`, `ExecutionRequest`, `ExecutionHandle`, `ExecutionState` (Task 3); `run_pipeline_task` from `backend.ml_pipeline.tasks`; `celery_app` from `backend.celery_app`; `get_settings`; `Settings.MAX_PARALLEL_BRANCH_WORKERS`, `Settings.EXECUTION_BACKEND`.
- Produces:
  - `class LocalExecutionBackend` — `name = "local"`; runs `run_pipeline_task` on a shared bounded `ThreadPoolExecutor`; tracks futures by external id.
  - `class CeleryExecutionBackend` — `name = "celery"`; dispatches `run_pipeline_batch_task.delay([(job_id, payload)])`; maps `celery_app.AsyncResult(...).state`.
  - `def get_execution_backend(settings: Settings | None = None) -> ExecutionBackend`.
  - `def make_external_id(prefix: str, job_id: str, attempt_id: str) -> str`.

- [ ] **Step 1: Write the failing tests (append to `tests/test_execution_backend_contract.py`)**

```python
from unittest.mock import MagicMock, patch

import pytest

from backend.config.base import Settings
from backend.ml_pipeline._execution.backends.registry import get_execution_backend


def _settings(**env: object) -> Settings:
    """Settings with a valid secret and overrides."""
    return Settings(SECRET_KEY="x" * 32, **env)


def test_registry_returns_local_by_default():
    """EXECUTION_BACKEND=local yields the local adapter."""
    backend = get_execution_backend(_settings(EXECUTION_BACKEND="local"))
    assert backend.name == "local"


def test_registry_returns_celery_when_selected():
    """EXECUTION_BACKEND=celery yields the celery adapter."""
    backend = get_execution_backend(_settings(EXECUTION_BACKEND="celery"))
    assert backend.name == "celery"


def test_registry_rejects_ray_until_plan_03():
    """Ray is not registered yet; selecting it fails loudly rather than silently."""
    with pytest.raises(ValueError, match="ray"):
        get_execution_backend(_settings(EXECUTION_BACKEND="ray"))


def test_local_backend_submits_to_run_pipeline_task():
    """LocalExecutionBackend.submit runs run_pipeline_task with the payload."""
    from backend.ml_pipeline._execution.backends.local import LocalExecutionBackend
    from backend.ml_pipeline._execution.backends.base import ExecutionRequest

    backend = LocalExecutionBackend(max_workers=1)
    with patch(
        "backend.ml_pipeline._execution.backends.local.run_pipeline_task"
    ) as mock_task:
        handle = backend.submit(
            ExecutionRequest(job_id="j1", attempt_id="a1", payload={"nodes": []})
        )
        backend.wait(handle.external_execution_id, timeout=5)
    mock_task.assert_called_once_with("j1", {"nodes": []})
    assert handle.external_execution_id.startswith("local:")


def test_celery_backend_submits_batch_and_maps_status():
    """CeleryExecutionBackend.submit delegates to run_pipeline_batch_task.delay."""
    from backend.ml_pipeline._execution.backends.base import (
        ExecutionRequest,
        ExecutionState,
    )
    from backend.ml_pipeline._execution.backends.celery import CeleryExecutionBackend

    backend = CeleryExecutionBackend()
    fake_task = MagicMock()
    fake_task.id = "celery-123"
    with patch(
        "backend.ml_pipeline._execution.backends.celery.run_pipeline_batch_task"
    ) as mock_batch:
        mock_batch.delay.return_value = fake_task
        handle = backend.submit(ExecutionRequest(job_id="j1", attempt_id="a1", payload={}))
    mock_batch.delay.assert_called_once_with([("j1", {})])
    assert handle.external_execution_id == "celery-123"

    with patch("backend.ml_pipeline._execution.backends.celery.celery_app") as mock_app:
        mock_app.AsyncResult.return_value.state = "SUCCESS"
        assert backend.status("celery-123") is ExecutionState.SUCCEEDED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_execution_backend_contract.py -v -k "registry or local_backend or celery_backend"`
Expected: FAIL — `ModuleNotFoundError` for `...backends.registry` / `...backends.local` / `...backends.celery`.

- [ ] **Step 3: Write `LocalExecutionBackend`**

Create `backend/ml_pipeline/_execution/backends/local.py`:

```python
"""In-process execution backend.

Runs the existing ``run_pipeline_task`` callable on a shared, bounded thread
pool so the FastAPI process can execute pipelines without Celery. This is the
local development / single-process fallback; the DB remains the source of
truth for job status, so ``logs`` returns an empty string and ``status``
reflects only the in-process future.
"""

import uuid
from concurrent.futures import Future, ThreadPoolExecutor

from backend.config import get_settings
from backend.ml_pipeline._execution.backends.base import (
    ExecutionHandle,
    ExecutionRequest,
    ExecutionState,
)
from backend.ml_pipeline.tasks import run_pipeline_task


class LocalExecutionBackend:
    """Execute pipeline jobs in-process on a bounded thread pool."""

    name = "local"

    def __init__(self, max_workers: int | None = None) -> None:
        """Create the backend, sizing the pool from settings when not given."""
        workers = max_workers or get_settings().MAX_PARALLEL_BRANCH_WORKERS
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers))
        self._futures: dict[str, Future[None]] = {}

    def submit(self, request: ExecutionRequest) -> ExecutionHandle:
        """Schedule ``run_pipeline_task`` for the request; return a local handle."""
        external_id = f"local:{request.job_id}:{request.attempt_id}:{uuid.uuid4().hex[:8]}"
        future = self._executor.submit(run_pipeline_task, request.job_id, request.payload)
        self._futures[external_id] = future
        return ExecutionHandle(external_execution_id=external_id)

    def cancel(self, external_execution_id: str) -> None:
        """Best-effort cancel of the in-process future (no effect once started)."""
        future = self._futures.get(external_execution_id)
        if future is not None:
            future.cancel()

    def status(self, external_execution_id: str) -> ExecutionState:
        """Map the tracked future's state to an ``ExecutionState``."""
        future = self._futures.get(external_execution_id)
        if future is None:
            return ExecutionState.MISSING
        if not future.done():
            return ExecutionState.RUNNING
        if future.cancelled():
            return ExecutionState.STOPPED
        return ExecutionState.FAILED if future.exception() else ExecutionState.SUCCEEDED

    def logs(self, external_execution_id: str) -> str:
        """Local execution logs live on the DB job row; return empty here."""
        return ""

    def wait(self, external_execution_id: str, timeout: float | None = None) -> None:
        """Block until the tracked future finishes (used by tests and drain)."""
        future = self._futures.get(external_execution_id)
        if future is not None:
            future.result(timeout=timeout)
```

- [ ] **Step 4: Write `CeleryExecutionBackend`**

Create `backend/ml_pipeline/_execution/backends/celery.py`:

```python
"""Celery execution backend.

Wraps the existing Celery dispatch so routes stop calling Celery directly.
``submit`` dispatches a single-branch ``run_pipeline_batch_task`` (the same
task the multi-branch path already batches into) and returns the Celery task
id as the external execution id. ``status`` maps Celery result states.
"""

from backend.celery_app import celery_app
from backend.ml_pipeline._execution.backends.base import (
    ExecutionHandle,
    ExecutionRequest,
    ExecutionState,
)
from backend.ml_pipeline.tasks import run_pipeline_batch_task

_CELERY_STATE_MAP: dict[str, ExecutionState] = {
    "PENDING": ExecutionState.PENDING,
    "RECEIVED": ExecutionState.PENDING,
    "STARTED": ExecutionState.RUNNING,
    "RETRY": ExecutionState.RUNNING,
    "SUCCESS": ExecutionState.SUCCEEDED,
    "FAILURE": ExecutionState.FAILED,
    "REVOKED": ExecutionState.STOPPED,
}


class CeleryExecutionBackend:
    """Dispatch and control pipeline jobs through Celery + Redis."""

    name = "celery"

    def submit(self, request: ExecutionRequest) -> ExecutionHandle:
        """Dispatch one branch as a Celery task and return its task id handle."""
        task = run_pipeline_batch_task.delay([(request.job_id, request.payload)])
        return ExecutionHandle(external_execution_id=task.id)

    def cancel(self, external_execution_id: str) -> None:
        """Revoke and terminate the Celery task (SIGTERM) for this execution."""
        celery_app.control.revoke(external_execution_id, terminate=True, signal="SIGTERM")

    def status(self, external_execution_id: str) -> ExecutionState:
        """Map the Celery result state to an ``ExecutionState``."""
        state = celery_app.AsyncResult(external_execution_id).state
        return _CELERY_STATE_MAP.get(state, ExecutionState.MISSING)

    def logs(self, external_execution_id: str) -> str:
        """Celery task logs are not fetched here; the DB job row carries logs."""
        return ""
```

- [ ] **Step 5: Write the registry**

Create `backend/ml_pipeline/_execution/backends/registry.py`:

```python
"""Execution backend registry.

Resolves the configured ``EXECUTION_BACKEND`` to a concrete adapter. The local
backend is a process-wide singleton so its thread pool and future registry are
shared. Selecting ``ray`` raises until plan 03 registers the Ray adapter —
production must never silently fall back to in-process execution.
"""

from backend.config import Settings, get_settings
from backend.ml_pipeline._execution.backends.base import ExecutionBackend
from backend.ml_pipeline._execution.backends.celery import CeleryExecutionBackend
from backend.ml_pipeline._execution.backends.local import LocalExecutionBackend

_local_singleton: LocalExecutionBackend | None = None


def make_external_id(prefix: str, job_id: str, attempt_id: str) -> str:
    """Build a stable, greppable external id for logging and reconciliation."""
    return f"{prefix}:{job_id}:{attempt_id}"


def get_execution_backend_by_name(
    name: str, settings: Settings | None = None
) -> ExecutionBackend:
    """Return the adapter for an explicit persisted backend name."""
    global _local_singleton
    settings = settings or get_settings()
    if name == "local":
        if _local_singleton is None:
            _local_singleton = LocalExecutionBackend()
        return _local_singleton
    if name == "celery":
        return CeleryExecutionBackend()
    raise ValueError(
        f"EXECUTION_BACKEND={name!r} is not available. The 'ray' backend is "
        "registered in the Ray Jobs runtime plan (03); production must not "
        "silently fall back to in-process execution."
    )


def get_execution_backend(settings: Settings | None = None) -> ExecutionBackend:
    """Return the adapter selected for new submissions."""
    settings = settings or get_settings()
    return get_execution_backend_by_name(settings.EXECUTION_BACKEND, settings)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_execution_backend_contract.py -v`
Expected: PASS (all tests, including the new registry/local/celery tests).

- [ ] **Step 7: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/backends/ tests/test_execution_backend_contract.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add backend/ml_pipeline/_execution/backends/local.py backend/ml_pipeline/_execution/backends/celery.py backend/ml_pipeline/_execution/backends/registry.py tests/test_execution_backend_contract.py
git commit -m "feat(execution): add local and celery adapters behind a registry

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Branch dispatcher and route wiring

**Files:**
- Create: `backend/ml_pipeline/_execution/backends/dispatch.py`
- Modify: `backend/ml_pipeline/_internal/_routers/run_pipeline.py:263-291` (`_dispatch_branch_tasks`) and `:354-365` (Celery/BackgroundTasks block inside `resubmit_job_from_graph`)
- Test: `tests/test_execution_backend_dispatch.py`
- Test (update): `tests/test_jobs_router_retry.py` (patch target for the local dispatch path)

**Interfaces:**
- Consumes: `get_execution_backend`, `ExecutionRequest` (Tasks 3–4); `JobManager.attach_celery_task_id` (existing, `backend/ml_pipeline/_execution/jobs.py:132`).
- Produces:
  - `async def dispatch_branches(task_payloads: list[tuple[str, dict[str, Any]]], *, settings: Settings, db: AsyncSession) -> None`
  - Behavior parity: for `EXECUTION_BACKEND=celery`, submit each branch via the adapter and attach the returned external id to the job (via `attach_celery_task_id`); for `local`, submit each branch via the shared local backend.

- [ ] **Step 1: Write the failing test**

Create `tests/test_execution_backend_dispatch.py`:

```python
"""Tests for the branch dispatcher that fans payloads to the selected backend."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.config.base import Settings
from backend.ml_pipeline._execution.backends.dispatch import dispatch_branches


def _settings(**env: object) -> Settings:
    """Settings with a valid secret and overrides."""
    return Settings(SECRET_KEY="x" * 32, **env)


@pytest.mark.asyncio
async def test_dispatch_local_submits_each_branch():
    """Local dispatch submits one execution per payload through the local backend."""
    payloads = [("j1", {"nodes": []}), ("j2", {"nodes": []})]
    fake_backend = MagicMock(name="local")
    fake_backend.name = "local"
    fake_backend.submit.return_value.external_execution_id = "local:x"
    with patch(
        "backend.ml_pipeline._execution.backends.dispatch.get_execution_backend",
        return_value=fake_backend,
    ):
        await dispatch_branches(payloads, settings=_settings(EXECUTION_BACKEND="local"), db=AsyncMock())
    assert fake_backend.submit.call_count == 2


@pytest.mark.asyncio
async def test_dispatch_celery_attaches_external_id():
    """Celery dispatch attaches the returned external id onto each job row."""
    payloads = [("j1", {"nodes": []})]
    fake_backend = MagicMock(name="celery")
    fake_backend.name = "celery"
    fake_backend.submit.return_value.external_execution_id = "celery-123"
    with (
        patch(
            "backend.ml_pipeline._execution.backends.dispatch.get_execution_backend",
            return_value=fake_backend,
        ),
        patch(
            "backend.ml_pipeline._execution.backends.dispatch.JobManager.attach_celery_task_id",
            new=AsyncMock(),
        ) as mock_attach,
    ):
        await dispatch_branches(payloads, settings=_settings(EXECUTION_BACKEND="celery"), db=AsyncMock())
    mock_attach.assert_awaited_once()
    assert mock_attach.await_args.args[2] == "celery-123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_execution_backend_dispatch.py -v`
Expected: FAIL — `ModuleNotFoundError: ...backends.dispatch`.

- [ ] **Step 3: Write the dispatcher**

Create `backend/ml_pipeline/_execution/backends/dispatch.py`:

```python
"""Branch dispatcher — the single fan-out point for pipeline submissions.

Routes and the retry service call this instead of touching Celery or
BackgroundTasks directly. Each branch payload becomes one execution submitted
through the configured backend. For the Celery backend the returned external
id is attached to the job row (under ``job_metadata.celery_task_id``) so
cancellation can revoke it, preserving today's behavior.
"""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings
from backend.ml_pipeline._execution.backends.base import ExecutionRequest
from backend.ml_pipeline._execution.backends.registry import get_execution_backend
from backend.ml_pipeline._execution.jobs import JobManager

logger = logging.getLogger(__name__)


async def dispatch_branches(
    task_payloads: list[tuple[str, dict[str, Any]]],
    *,
    settings: Settings,
    db: AsyncSession,
) -> None:
    """Submit each ``(job_id, payload)`` branch through the configured backend.

    The ``attempt_id`` equals ``job_id`` in this plan; plan 02 replaces it with
    a durable attempt id. Attaching the external id is a no-op for the local
    backend (its handles are not persisted).
    """
    if not task_payloads:
        return

    backend = get_execution_backend(settings)
    for job_id, payload in task_payloads:
        handle = backend.submit(ExecutionRequest(job_id=job_id, attempt_id=job_id, payload=payload))
        if backend.name == "celery":
            try:
                await JobManager.attach_celery_task_id(db, job_id, handle.external_execution_id)
            except Exception:
                logger.warning("Failed to attach external id for job %s", job_id)
```

- [ ] **Step 4: Wire `run_pipeline.py` to the dispatcher**

In `backend/ml_pipeline/_internal/_routers/run_pipeline.py`, replace the body of `_dispatch_branch_tasks` (lines 263–291) with a delegation, and drop the now-unused `_run_branches_concurrently` helper and the `ThreadPoolExecutor` import:

```python
async def _dispatch_branch_tasks(
    task_payloads: list[tuple],
    settings: Any,
    background_tasks: BackgroundTasks,
    db: AsyncSession,
) -> None:
    """Dispatch branch execution through the configured execution backend."""
    from backend.ml_pipeline._execution.backends.dispatch import dispatch_branches

    await dispatch_branches(task_payloads, settings=settings, db=db)
```

In `resubmit_job_from_graph`, replace the `if settings.USE_CELERY: ... else: background_tasks.add_task(...)` block (lines 354–364) with:

```python
    settings = get_settings()
    payload: dict[str, Any] = dict(branch_graph)
    from backend.ml_pipeline._execution.backends.dispatch import dispatch_branches

    await dispatch_branches([(new_job_id, payload)], settings=settings, db=db)
```

Leave the `background_tasks` parameters in the signatures (still part of the public function contracts and used by callers) but they are no longer the dispatch mechanism.

- [ ] **Step 5: Update the retry test's patch target**

In `tests/test_jobs_router_retry.py`, the local dispatch path now runs through the local backend, not `run_pipeline.run_pipeline_task`. Update `test_retry_succeeds_for_failed_training_job` to patch the local backend's submit and assert the new job was created:

```python
@pytest.mark.asyncio
async def test_retry_succeeds_for_failed_training_job(async_session, client):
    """POST .../retry on a failed training job with a stored graph creates a fresh job id."""
    await _insert_job(async_session, "job-1", status="failed")

    with patch(
        "backend.ml_pipeline._execution.backends.local.LocalExecutionBackend.submit"
    ) as mock_submit:
        mock_submit.return_value.external_execution_id = "local:job"
        response = client.post(f"{BASE}/jobs/job-1/retry")

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] != "job-1"
    assert "retry" in body["message"].lower()
    mock_submit.assert_called_once()
    submitted_request = mock_submit.call_args.args[0]
    assert submitted_request.job_id == body["job_id"]
    assert submitted_request.payload["nodes"][0]["node_id"] == "node-1"
```

Apply the same patch-target change to `test_retry_succeeds_for_cancelled_tuning_job` (replace the `run_pipeline.run_pipeline_task` patch with the `LocalExecutionBackend.submit` patch) and to `test_concurrent_retries_create_only_one_job`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_execution_backend_dispatch.py tests/test_jobs_router_retry.py tests/test_run_pipeline_helpers.py -q`
Expected: PASS.

- [ ] **Step 7: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/backends/dispatch.py backend/ml_pipeline/_internal/_routers/run_pipeline.py tests/test_execution_backend_dispatch.py tests/test_jobs_router_retry.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add backend/ml_pipeline/_execution/backends/dispatch.py backend/ml_pipeline/_internal/_routers/run_pipeline.py tests/test_execution_backend_dispatch.py tests/test_jobs_router_retry.py
git commit -m "refactor(run): dispatch branches through the execution backend registry

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: Cancellation through the backend adapter

**Files:**
- Modify: `backend/ml_pipeline/_execution/job_manager_base.py:54-103` (`_revoke_celery_task` and `_cancel_job`)
- Test: `tests/test_job_manager_base.py` (extend)

**Interfaces:**
- Consumes: `get_execution_backend` (Task 4); existing `job_metadata["celery_task_id"]` field.
- Produces: `TrainingJobManagerBase._cancel_external_execution(job_metadata: dict[str, Any]) -> None` — resolves the external id and delegates to `backend.cancel(...)`, preserving the best-effort (never-raise) contract.

- [ ] **Step 1: Write the failing test (append to `tests/test_job_manager_base.py`)**

```python
from unittest.mock import MagicMock, patch

from backend.ml_pipeline._execution.job_manager_base import TrainingJobManagerBase


def test_cancel_external_execution_delegates_to_backend():
    """The external id stored on the job is cancelled through the backend adapter."""
    fake_backend = MagicMock()
    with patch(
        "backend.ml_pipeline._execution.job_manager_base.get_execution_backend",
        return_value=fake_backend,
    ):
        TrainingJobManagerBase._cancel_external_execution({"celery_task_id": "celery-123"})
    fake_backend.cancel.assert_called_once_with("celery-123")


def test_cancel_external_execution_is_best_effort():
    """A backend error during cancel is swallowed so the user-visible cancel still returns."""
    fake_backend = MagicMock()
    fake_backend.cancel.side_effect = RuntimeError("broker down")
    with patch(
        "backend.ml_pipeline._execution.job_manager_base.get_execution_backend",
        return_value=fake_backend,
    ):
        # Must not raise.
        TrainingJobManagerBase._cancel_external_execution({"celery_task_id": "celery-123"})


def test_cancel_external_execution_noop_without_id():
    """No stored external id means there is nothing to cancel."""
    fake_backend = MagicMock()
    with patch(
        "backend.ml_pipeline._execution.job_manager_base.get_execution_backend",
        return_value=fake_backend,
    ):
        TrainingJobManagerBase._cancel_external_execution({})
    fake_backend.cancel.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_job_manager_base.py -v -k cancel_external`
Expected: FAIL — `AttributeError: ... has no attribute '_cancel_external_execution'`.

- [ ] **Step 3: Replace `_revoke_celery_task` with a backend-neutral cancel**

In `backend/ml_pipeline/_execution/job_manager_base.py`, replace the `_revoke_celery_task` static method with:

```python
@staticmethod
def _cancel_external_execution(job_metadata: dict[str, Any]) -> None:
    """Best-effort cancel of the external execution recorded in *job_metadata*.

    Reads the stored external execution id (still ``celery_task_id`` in this
    plan; plan 02 moves it onto the attempt row) and delegates to the
    configured backend. Any transport/network failure is swallowed so it never
    blocks the user-visible cancel — the late-write guard in
    ``_update_status_sync`` keeps the row CANCELLED even if a worker writes
    back.
    """
    external_id = job_metadata.get("celery_task_id")
    if not external_id:
        return
    try:
        from backend.ml_pipeline._execution.backends.registry import (  # noqa: PLC0415
            get_execution_backend,
        )

        get_execution_backend().cancel(external_id)
    except Exception:
        pass  # nosec B110 - best-effort cancel; guard protects final state
```

Update the single caller inside `_cancel_job` (line ~100) from:

```python
            TrainingJobManagerBase._revoke_celery_task(meta)
```

to:

```python
            TrainingJobManagerBase._cancel_external_execution(meta)
```

Add the import used by the test's patch target near the top of the module (module-level so the patch path resolves), replacing the lazy import inside the method is optional — keep the lazy import inside the method to avoid a circular import at module load, and ensure the test patches `backend.ml_pipeline._execution.job_manager_base.get_execution_backend`. To make that patch target valid, add at module top:

```python
from backend.ml_pipeline._execution.backends.registry import get_execution_backend
```

and change the method body to call `get_execution_backend()` directly (drop the inner import). If a circular import surfaces at load time, keep the top-level import removed and instead have the test patch `backend.ml_pipeline._execution.backends.registry.get_execution_backend`; the plan's default is the module-level import since `registry` does not import `job_manager_base`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_job_manager_base.py -v`
Expected: PASS (existing tests plus the three new cancel tests).

- [ ] **Step 5: Run the cancellation-path regression tests**

Run: `pytest tests/test_ml_pipeline_backend_fixes.py tests/test_execution.py -q`
Expected: PASS — the late-write guard behavior is unchanged.

- [ ] **Step 6: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/job_manager_base.py tests/test_job_manager_base.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add backend/ml_pipeline/_execution/job_manager_base.py tests/test_job_manager_base.py
git commit -m "refactor(cancel): route job cancellation through the execution backend

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: Event bus split (EVENT_BUS instead of USE_CELERY)

**Files:**
- Modify: `backend/realtime/events.py:36-77` (`_redis_client_sync` + `publish_job_event`)
- Modify: `backend/realtime/manager.py:64-76` (`ConnectionManager.start` transport selection)
- Test: `tests/test_event_bus_split.py`

**Interfaces:**
- Consumes: `settings.EVENT_BUS`, `settings.EVENT_BUS_URL`, `settings.REDIS_URL`, `settings.CELERY_BROKER_URL`.
- Produces:
  - `def _event_bus_url(settings) -> str` (resolution order: `EVENT_BUS_URL` → `REDIS_URL` → `CELERY_BROKER_URL`)
  - `publish_job_event` routes on `EVENT_BUS == "redis"` (redis) else in-process bus.
  - `ConnectionManager.start()` picks `_subscriber_loop` when `EVENT_BUS == "redis"`, else `_local_loop`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_event_bus_split.py`:

```python
"""Tests that realtime transport is driven by EVENT_BUS, not USE_CELERY."""

from unittest.mock import MagicMock, patch

from backend.config.base import Settings
from backend.realtime import events


def _settings(**env: object) -> Settings:
    """Settings with a valid secret and overrides."""
    return Settings(SECRET_KEY="x" * 32, **env)


def test_publish_uses_local_bus_when_event_bus_local():
    """EVENT_BUS=local publishes to the in-process bus and never touches Redis."""
    ev = events.JobEvent(event="status", job_id="j1", status="running")
    with (
        patch.object(events, "get_settings", return_value=_settings(EVENT_BUS="local")),
        patch("backend.realtime.local_bus.local_bus") as mock_local,
        patch.object(events, "_redis_client_sync") as mock_redis,
    ):
        events.publish_job_event(ev)
    mock_local.publish.assert_called_once()
    mock_redis.assert_not_called()


def test_publish_uses_redis_when_event_bus_redis():
    """EVENT_BUS=redis publishes to Redis pub/sub."""
    ev = events.JobEvent(event="status", job_id="j1", status="running")
    fake_client = MagicMock()
    with (
        patch.object(events, "get_settings", return_value=_settings(EVENT_BUS="redis")),
        patch.object(events, "_redis_client_sync", return_value=fake_client),
    ):
        events.publish_job_event(ev)
    fake_client.publish.assert_called_once()


def test_event_bus_url_resolution_prefers_explicit():
    """EVENT_BUS_URL wins over REDIS_URL and CELERY_BROKER_URL."""
    s = _settings(EVENT_BUS_URL="redis://a:6379/2", REDIS_URL="redis://b:6379/3")
    assert events._event_bus_url(s) == "redis://a:6379/2"


def test_event_bus_url_falls_back_to_celery_broker():
    """With nothing else set, the legacy CELERY_BROKER_URL is used unchanged."""
    s = _settings()
    assert events._event_bus_url(s) == s.CELERY_BROKER_URL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_event_bus_split.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_event_bus_url'` and the redis-routing assertion fails because the current code branches on `USE_CELERY`.

- [ ] **Step 3: Update `events.py`**

In `backend/realtime/events.py`, add the URL resolver and update the two functions:

```python
def _event_bus_url(settings: Any) -> str:
    """Resolve the event-bus Redis URL, preserving legacy fallbacks."""
    return settings.EVENT_BUS_URL or settings.REDIS_URL or settings.CELERY_BROKER_URL


def _redis_client_sync() -> Any:
    """Sync Redis client for the event bus.

    Imported lazily so the module is importable in environments without Redis
    (e.g. unit tests that exercise the engine but not the queue).
    """
    import redis

    return redis.Redis.from_url(_event_bus_url(get_settings()), decode_responses=True)
```

Replace the routing block in `publish_job_event`:

```python
    payload = orjson.dumps(event.model_dump(exclude_none=True)).decode()
    settings = get_settings()
    if settings.EVENT_BUS != "redis":
        # Lazy import avoids a hard cycle (manager imports events).
        from backend.realtime.local_bus import local_bus

        local_bus.publish(payload)
        return
    try:
        client = _redis_client_sync()
        client.publish(JOB_EVENTS_CHANNEL, payload)
    except Exception as exc:  # pragma: no cover - depends on live Redis
        logger.warning("publish_job_event failed for %s: %s", event.job_id, exc)
```

- [ ] **Step 4: Update `manager.py` transport selection**

In `backend/realtime/manager.py`, inside `ConnectionManager.start`, replace:

```python
        target = self._subscriber_loop if settings.USE_CELERY else self._local_loop
```

with:

```python
        target = self._subscriber_loop if settings.EVENT_BUS == "redis" else self._local_loop
```

In `_subscriber_loop`, replace the client URL:

```python
                client = aioredis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
```

with the event-bus resolver:

```python
                from backend.realtime.events import _event_bus_url

                client = aioredis.from_url(_event_bus_url(settings), decode_responses=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_event_bus_split.py tests/test_realtime_local_bus.py tests/test_realtime_manager.py -q`
Expected: PASS.

- [ ] **Step 6: Static checks**

Run: `ruff check backend/realtime/events.py backend/realtime/manager.py tests/test_event_bus_split.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add backend/realtime/events.py backend/realtime/manager.py tests/test_event_bus_split.py
git commit -m "refactor(realtime): drive event transport from EVENT_BUS

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 8: Foundation gate — full targeted regression + docs

**Files:**
- Modify: `.env.example` (document the two new keys)
- Test: run the job/realtime/config test subset

**Interfaces:** none new.

- [ ] **Step 1: Document the new environment keys**

Append to `.env.example` after the existing `USE_CELERY`/`CELERY_BROKER_URL` lines:

```bash
# --- Execution backend (plan 01) ---
# EXECUTION_BACKEND selects where pipeline jobs run: local | celery | ray.
# EVENT_BUS selects realtime transport: local | redis (independent of execution).
# Leaving USE_CELERY set keeps working: USE_CELERY=true implies EXECUTION_BACKEND=celery
# and EVENT_BUS=redis unless you set them explicitly.
EXECUTION_BACKEND=local
EVENT_BUS=local
# EVENT_BUS_URL=redis://localhost:6379/0
```

- [ ] **Step 2: Run the foundation regression subset**

Run:
```bash
pytest tests/test_sync_session_module.py tests/test_config_execution_backend.py \
  tests/test_execution_backend_contract.py tests/test_execution_backend_dispatch.py \
  tests/test_event_bus_split.py tests/test_job_manager_base.py \
  tests/test_jobs_router_retry.py tests/test_pipeline_task.py \
  tests/test_run_pipeline_helpers.py -q
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
git commit -m "docs(config): document EXECUTION_BACKEND and EVENT_BUS keys

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Definition of Done (Foundation Gate)

- `EXECUTION_BACKEND` and `EVENT_BUS` exist, default to `local`, and are derived from legacy `USE_CELERY` when only that is set.
- A neutral `backend/database/sync_session.py` is the single source of synchronous sessions; both task modules delegate to it and existing patch points still work.
- The typed `ExecutionBackend` protocol plus `LocalExecutionBackend`/`CeleryExecutionBackend` adapters exist behind `get_execution_backend`; selecting `ray` raises (no silent fallback).
- `/run` dispatch, retry resubmission, and job cancellation all go through the backend abstraction — no route or manager imports `celery_app` directly except the Celery adapter.
- Realtime transport is chosen by `EVENT_BUS`; Redis remains available for events even when Celery is not the executor.
- No public API path/model changed; the cancellation late-write guard and retry endpoint semantics are unchanged. Celery remains a working rollback target.
- Full `ruff` / `ruff format --check` / `ty` gate is green.
