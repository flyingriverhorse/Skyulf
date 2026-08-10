# Ray Jobs Pipeline Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Ray as an optional dependency and implement a `RayExecutionBackend` that submits a self-contained pipeline driver (`python -m backend.ray_jobs.run_pipeline --job-id … --attempt-id …`) via a mockable Ray Jobs adapter, where the driver loads all configuration from the database by id (never dataframes or secrets on the command line) and runs the existing pipeline execution path.

**Architecture:** FastAPI submits one Ray Job per attempt through `ray.job_submission.JobSubmissionClient`, wrapped behind a `RayJobClient` adapter so unit tests never need a live cluster. The Ray submission id becomes the attempt's external execution id. The driver process opens a synchronous DB session, loads the logical job's stored graph, and calls the same `execute_pipeline` used by the Celery/local paths — so results, artifacts, progress, and the attempt lifecycle stay identical. Production Ray mode requires PostgreSQL and S3-compatible storage and never silently falls back to in-process execution.

**Tech Stack:** Python 3.12, Ray Jobs (`ray[default]>=2.40,<3.0`, optional), FastAPI, SQLAlchemy 2.0 sync sessions, argparse, pytest with mocked Ray client.

## Global Constraints

- Add Ray as an **optional** dependency only (`ray[default]>=2.40,<3.0`); it is installed in Ray images/environments during coexistence, never pulled into the default backend install. Run the repository's Trivy/Codacy dependency scan after changing dependency manifests.
- The Ray driver loads the pipeline graph, dataset references, and storage configuration **from the database by `job_id`/`attempt_id`**. No dataframe contents, credentials, or secrets appear in command-line arguments, job metadata, or logs.
- `ray.job_submission.JobSubmissionClient` is used **only** through the `RayJobClient` adapter; every unit test mocks the adapter.
- Production Ray mode requires PostgreSQL and an S3-compatible artifact bucket. Selecting `ray` in a production environment without both raises at backend construction — production never silently falls back to in-process execution. Local Ray mode may use SQLite/local storage.
- Preserve the plan-01 execution contract (`submit`/`cancel`/`status`/`logs`, external id), the plan-02 attempt lifecycle (external id recorded on the attempt, `submission_failed` on submit error, `cancel_requested → cancelled`), the DB-as-truth model, the local fallback, and the Celery rollback path.
- No public API path or response model changes; no config/response shape exposed to the frontend changes in this plan, so no frontend files are touched.
- Target Python 3.12 idioms and full typing; avoid `Any` where a concrete type exists. Every new function/method has a 1–2 line docstring.
- Every implementation task follows TDD and ends with a focused commit whose message includes:
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`
- After Python changes run, in order:
  - `ruff check .`
  - `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
  - `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
- Depends on plans 01 (backend registry, `dispatch_branches`) and 02 (`ExecutionAttempt`, attempt transitions in `execute_pipeline`).

---

## File Structure

Create:

- `backend/config/mixins/execution.py` additions — `RAY_ADDRESS`, `RAY_JOB_ENTRYPOINT_PYTHON`, `RAY_NAMESPACE` (edit of the plan-01 file).
- `backend/ml_pipeline/_execution/backends/ray_client.py` — `RayJobClientProtocol` + `RayJobClient` adapter wrapping `JobSubmissionClient`.
- `backend/ml_pipeline/_execution/backends/ray.py` — `RayExecutionBackend`.
- `backend/ray_jobs/__init__.py` — package marker.
- `backend/ray_jobs/run_pipeline.py` — the Ray driver entrypoint.
- `requirements-ray.txt` — optional Ray requirements installed only in Ray images.
- `tests/test_ray_client_adapter.py`, `tests/test_ray_execution_backend.py`, `tests/test_ray_registry_guard.py`, `tests/test_ray_driver_run_pipeline.py`.

Modify:

- `pyproject.toml` — add a `ray` optional-dependencies extra.
- `backend/ml_pipeline/_execution/backends/registry.py` — register `ray` with the production guard.

---

### Task 1: Add Ray as an optional dependency + Ray settings

**Files:**
- Modify: `pyproject.toml:66-93` (add a `ray` extra)
- Create: `requirements-ray.txt`
- Modify: `backend/config/mixins/execution.py` (add Ray settings)
- Test: `tests/test_config_execution_backend.py` (extend)

**Interfaces:**
- Produces on `Settings`:
  - `RAY_ADDRESS: str | None` (default `None`) — Ray dashboard/head address for `JobSubmissionClient`.
  - `RAY_JOB_ENTRYPOINT_PYTHON: str` (default `"python"`).
  - `RAY_NAMESPACE: str | None` (default `None`).

- [ ] **Step 1: Write the failing test (extend `tests/test_config_execution_backend.py`)**

```python
def test_ray_settings_defaults():
    """Ray connection settings default to unset/local-friendly values."""
    s = _settings()
    assert s.RAY_ADDRESS is None
    assert s.RAY_JOB_ENTRYPOINT_PYTHON == "python"
    assert s.RAY_NAMESPACE is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_execution_backend.py -v -k ray_settings`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'RAY_ADDRESS'`.

- [ ] **Step 3: Add Ray settings to `ExecutionMixin`**

Append to `backend/config/mixins/execution.py`'s `ExecutionMixin`:

```python
    # ── Ray Jobs runtime (plan 03) ───────────────────────────────────────────
    # Ray dashboard/head address for JobSubmissionClient (e.g.
    # "http://ray-head:8265"). Required when EXECUTION_BACKEND=ray.
    RAY_ADDRESS: str | None = None
    # Python interpreter used in the Ray Job entrypoint command.
    RAY_JOB_ENTRYPOINT_PYTHON: str = "python"
    # Optional Ray namespace for submitted jobs.
    RAY_NAMESPACE: str | None = None
```

- [ ] **Step 4: Add the optional dependency to `pyproject.toml`**

In `[project.optional-dependencies]`, add:

```toml
ray = [
    "ray[default]>=2.40,<3.0",
]
```

Create `requirements-ray.txt`:

```text
# Ray Jobs runtime dependencies — installed only in Ray head/worker images
# during Celery→Ray coexistence, never in the default backend install.
ray[default]>=2.40,<3.0
```

- [ ] **Step 5: Run the dependency/security scan (manifest changed)**

Run the repository's Codacy Trivy dependency scan on the workspace root (and the local `pyscan-rs` scanner already in the dev group):
```bash
pyscan --path .
```
Then run the Codacy CLI Trivy analysis for the changed manifests (`pyproject.toml`, `requirements-ray.txt`) as required by the repository's dependency-scan rule. Resolve any high/critical advisory introduced by `ray[default]` before proceeding.

- [ ] **Step 6: Run the config test to verify it passes**

Run: `pytest tests/test_config_execution_backend.py -v`
Expected: PASS.

- [ ] **Step 7: Static checks**

Run: `ruff check backend/config/mixins/execution.py tests/test_config_execution_backend.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml requirements-ray.txt backend/config/mixins/execution.py tests/test_config_execution_backend.py
git commit -m "build(ray): add optional ray[default] dependency and Ray settings

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: RayJobClient adapter (mockable)

**Files:**
- Create: `backend/ml_pipeline/_execution/backends/ray_client.py`
- Test: `tests/test_ray_client_adapter.py`

**Interfaces:**
- Produces:
  - `class RayJobClientProtocol(Protocol)` with `submit_job(entrypoint: str, *, submission_id: str, metadata: dict[str, str], runtime_env: dict[str, Any] | None) -> str`, `stop_job(submission_id: str) -> bool`, `get_job_status(submission_id: str) -> str`, `get_job_logs(submission_id: str) -> str`.
  - `class RayJobClient` — concrete adapter constructing `ray.job_submission.JobSubmissionClient(address)` lazily and delegating.
- Consumes: `Settings.RAY_ADDRESS` (Task 1). Ray itself is imported lazily so importing this module never requires Ray to be installed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ray_client_adapter.py`:

```python
"""Unit tests for the Ray Jobs client adapter (no live cluster; SDK is patched)."""

import sys
import types
from unittest.mock import MagicMock

from backend.ml_pipeline._execution.backends.ray_client import RayJobClient


def _install_fake_ray(monkeypatch, fake_client):
    """Install a fake ray.job_submission.JobSubmissionClient into sys.modules."""
    mod = types.ModuleType("ray.job_submission")
    mod.JobSubmissionClient = MagicMock(return_value=fake_client)
    ray_pkg = types.ModuleType("ray")
    ray_pkg.job_submission = mod
    monkeypatch.setitem(sys.modules, "ray", ray_pkg)
    monkeypatch.setitem(sys.modules, "ray.job_submission", mod)
    return mod.JobSubmissionClient


def test_submit_delegates_to_sdk(monkeypatch):
    """submit_job forwards entrypoint/submission_id/metadata to the SDK."""
    fake = MagicMock()
    fake.submit_job.return_value = "skyulf-j1-a1"
    ctor = _install_fake_ray(monkeypatch, fake)

    client = RayJobClient(address="http://ray-head:8265")
    got = client.submit_job(
        "python -m backend.ray_jobs.run_pipeline --job-id j1 --attempt-id a1",
        submission_id="skyulf-j1-a1",
        metadata={"skyulf_job_id": "j1", "skyulf_attempt_id": "a1"},
        runtime_env=None,
    )
    assert got == "skyulf-j1-a1"
    ctor.assert_called_once_with("http://ray-head:8265")
    fake.submit_job.assert_called_once()
    kwargs = fake.submit_job.call_args.kwargs
    assert kwargs["submission_id"] == "skyulf-j1-a1"
    assert kwargs["metadata"]["skyulf_job_id"] == "j1"


def test_status_and_logs_and_stop(monkeypatch):
    """status/logs/stop delegate to the SDK and coerce status to str."""
    fake = MagicMock()
    fake.get_job_status.return_value = "RUNNING"
    fake.get_job_logs.return_value = "log line"
    fake.stop_job.return_value = True
    _install_fake_ray(monkeypatch, fake)

    client = RayJobClient(address="http://ray-head:8265")
    assert client.get_job_status("skyulf-j1-a1") == "RUNNING"
    assert client.get_job_logs("skyulf-j1-a1") == "log line"
    assert client.stop_job("skyulf-j1-a1") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ray_client_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: ...backends.ray_client`.

- [ ] **Step 3: Implement the adapter**

Create `backend/ml_pipeline/_execution/backends/ray_client.py`:

```python
"""Adapter around Ray's JobSubmissionClient.

All Ray Jobs SDK access goes through this thin adapter so the execution
backend and its unit tests never touch the real SDK directly. Ray is imported
lazily inside the constructor: importing this module must not require Ray to be
installed (Ray lives only in Ray images during coexistence).
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RayJobClientProtocol(Protocol):
    """The subset of the Ray Jobs SDK the execution backend depends on."""

    def submit_job(
        self,
        entrypoint: str,
        *,
        submission_id: str,
        metadata: dict[str, str],
        runtime_env: dict[str, Any] | None,
    ) -> str:
        """Submit a Ray Job and return its submission id."""
        ...

    def stop_job(self, submission_id: str) -> bool:
        """Request the Ray Job stop; return whether the stop was accepted."""
        ...

    def get_job_status(self, submission_id: str) -> str:
        """Return the Ray Job status as a string (PENDING/RUNNING/…)."""
        ...

    def get_job_logs(self, submission_id: str) -> str:
        """Return the Ray Job's captured logs."""
        ...


class RayJobClient:
    """Concrete Ray Jobs client bound to a dashboard/head address."""

    def __init__(self, address: str) -> None:
        """Create the underlying JobSubmissionClient, importing Ray lazily."""
        from ray.job_submission import JobSubmissionClient  # noqa: PLC0415

        self._client = JobSubmissionClient(address)

    def submit_job(
        self,
        entrypoint: str,
        *,
        submission_id: str,
        metadata: dict[str, str],
        runtime_env: dict[str, Any] | None,
    ) -> str:
        """Submit a Ray Job with a deterministic submission id and metadata."""
        return self._client.submit_job(
            entrypoint=entrypoint,
            submission_id=submission_id,
            metadata=metadata,
            runtime_env=runtime_env,
        )

    def stop_job(self, submission_id: str) -> bool:
        """Request termination of the Ray Job."""
        return bool(self._client.stop_job(submission_id))

    def get_job_status(self, submission_id: str) -> str:
        """Return the Ray Job status coerced to a plain string."""
        return str(self._client.get_job_status(submission_id))

    def get_job_logs(self, submission_id: str) -> str:
        """Return the Ray Job's logs."""
        return str(self._client.get_job_logs(submission_id))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ray_client_adapter.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/backends/ray_client.py tests/test_ray_client_adapter.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add backend/ml_pipeline/_execution/backends/ray_client.py tests/test_ray_client_adapter.py
git commit -m "feat(ray): add mockable Ray Jobs client adapter

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: RayExecutionBackend

**Files:**
- Create: `backend/ml_pipeline/_execution/backends/ray.py`
- Test: `tests/test_ray_execution_backend.py`

**Interfaces:**
- Consumes: `ExecutionBackend`, `ExecutionRequest`, `ExecutionHandle`, `ExecutionState` (plan 01); `RayJobClientProtocol` (Task 2); `Settings.RAY_ADDRESS`, `Settings.RAY_JOB_ENTRYPOINT_PYTHON`.
- Produces:
  - `class RayExecutionBackend` — `name = "ray"`; injects a `RayJobClientProtocol` (defaults to a lazily-built `RayJobClient`).
  - `submit` builds the entrypoint `"<python> -m backend.ray_jobs.run_pipeline --job-id <id> --attempt-id <id>"`, uses a deterministic `submission_id = f"skyulf-{job_id}-{attempt_id}"`, passes only ids in metadata, and `runtime_env=None` (no runtime pip installs), returning the submission id as the external id.
  - `cancel`/`status`/`logs` delegate to the adapter; `status` maps Ray status strings to `ExecutionState`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ray_execution_backend.py`:

```python
"""Unit tests for RayExecutionBackend with a mocked Ray client."""

from unittest.mock import MagicMock

import pytest

from backend.ml_pipeline._execution.backends.base import (
    ExecutionRequest,
    ExecutionState,
)
from backend.ml_pipeline._execution.backends.ray import RayExecutionBackend


def _backend(client):
    """Build a RayExecutionBackend with an injected fake client."""
    return RayExecutionBackend(client=client, entrypoint_python="python")


def test_submit_builds_entrypoint_and_returns_submission_id():
    """submit passes a no-secrets entrypoint and returns the submission id."""
    client = MagicMock()
    client.submit_job.return_value = "skyulf-j1-a1"
    backend = _backend(client)

    handle = backend.submit(ExecutionRequest(job_id="j1", attempt_id="a1", payload={"nodes": []}))

    assert handle.external_execution_id == "skyulf-j1-a1"
    call = client.submit_job.call_args
    entrypoint = call.args[0]
    assert entrypoint == "python -m backend.ray_jobs.run_pipeline --job-id j1 --attempt-id a1"
    assert call.kwargs["submission_id"] == "skyulf-j1-a1"
    assert call.kwargs["metadata"] == {"skyulf_job_id": "j1", "skyulf_attempt_id": "a1"}
    assert call.kwargs["runtime_env"] is None
    # The dataframe/payload is never placed on the command line.
    assert "nodes" not in entrypoint


@pytest.mark.parametrize(
    ("ray_status", "expected"),
    [
        ("PENDING", ExecutionState.PENDING),
        ("RUNNING", ExecutionState.RUNNING),
        ("SUCCEEDED", ExecutionState.SUCCEEDED),
        ("FAILED", ExecutionState.FAILED),
        ("STOPPED", ExecutionState.STOPPED),
        ("SomethingElse", ExecutionState.MISSING),
    ],
)
def test_status_maps_ray_states(ray_status, expected):
    """Ray status strings map to the neutral ExecutionState."""
    client = MagicMock()
    client.get_job_status.return_value = ray_status
    assert _backend(client).status("skyulf-j1-a1") is expected


def test_cancel_and_logs_delegate():
    """cancel and logs forward to the Ray client."""
    client = MagicMock()
    client.get_job_logs.return_value = "driver log"
    backend = _backend(client)
    backend.cancel("skyulf-j1-a1")
    client.stop_job.assert_called_once_with("skyulf-j1-a1")
    assert backend.logs("skyulf-j1-a1") == "driver log"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ray_execution_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: ...backends.ray`.

- [ ] **Step 3: Implement `RayExecutionBackend`**

Create `backend/ml_pipeline/_execution/backends/ray.py`:

```python
"""Ray Jobs execution backend.

Submits one Ray Job per attempt running the self-contained driver
``backend.ray_jobs.run_pipeline``. Only ids travel on the command line and in
metadata — the driver loads the graph, dataset references, and storage config
from the database. The Ray submission id is the attempt's external execution
id; a deterministic id (``skyulf-<job>-<attempt>``) makes resubmission
idempotent and links the Ray Job back to Skyulf.
"""

from backend.config import get_settings
from backend.ml_pipeline._execution.backends.base import (
    ExecutionHandle,
    ExecutionRequest,
    ExecutionState,
)
from backend.ml_pipeline._execution.backends.ray_client import RayJobClientProtocol

_RAY_STATE_MAP: dict[str, ExecutionState] = {
    "PENDING": ExecutionState.PENDING,
    "RUNNING": ExecutionState.RUNNING,
    "SUCCEEDED": ExecutionState.SUCCEEDED,
    "FAILED": ExecutionState.FAILED,
    "STOPPED": ExecutionState.STOPPED,
}


class RayExecutionBackend:
    """Submit and control pipeline attempts as Ray Jobs."""

    name = "ray"

    def __init__(
        self,
        client: RayJobClientProtocol | None = None,
        entrypoint_python: str | None = None,
    ) -> None:
        """Build the backend, defaulting the client to a live RayJobClient."""
        settings = get_settings()
        self._entrypoint_python = entrypoint_python or settings.RAY_JOB_ENTRYPOINT_PYTHON
        if client is not None:
            self._client: RayJobClientProtocol = client
        else:
            from backend.ml_pipeline._execution.backends.ray_client import (  # noqa: PLC0415
                RayJobClient,
            )

            if not settings.RAY_ADDRESS:
                raise ValueError("RAY_ADDRESS must be set to submit Ray Jobs")
            self._client = RayJobClient(settings.RAY_ADDRESS)

    def submit(self, request: ExecutionRequest) -> ExecutionHandle:
        """Submit the Ray driver for this attempt with only ids on the CLI."""
        submission_id = f"skyulf-{request.job_id}-{request.attempt_id}"
        entrypoint = (
            f"{self._entrypoint_python} -m backend.ray_jobs.run_pipeline "
            f"--job-id {request.job_id} --attempt-id {request.attempt_id}"
        )
        returned = self._client.submit_job(
            entrypoint,
            submission_id=submission_id,
            metadata={"skyulf_job_id": request.job_id, "skyulf_attempt_id": request.attempt_id},
            runtime_env=None,
        )
        return ExecutionHandle(external_execution_id=returned or submission_id)

    def cancel(self, external_execution_id: str) -> None:
        """Stop the Ray Job for this external id."""
        self._client.stop_job(external_execution_id)

    def status(self, external_execution_id: str) -> ExecutionState:
        """Map the Ray Job status to a neutral ExecutionState."""
        return _RAY_STATE_MAP.get(self._client.get_job_status(external_execution_id), ExecutionState.MISSING)

    def logs(self, external_execution_id: str) -> str:
        """Return the Ray Job's driver logs."""
        return self._client.get_job_logs(external_execution_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ray_execution_backend.py -v`
Expected: PASS (all, including parametrized status cases).

- [ ] **Step 5: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/backends/ray.py tests/test_ray_execution_backend.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add backend/ml_pipeline/_execution/backends/ray.py tests/test_ray_execution_backend.py
git commit -m "feat(ray): add RayExecutionBackend submitting per-attempt Ray Jobs

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Register Ray in the backend registry with the production guard

**Files:**
- Modify: `backend/ml_pipeline/_execution/backends/registry.py`
- Test: `tests/test_ray_registry_guard.py`

**Interfaces:**
- Consumes: `RayExecutionBackend` (Task 3); `Settings.environment_name`, `Settings.DATABASE_URL`, `Settings.S3_ARTIFACT_BUCKET`, `Settings.RAY_ADDRESS`.
- Produces: `get_execution_backend(settings)` returns a `RayExecutionBackend` for `EXECUTION_BACKEND=ray`, raising `RayBackendConfigurationError` in a production environment unless PostgreSQL and S3 are configured. Local environments may use SQLite/local storage.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ray_registry_guard.py`:

```python
"""The registry registers Ray and enforces production PostgreSQL + S3 requirements."""

from unittest.mock import MagicMock, patch

import pytest

from backend.config.base import Settings
from backend.ml_pipeline._execution.backends.registry import (
    RayBackendConfigurationError,
    get_execution_backend,
)


def _settings(**env: object) -> Settings:
    """Settings with a valid secret and overrides."""
    return Settings(SECRET_KEY="x" * 32, **env)


def test_local_ray_allows_sqlite(monkeypatch):
    """A development environment may run Ray with SQLite/local storage."""
    monkeypatch.setenv("FASTAPI_ENV", "development")
    s = _settings(EXECUTION_BACKEND="ray", RAY_ADDRESS="http://ray-head:8265")
    with patch(
        "backend.ml_pipeline._execution.backends.registry.RayExecutionBackend",
        return_value=MagicMock(name="ray"),
    ) as mock_ray:
        backend = get_execution_backend(s)
    assert backend is mock_ray.return_value


def test_production_ray_requires_postgres_and_s3(monkeypatch):
    """Production Ray without PostgreSQL + S3 raises rather than falling back."""
    monkeypatch.setenv("FASTAPI_ENV", "production")
    s = _settings(
        EXECUTION_BACKEND="ray",
        RAY_ADDRESS="http://ray-head:8265",
        DATABASE_URL="sqlite+aiosqlite:///./x.db",
        ENVIRONMENT="production",
    )
    with pytest.raises(RayBackendConfigurationError, match="PostgreSQL"):
        get_execution_backend(s)


def test_production_ray_ok_with_postgres_and_s3(monkeypatch):
    """Production Ray with PostgreSQL + S3 configured constructs the backend."""
    monkeypatch.setenv("FASTAPI_ENV", "production")
    s = _settings(
        EXECUTION_BACKEND="ray",
        RAY_ADDRESS="http://ray-head:8265",
        DATABASE_URL="postgresql+asyncpg://u:p@h:5432/db",
        S3_ARTIFACT_BUCKET="skyulf-artifacts",
        ENVIRONMENT="production",
    )
    with patch(
        "backend.ml_pipeline._execution.backends.registry.RayExecutionBackend",
        return_value=MagicMock(name="ray"),
    ):
        assert get_execution_backend(s).name is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ray_registry_guard.py -v`
Expected: FAIL — `ImportError: cannot import name 'RayBackendConfigurationError'` and `ray` still raises `ValueError`.

- [ ] **Step 3: Register Ray + add the production guard**

In `backend/ml_pipeline/_execution/backends/registry.py`, add the exception and Ray branch:

```python
class RayBackendConfigurationError(RuntimeError):
    """Raised when the Ray backend is selected without required production infra."""


def _is_postgres(database_url: str) -> bool:
    """Return True when the configured database is PostgreSQL."""
    return database_url.startswith(("postgresql", "postgres"))


def _validate_production_ray(settings: Settings) -> None:
    """Enforce PostgreSQL + S3 for Ray in production; never silently fall back."""
    if settings.environment_name != "production":
        return
    if not _is_postgres(settings.DATABASE_URL):
        raise RayBackendConfigurationError(
            "Production Ray mode requires PostgreSQL (set DB_TYPE=postgres / DATABASE_URL)."
        )
    if not settings.S3_ARTIFACT_BUCKET:
        raise RayBackendConfigurationError(
            "Production Ray mode requires S3-compatible shared storage (set S3_ARTIFACT_BUCKET)."
        )
```

Replace the `raise ValueError(...)` tail of `get_execution_backend` with a Ray branch:

```python
    if name == "ray":
        from backend.ml_pipeline._execution.backends.ray import (  # noqa: PLC0415
            RayExecutionBackend,
        )

        _validate_production_ray(settings)
        return RayExecutionBackend()
    raise ValueError(f"EXECUTION_BACKEND={name!r} is not a known execution backend")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ray_registry_guard.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Guard against regressions in the plan-01 registry tests**

Run: `pytest tests/test_execution_backend_contract.py -q`
Expected: PASS — `local`/`celery` selection is unchanged; the `ray` test that previously expected a `ValueError` is superseded by `tests/test_ray_registry_guard.py`; remove the now-obsolete `test_registry_rejects_ray_until_plan_03` from `tests/test_execution_backend_contract.py`.

- [ ] **Step 6: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/backends/registry.py tests/test_ray_registry_guard.py tests/test_execution_backend_contract.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add backend/ml_pipeline/_execution/backends/registry.py tests/test_ray_registry_guard.py tests/test_execution_backend_contract.py
git commit -m "feat(ray): register ray backend with production PostgreSQL+S3 guard

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Ray driver entrypoint (loads config by id, runs pipeline)

**Files:**
- Create: `backend/ray_jobs/__init__.py`
- Create: `backend/ray_jobs/run_pipeline.py`
- Test: `tests/test_ray_driver_run_pipeline.py`

**Interfaces:**
- Consumes: `backend.database.sync_session.get_sync_session`; `JobStrategyFactory.find_job`; `execute_pipeline`; `ExecutionAttemptRepository`, `AttemptStatus`; `JobStatus`.
- Produces:
  - `def _parse_args(argv: list[str] | None) -> argparse.Namespace` with `--job-id`, `--attempt-id`.
  - `def _payload_from_job(job) -> dict[str, Any]` — reconstructs the pipeline payload from the job's stored graph (no secrets).
  - `def run(session: Session, job_id: str, attempt_id: str) -> int` — orchestrates load → cancel-check → execute → exit code.
  - `def main(argv: list[str] | None = None) -> int` — process entrypoint.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ray_driver_run_pipeline.py`:

```python
"""Tests for the Ray driver: it loads config by id from the DB and runs the pipeline."""

from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.database.models import Base, ExecutionAttempt, TrainingJob
from backend.ray_jobs import run_pipeline as driver


def _session_with_job(status="queued"):
    """Build a sync in-memory session with one job + queued attempt."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add(TrainingJob(id="j1", pipeline_id="pipe", node_id="n1", dataset_source_id="d",
                            status=status, run_mode="fixed", model_type="rf",
                            graph={"pipeline_id": "pipe", "nodes": [
                                {"node_id": "n1", "step_type": "data_loader",
                                 "params": {"dataset_id": "d"}, "inputs": []}], "metadata": {}}))
    session.add(ExecutionAttempt(id="a1", job_id="j1", attempt_number=1, backend="ray",
                                 status="queued", is_final=False))
    session.commit()
    return session


def test_parse_args_reads_ids():
    """The driver parses --job-id and --attempt-id."""
    ns = driver._parse_args(["--job-id", "j1", "--attempt-id", "a1"])
    assert ns.job_id == "j1" and ns.attempt_id == "a1"


def test_payload_excludes_secrets():
    """The reconstructed payload is the stored graph with no credentials."""
    session = _session_with_job()
    job = session.get(TrainingJob, "j1")
    payload = driver._payload_from_job(job)
    assert payload["pipeline_id"] == "pipe"
    assert payload["nodes"][0]["node_id"] == "n1"
    assert "storage_options" not in payload
    session.close()


def test_run_invokes_execute_pipeline_and_returns_zero_on_success():
    """A successful run calls execute_pipeline with the DB payload and exits 0."""
    session = _session_with_job()

    def _fake_execute(job_id, payload, sess):
        sess.get(TrainingJob, job_id).status = "completed"
        sess.commit()

    with patch("backend.ray_jobs.run_pipeline.execute_pipeline", side_effect=_fake_execute) as mock_exec:
        rc = driver.run(session, "j1", "a1")
    assert rc == 0
    assert mock_exec.call_args.args[0] == "j1"
    assert mock_exec.call_args.args[1]["pipeline_id"] == "pipe"
    session.close()


def test_run_returns_nonzero_on_failure():
    """A failed job makes the driver exit nonzero so Ray reports FAILED."""
    session = _session_with_job()

    def _fake_execute(job_id, payload, sess):
        sess.get(TrainingJob, job_id).status = "failed"
        sess.commit()

    with patch("backend.ray_jobs.run_pipeline.execute_pipeline", side_effect=_fake_execute):
        rc = driver.run(session, "j1", "a1")
    assert rc == 1
    session.close()


def test_run_finalizes_cancel_requested_without_executing():
    """If cancellation was requested before the driver started, it finalizes cancelled."""
    session = _session_with_job(status="cancel_requested")
    with patch("backend.ray_jobs.run_pipeline.execute_pipeline") as mock_exec:
        rc = driver.run(session, "j1", "a1")
    mock_exec.assert_not_called()
    assert session.get(TrainingJob, "j1").status == "cancelled"
    assert session.get(ExecutionAttempt, "a1").status == "cancelled"
    assert rc == 0
    session.close()


def test_run_returns_nonzero_when_job_missing():
    """A missing job id is a hard driver failure (nonzero exit)."""
    session = _session_with_job()
    assert driver.run(session, "does-not-exist", "a1") == 1
    session.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ray_driver_run_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.ray_jobs'`.

- [ ] **Step 3: Implement the driver**

Create `backend/ray_jobs/__init__.py`:

```python
"""Ray Job driver entrypoints.

Modules here run inside Ray worker processes. They load everything they need
from the database by id and never accept dataframe contents or secrets on the
command line.
"""
```

Create `backend/ray_jobs/run_pipeline.py`:

```python
"""Ray Job driver: run one pipeline attempt loaded from the database.

Invoked as ``python -m backend.ray_jobs.run_pipeline --job-id … --attempt-id …``.
Loads the logical job's stored graph, runs the same ``execute_pipeline`` used by
the Celery/local paths, and exits 0 on success / nonzero on failure so Ray Jobs
reports SUCCEEDED/FAILED consistently with the durable DB state.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from backend.database.models import MLJob
from backend.database.sync_session import get_sync_session
from backend.ml_pipeline._execution.attempts import AttemptStatus, ExecutionAttemptRepository
from backend.ml_pipeline._execution.schemas import JobStatus
from backend.ml_pipeline._execution.strategies import JobStrategyFactory
from backend.ml_pipeline._services.pipeline_execution_service import execute_pipeline
from backend.utils.logging_utils import setup_universal_logging

logger = logging.getLogger(__name__)

_TERMINAL_OK = {JobStatus.COMPLETED.value, JobStatus.SUCCEEDED.value, JobStatus.CANCELLED.value}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the driver's --job-id / --attempt-id arguments."""
    parser = argparse.ArgumentParser(prog="backend.ray_jobs.run_pipeline")
    parser.add_argument("--job-id", required=True, dest="job_id")
    parser.add_argument("--attempt-id", required=True, dest="attempt_id")
    return parser.parse_args(argv)


def _payload_from_job(job: MLJob) -> dict[str, Any]:
    """Reconstruct the pipeline payload from the job's stored graph (no secrets)."""
    graph = job.graph if isinstance(job.graph, dict) else {}
    return {
        "pipeline_id": graph.get("pipeline_id", job.pipeline_id),
        "nodes": graph.get("nodes", []),
        "metadata": graph.get("metadata", {}),
    }


def _finalize_cancelled(session: Session, job: MLJob, attempt_id: str) -> None:
    """Confirm a pre-start cancellation on the job and its attempt."""
    job.status = JobStatus.CANCELLED.value
    job.finished_at = datetime.now(UTC)
    ExecutionAttemptRepository.mark_terminal_sync(session, attempt_id, AttemptStatus.CANCELLED)
    session.commit()


def run(session: Session, job_id: str, attempt_id: str) -> int:
    """Load the job, honor a pre-start cancel, run the pipeline, and return an exit code."""
    job, strategy = JobStrategyFactory.find_job(session, job_id)
    if job is None or strategy is None:
        logger.error("Ray driver: job %s not found", job_id)
        return 1

    if job.status in (JobStatus.CANCEL_REQUESTED.value, JobStatus.CANCELLED.value):
        logger.info("Ray driver: job %s already cancelled before start", job_id)
        _finalize_cancelled(session, job, attempt_id)
        return 0

    latest = ExecutionAttemptRepository.latest_attempt_sync(session, job_id)
    if latest is not None and latest.id != attempt_id:
        logger.warning("Ray driver: attempt mismatch (arg=%s latest=%s)", attempt_id, latest.id)

    payload = _payload_from_job(job)
    execute_pipeline(job_id, payload, session)

    session.refresh(job, ["status"])
    return 0 if job.status in _TERMINAL_OK else 1


def main(argv: list[str] | None = None) -> int:
    """Process entrypoint: set up logging, open a session, and run one attempt."""
    setup_universal_logging(
        log_file="logs/ray_driver.log", log_level="INFO", console_log_level="INFO"
    )
    args = _parse_args(argv)
    session = get_sync_session()
    try:
        return run(session, args.job_id, args.attempt_id)
    finally:
        session.close()


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ray_driver_run_pipeline.py -v`
Expected: PASS (all).

- [ ] **Step 5: Static checks**

Run: `ruff check backend/ray_jobs/ tests/test_ray_driver_run_pipeline.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add backend/ray_jobs/__init__.py backend/ray_jobs/run_pipeline.py tests/test_ray_driver_run_pipeline.py
git commit -m "feat(ray): add Ray Job driver that loads config by id and runs the pipeline

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: End-to-end submission flow through Ray (mocked)

**Files:**
- Test: `tests/test_ray_submission_flow.py`

**Interfaces:** none new — verifies plan-01/02 `dispatch_branches` + plan-03 Ray backend interoperate: a Ray submission records the Ray submission id on the attempt, and a submit failure marks `submission_failed`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ray_submission_flow.py`:

```python
"""End-to-end (mocked) Ray submission: dispatch records the Ray submission id on the attempt."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import MagicMock, patch

from backend.config.base import Settings
from backend.database.models import Base, ExecutionAttempt, TrainingJob
from backend.ml_pipeline._execution.attempts import ExecutionAttemptRepository
from backend.ml_pipeline._execution.backends.dispatch import dispatch_branches


def _settings(**env: object) -> Settings:
    """Settings selecting the Ray backend with a fake address."""
    return Settings(SECRET_KEY="x" * 32, EXECUTION_BACKEND="ray",
                    RAY_ADDRESS="http://ray-head:8265", **env)


@pytest.mark.asyncio
async def test_ray_dispatch_records_submission_id_on_attempt():
    """dispatch_branches with the Ray backend records the Ray submission id."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        session.add(TrainingJob(id="j1", pipeline_id="p", node_id="n", dataset_source_id="d",
                                status="queued", run_mode="fixed", model_type="rf", graph={}))
        await session.commit()
        await ExecutionAttemptRepository.create_initial_attempt(session, "j1", "ray")

        fake_client = MagicMock()
        fake_client.submit_job.return_value = "skyulf-j1-att"
        from backend.ml_pipeline._execution.backends.ray import RayExecutionBackend

        with patch(
            "backend.ml_pipeline._execution.backends.dispatch.get_execution_backend",
            return_value=RayExecutionBackend(client=fake_client, entrypoint_python="python"),
        ):
            await dispatch_branches([("j1", {"nodes": []})], settings=_settings(), db=session)

        attempt = await ExecutionAttemptRepository.latest_attempt(session, "j1")
        assert attempt is not None
        assert attempt.external_execution_id == "skyulf-j1-att"
    await engine.dispose()


@pytest.mark.asyncio
async def test_ray_submit_failure_marks_submission_failed():
    """A Ray submit exception flips the job/attempt to submission_failed."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        session.add(TrainingJob(id="j2", pipeline_id="p", node_id="n", dataset_source_id="d",
                                status="queued", run_mode="fixed", model_type="rf", graph={}))
        await session.commit()
        await ExecutionAttemptRepository.create_initial_attempt(session, "j2", "ray")

        fake_client = MagicMock()
        fake_client.submit_job.side_effect = RuntimeError("cluster unreachable")
        from backend.ml_pipeline._execution.backends.ray import RayExecutionBackend

        with patch(
            "backend.ml_pipeline._execution.backends.dispatch.get_execution_backend",
            return_value=RayExecutionBackend(client=fake_client, entrypoint_python="python"),
        ):
            await dispatch_branches([("j2", {})], settings=_settings(), db=session)

        job = await session.get(TrainingJob, "j2")
        attempt = await ExecutionAttemptRepository.latest_attempt(session, "j2")
        assert job.status == "submission_failed"
        assert attempt.status == "submission_failed"
    await engine.dispose()
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `pytest tests/test_ray_submission_flow.py -v`
Expected: initially may FAIL if `publish_job_event` requires Redis — patch it if needed. Because `EVENT_BUS` defaults to `local`, `publish_job_event` uses the in-process bus (a no-op when no listener is attached), so no patching is needed. Expected final: PASS (2 passed).

- [ ] **Step 3: Static checks**

Run: `ruff check tests/test_ray_submission_flow.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ray_submission_flow.py
git commit -m "test(ray): cover end-to-end Ray submission and submission-failure paths

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: Ray runtime gate — docs + regression

**Files:**
- Modify: `.env.example` (document `EXECUTION_BACKEND=ray` + `RAY_ADDRESS`)
- Modify: `ray-migration/README.md` only if a link is missing (handled by the parent reviewer)

- [ ] **Step 1: Document Ray env keys**

Append to `.env.example` under the execution-backend section:

```bash
# --- Ray Jobs runtime (plan 03) ---
# Set EXECUTION_BACKEND=ray and point RAY_ADDRESS at the Ray dashboard/head.
# Production Ray mode additionally requires DB_TYPE=postgres and S3_ARTIFACT_BUCKET.
# RAY_ADDRESS=http://ray-head:8265
# RAY_JOB_ENTRYPOINT_PYTHON=python
# RAY_NAMESPACE=skyulf
```

- [ ] **Step 2: Run the Ray runtime regression subset**

Run:
```bash
pytest tests/test_ray_client_adapter.py tests/test_ray_execution_backend.py \
  tests/test_ray_registry_guard.py tests/test_ray_driver_run_pipeline.py \
  tests/test_ray_submission_flow.py tests/test_execution_backend_contract.py \
  tests/test_execution_backend_dispatch.py -q
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
git commit -m "docs(ray): document EXECUTION_BACKEND=ray and RAY_ADDRESS

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Definition of Done (Ray Runtime Gate)

- `ray[default]>=2.40,<3.0` is an optional extra (`pyproject.toml` `ray` extra + `requirements-ray.txt`), installed only in Ray images; the dependency scan is clean.
- `RayJobClient` wraps `JobSubmissionClient` and is the only Ray SDK touchpoint; every test mocks it.
- `RayExecutionBackend` submits `python -m backend.ray_jobs.run_pipeline --job-id … --attempt-id …` with a deterministic submission id, ids-only metadata, and `runtime_env=None`; status maps Ray states; the Ray submission id becomes the attempt's external id.
- The registry returns Ray for `EXECUTION_BACKEND=ray`; production requires PostgreSQL + S3 and raises otherwise — never a silent in-process fallback.
- The Ray driver loads the graph from the DB by id (no dataframes/secrets on the CLI), runs the shared `execute_pipeline`, honors a pre-start cancel, and returns exit codes consistent with the durable job status.
- A whole pipeline can be submitted, queried, stopped, and completed on a single-node Ray cluster (validated end-to-end with the mocked client; live-cluster integration is exercised in plan 05).
- Full backend `ruff` / `ruff format --check` / `ty` gate is green.
