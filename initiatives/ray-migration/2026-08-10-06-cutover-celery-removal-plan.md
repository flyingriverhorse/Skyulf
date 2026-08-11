# Cutover and Celery Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove Ray/Celery result parity, roll Ray out per workload behind a feature flag, provide a drain-and-rollback boundary, then — only after the cutover gate passes — delete all Celery code, dependencies, and configuration while keeping Redis available as the event bus.

**Architecture:** A parity comparator diffs two completed jobs (schema, model, metrics within tolerance, status) so identical graphs run on Celery and Ray can be compared. A per-workload rollout flag routes selected job types to Ray while others stay on the previous backend (Celery remains a working rollback). A drain procedure waits for active attempts to finish before a backend switch. Once acceptance gates pass, Celery is removed in ordered steps: convert the `@shared_task`/`@celery_app.task` entrypoints to plain functions dispatched via FastAPI BackgroundTasks (EDA/ingestion) or the local backend (pipeline), delete the Celery app/worker/beat and the `CeleryExecutionBackend`, drop the `celery[redis]` dependency and `USE_CELERY`/`CELERY_*` settings, and update every reader to `EVENT_BUS`/`EXECUTION_BACKEND`.

**Tech Stack:** Python 3.12, FastAPI BackgroundTasks, Ray (execution), Redis (event bus, retained), pytest, Docker Compose, GitHub Actions.

## Global Constraints

- **Ordering is a hard gate:** Tasks 6–8 (Celery deletion) run **only after** Tasks 1–5 (parity, rollout flag, dual-run, drain, cutover gate) pass and Ray has demonstrated a measured benefit in production-like tests. If Ray shows no benefit, stop after Task 4 and keep the backend abstraction with Celery intact.
- Rollback to Celery must remain possible until the moment Task 7 deletes Celery. Before that boundary, flipping `EXECUTION_BACKEND` (or the rollout flag) back is a complete rollback.
- Redis is **not** removed by this plan; it stays the event bus (`EVENT_BUS=redis`). Only Celery (broker/worker/beat/result-backend usage) is removed.
- Preserve every public API path/response model, the DB-as-truth model, the cancellation late-write guard, retry semantics, the WebSocket invalidator pattern, the local fallback, and reconciliation.
- Run the repository's Trivy/Codacy dependency scan after removing `celery[redis]` from the manifests.
- Target Python 3.12 idioms and full typing; avoid `Any` where a concrete type exists. Every new function/method has a 1–2 line docstring.
- Every implementation task follows TDD and ends with a focused commit whose message includes:
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`
- **Validation commands change at Task 7** because `celery_worker.py` is deleted:
  - Tasks 1–6 (Celery still present): `ruff check .`; `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`; `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`.
  - Tasks 7–8 (Celery removed): `ruff check .`; `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py run_scheduler.py`; `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py run_scheduler.py`. Update `.github/workflows/pr_check.yml` to match (Task 8).
- `EXECUTION_BACKEND`/status enum shape exposed to the frontend does not change in this plan; no frontend files are touched.
- Depends on plans 01–05.

---

## File Structure

Create:

- `backend/ml_pipeline/_execution/parity.py` — `ParityReport`, `parity_input_from_job`, `compare_job_results`.
- `backend/ml_pipeline/_execution/rollout.py` — `resolve_backend_name_for_job`, `get_execution_backend_for_job`.
- `backend/ml_pipeline/_execution/drain.py` — `DrainReport`, `drain_active_attempts`.
- `tests/test_parity_compare.py`, `tests/test_rollout_flag.py`, `tests/test_dual_run.py`, `tests/test_drain.py`, `tests/test_celery_removed.py`.

Modify (Tasks 2–8):

- `backend/config/mixins/execution.py` — add `RAY_ROLLOUT_JOB_TYPES`; later drop celery-derivation.
- `backend/ml_pipeline/_execution/backends/dispatch.py` — per-job backend selection.
- `backend/ml_pipeline/tasks.py`, `backend/data_ingestion/tasks.py`, `backend/eda/tasks.py`, `backend/eda/router.py`, `backend/data_ingestion/service.py`, `backend/monitoring/tasks.py` — remove Celery task decorators/dispatch.
- `backend/ml_pipeline/_execution/backends/registry.py`, `backend/config/mixins/celery.py`, `backend/config/base.py`, `backend/realtime/events.py`, `backend/realtime/manager.py`, `backend/realtime/local_bus.py`, `backend/health/routes.py` — drop `USE_CELERY`/`CELERY_*`.
- `pyproject.toml`, `requirements-fastapi.txt`, `docker-compose.yml`, `.github/workflows/pr_check.yml`, `.env.example` — remove Celery dependency/config.
- Delete: `backend/celery_app.py`, `celery_worker.py`, `backend/monitoring/tasks.py`, `backend/ml_pipeline/_execution/backends/celery.py`.

---

### Task 1: Parity comparison core

**Files:**
- Create: `backend/ml_pipeline/_execution/parity.py`
- Test: `tests/test_parity_compare.py`

**Interfaces:**
- Produces:
  - `@dataclass(slots=True) class ParityReport` — `passed: bool`, `mismatches: list[str]`, plus booleans `status_match`, `model_type_match`, `model_family_match`, `metrics_within_tolerance`, `best_params_match`, `schema_match`.
  - `def parity_input_from_job(job: JobInfo) -> dict[str, Any]` — extract the comparable fields from a `JobInfo`.
  - `def compare_job_results(a: dict[str, Any], b: dict[str, Any], *, metric_tolerance: float = 1e-6) -> ParityReport`.
- Consumes: `JobInfo` (plan 01 schemas).

- [ ] **Step 1: Write the failing test**

Create `tests/test_parity_compare.py`:

```python
"""Parity comparison of two completed job result snapshots."""

from backend.ml_pipeline._execution.parity import compare_job_results


def _snap(**over):
    """Build a comparable job snapshot with sensible defaults."""
    base = {
        "status": "completed",
        "model_type": "random_forest",
        "model_family": "classification",
        "metrics": {"accuracy": 0.900000, "f1": 0.850000},
        "best_params": {"n_estimators": 100},
        "output_columns": ["a", "b", "prediction"],
    }
    base.update(over)
    return base


def test_identical_snapshots_pass():
    """Identical snapshots produce a passing parity report."""
    report = compare_job_results(_snap(), _snap())
    assert report.passed is True
    assert report.mismatches == []


def test_metrics_within_tolerance_pass():
    """Metrics differing below the tolerance still pass."""
    report = compare_job_results(_snap(), _snap(metrics={"accuracy": 0.9000004, "f1": 0.85}),
                                 metric_tolerance=1e-3)
    assert report.metrics_within_tolerance is True
    assert report.passed is True


def test_metric_beyond_tolerance_fails():
    """A metric gap beyond tolerance fails and is reported."""
    report = compare_job_results(_snap(), _snap(metrics={"accuracy": 0.80, "f1": 0.85}))
    assert report.metrics_within_tolerance is False
    assert report.passed is False
    assert any("accuracy" in m for m in report.mismatches)


def test_model_and_schema_mismatch_fail():
    """Different model type or output schema fails parity."""
    report = compare_job_results(_snap(), _snap(model_type="xgboost", output_columns=["a"]))
    assert report.model_type_match is False
    assert report.schema_match is False
    assert report.passed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parity_compare.py -v`
Expected: FAIL — `ModuleNotFoundError: ...parity`.

- [ ] **Step 3: Implement the comparator**

Create `backend/ml_pipeline/_execution/parity.py`:

```python
"""Compare two completed job snapshots for Ray/Celery result parity.

Compares user-visible outcome fields — status, model type/family, metrics
(within a floating-point tolerance), best params, and output schema — and
returns a structured report listing any mismatches.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ParityReport:
    """Structured result of comparing two job snapshots."""

    passed: bool = True
    status_match: bool = True
    model_type_match: bool = True
    model_family_match: bool = True
    metrics_within_tolerance: bool = True
    best_params_match: bool = True
    schema_match: bool = True
    mismatches: list[str] = field(default_factory=list)


def parity_input_from_job(job: Any) -> dict[str, Any]:
    """Extract the comparable fields from a JobInfo into a plain snapshot dict."""
    metrics = job.metrics or {}
    graph = job.graph or {}
    return {
        "status": str(job.status),
        "model_type": job.model_type,
        "model_family": job.model_family,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "best_params": (job.result or {}).get("best_params") if job.result else None,
        "output_columns": (metrics.get("output_columns") if isinstance(metrics, dict) else None)
        or graph.get("output_columns"),
    }


def _metrics_within(a: dict[str, Any], b: dict[str, Any], tol: float) -> list[str]:
    """Return mismatch messages for numeric metrics differing beyond ``tol``."""
    mismatches: list[str] = []
    for key in set(a) | set(b):
        va, vb = a.get(key), b.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if abs(float(va) - float(vb)) > tol:
                mismatches.append(f"metric {key}: {va} vs {vb} (>|{tol}|)")
        elif va != vb:
            mismatches.append(f"metric {key}: {va} vs {vb}")
    return mismatches


def compare_job_results(
    a: dict[str, Any], b: dict[str, Any], *, metric_tolerance: float = 1e-6
) -> ParityReport:
    """Compare two job snapshots and return a parity report."""
    report = ParityReport()

    if a.get("status") != b.get("status"):
        report.status_match = False
        report.mismatches.append(f"status: {a.get('status')} vs {b.get('status')}")
    if a.get("model_type") != b.get("model_type"):
        report.model_type_match = False
        report.mismatches.append(f"model_type: {a.get('model_type')} vs {b.get('model_type')}")
    if a.get("model_family") != b.get("model_family"):
        report.model_family_match = False
        report.mismatches.append(f"model_family: {a.get('model_family')} vs {b.get('model_family')}")
    if a.get("best_params") != b.get("best_params"):
        report.best_params_match = False
        report.mismatches.append("best_params differ")
    if sorted(a.get("output_columns") or []) != sorted(b.get("output_columns") or []):
        report.schema_match = False
        report.mismatches.append("output_columns differ")

    metric_mismatches = _metrics_within(a.get("metrics") or {}, b.get("metrics") or {}, metric_tolerance)
    if metric_mismatches:
        report.metrics_within_tolerance = False
        report.mismatches.extend(metric_mismatches)

    report.passed = not report.mismatches
    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_parity_compare.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/parity.py tests/test_parity_compare.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add backend/ml_pipeline/_execution/parity.py tests/test_parity_compare.py
git commit -m "feat(cutover): add Ray/Celery job result parity comparator

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Per-workload rollout flag + backend selection

**Files:**
- Modify: `backend/config/mixins/execution.py` (add `RAY_ROLLOUT_JOB_TYPES`)
- Create: `backend/ml_pipeline/_execution/rollout.py`
- Modify: `backend/ml_pipeline/_execution/backends/dispatch.py` (select backend per job)
- Test: `tests/test_rollout_flag.py`

**Interfaces:**
- Produces:
  - Setting `RAY_ROLLOUT_JOB_TYPES: str` (comma-separated, default `""`).
  - `def resolve_backend_name_for_job(job_type: str, settings) -> str` — returns `"ray"` for gated job types, else `settings.EXECUTION_BACKEND`.
  - `def get_execution_backend_for_job(job_type: str, settings) -> ExecutionBackend`.
- Consumes: `get_execution_backend`, `get_execution_backend` registry construction with a forced name.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rollout_flag.py`:

```python
"""Per-workload rollout routes gated job types to Ray while others keep the default."""

from backend.config.base import Settings
from backend.ml_pipeline._execution.rollout import resolve_backend_name_for_job


def _settings(**env: object) -> Settings:
    """Settings with a valid secret and overrides."""
    return Settings(SECRET_KEY="x" * 32, **env)


def test_gated_job_type_routes_to_ray():
    """A job type listed in RAY_ROLLOUT_JOB_TYPES routes to Ray."""
    s = _settings(EXECUTION_BACKEND="celery", RAY_ROLLOUT_JOB_TYPES="tuning")
    assert resolve_backend_name_for_job("tuning", s) == "ray"
    assert resolve_backend_name_for_job("training", s) == "celery"


def test_no_gating_uses_default_backend():
    """With no rollout list, every job uses the default EXECUTION_BACKEND."""
    s = _settings(EXECUTION_BACKEND="local")
    assert resolve_backend_name_for_job("tuning", s) == "local"


def test_rollout_list_is_whitespace_tolerant():
    """The comma-separated rollout list tolerates spaces."""
    s = _settings(EXECUTION_BACKEND="celery", RAY_ROLLOUT_JOB_TYPES=" tuning , training ")
    assert resolve_backend_name_for_job("training", s) == "ray"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rollout_flag.py -v`
Expected: FAIL — `ModuleNotFoundError: ...rollout`.

- [ ] **Step 3: Add the setting**

Append to `backend/config/mixins/execution.py`'s `ExecutionMixin`:

```python
    # ── Ray rollout (plan 06) ────────────────────────────────────────────────
    # Comma-separated job types (e.g. "tuning" or "tuning,training") routed to
    # Ray during coexistence while other workloads keep EXECUTION_BACKEND.
    RAY_ROLLOUT_JOB_TYPES: str = ""
```

- [ ] **Step 4: Implement the resolver**

Create `backend/ml_pipeline/_execution/rollout.py`:

```python
"""Per-workload Ray rollout selection.

Lets specific job types run on Ray behind a flag while everything else keeps
the configured ``EXECUTION_BACKEND`` — so a workload can be piloted on Ray with
Celery still available as an immediate rollback for the rest.
"""

from backend.config import Settings
from backend.ml_pipeline._execution.backends.base import ExecutionBackend


def _rollout_set(settings: Settings) -> set[str]:
    """Parse the comma-separated rollout job-type list into a set."""
    return {p.strip() for p in settings.RAY_ROLLOUT_JOB_TYPES.split(",") if p.strip()}


def resolve_backend_name_for_job(job_type: str, settings: Settings) -> str:
    """Return the backend name for a job type, honoring the Ray rollout list."""
    if job_type in _rollout_set(settings):
        return "ray"
    return settings.EXECUTION_BACKEND


def get_execution_backend_for_job(job_type: str, settings: Settings) -> ExecutionBackend:
    """Construct the execution backend for a job type per the rollout flag."""
    from backend.ml_pipeline._execution.backends.registry import (  # noqa: PLC0415
        _construct_backend,
    )

    return _construct_backend(resolve_backend_name_for_job(job_type, settings), settings)
```

Refactor `backend/ml_pipeline/_execution/backends/registry.py` so a named constructor is reusable: extract the body of `get_execution_backend`'s name-switch into `def _construct_backend(name: str, settings: Settings) -> ExecutionBackend` and have `get_execution_backend` call it with `settings.EXECUTION_BACKEND`. The local singleton, celery, and ray branches (with the production guard) move into `_construct_backend` unchanged.

- [ ] **Step 5: Use per-job backend selection in the dispatcher**

In `backend/ml_pipeline/_execution/backends/dispatch.py`, replace the single `backend = get_execution_backend(settings)` with per-job selection based on the payload's `job_type` (already injected in plan 04):

```python
    from backend.ml_pipeline._execution.rollout import get_execution_backend_for_job

    for job_id, payload in task_payloads:
        job_row = await db.get(TrainingJob, job_id)
        job_type = "tuning" if job_row is not None and job_row.run_mode == "tuned" else "training"
        payload = {**payload, "job_type": job_type}
        payload.setdefault("pipeline_id", job_row.pipeline_id if job_row is not None else "")
        backend = get_execution_backend_for_job(job_type, settings)
        attempt = await ExecutionAttemptRepository.latest_attempt(db, job_id)
        # ... (unchanged submit/record/submission_failed logic) ...
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_rollout_flag.py tests/test_execution_backend_dispatch.py tests/test_ray_submission_flow.py -q`
Expected: PASS.

- [ ] **Step 7: Static checks**

Run: `ruff check backend/config/mixins/execution.py backend/ml_pipeline/_execution/rollout.py backend/ml_pipeline/_execution/backends/registry.py backend/ml_pipeline/_execution/backends/dispatch.py tests/test_rollout_flag.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add backend/config/mixins/execution.py backend/ml_pipeline/_execution/rollout.py backend/ml_pipeline/_execution/backends/registry.py backend/ml_pipeline/_execution/backends/dispatch.py tests/test_rollout_flag.py
git commit -m "feat(cutover): route gated job types to Ray behind a rollout flag

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Dual-run parity harness

**Files:**
- Modify: `backend/ml_pipeline/_execution/parity.py` (add `run_dual_comparison`)
- Test: `tests/test_dual_run.py`

**Interfaces:**
- Produces: `def run_dual_comparison(job_a: JobInfo, job_b: JobInfo, *, metric_tolerance: float = 1e-6) -> ParityReport` — builds snapshots from two completed `JobInfo` objects (one Ray, one Celery) and compares them.
- Consumes: `parity_input_from_job`, `compare_job_results` (Task 1).

- [ ] **Step 1: Write the failing test**

Create `tests/test_dual_run.py`:

```python
"""Dual-run harness compares two completed JobInfo results end-to-end."""

from backend.ml_pipeline._execution.parity import run_dual_comparison
from backend.ml_pipeline._execution.schemas import JobInfo, JobStatus


def _job(model_type="random_forest", acc=0.9):
    """Build a completed JobInfo with the given model/metrics."""
    return JobInfo(
        job_id="j", pipeline_id="p", node_id="n", job_type="training",
        status=JobStatus.COMPLETED, start_time=None,
        model_type=model_type, model_family="classification",
        metrics={"accuracy": acc}, result={"best_params": {"n_estimators": 100}},
    )


def test_dual_run_matching_jobs_pass():
    """Two equivalent jobs (Ray vs Celery) pass parity."""
    report = run_dual_comparison(_job(), _job(acc=0.9000001), metric_tolerance=1e-3)
    assert report.passed is True


def test_dual_run_divergent_jobs_fail():
    """Divergent model types fail parity and are reported."""
    report = run_dual_comparison(_job(model_type="xgboost"), _job())
    assert report.passed is False
    assert any("model_type" in m for m in report.mismatches)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dual_run.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_dual_comparison'`.

- [ ] **Step 3: Implement the harness (append to `parity.py`)**

```python
def run_dual_comparison(job_a: Any, job_b: Any, *, metric_tolerance: float = 1e-6) -> ParityReport:
    """Compare two completed JobInfo results (e.g. one Ray, one Celery) for parity."""
    return compare_job_results(
        parity_input_from_job(job_a),
        parity_input_from_job(job_b),
        metric_tolerance=metric_tolerance,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dual_run.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Document the dual-run procedure**

Add a short "Dual-run parity procedure" note to `ray-migration/README.md` describing: submit the same saved graph twice with `RAY_ROLLOUT_JOB_TYPES` forcing one job onto Ray and the other onto Celery, wait for both to reach `completed`, fetch both via `JobManager.get_job`, and call `run_dual_comparison`; record the `ParityReport`. (Docs only; no test.)

- [ ] **Step 6: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/parity.py tests/test_dual_run.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add backend/ml_pipeline/_execution/parity.py tests/test_dual_run.py ray-migration/README.md
git commit -m "feat(cutover): add dual-run parity harness and procedure

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Drain procedure + rollback boundary

**Files:**
- Create: `backend/ml_pipeline/_execution/drain.py`
- Test: `tests/test_drain.py`

**Interfaces:**
- Produces:
  - `@dataclass class DrainReport` — `drained: bool`, `remaining: int`, `waited_seconds: float`.
  - `def drain_active_attempts(session: Session, *, timeout_seconds: float, poll_seconds: float = 2.0, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> DrainReport` — polls until no active attempts remain or the timeout elapses.
- Consumes: `ExecutionAttemptRepository.active_attempts_with_external_id_sync` (plan 05) plus a count of all active attempts.

- [ ] **Step 1: Write the failing test**

Create `tests/test_drain.py`:

```python
"""Draining waits for active attempts to finish before a backend switch."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.database.models import Base, ExecutionAttempt, TrainingJob
from backend.ml_pipeline._execution.drain import drain_active_attempts


def _session(active=True):
    """Build a session with one job whose attempt is active or terminal."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add(TrainingJob(id="j1", pipeline_id="p", node_id="n", dataset_source_id="d",
                            status="running", run_mode="fixed", model_type="rf", graph={}))
    session.add(ExecutionAttempt(id="a1", job_id="j1", attempt_number=1, backend="ray",
                                 status="running" if active else "succeeded", is_final=not active))
    session.commit()
    return session


def test_drain_returns_immediately_when_idle():
    """With no active attempts, drain completes at once."""
    session = _session(active=False)
    report = drain_active_attempts(session, timeout_seconds=10, clock=lambda: 0.0, sleep=lambda _s: None)
    assert report.drained is True
    assert report.remaining == 0
    session.close()


def test_drain_times_out_when_attempts_stay_active():
    """A never-finishing attempt makes drain time out with remaining > 0."""
    session = _session(active=True)
    ticks = iter([0.0, 5.0, 11.0, 12.0])
    report = drain_active_attempts(
        session, timeout_seconds=10, poll_seconds=5,
        clock=lambda: next(ticks), sleep=lambda _s: None,
    )
    assert report.drained is False
    assert report.remaining == 1
    session.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_drain.py -v`
Expected: FAIL — `ModuleNotFoundError: ...drain`.

- [ ] **Step 3: Implement drain**

Create `backend/ml_pipeline/_execution/drain.py`:

```python
"""Drain active execution attempts before switching or removing a backend.

Used at the rollback boundary: before flipping ``EXECUTION_BACKEND`` (or before
removing Celery), wait for in-flight attempts to reach a terminal state so no
running work is orphaned by the switch.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from backend.database.models import ExecutionAttempt

_ACTIVE = ("queued", "running", "cancel_requested")


@dataclass(slots=True)
class DrainReport:
    """Outcome of a drain wait."""

    drained: bool
    remaining: int
    waited_seconds: float


def _count_active(session: Session) -> int:
    """Count attempts still in a non-terminal state."""
    return (
        session.query(ExecutionAttempt)
        .filter(ExecutionAttempt.status.in_(_ACTIVE))
        .count()
    )


def drain_active_attempts(
    session: Session,
    *,
    timeout_seconds: float,
    poll_seconds: float = 2.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> DrainReport:
    """Wait until no active attempts remain or the timeout elapses."""
    start = clock()
    remaining = _count_active(session)
    while remaining > 0:
        if (clock() - start) >= timeout_seconds:
            return DrainReport(drained=False, remaining=remaining, waited_seconds=clock() - start)
        sleep(poll_seconds)
        session.expire_all()
        remaining = _count_active(session)
    return DrainReport(drained=True, remaining=0, waited_seconds=clock() - start)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_drain.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Document the rollback boundary**

Add a "Rollback boundary" note to `ray-migration/README.md`: before Task 7, rollback = `drain_active_attempts` then set `EXECUTION_BACKEND=celery` (or clear `RAY_ROLLOUT_JOB_TYPES`); after Task 7 deletes Celery, the boundary is closed and rollback means reverting the deletion commits.

- [ ] **Step 6: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/drain.py tests/test_drain.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add backend/ml_pipeline/_execution/drain.py tests/test_drain.py ray-migration/README.md
git commit -m "feat(cutover): add drain procedure and document rollback boundary

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Cutover gate checklist (default switch)

**Files:**
- Modify: `.env.example` (document making Ray the default)
- Modify: `ray-migration/README.md` (cutover acceptance checklist)

**Interfaces:** none (documentation + operational gate). No code default is changed here — flipping the deployment `EXECUTION_BACKEND` to `ray` is an env change gated by the checklist.

- [ ] **Step 1: Record the cutover acceptance checklist**

Add a "Cutover acceptance gate" section to `ray-migration/README.md` requiring, before Tasks 6–8:
- Backend parity tests pass (Task 1/3) for representative training and tuning graphs within tolerance.
- Cancellation (`cancel_requested → cancelled`), retry (new attempt), reconciliation (worker/head loss), and no-silent-fallback tests pass on a live single-node Ray cluster.
- Measured benefit: Ray meets or beats Celery on queue wait + total runtime for the piloted workloads (record the numbers).
- Drain completes cleanly under load.
If any item fails, stop and keep Celery.

- [ ] **Step 2: Document the default switch**

Append to `.env.example`:

```bash
# --- Cutover (plan 06) ---
# After the cutover acceptance gate passes, make Ray the default by setting:
#   EXECUTION_BACKEND=ray
# Pilot individual workloads first with RAY_ROLLOUT_JOB_TYPES=tuning (Celery stays
# the rollback for everything else until the gate passes).
```

- [ ] **Step 3: Validate docs render (no code change)**

Run: `python -c "import pathlib; assert pathlib.Path('ray-migration/README.md').read_text().find('Cutover acceptance gate') != -1; print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add .env.example ray-migration/README.md
git commit -m "docs(cutover): add acceptance gate checklist and default-switch guidance

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: De-Celery the task modules (convert to plain functions)

> **Gate:** Do not start Task 6 until the Task 5 acceptance gate passes.

**Files:**
- Modify: `backend/ml_pipeline/tasks.py:8,82,101` (drop `@shared_task`)
- Modify: `backend/data_ingestion/tasks.py:6,145` (drop `@shared_task`)
- Modify: `backend/data_ingestion/service.py:300-303` (BackgroundTasks only)
- Modify: `backend/eda/tasks.py:242-259` (plain function; no `celery_app` task)
- Modify: `backend/eda/router.py:14,113-131` (BackgroundTasks only; drop `celery_app` import)
- Test: `tests/test_pipeline_task.py`, `tests/test_data_ingestion.py`, `tests/test_eda_tasks_extra.py` (adjust patch points if needed)

**Interfaces:**
- `run_pipeline_task`/`run_pipeline_batch_task`/`ingest_data_task`/`generate_profile` become **plain functions** (no Celery). The Local execution backend already calls `run_pipeline_task` directly; EDA and ingestion dispatch via FastAPI BackgroundTasks.

- [ ] **Step 1: Write/adjust the failing test**

Add to `tests/test_eda_tasks_extra.py` (or create it) a test that the EDA entrypoint is a plain callable and the router dispatches via BackgroundTasks:

```python
def test_generate_profile_is_plain_callable():
    """The EDA profile entrypoint is a plain function, not a Celery task."""
    from backend.eda import tasks

    assert callable(tasks.generate_profile)
    assert not hasattr(tasks.generate_profile, "delay")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eda_tasks_extra.py -v -k plain_callable`
Expected: FAIL — the current entrypoint is `generate_profile_celery` (a Celery task) guarded by an import.

- [ ] **Step 3: Convert `backend/ml_pipeline/tasks.py`**

Remove `from celery import shared_task` and the two `@shared_task(...)` decorators so `run_pipeline_task(job_id, pipeline_config_dict)` and `run_pipeline_batch_task(branches)` are plain functions. Their bodies are unchanged. (The Local backend and dispatcher already call them as plain callables.)

- [ ] **Step 4: Convert `backend/data_ingestion/tasks.py` + `service.py`**

Remove `from celery import shared_task` and the `@shared_task(...)` decorator from `ingest_data_task`. In `service.py::_trigger_ingestion`, drop the `if settings.USE_CELERY: ingest_data_task.delay(...)` branch, keeping the BackgroundTasks and thread-fallback branches:

```python
    def _trigger_ingestion(
        self, settings: Any, source_id: int, background_tasks: BackgroundTasks | None
    ) -> None:
        """Kick off ingestion for `source_id` via BackgroundTasks or a thread fallback."""
        if background_tasks:
            background_tasks.add_task(ingest_data_task, source_id)
        else:
            import asyncio

            _task = asyncio.create_task(asyncio.to_thread(ingest_data_task, source_id))
            self._retain_ingestion_task(_task, source_id)
```

(Preserve the existing `_on_done`/strong-reference logic; extract it into `_retain_ingestion_task` if it is currently inline, or keep it inline unchanged.)

- [ ] **Step 5: Convert `backend/eda/tasks.py` + `router.py`**

In `eda/tasks.py`, replace the Celery-guarded block with a plain function:

```python
def generate_profile(report_id: int) -> None:
    """Run EDA analysis for a report on a fresh event loop (BackgroundTasks entrypoint)."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_eda_background(report_id))
    finally:
        asyncio.set_event_loop(None)
        loop.close()
```

In `eda/router.py`, drop `from backend.celery_app import celery_app` and simplify `_dispatch_analysis_job`:

```python
async def _dispatch_analysis_job(
    report: EDAReport, background_tasks: BackgroundTasks, session: AsyncSession
) -> None:
    """Dispatch the EDA job via FastAPI BackgroundTasks."""
    background_tasks.add_task(generate_profile, report.id)
```

Update the import in `eda/router.py` to reference `generate_profile` instead of `generate_profile_celery`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_pipeline_task.py tests/test_data_ingestion.py tests/test_eda_tasks_extra.py tests/test_eda_api.py -q`
Expected: PASS. (If any test patched `...run_pipeline_task` as a Celery task attribute or `generate_profile_celery`, update the patch target to the plain function.)

- [ ] **Step 7: Static checks**

Run: `ruff check backend/ml_pipeline/tasks.py backend/data_ingestion/tasks.py backend/data_ingestion/service.py backend/eda/tasks.py backend/eda/router.py tests/test_eda_tasks_extra.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add backend/ml_pipeline/tasks.py backend/data_ingestion/tasks.py backend/data_ingestion/service.py backend/eda/tasks.py backend/eda/router.py tests/test_eda_tasks_extra.py
git commit -m "refactor(cutover): convert Celery tasks to plain BackgroundTasks callables

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: Delete Celery app/worker/beat, adapter, and settings

> **Gate:** Only after Task 6. This is the reversible-boundary deletion.

**Files:**
- Delete: `backend/celery_app.py`, `celery_worker.py`, `backend/monitoring/tasks.py`, `backend/ml_pipeline/_execution/backends/celery.py`
- Modify: `backend/ml_pipeline/_execution/backends/registry.py` (drop the celery branch), `backend/ml_pipeline/_execution/backends/base.py` (drop `"celery"` from the name Literal), `backend/config/mixins/execution.py` (`ExecutionBackendName`), `backend/config/mixins/celery.py` (drop `USE_CELERY`/`CELERY_*`), `backend/config/base.py` (simplify alias validator), `backend/realtime/events.py`, `backend/realtime/manager.py`, `backend/realtime/local_bus.py` (docstring), `backend/health/routes.py`
- Test: `tests/test_celery_removed.py`

**Interfaces:**
- `ExecutionBackendName` becomes `Literal["local", "ray"]`; selecting `celery` raises. `EVENT_BUS`/`EXECUTION_BACKEND` are the only knobs. Redis (event bus) is retained.

- [ ] **Step 1: Write the failing test**

Create `tests/test_celery_removed.py`:

```python
"""After cutover, Celery is fully removed and the app imports without it."""

import importlib

import pytest


def test_celery_modules_are_gone():
    """The Celery app/worker/adapter modules no longer import."""
    for mod in (
        "backend.celery_app",
        "backend.ml_pipeline._execution.backends.celery",
        "backend.monitoring.tasks",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)


def test_settings_have_no_celery_fields():
    """USE_CELERY / CELERY_* settings are removed."""
    from backend.config.base import Settings

    s = Settings(SECRET_KEY="x" * 32)
    for attr in ("USE_CELERY", "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND", "CELERY_TASK_DEFAULT_QUEUE"):
        assert not hasattr(s, attr)


def test_registry_rejects_celery():
    """Selecting the removed celery backend raises."""
    from backend.config.base import Settings
    from backend.ml_pipeline._execution.backends.registry import get_execution_backend

    with pytest.raises(ValueError, match="celery|not a known"):
        get_execution_backend(Settings(SECRET_KEY="x" * 32, EXECUTION_BACKEND="celery"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_celery_removed.py -v`
Expected: FAIL — the modules still import and settings still expose `USE_CELERY`.

- [ ] **Step 3: Delete Celery modules**

```bash
git rm backend/celery_app.py celery_worker.py backend/monitoring/tasks.py backend/ml_pipeline/_execution/backends/celery.py
```

- [ ] **Step 4: Remove the celery branch from the registry and the name Literal**

In `backend/ml_pipeline/_execution/backends/registry.py`, delete the `if name == "celery": return CeleryExecutionBackend()` branch (and its import) from `_construct_backend`. In `backend/config/mixins/execution.py`, change:

```python
ExecutionBackendName = Literal["local", "ray"]
```

and in `backend/ml_pipeline/_execution/backends/base.py`, if a name Literal exists, drop `"celery"`.

- [ ] **Step 5: Remove Celery settings and simplify the alias validator**

In `backend/config/mixins/celery.py`, delete `USE_CELERY`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, and `CELERY_TASK_DEFAULT_QUEUE` (keep `ERROR_LOG_RETENTION_DAYS`, `TUNING_*`, `JOB_*`, `MAX_PARALLEL_BRANCH_WORKERS`, monitoring caps). Rename the class docstring to "Background job lifecycle and tuning settings" (leave the class name to avoid churn, or rename to `JobsMixin` and update `backend/config/base.py`). In `backend/config/base.py`, delete the `sync_execution_backend_aliases` validator entirely (nothing to reconcile now).

- [ ] **Step 6: Update the remaining `USE_CELERY` readers**

- `backend/realtime/events.py`: `_event_bus_url` becomes `settings.EVENT_BUS_URL or settings.REDIS_URL or "redis://127.0.0.1:6379/0"` (drop the `CELERY_BROKER_URL` fallback). Its routing already uses `EVENT_BUS` (plan 01).
- `backend/realtime/manager.py`: already selects on `EVENT_BUS` (plan 01); ensure no `CELERY_BROKER_URL` reference remains (the `_event_bus_url` import handles the URL).
- `backend/realtime/local_bus.py`: update the module docstring to reference `EVENT_BUS=local` instead of `USE_CELERY=False`.
- `backend/health/routes.py`: replace the `if settings.USE_CELERY:` Redis cache check with `if settings.EVENT_BUS == "redis":`, and use `_event_bus_url(settings)` for the ping URL.

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_celery_removed.py tests/test_event_bus_split.py tests/test_realtime_manager.py tests/test_config_execution_backend.py -q`
Expected: PASS. Remove or update `tests/test_execution_backend_contract.py` cases that referenced the celery adapter, and any test importing `backend.celery_app`.

- [ ] **Step 8: Static checks (updated file list — no `celery_worker.py`)**

Run: `ruff check .`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py run_scheduler.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py run_scheduler.py`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat(cutover): remove Celery app, worker, beat, adapter, and settings

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 8: Remove the Celery dependency + update manifests, compose, CI, docs

**Files:**
- Modify: `pyproject.toml:24` (drop `celery[redis]`), `requirements-fastapi.txt:18` (drop `celery[redis]`)
- Modify: `docker-compose.yml` (remove the `worker` service + `USE_CELERY`/`CELERY_*` env)
- Modify: `.github/workflows/pr_check.yml:64,67` (swap `celery_worker.py` → `run_scheduler.py`)
- Modify: `.env.example` (remove `USE_CELERY`/`CELERY_*`)
- Test: dependency scan + full suite

**Interfaces:** none. Redis stays (event bus); only Celery leaves.

- [ ] **Step 1: Drop the dependency from the manifests**

In `pyproject.toml`, remove the line `"celery[redis]>=5.4.0,<6.0.0",` from `[project].dependencies`. In `requirements-fastapi.txt`, remove `celery[redis]>=5.4.0,<6.0.0` (keep `redis>=5.0.0,<6.0.0`).

- [ ] **Step 2: Run the dependency/security scan (manifest changed)**

Run the repository's Codacy Trivy scan on `pyproject.toml`/`requirements-fastapi.txt` and `pyscan --path .`. Removing a dependency should not introduce advisories; confirm the scan is clean.

- [ ] **Step 3: Update Docker Compose (dev)**

In `docker-compose.yml`, delete the `worker` service block entirely, and in the `api` service replace the Celery env with execution/event settings:

```yaml
    environment:
      FASTAPI_ENV: development
      HOST: 0.0.0.0
      PORT: 8000
      EXECUTION_BACKEND: local
      EVENT_BUS: redis
      EVENT_BUS_URL: redis://redis:6379/0
      REDIS_URL: redis://redis:6379/0
      DATABASE_URL: sqlite+aiosqlite:///./mlops_database.db
      MAX_UPLOAD_SIZE: "10737418240"
```

(Keep the `redis` service — it is the event bus. For a Ray dev stack use `docker-compose.ray.yml` from plan 05.)

- [ ] **Step 4: Update the CI static-check file list**

In `.github/workflows/pr_check.yml`, change the two check commands to drop `celery_worker.py` and add `run_scheduler.py`:

```yaml
      - name: Format check (Ruff)
        run: bash .github/scripts/run_check.sh "Ruff format" ruff format --check backend skyulf-core tests run_skyulf.py run_scheduler.py

      - name: Type check (Ty)
        run: bash .github/scripts/run_check.sh "Ty type check" ty check backend skyulf-core/skyulf skyulf-core/tests run_skyulf.py run_scheduler.py
```

- [ ] **Step 5: Remove Celery keys from `.env.example`**

Delete the `USE_CELERY`, `CELERY_BROKER_URL`, and any `CELERY_RESULT_BACKEND`/`CELERY_TASK_DEFAULT_QUEUE` lines from `.env.example`. Ensure `EXECUTION_BACKEND`, `EVENT_BUS`, and `EVENT_BUS_URL` remain documented.

- [ ] **Step 6: Full validation (post-cutover file list)**

Run the full backend suite plus the static gate:
```bash
pytest -q
ruff check .
ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py run_scheduler.py
ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py run_scheduler.py
```
Expected: all tests pass; no lint/format/type errors. Fix any residual `USE_CELERY`/`celery` reference the suite surfaces.

- [ ] **Step 7: Confirm no Celery references remain**

Run: `grep -rn "celery\|USE_CELERY\|shared_task\|celery_worker" backend/ *.py .github/ docker-compose.yml requirements-fastapi.txt pyproject.toml | grep -iv "ray-migration"`
Expected: no matches (or only historical CHANGELOG/docs entries outside code). Resolve any code match.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml requirements-fastapi.txt docker-compose.yml .github/workflows/pr_check.yml .env.example
git commit -m "build(cutover): drop celery[redis] dependency and update compose/CI/docs

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Definition of Done (Cutover Gate)

- A parity comparator + dual-run harness diff Ray vs Celery job results (schema, model, metrics within tolerance, status); a documented procedure records the report.
- A per-workload rollout flag routes selected job types to Ray while Celery remains a working rollback for the rest.
- A drain procedure waits for active attempts before a switch; the rollback boundary is documented and open until Task 7.
- The cutover acceptance checklist (parity + cancellation/retry/reconciliation/no-silent-fallback + measured benefit + clean drain) gates the deletion tasks; if unmet, Celery stays.
- After the gate: Celery tasks are plain BackgroundTasks callables; the Celery app/worker/beat/adapter and `USE_CELERY`/`CELERY_*` settings are deleted; the `celery[redis]` dependency is removed and the scan is clean; Redis remains the event bus.
- Compose, CI static-check file lists, and `.env.example` no longer reference Celery; no `celery`/`shared_task`/`USE_CELERY` reference remains in code.
- The full test suite passes and the post-cutover `ruff` / `ruff format --check` / `ty` gate (with `run_scheduler.py`, without `celery_worker.py`) is green.
