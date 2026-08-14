# Distributed Compute Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distribute independent pipeline branches across Ray workers by declaring per-attempt CPU/GPU resources on each Ray Job, align worker BLAS/estimator threads to the reserved CPUs to prevent oversubscription, run hyperparameter tuning through Ray's joblib backend (not Ray Tune), and enforce shared/durable artifact URI rules.

**Architecture:** Each branch is already submitted as its own Ray Job (plan 03), so branch parallelism comes from the Ray scheduler placing per-attempt jobs across the cluster. This plan makes that placement resource-aware: `RayExecutionBackend` declares `entrypoint_num_cpus`/`entrypoint_num_gpus`/`entrypoint_memory` per attempt and passes the reserved CPU count to the driver via `runtime_env` env vars (no package installs). The driver aligns `OMP/MKL/OpenBLAS/NUMEXPR` threads and estimator `n_jobs` to those CPUs. Tuning registers Ray's joblib backend and runs the existing searchers under `parallel_backend("ray")`, preserving one code path for local and distributed execution. Datasets/artifacts are exchanged as URIs; production Ray requires durable (S3) URIs.

**Tech Stack:** Python 3.12, Ray Core + `ray.util.joblib`, joblib/sklearn/XGBoost/LightGBM thread controls, SQLAlchemy sync sessions, pytest with mocked Ray.

## Global Constraints

- Do **not** adopt Ray Tune, Ray Data, Ray Train, or Ray Serve in this plan. Tuning uses Ray's joblib backend only; the existing search strategies, metrics, CV, time-series ordering, threshold tuning, wrappers, and result schemas are unchanged.
- Keep one tuning code path for local and distributed execution — Ray affects only the joblib `parallel_backend` and thread alignment, not the searcher logic.
- Every distributed task declares logical CPU/GPU/memory; the worker environment and estimators align (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, sklearn `n_jobs`, XGBoost/LightGBM threads, Ray `num_cpus`) so nested parallelism never oversubscribes a host.
- Workers receive dataset/artifact **URIs and ids**, never large dataframes by value or secrets. Production Ray requires durable S3-compatible URIs (local paths are rejected); local Ray may use local paths.
- `runtime_env` may carry env vars only — never runtime `pip`/`conda` installs (production package installation stays disabled).
- Preserve the plan 01–03 contracts (execution protocol, attempt lifecycle, Ray driver loading config by id), the DB-as-truth model, the local fallback, and the Celery rollback path.
- No public API path/response-model changes and no config/response shape exposed to the frontend changes in this plan, so no frontend files are touched.
- Target Python 3.12 idioms and full typing; avoid `Any` where a concrete type exists. Every new function/method has a 1–2 line docstring.
- Every implementation task follows TDD and ends with a focused commit whose message includes:
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`
- After Python changes run, in order:
  - `ruff check .`
  - `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
  - `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
- Depends on plans 01–03. Where this plan changes the tuning config injected in `backend/ml_pipeline/_execution/engine/_node_runners.py`, run the tuning test subset; skyulf-core's searcher code is unchanged, so its suite is a spot-check, not a full rerun.

---

## File Structure

Create:

- `backend/ml_pipeline/_execution/resources.py` — `ResourceSpec`, `align_thread_env`, `resource_spec_for_job`.
- `backend/ml_pipeline/_execution/backends/ray_joblib.py` — `register_ray_joblib` (idempotent).
- `backend/ml_pipeline/_execution/shared_inputs.py` — `ensure_durable_artifact_uri`, `ray_put_shared`.
- `tests/test_resource_alignment.py`, `tests/test_ray_joblib_backend.py`, `tests/test_shared_inputs.py`, `tests/test_ray_resource_submission.py`.

Modify:

- `backend/config/mixins/execution.py` — add `RAY_ENTRYPOINT_NUM_CPUS`, `RAY_ENTRYPOINT_NUM_GPUS`, `RAY_TUNING_NUM_CPUS`.
- `backend/ml_pipeline/_execution/backends/ray_client.py` — `submit_job` accepts entrypoint resource params.
- `backend/ml_pipeline/_execution/backends/ray.py` — declare resources + pass reserved CPU count via `runtime_env` env vars.
- `backend/ray_jobs/run_pipeline.py` — align threads from the reserved CPU env var; register the Ray joblib backend.
- `backend/ml_pipeline/_execution/engine/_node_runners.py:596-635` — inject `parallel_backend="ray"` when `EXECUTION_BACKEND=ray`.
- `tests/test_ray_execution_backend.py`, `tests/test_ray_client_adapter.py` — update `runtime_env`/resource expectations.

---

### Task 1: Resource spec + thread alignment

**Files:**
- Create: `backend/ml_pipeline/_execution/resources.py`
- Modify: `backend/config/mixins/execution.py` (add resource settings)
- Test: `tests/test_resource_alignment.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True, slots=True) class ResourceSpec` — `num_cpus: int`, `num_gpus: int`, `memory_mb: int | None`.
  - `def align_thread_env(num_threads: int, *, apply: bool = True) -> dict[str, str]` — sets/returns `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`.
  - `def resource_spec_for_job(job_type: str, settings) -> ResourceSpec` — training vs tuning CPU/GPU reservation.
- Consumes: new settings `RAY_ENTRYPOINT_NUM_CPUS`, `RAY_ENTRYPOINT_NUM_GPUS`, `RAY_TUNING_NUM_CPUS`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_resource_alignment.py`:

```python
"""Resource declarations and thread alignment prevent CPU oversubscription."""

import os

from backend.config.base import Settings
from backend.ml_pipeline._execution.resources import (
    ResourceSpec,
    align_thread_env,
    resource_spec_for_job,
)


def _settings(**env: object) -> Settings:
    """Settings with a valid secret and overrides."""
    return Settings(SECRET_KEY="x" * 32, **env)


def test_align_thread_env_returns_all_blas_vars():
    """All four BLAS/threadpool env vars are aligned to the requested count."""
    got = align_thread_env(3, apply=False)
    assert got == {
        "OMP_NUM_THREADS": "3",
        "MKL_NUM_THREADS": "3",
        "OPENBLAS_NUM_THREADS": "3",
        "NUMEXPR_NUM_THREADS": "3",
    }


def test_align_thread_env_applies_to_environ(monkeypatch):
    """apply=True writes the values into os.environ."""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(key, raising=False)
    align_thread_env(2, apply=True)
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "2"


def test_resource_spec_training_vs_tuning():
    """Tuning reserves more CPUs than a single fixed training fit."""
    s = _settings(RAY_ENTRYPOINT_NUM_CPUS=1, RAY_TUNING_NUM_CPUS=4)
    train = resource_spec_for_job("training", s)
    tune = resource_spec_for_job("tuning", s)
    assert isinstance(train, ResourceSpec)
    assert train.num_cpus == 1
    assert tune.num_cpus == 4
    assert train.num_gpus == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resource_alignment.py -v`
Expected: FAIL — `ModuleNotFoundError: ...resources` and missing settings.

- [ ] **Step 3: Add resource settings**

Append to `backend/config/mixins/execution.py`'s `ExecutionMixin`:

```python
    # ── Ray resource reservations (plan 04) ──────────────────────────────────
    # Logical CPUs/GPUs reserved for a single fixed-training Ray Job.
    RAY_ENTRYPOINT_NUM_CPUS: int = 1
    RAY_ENTRYPOINT_NUM_GPUS: int = 0
    # Logical CPUs reserved for a tuning Ray Job (parallel trials via Ray joblib).
    RAY_TUNING_NUM_CPUS: int = 4
```

- [ ] **Step 4: Implement `resources.py`**

Create `backend/ml_pipeline/_execution/resources.py`:

```python
"""Resource declarations and thread alignment for distributed execution.

Ray ``num_cpus`` is a scheduling reservation, not physical enforcement, so the
worker's BLAS threadpools and estimator ``n_jobs`` must be aligned to the same
number or nested parallelism oversubscribes the host. These helpers centralize
that alignment so training (single fit) and tuning (parallel trials) both stay
within their reserved CPU budget.
"""

import os
from dataclasses import dataclass

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """Logical resource reservation for one distributed execution."""

    num_cpus: int
    num_gpus: int
    memory_mb: int | None = None


def align_thread_env(num_threads: int, *, apply: bool = True) -> dict[str, str]:
    """Align all BLAS/threadpool env vars to ``num_threads``; optionally apply them."""
    value = str(max(1, num_threads))
    env = {key: value for key in _THREAD_ENV_VARS}
    if apply:
        os.environ.update(env)
    return env


def resource_spec_for_job(job_type: str, settings) -> ResourceSpec:
    """Return the CPU/GPU reservation for a job type (tuning reserves more CPUs)."""
    if job_type == "tuning":
        return ResourceSpec(
            num_cpus=settings.RAY_TUNING_NUM_CPUS,
            num_gpus=settings.RAY_ENTRYPOINT_NUM_GPUS,
        )
    return ResourceSpec(
        num_cpus=settings.RAY_ENTRYPOINT_NUM_CPUS,
        num_gpus=settings.RAY_ENTRYPOINT_NUM_GPUS,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_resource_alignment.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/resources.py backend/config/mixins/execution.py tests/test_resource_alignment.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add backend/ml_pipeline/_execution/resources.py backend/config/mixins/execution.py tests/test_resource_alignment.py
git commit -m "feat(ray): add resource spec and BLAS thread alignment helpers

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Declare per-attempt Ray resources + propagate reserved CPUs

**Files:**
- Modify: `backend/ml_pipeline/_execution/backends/ray_client.py` (`submit_job` resource params)
- Modify: `backend/ml_pipeline/_execution/backends/ray.py` (`submit` declares resources, passes CPU env var)
- Modify: `backend/ray_jobs/run_pipeline.py` (`main` aligns threads from the reserved CPU env var)
- Test: `tests/test_ray_resource_submission.py`; update `tests/test_ray_execution_backend.py`, `tests/test_ray_client_adapter.py`

**Interfaces:**
- Consumes: `ResourceSpec`, `resource_spec_for_job`, `align_thread_env` (Task 1); `Settings.RAY_*`.
- Produces:
  - `RayJobClient.submit_job(..., entrypoint_num_cpus: float | None, entrypoint_num_gpus: float | None, entrypoint_memory: int | None)` — forwards to the SDK.
  - `RayExecutionBackend.submit` derives a `ResourceSpec` from the payload's job type, submits with those entrypoint resources, and sets `runtime_env={"env_vars": {"SKYULF_NUM_CPUS": str(n)}}`.
  - Driver `main()` reads `SKYULF_NUM_CPUS` and calls `align_thread_env(n)` before executing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ray_resource_submission.py`:

```python
"""RayExecutionBackend declares entrypoint resources and propagates reserved CPUs."""

from unittest.mock import MagicMock

from backend.ml_pipeline._execution.backends.base import ExecutionRequest
from backend.ml_pipeline._execution.backends.ray import RayExecutionBackend


def test_training_submission_declares_one_cpu_and_env_var(monkeypatch):
    """A training attempt reserves 1 CPU and exports SKYULF_NUM_CPUS=1."""
    client = MagicMock()
    client.submit_job.return_value = "skyulf-j1-a1"
    monkeypatch.setenv("RAY_ENTRYPOINT_NUM_CPUS", "1")
    backend = RayExecutionBackend(client=client, entrypoint_python="python")

    backend.submit(ExecutionRequest(job_id="j1", attempt_id="a1", payload={"job_type": "training"}))

    kwargs = client.submit_job.call_args.kwargs
    assert kwargs["entrypoint_num_cpus"] == 1
    assert kwargs["runtime_env"] == {"env_vars": {"SKYULF_NUM_CPUS": "1"}}


def test_tuning_submission_declares_more_cpus(monkeypatch):
    """A tuning attempt reserves RAY_TUNING_NUM_CPUS CPUs."""
    client = MagicMock()
    client.submit_job.return_value = "skyulf-j2-a1"
    monkeypatch.setenv("RAY_TUNING_NUM_CPUS", "4")
    backend = RayExecutionBackend(client=client, entrypoint_python="python")

    backend.submit(ExecutionRequest(job_id="j2", attempt_id="a1", payload={"job_type": "tuning"}))

    kwargs = client.submit_job.call_args.kwargs
    assert kwargs["entrypoint_num_cpus"] == 4
    assert kwargs["runtime_env"]["env_vars"]["SKYULF_NUM_CPUS"] == "4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ray_resource_submission.py -v`
Expected: FAIL — `submit` does not yet pass `entrypoint_num_cpus`/env vars.

- [ ] **Step 3: Extend the Ray client adapter**

In `backend/ml_pipeline/_execution/backends/ray_client.py`, extend both the protocol and the concrete `submit_job` signatures:

```python
    def submit_job(
        self,
        entrypoint: str,
        *,
        submission_id: str,
        metadata: dict[str, str],
        runtime_env: dict[str, Any] | None,
        entrypoint_num_cpus: float | None = None,
        entrypoint_num_gpus: float | None = None,
        entrypoint_memory: int | None = None,
    ) -> str:
        """Submit a Ray Job with a deterministic submission id, metadata, and resources."""
        return self._client.submit_job(
            entrypoint=entrypoint,
            submission_id=submission_id,
            metadata=metadata,
            runtime_env=runtime_env,
            entrypoint_num_cpus=entrypoint_num_cpus,
            entrypoint_num_gpus=entrypoint_num_gpus,
            entrypoint_memory=entrypoint_memory,
        )
```

Apply the matching signature to `RayJobClientProtocol.submit_job` (same params, `...` body).

- [ ] **Step 4: Declare resources in `RayExecutionBackend.submit`**

In `backend/ml_pipeline/_execution/backends/ray.py`, update `submit`:

```python
    def submit(self, request: ExecutionRequest) -> ExecutionHandle:
        """Submit the Ray driver with declared resources and reserved-CPU env var."""
        from backend.config import get_settings  # noqa: PLC0415
        from backend.ml_pipeline._execution.resources import resource_spec_for_job  # noqa: PLC0415

        settings = get_settings()
        job_type = str(request.payload.get("job_type", "training"))
        spec = resource_spec_for_job(job_type, settings)

        submission_id = f"skyulf-{request.job_id}-{request.attempt_id}"
        entrypoint = (
            f"{self._entrypoint_python} -m backend.ray_jobs.run_pipeline "
            f"--job-id {request.job_id} --attempt-id {request.attempt_id}"
        )
        returned = self._client.submit_job(
            entrypoint,
            submission_id=submission_id,
            metadata={"skyulf_job_id": request.job_id, "skyulf_attempt_id": request.attempt_id},
            runtime_env={"env_vars": {"SKYULF_NUM_CPUS": str(spec.num_cpus)}},
            entrypoint_num_cpus=spec.num_cpus,
            entrypoint_num_gpus=spec.num_gpus or None,
            entrypoint_memory=spec.memory_mb,
        )
        return ExecutionHandle(external_execution_id=returned or submission_id)
```

The `job_type` on the payload is set at dispatch time from the job's `run_mode`. Update `dispatch_branches` (plan 01/02) to include it: when building the branch payload for submission, add `payload = {**payload, "job_type": <"tuning" if job.run_mode == "tuned" else "training">}`. Concretely, in `backend/ml_pipeline/_execution/backends/dispatch.py`, resolve the job's `run_mode` and inject `job_type` into the payload before `backend.submit(...)`:

```python
        job_row = await db.get(TrainingJob, job_id)
        job_type = "tuning" if job_row is not None and job_row.run_mode == "tuned" else "training"
        payload = {**payload, "job_type": job_type}
```

(`job_type` is a routing hint for resource sizing only; the driver ignores it and reads the graph from the DB.)

- [ ] **Step 5: Align threads in the driver**

In `backend/ray_jobs/run_pipeline.py`, at the start of `main()` (before opening the session), align threads to the reserved CPUs:

```python
    import os  # noqa: PLC0415

    from backend.ml_pipeline._execution.resources import align_thread_env  # noqa: PLC0415

    reserved_cpus = int(os.environ.get("SKYULF_NUM_CPUS", "1"))
    align_thread_env(reserved_cpus)
```

- [ ] **Step 6: Update plan-03 expectations for `runtime_env` and adapter resources**

In `tests/test_ray_execution_backend.py::test_submit_builds_entrypoint_and_returns_submission_id`, replace the `runtime_env is None` assertion with:

```python
    assert call.kwargs["runtime_env"] == {"env_vars": {"SKYULF_NUM_CPUS": "1"}}
    assert call.kwargs["entrypoint_num_cpus"] == 1
```

In `tests/test_ray_client_adapter.py::test_submit_delegates_to_sdk`, add resource kwargs to the call and assert they forward:

```python
    got = client.submit_job(
        "python -m backend.ray_jobs.run_pipeline --job-id j1 --attempt-id a1",
        submission_id="skyulf-j1-a1",
        metadata={"skyulf_job_id": "j1", "skyulf_attempt_id": "a1"},
        runtime_env={"env_vars": {"SKYULF_NUM_CPUS": "1"}},
        entrypoint_num_cpus=1,
    )
    assert fake.submit_job.call_args.kwargs["entrypoint_num_cpus"] == 1
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_ray_resource_submission.py tests/test_ray_execution_backend.py tests/test_ray_client_adapter.py tests/test_ray_driver_run_pipeline.py -q`
Expected: PASS.

- [ ] **Step 8: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/backends/ray_client.py backend/ml_pipeline/_execution/backends/ray.py backend/ml_pipeline/_execution/backends/dispatch.py backend/ray_jobs/run_pipeline.py tests/test_ray_resource_submission.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add backend/ml_pipeline/_execution/backends/ray_client.py backend/ml_pipeline/_execution/backends/ray.py backend/ml_pipeline/_execution/backends/dispatch.py backend/ray_jobs/run_pipeline.py tests/test_ray_resource_submission.py tests/test_ray_execution_backend.py tests/test_ray_client_adapter.py
git commit -m "feat(ray): declare per-attempt resources and align worker threads

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Ray joblib tuning backend

**Files:**
- Create: `backend/ml_pipeline/_execution/backends/ray_joblib.py`
- Modify: `backend/ray_jobs/run_pipeline.py` (register the backend in `main`)
- Modify: `backend/ml_pipeline/_execution/engine/_node_runners.py:596-635` (inject `parallel_backend="ray"` under Ray)
- Test: `tests/test_ray_joblib_backend.py`, `tests/test_node_runners_extra.py` (extend)

**Interfaces:**
- Produces:
  - `def register_ray_joblib() -> bool` — idempotently registers Ray's joblib backend; returns whether it registered this call.
  - `_prepare_tuning_config` sets `tuning_params["parallel_backend"] = "ray"` and `tuning_params["n_jobs"] = -1` when `EXECUTION_BACKEND == "ray"` and no explicit `TUNING_PARALLEL_BACKEND` override is set; otherwise unchanged.
- Consumes: `ray.util.joblib.register_ray` (lazy); the existing skyulf-core tuning engine, which already runs `with parallel_backend(config.parallel_backend): searcher.fit(...)` (`skyulf-core/skyulf/modeling/_tuning/engine.py:1034`) — no skyulf-core change needed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ray_joblib_backend.py`:

```python
"""Registering Ray's joblib backend is idempotent and lazy."""

import sys
import types
from unittest.mock import MagicMock

from backend.ml_pipeline._execution.backends import ray_joblib


def test_register_ray_joblib_calls_sdk_once(monkeypatch):
    """register_ray is called once; a second call is a no-op."""
    ray_joblib._REGISTERED = False  # reset module state
    fake_register = MagicMock()
    util = types.ModuleType("ray.util.joblib")
    util.register_ray = fake_register
    monkeypatch.setitem(sys.modules, "ray.util.joblib", util)

    assert ray_joblib.register_ray_joblib() is True
    assert ray_joblib.register_ray_joblib() is False
    fake_register.assert_called_once()
```

Extend `tests/test_node_runners_extra.py`:

```python
def test_tuning_config_uses_ray_backend_under_ray(monkeypatch):
    """When EXECUTION_BACKEND=ray, tuning runs under the ray joblib backend."""
    from unittest.mock import MagicMock

    from backend.ml_pipeline._execution.engine._node_runners import NodeRunnersMixin
    from backend.ml_pipeline._execution.schemas import NodeConfig

    fake_settings = MagicMock()
    fake_settings.TUNING_N_JOBS = 1
    fake_settings.TUNING_PARALLEL_BACKEND = ""
    fake_settings.EXECUTION_BACKEND = "ray"
    monkeypatch.setattr(
        "backend.ml_pipeline._execution.engine._node_runners.get_settings",
        lambda: fake_settings,
    )

    calc = MagicMock()
    calc.problem_type = "classification"
    calc.build_tuning_search_space.return_value = {}
    node = NodeConfig(node_id="n", step_type="training",
                      params={"tuning_config": {"strategy": "random"}}, inputs=[])
    runner = NodeRunnersMixin()
    cfg = runner._prepare_tuning_config(node, calc)
    assert cfg["parallel_backend"] == "ray"
    assert cfg["n_jobs"] == -1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ray_joblib_backend.py tests/test_node_runners_extra.py -v -k "ray"`
Expected: FAIL — `ModuleNotFoundError: ...ray_joblib`; tuning config still uses the raw settings values.

- [ ] **Step 3: Implement the joblib registration helper**

Create `backend/ml_pipeline/_execution/backends/ray_joblib.py`:

```python
"""Register Ray's joblib backend for distributed hyperparameter search.

Registering ``ray`` as a joblib backend lets the existing sklearn/Optuna
searchers fan trials across the Ray cluster via ``parallel_backend("ray")``
without any change to the search logic. Registration is process-wide and
idempotent; the Ray driver calls it once at startup.
"""

import logging

logger = logging.getLogger(__name__)

_REGISTERED = False


def register_ray_joblib() -> bool:
    """Register Ray's joblib backend once per process; return whether it registered now."""
    global _REGISTERED
    if _REGISTERED:
        return False
    from ray.util.joblib import register_ray  # noqa: PLC0415

    register_ray()
    _REGISTERED = True
    logger.info("Registered Ray joblib backend for distributed tuning")
    return True
```

- [ ] **Step 4: Register the backend in the Ray driver**

In `backend/ray_jobs/run_pipeline.py`'s `main()`, after `align_thread_env(...)` and before opening the session, register the joblib backend best-effort:

```python
    from backend.ml_pipeline._execution.backends.ray_joblib import (  # noqa: PLC0415
        register_ray_joblib,
    )

    try:
        register_ray_joblib()
    except Exception as exc:  # pragma: no cover - depends on ray availability
        logger.warning("Ray joblib backend registration skipped: %s", exc)
```

- [ ] **Step 5: Inject the Ray parallel backend in `_prepare_tuning_config`**

In `backend/ml_pipeline/_execution/engine/_node_runners.py`, replace the two injection lines:

```python
        tuning_params["n_jobs"] = settings.TUNING_N_JOBS
        tuning_params["parallel_backend"] = settings.TUNING_PARALLEL_BACKEND
```

with Ray-aware selection:

```python
        if settings.EXECUTION_BACKEND == "ray" and not settings.TUNING_PARALLEL_BACKEND:
            # Under Ray, run trials through Ray's joblib backend and let it use
            # all CPUs reserved for this Ray Job (aligned via SKYULF_NUM_CPUS).
            tuning_params["n_jobs"] = -1
            tuning_params["parallel_backend"] = "ray"
        else:
            tuning_params["n_jobs"] = settings.TUNING_N_JOBS
            tuning_params["parallel_backend"] = settings.TUNING_PARALLEL_BACKEND
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_ray_joblib_backend.py tests/test_node_runners_extra.py -q`
Expected: PASS.

- [ ] **Step 7: Spot-check the tuning path is behaviorally unchanged for local/celery**

Run: `pytest tests/test_cross_validation_all_methods.py tests/test_cv_basic_vs_advanced.py -q`
Expected: PASS — the local/celery branch of the injection is byte-identical to before, so tuning behavior is unchanged when `EXECUTION_BACKEND != "ray"`.

- [ ] **Step 8: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/backends/ray_joblib.py backend/ray_jobs/run_pipeline.py backend/ml_pipeline/_execution/engine/_node_runners.py tests/test_ray_joblib_backend.py tests/test_node_runners_extra.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add backend/ml_pipeline/_execution/backends/ray_joblib.py backend/ray_jobs/run_pipeline.py backend/ml_pipeline/_execution/engine/_node_runners.py tests/test_ray_joblib_backend.py tests/test_node_runners_extra.py
git commit -m "feat(ray): run hyperparameter tuning through Ray's joblib backend

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Shared/durable artifact URI rules

**Files:**
- Create: `backend/ml_pipeline/_execution/shared_inputs.py`
- Modify: `backend/ml_pipeline/_services/pipeline_execution_service.py:190-197` (validate the base artifact URI under Ray)
- Test: `tests/test_shared_inputs.py`

**Interfaces:**
- Produces:
  - `def ensure_durable_artifact_uri(uri: str, settings) -> str` — raises `NonDurableArtifactError` when a production Ray run resolves a non-`s3://` (local) artifact URI; returns the URI unchanged otherwise.
  - `def ray_put_shared(obj: object) -> object` — thin wrapper over `ray.put` for placing an immutable input in the object store once (for intra-node branch sharing); raises `RuntimeError` if Ray is not initialized.
  - `class NonDurableArtifactError(RuntimeError)`.
- Consumes: `Settings.EXECUTION_BACKEND`, `Settings.environment_name`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_shared_inputs.py`:

```python
"""Shared-input and durable-artifact URI rules for distributed execution."""

import sys
import types
from unittest.mock import MagicMock

import pytest

from backend.config.base import Settings
from backend.ml_pipeline._execution.shared_inputs import (
    NonDurableArtifactError,
    ensure_durable_artifact_uri,
    ray_put_shared,
)


def _settings(**env: object) -> Settings:
    """Settings with a valid secret and overrides."""
    return Settings(SECRET_KEY="x" * 32, **env)


def test_local_uri_rejected_under_production_ray(monkeypatch):
    """Production Ray must not resolve a local artifact path (not cross-node durable)."""
    monkeypatch.setenv("FASTAPI_ENV", "production")
    s = _settings(EXECUTION_BACKEND="ray", ENVIRONMENT="production")
    with pytest.raises(NonDurableArtifactError):
        ensure_durable_artifact_uri("/var/artifacts/job1", s)


def test_s3_uri_allowed_under_production_ray(monkeypatch):
    """An s3:// artifact URI is durable and accepted under production Ray."""
    monkeypatch.setenv("FASTAPI_ENV", "production")
    s = _settings(EXECUTION_BACKEND="ray", ENVIRONMENT="production")
    assert ensure_durable_artifact_uri("s3://bucket/job1", s) == "s3://bucket/job1"


def test_local_uri_allowed_for_local_backend():
    """Local/dev execution may use local artifact paths."""
    s = _settings(EXECUTION_BACKEND="local")
    assert ensure_durable_artifact_uri("/var/artifacts/job1", s) == "/var/artifacts/job1"


def test_ray_put_shared_delegates(monkeypatch):
    """ray_put_shared places the object via ray.put when Ray is initialized."""
    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: True
    fake_ray.put = MagicMock(return_value="ObjectRef(x)")
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    assert ray_put_shared({"df": 1}) == "ObjectRef(x)"
    fake_ray.put.assert_called_once()


def test_ray_put_shared_requires_initialized_ray(monkeypatch):
    """ray_put_shared refuses to run when Ray is not initialized."""
    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: False
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    with pytest.raises(RuntimeError):
        ray_put_shared({"df": 1})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_shared_inputs.py -v`
Expected: FAIL — `ModuleNotFoundError: ...shared_inputs`.

- [ ] **Step 3: Implement `shared_inputs.py`**

Create `backend/ml_pipeline/_execution/shared_inputs.py`:

```python
"""Shared-input and durable-artifact URI rules for distributed execution.

Ray workers receive URIs and ids, never large dataframes by value. For inputs
shared across branches on one node, place an immutable representation in the
object store once with ``ray.put``. Production Ray requires durable
(S3-compatible) artifact URIs — a local filesystem path is not visible to other
nodes and is rejected.
"""


class NonDurableArtifactError(RuntimeError):
    """Raised when a production Ray run resolves a non-durable (local) artifact URI."""


def ensure_durable_artifact_uri(uri: str, settings) -> str:
    """Return the URI unchanged, or raise when it is non-durable under production Ray."""
    if settings.EXECUTION_BACKEND == "ray" and settings.environment_name == "production":
        if not str(uri).startswith("s3://"):
            raise NonDurableArtifactError(
                "Production Ray mode requires an S3-compatible artifact URI; "
                f"got a non-durable path: {uri!r}"
            )
    return uri


def ray_put_shared(obj: object) -> object:
    """Place an immutable input in the Ray object store once and return its ref."""
    import ray  # noqa: PLC0415

    if not ray.is_initialized():
        raise RuntimeError("ray_put_shared requires an initialized Ray runtime")
    return ray.put(obj)
```

- [ ] **Step 4: Enforce the durable-URI rule in the execution service**

In `backend/ml_pipeline/_services/pipeline_execution_service.py`, right after the artifact store is created and `job.artifact_uri` is set (around line 195), validate it:

```python
        from backend.config import get_settings  # noqa: PLC0415
        from backend.ml_pipeline._execution.shared_inputs import ensure_durable_artifact_uri  # noqa: PLC0415

        ensure_durable_artifact_uri(base_artifact_uri, get_settings())
        job.artifact_uri = base_artifact_uri
        session.commit()
```

This raises early (caught by `execute_pipeline`'s existing handler, which records the failure and classifies it) if a production Ray run would write to a non-durable location.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_shared_inputs.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Static checks**

Run: `ruff check backend/ml_pipeline/_execution/shared_inputs.py backend/ml_pipeline/_services/pipeline_execution_service.py tests/test_shared_inputs.py`
Then: `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
Then: `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add backend/ml_pipeline/_execution/shared_inputs.py backend/ml_pipeline/_services/pipeline_execution_service.py tests/test_shared_inputs.py
git commit -m "feat(ray): enforce durable artifact URIs and add shared-input helper

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Distributed compute gate — regression + docs

**Files:**
- Modify: `.env.example` (document the Ray resource keys)

- [ ] **Step 1: Document the resource keys**

Append to `.env.example` under the Ray runtime section:

```bash
# --- Ray resources & tuning (plan 04) ---
# CPUs/GPUs reserved per training Ray Job; CPUs per tuning Ray Job (parallel trials).
# RAY_ENTRYPOINT_NUM_CPUS=1
# RAY_ENTRYPOINT_NUM_GPUS=0
# RAY_TUNING_NUM_CPUS=4
```

- [ ] **Step 2: Run the distributed-compute regression subset**

Run:
```bash
pytest tests/test_resource_alignment.py tests/test_ray_joblib_backend.py \
  tests/test_shared_inputs.py tests/test_ray_resource_submission.py \
  tests/test_ray_execution_backend.py tests/test_ray_client_adapter.py \
  tests/test_ray_driver_run_pipeline.py tests/test_node_runners_extra.py -q
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
git commit -m "docs(ray): document Ray resource reservation env keys

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Definition of Done (Compute Gate)

- Each Ray Job declares `entrypoint_num_cpus`/`num_gpus`/`memory` sized by job type; the Ray scheduler distributes branch attempts across workers.
- The driver aligns `OMP/MKL/OpenBLAS/NUMEXPR` threads to the reserved CPUs (via `SKYULF_NUM_CPUS`); nested estimator parallelism cannot oversubscribe a host.
- Hyperparameter tuning runs through Ray's joblib backend under `parallel_backend("ray")` with unchanged search logic; local/celery tuning behavior is byte-identical to before.
- Ray workers exchange URIs/ids, not dataframes; production Ray rejects non-durable (local) artifact URIs; a `ray_put_shared` helper exists for intra-node immutable inputs.
- Ray Tune / Ray Data / Ray Train / Ray Serve are **not** used.
- Full backend `ruff` / `ruff format --check` / `ty` gate is green.
