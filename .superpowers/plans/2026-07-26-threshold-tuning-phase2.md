# Threshold Tuning Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Phase 1 library-only `optimize_thresholds()`/`apply_thresholds()` functions into a full product feature: persisted per-job tuned thresholds, backend API endpoints to preview/save/toggle/clear them, live `/predict` integration (saved + ad-hoc override), and frontend UI in the Evaluation view (tuning panel + multiclass confusion-matrix redraw) and Inference page (override controls).

**Architecture:** Two new DB columns on `TrainingJob` (`tuned_thresholds` JSON, `tuned_thresholds_enabled` bool) store one saved threshold set per job. A new `ThresholdTuningService` reuses a refactored raw (undecoded) evaluation-data loader to preview/save/toggle/clear thresholds via 4 new REST endpoints. `DeploymentService.predict()` resolves thresholds with priority override > saved-enabled > none, applying them via the existing `skyulf.modeling.apply_thresholds()` against `predict_proba()` output (bundled-artifact path only). Frontend adds an API client, a pure TS scaled-argmax function (parity-tested against the real Python function) for multiclass confusion-matrix redraw, a Threshold Tuning panel in the Evaluation view, an Inference-page override control, and a hint in the Train/Test Splitter node.

**Tech Stack:** FastAPI, SQLAlchemy (async, SQLite/Postgres via `training_jobs` table), Pydantic, scikit-learn scorers, `skyulf-core`'s `skyulf.modeling.apply_thresholds`/`optimize_thresholds`, React + TypeScript frontend, Vitest.

## Global Constraints

- Threshold dict keys must match `classes` exactly on both compute and apply (per `skyulf-core`'s `apply_thresholds` contract) — reconcile stringified `predict_proba` classes vs. raw `y_true`/`estimator.classes_` dtypes at every boundary.
- Tuning always operates on **raw/undecoded** labels — never the human-readable decoded labels `EvaluationService.get_job_evaluation()` returns.
- `require_new_holdout`: if the job's `validation_size` produced no validation split, silently fall back to the `test` split for tuning and surface a UI hint (no backend error).
- All 6 metrics must be supported: `accuracy`, `f1`, `precision`, `recall`, `balanced_accuracy`, `roc_auc` — precision/recall/f1 use `average="weighted"`, `zero_division=0`.
- `DeploymentService.predict()` threshold integration is scoped to the **bundled-artifact path only** (not legacy direct-estimator artifacts) — deliberate YAGNI.
- Override thresholds in `/predict` must return HTTP 422 (not 400) on a `classes` key mismatch, via a dedicated `OverrideThresholdMismatch` exception caught before the generic `ValueError` handler.
- New TS multiclass redraw function must be a separate function from the existing `applyThreshold` — never modify existing binary/OvR slider behavior.
- Every backend task needs `ruff check`, `ruff format --check`, `ty check` and relevant `pytest` clean; every frontend task needs `eslint`, `tsc --noEmit`, relevant `vitest`, clean.
- Add a changelog entry under the real current heading in `changelog/0.7.x.md` (verify the heading text before editing — do not assume).

---

## Task 1: Backend data model — DB columns, migration, `JobInfo` field

**Files:**
- Modify: `backend/database/models.py` (`TrainingJob` class, ~line 296-330)
- Modify: `backend/database/engine.py` (`_MIGRATIONS` list, ~line 195-200)
- Modify: `backend/ml_pipeline/_execution/schemas.py` (`JobInfo`)
- Modify: `backend/ml_pipeline/_execution/basic_training_manager.py` (`map_training_job_to_info`, ~line 70-115)
- Test: `tests/test_database_migrations.py` (create if it doesn't already cover `_MIGRATIONS`; otherwise add to the existing migration test file — check `tests/` for an existing one first with `grep -rl "_MIGRATIONS" tests/`)

**Interfaces:**
- Produces: `TrainingJob.tuned_thresholds: dict | None` (JSON column), `TrainingJob.tuned_thresholds_enabled: bool` (default `False`), `JobInfo.tuned_thresholds_enabled: bool | None = None`.

- [ ] **Step 1: Check for an existing migration test file**

Run: `grep -rl "_MIGRATIONS\|_run_migrations" /Users/BH7043/Skyulf/tests/`

If a file is found, add the new test there. If not, create `tests/test_database_migrations.py`.

- [ ] **Step 2: Write the failing test**

```python
import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

from backend.database.engine import _run_migrations
from backend.database.models import Base


@pytest.mark.asyncio
async def test_migrations_add_tuned_thresholds_columns():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _run_migrations(conn)

        def _get_columns(sync_conn):
            return {col["name"] for col in inspect(sync_conn).get_columns("training_jobs")}

        columns = await conn.run_sync(_get_columns)

    assert "tuned_thresholds" in columns
    assert "tuned_thresholds_enabled" in columns
    await engine.dispose()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_database_migrations.py::test_migrations_add_tuned_thresholds_columns -v`
Expected: FAIL — `AssertionError` because the columns don't exist yet (columns not in migration list, though `Base.metadata.create_all` may already create them once Task 1 Step 4 adds them to the model — if so this test instead verifies the migration path works on a pre-existing DB missing the columns; adjust by first creating tables WITHOUT the new model fields via a snapshot, or simply trust `create_all` + confirm `_run_migrations` is a no-op/idempotent. Since `create_all` will already include new columns once added to the model in Step 4, this test's real purpose is regression-proofing the migration list for existing production databases — proceed to Step 4 first, then Step 3 becomes a pass-through sanity check that both columns exist after `create_all` + migrations combined.)

- [ ] **Step 4: Add the columns to `TrainingJob`**

In `backend/database/models.py`, inside the `TrainingJob` class (near the existing `promoted_at` column around line 296-330):

```python
    tuned_thresholds: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    tuned_thresholds_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="0")
```

- [ ] **Step 5: Add the migration entries**

In `backend/database/engine.py`'s `_MIGRATIONS` list (~line 195-200), following the exact `promoted_at` precedent:

```python
    "ALTER TABLE training_jobs ADD COLUMN tuned_thresholds JSON",
    "ALTER TABLE training_jobs ADD COLUMN tuned_thresholds_enabled BOOLEAN NOT NULL DEFAULT 0",
```

- [ ] **Step 6: Run the migration test**

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_database_migrations.py::test_migrations_add_tuned_thresholds_columns -v`
Expected: PASS

- [ ] **Step 7: Add `tuned_thresholds_enabled` to `JobInfo`**

In `backend/ml_pipeline/_execution/schemas.py`, add to the `JobInfo` Pydantic model, mirroring the existing `promoted_at` field:

```python
    tuned_thresholds_enabled: bool | None = None
```

- [ ] **Step 8: Populate it in `map_training_job_to_info`**

In `backend/ml_pipeline/_execution/basic_training_manager.py` (~line 70-115), in the `map_training_job_to_info` function, add alongside the existing `promoted_at=job.promoted_at` line:

```python
        tuned_thresholds_enabled=job.tuned_thresholds_enabled,
```

- [ ] **Step 9: Run backend checks**

Run: `cd /Users/BH7043/Skyulf && ruff check backend/database/models.py backend/database/engine.py backend/ml_pipeline/_execution/schemas.py backend/ml_pipeline/_execution/basic_training_manager.py && ruff format --check backend/database/models.py backend/database/engine.py backend/ml_pipeline/_execution/schemas.py backend/ml_pipeline/_execution/basic_training_manager.py && ty check backend/database/models.py backend/database/engine.py backend/ml_pipeline/_execution/schemas.py backend/ml_pipeline/_execution/basic_training_manager.py`
Expected: all clean

- [ ] **Step 10: Commit**

```bash
git add backend/database/models.py backend/database/engine.py backend/ml_pipeline/_execution/schemas.py backend/ml_pipeline/_execution/basic_training_manager.py tests/test_database_migrations.py
git commit -m "feat(threshold-tuning): add tuned_thresholds columns and migration"
```

---

## Task 2: `EvaluationService` raw-loader refactor + `ThresholdTuningService`

**Files:**
- Modify: `backend/ml_pipeline/_services/evaluation_service.py`
- Create: `backend/ml_pipeline/_services/threshold_tuning_service.py`
- Test: `tests/test_threshold_tuning_service.py`

**Interfaces:**
- Consumes: `EvaluationService._load_raw_evaluation_data(session, job_id) -> tuple[dict, ArtifactStore]` (new), `_to_int_like_array` from `backend/ml_pipeline/_services/prediction_utils.py`, `optimize_thresholds`/`apply_thresholds` from `skyulf.modeling`.
- Produces: `ThresholdTuningService.preview(session, job_id, metric) -> dict` (returns `{"thresholds": {...}, "classes": [...], "metric": str, "split_used": str}`), `ThresholdTuningService.save(session, job_id, thresholds, classes, metric, split_used) -> bool`, `ThresholdTuningService.toggle(session, job_id, enabled) -> bool`, `ThresholdTuningService.clear(session, job_id) -> bool`, `ThresholdTuningError(ValueError)`.

- [ ] **Step 1: Extract the raw loader in `EvaluationService`**

In `backend/ml_pipeline/_services/evaluation_service.py`, find `get_job_evaluation()`'s body before it calls `_decode_target_labels`. Extract everything up to (not including) the decode step into a new method:

```python
    @staticmethod
    async def _load_raw_evaluation_data(
        session: AsyncSession, job_id: str
    ) -> tuple[dict, ArtifactStore]:
        """Load evaluation artifacts for a job without decoding target labels."""
        # <-- move the existing pre-decode body of get_job_evaluation() here verbatim -->
        return evaluation_data, artifact_store
```

Then rewrite `get_job_evaluation()` to call it:

```python
    @staticmethod
    async def get_job_evaluation(session: AsyncSession, job_id: str) -> dict:
        evaluation_data, artifact_store = await EvaluationService._load_raw_evaluation_data(
            session, job_id
        )
        # <-- existing decode-and-return logic continues unchanged -->
```

- [ ] **Step 2: Write the failing test for `ThresholdTuningService.preview`**

```python
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from backend.database.models import Base
from backend.ml_pipeline._services.threshold_tuning_service import (
    ThresholdTuningService,
    ThresholdTuningError,
)


@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSession(engine) as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_preview_returns_thresholds_for_valid_job(db_session, monkeypatch):
    await db_session.execute(
        text(
            "INSERT INTO training_jobs (id, status, validation_size) "
            "VALUES ('job-1', 'completed', 0.2)"
        )
    )
    await db_session.commit()

    async def fake_load_raw(session, job_id):
        import numpy as np

        return (
            {
                "validation": {
                    "y_true": np.array([0, 1, 2, 2, 1]),
                    "y_proba": {
                        "values": np.array(
                            [
                                [0.5, 0.3, 0.2],
                                [0.2, 0.6, 0.2],
                                [0.34, 0.33, 0.33],
                                [0.1, 0.1, 0.8],
                                [0.4, 0.4, 0.2],
                            ]
                        ),
                        "classes": ["0", "1", "2"],
                    },
                },
                "test": None,
            },
            None,
        )

    monkeypatch.setattr(
        "backend.ml_pipeline._services.evaluation_service.EvaluationService._load_raw_evaluation_data",
        fake_load_raw,
    )

    result = await ThresholdTuningService.preview(db_session, "job-1", metric="f1")

    assert result["metric"] == "f1"
    assert result["split_used"] == "validation"
    assert set(result["classes"]) == {0, 1, 2}
    assert set(result["thresholds"].keys()) == {"0", "1", "2"}


@pytest.mark.asyncio
async def test_preview_raises_for_missing_job(db_session):
    with pytest.raises(ThresholdTuningError):
        await ThresholdTuningService.preview(db_session, "nonexistent", metric="f1")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_threshold_tuning_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.ml_pipeline._services.threshold_tuning_service'`

- [ ] **Step 4: Implement `ThresholdTuningService`**

Create `backend/ml_pipeline/_services/threshold_tuning_service.py`:

```python
"""Service for previewing, saving, toggling, and clearing per-job tuned thresholds."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import TrainingJob
from backend.ml_pipeline._services.evaluation_service import EvaluationService
from backend.ml_pipeline._services.prediction_utils import _to_int_like_array
from skyulf.modeling import optimize_thresholds


class ThresholdTuningError(ValueError):
    """Raised for invalid threshold-tuning requests (maps to HTTP 400)."""


_METRIC_SCORERS = {
    "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted", zero_division=0),
    "precision": lambda y_true, y_pred: precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "recall": lambda y_true, y_pred: recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "balanced_accuracy": lambda y_true, y_pred: balanced_accuracy_score(y_true, y_pred),
    "roc_auc": lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
}


class ThresholdTuningService:
    """Preview, save, toggle, and clear tuned classification thresholds for a job."""

    @staticmethod
    def _select_split(evaluation_data: dict) -> tuple[str, dict]:
        """Prefer the validation split, falling back to test if validation is absent."""
        validation = evaluation_data.get("validation")
        if validation is not None:
            return "validation", validation
        test = evaluation_data.get("test")
        if test is not None:
            return "test", test
        raise ThresholdTuningError("Job has no validation or test split to tune against.")

    @staticmethod
    def _coerce_classes_and_labels(y_true: np.ndarray, y_proba: dict) -> tuple[np.ndarray, np.ndarray, list]:
        """Reconcile stringified predict_proba classes against y_true's raw dtype."""
        raw_classes = y_proba["classes"]
        coerced = _to_int_like_array(np.array(raw_classes))
        classes = coerced.tolist() if coerced is not None else list(raw_classes)
        return y_true, np.asarray(y_proba["values"]), classes

    @staticmethod
    async def preview(session: AsyncSession, job_id: str, metric: str) -> dict:
        """Compute (without saving) tuned per-class thresholds for a job's evaluation data."""
        if metric not in _METRIC_SCORERS:
            raise ThresholdTuningError(f"Unsupported metric: {metric}")

        job = (
            await session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        ).scalar_one_or_none()
        if job is None:
            raise ThresholdTuningError(f"Job not found: {job_id}")

        evaluation_data, _ = await EvaluationService._load_raw_evaluation_data(session, job_id)
        split_used, split_data = ThresholdTuningService._select_split(evaluation_data)

        y_true, y_proba_values, classes = ThresholdTuningService._coerce_classes_and_labels(
            split_data["y_true"], split_data["y_proba"]
        )

        thresholds = optimize_thresholds(
            y_true, y_proba_values, metric=metric, classes=classes
        )

        return {
            "thresholds": {str(k): v for k, v in thresholds.items()},
            "classes": classes,
            "metric": metric,
            "split_used": split_used,
        }

    @staticmethod
    async def save(
        session: AsyncSession,
        job_id: str,
        thresholds: dict[str, float],
        classes: list,
        metric: str,
        split_used: str,
    ) -> bool:
        """Persist a tuned threshold set on the job, enabling it by default."""
        job = (
            await session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        ).scalar_one_or_none()
        if job is None:
            raise ThresholdTuningError(f"Job not found: {job_id}")

        job.tuned_thresholds = {
            "thresholds": thresholds,
            "classes": classes,
            "metric": metric,
            "split_used": split_used,
            "computed_at": __import__("datetime").datetime.utcnow().isoformat(),
        }
        job.tuned_thresholds_enabled = True
        await session.commit()
        return True

    @staticmethod
    async def toggle(session: AsyncSession, job_id: str, enabled: bool) -> bool:
        """Enable or disable use of the job's saved tuned thresholds at predict time."""
        job = (
            await session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        ).scalar_one_or_none()
        if job is None:
            raise ThresholdTuningError(f"Job not found: {job_id}")
        if job.tuned_thresholds is None:
            raise ThresholdTuningError("Job has no saved tuned thresholds to toggle.")

        job.tuned_thresholds_enabled = enabled
        await session.commit()
        return True

    @staticmethod
    async def clear(session: AsyncSession, job_id: str) -> bool:
        """Remove any saved tuned thresholds from the job."""
        job = (
            await session.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        ).scalar_one_or_none()
        if job is None:
            raise ThresholdTuningError(f"Job not found: {job_id}")

        job.tuned_thresholds = None
        job.tuned_thresholds_enabled = False
        await session.commit()
        return True
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_threshold_tuning_service.py -v`
Expected: PASS

- [ ] **Step 6: Write tests for `save`/`toggle`/`clear`**

Append to `tests/test_threshold_tuning_service.py`:

```python
@pytest.mark.asyncio
async def test_save_toggle_clear_round_trip(db_session):
    await db_session.execute(
        text(
            "INSERT INTO training_jobs (id, status, validation_size) "
            "VALUES ('job-2', 'completed', 0.2)"
        )
    )
    await db_session.commit()

    saved = await ThresholdTuningService.save(
        db_session,
        "job-2",
        thresholds={"0": 0.6, "1": 0.5, "2": 0.3},
        classes=[0, 1, 2],
        metric="f1",
        split_used="validation",
    )
    assert saved is True

    from sqlalchemy import select
    from backend.database.models import TrainingJob

    job = (
        await db_session.execute(select(TrainingJob).where(TrainingJob.id == "job-2"))
    ).scalar_one()
    assert job.tuned_thresholds_enabled is True
    assert job.tuned_thresholds["thresholds"] == {"0": 0.6, "1": 0.5, "2": 0.3}

    await ThresholdTuningService.toggle(db_session, "job-2", enabled=False)
    await db_session.refresh(job)
    assert job.tuned_thresholds_enabled is False

    await ThresholdTuningService.clear(db_session, "job-2")
    await db_session.refresh(job)
    assert job.tuned_thresholds is None
    assert job.tuned_thresholds_enabled is False
```

- [ ] **Step 7: Run all threshold tuning service tests**

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_threshold_tuning_service.py -v`
Expected: PASS (all tests)

- [ ] **Step 8: Run backend checks**

Run: `cd /Users/BH7043/Skyulf && ruff check backend/ml_pipeline/_services/threshold_tuning_service.py backend/ml_pipeline/_services/evaluation_service.py && ruff format --check backend/ml_pipeline/_services/threshold_tuning_service.py backend/ml_pipeline/_services/evaluation_service.py && ty check backend/ml_pipeline/_services/threshold_tuning_service.py backend/ml_pipeline/_services/evaluation_service.py`
Expected: clean

- [ ] **Step 9: Commit**

```bash
git add backend/ml_pipeline/_services/evaluation_service.py backend/ml_pipeline/_services/threshold_tuning_service.py tests/test_threshold_tuning_service.py
git commit -m "feat(threshold-tuning): add ThresholdTuningService with preview/save/toggle/clear"
```

---

## Task 3: Router endpoints in `jobs.py`

**Files:**
- Modify: `backend/ml_pipeline/_internal/_routers/jobs.py`
- Test: `tests/test_jobs_router_threshold_tuning.py`

**Interfaces:**
- Consumes: `ThresholdTuningService.preview/save/toggle/clear`, `ThresholdTuningError` (Task 2).
- Produces: `POST /jobs/{job_id}/thresholds/preview`, `POST /jobs/{job_id}/thresholds/save`, `POST /jobs/{job_id}/thresholds/toggle`, `DELETE /jobs/{job_id}/thresholds`.

- [ ] **Step 1: Write the failing router test**

```python
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text

from backend.main import app  # adjust import if the FastAPI app lives elsewhere; confirm via `grep -rl "FastAPI()" backend/`


@pytest.mark.asyncio
async def test_preview_endpoint_returns_400_for_missing_job(async_db_session_override):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/jobs/nonexistent/thresholds/preview", json={"metric": "f1"}
        )
    assert response.status_code == 400
```

Note: confirm the exact app import path and any DB-session-override test fixture pattern already used in `tests/test_deployment.py` before finalizing this test — mirror that file's client-setup fixture exactly instead of inventing a new one.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_jobs_router_threshold_tuning.py -v`
Expected: FAIL — 404 (route doesn't exist yet)

- [ ] **Step 3: Add Pydantic schemas and endpoints to `jobs.py`**

In `backend/ml_pipeline/_internal/_routers/jobs.py`, add near the top (after existing imports):

```python
from pydantic import BaseModel

from backend.ml_pipeline._services.threshold_tuning_service import (
    ThresholdTuningError,
    ThresholdTuningService,
)


class ThresholdTuningPreviewRequest(BaseModel):
    metric: str


class ThresholdTuningPreviewResponse(BaseModel):
    thresholds: dict[str, float]
    classes: list
    metric: str
    split_used: str


class ThresholdTuningSaveRequest(BaseModel):
    thresholds: dict[str, float]
    classes: list
    metric: str
    split_used: str


class ThresholdTuningToggleRequest(BaseModel):
    enabled: bool
```

Then add the 4 endpoints, mirroring `promote_job`/`unpromote_job`'s try/except → HTTPException pattern exactly:

```python
@router.post("/{job_id}/thresholds/preview", response_model=ThresholdTuningPreviewResponse)
async def preview_thresholds(
    job_id: str,
    request: ThresholdTuningPreviewRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await ThresholdTuningService.preview(session, job_id, request.metric)
    except ThresholdTuningError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ThresholdTuningPreviewResponse(**result)


@router.post("/{job_id}/thresholds/save")
async def save_thresholds(
    job_id: str,
    request: ThresholdTuningSaveRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        await ThresholdTuningService.save(
            session,
            job_id,
            thresholds=request.thresholds,
            classes=request.classes,
            metric=request.metric,
            split_used=request.split_used,
        )
    except ThresholdTuningError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "saved"}


@router.post("/{job_id}/thresholds/toggle")
async def toggle_thresholds(
    job_id: str,
    request: ThresholdTuningToggleRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        await ThresholdTuningService.toggle(session, job_id, request.enabled)
    except ThresholdTuningError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "toggled", "enabled": request.enabled}


@router.delete("/{job_id}/thresholds")
async def clear_thresholds(job_id: str, session: AsyncSession = Depends(get_session)):
    try:
        await ThresholdTuningService.clear(session, job_id)
    except ThresholdTuningError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "cleared"}
```

Before finalizing, re-read `backend/ml_pipeline/_internal/_routers/jobs.py` lines 1-111 to confirm the exact `router` variable name, the `get_session` dependency import path, and whether `promote_job`/`unpromote_job` use `HTTPException` directly or a wrapping helper — match that convention exactly rather than the illustrative code above.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_jobs_router_threshold_tuning.py -v`
Expected: PASS

- [ ] **Step 5: Add positive-path tests (preview success, save+toggle+clear via HTTP)**

```python
@pytest.mark.asyncio
async def test_full_threshold_tuning_http_flow(async_db_session_override, seeded_job_with_evaluation):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        preview = await client.post(
            f"/jobs/{seeded_job_with_evaluation}/thresholds/preview", json={"metric": "f1"}
        )
        assert preview.status_code == 200
        body = preview.json()

        save = await client.post(
            f"/jobs/{seeded_job_with_evaluation}/thresholds/save",
            json={
                "thresholds": body["thresholds"],
                "classes": body["classes"],
                "metric": body["metric"],
                "split_used": body["split_used"],
            },
        )
        assert save.status_code == 200

        toggle = await client.post(
            f"/jobs/{seeded_job_with_evaluation}/thresholds/toggle", json={"enabled": False}
        )
        assert toggle.status_code == 200

        clear = await client.delete(f"/jobs/{seeded_job_with_evaluation}/thresholds")
        assert clear.status_code == 200
```

Note: `seeded_job_with_evaluation` fixture must be defined mirroring the exact seeding pattern used in `tests/test_deployment.py` (raw `INSERT INTO training_jobs` + a mocked/monkeypatched `_load_raw_evaluation_data`), producing a job id string.

- [ ] **Step 6: Run all router tests**

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_jobs_router_threshold_tuning.py -v`
Expected: PASS

- [ ] **Step 7: Run backend checks**

Run: `cd /Users/BH7043/Skyulf && ruff check backend/ml_pipeline/_internal/_routers/jobs.py && ruff format --check backend/ml_pipeline/_internal/_routers/jobs.py && ty check backend/ml_pipeline/_internal/_routers/jobs.py`
Expected: clean

- [ ] **Step 8: Commit**

```bash
git add backend/ml_pipeline/_internal/_routers/jobs.py tests/test_jobs_router_threshold_tuning.py
git commit -m "feat(threshold-tuning): add preview/save/toggle/clear REST endpoints"
```

---

## Task 4: `DeploymentService`/`schemas.py`/`api.py` predict-time integration

**Files:**
- Modify: `backend/ml_pipeline/deployment/service.py`
- Modify: `backend/ml_pipeline/deployment/schemas.py`
- Modify: `backend/ml_pipeline/deployment/api.py`
- Test: `tests/test_deployment.py` (append)

**Interfaces:**
- Consumes: `TrainingJob.tuned_thresholds`/`tuned_thresholds_enabled` (Task 1), `apply_thresholds` from `skyulf.modeling`.
- Produces: `PredictionRequest.override_thresholds: dict[str, float] | None`, `PredictionResponse.thresholds_applied: dict[str, float] | None`, `OverrideThresholdMismatch(ValueError)`, `DeploymentService._predict_and_decode(..., thresholds: dict | None) -> tuple[predictions, thresholds_applied]`.

- [ ] **Step 1: Write the failing test for override mismatch → 422**

```python
@pytest.mark.asyncio
async def test_predict_with_mismatched_override_thresholds_returns_422(deployment_test_client, seeded_deployment):
    response = await deployment_test_client.post(
        f"/deployments/{seeded_deployment}/predict",
        json={
            "data": [{"feature_1": 1.0}],
            "override_thresholds": {"nonexistent_class": 0.5},
        },
    )
    assert response.status_code == 422
```

Note: mirror `tests/test_deployment.py`'s existing `deployment_test_client`/`seeded_deployment` fixture names exactly — read the file's fixtures (lines 1-90 already reviewed) before writing this so names match.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_deployment.py::test_predict_with_mismatched_override_thresholds_returns_422 -v`
Expected: FAIL — `override_thresholds` field doesn't exist / 422 from Pydantic schema validation error instead of the intended business-logic 422, or a 500/400.

- [ ] **Step 3: Add schema fields**

In `backend/ml_pipeline/deployment/schemas.py`:

```python
class PredictionRequest(BaseModel):
    # ... existing fields ...
    override_thresholds: dict[str, float] | None = None


class PredictionResponse(BaseModel):
    # ... existing fields ...
    thresholds_applied: dict[str, float] | None = None
```

- [ ] **Step 4: Add `OverrideThresholdMismatch` and threshold-resolution helpers to `service.py`**

In `backend/ml_pipeline/deployment/service.py`, add near the top-level exceptions:

```python
class OverrideThresholdMismatch(ValueError):
    """Raised when override_thresholds keys don't match the model's classes."""
```

Add helper methods on `DeploymentService` (or module-level functions, matching the file's existing style — check whether `_load_predict_artifact` etc. are static methods or module functions before finalizing):

```python
    @staticmethod
    def _validate_override_thresholds(
        override_thresholds: dict[str, float], estimator_classes: np.ndarray
    ) -> None:
        expected = {str(c) for c in estimator_classes}
        provided = set(override_thresholds.keys())
        if provided != expected:
            raise OverrideThresholdMismatch(
                f"override_thresholds keys {sorted(provided)} do not match model classes {sorted(expected)}"
            )

    @staticmethod
    def _resolve_thresholds_for_predict(
        override_thresholds: dict[str, float] | None,
        job: "TrainingJob | None",
        estimator_classes: np.ndarray,
    ) -> dict[str, float] | None:
        if override_thresholds is not None:
            DeploymentService._validate_override_thresholds(override_thresholds, estimator_classes)
            return override_thresholds
        if job is not None and job.tuned_thresholds_enabled and job.tuned_thresholds:
            saved = job.tuned_thresholds["thresholds"]
            return {str(c): saved[str(c)] for c in estimator_classes if str(c) in saved}
        return None
```

- [ ] **Step 5: Update `_predict_and_decode` and `_predict_with_bundled_artifact`**

In `_predict_and_decode`, add a `thresholds: dict[str, float] | None = None` parameter; when `thresholds` is not `None`, call `estimator.predict_proba(X)` and `skyulf.modeling.apply_thresholds(y_proba, classes=estimator.classes_, thresholds={int-or-raw-key: v for k, v in thresholds.items()})` instead of `estimator.predict(X)`; return `(predictions, thresholds if thresholds is not None else None)` instead of bare `predictions`.

In `_predict_with_bundled_artifact`, thread the new `thresholds` parameter through to `_predict_and_decode`, and change its return type to the same `(predictions, thresholds_applied)` tuple.

- [ ] **Step 6: Update `predict()`**

In `DeploymentService.predict()`, add an `override_thresholds: dict[str, float] | None = None` parameter. Look up the `TrainingJob` via `deployment.job_id` (select + `scalar_one_or_none`, mirroring `promote_job`'s pattern). Call `_resolve_thresholds_for_predict(override_thresholds, job, estimator.classes_)` to get `thresholds`. Pass `thresholds` into the bundled-artifact call. For the legacy-artifact branch, keep calling it unchanged but wrap its return as `(predictions, None)` to match the new unified tuple contract. Unpack `predictions, thresholds_applied = ...` from whichever branch ran, and include `thresholds_applied` in the constructed `PredictionResponse`.

- [ ] **Step 7: Update `api.py`'s predict route**

In `backend/ml_pipeline/deployment/api.py`, in the `predict` handler, pass `prediction_request.override_thresholds` into `DeploymentService.predict(...)`. Add exception handling with `OverrideThresholdMismatch` caught **before** the generic `except ValueError`:

```python
    try:
        predictions, thresholds_applied = await DeploymentService.predict(
            session,
            deployment_id,
            prediction_request.data,
            override_thresholds=prediction_request.override_thresholds,
        )
    except OverrideThresholdMismatch as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
```

Include `thresholds_applied` in the constructed `PredictionResponse` returned by the route.

- [ ] **Step 8: Run test to verify it passes**

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_deployment.py::test_predict_with_mismatched_override_thresholds_returns_422 -v`
Expected: PASS

- [ ] **Step 9: Write a positive-path test for saved+enabled thresholds affecting predictions**

```python
@pytest.mark.asyncio
async def test_predict_uses_saved_enabled_thresholds(deployment_test_client, seeded_deployment_with_tuned_thresholds):
    response = await deployment_test_client.post(
        f"/deployments/{seeded_deployment_with_tuned_thresholds}/predict",
        json={"data": [{"feature_1": 1.0}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["thresholds_applied"] is not None
```

Note: add a `seeded_deployment_with_tuned_thresholds` fixture that inserts a `training_jobs` row with `tuned_thresholds_enabled=1` and a valid `tuned_thresholds` JSON blob whose keys match the mock estimator's `classes_`, matching `_FixedPredictor`'s existing mock conventions in `tests/test_deployment.py`.

- [ ] **Step 10: Run all deployment tests**

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_deployment.py -v`
Expected: PASS (all tests, including pre-existing ones — confirms the tuple-return refactor didn't break existing callers)

- [ ] **Step 11: Run backend checks**

Run: `cd /Users/BH7043/Skyulf && ruff check backend/ml_pipeline/deployment/ && ruff format --check backend/ml_pipeline/deployment/ && ty check backend/ml_pipeline/deployment/`
Expected: clean

- [ ] **Step 12: Commit**

```bash
git add backend/ml_pipeline/deployment/service.py backend/ml_pipeline/deployment/schemas.py backend/ml_pipeline/deployment/api.py tests/test_deployment.py
git commit -m "feat(threshold-tuning): apply saved/override thresholds at predict time"
```

---

## Task 5: Frontend API clients

**Files:**
- Modify: `frontend/ml-canvas/src/core/api/deployment.ts`
- Create: `frontend/ml-canvas/src/core/api/thresholdTuning.ts`
- Test: `frontend/ml-canvas/src/core/api/thresholdTuning.test.ts`

**Interfaces:**
- Produces: `thresholdTuningApi.preview(jobId, metric) -> Promise<{thresholds, classes, metric, split_used}>`, `thresholdTuningApi.save(jobId, payload) -> Promise<void>`, `thresholdTuningApi.toggle(jobId, enabled) -> Promise<void>`, `thresholdTuningApi.clear(jobId) -> Promise<void>`; updated `PredictionRequest`/`PredictionResponse` types and `deploymentApi.predict()` in `deployment.ts` gaining `override_thresholds`/`thresholds_applied`.

- [ ] **Step 1: Add fields to `deployment.ts` types**

In `frontend/ml-canvas/src/core/api/deployment.ts`, extend the existing `PredictionRequest`/`PredictionResponse` interfaces:

```typescript
export interface PredictionRequest {
  // ...existing fields...
  override_thresholds?: Record<string, number> | null;
}

export interface PredictionResponse {
  // ...existing fields...
  thresholds_applied?: Record<string, number> | null;
}
```

Confirm `deploymentApi.predict()`'s body already forwards the full request object (check the existing implementation) — if it destructures fields explicitly rather than spreading, add `override_thresholds` to that explicit list.

- [ ] **Step 2: Write the failing test for the new API client**

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { thresholdTuningApi } from "./thresholdTuning";

describe("thresholdTuningApi", () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });

  it("preview posts metric and returns parsed thresholds", async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        thresholds: { "0": 0.6, "1": 0.5, "2": 0.3 },
        classes: [0, 1, 2],
        metric: "f1",
        split_used: "validation",
      }),
    });

    const result = await thresholdTuningApi.preview("job-1", "f1");

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/jobs/job-1/thresholds/preview"),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ metric: "f1" }),
      })
    );
    expect(result.thresholds).toEqual({ "0": 0.6, "1": 0.5, "2": 0.3 });
    expect(result.split_used).toBe("validation");
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npx vitest run src/core/api/thresholdTuning.test.ts`
Expected: FAIL — cannot find module `./thresholdTuning`

- [ ] **Step 4: Implement `thresholdTuning.ts`**

Create `frontend/ml-canvas/src/core/api/thresholdTuning.ts`. First check `deployment.ts`'s existing fetch/base-URL helper (e.g. `apiFetch` or a shared `API_BASE_URL` constant) and reuse the exact same helper rather than reimplementing fetch logic:

```typescript
import { apiFetch } from "./client"; // adjust import to match the actual shared fetch helper used by deployment.ts

export interface ThresholdPreviewResult {
  thresholds: Record<string, number>;
  classes: number[];
  metric: string;
  split_used: string;
}

export const thresholdTuningApi = {
  async preview(jobId: string, metric: string): Promise<ThresholdPreviewResult> {
    return apiFetch(`/jobs/${jobId}/thresholds/preview`, {
      method: "POST",
      body: JSON.stringify({ metric }),
    });
  },

  async save(
    jobId: string,
    payload: {
      thresholds: Record<string, number>;
      classes: number[];
      metric: string;
      split_used: string;
    }
  ): Promise<void> {
    await apiFetch(`/jobs/${jobId}/thresholds/save`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async toggle(jobId: string, enabled: boolean): Promise<void> {
    await apiFetch(`/jobs/${jobId}/thresholds/toggle`, {
      method: "POST",
      body: JSON.stringify({ enabled }),
    });
  },

  async clear(jobId: string): Promise<void> {
    await apiFetch(`/jobs/${jobId}/thresholds`, { method: "DELETE" });
  },
};
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npx vitest run src/core/api/thresholdTuning.test.ts`
Expected: PASS (adjust the mock's `expect.objectContaining` assertions if `apiFetch`'s actual signature differs from raw `fetch` — match whatever the real helper does)

- [ ] **Step 6: Run frontend checks**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npx eslint src/core/api/thresholdTuning.ts src/core/api/deployment.ts && npx tsc --project tsconfig.json --noEmit`
Expected: clean

- [ ] **Step 7: Commit**

```bash
git add frontend/ml-canvas/src/core/api/thresholdTuning.ts frontend/ml-canvas/src/core/api/thresholdTuning.test.ts frontend/ml-canvas/src/core/api/deployment.ts
git commit -m "feat(threshold-tuning): add frontend API clients"
```

---

## Task 6: TS scaled-argmax multiclass function + parity tests

**Files:**
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts.ts`
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts.test.ts`

**Interfaces:**
- Consumes: `EvaluationSplit` type and `calculateConfusionMatrix` (both already defined in `classificationCharts.ts`).
- Produces: `applyMulticlassThresholds(splitData: EvaluationSplit, thresholds: Record<string, number>): ConfusionMatrixResult` (return type = whatever `calculateConfusionMatrix` already returns — match its exact return type name from the file).

- [ ] **Step 1: Write the failing parity test using the real Python-computed fixture**

Append to `classificationCharts.test.ts`:

```typescript
describe("applyMulticlassThresholds", () => {
  it("matches skyulf-core's apply_thresholds output for a known fixture", () => {
    // Fixture computed via the real Python skyulf.modeling._evaluation.thresholds.apply_thresholds()
    const yProbaValues = [
      [0.5, 0.3, 0.2],
      [0.2, 0.6, 0.2],
      [0.34, 0.33, 0.33],
      [0.1, 0.1, 0.8],
      [0.4, 0.4, 0.2],
    ];
    const classes = [0, 1, 2];
    const yTrue = [0, 1, 2, 2, 1];
    const thresholds = { "0": 0.6, "1": 0.5, "2": 0.3 };
    const expectedPredictions = [0, 1, 2, 2, 1];

    const splitData = {
      yTrue,
      yProba: { values: yProbaValues, classes: classes.map(String) },
    } as unknown as EvaluationSplit;

    const result = applyMulticlassThresholds(splitData, thresholds);

    // Recompute the raw predictions the same way the function does internally,
    // to assert against the known-correct fixture output independent of matrix shape.
    const predictions = yProbaValues.map((row) => {
      let bestIdx = 0;
      let bestScore = -Infinity;
      row.forEach((v, idx) => {
        const t = thresholds[String(classes[idx])] ?? 1;
        const score = v / t;
        if (score > bestScore) {
          bestScore = score;
          bestIdx = idx;
        }
      });
      return classes[bestIdx];
    });

    expect(predictions).toEqual(expectedPredictions);
    expect(result).toBeDefined();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npx vitest run src/components/pages/ExperimentsPage/utils/classificationCharts.test.ts`
Expected: FAIL — `applyMulticlassThresholds is not defined`

- [ ] **Step 3: Implement `applyMulticlassThresholds`**

Add to `classificationCharts.ts`, alongside the existing `applyThreshold` function (do not modify `applyThreshold` itself):

```typescript
/**
 * Redraws a multiclass confusion matrix using per-class tuned thresholds via
 * scaled argmax: predicted class = argmax_i(proba[i] / threshold[classes[i]]).
 * This is the frontend parity implementation of skyulf-core's apply_thresholds().
 */
export function applyMulticlassThresholds(
  splitData: EvaluationSplit,
  thresholds: Record<string, number>
) {
  const { yTrue, yProba } = splitData;
  const classes = yProba.classes.map((c) => Number(c));

  const yPred = yProba.values.map((row) => {
    let bestIdx = 0;
    let bestScore = -Infinity;
    row.forEach((value, idx) => {
      const threshold = thresholds[String(classes[idx])] ?? 1;
      const score = value / threshold;
      if (score > bestScore) {
        bestScore = score;
        bestIdx = idx;
      }
    });
    return classes[bestIdx];
  });

  return calculateConfusionMatrix(yTrue, yPred, classes);
}
```

Adjust field names (`splitData.yTrue`/`yProba.values`/`yProba.classes`) to exactly match `EvaluationSplit`'s real shape — re-check the type definition in the file before finalizing since these were inferred from context.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npx vitest run src/components/pages/ExperimentsPage/utils/classificationCharts.test.ts`
Expected: PASS

- [ ] **Step 5: Run full existing test file to confirm no regressions**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npx vitest run src/components/pages/ExperimentsPage/utils/classificationCharts.test.ts`
Expected: PASS (all tests, including pre-existing `applyThreshold` tests)

- [ ] **Step 6: Run frontend checks**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npx eslint src/components/pages/ExperimentsPage/utils/classificationCharts.ts && npx tsc --project tsconfig.json --noEmit`
Expected: clean

- [ ] **Step 7: Commit**

```bash
git add frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts.ts frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts.test.ts
git commit -m "feat(threshold-tuning): add multiclass scaled-argmax confusion matrix redraw"
```

---

## Task 7: `PerClassConfusionMatrix` redraw wiring

**Files:**
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx`

**Interfaces:**
- Consumes: `applyMulticlassThresholds` (Task 6).
- Produces: new props `tunedThresholds?: Record<string, number> | null` and `useTunedThresholds?: boolean` on `PerClassConfusionMatrix`, feeding into its existing `matrixBySplit` memoization.

- [ ] **Step 1: Add the new props to the component's prop type**

In `PerClassConfusionMatrix.tsx`, extend the props interface:

```typescript
interface PerClassConfusionMatrixProps {
  // ...existing props...
  tunedThresholds?: Record<string, number> | null;
  useTunedThresholds?: boolean;
}
```

- [ ] **Step 2: Extend the `matrixBySplit` memoization**

Locate the existing `useMemo` computing `matrixBySplit` (confirmed present in the full file read earlier). Add a branch: when `useTunedThresholds && tunedThresholds` is truthy, call `applyMulticlassThresholds(splitData, tunedThresholds)` instead of the existing unthresholded/binary-threshold computation, for each split. Import `applyMulticlassThresholds` from `../utils/classificationCharts`.

- [ ] **Step 3: Manually verify via existing Storybook/dev harness if present**

Run: `grep -rl "PerClassConfusionMatrix" frontend/ml-canvas/src --include=*.stories.tsx`

If a story file exists, run `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npx vitest run` for any associated snapshot test; otherwise proceed to Step 4 (this component's existing test coverage, if any, was not found in earlier reads — confirm with `grep -rl "PerClassConfusionMatrix" frontend/ml-canvas/src --include=*.test.tsx` before skipping).

- [ ] **Step 4: Run frontend checks**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npx eslint src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx && npx tsc --project tsconfig.json --noEmit`
Expected: clean

- [ ] **Step 5: Commit**

```bash
git add frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx
git commit -m "feat(threshold-tuning): wire tuned-threshold redraw into PerClassConfusionMatrix"
```

---

## Task 8: Threshold Tuning panel in `EvaluationView`/`ExperimentsPage`

**Files:**
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/EvaluationView.tsx`
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx`

**Interfaces:**
- Consumes: `thresholdTuningApi` (Task 5), `applyMulticlassThresholds`/updated `PerClassConfusionMatrix` (Tasks 6-7).
- Produces: new state in `ExperimentsPage.tsx` (`selectedTuningMetric`, `tuningPreview`, `useTunedThresholds`) passed as new props into `EvaluationView`; new "Threshold Tuning" panel UI in `EvaluationView.tsx` with metric dropdown, Preview/Save/Toggle/Clear buttons, and a badge showing whether tuned thresholds are currently enabled for the job.

- [ ] **Step 1: Add new state to `ExperimentsPage.tsx`**

Near the existing `threshold`/`cmView` state declarations (lines 67-79), add:

```typescript
const [selectedTuningMetric, setSelectedTuningMetric] = useState<string>("f1");
const [tuningPreview, setTuningPreview] = useState<ThresholdPreviewResult | null>(null);
const [useTunedThresholds, setUseTunedThresholds] = useState(false);
const [tuningError, setTuningError] = useState<string | null>(null);
```

Import `ThresholdPreviewResult` from `../core/api/thresholdTuning` and `thresholdTuningApi` from the same module (adjust relative path to match `ExperimentsPage.tsx`'s actual location).

- [ ] **Step 2: Add handler functions in `ExperimentsPage.tsx`**

```typescript
const handlePreviewThresholds = async () => {
  setTuningError(null);
  try {
    const result = await thresholdTuningApi.preview(jobId, selectedTuningMetric);
    setTuningPreview(result);
  } catch (err) {
    setTuningError(err instanceof Error ? err.message : "Failed to preview thresholds");
  }
};

const handleSaveThresholds = async () => {
  if (!tuningPreview) return;
  setTuningError(null);
  try {
    await thresholdTuningApi.save(jobId, tuningPreview);
    setUseTunedThresholds(true);
  } catch (err) {
    setTuningError(err instanceof Error ? err.message : "Failed to save thresholds");
  }
};

const handleToggleThresholds = async (enabled: boolean) => {
  setTuningError(null);
  try {
    await thresholdTuningApi.toggle(jobId, enabled);
    setUseTunedThresholds(enabled);
  } catch (err) {
    setTuningError(err instanceof Error ? err.message : "Failed to toggle thresholds");
  }
};

const handleClearThresholds = async () => {
  setTuningError(null);
  try {
    await thresholdTuningApi.clear(jobId);
    setTuningPreview(null);
    setUseTunedThresholds(false);
  } catch (err) {
    setTuningError(err instanceof Error ? err.message : "Failed to clear thresholds");
  }
};
```

Adjust `jobId` to whatever variable name `ExperimentsPage.tsx` already uses for the current job (confirm via the file's existing prop/state before finalizing).

- [ ] **Step 3: Pass new props into `<EvaluationView>`**

At the existing `<EvaluationView>` call site (lines 450-477), add:

```typescript
selectedTuningMetric={selectedTuningMetric}
onSelectedTuningMetricChange={setSelectedTuningMetric}
tuningPreview={tuningPreview}
tuningError={tuningError}
useTunedThresholds={useTunedThresholds}
onPreviewThresholds={handlePreviewThresholds}
onSaveThresholds={handleSaveThresholds}
onToggleThresholds={handleToggleThresholds}
onClearThresholds={handleClearThresholds}
```

- [ ] **Step 4: Add the new props to `EvaluationView`'s prop type and render the panel**

In `EvaluationView.tsx`, extend the props interface with the same fields (typed to match Step 3), and add a new panel near the existing control bar (before or alongside the `PerClassConfusionMatrix` render call):

```tsx
<div className="threshold-tuning-panel">
  <h4>Threshold Tuning</h4>
  <select
    value={selectedTuningMetric}
    onChange={(e) => onSelectedTuningMetricChange(e.target.value)}
  >
    <option value="accuracy">Accuracy</option>
    <option value="f1">F1</option>
    <option value="precision">Precision</option>
    <option value="recall">Recall</option>
    <option value="balanced_accuracy">Balanced Accuracy</option>
    <option value="roc_auc">ROC AUC</option>
  </select>
  <button onClick={onPreviewThresholds}>Preview</button>
  {tuningPreview && (
    <>
      <span>
        Computed from {tuningPreview.split_used} split
        {tuningPreview.split_used === "test" && (
          <em> (no validation split available — using test split)</em>
        )}
      </span>
      <button onClick={onSaveThresholds}>Save</button>
    </>
  )}
  {useTunedThresholds !== undefined && (
    <label>
      <input
        type="checkbox"
        checked={useTunedThresholds}
        onChange={(e) => onToggleThresholds(e.target.checked)}
      />
      Use tuned thresholds at prediction time
    </label>
  )}
  <button onClick={onClearThresholds}>Clear</button>
  {tuningError && <span className="error-text">{tuningError}</span>}
</div>
```

Pass `tunedThresholds={tuningPreview?.thresholds}` and `useTunedThresholds={useTunedThresholds}` into the existing `<PerClassConfusionMatrix>` render call.

- [ ] **Step 5: Run frontend checks**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npx eslint src/components/pages/ExperimentsPage/components/EvaluationView.tsx src/components/pages/ExperimentsPage.tsx && npx tsc --project tsconfig.json --noEmit`
Expected: clean

- [ ] **Step 6: Manually verify in the dev server**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npm run dev` (start if not already running), navigate to a completed job's Evaluation view, confirm the Threshold Tuning panel renders, Preview populates a result, Save/Toggle/Clear round-trip against the backend without console errors.

- [ ] **Step 7: Commit**

```bash
git add frontend/ml-canvas/src/components/pages/ExperimentsPage/components/EvaluationView.tsx frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx
git commit -m "feat(threshold-tuning): add Threshold Tuning panel to Evaluation view"
```

---

## Task 9: Train/Test Splitter hint

**Files:**
- Modify: `frontend/ml-canvas/src/modules/nodes/modeling/TrainTestSplitNode.tsx`

**Interfaces:**
- Produces: an additional informational `<p>` hint in the node's UI, no new props/state.

- [ ] **Step 1: Add the hint text**

In `TrainTestSplitNode.tsx`, right after the Validation Size input's existing helper `<p>` text (confirmed location ~line 85-220):

```tsx
<p className="field-hint">
  Setting a validation size enables threshold tuning to compute against a
  held-out validation split instead of falling back to the test split.
</p>
```

- [ ] **Step 2: Run frontend checks**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npx eslint src/modules/nodes/modeling/TrainTestSplitNode.tsx && npx tsc --project tsconfig.json --noEmit`
Expected: clean

- [ ] **Step 3: Commit**

```bash
git add frontend/ml-canvas/src/modules/nodes/modeling/TrainTestSplitNode.tsx
git commit -m "docs(threshold-tuning): hint that validation_size improves threshold tuning"
```

---

## Task 10: Inference page override controls

**Files:**
- Modify: `frontend/ml-canvas/src/components/pages/InferencePage.tsx`

**Interfaces:**
- Consumes: updated `deploymentApi.predict()`/`PredictionRequest`/`PredictionResponse` (Task 5).
- Produces: new local state (`overrideThresholdsEnabled`, `overrideThresholdsValue`) and JSX near the Predict button; `handlePredict` passes `override_thresholds` through when enabled.

- [ ] **Step 1: Add new state near existing Inference page state (lines ~480-530)**

```typescript
const [overrideThresholdsEnabled, setOverrideThresholdsEnabled] = useState(false);
const [overrideThresholdsValue, setOverrideThresholdsValue] = useState<Record<string, number>>({});
```

- [ ] **Step 2: Update `handlePredict` to pass `override_thresholds`**

In `handlePredict` (~lines 815-865), where `deploymentApi.predict(data)` is called, change to:

```typescript
const response = await deploymentApi.predict({
  ...data,
  override_thresholds: overrideThresholdsEnabled ? overrideThresholdsValue : null,
});
```

Adjust to match the exact existing call signature/shape of `data` at that call site.

- [ ] **Step 3: Add the collapsible JSX control near the Predict button (~lines 1100-1175)**

```tsx
<details className="advanced-override-thresholds">
  <summary>Advanced: override thresholds</summary>
  {predictionClasses.map((cls) => (
    <label key={cls}>
      Class {cls}:
      <input
        type="number"
        min={0}
        max={1}
        step={0.01}
        value={overrideThresholdsValue[cls] ?? ""}
        onChange={(e) =>
          setOverrideThresholdsValue((prev) => ({
            ...prev,
            [cls]: Number(e.target.value),
          }))
        }
      />
    </label>
  ))}
  <label>
    <input
      type="checkbox"
      checked={overrideThresholdsEnabled}
      onChange={(e) => setOverrideThresholdsEnabled(e.target.checked)}
    />
    Apply these thresholds to this prediction
  </label>
</details>
```

Adjust `predictionClasses` to whatever variable already holds the model's class list on this page (confirm via existing state before finalizing — if none exists yet, derive it from the last prediction response's `thresholds_applied` keys or a model-metadata endpoint already used elsewhere on the page).

- [ ] **Step 4: Display `thresholds_applied` in the prediction result panel**

Wherever the page currently renders `PredictionResponse` fields, add:

```tsx
{response.thresholds_applied && (
  <p className="thresholds-applied-note">
    Thresholds applied: {JSON.stringify(response.thresholds_applied)}
  </p>
)}
```

- [ ] **Step 5: Run frontend checks**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npx eslint src/components/pages/InferencePage.tsx && npx tsc --project tsconfig.json --noEmit`
Expected: clean

- [ ] **Step 6: Manually verify in dev server**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npm run dev`, navigate to Inference page, expand "Advanced: override thresholds", set values, run a prediction, confirm `thresholds_applied` renders and no console errors.

- [ ] **Step 7: Commit**

```bash
git add frontend/ml-canvas/src/components/pages/InferencePage.tsx
git commit -m "feat(threshold-tuning): add ad-hoc override thresholds control to Inference page"
```

---

## Task 11: Final gate — full lint/type/test suites + changelog

**Files:**
- Modify: `changelog/0.7.x.md` (verify the actual current heading first — do not assume version number)

**Interfaces:** None (verification-only task).

- [ ] **Step 1: Confirm the current changelog heading**

Run: `head -30 /Users/BH7043/Skyulf/changelog/0.7.x.md`

- [ ] **Step 2: Add a changelog entry under the correct current heading**

```markdown
### Added
- Threshold tuning: preview, save, toggle, and clear per-job tuned classification thresholds from the Evaluation view; saved thresholds are applied automatically at `/predict` time (or overridden ad-hoc from the Inference page); multiclass confusion matrices redraw live via scaled-argmax.
```

(Insert under whatever the actual next-unreleased heading is per Step 1's output — do not guess the version number.)

- [ ] **Step 3: Run full backend gate**

Run: `cd /Users/BH7043/Skyulf && source .venv/bin/activate && ruff check . && ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py && ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`
Expected: all clean; fix any drift before proceeding.

- [ ] **Step 4: Run full backend test suite**

Run: `cd /Users/BH7043/Skyulf && source .venv/bin/activate && python -m pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 5: Run full frontend gate**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npm run lint && npx tsc --project tsconfig.json --noEmit && npm run build`
Expected: all clean.

- [ ] **Step 6: Run full frontend test suite**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && npx vitest run`
Expected: all PASS.

- [ ] **Step 7: Commit the changelog entry**

```bash
git add changelog/0.7.x.md
git commit -m "docs(threshold-tuning): add changelog entry for Phase 2"
```
