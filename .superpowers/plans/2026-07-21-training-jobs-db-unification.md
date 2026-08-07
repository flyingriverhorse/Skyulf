# Training Jobs DB Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two parallel job tables (`BasicTrainingJob`, `AdvancedTuningJob`) with one
unified `TrainingJob` model/table (`training_jobs`), discriminated by a new `run_mode: "fixed" |
"tuned"` column, with no data migration (dev DB is dropped and recreated).

**Architecture:** `TrainingJob(MLJob)` becomes the single source of truth. All call sites that
previously queried/looped over both `BasicTrainingJob` and `AdvancedTuningJob` now query
`TrainingJob` filtered by `run_mode`. `BasicTrainingManager`/`AdvancedTuningManager` stay as
separate classes (per the approved design) but both operate on `TrainingJob`, filtered by
`run_mode="fixed"` / `run_mode="tuned"` respectively — this filter is load-bearing: without it,
`JobManager`'s "try basic first, then advanced" fallback pattern would incorrectly match a tuned
job's row when looking it up via the basic manager (same table, same id).

**Tech Stack:** Python 3.11+, SQLAlchemy (async + sync sessions), FastAPI, pytest.

## Global Constraints

- No Alembic migration, no `_legacy` table retention — the dev DB is dropped and recreated
  against the new schema (per user direction).
- `basic_training_manager.py` and `advanced_tuning_manager.py` remain separate classes; do not
  merge their logic.
- Every query that previously scanned/looped over `(BasicTrainingJob, AdvancedTuningJob)` for a
  single-mode lookup must add an explicit `TrainingJob.run_mode == "fixed"` or `"tuned"` filter —
  omitting this silently breaks correctness now that both modes share one table/id-space.
- Run `ruff check` and the relevant `pytest` file(s) after every task; run the full backend test
  suite at the end.

---

### Task 1: Unified `TrainingJob` model + `Deployment` FK

**Files:**
- Modify: `backend/database/models.py:298-346` (replace `BasicTrainingJob`/`AdvancedTuningJob`
  with `TrainingJob`; add FK to `Deployment.job_id`)

**Interfaces:**
- Produces: `TrainingJob(MLJob)` — table `training_jobs`, columns: all `MLJob` base columns +
  `run_mode: str`, `version: int`, `hyperparameters: JSON | None`, `search_strategy: str | None`,
  `search_space: JSON | None`, `baseline_hyperparameters: JSON | None`, `n_iterations: int | None`,
  `scoring: str | None`, `random_state: int | None`, `cross_validation: JSON | None`,
  `results: JSON | None`, `best_params: JSON | None`, `best_score: float | None`. Method
  `to_dict() -> dict` merging base + all the above fields (including `run_mode` and `version`).

- [ ] **Step 1: Replace the two model classes with `TrainingJob`**

In `backend/database/models.py`, replace the entire `BasicTrainingJob` and `AdvancedTuningJob`
class definitions (currently lines ~298-346) with:

```python
class TrainingJob(MLJob):
    """Unified background training/tuning job, discriminated by `run_mode`.

    `run_mode="fixed"` is a single fixed-hyperparameter fit (formerly
    `BasicTrainingJob`); `run_mode="tuned"` is a hyperparameter search
    (formerly `AdvancedTuningJob`). Both share one id-space/table so a job's
    id alone never ambiguously refers to two different rows.
    """

    __tablename__ = "training_jobs"

    run_mode: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # fixed-mode only
    hyperparameters: Mapped[Any | None] = mapped_column(JSON, nullable=True)

    # tuned-mode only (nullable; unused when run_mode == "fixed")
    search_strategy: Mapped[str | None] = mapped_column(String(20), nullable=True)
    search_space: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    baseline_hyperparameters: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    n_iterations: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scoring: Mapped[str | None] = mapped_column(String(100), nullable=True)
    random_state: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cross_validation: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    results: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    best_params: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    best_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    owner = relationship("User", backref="training_jobs")

    def to_dict(self) -> dict:
        data = self.to_dict_base()
        data.update(
            {
                "run_mode": self.run_mode,
                "version": self.version,
                "hyperparameters": self.hyperparameters,
                "search_strategy": self.search_strategy,
                "search_space": self.search_space,
                "baseline_hyperparameters": self.baseline_hyperparameters,
                "n_iterations": self.n_iterations,
                "scoring": self.scoring,
                "random_state": self.random_state,
                "cross_validation": self.cross_validation,
                "results": self.results,
                "best_params": self.best_params,
                "best_score": self.best_score,
            }
        )
        return data
```

- [ ] **Step 2: Add the FK constraint on `Deployment.job_id`**

Find this in `backend/database/models.py` (in the `Deployment` class):

```python
    # ID of the TrainingJob or HyperparameterTuningJob
    job_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
```

Replace with:

```python
    # ID of the unified TrainingJob (fixed or tuned run_mode)
    job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("training_jobs.id"), nullable=False, index=True
    )
```

- [ ] **Step 3: Verify the module still imports cleanly**

Run: `cd /Users/BH7043/Skyulf && python -c "from backend.database import models"`
Expected: no errors (other files still reference `BasicTrainingJob`/`AdvancedTuningJob` at this
point, but that import itself doesn't touch them, so this is just a models.py syntax/definition
sanity check — the full app will not import cleanly until later tasks finish).

- [ ] **Step 4: Commit**

```bash
cd /Users/BH7043/Skyulf
git add backend/database/models.py
git commit -m "refactor(db): unify BasicTrainingJob/AdvancedTuningJob into TrainingJob(run_mode)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: `run_mode`-aware filtering in `TrainingJobManagerBase`

**Files:**
- Modify: `backend/ml_pipeline/_execution/job_manager_base.py`

**Interfaces:**
- Consumes: `TrainingJob` from Task 1.
- Produces: `_cancel_job(session, model, job_id, run_mode=None)`,
  `_update_status_sync(session, model, job_id, status, error, result, logs, apply_fields_fn,
  run_mode=None)` — both now accept an optional `run_mode` filter so callers scoped to one mode
  never touch a row belonging to the other mode.

- [ ] **Step 1: Add `run_mode` parameter to `_cancel_job`**

Replace:

```python
    @staticmethod
    async def _cancel_job(
        session: AsyncSession, model: type[Any], job_id: str
    ) -> bool:
        """Cancel a QUEUED/RUNNING job row and revoke its Celery task.

        Returns True if the job was found and cancelled, False otherwise.
        """
        stmt = select(model).where(model.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()
```

With:

```python
    @staticmethod
    async def _cancel_job(
        session: AsyncSession, model: type[Any], job_id: str, run_mode: str | None = None
    ) -> bool:
        """Cancel a QUEUED/RUNNING job row and revoke its Celery task.

        `run_mode`, when given, scopes the lookup to that mode so a caller
        scoped to "fixed" jobs (e.g. BasicTrainingManager) never cancels a
        "tuned" row that happens to share the same underlying table.

        Returns True if the job was found and cancelled, False otherwise.
        """
        stmt = select(model).where(model.id == job_id)
        if run_mode is not None:
            stmt = stmt.where(model.run_mode == run_mode)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()
```

- [ ] **Step 2: Add `run_mode` parameter to `_update_status_sync`**

Replace:

```python
    @staticmethod
    def _update_status_sync(
        session: Session,
        model: type[Any],
        job_id: str,
        status: JobStatus | None,
        error: str | None,
        result: dict[str, Any] | None,
        logs: list[str] | None,
        apply_fields_fn: Callable[
            [Any, JobStatus | None, str | None, list[str] | None, dict[str, Any] | None],
            None,
        ],
    ) -> bool:
        """Update a job row; guard against overwriting a CANCELLED status.

        *apply_fields_fn* is the model-specific hook that writes
        status/error/logs/result onto the concrete job row.  Returns True if
        the job was found, False otherwise.
        """
        job = session.query(model).filter(model.id == job_id).first()
        if not job:
            return False
```

With:

```python
    @staticmethod
    def _update_status_sync(
        session: Session,
        model: type[Any],
        job_id: str,
        status: JobStatus | None,
        error: str | None,
        result: dict[str, Any] | None,
        logs: list[str] | None,
        apply_fields_fn: Callable[
            [Any, JobStatus | None, str | None, list[str] | None, dict[str, Any] | None],
            None,
        ],
        run_mode: str | None = None,
    ) -> bool:
        """Update a job row; guard against overwriting a CANCELLED status.

        *apply_fields_fn* is the model-specific hook that writes
        status/error/logs/result onto the concrete job row. `run_mode`, when
        given, scopes the lookup to that mode (see `_cancel_job` for why this
        matters now that both modes share one table). Returns True if the
        job was found, False otherwise.
        """
        query = session.query(model).filter(model.id == job_id)
        if run_mode is not None:
            query = query.filter(model.run_mode == run_mode)
        job = query.first()
        if not job:
            return False
```

- [ ] **Step 3: Sanity-check syntax**

Run: `cd /Users/BH7043/Skyulf && python -m py_compile backend/ml_pipeline/_execution/job_manager_base.py`
Expected: no output (success).

- [ ] **Step 4: Commit**

```bash
cd /Users/BH7043/Skyulf
git add backend/ml_pipeline/_execution/job_manager_base.py
git commit -m "refactor(job-manager-base): support run_mode-scoped job lookups

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: `basic_training_manager.py` → `TrainingJob(run_mode="fixed")`

**Files:**
- Modify: `backend/ml_pipeline/_execution/basic_training_manager.py`

**Interfaces:**
- Consumes: `TrainingJob` (Task 1), `_cancel_job`/`_update_status_sync` with `run_mode` (Task 2).
- Produces: `BasicTrainingManager` — same public method signatures as before
  (`create_training_job`, `map_training_job_to_info`, `cancel_training_job`,
  `update_status_sync`, `get_training_job`, `list_training_jobs`), now all scoped to
  `TrainingJob.run_mode == "fixed"`.

- [ ] **Step 1: Swap the import and type annotations**

Replace:
```python
from backend.database.models import BasicTrainingJob
```
With:
```python
from backend.database.models import TrainingJob
```

Then throughout the file, replace every occurrence of the type annotation/reference
`BasicTrainingJob` with `TrainingJob` (parameter type hints in `map_training_job_to_info`,
`_update_training_result`, `_append_job_logs`, `_handle_cancelled_status_update`,
`_apply_status_update_fields` — a straightforward find/replace of the identifier).

- [ ] **Step 2: Set `run_mode="fixed"` on job creation**

Replace:
```python
        job = BasicTrainingJob(
            id=job_id,
            pipeline_id=pipeline_id,
            node_id=node_id,
            dataset_source_id=dataset_id,
            user_id=user_id,
            status=JobStatus.QUEUED.value,
            version=version,
            model_type=model_type_val,
            graph=graph,
            job_metadata={"branch_index": branch_index},
            started_at=datetime.now(UTC),
        )
```
With:
```python
        job = TrainingJob(
            id=job_id,
            pipeline_id=pipeline_id,
            node_id=node_id,
            dataset_source_id=dataset_id,
            user_id=user_id,
            status=JobStatus.QUEUED.value,
            run_mode="fixed",
            version=version,
            model_type=model_type_val,
            graph=graph,
            job_metadata={"branch_index": branch_index},
            started_at=datetime.now(UTC),
        )
```

- [ ] **Step 3: Scope `cancel_training_job` to `run_mode="fixed"`**

Replace:
```python
        return await TrainingJobManagerBase._cancel_job(session, BasicTrainingJob, job_id)
```
With:
```python
        return await TrainingJobManagerBase._cancel_job(
            session, TrainingJob, job_id, run_mode="fixed"
        )
```

- [ ] **Step 4: Scope `update_status_sync` to `run_mode="fixed"`**

Replace:
```python
        return TrainingJobManagerBase._update_status_sync(
            session,
            BasicTrainingJob,
            job_id,
            status,
            error,
            result,
            logs,
            BasicTrainingManager._apply_status_update_fields,
        )
```
With:
```python
        return TrainingJobManagerBase._update_status_sync(
            session,
            TrainingJob,
            job_id,
            status,
            error,
            result,
            logs,
            BasicTrainingManager._apply_status_update_fields,
            run_mode="fixed",
        )
```

- [ ] **Step 5: Scope `get_training_job` to `run_mode="fixed"`**

Replace:
```python
        stmt = select(BasicTrainingJob).where(BasicTrainingJob.id == job_id)
```
(the one inside `get_training_job`) with:
```python
        stmt = select(TrainingJob).where(
            TrainingJob.id == job_id, TrainingJob.run_mode == "fixed"
        )
```

- [ ] **Step 6: Scope `list_training_jobs` to `run_mode="fixed"`**

Replace:
```python
        result_train = await session.execute(
            select(BasicTrainingJob)
            .where(BasicTrainingJob.model_type != "preview")
            .order_by(BasicTrainingJob.started_at.desc())
            .limit(effective_limit)
            .offset(skip)
        )
```
With:
```python
        result_train = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.run_mode == "fixed", TrainingJob.model_type != "preview")
            .order_by(TrainingJob.started_at.desc())
            .limit(effective_limit)
            .offset(skip)
        )
```

- [ ] **Step 7: Sanity-check syntax and remaining references**

Run: `cd /Users/BH7043/Skyulf && python -m py_compile backend/ml_pipeline/_execution/basic_training_manager.py && grep -n "BasicTrainingJob" backend/ml_pipeline/_execution/basic_training_manager.py`
Expected: `py_compile` succeeds silently; the `grep` prints no matches (empty output).

- [ ] **Step 8: Commit**

```bash
cd /Users/BH7043/Skyulf
git add backend/ml_pipeline/_execution/basic_training_manager.py
git commit -m "refactor(basic-training-manager): use TrainingJob(run_mode='fixed')

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: `advanced_tuning_manager.py` → `TrainingJob(run_mode="tuned")`

**Files:**
- Modify: `backend/ml_pipeline/_execution/advanced_tuning_manager.py`

**Interfaces:**
- Consumes: `TrainingJob` (Task 1), `_cancel_job`/`_update_status_sync` with `run_mode` (Task 2).
- Produces: `AdvancedTuningManager` — same public method signatures as before, now scoped to
  `TrainingJob.run_mode == "tuned"`; `run_number` field is gone, replaced by `version` everywhere
  (including `JobInfo.version` mapping).

- [ ] **Step 1: Swap the import**

Replace:
```python
from backend.database.models import AdvancedTuningJob
```
With:
```python
from backend.database.models import TrainingJob
```

Then replace every remaining type-annotation occurrence of `AdvancedTuningJob` in this file with
`TrainingJob` (in `map_tuning_job_to_info`, `_update_tuning_result`, `_append_job_logs`,
`_handle_cancelled_status_update`, `_apply_status_update_fields`).

- [ ] **Step 2: Set `run_mode="tuned"` and rename `run_number` → `version` on job creation**

Replace:
```python
        job = AdvancedTuningJob(
            id=job_id,
            pipeline_id=pipeline_id,
            node_id=node_id,
            dataset_source_id=dataset_id,
            user_id=user_id,
            status=JobStatus.QUEUED.value,
            run_number=next_version,
            model_type=model_type,
            search_strategy=search_strategy,
            graph=graph,
            job_metadata={"branch_index": branch_index},
            started_at=datetime.now(UTC),
        )
```
With:
```python
        job = TrainingJob(
            id=job_id,
            pipeline_id=pipeline_id,
            node_id=node_id,
            dataset_source_id=dataset_id,
            user_id=user_id,
            status=JobStatus.QUEUED.value,
            run_mode="tuned",
            version=next_version,
            model_type=model_type,
            search_strategy=search_strategy,
            graph=graph,
            job_metadata={"branch_index": branch_index},
            started_at=datetime.now(UTC),
        )
```

- [ ] **Step 3: Rename `run_number` → `version` in `map_tuning_job_to_info`**

Replace:
```python
            version=type_cast(int | None, job.run_number),
```
With:
```python
            version=type_cast(int | None, job.version),
```

- [ ] **Step 4: Scope `cancel_tuning_job` to `run_mode="tuned"`**

Replace:
```python
        return await TrainingJobManagerBase._cancel_job(session, AdvancedTuningJob, job_id)
```
With:
```python
        return await TrainingJobManagerBase._cancel_job(
            session, TrainingJob, job_id, run_mode="tuned"
        )
```

- [ ] **Step 5: Scope `update_status_sync` to `run_mode="tuned"`**

Replace:
```python
        return TrainingJobManagerBase._update_status_sync(
            session,
            AdvancedTuningJob,
            job_id,
            status,
            error,
            result,
            logs,
            AdvancedTuningManager._apply_status_update_fields,
        )
```
With:
```python
        return TrainingJobManagerBase._update_status_sync(
            session,
            TrainingJob,
            job_id,
            status,
            error,
            result,
            logs,
            AdvancedTuningManager._apply_status_update_fields,
            run_mode="tuned",
        )
```

- [ ] **Step 6: Scope `get_tuning_job` to `run_mode="tuned"`**

Replace:
```python
        stmt = select(AdvancedTuningJob).where(AdvancedTuningJob.id == job_id)
```
(the one inside `get_tuning_job`) with:
```python
        stmt = select(TrainingJob).where(
            TrainingJob.id == job_id, TrainingJob.run_mode == "tuned"
        )
```

- [ ] **Step 7: Scope `list_tuning_jobs` to `run_mode="tuned"`**

Replace:
```python
        result_tune = await session.execute(
            select(AdvancedTuningJob)
            .order_by(AdvancedTuningJob.started_at.desc())
            .limit(limit)
            .offset(skip)
        )
```
With:
```python
        result_tune = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.run_mode == "tuned")
            .order_by(TrainingJob.started_at.desc())
            .limit(limit)
            .offset(skip)
        )
```

- [ ] **Step 8: Scope `get_latest_tuning_job_for_node` to `run_mode="tuned"`**

Replace:
```python
        result = await session.execute(
            select(AdvancedTuningJob)
            .where(AdvancedTuningJob.node_id == node_id)
            .where(AdvancedTuningJob.status == JobStatus.COMPLETED.value)
            .order_by(AdvancedTuningJob.finished_at.desc())
            .limit(1)
        )
```
With:
```python
        result = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.run_mode == "tuned")
            .where(TrainingJob.node_id == node_id)
            .where(TrainingJob.status == JobStatus.COMPLETED.value)
            .order_by(TrainingJob.finished_at.desc())
            .limit(1)
        )
```

- [ ] **Step 9: Scope `get_best_tuning_job_for_model` to `run_mode="tuned"`**

Replace:
```python
        result = await session.execute(
            select(AdvancedTuningJob)
            .where(AdvancedTuningJob.model_type == model_type)
            .where(AdvancedTuningJob.status == JobStatus.COMPLETED.value)
            .order_by(AdvancedTuningJob.finished_at.desc())
            .limit(1)
        )
```
With:
```python
        result = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.run_mode == "tuned")
            .where(TrainingJob.model_type == model_type)
            .where(TrainingJob.status == JobStatus.COMPLETED.value)
            .order_by(TrainingJob.finished_at.desc())
            .limit(1)
        )
```

- [ ] **Step 10: Scope `get_tuning_jobs_for_model` to `run_mode="tuned"`**

Replace:
```python
        result = await session.execute(
            select(AdvancedTuningJob)
            .where(AdvancedTuningJob.model_type == model_type)
            .where(AdvancedTuningJob.status == JobStatus.COMPLETED.value)
            .order_by(AdvancedTuningJob.finished_at.desc())
            .limit(effective_limit)
        )
```
With:
```python
        result = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.run_mode == "tuned")
            .where(TrainingJob.model_type == model_type)
            .where(TrainingJob.status == JobStatus.COMPLETED.value)
            .order_by(TrainingJob.finished_at.desc())
            .limit(effective_limit)
        )
```

- [ ] **Step 11: Sanity-check syntax and remaining references**

Run: `cd /Users/BH7043/Skyulf && python -m py_compile backend/ml_pipeline/_execution/advanced_tuning_manager.py && grep -n "AdvancedTuningJob\|run_number" backend/ml_pipeline/_execution/advanced_tuning_manager.py`
Expected: `py_compile` succeeds silently; `grep` prints no matches.

- [ ] **Step 12: Commit**

```bash
cd /Users/BH7043/Skyulf
git add backend/ml_pipeline/_execution/advanced_tuning_manager.py
git commit -m "refactor(advanced-tuning-manager): use TrainingJob(run_mode='tuned')

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: `strategies.py` — dispatch by `run_mode`, not `isinstance`

**Files:**
- Modify: `backend/ml_pipeline/_execution/strategies.py`

**Interfaces:**
- Consumes: `TrainingJob` (Task 1).
- Produces: `JobStrategy.get_job_model()` returns `TrainingJob` for both strategies;
  `JobStrategy.get_job(session, job_id, run_mode)` takes an explicit `run_mode`;
  `JobStrategyFactory.get_strategy_by_job(job)` dispatches on `job.run_mode`;
  `JobStrategyFactory.find_job(session, job_id)` does a single lookup (no more double-query
  across two identical-table strategies).

- [ ] **Step 1: Swap the import**

Replace:
```python
from backend.database.models import AdvancedTuningJob, BasicTrainingJob, MLJob
```
With:
```python
from backend.database.models import MLJob, TrainingJob
```

- [ ] **Step 2: Make `get_job` accept an explicit `run_mode` and each strategy declare its mode**

Replace:
```python
class JobStrategy(ABC):
    """
    Abstract base class for job execution strategies.
    Encapsulates logic specific to different job types (Training, Tuning, etc.).
    """

    @abstractmethod
    def get_job_model(self) -> type[MLJob]:
        """Returns the SQLAlchemy model class for this job type."""

    def get_job(self, session: Session, job_id: str) -> MLJob | None:
        """Fetches the job from the database."""
        return session.query(self.get_job_model()).filter(self.get_job_model().id == job_id).first()
```
With:
```python
class JobStrategy(ABC):
    """
    Abstract base class for job execution strategies.
    Encapsulates logic specific to different job types (Training, Tuning, etc.).
    """

    run_mode: str

    def get_job_model(self) -> type[MLJob]:
        """Returns the SQLAlchemy model class for this job type."""
        return TrainingJob

    def get_job(self, session: Session, job_id: str) -> MLJob | None:
        """Fetches the job from the database, scoped to this strategy's `run_mode`."""
        model = self.get_job_model()
        return (
            session.query(model)
            .filter(model.id == job_id, model.run_mode == self.run_mode)
            .first()
        )
```

- [ ] **Step 3: Remove the now-redundant `get_job_model` overrides and declare `run_mode` per subclass**

Replace:
```python
class BasicTrainingStrategy(JobStrategy):
    def get_job_model(self) -> type[MLJob]:
        return BasicTrainingJob

    def get_initial_log(self, job: MLJob) -> str:
```
With:
```python
class BasicTrainingStrategy(JobStrategy):
    run_mode = "fixed"

    def get_initial_log(self, job: MLJob) -> str:
```

Replace:
```python
class AdvancedTuningStrategy(JobStrategy):
    def get_job_model(self) -> type[MLJob]:
        return AdvancedTuningJob

    def get_initial_log(self, job: MLJob) -> str:
```
With:
```python
class AdvancedTuningStrategy(JobStrategy):
    run_mode = "tuned"

    def get_initial_log(self, job: MLJob) -> str:
```

- [ ] **Step 4: Dispatch `get_strategy_by_job` on `run_mode`, and simplify `find_job` to one query**

Replace:
```python
    @classmethod
    def get_strategy_by_job(cls, job: MLJob) -> JobStrategy:
        if isinstance(job, BasicTrainingJob):
            return cls._strategies[StepType.BASIC_TRAINING]
        elif isinstance(job, AdvancedTuningJob):
            return cls._strategies[StepType.ADVANCED_TUNING]
        else:
            raise ValueError(f"Unknown job type: {type(job)}")

    @classmethod
    def find_job(cls, session: Session, job_id: str) -> tuple[MLJob | None, JobStrategy | None]:
        """
        Tries to find the job in all known tables.
        Returns (job, strategy) or (None, None).
        """
        for strategy in cls._strategies.values():
            job = strategy.get_job(session, job_id)
            if job:
                return job, strategy
        return None, None
```
With:
```python
    @classmethod
    def get_strategy_by_job(cls, job: MLJob) -> JobStrategy:
        run_mode = getattr(job, "run_mode", None)
        if run_mode == "fixed":
            return cls._strategies[StepType.BASIC_TRAINING]
        elif run_mode == "tuned":
            return cls._strategies[StepType.ADVANCED_TUNING]
        else:
            raise ValueError(f"Unknown job run_mode: {run_mode!r} (job type: {type(job)})")

    @classmethod
    def find_job(cls, session: Session, job_id: str) -> tuple[MLJob | None, JobStrategy | None]:
        """
        Looks up the job by id (single shared table) and resolves its strategy
        from `run_mode`. Returns (job, strategy) or (None, None).
        """
        job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job is None:
            return None, None
        return job, cls.get_strategy_by_job(job)
```

- [ ] **Step 5: Sanity-check syntax and remaining references**

Run: `cd /Users/BH7043/Skyulf && python -m py_compile backend/ml_pipeline/_execution/strategies.py && grep -n "BasicTrainingJob\|AdvancedTuningJob" backend/ml_pipeline/_execution/strategies.py`
Expected: `py_compile` succeeds silently; `grep` prints no matches.

- [ ] **Step 6: Commit**

```bash
cd /Users/BH7043/Skyulf
git add backend/ml_pipeline/_execution/strategies.py
git commit -m "refactor(strategies): dispatch job strategy by run_mode, not table/isinstance

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: `job_service.py` — single-table lookup

**Files:**
- Modify: `backend/ml_pipeline/_services/job_service.py`

**Interfaces:**
- Consumes: `TrainingJob` (Task 1).
- Produces: `JobService.get_job_by_id_sync(session, job_id) -> TrainingJob | None`,
  `JobService.get_job_by_id(session, job_id) -> TrainingJob | None` — both now a single query
  instead of trying two tables.

- [ ] **Step 1: Replace the whole file body with the single-table version**

Replace the full contents of `backend/ml_pipeline/_services/job_service.py` with:

```python
"""
Job Service
-----------
Centralized service for retrieving job entities (TrainingJob) from the
database. TrainingJob is discriminated by `run_mode` ("fixed" | "tuned"),
so a single query by id is always unambiguous.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from backend.database.models import TrainingJob


class JobService:
    """Service for managing and retrieving Job entities."""

    @staticmethod
    def get_job_by_id_sync(session: Session, job_id: str) -> TrainingJob | None:
        """Retrieves a job by ID (Synchronous)."""
        return session.query(TrainingJob).filter(TrainingJob.id == job_id).first()

    @staticmethod
    async def get_job_by_id(session: AsyncSession, job_id: str) -> TrainingJob | None:
        """
        Retrieves a job by ID.

        Args:
            session: The async database session.
            job_id: The unique identifier of the job.

        Returns:
            The TrainingJob entity if found, else None.
        """
        stmt = select(TrainingJob).where(TrainingJob.id == job_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
```

- [ ] **Step 2: Sanity-check syntax**

Run: `cd /Users/BH7043/Skyulf && python -m py_compile backend/ml_pipeline/_services/job_service.py`
Expected: no output (success).

- [ ] **Step 3: Commit**

```bash
cd /Users/BH7043/Skyulf
git add backend/ml_pipeline/_services/job_service.py
git commit -m "refactor(job-service): single-table TrainingJob lookup

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: `jobs.py` facade (`JobManager`) — collapse dual-model loops

**Files:**
- Modify: `backend/ml_pipeline/_execution/jobs.py`

**Interfaces:**
- Consumes: `TrainingJob` (Task 1); `BasicTrainingManager`/`AdvancedTuningManager` (Tasks 3-4,
  unchanged public signatures).
- Produces: `JobManager` — all public method signatures unchanged; internals that looped over
  `(BasicTrainingJob, AdvancedTuningJob)` now either query `TrainingJob` once, or (where the
  method must stay run_mode-aware, e.g. `cancel_job`/`get_job`/`update_status_sync`) keep the
  "try basic manager, then tuning manager" delegation, which remains correct because Tasks 3-4
  scoped those managers' queries by `run_mode`.

- [ ] **Step 1: Swap the import**

Replace:
```python
from backend.database.models import AdvancedTuningJob, BasicTrainingJob
```
With:
```python
from backend.database.models import TrainingJob
```

- [ ] **Step 2: Collapse `find_active_job`'s dual-model loop into a single query**

Replace:
```python
        active = {JobStatus.QUEUED.value, JobStatus.RUNNING.value}
        for model in (BasicTrainingJob, AdvancedTuningJob):
            # Fetch all recent active candidates then filter by branch_index
            # in Python — avoids cross-db JSON operator differences.
            stmt = (
                select(model.id, model.job_metadata)
                .where(
                    model.dataset_source_id == dataset_id,
                    model.node_id == node_id,
                    model.status.in_(active),
                    model.created_at >= cutoff,
                )
                .with_for_update(skip_locked=True)
                .limit(20)
            )
            rows = (await session.execute(stmt)).all()
            for job_id, meta in rows:
                stored_idx = (meta or {}).get("branch_index", 0)
                if stored_idx == branch_index:
                    return job_id
        return None
```
With:
```python
        active = {JobStatus.QUEUED.value, JobStatus.RUNNING.value}
        # Fetch all recent active candidates (either run_mode) then filter by
        # branch_index in Python — avoids cross-db JSON operator differences.
        stmt = (
            select(TrainingJob.id, TrainingJob.job_metadata)
            .where(
                TrainingJob.dataset_source_id == dataset_id,
                TrainingJob.node_id == node_id,
                TrainingJob.status.in_(active),
                TrainingJob.created_at >= cutoff,
            )
            .with_for_update(skip_locked=True)
            .limit(20)
        )
        rows = (await session.execute(stmt)).all()
        for job_id, meta in rows:
            stored_idx = (meta or {}).get("branch_index", 0)
            if stored_idx == branch_index:
                return job_id
        return None
```

- [ ] **Step 3: Collapse `attach_celery_task_id`'s dual-model loop into a single query**

Replace:
```python
        for model in (BasicTrainingJob, AdvancedTuningJob):
            stmt = select(model).where(model.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job is None:
                continue
            meta = dict(job.job_metadata) if isinstance(job.job_metadata, dict) else {}
            meta["celery_task_id"] = task_id
            job.job_metadata = meta
            await session.commit()
            return
```
With:
```python
        stmt = select(TrainingJob).where(TrainingJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()
        if job is None:
            return
        meta = dict(job.job_metadata) if isinstance(job.job_metadata, dict) else {}
        meta["celery_task_id"] = task_id
        job.job_metadata = meta
        await session.commit()
```

- [ ] **Step 4: Collapse `promote_job`'s dual-model loop into a single query**

Replace:
```python
        for model_cls in (BasicTrainingJob, AdvancedTuningJob):
            stmt = select(model_cls).where(model_cls.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job:
                if job.status != "completed":
                    return False
                job.promoted_at = datetime.now(UTC)  # type: ignore[assignment]
                await session.commit()
                return True
        return False
```
With:
```python
        stmt = select(TrainingJob).where(TrainingJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()
        if job is None:
            return False
        if job.status != "completed":
            return False
        job.promoted_at = datetime.now(UTC)  # type: ignore[assignment]
        await session.commit()
        return True
```

- [ ] **Step 5: Collapse `unpromote_job`'s dual-model loop into a single query**

Replace:
```python
        for model_cls in (BasicTrainingJob, AdvancedTuningJob):
            stmt = select(model_cls).where(model_cls.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job:
                job.promoted_at = None  # type: ignore[assignment]
                await session.commit()
                return True
        return False
```
With:
```python
        stmt = select(TrainingJob).where(TrainingJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()
        if job is None:
            return False
        job.promoted_at = None  # type: ignore[assignment]
        await session.commit()
        return True
```

Note: `cancel_job`, `update_status_sync`, `get_job`, and `list_jobs` in this file already
delegate to `BasicTrainingManager`/`AdvancedTuningManager` (which Tasks 3-4 scoped by
`run_mode`) — leave those method bodies as-is; only the import line changes for them (already
covered by Step 1).

- [ ] **Step 6: Sanity-check syntax and remaining references**

Run: `cd /Users/BH7043/Skyulf && python -m py_compile backend/ml_pipeline/_execution/jobs.py && grep -n "BasicTrainingJob\|AdvancedTuningJob" backend/ml_pipeline/_execution/jobs.py`
Expected: `py_compile` succeeds silently; `grep` prints no matches.

- [ ] **Step 7: Commit**

```bash
cd /Users/BH7043/Skyulf
git add backend/ml_pipeline/_execution/jobs.py
git commit -m "refactor(job-manager): collapse dual-table loops onto unified TrainingJob

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 8: `meta.py` — unified job counts

**Files:**
- Modify: `backend/ml_pipeline/_internal/_routers/meta.py`

**Interfaces:**
- Consumes: `TrainingJob` (Task 1).
- Produces: `/stats` endpoint's `training_count`/`tuning_count` computed from `TrainingJob`
  filtered by `run_mode`, same response shape as before.

- [ ] **Step 1: Swap the import**

Replace:
```python
from backend.database.models import (
    AdvancedTuningJob,
    BasicTrainingJob,
    DataSource,
    Deployment,
)
```
With:
```python
from backend.database.models import (
    DataSource,
    Deployment,
    TrainingJob,
)
```

- [ ] **Step 2: Scope the two counts by `run_mode`**

Replace:
```python
    training_count = await session.scalar(select(func.count(BasicTrainingJob.id)))
    tuning_count = await session.scalar(select(func.count(AdvancedTuningJob.id)))
```
With:
```python
    training_count = await session.scalar(
        select(func.count(TrainingJob.id)).where(TrainingJob.run_mode == "fixed")
    )
    tuning_count = await session.scalar(
        select(func.count(TrainingJob.id)).where(TrainingJob.run_mode == "tuned")
    )
```

- [ ] **Step 3: Sanity-check syntax and remaining references**

Run: `cd /Users/BH7043/Skyulf && python -m py_compile backend/ml_pipeline/_internal/_routers/meta.py && grep -n "BasicTrainingJob\|AdvancedTuningJob" backend/ml_pipeline/_internal/_routers/meta.py`
Expected: `py_compile` succeeds silently; `grep` prints no matches.

- [ ] **Step 4: Commit**

```bash
cd /Users/BH7043/Skyulf
git add backend/ml_pipeline/_internal/_routers/meta.py
git commit -m "refactor(meta-router): compute /stats job counts from unified TrainingJob

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 9: `model_registry/service.py` — unified version counting + registry listing

**Files:**
- Modify: `backend/ml_pipeline/model_registry/service.py`

**Interfaces:**
- Consumes: `TrainingJob` (Task 1).
- Produces: `ModelRegistryService.get_next_version` (unchanged signature/behavior, now reads the
  max from one table filtered by `run_mode`), `list_models`, `get_model_versions` (unchanged
  signatures, now built from one query each instead of two).

- [ ] **Step 1: Swap the import**

Replace:
```python
from backend.database.models import (
    AdvancedTuningJob,
    BasicTrainingJob,
    DataSource,
    Deployment,
    ModelVersionCounter,
)
```
With:
```python
from backend.database.models import (
    DataSource,
    Deployment,
    ModelVersionCounter,
    TrainingJob,
)
```

- [ ] **Step 2: Update `_compute_seed_version` to read both run_modes from one table**

Replace:
```python
        stmt_train = select(func.max(BasicTrainingJob.version)).where(
            BasicTrainingJob.dataset_source_id == dataset_id,
            BasicTrainingJob.model_type == model_type,
        )
        result_train = await session.execute(stmt_train)
        max_train = result_train.scalar() or 0

        stmt_tune = select(func.max(AdvancedTuningJob.run_number)).where(
            AdvancedTuningJob.dataset_source_id == dataset_id,
            AdvancedTuningJob.model_type == model_type,
        )
        result_tune = await session.execute(stmt_tune)
        max_tune = result_tune.scalar() or 0

        return max(max_train, max_tune) + 1
```
With:
```python
        stmt = select(func.max(TrainingJob.version)).where(
            TrainingJob.dataset_source_id == dataset_id,
            TrainingJob.model_type == model_type,
        )
        result = await session.execute(stmt)
        max_version = result.scalar() or 0

        return max_version + 1
```

- [ ] **Step 3: Update the docstring reference to the old field names**

Replace:
```python
        Versions are shared between ``BasicTrainingJob.version`` and
        ``AdvancedTuningJob.run_number`` (same sequence), backed by a single
        ``ModelVersionCounter`` row per (dataset_id, model_type). The previous
```
With:
```python
        Both "fixed" and "tuned" run_modes of ``TrainingJob`` share the same
        ``version`` column and sequence, backed by a single
        ``ModelVersionCounter`` row per (dataset_id, model_type). The previous
```

- [ ] **Step 4: Update `get_registry_stats` to count by `run_mode`**

Replace:
```python
        # Count total versions (completed jobs)
        train_count = await session.scalar(
            select(func.count(BasicTrainingJob.id)).where(BasicTrainingJob.status == "completed")
        )
        tune_count = await session.scalar(
            select(func.count(AdvancedTuningJob.id)).where(AdvancedTuningJob.status == "completed")
        )
```
With:
```python
        # Count total versions (completed jobs)
        train_count = await session.scalar(
            select(func.count(TrainingJob.id)).where(
                TrainingJob.run_mode == "fixed", TrainingJob.status == "completed"
            )
        )
        tune_count = await session.scalar(
            select(func.count(TrainingJob.id)).where(
                TrainingJob.run_mode == "tuned", TrainingJob.status == "completed"
            )
        )
```

- [ ] **Step 5: Update `_fetch_completed_train_jobs`/`_fetch_completed_tune_jobs`**

Replace:
```python
    @staticmethod
    async def _fetch_completed_train_jobs(session: AsyncSession) -> list[BasicTrainingJob]:
        """Fetches completed training jobs ordered by newest first."""
        train_jobs_result = await session.execute(
            select(BasicTrainingJob)
            .where(BasicTrainingJob.status == "completed")
            .order_by(BasicTrainingJob.created_at.desc())
        )
        return list(train_jobs_result.scalars().all())

    @staticmethod
    async def _fetch_completed_tune_jobs(session: AsyncSession) -> list[AdvancedTuningJob]:
        """Fetches completed tuning jobs ordered by newest first."""
        tune_jobs_result = await session.execute(
            select(AdvancedTuningJob)
            .where(AdvancedTuningJob.status == "completed")
            .order_by(AdvancedTuningJob.created_at.desc())
        )
        return list(tune_jobs_result.scalars().all())
```
With:
```python
    @staticmethod
    async def _fetch_completed_train_jobs(session: AsyncSession) -> list[TrainingJob]:
        """Fetches completed fixed-mode training jobs ordered by newest first."""
        train_jobs_result = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.run_mode == "fixed", TrainingJob.status == "completed")
            .order_by(TrainingJob.created_at.desc())
        )
        return list(train_jobs_result.scalars().all())

    @staticmethod
    async def _fetch_completed_tune_jobs(session: AsyncSession) -> list[TrainingJob]:
        """Fetches completed tuned-mode tuning jobs ordered by newest first."""
        tune_jobs_result = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.run_mode == "tuned", TrainingJob.status == "completed")
            .order_by(TrainingJob.created_at.desc())
        )
        return list(tune_jobs_result.scalars().all())
```

- [ ] **Step 6: Update `_train_job_to_version`/`_tune_job_to_version`/`_group_versions_by_model_and_dataset` type hints**

Replace:
```python
    @staticmethod
    def _train_job_to_version(
        job: BasicTrainingJob, deployed_job_ids: dict[Any, Any]
    ) -> ModelVersion:
```
With:
```python
    @staticmethod
    def _train_job_to_version(
        job: TrainingJob, deployed_job_ids: dict[Any, Any]
    ) -> ModelVersion:
```

Replace:
```python
    @staticmethod
    def _tune_job_to_version(
        job: AdvancedTuningJob, deployed_job_ids: dict[Any, Any]
    ) -> ModelVersion:
        """Converts a completed tuning job into a ModelVersion entry, using run_number as version."""
        # For tuning jobs, we use run_number as version
        # And best_params as hyperparameters
        metrics = dict(cast(dict[str, Any] | None, job.metrics) or {})
        if job.best_score is not None:
            metrics["best_score"] = job.best_score

        return ModelVersion(
            job_id=job.id,
            pipeline_id=job.pipeline_id,
            node_id=job.node_id,
            model_type=cast(str, job.model_type or "unknown"),
            version=job.run_number,
            source="tuning",
```
With:
```python
    @staticmethod
    def _tune_job_to_version(
        job: TrainingJob, deployed_job_ids: dict[Any, Any]
    ) -> ModelVersion:
        """Converts a completed tuning job into a ModelVersion entry, using version as the version."""
        # For tuning jobs, best_params is used as hyperparameters
        metrics = dict(cast(dict[str, Any] | None, job.metrics) or {})
        if job.best_score is not None:
            metrics["best_score"] = job.best_score

        return ModelVersion(
            job_id=job.id,
            pipeline_id=job.pipeline_id,
            node_id=job.node_id,
            model_type=cast(str, job.model_type or "unknown"),
            version=job.version,
            source="tuning",
```

Replace:
```python
    @staticmethod
    def _group_versions_by_model_and_dataset(
        train_jobs: list[BasicTrainingJob],
        tune_jobs: list[AdvancedTuningJob],
        deployed_job_ids: dict[Any, Any],
    ) -> dict[tuple[str, str], list[ModelVersion]]:
```
With:
```python
    @staticmethod
    def _group_versions_by_model_and_dataset(
        train_jobs: list[TrainingJob],
        tune_jobs: list[TrainingJob],
        deployed_job_ids: dict[Any, Any],
    ) -> dict[tuple[str, str], list[ModelVersion]]:
```

- [ ] **Step 7: Update `get_model_versions` to filter one table by `run_mode` twice**

Replace:
```python
        # Training Jobs
        train_jobs = await session.execute(
            select(BasicTrainingJob)
            .where(BasicTrainingJob.status == "completed")
            .where(BasicTrainingJob.model_type == model_type)
            .order_by(BasicTrainingJob.created_at.desc())
        )
```
With:
```python
        # Training Jobs (fixed mode)
        train_jobs = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.run_mode == "fixed")
            .where(TrainingJob.status == "completed")
            .where(TrainingJob.model_type == model_type)
            .order_by(TrainingJob.created_at.desc())
        )
```

Replace:
```python
        # Tuning Jobs
        tune_jobs = await session.execute(
            select(AdvancedTuningJob)
            .where(AdvancedTuningJob.status == "completed")
            .where(AdvancedTuningJob.model_type == model_type)
            .order_by(AdvancedTuningJob.created_at.desc())
        )
```
With:
```python
        # Tuning Jobs (tuned mode)
        tune_jobs = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.run_mode == "tuned")
            .where(TrainingJob.status == "completed")
            .where(TrainingJob.model_type == model_type)
            .order_by(TrainingJob.created_at.desc())
        )
```

And, in the loop right below that builds `ModelVersion` from `tune_jobs`, replace:
```python
                    version=job.run_number,
```
With:
```python
                    version=job.version,
```

- [ ] **Step 8: Sanity-check syntax and remaining references**

Run: `cd /Users/BH7043/Skyulf && python -m py_compile backend/ml_pipeline/model_registry/service.py && grep -n "BasicTrainingJob\|AdvancedTuningJob\|run_number" backend/ml_pipeline/model_registry/service.py`
Expected: `py_compile` succeeds silently; `grep` prints no matches.

- [ ] **Step 9: Commit**

```bash
cd /Users/BH7043/Skyulf
git add backend/ml_pipeline/model_registry/service.py
git commit -m "refactor(model-registry): compute versions/registry from unified TrainingJob

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 10: `monitoring/router.py` — unified drift/description/slow-node scans

**Files:**
- Modify: `backend/monitoring/router.py`

**Interfaces:**
- Consumes: `TrainingJob` (Task 1).
- Produces: `_fetch_drift_job_rows`, `update_job_description`, `_load_feature_importances`,
  `_scan_slow_node_jobs` — same signatures/behavior, now single-table queries instead of looping
  over two model classes (since both loop iterations would otherwise touch the exact same rows).

- [ ] **Step 1: Swap the import**

Replace:
```python
from backend.database.models import (
    AdvancedTuningJob,
    BasicTrainingJob,
    DriftCheckResult,
    ErrorEvent,
```
With:
```python
from backend.database.models import (
    DriftCheckResult,
    ErrorEvent,
    TrainingJob,
```
(Keep whatever other names already followed `ErrorEvent,` in that import block — only the first
two lines change.)

- [ ] **Step 2: Collapse `_fetch_drift_job_rows` to one query**

Replace:
```python
async def _fetch_drift_job_rows(
    db: AsyncSession, job_ids: list[str]
) -> dict[str, BasicTrainingJob | AdvancedTuningJob]:
    """Look up training/tuning job rows for the given ids across both job tables."""
    db_jobs: dict[str, BasicTrainingJob | AdvancedTuningJob] = {}
    try:
        for model_cls in (BasicTrainingJob, AdvancedTuningJob):
            stmt = select(model_cls).where(model_cls.id.in_(job_ids))
            result = await db.execute(stmt)
            for row in result.scalars().all():
                db_jobs[str(row.id)] = row
    except Exception:
        logger.warning("Could not enrich drift jobs from DB", exc_info=True)
    return db_jobs
```
With:
```python
async def _fetch_drift_job_rows(db: AsyncSession, job_ids: list[str]) -> dict[str, TrainingJob]:
    """Look up training/tuning job rows (either run_mode) for the given ids."""
    db_jobs: dict[str, TrainingJob] = {}
    try:
        stmt = select(TrainingJob).where(TrainingJob.id.in_(job_ids))
        result = await db.execute(stmt)
        for row in result.scalars().all():
            db_jobs[str(row.id)] = row
    except Exception:
        logger.warning("Could not enrich drift jobs from DB", exc_info=True)
    return db_jobs
```

- [ ] **Step 3: Update the `_extract_drift_target_column`/`_enrich_drift_job` type hints**

Replace:
```python
def _extract_drift_target_column(db_row: BasicTrainingJob | AdvancedTuningJob) -> str | None:
```
With:
```python
def _extract_drift_target_column(db_row: TrainingJob) -> str | None:
```

Replace:
```python
def _enrich_drift_job(job: DriftJobOption, db_row: BasicTrainingJob | AdvancedTuningJob) -> None:
```
With:
```python
def _enrich_drift_job(job: DriftJobOption, db_row: TrainingJob) -> None:
```

- [ ] **Step 4: Collapse `update_job_description`'s loop to one query**

Replace:
```python
    for model_cls in (BasicTrainingJob, AdvancedTuningJob):
        stmt = select(model_cls).where(model_cls.id == job_id)
        result = await db.execute(stmt)
        row = result.scalar_one_or_none()
        if row:
            meta_raw: dict[str, Any] = cast(dict[str, Any], row.job_metadata or {})
            if not isinstance(meta_raw, dict):
                meta_raw = {}
            meta_raw["description"] = body.description
            row.job_metadata = cast(Any, meta_raw)
            await db.commit()
            return {"status": "ok"}

    raise HTTPException(status_code=404, detail="Job not found")
```
With:
```python
    stmt = select(TrainingJob).where(TrainingJob.id == job_id)
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()
    if row:
        meta_raw: dict[str, Any] = cast(dict[str, Any], row.job_metadata or {})
        if not isinstance(meta_raw, dict):
            meta_raw = {}
        meta_raw["description"] = body.description
        row.job_metadata = cast(Any, meta_raw)
        await db.commit()
        return {"status": "ok"}

    raise HTTPException(status_code=404, detail="Job not found")
```

- [ ] **Step 5: Collapse `_load_feature_importances`'s loop to one query**

Replace:
```python
    try:
        for model_cls in (BasicTrainingJob, AdvancedTuningJob):
            stmt = select(model_cls).where(model_cls.id == job_id)
            result = await db.execute(stmt)
            row = result.scalar_one_or_none()
            if row:
                job_metrics: dict[str, Any] = cast(dict[str, Any], row.metrics or {})
                if "feature_importances" in job_metrics:
                    feature_importances = job_metrics["feature_importances"]
                break
    except Exception:
        logger.warning("Could not load feature importances for job %s", job_id)
    return feature_importances
```
With:
```python
    try:
        stmt = select(TrainingJob).where(TrainingJob.id == job_id)
        result = await db.execute(stmt)
        row = result.scalar_one_or_none()
        if row:
            job_metrics: dict[str, Any] = cast(dict[str, Any], row.metrics or {})
            if "feature_importances" in job_metrics:
                feature_importances = job_metrics["feature_importances"]
    except Exception:
        logger.warning("Could not load feature importances for job %s", job_id)
    return feature_importances
```

- [ ] **Step 6: Collapse `_scan_slow_node_jobs`'s loop to one query**

Replace:
```python
    # Scan both job tables — same metrics shape, different rows.
    for model in (BasicTrainingJob, AdvancedTuningJob):
        stmt = select(model).where(
            model.status == "completed",
            model.finished_at.isnot(None),
            model.finished_at >= cutoff,
        )
        result = await db.execute(stmt)
        for job in result.scalars().all():
            jobs_scanned += 1
            metrics = job.metrics or {}
            timings = metrics.get("node_timings") if isinstance(metrics, dict) else None
            if not isinstance(timings, list):
                continue
            for entry in timings:
                if _accumulate_node_timing(entry, by_step, sample_node):
                    runs_seen += 1

    return by_step, sample_node, jobs_scanned, runs_seen
```
With:
```python
    # Scan the unified table — both run_modes share the same metrics shape.
    stmt = select(TrainingJob).where(
        TrainingJob.status == "completed",
        TrainingJob.finished_at.isnot(None),
        TrainingJob.finished_at >= cutoff,
    )
    result = await db.execute(stmt)
    for job in result.scalars().all():
        jobs_scanned += 1
        metrics = job.metrics or {}
        timings = metrics.get("node_timings") if isinstance(metrics, dict) else None
        if not isinstance(timings, list):
            continue
        for entry in timings:
            if _accumulate_node_timing(entry, by_step, sample_node):
                runs_seen += 1

    return by_step, sample_node, jobs_scanned, runs_seen
```

- [ ] **Step 7: Sanity-check syntax and remaining references**

Run: `cd /Users/BH7043/Skyulf && python -m py_compile backend/monitoring/router.py && grep -n "BasicTrainingJob\|AdvancedTuningJob" backend/monitoring/router.py`
Expected: `py_compile` succeeds silently; `grep` prints no matches.

- [ ] **Step 8: Commit**

```bash
cd /Users/BH7043/Skyulf
git add backend/monitoring/router.py
git commit -m "refactor(monitoring-router): single-table TrainingJob scans

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 11: Frontend comment reference update

**Files:**
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/jobMeta.ts:165`

**Interfaces:**
- No code/behavior change — comment-only.

- [ ] **Step 1: Update the stale model-name reference in the comment**

Find the comment mentioning:
```
 * backend's `AdvancedTuningJob` model (`nullable=False` there), so it's
```
Replace with:
```
 * backend's `TrainingJob` model when `run_mode === "tuned"` (`nullable=False`
 * there), so it's
```
(Adjust surrounding line wrapping as needed so the comment still reads naturally — keep it to two
lines.)

- [ ] **Step 2: Verify no functional TS reference exists**

Run: `cd /Users/BH7043/Skyulf/frontend/ml-canvas && grep -n "BasicTrainingJob\|AdvancedTuningJob" src/components/pages/ExperimentsPage/utils/jobMeta.ts`
Expected: no matches (comment updated, no code ever referenced these names directly).

- [ ] **Step 3: Commit**

```bash
cd /Users/BH7043/Skyulf
git add frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/jobMeta.ts
git commit -m "docs(frontend): update jobMeta.ts comment for unified TrainingJob model

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 12: Update all backend tests referencing the old models

**Files:**
- Modify: `tests/test_versioning_logic.py`
- Modify: `tests/test_monitoring_router_extra.py`
- Modify: `tests/test_pipeline_task.py`
- Modify: `tests/test_ml_pipeline_backend_fixes.py`
- Modify: `tests/test_backend_strategies.py`
- Modify: `tests/test_main_stale_job_cutoff.py`
- Modify: `tests/test_pagination_and_thresholds_settings.py`
- Modify: `tests/test_jobs_settings_driven_constants.py`
- Modify: `tests/test_model_version_race.py`

**Interfaces:**
- Consumes: `TrainingJob` (Task 1) and all the refactored modules from Tasks 2-10.
- Produces: all nine test files pass against the unified model, asserting the same behaviors as
  before (idempotency-window matching, version-race safety, strategy dispatch, stale-job cutoff,
  pagination/threshold settings, monitoring-router drift enrichment).

- [ ] **Step 1: `tests/test_versioning_logic.py`**

Replace:
```python
from backend.database.models import AdvancedTuningJob, BasicTrainingJob, ModelVersionCounter
```
With:
```python
from backend.database.models import ModelVersionCounter, TrainingJob
```

Replace:
```python
            delete(BasicTrainingJob).where(BasicTrainingJob.pipeline_id == pipeline_id)
```
With:
```python
            delete(TrainingJob).where(TrainingJob.pipeline_id == pipeline_id)
```
(the corresponding `AdvancedTuningJob` delete line right below it becomes redundant since it's
now the same table/filter — delete that second `delete(AdvancedTuningJob)...` line entirely,
i.e. keep only one `delete(TrainingJob).where(TrainingJob.pipeline_id == pipeline_id)` cleanup
statement.)

Replace each of:
```python
        job1 = await session.get(BasicTrainingJob, job1_id)
```
```python
        job2 = await session.get(AdvancedTuningJob, job2_id)
```
```python
        job3 = await session.get(BasicTrainingJob, job3_id)
```
```python
        job4 = await session.get(AdvancedTuningJob, job4_id)
```
With:
```python
        job1 = await session.get(TrainingJob, job1_id)
```
```python
        job2 = await session.get(TrainingJob, job2_id)
```
```python
        job3 = await session.get(TrainingJob, job3_id)
```
```python
        job4 = await session.get(TrainingJob, job4_id)
```
(`session.get(Model, id)` works identically for a shared table — only the model name changes.)

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_versioning_logic.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 2: `tests/test_monitoring_router_extra.py`**

This file only references the two old model names in docstrings/comments (no imports or actual
usage per the grep scan). Update the two comments:

Replace:
```python
    """Build a lightweight mock standing in for a BasicTrainingJob/AdvancedTuningJob row."""
```
With:
```python
    """Build a lightweight mock standing in for a TrainingJob row."""
```

Replace:
```python
    """Rows from both BasicTrainingJob and AdvancedTuningJob are merged by id."""
```
With:
```python
    """Rows from both run_modes ("fixed" and "tuned") are merged by id."""
```

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_monitoring_router_extra.py -v`
Expected: all tests in the file PASS (no behavior touched, only comments).

- [ ] **Step 3: `tests/test_pipeline_task.py`**

Replace:
```python
from backend.database.models import AdvancedTuningJob, BasicTrainingJob
```
With:
```python
from backend.database.models import TrainingJob
```

Replace:
```python
    job = BasicTrainingJob(id=MOCK_JOB_ID, status="queued")
```
(the one around line 52) with:
```python
    job = TrainingJob(id=MOCK_JOB_ID, status="queued", run_mode="fixed")
```

Replace:
```python
    job = AdvancedTuningJob(id=MOCK_JOB_ID, status="queued")

    # First query (BasicTrainingJob) returns None
    # Second query (AdvancedTuningJob) returns job
```
With:
```python
    job = TrainingJob(id=MOCK_JOB_ID, status="queued", run_mode="tuned")

    # Single query against the unified TrainingJob table returns job
```

Then find the mock-branching logic right after (around the old lines 100-102):
```python
        if args[0] == BasicTrainingJob:
            return None
        elif args[0] == AdvancedTuningJob:
            return job
```
Since there's now only one model class being queried (no more branching by model class), replace
this whole conditional with a direct return of `job` from the mock (adjust to match the specific
mock/fixture structure in the file — the intent is: the single `TrainingJob` query returns `job`
unconditionally instead of branching on which of two model classes was passed). Read the
surrounding ~15 lines of the test to confirm the exact mock signature before editing, since it
depends on whatever query-execution mock helper this file defines earlier.

Replace the two remaining occurrences of:
```python
    job = BasicTrainingJob(id=MOCK_JOB_ID, status="queued")
```
(around lines 130 and 156) with:
```python
    job = TrainingJob(id=MOCK_JOB_ID, status="queued", run_mode="fixed")
```

Replace:
```python
        if model_name == "BasicTrainingJob":
```
With:
```python
        if model_name == "TrainingJob":
```

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_pipeline_task.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 4: `tests/test_ml_pipeline_backend_fixes.py`**

Replace:
```python
from backend.database.models import BasicTrainingJob
```
With:
```python
from backend.database.models import TrainingJob
```

Replace both occurrences of:
```python
    job = BasicTrainingJob(id=job_id, status="running")
```
(around lines 286 and 312) with:
```python
    job = TrainingJob(id=job_id, status="running", run_mode="fixed")
```

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_ml_pipeline_backend_fixes.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 5: `tests/test_backend_strategies.py`**

Replace:
```python
from backend.database.models import AdvancedTuningJob, BasicTrainingJob
```
With:
```python
from backend.database.models import TrainingJob
```

Replace:
```python
        self.job = MagicMock(spec=BasicTrainingJob)
```
With:
```python
        self.job = MagicMock(spec=TrainingJob)
        self.job.run_mode = "fixed"
```

Replace:
```python
        self.assertEqual(self.strategy.get_job_model(), BasicTrainingJob)
```
With:
```python
        self.assertEqual(self.strategy.get_job_model(), TrainingJob)
```

Replace:
```python
        self.job = MagicMock(spec=AdvancedTuningJob)
```
With:
```python
        self.job = MagicMock(spec=TrainingJob)
        self.job.run_mode = "tuned"
```

Replace:
```python
        self.assertEqual(self.strategy.get_job_model(), AdvancedTuningJob)
```
With:
```python
        self.assertEqual(self.strategy.get_job_model(), TrainingJob)
```

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_backend_strategies.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 6: `tests/test_main_stale_job_cutoff.py`**

Replace:
```python
from backend.database.models import AdvancedTuningJob, Base, BasicTrainingJob
```
With:
```python
from backend.database.models import Base, TrainingJob
```

Replace:
```python
        tables=[BasicTrainingJob.__table__, AdvancedTuningJob.__table__],
```
With:
```python
        tables=[TrainingJob.__table__],
```

Replace:
```python
    """Insert a minimal 'running' BasicTrainingJob row started at the given time."""
```
With:
```python
    """Insert a minimal 'running' TrainingJob row (run_mode='fixed') started at the given time."""
```

Replace:
```python
            BasicTrainingJob.__table__.insert().values(
```
With:
```python
            TrainingJob.__table__.insert().values(
```
(Also check the `.values(...)` call right below this line — since `run_mode` is now
`nullable=False`, add `run_mode="fixed"` to that `.values(...)` call alongside whatever fields
it already sets, e.g. `id=...`, `status="running"`, etc.)

Replace both occurrences of:
```python
            BasicTrainingJob.__table__.select().where(BasicTrainingJob.__table__.c.id == "job-90m")
```
With:
```python
            TrainingJob.__table__.select().where(TrainingJob.__table__.c.id == "job-90m")
```

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_main_stale_job_cutoff.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 7: `tests/test_pagination_and_thresholds_settings.py`**

Replace:
```python
    AdvancedTuningJob,
```
and
```python
    BasicTrainingJob,
```
(two separate lines inside the existing multi-line import from `backend.database.models`) with a
single:
```python
    TrainingJob,
```

Replace:
```python
def _make_training_job(idx: int) -> BasicTrainingJob:
    return BasicTrainingJob(
```
With:
```python
def _make_training_job(idx: int) -> TrainingJob:
    return TrainingJob(
```
(Add `run_mode="fixed",` into that constructor call's keyword arguments — check the existing
argument list at that call site and insert it alongside the other fields already being set, e.g.
right after `status=...` or `id=...`.)

Replace:
```python
def _make_tuning_job(idx: int, model_type: str = "classifier") -> AdvancedTuningJob:
    return AdvancedTuningJob(
```
With:
```python
def _make_tuning_job(idx: int, model_type: str = "classifier") -> TrainingJob:
    return TrainingJob(
```
(Similarly add `run_mode="tuned",` into that constructor call's keyword arguments, and if the
original constructor set `run_number=...`, rename that keyword to `version=...`.)

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_pagination_and_thresholds_settings.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 8: `tests/test_jobs_settings_driven_constants.py`**

Replace:
```python
from backend.database.models import Base, BasicTrainingJob
```
With:
```python
from backend.database.models import Base, TrainingJob
```

Replace:
```python
    """Insert a minimal running BasicTrainingJob row with a specific created_at."""
```
With:
```python
    """Insert a minimal running TrainingJob row (run_mode='fixed') with a specific created_at."""
```

Find the constructor call right after:
```python
    job = BasicTrainingJob(
```
Replace with:
```python
    job = TrainingJob(
```
(Add `run_mode="fixed",` into that constructor's keyword arguments alongside the other fields
already set there.)

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_jobs_settings_driven_constants.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 9: `tests/test_model_version_race.py`**

Replace:
```python
from backend.database.models import AdvancedTuningJob, BasicTrainingJob, ModelVersionCounter
```
With:
```python
from backend.database.models import ModelVersionCounter, TrainingJob
```

Replace:
```python
            delete(BasicTrainingJob).where(BasicTrainingJob.dataset_source_id == dataset_id)
```
With:
```python
            delete(TrainingJob).where(TrainingJob.dataset_source_id == dataset_id)
```
(delete the corresponding `delete(AdvancedTuningJob)...` cleanup line right below it — same
table/filter now, so it's redundant, matching the pattern from Step 1.)

Replace:
```python
    the max of pre-existing BasicTrainingJob.version / AdvancedTuningJob.run_number
```
With:
```python
    the max of pre-existing TrainingJob.version rows (either run_mode)
```

Find the constructor call right after:
```python
            BasicTrainingJob(
```
Replace with:
```python
            TrainingJob(
```
(Add `run_mode="fixed",` into that constructor's keyword arguments.)

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/test_model_version_race.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 10: Commit all test updates**

```bash
cd /Users/BH7043/Skyulf
git add tests/test_versioning_logic.py tests/test_monitoring_router_extra.py \
        tests/test_pipeline_task.py tests/test_ml_pipeline_backend_fixes.py \
        tests/test_backend_strategies.py tests/test_main_stale_job_cutoff.py \
        tests/test_pagination_and_thresholds_settings.py \
        tests/test_jobs_settings_driven_constants.py tests/test_model_version_race.py
git commit -m "test: update job-model tests for unified TrainingJob(run_mode)

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 13: Drop/recreate dev DB and run full verification

**Files:** none (verification-only task)

**Interfaces:** N/A — final gate confirming the whole unification works end-to-end.

- [ ] **Step 1: Confirm no remaining references to the old model names anywhere in the repo**

Run: `cd /Users/BH7043/Skyulf && grep -rn "BasicTrainingJob\|AdvancedTuningJob" backend/ tests/ frontend/ml-canvas/src/ --include="*.py" --include="*.ts" --include="*.tsx"`
Expected: no output (empty). If anything remains, fix it before continuing.

- [ ] **Step 2: Drop and recreate the local dev database**

Locate the dev DB file/connection string (check `backend/config` / `.env` for the configured
SQLite/Postgres DB). For the common SQLite dev setup:

```bash
cd /Users/BH7043/Skyulf
find . -maxdepth 2 -name "*.db" -not -path "./.venv/*"
```

Delete the found dev DB file(s) (e.g. `rm skyulf.db` — substitute the actual filename found), then
let the app recreate the schema on next startup via its existing `Base.metadata.create_all` /
startup migration path (check `backend/database/engine.py` for how tables are created on boot —
do not write a new creation script, reuse whatever the app already does on startup).

- [ ] **Step 3: Run ruff on all touched files**

Run: `cd /Users/BH7043/Skyulf && ruff check backend/database/models.py backend/ml_pipeline/_execution/job_manager_base.py backend/ml_pipeline/_execution/basic_training_manager.py backend/ml_pipeline/_execution/advanced_tuning_manager.py backend/ml_pipeline/_execution/strategies.py backend/ml_pipeline/_execution/jobs.py backend/ml_pipeline/_services/job_service.py backend/ml_pipeline/_internal/_routers/meta.py backend/ml_pipeline/model_registry/service.py backend/monitoring/router.py`
Expected: no lint errors (or only pre-existing ones unrelated to this change).

- [ ] **Step 4: Run the full backend test suite**

Run: `cd /Users/BH7043/Skyulf && python -m pytest tests/ -x -q`
Expected: all tests PASS. If anything fails, use systematic-debugging to find and fix the root
cause before proceeding — do not skip or weaken assertions to force a pass.

- [ ] **Step 5: Start the backend and smoke-test the `/stats` and job-listing endpoints**

Run: `cd /Users/BH7043/Skyulf && (uvicorn backend.main:app --port 8010 &) && sleep 3 && curl -s http://localhost:8010/api/stats && echo && curl -s http://localhost:8010/api/jobs?limit=5`
Expected: both return valid JSON with no 500 errors (empty job list is fine on a freshly recreated
DB). Stop the server afterward: find its PID via `lsof -ti:8010` and `kill <pid>`.

- [ ] **Step 6: Final commit (if Step 5 required any fixes)**

If Step 4 or 5 required any code fixes, commit them:

```bash
cd /Users/BH7043/Skyulf
git add -A
git commit -m "fix: address test/runtime issues found during TrainingJob unification verification

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

If no fixes were needed, skip this step — Task 13 is verification-only.

---

## Post-plan note

Update `basic_training_advanced_tuning_unification_plan.md`'s Phase 4 section to mark it done
(schema-unification-only variant, no migration) once all tasks above are complete and verified.
