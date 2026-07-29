# Design: Unify `basic_training_jobs` / `advanced_tuning_jobs` into `training_jobs`

Status: Approved
Context: Phase 4 of `basic_training_advanced_tuning_unification_plan.md`, scoped down to
schema-only unification (no migration/backfill — DB will be dropped and recreated).

## Goal

Replace the two parallel job tables (`BasicTrainingJob`, `AdvancedTuningJob`) with a single
`TrainingJob` model/table (`training_jobs`), matching the `run_mode: 'fixed' | 'tuned'`
convention already used by the pipeline engine (`_resolve_run_mode`). No data migration is
needed or written — the existing dev DB will simply be dropped and recreated against the new
schema.

## Schema

`TrainingJob(MLJob)` — table `training_jobs`:

- All existing `MLJob` base columns unchanged (`id`, `pipeline_id`, `node_id`,
  `dataset_source_id`, `user_id`, `status`, `model_type`, `metadata`, `metrics`, `graph`,
  `artifact_uri`, `error_message`, `progress`, `current_step`, `logs`, `started_at`,
  `finished_at`, `promoted_at`, timestamps).
- `run_mode: str` (new) — `"fixed"` or `"tuned"`, discriminator column.
- `version: int` — unified sequence. Replaces both `BasicTrainingJob.version` and
  `AdvancedTuningJob.run_number` (already documented in `model_registry/service.py` as sharing
  one sequence).
- `hyperparameters: JSON | None` — fixed-mode hyperparameters (also usable as the tuned-mode
  baseline where applicable).
- Tuning-only columns, all nullable, populated only when `run_mode == "tuned"`:
  `search_strategy`, `search_space`, `baseline_hyperparameters`, `n_iterations`, `scoring`,
  `random_state`, `cross_validation`, `results`, `best_params`, `best_score`.
- `to_dict()` merges the fields from both old `to_dict()` implementations.

`Deployment.job_id` gets a real `ForeignKey("training_jobs.id")` constraint (previously a loose
string column with no FK).

## Code changes (mechanical, no behavior change)

Replace the "try `BasicTrainingJob`, then `AdvancedTuningJob`" dual-lookup pattern with a single
`TrainingJob` query (filtered by `run_mode` where the old code was filtered by table) in:

- `backend/database/models.py` — the model itself
- `backend/ml_pipeline/_services/job_service.py`
- `backend/ml_pipeline/_execution/jobs.py`
- `backend/ml_pipeline/_execution/basic_training_manager.py`
- `backend/ml_pipeline/_execution/advanced_tuning_manager.py`
- `backend/ml_pipeline/_execution/strategies.py`
- `backend/ml_pipeline/_internal/_routers/meta.py`
- `backend/ml_pipeline/model_registry/service.py`
- `backend/monitoring/router.py`
- `frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/jobMeta.ts` (if it references
  job/table type strings tied to the old split)
- Test files referencing `BasicTrainingJob`/`AdvancedTuningJob`: `tests/test_versioning_logic.py`,
  `tests/test_monitoring_router_extra.py`, `tests/test_pipeline_task.py`,
  `tests/test_ml_pipeline_backend_fixes.py`, `tests/test_backend_strategies.py`,
  `tests/test_main_stale_job_cutoff.py`, `tests/test_pagination_and_thresholds_settings.py`,
  `tests/test_jobs_settings_driven_constants.py`, `tests/test_model_version_race.py`

## Explicitly out of scope

- No Alembic migration or backfill script — DB is dropped/recreated instead.
- No `_legacy` table retention.
- `basic_training_manager.py` and `advanced_tuning_manager.py` remain **separate classes**
  (both now backed by `TrainingJob`, filtered by `run_mode`) — merging their logic into one
  manager class is a Phase 5-style code-consolidation concern, not part of this DB unification.

## Validation

- `pytest` (targeted: the test files listed above, then full backend suite)
- `ruff` / existing lint
- Frontend: `tsc --noEmit`, `vitest run` if `jobMeta.ts` changes
