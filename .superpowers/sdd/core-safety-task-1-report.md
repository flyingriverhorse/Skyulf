Status: REPAIRED_PENDING_REVIEW
Base commit: 748929e5

Repair summary:
- Fixed the Task 1 regression where pipeline TargetEncoder training always called sklearn
  `fit_transform` with `cv=5`, which crashed on undersized training splits.
- Kept normal eligible training behavior unchanged: sklearn TargetEncoder still uses
  deterministic five-fold cross-fitting with `cv=5`, `shuffle=True`, and `random_state=42`.
- Preserved direct Calculator/Applier semantics: explicit `fit(...).transform(...)` still
  behaves exactly as before outside the pipeline training hook.
- Added direct regression coverage for the new dispatcher helper and for the small-split CV
  policy that now protects pipeline training rows.

Explicit policy:
- Pipeline training rows use the largest leakage-safe deterministic fold count instead of
  blindly forcing five folds.
- Classification targets (`binary` / `multiclass`, including `auto` when sklearn infers them):
  use `cv=min(5, smallest_target_class_count)`.
- Regression targets (including `auto` when sklearn infers continuous targets):
  use `cv=min(5, n_training_rows)`.
- A one-row training split raises a clear `ValueError` because no leakage-safe cross-fit
  representation exists.
- A classification training split where any target class appears only once raises a clear
  `ValueError` instead of surfacing sklearn's accidental `n_splits` / class-membership error.

Files changed:
- `skyulf-core/skyulf/preprocessing/encoding/target.py`
- `skyulf-core/tests/test_encoding_target.py`
- `skyulf-core/tests/test_preprocessing_dispatcher.py`
- `docs/reference/preprocessing_nodes.md`
- `.superpowers/sdd/core-safety-task-1-report.md`
- `.superpowers/sdd/progress.md`

Validation:
- `source .venv/bin/activate && pytest skyulf-core/tests/test_encoding_target.py::test_feature_engineer_cross_fits_small_target_encoder_training_rows skyulf-core/tests/test_encoding_target.py::test_resolve_target_encoder_training_cv_uses_smallest_class_count skyulf-core/tests/test_encoding_target.py::test_resolve_target_encoder_training_cv_rejects_single_row skyulf-core/tests/test_encoding_target.py::test_resolve_target_encoder_training_cv_rejects_singleton_class skyulf-core/tests/test_preprocessing_dispatcher.py::test_fit_transform_train_dual_engine_dispatches_to_pandas_path skyulf-core/tests/test_preprocessing_dispatcher.py::test_fit_transform_train_dual_engine_dispatches_to_polars_path -q` -> `7 passed`
- `source .venv/bin/activate && pytest skyulf-core/tests/test_encoding_target.py skyulf-core/tests/test_preprocessing_dispatcher.py -q` -> `48 passed`
- `source .venv/bin/activate && ruff check .` -> `All checks passed!`
- `source .venv/bin/activate && ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py` -> `569 files already formatted`
- `source .venv/bin/activate && ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py` -> `All checks passed!`

Notes:
- This repair is intentionally limited to Task 1's TargetEncoder training-path regression.
- Task 1 is not marked accepted here; the independent review should be re-run on this repair.
