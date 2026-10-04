# Model-set output selection implementation plan

Latest status: live acceptance completed; see
[report112](112-clean-databricks-acceptance.md). Pending statements below describe
the earlier implementation checkpoints.

Approved design: model rules and combined rules live in features/scoring.py.
Publishing offers all columns, combined outputs only, or separate named views
over one physical Delta table. Calculation errors propagate before publication.

## Files and execution

- [x] Add `model_set_output.py` for validated publication policies, column
  selection and owned view provisioning. Test with real fitted set artifacts;
  reject combined-only without rules, unknown branches, duplicate/unsafe names,
  view/table collisions and unrelated existing views before publishing data.
- [x] Integrate publication into `model_set_batch.py`: validate before Spark,
  provision exact projected schema, create/verify views before the sole Delta
  write, preserve no-op, history, provenance and failure handling.
- [x] Update `model_set_project.py` and `branch_notebook.py`: bind destinations
  to the deployment, capture combined rules from features during training only.
  Preserve legacy composition packages and saved-artifact replay.
- [x] Consolidate template rules in `features/scoring.py`; remove the template
  composition directory. Keep `build_scoring` contract via a package export.
  Add conditional output-mode/table/view prompts and editable per-model view
  overrides in `modeling/model_set.py`. Verify actual CLI generation.
- [x] Update user-facing documentation, queue evidence and changelog. Run
  affected suites, full Ruff/format/Ty and Lizard CCN10; report skipped cloud or
  Delta tests explicitly. Do not commit or push without a new request.

## Verification contract

Write failing tests before production changes. Exercise all/combined-only/view
modes with component and business values; keep keys, status and set identity.
All views project the same committed table, never independent data copies.
DDL setup is not transactional across views: a failed setup can leave empty or
old-snapshot views, but cannot publish a new subset of calculated predictions.
Never overwrite an unrelated catalog object. Schema changes need a new target.
Reusing a view with a changed source/projection fails explicitly.

## Verification results (2026-09-29)

- Red tests reproduced missing publication selection, missing shared combined
  rule capture, and missing conditional initializer questions before changes.
- Local contract suite: 165 passed, 1 Windows symlink skip. Subsequent expanded
  shared-scoring suite: 3 passed, including a real two-model saved-package replay
  after editable source changes and positive/negative profit arithmetic.
- Actual Databricks CLI generation: 94 passed, including all three publication
  modes with custom table/view names and existing template regressions.
- Output/batch/import-order plus Delta suite: 28 passed, 8 skipped because local
  Spark/Delta/Java are unavailable. The eight include new combined-only physical
  schema and common-view publication/failure tests; they have NOT run live yet.
- Full Ruff, format, Ty and backend/Core CCN10 passed. `git diff --check` passed.
- Fresh Core wheel built with uv, SHA256
  `1aa48350cdeaa88e11194791f71fa69c009e236db2f72733d7494d14136df8ca`.
  The generated separate-views Bundle passed `bundle validate --strict --target dev`
  against profile `skyulf`, with no warnings. No upload, deployment or job run.
- Evidence/temp paths: `rehearsals/sm36c_outputs/`; do not commit generated wheels,
  uv cache, generated projects or test output. No commit/push performed.

The preceding SM-36c cloud runs in report109 predate this follow-up. Do not treat
their success as proof of these newly added view/storage modes. Live acceptance
is the remaining verification step before claiming Databricks-tested delivery.

## Naming and explanation follow-up

Added a conditional `model_set_name` initializer question. Blank preserves the
registered `<project_name>_set`; a custom simple name replaces that default while
retaining metadata-schema ownership and deployment suffix. Existing physical
table, per-model view prefix and combined-view name questions keep their defaults
and custom overrides. The generated `model_set.py` now documents all three modes
with a revenue/cost/profit example, metadata retention, and insert-vs-update/delete
behavior. Full rebuild is explicitly described as a set-change policy, not an
unconditional source-correction synchronization switch.

Verification: 102 template/layout/batch/output tests and 6 actual CLI generation
tests passed. Generated custom-name Bundle strict validation passed without
warnings. Full Ruff/format/Ty/CCN10 passed. Two pre-existing prompt tests omitted
training_layout (confirmed against HEAD); their inputs now include actual
initializer defaults. No live cloud job or commit was added by this follow-up.
