# SM-33F: understandable Bundle initialization and Python preprocessing

Date: 2026-09-26. Status: DONE for local acceptance; included in SM-33F/G delivery.
SM-33D/E baseline: `516b3f86`, DCO sign-off and hooks passed; no push.
This follow-up implements the user's setup feedback before SM-34.

## Delivered behavior

- Training input, prediction input and prediction output have distinct explanations.
  `prediction_table_name` chooses the output basename; blank keeps the project
  default. The configured output schema and target suffix still apply.
- Initializer key/feature fields accept comma-separated names. Generated config
  preserves ordered JSON arrays. Invalid identifiers and empty items are rejected.
- Classification/regression have separate model/metric menus. Tests compare the
  menus with Core metadata and metric contracts. Optional packages and required
  estimator settings still apply; listing does not guarantee default fit success.
- Snapshots, date formats, availability cutoffs, date-only handling, samples, CV
  and timezones have concrete English explanations.
- `scheduled` replaces the paused/monthly initializer choices. Selecting it emits
  `pause_status: UNPAUSED`, including dev, so deployment enables the selected cron.
  Default `manual` adds no schedule. Six-month cron is independent of the data
  window. Runtime action `train_monthly` retains its name but imposes no second
  monthly timer. Initialization and validation alone do not create active jobs.

## Python preprocessing

Generated `src/preprocessing.py` defines `build_preprocessing()`, returning an
ordered list of normal Core nodes. It starts empty and includes commented
imputation/scaling examples plus a custom learned mean-centering example.
Keep the JSON preprocessing list empty; conflicting nonempty JSON is rejected.
Model parameters and workflow settings stay in config/workflow.json.

Users can define top-level Calculator/Applier classes in that same file and
include them with `custom_step`. The runtime calls existing Core fitting/CV.
Fit returns learned state; apply reuses it. User steps must preserve row count
and order and support the project's engines. CV refits them inside each fold.

The source is saved alongside manifest/pipeline.pkl and included through the
existing MLflow model package. Loading checks its checksum and installs a
source-specific module before unpickling. Existing artifacts without source
remain supported. Different source versions can coexist without replacing each
other's same-named classes. Scoring and operator actions ignore the current
editable file; changes affect the next training run.

This is trusted executable model content, like pickle, not a sandbox. The supported
project source is one self-contained UTF-8 file up to 64 KiB. Dependencies must be
installed; sibling project modules are not automatically shipped. Preview executes
the trusted recipe; keep data access/training out of imports and the recipe factory.

This delivers the single-file portion of SM-36a. Broader package/dependency
shipping, row eligibility/output rules and multiple-model branches remain open.

## Verification

- 56 real CLI generation tests: all published examples, comma-separated columns,
  task-specific menus, output naming, enabled six-month schedule, generated Python
  recipe fit/save/load, pandas/Polars and regression/classification, preview/CV,
  existing lifecycle/compute combinations. Log: `.cache/sm33f-cli-python.log`.
- 15 new Python tests: pandas/Polars fitted state; fold-local means; local and
  real MLflow package loads in fresh subprocesses without the current project;
  checksum rejection before source execution; simultaneous v1/v2 code isolation;
  current project loading only for training, never score/approve/reject/rollback.
  Log: `.cache/sm33f-python-final.log`.
- 201 existing artifact/batch/CV/retraining/runtime/template/validation/promotion
  tests passed; one optional PySpark test skipped in this Python environment.
  Log: `.cache/sm33f-regression.log`.
- 11 existing local MLflow/preview tests passed.
  Log: `.cache/sm33f-model-preview.log`.
- Full ty, scoped Ruff/format and strict MkDocs passed. The matching 0.9.0 wheel built and contains
  both project source-loading modules. Strict CLI dev validation passed. Resolved
  schedule: `0 0 3 1 1,7 ?`, `Europe/Copenhagen`, `UNPAUSED`; sync includes
  src/preprocessing.py. No warning was accepted as strict validation success.
- Independent review found stale JSON preprocessing instructions and an unclear
  regression metric description; both were corrected.

No live deployment, data/model/alias change or cloud job execution occurred for
this follow-up. Previous SM-33E personal serverless evidence remains report 60;
it does not prove the new custom-code path on Databricks. Existing generated
projects were not overwritten. Regenerate or migrate them and use the matching
wheel before trying this new entrypoint.

Next: SM-34 independent score/train scheduling and further window controls.
The user-requested delivery includes this report and updated reports 37/58.
Ignored rehearsal artifacts and unrelated `.tmp-review-model/` are excluded.

## SM-33G follow-up: answer-driven questions

Date: 2026-09-26. Status: DONE for local setup acceptance; included in SM-33F/G delivery.
The user explicitly chose questions over inspecting source data.

New initialization-only `event_time_kind`/`result_time_kind` choices distinguish
database timestamp instants, local timestamps without timezone, dates and text.
For text, `event_text_kind`/`result_text_kind` distinguish offset datetime, local
datetime and date-only. Existing format/timezone/date-only questions appear only
when relevant. Unused date branches remain hidden. Defaults preserve existing
timestamp behavior; no automatic schema/value discovery or conversion was added.
Generated runtime config still uses the same parsing fields. Actual values are
validated at execution. The random-window example now declares its text formats.

Verification:

- 14 new tests failed before implementation, then passed for both date branches
  across six representations and the disabled case.
- 37 template/preview tests passed, including CV/cron/policy compute visibility.
  `.cache/sm33g-check.log`.
- 56 actual CLI generation tests passed; `.cache/sm33g-cli.log`.
- Scoped Ruff/ty, strict MkDocs and strict dev Bundle validation passed;
  `.cache/sm33g-docs.log`, `.cache/sm33g-validate.log`.
- No data inspection, deployment, training or cloud job execution. This acceptance
  covers generated configuration and visibility predicates, not a live data run.

The generated README and user guide explain this flow. SM-34 remains next.

Commit verification (2026-09-26): the combined local suite passed 243 tests with
one optional PySpark skip; actual CLI generation passed 56 tests. Logs:
`.cache/sm33fg-commit-tests.log`, `.cache/sm33fg-commit-cli.log`.
All applicable staged-file hooks passed, including Ruff/format and full ty;
`.cache/sm33fg-precommit.log`. Changelog line endings were normalized to the
existing `.gitattributes` LF rule without changing global Git configuration.
