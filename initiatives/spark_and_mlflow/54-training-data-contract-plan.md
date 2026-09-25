# Pre-SM-34 training data contract implementation plan

Date: 2026-09-25. Approved scope: the user's discussion of record identities,
optional event/result dates, explicit parsing/timezones, random/temporal splits,
Core CV reuse and monthly lookback semantics. Baseline: `0c0fd17f` (SM-33).

**Goal:** Make the local Bundle usable with ordinary labeled tables as well as
time-dependent data, with explicit, reproducible training/evaluation rules.

**Architecture:** Keep Bundle entrypoints thin. Use the same readable field names
throughout workflow settings, runtime contracts and saved training evidence. Reuse Core
splitters and CV. Separate source snapshot, row selection, final evaluation
split, optional training CV and scheduling. Spark must filter normalized dates
before bounded local materialization; pandas/Polars fit and inference stay local.

**Execution:** Use task-by-task TDD and review. Start with SM-33A. Do not begin
SM-34 schedules until SM-33A through SM-33E are complete. Status is recorded in
OPEN_QUEUE.md; this document defines the work, not a claim of implementation.

## Constraints and decisions

- English files, docs, examples and diagrams. Continue the user's `090` branch.
- Two existing jobs, no new default admission tables, no hardcoded company data.
- Prefer `record_key_columns` and `result_available_at_column` in user settings.
  Keep `event_column` with explicit explanatory copy: the source observation
  timestamp column, not a date boundary or the job schedule.
- Use the new field names directly; do not retain aliases or a normalization
  module. The user explicitly accepts recreating experimental projects/models.
  No production compatibility requirement applies to this rename.
- Column names identify existing source columns; never invent IDs, timestamps
  or label-availability dates. Composite keys identify records, not implicit CV
  groups. Source row order must not accidentally determine random splits.
- No claim of arbitrary date-string autodetection. Native instant timestamps,
  local timestamps, dates and strings need explicit semantics. Unknown formats,
  missing source zones and ambiguous/nonexistent DST instants fail clearly.
- Date-free training must work for both engines. Missing dates must not silently
  switch an explicitly temporal configuration to random behavior.
- Availability filtering is independent of temporal/random splitting. Disabled
  means the user supplies a labeled dataset; it does not mean filling unknown
  targets. Enabled means per-row availability <= the pinned result cutoff;
  unavailable/null availability is counted and excluded under explicit rules.
- Final evaluation rows stay outside model fitting, FE fitting and training CV.
  CV reports do not replace the shared candidate/champion evaluation dataset.
- A monthly job can train on date-free data. Rolling calendar selection applies
  only when explicitly selected and always names its event column/timezone.
- Pin Delta source version, selection, split policy and membership evidence per
  run. Approval must replay the saved evaluation rather than draw a new split.
- No data truncation on budget overflow and no unbounded collect as a parsing fix.

## SM-33A - Direct, readable field names

**Files:** existing Core frame contracts, Databricks training/scoring/workflow
modules, their callers/tests/examples, Bundle schema/config and user guides.

**Contract:** `record_key_columns` and `result_available_at_column` are the
actual field names, including serialized training specs. Remove the temporary
`_workflow_fields.py` adapter and old initializer fields. Earlier experimental
projects and models will be recreated; no compatibility layer is required.

- [x] Remove the alias adapter and its compatibility-only tests.
- [x] Rename fields directly across training, scoring and frame contracts.
- [x] Simplify initializer prompts and output; update examples and documentation.
- [x] Verify real pandas/Polars lifecycle tests using newly saved evidence.
- [x] Verify actual CLI generation, frame/Delta contracts, lint, types and docs.

## SM-33B — Explicit timestamp parsing before filtering

**Files:** a focused `training_dates.py` under the Databricks integration;
`local_retraining.py`, `workflow_config.py`; focused date tests and actual
Delta reader tests under `skyulf-core/tests/integrations/`.

**Contract:** Event and availability columns each have explicit source parsing
rules. A format is required for strings; local timestamps/date-only inputs need
declared timezone/calendar semantics. Already aware timestamps retain their
instant. Normalize before source filtering, ordering, label filtering and
local conversion; preserve the instant across Spark-to-Python transport.

- [ ] Reproduce the current unsafe naive-as-UTC assumption and boundary shift.
- [ ] Add shared parsed-date rules with native/explicit-format modes. Document
  syntax per engine; do not pass pandas format strings into Spark unchecked.
- [ ] Reject ambiguous day/month strings, missing years, invalid dates,
  missing timezone for local times and DST overlaps/gaps without guessing.
- [ ] Implement bounded, deterministic source projection/filtering on normalized
  values; do not hide invalid/null event rows by filtering them away first.
- [ ] Verify equivalent `03:00+03:00`/`00:00Z` instants, different event/result
  zones, date-only policy, offset mixtures, null results and exact boundaries.
- [ ] Verify with non-UTC Spark session and process timezone, pandas/Polars and
  real Delta reads. Valid rows must retain the same split and availability.

## SM-33C — Date-free training and independent result availability

**Files:** `local_retraining.py`, `local_workflow.py`, `local_approval.py`,
`workflow_config.py`, Core `preprocessing/split.py` as a reused dependency;
template initialization/config, tests and examples.

**Contract:** Explicit `split_strategy=random|temporal`. Random requires test
proportion/seed (and optional class stratification), not event/date fields.
Temporal requires the event mapping and valid boundaries. Independently choose
availability disabled or a result-date mapping plus pinned cutoff. Manual
snapshot version still identifies data independently of event dates.

- [ ] Test a table containing only keys, features and target on both engines.
- [ ] Reuse Core DataSplitter for deterministic random holdout; order by stable
  record identity before splitting and persist enough evidence to replay.
- [ ] Permit absent/null time fields only when unused. Reject contradictory
  policies and malformed active fields; do not infer a split from missing data.
- [ ] Make availability filtering optional and per-row. Count late/unknown
  outcomes and define null-target failure behavior for the labeled-data mode.
- [ ] Separate observation window end from result-availability cutoff in new
  settings; document the new contract for regenerated projects.
- [ ] Extend saved evidence/versioned readers so approval and retries use the
  same snapshot and holdout. Use newly generated evidence for validation.
- [ ] Test delayed outcomes becoming eligible in a later snapshot; no target
  leakage, deterministic class stratification and bounded-read failures.

## SM-33D — Core CV connection and selection/window separation

**Files:** `local_retraining.py`/`local_workflow.py`, `workflow_config.py`, existing
`modeling/cross_validation.py`, pipeline CV APIs and `_tuning/splitters.py` as
dependencies; template config and training/CV integration tests.

**Contract:** Optional CV uses existing Core APIs inside the training partition.
Expose supported K-fold/stratified/shuffle/time-series choices and bounded fold
settings. Temporal CV consumes correctly ordered data and never shuffles. Keep
the final heldout dataset identical for candidate/champion comparison. Model
search/trial tuning and explainability remain SM-36 scope.

- [ ] Audit/reuse existing pipeline CV entrypoints and FE fold-refitting rules.
  Reject unsupported combinations rather than silently downgrade requested CV.
- [ ] Wire optional CV with deterministic seeds/folds and MLflow evidence;
  tests must show heldout rows never reach fitting or fold preprocessing.
- [ ] Make full-snapshot selection and rolling-calendar selection explicit.
  Rolling lookback names its event column and window timezone; clarify whether
  counts include the holdout period. Document the chosen semantics explicitly,
  without an unexplained default in date-free mode.
- [ ] Separate job execution date/cron from data selection. Monthly invocation
  of random full-snapshot training must not require event dates or lookback.
- [ ] Verify month/year boundaries, source ordering, time-series folds and
  recorded candidate/champion comparison on the same final evaluation rows.

## SM-33E — Generic template, operator documentation and live acceptance

**Files:** template schema/config/examples/README, both Bundle user guides and
Mermaid walkthrough; focused generated-project tests; a new initiative report
and bounded rehearsal driver using existing personal workspace resources.

- [ ] Hide irrelevant date prompts for random/no-availability initialization;
  show explicit column, format, source zone and cutoff controls only when used.
- [ ] Provide complete English random/date-free and temporal/delayed-result
  examples, including optional availability with random splitting.
- [ ] Explain source schema, per-row dates vs global boundaries, missing targets,
  composite keys, split vs CV vs cron, lookback and migration in Mermaid/tables.
- [ ] Generate/strictly validate projects for both engines and both task types.
- [ ] Exercise actual Databricks training, MLflow metrics/artifacts, saved
  evidence approval, scoring/new rows/no-op with date-free and temporal inputs.
  Include non-UTC boundaries and delayed-result eligibility. Reuse existing test
  schema/jobs where possible; no destructive cleanup or production mutation.
- [ ] Record exact live resources/runs, local checks and limitations. Mark
  SM-34 READY only after this acceptance passes; company production remains a
  separate later gate.

## Rulings

- 2026-09-25: Start with SM-33A after committing SM-33, as requested. This first
  slice changes vocabulary only; it must not claim optional dates or new CV.
- 2026-09-25: User explicitly rejected field compatibility overhead because
  this work is pre-production. Use direct renames and recreate old test models
  when running the combined live acceptance; do not add a field adapter.
