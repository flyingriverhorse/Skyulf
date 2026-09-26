# Pre-split cleaning and existing leakage safeguards

Date: 2026-09-26. Cleanup audit complete; SM-33H1/H2 remain planned, not delivered.
Baseline: `8a88b2b2`. User requested inspecting existing nodes/leakage controls
before adding Bundle tasks. A subsequent requested Core temporal review is recorded
below; no cloud resources were changed.

For implementation, use the executing-plans workflow task by task, with failing
regression tests before runtime changes and verification before completion.

**Goal:** Run explicit training-row cleanup before Bundle splitting while reusing
Core nodes, operation-aware leakage validation and existing artifact/replay paths.

**Architecture:** Keep the recipe in the generated `src/preprocessing.py`.
Introduce a separate ordered pre-split recipe, not arbitrary unvalidated Python
execution over the full dataset. Reuse Core Calculator/Applier implementations
and the shared classifier; do not copy algorithms or maintain another leakage list.

**Tech stack:** Existing pandas/Polars Core, bounded Delta reads, Python project
source snapshots, MLflow artifacts and Databricks Bundle entrypoints.

## What already exists

- `skyulf/leakage.py`: `step_learns_from_data`, `leakage_exemption_reason` and
  `validate_leakage_safety`. Actual parameters refine conservative registry flags.
  Unknown nodes/modes are treated as data-dependent. Fixed-mode exemptions do
  not prove target/future-feature provenance.
- `skyulf/pipeline/_pipeline.py`: checks before fit/get_fitted_split. An external
  `SplitDataset` marks the input already split. With no boundary configured,
  the validator returns an advisory, even in raise mode: calling it on a bare
  cleanup list is NOT sufficient to enforce pre-split admission.
- `backend/ml_pipeline/_execution/_leakage_validation.py`: DAG/branch-aware
  validation imports the Core classifier. Core/Bundle must not import backend.
- `skyulf/preprocessing/base.py`: fits on the train portion of SplitDataset and
  applies learned state to heldout portions. `fold_adapter.py` and Bundle
  `local_cv.py` already provide fold-local fitting; CV alone is not an exemption.
- `integrations/databricks/local_retraining.py`: bounded read, split, CV, then
  fit_local_workflow. No project pre-split cleanup hook exists. Current split
  validation rejects missing available targets and null/duplicate record keys.
- `integrations/databricks/project.py`: currently resolves only build_preprocessing.
  Existing custom classes are unknown to leakage metadata unless separately
  declared; do not automatically permit them before split.
- `preprocessing/pipeline.py`: ordinary inference skips DropMissingRows,
  Deduplicate and resampling; preserve_rows checks the steps it applies. A new
  source-cleaning phase cannot assume its train filters already execute in score.
- `integrations/databricks/local_approval.py` and `local_workflow.py` rebuild a
  holdout from saved training evidence. They must replay any new training filters.

## Operation audit

This table describes the CURRENT Core classifier, not delivered Bundle support
for running these nodes before its external split.

| Existing operation | Core pre-split classification | Bundle scope / caveat |
| --- | --- | --- |
| DropMissingRows with explicit subset/how/threshold | Fixed, permitted | First implementation: training eligibility; preserve keys and target alignment. Its per-row missing percentage is different from learned column missingness. |
| ManualBounds with configured lower/upper bounds | Fixed, permitted | First implementation: explicit domain bounds; null/NaN values are retained by this node, so combine with an explicit missing-row policy if required. |
| TextCleaning, AliasReplacement, ValueReplacement, InvalidValueReplacement | Fixed, permitted | Reuse existing nodes. Keep feature-changing steps after split initially so saved FE also applies them during inference. |
| SimpleTransformation, DateFeatures, FeatureInteraction, GeoDistance, H3Index | Registry says not learned | Other fixed feature operations, not initial row-cleaning scope. Preserve feature availability, parameters and optional dependencies; keep them in ordinary saved FE initially. |
| SimpleImputer(strategy=constant) | Fixed, permitted | Mean/median/most-frequent modes are learned; initial Bundle recipe still keeps feature transforms after split. |
| DropMissingColumns with an explicit list and no positive missing_threshold | Fixed, permitted | A positive threshold learns the column drop list and is blocked before split. Do not drop keys, target or required time fields. |
| Casting to a configured non-categorical type | Fixed, permitted | Categorical conversion learns vocabulary and is blocked before split. |
| MissingIndicator with explicit columns | Fixed, permitted | Automatic missing-column discovery is learned. |
| CustomBinning with explicit columns and fixed edges | Fixed, permitted | Automatic column selection is learned, even with fixed bin edges. |
| HashEncoder with explicit columns; explicit polynomial features; fixed math/feature formulas | Parameter-dependent exemptions exist | Not a guarantee against referencing a target/future value; no automatic Bundle-wide enablement. |
| StandardScaler and other fitted scalers, ordinary OneHotEncoder, learned imputers, IQR/ZScore/Winsorize/EllipticEnvelope, learned feature selection | Learned, blocked | Fit after split and within each CV fold; apply saved state to holdout/inference. |
| Deduplicate | Currently marked learned, blocked | Fit stores config only, but survivor selection depends on other rows. Do not bypass this policy; audit duplicates/group isolation as separate scope. |
| LagFeatures / RollingAggregate | Registry says not learned | Temporal review fixed missing-sort fallback, invalid direct lags and known current-target rolling. Automatic availability/history is still absent; not initial pre-split cleanup. |
| Unknown custom code | Treated as learned, blocked | A Python filename or developer assertion is not a provenance proof. No unrestricted pre-split callback in the first delivery. |

The earlier conversational suggestion that Deduplicate could simply be placed
before split was too broad for the existing contract. Its current rejection is
preserved until separately reviewed. No-op or target-only exemptions also do not
automatically qualify a step for the training-filter execution contract.

## Follow-up Core temporal review (2026-09-26)

The user requested actual lag/rolling verification before starting SM-33H1.
Reproduced and corrected:

- A missing declared `sort_by` silently retained input order, allowing a past-row
  feature to read a chronologically future observation. Both engines now reject it.
- Direct lag artifacts accepted zero/negative shifts. The applier now requires
  positive integers; calculator normalization remains unchanged.
- Current-target rolling exposed the same row's answer, including window 1.
  Core rejects this known-target definition before fit irrespective of split
  placement or warning mode. Direct calculator target context and backend
  selected-branch admission share that rule.

Verified boundaries: configured groups isolate histories and appending future
observations does not change prior features on either engine. Training state does
not include history: independently applied batches start their own windows. Rolling
still includes the current observed feature, which must be available at prediction
time. Stable sort ties/null handling and caller-ordered mode are unchanged.

This is not general point-in-time certification. Result availability, forecast
horizon, gap/embargo policy, tied/missing time policy, cross-batch context retrieval
and context-row removal remain explicit SM-36a work. Unknown target provenance
cannot be inferred from an arbitrary standalone frame. Do not enable these nodes
as generic pre-split cleaning based on `learns_from_data=False`.

Regression evidence is in `test_temporal_leakage_guards.py` and the existing backend
leakage validation tests. The original Core reproduction failed 17 tests (4 causal
prefix cases already passed); backend admission failed 3 tests before the fix.
Final relevant suites passed: 1,981 Core and 1,731 backend tests; scoped Ruff,
repository ty and strict MkDocs passed. No Databricks execution was performed.

## SM-33H1 — Core-backed training filters before split

Status: READY. Dependency: SM-33G. First scope: DropMissingRows and ManualBounds.

Primary files: `integrations/databricks/project.py`, `local_retraining.py`,
`workflow_config.py`, `job_runtime.py`, generated `src/preprocessing.py`, preview,
and the existing integration test modules. Reuse `leakage.py`, registry and node
implementations; only amend Core if a reproduced shared defect requires it.

- [ ] Add failing tests for configured filters executing BEFORE the real Bundle
  split and for StandardScaler/mean imputation/unknown custom code being refused
  before reading/fitting data. Test the actual train entrypoint, not only a helper.
- [ ] Add optional `build_pre_split_steps()` in the SAME project Python file;
  absent/empty means existing behavior. It returns ordinary Core step dictionaries.
  The name is a planned API until implemented. Keep build_preprocessing unchanged.
- [ ] Reuse step_learns_from_data with authoritative target context for admission;
  independently enforce the narrower initial row-filter contract. Do not treat
  on_leakage=warn/ignore or a no-split advisory as permission to bypass that contract.
- [ ] Require explicit existing filter columns/bounds, rejecting missing-column
  silent no-ops. Preserve identities and target pairing. Training-target missingness
  may be an explicit filter, never an automatic inference rule.
- [ ] Apply filters via existing calculators/appliers using the selected engine,
  then convert only at the existing splitter boundary if needed. Keep loaded raw
  data within current row/byte limits; cleanup is not a way to exceed read budgets.
- [ ] Make source projection include declared filter-only columns without exposing
  them as model features. Do not remove/overwrite keys or required time metadata.
  Invalid keys remain explicit errors; duplicate resolution is separate scope.
- [ ] Fix and document order: source snapshot/window/availability selection,
  optional current deterministic source sample, bounded transfer, training filters,
  final train/holdout split, fold-local learned FE/model. Do not silently move or
  refill the existing source sample; report that filters can reduce its final size.
- [ ] Save requested filters and counts by step. Report empty/undersized partitions
  and stratification failures clearly before model publication. Report metrics as
  evaluated on the declared eligible population, with exclusion counts visible.
- [ ] Test both engines, null/NaN, duplicated pandas indices, missing target,
  all-rejected input, explicit bounds, key/order/label preservation, sample size,
  random/temporal split and CV. Preserve existing learned-before-split failures.
- [ ] Document an English same-file example and offline preview of phase order.

Suggested project recipe (NOT implemented yet):

```python
def build_pre_split_steps():
    """Declare training eligibility using existing Core nodes."""
    return [
        {"name": "known_target", "transformer": "DropMissingRows",
         "params": {"subset": ["target"], "how": "any"}},
        {"name": "valid_age", "transformer": "ManualBounds",
         "params": {"bounds": {"age": {"lower": 0, "upper": 120}}}},
    ]
```

## SM-33H2 — Saved cleanup evidence, lifecycle replay and acceptance

Status: WAIT for SM-33H1. Required before marking the combined feature complete.

Primary files: `local_retraining.py`, `local_approval.py`, `local_workflow.py`,
existing training evidence/manifest helpers, MLflow tests, generated preview/README
and `docs/user_guide/databricks_bundle.md`.

- [ ] Add failing artifact/approval tests first: change the editable Python file
  after training; approval must still reconstruct the exact original holdout.
- [ ] Version and save the filter recipe/code identity, ordered survivor/holdout
  membership evidence and counts alongside existing MLflow training artifacts.
  Reuse project source snapshots; do not create Delta admission tables for this.
- [ ] Route automatic comparison, manual approve/reject and replay through saved
  rules. Preserve concrete source version, sample and holdout hashes. Never read
  the current project file as a substitute for saved training evidence.
- [ ] Compare candidate/champion on the SAME candidate evaluation population.
  A champion's historic training filters must not silently shrink that holdout.
- [ ] Keep target-based training filters out of scoring; inference must work without
  a target column. Explain that rows excluded from training may still be scored.
  Shared prediction eligibility is a separate opt-in contract under SM-36a.
- [ ] Verify fresh-process pandas/Polars MLflow load, training/CV metrics, reject,
  approve, rollback and unchanged scoring output identity. Tampered rules/evidence
  must fail; changing current source code must not alter old-model replay.
- [ ] Run real CLI generation/preview/strict validation. Plan a bounded personal
  Databricks rehearsal and execute only within explicit live authorization; record
  local and live results separately. Do not claim local evidence proves live support.

## SM-36a remaining policy work (explicitly separate)

- [ ] Review pre-split duplicate handling: exact vs entity duplicates, target conflicts,
  deterministic keeper ordering and group isolation across folds/holdout. Reproduce
  current behavior and change shared Core/backend policy only with parity tests.
- [ ] If feature-changing operations are moved before split, capture/reapply their
  fixed transformations exactly once for heldout/raw inference. Preserve raw input
  schema and do not silently omit or double-apply source normalization.
- [ ] Add explicit keyed scoring eligibility/rejection results, coverage counts and
  incremental cursor/publication semantics; no silent row loss or retry loops.
- [ ] Separately assess lag/rolling history, group and temporal boundaries and serving
  context. A learns_from_data=False flag is not sufficient evidence for this scope.
- [ ] Broader custom code/dependency packaging and output rules remain as in report 58.

## Audit evidence and implementation checks

- 940 existing Core leakage/cleaning operation tests passed:
  `test_leakage_safety_validation.py`, `test_cleaning_operation_leakage.py`,
  `integration/test_leakage_operation_contract.py`.
  Log: `.cache/sm33h-leakage-audit.log`.
- 109 existing backend leakage/graph tests passed:
  `tests/unit/test_leakage_validation.py`,
  `tests/integration/test_leakage_graph_semantics.py`.
  Log: `.cache/sm33h-backend-leakage.log`.
- Direct registry/classifier probes confirmed the distinctions above. A real
  DropMissingRows(target) -> ManualBounds(age >= 0) chain kept keys [1, 4] and
  correctly aligned targets [1.0, 4.0] on both pandas and Polars.
- These verify existing building blocks, NOT a delivered pre-split Bundle hook.
  Add runtime/approval/MLflow tests above before marking H1/H2 complete. Broaden
  to existing Bundle/runtime/CV suites; run scoped Ruff, full ty, CLI validation
  and docs checks. Commit only after the applicable checks pass.
