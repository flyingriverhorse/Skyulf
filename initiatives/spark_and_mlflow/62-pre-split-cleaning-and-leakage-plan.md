# Pre-split cleaning and existing leakage safeguards

Date: 2026-09-26. SM-33H1/H2 verified; H2 personal serverless acceptance passed. SM-33H3 local and personal serverless acceptance passed; SM-34 is READY.
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

## Baseline before SM-33H1

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

Status: DONE for H1 scope. Dependency: SM-33G. First scope: DropMissingRows and ManualBounds.
Implementation baseline: `522c6e82` (temporal guard review committed with passing hooks).

Implementation boundary: because both automatic promotion and manual approval
reconstruct the holdout through the shared splitter, H1 must carry the recipe in
the saved training specification and use it on those paths. This minimum replay
coherence belongs to H1; H2 retains extended membership/code evidence, tampering,
fresh-process MLflow and live acceptance tests. Do not expose a cleaned training
path with an unfiltered promotion holdout.

Primary files: `integrations/databricks/project.py`, `local_retraining.py`,
`workflow_config.py`, `job_runtime.py`, generated `src/preprocessing.py`, preview,
and the existing integration test modules. Reuse `leakage.py`, registry and node
implementations; only amend Core if a reproduced shared defect requires it.

- [x] Add failing tests for configured filters executing BEFORE the real Bundle
  split and for StandardScaler/mean imputation/unknown custom code being refused
  before reading/fitting data. Test the actual train entrypoint, not only a helper.
- [x] Add optional `build_pre_split_steps()` in the SAME project Python file;
  absent/empty means existing behavior. It returns ordinary Core step dictionaries.
  build_preprocessing remains unchanged.
- [x] Reuse step_learns_from_data with authoritative target context for admission;
  independently enforce the narrower initial row-filter contract. Do not treat
  on_leakage=warn/ignore or a no-split advisory as permission to bypass that contract.
- [x] Require explicit existing filter columns/bounds, rejecting missing-column
  silent no-ops. Preserve identities and target pairing. Training-target missingness
  may be an explicit filter, never an automatic inference rule.
- [x] Apply filters via existing calculators/appliers using the selected engine,
  then convert only at the existing splitter boundary if needed. Keep loaded raw
  data within current row/byte limits; cleanup is not a way to exceed read budgets.
- [x] Make source projection include declared filter-only columns without exposing
  them as model features. Do not remove/overwrite keys or required time metadata.
  Invalid keys remain explicit errors; duplicate resolution is separate scope.
- [x] Fix and document order: source snapshot/window/availability selection,
  optional current deterministic source sample, bounded transfer, training filters,
  final train/holdout split, fold-local learned FE/model. Do not silently move or
  refill the existing source sample; report that filters can reduce its final size.
- [x] Save requested filters and counts by step. Report empty/undersized partitions
  and stratification failures clearly before model publication. Report metrics as
  evaluated on the declared eligible population, with exclusion counts visible.
- [x] Test both engines, null/NaN, duplicated pandas indices, missing target,
  all-rejected input, explicit bounds, key/order/label preservation, sample size,
  random/temporal split and CV. Preserve existing learned-before-split failures.
- [x] Document an English same-file example and offline preview of phase order.

Implemented project recipe:

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

Status: DONE for the documented local and personal serverless acceptance scope.
H1 was committed as `55b31ca9`; H2 changes remain uncommitted. H3 node expansion
is separate and has not been implemented.

Primary files: `local_retraining.py`, `local_approval.py`, `local_workflow.py`,
existing training evidence/manifest helpers, MLflow tests, generated preview/README
and `docs/user_guide/databricks_bundle_walkthrough.md`.

H2 implementation now saves `training_filter_evidence.json` with recipe/code
identity, ordered pre-filter/survivor/train/holdout membership and counts, bound
to the saved comparison. Automatic and manual lifecycle paths load the saved
artifact and reject evidence or engine mismatches. Current project edits do not
replace the saved recipe. Legacy artifacts without the new evidence keep their
previous identity; they do not gain the new guarantees retroactively.

Local acceptance: real SQLite MLflow rehearsed pandas and Polars, three versions
each, bootstrap, promotion, rejection and rollback to v1. The full real WSL
Spark/Delta date/split module passed 15 tests; 56 installed-CLI generation tests,
strict generated dev validation with the new wheel and strict MkDocs passed.
The final changed-path suite passed 144 tests (five existing policy deprecation
warnings), with full ty and scoped Ruff/format passing. The earlier 248-pass
suite overlaps and predates final automatic-path tightening. Independent review
found a JSON-null receipt validation bypass, fixed with two real MLflow red/green
tests and 102 passing related lifecycle tests. Scoped round-two review approved
both fixes, including exact key/prediction identity checks in the prepared live
harness. The approved live run below also passed; local and cloud evidence are
recorded separately.

The approved bounded personal rehearsal reused
`workspace.skyulf_lifecycle_test` with new `sm33h2_20260926_r1_*` resources:
one 240+3-row synthetic source, two prediction outputs and two models. One
ephemeral serverless run with three sequential tasks succeeded in 701.797
seconds, within its 900-second limit. No new schema or persistent job was
created, and existing Bundle jobs were not redeployed. Results are below.

- [x] Add failing artifact/approval tests first: change the editable Python file
  after training; approval must still reconstruct the exact original holdout.
- [x] Version and save the filter recipe/code identity, ordered survivor/holdout
  membership evidence and counts alongside existing MLflow training artifacts.
  Reuse project source snapshots; do not create Delta admission tables for this.
- [x] Route automatic comparison, manual approve/reject and replay through saved
  rules. Preserve concrete source version, sample and holdout hashes. Never read
  the current project file as a substitute for saved training evidence.
- [x] Compare candidate/champion on the SAME candidate evaluation population.
  A champion's historic training filters must not silently shrink that holdout.
- [x] Keep target-based training filters out of scoring; inference must work without
  a target column. Explain that rows excluded from training may still be scored.
  Shared prediction eligibility is a separate opt-in contract under SM-36a.
- [x] Verify fresh-process pandas/Polars MLflow load, training/CV metrics, reject,
  approve, rollback and unchanged scoring output identity. Tampered rules/evidence
  must fail; changing current source code must not alter old-model replay.
- [x] Run real CLI generation/preview/strict validation. Plan a bounded personal
  Databricks rehearsal and execute only within explicit live authorization; record
  local and live results separately. Do not claim local evidence proves live support.

## SM-33H3 — All existing nodes in the correct phase

Status: DONE for the documented local and personal serverless scope on
2026-09-26. Run 848857785722024 passed all three tasks; SM-34 is READY. The
matrix retains explicit optional/conditional limitations rather than claiming
every parameter mode is production-verified.

**Goal:** Make every existing Core preprocessing node usable through the local
Bundle in its appropriate phase, reusing the implementation and preserving
training, holdout, CV, artifact and inference semantics on pandas and Polars.

**User decision:** Prioritize existing cleaning nodes in the right order. Do not
implement the newly suggested generic business-rule filter, group-disjoint split
or data-quality threshold system now. No new cleaning algorithm is requested.
Broad native Spark expansion and new temporal-history retrieval remain separate.

**Follow-up request:** Allow project-owned custom pre-split logic in the same
Python file, analogous to custom ordinary preprocessing. Implemented through
`custom_step(..., pre_split={"effect": "filter", "required_columns": [...],
"learns_from_data": False})`, reusing isolated source registration and snapshots.
Custom value-changing steps remain ordinary preprocessing; arbitrary custom
normalization is not admitted as a pre-split filter.

**Architecture:** Keep `build_pre_split_steps()` and `build_preprocessing()` in
the same generated `src/preprocessing.py`. Reuse the registry, operation-aware
leakage classifier, Calculator/Applier implementations and saved project code.
All-node coverage means valid placement for each node/mode, not permission to
run every node before split. Ordinary preprocessing already supports many nodes;
extend only demonstrated integration gaps rather than wrapping each node again.

**Implementation sequence and reviewed boundaries:**

1. Audit the live registry, modes and existing tests; distinguish valid Core
   placement from delivered Bundle integration and verified execution.
2. Reuse fixed in-place Core normalization on a working copy for eligibility,
   retaining source keys and raw model features. Feed selected raw features
   through a fixed normalization prefix in the existing model pipeline so
   training, CV, heldout and prediction each see a single value transformation.
   Keep normalized target values separate; target-only rules never require a
   target in prediction input. Reject writes to source keys and window columns.
3. Bind target-normalization semantics to saved model comparison, including
   regression units and class mappings whose label sets might be unchanged.
   Preserve JSON recipe meaning across source/spec replay, including numeric
   replacement keys; do not silently merge colliding keys or nonfinite values.
4. Extend the same phase contract to existing Deduplicate and explicitly opted-in
   custom eligibility. Preserve source-key guards, target pairing and saved
   source registration before validating custom recipes in a fresh process.
5. Exercise remaining families in their supported ordinary/train-only phases,
   update matrix evidence, docs and offline preview, then run the final gates.
   All-node integration does not mean every fixed operation is admitted before
   split; derived feature creation may remain ordinary preprocessing.

**Files and responsibilities:**

- `skyulf-core/skyulf/registry.py`, `leakage.py` and `preprocessing/`: authoritative
  inventory, parameter-dependent placement, fit/apply, row alignment and replay.
  Change shared behavior only for reproduced defects with regression coverage.
- `skyulf-core/skyulf/integrations/databricks/project.py`, `local_retraining.py`,
  `workflow_config.py`, `local_cv.py`, `local_approval.py`, `local_workflow.py`:
  load ordered recipes, project source columns, enforce phase boundaries and
  reproduce saved evaluation inputs. Reuse existing paths rather than a parallel
  node-execution framework.
- `skyulf-core/skyulf/inference/local_pipeline.py`, `project_code.py` and existing
  MLflow integration: preserve necessary normalization and fitted state when a
  raw scoring frame is supplied; never require the training target at inference.
- Extend `skyulf-core/tests/integrations/test_databricks_pre_split_filters.py`,
  `test_databricks_project_preprocessing.py`, `test_databricks_local_cv.py`,
  `test_databricks_local_approval.py` and relevant existing Core node tests.
- Generated `src/preprocessing.py`, `README.md.tmpl` and
  `docs/user_guide/databricks_bundle_walkthrough.md`: editable Python examples,
  placement matrix and explicit training-only versus prediction behavior.

Acceptance sequence:

- [x] Inventory every current registry node and alias, grouping parameter modes
  that change placement or row behavior. Start from the audited 58 calculator
  classes / 62 IDs, but detect registry additions rather than freezing that count.
  Record valid pre-split, post-split, train-only, heldout and inference behavior,
  engine/dependency requirements and the test covering each route. Include text,
  geo, inspection and temporal nodes; do not silently omit optional families.
- [x] Add failing tests for missing integration first. Include an ordered
  `ValueReplacement(-999 -> null) -> DropMissingRows` recipe, text/alias/casting
  normalization before cleanup and fixed invalid-value handling. Specify how
  normalization affects keys, target and time columns; preserve source identity
  and source-window semantics or reject conflicting configurations explicitly.
- [x] Integrate safe fixed modes through existing Core implementations. Keep
  learned imputation/scaling/encoding/binning/feature selection after split and
  refit them inside training folds. Admit modes using shared leakage rules;
  constant/fixed and learned modes of one node must not be conflated.
- [x] Integrate existing Deduplicate subset/keep behavior with deterministic
  ordering, preserved X/y alignment and explicit target-conflict handling. Test
  duplicate source keys separately from duplicate features; do not silently
  remove current key guards or collapse different observations of one customer.
  This reuses Deduplicate and does not introduce group-aware splitting.
- [x] Cover remaining existing node families in ordinary preprocessing using
  real recipes. Keep resampling train/fold-only, splitters at the workflow split
  boundary and inspection nodes read-only. Validate existing lag/rolling order,
  history requirements and row preservation without inventing history retrieval.
  A rejection of invalid placement must not substitute for testing valid usage.
- [x] Save and replay ordered normalization exactly once for raw heldout/scoring
  input and manual/automatic comparison. Train-only row exclusions must not drop
  prediction requests. Handle target-only normalization without requiring target
  at inference; detect incompatible recipe/schema/state before publication.
- [x] Design and implement explicit opt-in custom pre-split steps in the same
  `src/preprocessing.py`, reusing source-isolated custom Calculator/Applier
  registration and saved code. Start with fixed training-row eligibility, for
  example excluding project test accounts, with declared required columns and
  row effects. Do not create a new built-in business-rule filter for this.
  Preserve immutable source keys, row order and target pairing; reject added
  rows, undeclared column changes, missing inputs and invalid return types. For
  custom normalization, reuse the saved once-only inference replay contract
  above rather than treating value changes as a row filter. Keep learned custom
  statistics in post-split/fold-local preprocessing. Metadata is an explicit
  developer assertion, not proof that arbitrary Python is leakage-free; document
  this limitation and do not silently exempt all existing custom nodes.
- [x] Test custom pre-split behavior on both engines, saved-source approval
  replay after editing the project file, and target-free inference that omits
  training-only filters. Include negative row/key/target mutation cases and an
  English commented Bundle example. Preserve absent/empty-hook behavior and
  existing custom ordinary preprocessing compatibility.
- [ ] Test mixed chains and each supported node/mode on pandas and Polars through
  fit, heldout evaluation, CV and fresh-process local/MLflow artifact loading.
  Assert feature values/order, row identity, target alignment and prediction
  equivalence; test unseen categories, nulls and no-op configuration. Exercise
  optional dependency lanes explicitly and report unverified cases as open.
- [x] Update offline preview and English Python examples with the actual order
  and supported placement for every existing node. Run relevant Core/integration
  suites, Ruff, full ty, real CLI generation, strict generated Bundle validation
  and strict docs. Plan a bounded live rehearsal under the applicable explicit
  authorization; keep local and live evidence separate. Close only after the
  matrix has evidence or an explicitly tracked remaining limitation per mode.

### H3 reviewed local delivery and remaining acceptance

The final scoped review approved the implementation with no actionable findings.
Five fixed cleaners, deterministic Deduplicate and opt-in custom eligibility now
use the existing Core nodes. Raw feature replay is applied once inside the saved
model; normalized targets remain separate. The saved source is restored before
approval validates a custom recipe. Runtime guards reject changed survivor
values, schema, dtypes, keys and order, including in-place custom mutations.

Verification on 2026-09-26:

- Final combined affected suites: **289 passed**, five existing policy warnings.
- Actual CLI template generation: **56 passed**. Scoped Ruff/format, full ty,
  strict MkDocs and strict generated `dev` Bundle validation passed.
- New ordinary recipe suite: **30 passed** (11 node recipes per engine, six empty
  vectorizer cases, two executable generated custom-filter examples). The initial
  vectorizer fix plus existing vectorization suite passed 94 tests.
- Existing five-model, behavioral and temporal suites: **188 passed, 2 skipped**;
  skips require optional `h3`. Prior 30 bounded node fixtures: **60/60 passed**.
  These overlap other coverage and are not summed into a unique test total.
- New custom eligibility suite: **53 passed**; includes real fresh-process MLflow
  saved-source approval, feature-only prediction and declaration tampering.
- Real local SQLite MLflow rehearsal: both engines retained 47 of 240 source
  rows, heldout RMSE 6.877804181392917, champion v1. Only Delta reads were mocked;
  cloud dependency resolution and Delta prediction writes remain unverified here.

The [80-mode matrix](63-sm33h3-node-phase-matrix.csv) covers 62 non-model IDs / 58
calculators and separates existing tests, newly executed fixtures and limitations.
The remaining unchecked acceptance above is deliberately not a claim that every
parameter mode was run through fresh-process MLflow. Optional `H3Index` and
`sentence_embedder` packaging/execution remain open. Temporal history retrieval,
temporal CV policy, custom pre-split value normalization and broader packaging
remain SM-36a. Derived features stay ordinary preprocessing; row-dropping
inference stays guarded. Inspection artifacts work, but no separate MLflow
inspection-report UI is claimed. New Spark work remains parked.

The [bounded H3 serverless rehearsal](rehearsals/sm33h3_live/README.md) is fully
executed under the user's separate H3 approval with the verified current wheel.
All three tasks passed in 437.664 seconds; detailed evidence appears below.
No H3 commit or push has been made.

## SM-36a remaining policy work (explicitly separate)

- Existing Deduplicate integration and fixed normalization replay moved to
  SM-33H3. Group-disjoint split is parked; it is not part of that dedup task.
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


## SM-33H1 delivery evidence (2026-09-26)

Baseline temporal guards were committed as `522c6e82` with DCO and passing hooks.
The H1 implementation is included in the requested delivery commit. Initial real training tests failed before
implementation; the affected integration group then passed 202 tests. Review
found repeated engine conversions, large-integer target comparison loss and a
missing saved-cleanup manual approval test. All were fixed; scoped re-review
approved the changes. Polars now stays native across the complete filter chain.

Final verification after those fixes:

- 71 affected tests passed, including both engines, real local-MLflow approval
  replay, generated-notebook loading and post-filter stratification failure.
- The broader Bundle/runtime/CV run passed 128 tests and exposed four stale
  generated-notebook fixtures. After repairing their directory layout, the full
  notebook module passed all 16 tests (also included in the final 71 above).
- 56 real Databricks CLI generation tests passed. A newly built wheel was placed
  in a generated project's dist directory; strict dev Bundle validation passed.
- Real local WSL Spark/Delta regression passed (1 test, 14 deselected): source
  sampling precedes explicit target cleanup, with no sample refill. The Windows
  environment's missing-Delta skip is not counted as verification.
- Focused Ruff/format, repository ty and final strict MkDocs passed.

The saved training specification carries the recipe, and pre_split_filters.json
reports requested steps and per-step exclusions. Automatic/manual comparison
replays the filtered split. H2 still owns extended code/survivor integrity,
fresh-process MLflow and live acceptance. No deployment or live Databricks job
was executed in H1. Initial bounds accept numeric non-Boolean columns only.

## Full preprocessing inventory and reuse correction (2026-09-26)

The follow-up review inventoried all 94 Python files under preprocessing:
58 registered calculator classes expose 62 registration IDs, including aliases.
It inspected registrations, fit/apply entrypoints, parameter modes and execution
helpers, with deeper source review and targeted tests for the cleaning, casting,
validation, filtering, split and inference behaviors discussed with the user.
This is a capability/placement audit, not a claim that every node was retested
on every engine or on Databricks.

Earlier conversational suggestions overstated missing functionality. Reuse the
following implementations instead of adding parallel cleaning algorithms.

| Family | Existing implementations | Placement and remaining integration |
| --- | --- | --- |
| Text normalization | TextCleaning: trim, case, special characters, regex operations | Existing saved FE; not admitted to H1's row-filter-only pre-split recipe. |
| Explicit value mapping | ValueReplacement: flat/per-column mapping and to_replace/value; AliasReplacement: boolean, country, custom aliases | Sentinel and category normalization already exist. They are value transformations, not generic row predicates. |
| Invalid numeric values | InvalidValueReplacement: positive/negative infinity, negative/zero rules, configured ranges and age/percentage presets | Replacement already exists; distinguishes replacing a cell from removing its row. |
| Types | Casting: numeric, integer, Boolean, text, datetime and categorical modes; coerce_on_error | Strict/coercing modes already exist. Categorical vocabulary is learned. Casting uses mixed date parsing; it is not the Bundle's explicit timestamp/timezone policy. |
| Missingness | DropMissingRows, DropMissingColumns, MissingIndicator | H1 admits explicit row cleanup. Positive column-missingness thresholds and automatic missing-indicator selection learn from data. |
| Duplicate rows | Deduplicate: subset, first/last/none, aligned target filtering on both engines | Algorithm exists. Pre-split eligibility, deterministic keeper policy and group leakage remain separate work. H1 rejects duplicate record keys before filtering. |
| Bounds/outliers | ManualBounds, IQR, ZScore, EllipticEnvelope, Winsorize | ManualBounds is admitted in H1. Statistical methods fit after split. Winsorize clips values and does not remove rows. |
| Imputation | SimpleImputer, KNNImputer, IterativeImputer | Reuse learned train/fold state. Constant imputation is a fixed mode, but remains ordinary FE in H1. |
| Scaling | StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler | Existing learned FE, after split and inside training folds. |
| Encoding | LabelEncoder, OrdinalEncoder, OneHotEncoder, DummyEncoder, HashEncoder, TargetEncoder, WOEEncoder | Existing encoders; target encoding is not target-domain validation. Hash and target-only modes have parameter-specific leakage rules. |
| Binning/transforms | GeneralBinning, CustomBinning, KBinsDiscretizer, SimpleTransformation, GeneralTransformation, PowerTransformer | Fixed edges/formulas differ from learned edges/power parameters; reuse operation-aware admission. |
| Feature construction | FeatureGeneration/FeatureMath aliases, FeatureInteraction, PolynomialFeatures aliases | Arithmetic, ratio, similarity, datetime extraction and training-fitted group aggregates exist. No generic Boolean row-filter node was found. Group aggregates are not group-aware splitting. |
| Feature selection | VarianceThreshold, CorrelationThreshold, UnivariateSelection, ModelBasedSelection and feature_selection facade | Existing learned selection; reuse after split. |
| Time/geography | DateFeatures, LagFeatures, RollingAggregate, GeoDistance, H3Index | Existing feature generation. Temporal history/availability constraints remain explicit; the guard fixes do not provide historical data retrieval. |
| Text vectors | tokenizer, count_vectorizer, tfidf_vectorizer, hashing_vectorizer, sentence_embedder | Existing implementations; not required to add pre-split cleaning. Optional dependencies and inference portability remain node-specific. |
| Class balance | Oversampling and Undersampling, including random/SMOTE-family and cleaning methods | Already implemented. Train/fold-only; never use synthetic/resampled rows as the holdout. |
| Splitting | Split/TrainTestSplitter aliases, feature_target_split | Random/stratified split exists here; Bundle temporal selection exists outside this folder. No group-disjoint split configuration was found in the audited Core/Bundle split paths. |
| Inspection | DatasetProfile and DataSnapshot | Existing statistics/sample artifacts, not automatic quality-threshold enforcement. |
| Shared execution | base, dispatcher, pipeline, fold_adapter, schema/artifact/portable-state helpers and Spark adapters | Reuse existing fit/apply, row alignment, schema checks and fold-local fitting. Local support does not imply native Spark support. |

Related existing validation outside preprocessing also matters:

- `skyulf/profiling/expect.py` provides `expect_columns_exist`, `expect_no_nulls`,
  `expect_unique` and `expect_value_range`, with pandas/Polars handling. These
  raise validation errors; they do not remove rows. Range checks ignore missing
  values, so combine with a no-null check when missing values are forbidden.
- `skyulf/core/schema.py` supplies schema compatibility checks, including opt-in
  dtype and column-order checks. A schema check is not a full value-quality check.
- H1 already records per-step exclusions and guards key/target alignment. Those
  counts are not yet a configurable maximum-exclusion-rate or class-loss gate.

Audit candidates below are historical suggestions, not newly implemented work.
The subsequent user decision activates existing-node integration under SM-33H3;
new predicates, group splitting and data-quality thresholds are parked.

1. A declared row-predicate contract for equality/membership/AND/OR and column
   comparisons. Check null semantics, identity preservation and replay before
   admitting it; do not pretend an arbitrary Python callback is validated.
2. Reuse fixed normalization nodes before filtering only with an explicit saved
   train/inference normalization contract. Otherwise keep them in ordinary FE
   or in the source preparation shared by training and scoring.
3. Reuse Deduplicate with a reviewed duplicate policy; add group-disjoint splitting
   where the evaluation objective requires it. Repeated customer observations
   are not automatically duplicate records.
4. Connect existing expectations/counts to configurable quality gates. Extend
   only missing rules such as allowed target labels, cross-column consistency
   and maximum exclusion fraction. These are not all preprocessing nodes.
5. Feature availability and join cardinality belong to source/temporal contracts;
   neither a splitter nor a fixed-transformation flag proves those properties.

The updated sequence is SM-33H2 -> SM-33H3 -> SM-34. SM-36a retains the separate
scope above. This review does not open every existing node to pre-split use or
modify runtime code.

Fresh verification: 1,448 focused cleaning/casting/missingness/leakage/schema/
expectation/Bundle tests passed; 147 additional value-replacement, inspection,
split and resampling tests passed (1,595 total). One joblib Windows physical-CPU
detection warning fell back to logical cores. No Databricks execution occurred.
Logs: `.cache/preprocessing-audit-20260926.log` and
`.cache/preprocessing-audit-extra-20260926.log`.

## SM-33H2 live acceptance evidence (2026-09-26)

The user explicitly approved this exact personal rehearsal. [Run 926706369150614](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/886322042039901/run/926706369150614)
finished SUCCESS in 11 minutes 42 seconds. All three tasks succeeded:
`pandas_train` (967082093196374), `polars_train` (1090464827962261), and
`score` (947036177005319). Exactly one run was submitted; no retry was needed.

Wheel SHA-256: `d5982ff99ebfc8e9e5e572be05ae1b60a3fd88d974084943cf8336945cba186d`.
Uploaded notebook SHA-256: `3d4e5394004d1a234cc757a869f3875bdc792857ea9517e4222d5e054ccbb854`.
Workspace folder: `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_lifecycle_test/sm33h2/r1`.
The MLflow experiment is the `experiment` child of that folder. Local raw task
responses and a checked summary are retained in
`initiatives/spark_and_mlflow/rehearsals/sm33h2_live/` (ignored rehearsal artifacts).

### Training, saved code and lifecycle

Both engines ran Core DropMissingRows(target) and ManualBounds(age) before the
split, then SimpleImputer and StandardScaler with LinearRegression and three-fold
CV. v1 retained 160 eligible rows (120 train, 40 holdout); v2/v3 retained 213
(159 train, 54 holdout). The candidate and champion comparison used the
candidate's evaluation population. Changing the current Python recipe to raise
after training did not prevent approval from loading the recorded recipe.

| Engine | Version | Heldout RMSE | CV RMSE mean | MLflow training run |
|---|---|---|---|---|
| pandas | 1 | 33.954801 | 34.129708 | `2a7980c5e3b84864a798d81aa329a682` |
| pandas | 2 | 1.979446 | 2.453322 | `f868ad693d4a4b7faf480ee7a3788254` |
| pandas | 3 | 34.576033 | 34.286389 | `d69f68cba0d04b62b66a3b08c9a46b0b` |
| polars | 1 | 33.954801 | 34.129708 | `7389310615cf45b7a25718f63c71cad9` |
| polars | 2 | 1.979446 | 2.453322 | `98275bbf7d404d9e872ca664e558cf2d` |
| polars | 3 | 34.576033 | 34.286389 | `aa58e3d710fd4d6db5f2bcc8abfeb079` |

For each model, v1 became the initial champion, v2 was approved and promoted,
v3 was explicitly rejected, and rollback restored v1. Final champion v1 is
intentional test state. The weak no-intercept model is deliberate lifecycle
test data, not a recommended production model or a predictive-quality benchmark.

### Prediction publication

| Engine | Initial predictions | New predictions after append | Final rows | Next run | Delta version after append / no-op |
|---|---:|---:|---:|---|---|
| pandas | 240 | 3 | 243 | noop=true | 2 / 2 |
| Polars | 240 | 3 | 243 | noop=true | 2 / 2 |

The separately started score task loaded the registered artifacts afresh.
Local artifact and MLflow pyfunc predictions agreed on feature-only input with
no target or age filter column. Published keys were exactly 0..239 initially
and 0..242 after append. Every prediction was finite and used model version 1.
All original 240 key/prediction/version triples stayed unchanged after append;
the complete output and its Delta version stayed unchanged after the no-op.
Thus rows excluded by training eligibility still received predictions.

Retained resources under `workspace.skyulf_lifecycle_test`:

- Source: `sm33h2_20260926_r1_source` (243 synthetic rows).
- Outputs: `sm33h2_20260926_r1_pandas_predictions` and
  `sm33h2_20260926_r1_polars_predictions` (243 rows each).
- Models: `sm33h2_20260926_r1_pandas_model` and
  `sm33h2_20260926_r1_polars_model` (three versions each; champion v1).

This verifies bounded local pandas/Polars training and inference with Spark/Delta
I/O on personal serverless compute. Intentional evidence corruption is covered
by local tests, not by cloud artifact mutation. H3 all-node/custom-pre-split
integration, company production permissions and wider concurrent-writer testing
remain their own tasks. SM-33H2 is DONE; SM-33H3 is READY before SM-34.

## SM-33H3 live acceptance evidence (2026-09-26)

The explicitly approved [run 848857785722024](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/743740415302976/run/848857785722024)
completed **SUCCESS in 437.664 seconds**. Exactly one serverless submission was
made, with no retry, new schema or persistent job. All three tasks succeeded:

| Task | Task run ID | Result |
|---|---|---|
| pandas_train | 1078163679226229 | SUCCESS |
| polars_train | 179579493001815 | SUCCESS |
| score | 679007598316858 | SUCCESS |

Both engines used the existing Core nodes and the custom project filter. Cleanup
retained 47 of 240 rows: 15 missing targets, 10 invalid ages, 12 custom exclusions
and 156 duplicate observations were removed. The final split contained 35
training and 12 heldout rows. The two runs logged the same 30 heldout/CV metrics:

| Metric | pandas | Polars |
|---|---:|---:|
| Heldout MAE | 4.903793942628702 | 4.903793942628702 |
| Heldout RMSE | 6.877804181392917 | 6.877804181392917 |
| Heldout R² | 0.5494839013563249 | 0.5494839013563249 |
| CV RMSE mean | 4.4513347280751 | 4.4513347280751 |

MLflow training runs: pandas `c0b16dd97d4145578773cb65d2602808`, Polars
`f258dfcaf4f446d68d81ad11455790a1`. The source Python file was replaced after
training. A separate task restored the saved code, validated the saved recipe
and evidence, and approved champion v1 for both models. Target/filter columns
were absent from direct saved-model and MLflow prediction inputs; their
predictions matched. Non-idempotent feature replacement was checked explicitly
as `1 -> 2`, `2 -> 3`, with no second application.

Each score output first contained exactly keys 0..239, including rows excluded
from training. After appending three source records, only those three were
scored and each output contained exactly keys 0..242. The original 240
key/prediction/model-version triples remained unchanged. Another score returned
zero input/output rows and `noop=true`; Delta version stayed **2** and exact
published contents remained unchanged. Every prediction was finite and used v1.

Resources retained in `workspace.skyulf_lifecycle_test`:

- `sm33h3_20260926_r1_source`: 243 rows.
- `sm33h3_20260926_r1_pandas_predictions`: 243 predictions.
- `sm33h3_20260926_r1_polars_predictions`: 243 predictions.
- `sm33h3_20260926_r1_pandas_model` and `sm33h3_20260926_r1_polars_model`:
  one version each, champion v1.
- Notebook, wheel and experiment under
  `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_lifecycle_test/sm33h3/r1`.

Wheel SHA-256: `ad08f6d91e3f329eb03294e36ed7d2004019fe14bfcca13cbda51aa196369c24`.
CLI upload used the locally verified wheel containing all 248 current Core
Python files. Raw task outputs and `acceptance-summary.json` are retained in
`rehearsals/sm33h3_live`; `verify_results.py` passed against those actual outputs.

This proves the documented mixed recipe and batch workflow on personal
serverless, not every parameter mode or company production environment.
Optional H3Index/sentence-model execution, temporal history/CV policy and custom
pre-split value normalization remain explicitly tracked under SM-36a and
matrix63. Serving and native Spark expansion are unchanged. **SM-33H3 is DONE
for this scope; SM-34 is READY.** No commit or push was made.
