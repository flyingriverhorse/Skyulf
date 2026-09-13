# Opus core audit — open fix queue

**Source audit:** [`opus_core_analysis.md`](opus_core_analysis.md) (master report)
+ [`opus_core_analysis/README.md`](opus_core_analysis/README.md) (index of the 19
per-area report files `00`–`18`).
**Baseline:** commit `93d7719e` (master), audit run 2026-08-31 → 09-01 by 15
parallel read-only agents (Claude Opus 5). 116 findings: 5 🔴 / 45 🟠 / 44 🟡 /
22 ⚪, plus OC-160–320 filed by later reviews. OC-100 was retracted as a false
positive and is not counted; the corrections pass stays in the archive.

**Status key:** ⬜ open · 🟨 in progress · ✅ done · ⏭️ parked

Original severity and effort are the audit's own; follow-up ratings
reflect the verified scope, not the original scanner severity. A status cell
retains measured evidence and limitations; repair verification lives in the
archive Log.

---

## Live — fix queue

**Current status (2026-09-13): 1 open / 4 parked.**
OC-07/08/10/91/280 are resolved. The latest OC-06 work exposes Manual Bounds
and Geo Distance in Canvas. H3 is deferred by user choice; intermediate-stage
profile inspection remains the unresolved scope of OC-06.
Verification and limitations are recorded in the
[archive Log](opus_core_analysis-tracker.md#log).

Completed findings and their verification history remain in the archive.

### Now — silent wrongness reaching users

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|


### Next — wrong results in realistic configs

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|


### Then — decide deployment model first

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-71 | 🟠 | **No authentication or authorization anywhere on the API** (`main.py:373-395`, `database/models.py:151-159`) — **confirm intent first**: single-tenant self-hosted → documentation task; multi-tenant → highest-priority item in the entire report (scaffolded `User` model + dead `AUTH_FALLBACK_*` settings suggest the latter was intended) | decision + ~1 week | ⏭️ parked — user requested pause | PARKED!
| OC-72 | 🟡 | Insecure-by-default config: unset `FASTAPI_ENV` fails open to wildcard CORS + credentials (`config/factory.py:26`, `main.py:359-366`) | small | ⏭️ parked — with OC-71 |
| OC-73 | ⚪ | `DataSource.credentials` documented encrypted, stored plaintext JSON (`database/models.py:107`) | small | ⏭️ parked — with OC-71 |

### Remaining — evaluation & explainability

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|

### Remaining — backend infrastructure

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-185 | 🟡 | Authorization is stubbed in three mutually inconsistent pieces. `database/models.py:157 has_permission` is `return True  # Placeholder` with **zero callers**; `data_ingestion/dependencies.py:26,31 require_data_access`/`require_data_admin` are async no-ops wired to no route; and `data_ingestion/router.py:148,169` hardcode `user_id = 1` under an explicit `# KNOWN-GAP: Auth not implemented yet`, so every source belongs to one user and is visible to everyone. Nothing is exploitable *through* `has_permission` today precisely because nothing calls it — the risk is that the first caller gets an always-yes check shaped like a real API. Needs an authz decision before code | decision + ~1 week | ⏭️ parked — user requested pause |


### Remaining — direct-audit modules

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|

### Remaining — file-coverage closure

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|

### Remaining — cross-cutting & packaging

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-06 | 🟡 | **Intermediate-stage profile inspection needs an explicit Canvas scope.** | medium | ⬜ open — ManualBounds and GeoDistance added and tested on 2026-09-13; user explicitly deferred H3 in Canvas. Clustering, feature generation and custom binning were already reachable through existing nodes. Data Preview covers snapshot inspection, but source EDA is not an equivalent arbitrary intermediate-stage profile artifact. Remaining work is this inspection scope; R1 tracks future registry/UI drift separately. |

### Remaining — encoding / cleaning / imputation / scaling / drop / resampling

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|


### Remaining — feature generation / selection / vectorization / transformations

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|


### Remaining — profiling (outside the OC-39–46 cluster)

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|


### Remaining — core / engines / pipeline

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|


### Remaining — outliers / casting / binning / timeseries / geo

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|


### Remaining — frontend

The frontend CCN inventory recorded **0 functions in 0 files above CCN 10**;
its separate report was removed. The source-wide strict gate still passes.
The informational report stays at 8, listing 133 optional CCN 9/10 functions in
96 files. Complexity cleanup is complete under the accepted limit; the functional
findings below remain open and their audit counts are unchanged.

The [scanner follow-up](frontend_static_analysis_review_2026-09-10.md) replaces
the four reported dynamic regex/object-lookup patterns and explains file-level
complexity deltas. Local verification passes; external scanner confirmation is
pending. No new application exploit was established, so OC counts are unchanged.
The same report records the three job-log regex performance fixes, preserved
highlighting and the passing 100,000-digit browser check; no OC row was closed.
The subsequent Codacy artifact cleanup deletes the seven accidentally committed
local verification files and excludes their directory from future scans.
The separately reported lockfile issue still needs its exact advisory details.

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|


### Remaining — tests / packaging / CI (outside the Ongoing tier)

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|

### Pending policy decision — PCA non-finite inputs

The current explicit/tested conversion maps infinities to zero and NaN to the
column mean. The prepared-matrix PCA is mathematically consistent, but the choice
is not explained to users. Decide whether to retain and document it or change it
with compatibility coverage. This is outside the OC defect count and is not a
confirmed PCA calculation defect or a parked OC record.

---

## R1 — systemic fix: the hand-duplicated core↔frontend contract

Retires 8 findings as a class (OC-13, OC-14, OC-15, OC-19, OC-20, OC-53, OC-61, OC-66
plus the OC-06-class gaps). Doing R1 first is tempting but leaves users on broken
behaviour longer — the master report sequences it after the individual no-ops.

| Step | Work | Status |
|---|---|---|
| 1 | `@node_meta` as single source of truth — the `choices` tuple must be the *same object* the implementation branches on | ⬜ open |
| 2 | Emit `node-contract.json` + generated `nodeContract.ts` (literal-union types, defaults, choices, labels, help text); commit the generated file; CI no-diff check (lockfile pattern) | ⬜ open |
| 3 | Drift fails loudly at every layer — TS: node components import union types from `nodeContract.ts` (compile-time); `pipelineConverter.ts`: validate `node.data` against the contract (canvas-time error); backend: reject unknown param keys + out-of-choices values (any client); CI: assert every registry id is in the contract or on an explicit `INTENTIONALLY_HEADLESS` allow-list (catches OC-06-class gaps) | ⬜ open |
| 4 | Generate tooltip/help metadata too (OC-61, DATE_METHOD_META are metadata drift) | ⬜ open |

**Sequencing:** steps 1–2 are additive — land without touching any node. Step 3's
backend strictness goes behind a warn-only flag for one release (log every rejected
key — also the fastest way to find drift the audit missed).

---

## Planned enhancement — DRIFT-01: raw-data and model-feature drift modes

**Status:** ⬜ open — specification recorded at the user's request on 2026-09-10;
deferred for a later implementation pass. This is a cross-layer enhancement,
not a newly reproduced OC finding. At filing, the audit contained **63 open /
4 parked** findings; this enhancement did not change those counts. OC-224 remains
fixed; OC-223 is a separate alert-note lifecycle bug.

**Goal:** let the user upload one raw dataset and choose which representation
to monitor. Both modes must compare the same preprocessing stage on each side.

| Mode | Reference | Current data | Question answered |
|---|---|---|---|
| Raw data | Saved raw loader snapshot | Uploaded raw dataset | Has the incoming source distribution/schema changed? |
| Model features | The selected raw reference passed through the frozen prediction preprocessing path | The raw upload passed through that same path | Have the features supplied to the model at prediction time changed? |

**Current state:** OC-224 implements raw-source comparison using the selected
model's unique saved loader snapshot. `DriftCalculator` itself is unchanged.
The model-feature mode, mode selector and persisted mode/provenance fields
described below do not exist yet.

### Reference population and compatibility

- Start with the same saved raw reference rows in both modes. For the reported
  Iris job this is 150 source rows; the model card still reports its 120 training
  rows. Preserve sampling if the loader originally sampled the source.
- Keep representation (raw/model features) separate from population (loaded
  source/training subset). A later training-only baseline requires reliable raw
  training-row provenance. Do not reconstruct it from row counts, reset indexes
  or split seeds after arbitrary filtering, reordering or branch merges.
- Build the model-feature reference by applying the frozen prediction transform
  to those reference rows. Do not reuse the old splitter snapshot, an arbitrary
  intermediate frame or a SMOTE/resampled training matrix as an equivalent
  reference. Record any legitimate row-count change and its reason.
- Preserve existing raw-mode behavior and defaults. Existing jobs may enable
  feature mode only when their source and fitted preprocessing artifacts are
  sufficient and their replay is verified. Otherwise report the mode unavailable
  with a clear reason; require retraining only when the required state is absent.
- Historical checks must retain their original meaning. Pre-OC-224 checks may
  have compared different stages, and graphless legacy baselines are unverified;
  do not silently label those results as a verified raw or feature mode.

### Implementation work

- [ ] **Core: expose shared prediction-feature preparation.** Add a public,
  in-memory API for obtaining the features actually passed to the estimator;
  its name is to be decided. Reuse it for prediction and feature drift so target
  handling, input validation, transformations, column order and dtype handling
  cannot diverge. Keep `DriftCalculator(reference, current)` backward-compatible;
  no database, filesystem or backend imports in `skyulf-core`.
- [ ] **Replay: verify fitted pipelines and branch semantics.** Reuse training's
  fitted scaler/encoder/selector state; never call `fit` on uploaded data.
  Preserve prediction-time handling of splitters, resampling and row-dropping
  operations. `_build_composite_feature_engineer` currently concatenates ancestor
  steps; this alone does not prove exact replay of parallel branches or merges.
  Verify equivalence before enabling that graph shape; unsupported shapes must
  fail explicitly. Define column exclusions, including raw inputs used to create
  derived features before being dropped; the current explicit-drop convention
  is not complete feature lineage.
- [ ] **Backend artifacts: save both references and their identity.** Persist
  raw and prepared-feature snapshots, representation/population, row counts,
  ordered schema and model/pipeline identity or fingerprint. These are proposed
  metadata, not current artifact fields. Rebuild/invalidate the feature reference
  when the fitted pipeline changes; never silently reuse another model's state.
- [ ] **API and persistence: carry the selected mode end to end.** Extend drift
  requests, responses, saved results and alert details with mode and reference
  provenance. Scope history/trends and configurable thresholds by the appropriate
  mode/reference so different representations are not compared as one series.
  Add intentional migrations and a legacy/unknown interpretation for old rows.
  Record parsing, preprocessing and missing-reference failures as failed checks,
  not zero-drift successes; preserve existing alert lifecycle behavior.
- [ ] **Frontend: add Raw data / Model features selection.** Accept a raw upload
  in either mode, show which reference/population and row counts are in use, and
  explain unavailable modes. Preserve the selected mode in results, history,
  investigation links and CSV exports; do not require manual preprocessing.
- [ ] **Documentation and delivery:** add backend/UI and standalone Core examples,
  describe baseline selection and legacy limitations, and record concise release
  notes when implemented. Update this item and the tracker after verified work.

### Acceptance tests

- [ ] Identical reference/upload rows yield no drift in both modes for supported
  deterministic pipelines, including log transforms and fitted scaling/encoding.
  Do not require exact zero when comparing different sampled populations.
- [ ] Feature-mode preparation matches the matrix supplied by real prediction,
  including names, order, values and dtypes; check linear and supported branched
  graphs, fitted feature selection, unseen categories and generated features.
- [ ] New uploads do not refit/mutate preprocessing, resplit data or synthesize
  training rows. A large real shift remains detectable with the frozen transform;
  missing/new raw columns and transformation failures remain visible.
- [ ] Mode switching uses the correct reference, thresholds, history and export;
  old responses cannot replace results for a newly selected mode/reference.
- [ ] Retraining/pipeline changes cannot reuse stale feature references. Cover
  old jobs, missing artifacts, graphless metadata, ambiguous sources, sampled
  loaders and unsupported branch replay with explicit expected outcomes.
- [ ] Standalone Core users can obtain prediction features and run both drift
  comparisons without the backend. Existing Core calculator callers still work.

**Starting points:** `backend/monitoring/drift_reference.py`,
`backend/monitoring/router.py`, `backend/database/models.py`,
`backend/ml_pipeline/_execution/engine/_feature_eng.py`,
`backend/ml_pipeline/_execution/engine/_artifacts.py`,
`backend/ml_pipeline/deployment/service.py`,
`skyulf-core/skyulf/pipeline/_pipeline.py`,
`skyulf-core/skyulf/preprocessing/pipeline.py`,
`skyulf-core/skyulf/profiling/drift.py`,
`frontend/ml-canvas/src/core/api/monitoring.ts`,
`frontend/ml-canvas/src/pages/DataDriftPage.tsx` and `src/pages/drift/`.
Relevant existing regressions: `tests/integration/test_drift_reference_space.py`
and `tests/integration/test_drift_target_columns.py`.

---

## Reproduction evidence — open findings

Each block is the executed reproduction behind the row above it: the input, the
observed behaviour, and the fix/verification target. Source paths are relative
to `skyulf-core/skyulf/` unless written out, and line numbers refer to the
source read when the finding was filed, so they may have moved.
