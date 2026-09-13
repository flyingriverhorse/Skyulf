# Opus core audit — open fix queue

The live half of the Opus core audit: every finding still open, with the
reproduction evidence needed to fix it. Closed findings, the corrections pass and
the whole fix Log stay in [the archive](opus_core_analysis-tracker.md).

**Split 2026-09-06:** 100 open rows moved here; the archive keeps its 63 closed
rows in the same tier and domain grouping, so a finding stays where it was filed.
When one is fixed, move its row back to the archive with a one-sentence status
note and write the Log entry there — the Log is one unbroken stream, and this
file deliberately carries no history.

**Source audit:** [`opus_core_analysis.md`](opus_core_analysis.md) (master report)
+ [`opus_core_analysis/README.md`](opus_core_analysis/README.md) (index of the 19
per-area report files `00`–`18`).
**Baseline:** commit `93d7719e` (master), audit run 2026-08-31 → 09-01 by 15
parallel read-only agents (Claude Opus 5). 116 findings: 5 🔴 / 45 🟠 / 44 🟡 /
22 ⚪, plus OC-160–320 filed by later reviews. OC-100 was retracted as a false
positive and is not counted; the corrections pass stays in the archive.

**Status key:** ⬜ open · 🟨 in progress · ✅ done · ⏭️ parked

Original severity and effort are the audit's own; Qwen follow-up ratings
reflect the verified scope, not the original scanner severity. A status cell
retains measured evidence and limitations; repair verification lives in the
archive Log.

---

## Live — fix queue

**Current status (2026-09-13): 13 open / 4 parked.**
The latest fixes close OC-11/47/80/255/286/289. Verification and
limitations are recorded in the [archive Log](opus_core_analysis-tracker.md#log).
The earlier batch reports were removed by the user; their closure summaries
remain in that Log.
The verified Qwen follow-up added **48 findings, OC-271–318**,
grouped below by priority and domain. Qwen #1/#50/#57 reuse OC-253/65/64;
#13 is already fixed as OC-268. #18/#41 share OC-286, now closed with both
profile and job-result persistence covered. Policy-only #36 is recorded separately
below, outside the defect count. Reproduction and filing history are in
the [archive Log](opus_core_analysis-tracker.md#log).

Ordered by the master report's suggested fix order: **Now** (silent wrongness
reaching users), **Next** (wrong results in realistic configs), **Then** (decide
deployment model), **Ongoing** (remove the hiding conditions). Remaining findings
follow, grouped by domain.

The original Now/Next/Ongoing rows and their completed verification history
remain in the archive. The Qwen follow-up adds new Now and Next work below;
remaining findings are grouped by domain.

### Now — silent wrongness reaching users

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-281 | 🟠 | **Prediction-time row filtering loses the input-to-prediction contract** (`skyulf-core/skyulf/preprocessing/pipeline.py:44,327`) — Qwen #12. Define and enforce inference row retention or explicit row provenance through Core and deployment serialization; keep fold-scoring OC-251 separate. | medium | ⬜ open — An IQR pipeline returns two predictions for [2,1000,4]; pandas retains indices [0,2], Polars returns [0,1], and serving serializes values without indices; Winsorize preserves all three rows. |


### Next — wrong results in realistic configs

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-277 | 🟠 | **Concurrent submissions bypass duplicate-job protection** (`backend/ml_pipeline/_execution/jobs.py:104-119`) — Qwen #8. Make job reservation atomic across API processes and preserve one lock for an in-process key while waiters exist; coordinate lock cleanup with OC-310. | medium | ⬜ open — Two controlled OS processes create two queued jobs with versions 1/2 after the same absent-row check, and waiting coroutines reach two simultaneous entries for one key; one request alone does not duplicate a job and PostgreSQL was not exercised. |
| OC-280 | 🟡 | **Configured default rate limits are not applied to undecorated routes** (`backend/middleware/rate_limiter.py:10-19`) — Qwen #11. Wire default limiting into the actual app and retain explicit per-route limits; authentication decisions remain separately parked. | medium | ⬜ open — An undecorated mutation route accepts 230/230 requests while an explicit-limit control returns 429 after 60; the current inventory has 34 mutation routes, eight decorated and 26 without default enforcement. |


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
| OC-310 | 🟡 | **Failed job submission leaks entries in the per-key lock registry** (`backend/ml_pipeline/_internal/_routers/run_pipeline.py:190-212`) — Qwen #46. Validate job types and clean up reservation failures without evicting locks still owned or awaited; coordinate with OC-277. | small | ⬜ open — Four actual run requests with schema-accepted unknown job types return 500 and leave four unlocked registry entries; the UI does not submit those values and the route has an explicit 20/min rate limit. |


### Remaining — direct-audit modules

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-91 | 🟡 | Three public `core/` seams (263 lines) have zero call sites; one duplicates a differently-shaped backend class name | small | ⬜ open |

### Remaining — file-coverage closure

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|

### Remaining — cross-cutting & packaging

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-06 | 🟡 | 6 registered nodes unreachable from the UI (incl. all of `geo/`) — `registry.py` vs `frontend/` | small | ⬜ open — R1 step 3 catches this class |
| OC-07 | 🟡 | Node-id naming split 55 PascalCase / 45 snake_case + redundant aliases (`registry.py`) | half day | ⬜ open |
| OC-08 | 🟡 | Public-API name collision: `DatasetProfile` means two things (`skyulf/__init__.py:32-46`) | small | ⬜ open |
| OC-10 | ⚪ | 4 dead `infer_output_schema` overrides that only `return None` (`vectorization/*`) | mechanical | ⬜ open — **re-measured 2026-09-06: five, not four** (`count_vectorizer.py:147`, `hashing_vectorizer.py:135`, `tfidf_vectorizer.py:141`, `tokenizer.py:179`, `sentence_embedder.py:204`). `BaseCalculator.infer_output_schema` already ends in `return None` (`preprocessing/base.py:129`), so all five are behaviourally identical to inheriting. **Recommend folding into OC-03 rather than deleting standalone:** each override carries the per-node *reason* the schema is unknowable (learned vocabulary, model-loaded embedding width, data-dependent column survival), which is exactly the documentation OC-03's parametrized "predicted == actual for every node" test needs beside it, and OC-03 will touch these same five files |

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
| OC-64 | 🟠 | **F-14 only partially fixed** — engine registry global still an unlocked race (`engines/registry.py:60,86-91`) | small | ⬜ open Qwen recheck: ordered threads reproduce the shared-default race, but no production set_active_engine call was found; current live-request impact is unproven. Qwen #57. |
| OC-65 | 🟡 | polars `to_numpy()` zero-width "parity fix" does not achieve parity (`engines/polars_engine.py`) | small | ⬜ open Qwen recheck: a three-row, zero-column selection reaches conversion as pandas (3,0) versus Polars (0,0), so Polars has already lost its height. Qwen #50. |


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
| OC-54 | 🟡 | `DebugNode` is dead code that would silently no-op if wired up (`nodes/DebugNode.tsx`) | small | ⬜ open |
| OC-57 | ⚪ | `any`-typed chart props bypass type safety in EDA components (`modules/eda/`) | small | ⬜ open |


### Remaining — tests / packaging / CI (outside the Ongoing tier)

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|

### Qwen review — decisions outside the defect count

- **#36, PCA infinity policy:** the current explicit/tested conversion maps
  infinities to zero and NaN to the column mean. The prepared-matrix PCA is
  mathematically consistent, but the choice is not explained to users. Decide
  whether to retain and document it or change it with compatibility coverage;
  this is not a confirmed PCA calculation defect or a parked OC record.
- **#20/#35/#56:** no standalone runtime defect was established. Histogram
  counts match the right-closed convention; the inline-dispatch regex gap is an
  optional code-standard/test improvement; scalar dtype hashing needs a concrete
  supported-model counterexample or an explicit contract before filing a bug.
- **#13:** remains closed as OC-268. The four user-parked authentication/config
  findings OC-71/72/73/185 retain their previous status.

The standalone Qwen files were removed by the user after their actionable
content was transferred here. Claim numbers remain beside the evidence in
each OC row; fixed rows and their verification move to the archive. The
original disposition covers all 57 claims: 48 new records (including merged
#18/#41), three existing open matches, one already fixed claim, one policy
decision and three claims not established as runtime defects.

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

Findings filed before 2026-09-05 — OC-169 and OC-178–182 among them — keep
their reproduction detail in the archive's `## Log` entries instead.

### 2026-09-06 — remaining-source continuation (findings added as verified)

All entries below have executed reproduction evidence. Source paths are
relative to `skyulf-core/skyulf/`; line numbers refer to the source read during
the review and may move with concurrent edits. The main reviewer independently
reproduced the filed symptoms before completing the source ledger.

Each finding's queue row — severity, effort, status — now lives in the
**Live — fix queue** above, under the domain table it belongs to. What follows
here is the reproduction detail those rows deliberately do not repeat.

**OC-195 — wrapped pandas clustering loses numeric filtering.** Executed
`KMeansCalculator().fit(X,None,{'n_clusters':2})` for pandas
`x=[0.,.1,.2,10.,10.1,10.2]`, `text=['name']*6`: it fits one feature.
Passing `SkyulfPandasWrapper(X)` instead raises
`ValueError: could not convert string to float: 'name'`.
**Fix/verification target:** recognize both supported wrappers through the
public adapter interface and preserve the raw-frame behavior for every
clustering calculator/applier sharing the helper.

**OC-196 — GaussianMixture fit/predict/probability feature mismatch.** Fit
`GaussianMixtureCalculator` on pandas
`x=[0.,.1,.2,10.,10.1,10.2]`, `ref=[0,0,0,1,1,1]`, with
`reference_column='ref', n_components=2`. On that same frame, the public
applier's `predict` succeeds, while `predict_proba` raises
`X has 2 features, but GaussianMixture is expecting 1 features as input`.
**Fix/verification target:** share fitted feature selection between prediction
methods; verify probabilities also work with excluded text/reference columns.

**OC-197 — reference-crosstab internal names collide.** Executed
`_compute_reference_crosstab_polars` with labels `[0,0,1,1]` and reference
values `['a','b','a','b']`: a Series named `species` yields the expected four
counts. Naming it `count` raises `DuplicateError`; naming it
`__skyulf_cluster__` raises a duplicate-group-key error. The review also
reproduced `count` through public clustering evaluation. **Fix/verification
target:** choose independent collision-safe names for cluster, reference and
count columns. OC-161 concerns centroid features; this is the separate
reference-label aggregation path.
