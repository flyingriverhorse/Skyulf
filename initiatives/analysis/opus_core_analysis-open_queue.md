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
22 ⚪, plus OC-160–229 filed by later reviews. OC-100 was retracted as a false
positive and is not counted; the corrections pass stays in the archive.

**Status key:** ⬜ open · 🟨 in progress · ✅ done · ⏭️ parked

Severity and effort are the audit's own. A status cell is one sentence; the
detail lives in the archive's `## Log` section.

---

## Live — fix queue

Ordered by the master report's suggested fix order: **Now** (silent wrongness
reaching users), **Next** (wrong results in realistic configs), **Then** (decide
deployment model), **Ongoing** (remove the hiding conditions). Remaining findings
follow, grouped by domain.

The **Now**, **Next** and **Ongoing** tiers have no open findings left — those rows
are in the archive. **Next** closed its last two filed rows (OC-177, OC-164) on
2026-09-06, immediately re-opened with OC-207 filed *while* fixing OC-164, and
closed that on 2026-09-07.

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
| OC-110 | 🟠 | Semantic-type inference misclassifies small categorical columns as `Text`, so task type never inferred (`profiling/_analyzer/column.py`, `analyzer.py:502`) | small | ⬜ open |
| OC-113 | 🟠 | Near-perfect multicollinearity silently reports VIF = 1.0 — `max(1.0, …)` clamps numerical garbage (`numeric.py:32-63`) | small | ⬜ open |
| OC-120 | 🟠 | `Decimal` columns silently skipped by every auto-numeric node; crash pandas when selected explicitly (`engines/__init__.py`, `preprocessing/_helpers.py`) | small | ⬜ open |
| OC-91 | 🟡 | Three public `core/` seams (263 lines) have zero call sites; one duplicates a differently-shaped backend class name | small | ⬜ open |
| OC-101 | 🟡 | `calibrated_classifier`'s `random_state` no-op for two independent reasons (estimator rejects it AND factories hardcode the seed) | small | ⬜ open |
| OC-111 | 🟡 | A profiling recommendation branch is unreachable | small | ⬜ open |
| OC-102 | ⚪ | Five tunable models return an empty search space from the live `/defaults` endpoint (`hyperparameters/_registry.py`) | small | ⬜ open |
| OC-121 | ⚪ | polars `Enum` columns invisible to text auto-detection, diverging from pandas `Categorical` (`_helpers.py:148-157`) | small | ⬜ open |
| OC-90 | ⚪ | Unknown split config keys silently dropped instead of rejected (`preprocessing/split.py`) | small | ⬜ open |

### Remaining — file-coverage closure

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-140 | 🟠 | `InvalidValueReplacement` diverges across engines on non-numeric columns (pandas silently NaNs, polars raises) | small | ⬜ open |
| OC-142 | 🟠 | EDA correlation ratio η exceeds 1.0 with nulls; null-heavy columns rank as strongest association | small | ⬜ open |
| OC-144 | ⚪ | Geo distance column named `_km` even when the unit is miles | ~~1 line~~ **small, not 1 line** — scoped 2026-09-06 | ⬜ open — **not a one-liner; blast radius measured.** Four code sites (`geo/distance.py:83` pandas apply, `:112` polars apply, `:163` `node_meta` default, `:185` `fit`), **10** assertions in `tests/integration/test_geo_nodes.py` (incl. `:91`, which reads `result_km["geo_distance_km"]` while converting to miles — the mislabel the finding describes, baked into a test), and `docs/reference/preprocessing_nodes.md:630`. **The structural detail that decides the fix:** the two apply-path fallbacks are unreachable in the normal pipeline, because `fit` always writes `output_column` into the artifact — so the *declared* `node_meta` default is what really picks the name. `node_meta` params are a static dict and cannot be unit-dependent, so `f"geo_distance_{unit}"` has to be resolved in `fit` (declaring `""` = auto, or dropping the key), not patched at the four sites independently. Frontend impact is nil — all of `geo/` is UI-unreachable per OC-06 |

### Remaining — cross-cutting & packaging

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-06 | 🟡 | 6 registered nodes unreachable from the UI (incl. all of `geo/`) — `registry.py` vs `frontend/` | small | ⬜ open — R1 step 3 catches this class |
| OC-07 | 🟡 | Node-id naming split 55 PascalCase / 45 snake_case + redundant aliases (`registry.py`) | half day | ⬜ open |
| OC-08 | 🟡 | Public-API name collision: `DatasetProfile` means two things (`skyulf/__init__.py:32-46`) | small | ⬜ open |
| OC-10 | ⚪ | 4 dead `infer_output_schema` overrides that only `return None` (`vectorization/*`) | mechanical | ⬜ open — **re-measured 2026-09-06: five, not four** (`count_vectorizer.py:147`, `hashing_vectorizer.py:135`, `tfidf_vectorizer.py:141`, `tokenizer.py:179`, `sentence_embedder.py:204`). `BaseCalculator.infer_output_schema` already ends in `return None` (`preprocessing/base.py:129`), so all five are behaviourally identical to inheriting. **Recommend folding into OC-03 rather than deleting standalone:** each override carries the per-node *reason* the schema is unknowable (learned vocabulary, model-loaded embedding width, data-dependent column survival), which is exactly the documentation OC-03's parametrized "predicted == actual for every node" test needs beside it, and OC-03 will touch these same five files |
| OC-11 | ⚪ | Mega smoke test silently skips nodes with empty params (`tests/unit/test_all_nodes_smoke.py`) | small | ⬜ open |

### Remaining — encoding / cleaning / imputation / scaling / drop

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-18 | 🟡 | One-hot/dummy generated names can collide with existing columns (`encoding/one_hot.py:68-92`, `dummy.py:76-99`) | small | ⬜ open |

### Remaining — feature generation / selection / vectorization / transformations

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-23 | 🟠 | Polars `ratio` flips the sign of near-zero negative denominators (`feature_generation/_polars_ops.py:97-112`) | small | ⬜ open |
| OC-24 | 🟠 | Polars group aggregates treat null group keys differently from pandas (`_polars_ops.py:222-234`) | small | ⬜ open |
| OC-29 | 🟡 | `FeatureGeneration` advertises `polynomial` but silently skips it (`feature_generation/_common.py:24-31`) | small | ⬜ open |
| OC-30 | 🟡 | Datetime extraction ignores the UI output name, overwrites collisions (`_pandas_ops.py:173-184`) | small | ⬜ open |

### Remaining — profiling (outside the OC-39–46 cluster)

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-191 | 🟡 | All-null and Polars Enum columns are classified as text and sent to string-only aggregates, aborting the whole profile (`profiling/analyzer.py`, `_analyzer/column.py`) | small | ⬜ open |
| OC-192 | 🟡 | Decomposition's categorical null bucket displays as `Unknown`, but drilling into it filters for the literal string and silently loses the bucket's rows (`profiling/_analyzer/decomposition.py:71-76`) | small | ⬜ open |
| OC-199 | 🟡 | Explicit latitude/longitude selections bypass `exclude_cols`, returning coordinates for columns excluded from the profile (`profiling/_analyzer/geo.py:58-59`) | small | ⬜ open |
| OC-47 | 🟡 | Common-column dtype drift can silently disappear (`profiling/drift.py:136-153`) | small | ⬜ open |
| OC-48 | 🟡 | Expectations pass vacuously on empty frames (`profiling/expect.py:92-209`) | small | ⬜ open |
| OC-49 | 🟡 | Valid partially-unlabelled PCA payloads crash plotting (`profiling/visualizer.py:716-737`) | small | ⬜ open |
| OC-50 | 🟡 | Binary targets miss class-balance advice or flip to regression by sample size (`recommendations.py:147-152`) | small | ⬜ open |
| OC-51 | 🟡 | Transform advice can be mathematically invalid and self-contradictory (`recommendations.py:66-78,129-139`) | small | ⬜ open |
| OC-52 | ⚪ | Categorical colour mapping is process-nondeterministic (`visualizer.py:710-713`) | small | ⬜ open |

### Remaining — core / engines / pipeline

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-64 | 🟠 | **F-14 only partially fixed** — engine registry global still an unlocked race (`engines/registry.py:60,86-91`) | small | ⬜ open |
| OC-65 | 🟡 | polars `to_numpy()` zero-width "parity fix" does not achieve parity (`engines/polars_engine.py`) | small | ⬜ open |
| OC-74 | 🟡 | `NodeRegistry.list_models()` hides all 4 Ensemble models; `category` arg dead (`registry.py:101-108`) | small | ⬜ open |

### Remaining — outliers / casting / binning / timeseries / geo

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-59 | 🟠 | `DatasetProfile` numeric-column coverage completely different between engines (`preprocessing/inspection/`) | small | ⬜ open |
| OC-60 | 🟠 | `GeneralBinning`'s `missing_strategy: "label"` silent no-op on polars (`preprocessing/bucketing.py`) | small | ⬜ open |

### Remaining — modeling / tuning

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-187 | 🟡 | LightGBM's advertised `subsample` control and default search dimension have no effect: both calculators retain native `subsample_freq=0`, disabling row bagging (`modeling/hyperparameters/_tree.py:576`, `_registry.py:298,309`; `classification.py:754`, `regression.py:547`) | small | ⬜ open |

### Remaining — frontend

The separate [frontend CCN inventory](frontend_ccn_remaining_2026-09-10.md) now
has **0 functions in 0 files above CCN 10**: the source-wide strict gate passes.
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
| OC-223 | 🟡 | Completing a pending drift disposition clears the newly typed note after switching to another alert | small | ⬜ open — original modal characterization submits on one alert, switches IDs, types a new note and resolves the old callback; the new note becomes empty. Review modal and parent request lifetimes together before repair. |
| OC-225 | 🟡 | Selecting a job removes the Jobs drawer's accessible name; the detail Back button is also unnamed | small | ⬜ open — the drawer keeps `aria-labelledby="jobs-drawer-title"` after its history heading unmounts. Keep a valid dialog label in both views and label the icon-only Back action; verify real-card keyboard navigation and screen-reader names. |
| OC-226 | 🟡 | Late Segmentation hyperparameter definitions can restore an old model/reference column and call its old update callback after unmount | small | ⬜ open — original-source tests resolve defaults after newer config/model changes and observe the captured old config being emitted. Scope requests to the active model/settings lifetime and merge defaults into current config; cover reference edits, reversed responses and unmount. |
| OC-227 | 🟡 | A completed node drag creates no undo entry because history ignores positions when either the previous or next node is dragging | small | ⬜ open — original public-store characterization sends two `dragging: true` positions followed by `dragging: false`; history stays empty. Capture one pre-drag snapshot and one completed move without recording each frame; cover single/group drags, undo/redo and selection-only changes. |
| OC-56 | ⚪ | `useSchemaPreview` does not cancel in-flight requests on unmount (`hooks/useSchemaPreview.ts`) | small | ⬜ open |
| OC-57 | ⚪ | `any`-typed chart props bypass type safety in EDA components (`modules/eda/`) | small | ⬜ open |

### Remaining — tests / packaging / CI (outside the Ongoing tier)

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-80 | 🟡 | 3 weakest-covered modules untested exactly where silence is dangerous (`_sklearn_compat.py`, `value_replacement.py`, `config_validation.py`) | ~1 day | ⬜ open |
| OC-213 | ⚪ | Leakage examples produce 22 ty diagnostics: 16 in the notebook and six in the Python script, caused by heterogeneous configuration inference and un-narrowed `SplitDataset` slots (`skyulf-core/examples/09_leakage_safety.ipynb`, `09_leakage_safety.py`) | small | ⬜ open — both examples execute successfully; type information needs correction |

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
not a newly reproduced OC finding. It does not change the **63 open / 4 parked**
audit counts. OC-224 remains fixed; OC-223 is a separate alert-note lifecycle bug.

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

### 2026-09-08 — OC-213–215: Problems panel review

Source: [the complete diagnostic disposition](problems_panel_review-2026-09-08.md),
against working tree `f12dde9f`. The repeated export contains 105 distinct
file/rule/locations, including external type stubs, obsolete rules, and optional
style suggestions. Only the actionable example/dependency findings are filed here.
OC-214 and OC-215 are now closed in the archive; OC-213 remains open.

**OC-213 — leakage examples have inaccurate inferred types.** Run
`.venv\Scripts\python.exe -m ty check skyulf-core/examples/09_leakage_safety.ipynb
skyulf-core/examples/09_leakage_safety.py --output-format concise`: 22 diagnostics.
The notebook's unannotated configuration conflates modeling dictionaries with
preprocessing lists; `SplitDataset` slots also admit several frame types and
`(X,y)` tuples, so pandas indexing, subtraction, and `.drop(columns=...)` are
not justified by their declared types. Executing all 12 notebook code cells in
order and running the Python example succeeds, including assertions. This is
an example typing defect, not a reproduced training failure.
**Fix/verification target:** annotate the configuration appropriately and retain
named pandas partition variables, or narrow slot types explicitly. Preserve
the general `SplitDataset` contract and the example assertions. Rerun the
explicit example ty command and execute both examples.

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

**OC-199 — explicitly selected coordinates survive exclusion.** Executed
`EDAAnalyzer(pl.DataFrame({'lat':[1.,2.,3.], 'lon':[10.,20.,30.],
'x':[1.,2.,3.]})).analyze(exclude_cols=['lat','lon'],lat_col='lat',lon_col='lon')`.
The result still contains all three coordinate pairs in
`geospatial.sample_points`, plus their bounds and centroid, although the
per-column profile excludes them. **Fix/verification target:** apply the
exclusion policy consistently before explicit geospatial selection.

**OC-191 — unsupported string aggregation on valid dtypes.** Executed
`EDAAnalyzer(pl.DataFrame({'x': [None,None]})).analyze()` raises
`SchemaError: expected String, got null`; using
`pl.Series(['a','b'], dtype=pl.Enum(['a','b']))` raises the equivalent Enum
error. **Fix/verification target:** handle null-only columns and recognize or
normalize Enum before text aggregates. OC-121 concerns preprocessing
auto-selection; this finding concerns profiling aborting completely.

**OC-192 — categorical null drill-down loses the selected group.** With
`group=['a',None,'b']` and `v=[1,2,3]`, a decomposition sum split publishes an
`Unknown` bucket valued **2**. Applying
`{'column':'group','operator':'==','value':'Unknown'}` returns a total of
**0**. Only numeric columns recognize the null sentinel.
**Fix/verification target:** preserve null identity across split output and
filter input for every dtype, including genuine literal `Unknown` values.

**OC-187 — LightGBM row-subsampling control is inert.** Executed both public
`LGBMRegressorCalculator.fit` and `LGBMClassifierCalculator.fit` on 300-row,
6-feature sklearn generated datasets (`random_state=7`). With 20 trees, one
worker and seed 7, changing only `subsample` from 1.0 to 0.4 produced exactly
identical predictions/probabilities (maximum difference **0.0**). Setting
`subsample_freq=1` as a control produced maximum differences **85.66120232221425**
for regression and **0.16583687788133306** for classification. The metadata
exposes `subsample` but no frequency control; the default search proposes
`[0.6, 0.8, 1.0]` while the estimator frequency remains zero. This silently
ignores a requested regularization setting and wastes trials on equivalent
models. **Fix/verification target:** define and expose the bagging activation
policy for supported boosting modes, and verify a selected fraction changes
the fitted model when bagging is enabled. No implementation change made.
