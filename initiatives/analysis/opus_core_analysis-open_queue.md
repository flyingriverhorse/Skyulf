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
22 ⚪, plus OC-160–319 filed by later reviews. OC-100 was retracted as a false
positive and is not counted; the corrections pass stays in the archive.

**Status key:** ⬜ open · 🟨 in progress · ✅ done · ⏭️ parked

Original severity and effort are the audit's own; Qwen follow-up ratings
reflect the verified scope, not the original scanner severity. A status cell
retains measured evidence and limitations; repair verification lives in the
archive Log.

---

## Live — fix queue

**Current status (2026-09-12): 58 open / 4 parked.**
OC-264/275/282/285/294/295 are fixed in the latest six-finding batch.
One additional search/refit mismatch is filed as OC-319; verification is in the archive.
The verified Qwen follow-up added **48 findings, OC-271–318**,
grouped below by priority and domain. Qwen #1/#50/#57 reuse OC-253/65/64;
#13 is already fixed as OC-268. #18/#41 share OC-286, which requires both
persistence paths to be repaired. Policy-only #36 is recorded separately
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
| OC-272 | 🟡 | **Supported fitted models cannot produce fingerprints or model cards** (`skyulf-core/skyulf/pipeline/seal.py:174-185`) — Qwen #3. Canonicalize supported fitted Cython loss and NumPy Generator state while continuing to reject unknown state explicitly. | medium | ⬜ open — Actual fitted GradientBoosting, HistGB and SGD pipelines raise TypeError in both public APIs; LR/RF controls pass, and the claimed 10/38 catalog ratio was not remeasured. |
| OC-277 | 🟠 | **Concurrent submissions bypass duplicate-job protection** (`backend/ml_pipeline/_execution/jobs.py:104-119`) — Qwen #8. Make job reservation atomic across API processes and preserve one lock for an in-process key while waiters exist; coordinate lock cleanup with OC-310. | medium | ⬜ open — Two controlled OS processes create two queued jobs with versions 1/2 after the same absent-row check, and waiting coroutines reach two simultaneous entries for one key; one request alone does not duplicate a job and PostgreSQL was not exercised. |
| OC-278 | 🟠 | **Deployment promotion deactivates a working model before validating the replacement** (`backend/ml_pipeline/deployment/service.py:131-175,260-272`) — Qwen #9. Verify artifact usability before atomic promotion and preserve the active deployment if validation fails. | medium | ⬜ open — A missing-artifact deploy returns 200 and disables a working deployment, then predict returns 400; schema loading does run after commit, so the defect is the validation order rather than a complete absence of loading. |
| OC-279 | 🟠 | **Synchronous Preview execution blocks its API event loop** (`backend/ml_pipeline/_internal/_routers/preview.py:767-776,815-865`) — Qwen #10. Move synchronous graph work off the request event loop and verify concurrency with the actual Preview path. | medium | ⬜ open — A real 1000x16 input and 60 scalers take 0.853 s and stall a 5 ms heartbeat for 0.851 s; the loader is sampled and training nodes are excluded, so this does not establish that all workers stop. |
| OC-280 | 🟡 | **Configured default rate limits are not applied to undecorated routes** (`backend/middleware/rate_limiter.py:10-19`) — Qwen #11. Wire default limiting into the actual app and retain explicit per-route limits; authentication decisions remain separately parked. | medium | ⬜ open — An undecorated mutation route accepts 230/230 requests while an explicit-limit control returns 429 after 60; the current inventory has 34 mutation routes, eight decorated and 26 without default enforcement. |
| OC-286 | 🟠 | **Non-finite profile and preview metrics cross JSON persistence boundaries** (`skyulf-core/skyulf/profiling/schemas.py:259-273`; `backend/ml_pipeline/_execution/strategies.py:130-163`) — Qwen #18 / #41. Enforce a consistent finite-or-null JSON contract for both EDA profile persistence and background preview/job metrics; closure requires coverage of both write paths. | medium | ⬜ open — Qwen #18 typed profile fields retain inf/NaN and #41 actual preview success persists Infinity to SQLite; PostgreSQL/MySQL JSON binders emit non-standard JSON, but live server rejection was not tested and orjson/model_dump_json controls sanitize it. |
| OC-289 | 🟡 | **User-controlled identifiers can inject new lines into logs** (`backend/ml_pipeline/model_registry/api.py:43-51`) — Qwen #22. Sanitize identifiers and exception text at the affected logging boundaries while retaining prior credential redaction. | small | ⬜ open — A real missing-job request containing percent-encoded LF returns 404 but writes a raw newline into the registry log; this demonstrates log-integrity loss, not code execution. |
| OC-292 | 🟡 | **Canvas accepts Segmentation-to-Ensemble connections that convert to invalid data inputs** (`frontend/ml-canvas/src/core/utils/pipelineConversion/ensemble.ts:11-16`) — Qwen #25. Align connection validation, ensemble conversion and backend input expectations for unsupported model families. | medium | ⬜ open — Registered port validation accepts the connection and the actual converter lists the segmentation training node as Ensemble data input; the backend rejects its Model artifact where Dataset is required. |
| OC-318 | 🟡 | **Empty scaling range fields pass validation and become null bounds** (`frontend/ml-canvas/src/modules/nodes/processing/scaling/ScalingControls.tsx:64,91`) — Qwen #55. Validate finite MinMax/Robust range inputs in the UI and reject invalid serialized bounds at the Core boundary. | small | ⬜ open — Clearing the real controls produces NaN, valid=true and JSON [null,1] or [null,75]; actual Core fitting fails with a None comparison TypeError. |
| OC-319 | 🟡 | **Explicit Logistic Regression penalty/ratio settings describe different models during search and refit** (`skyulf-core/skyulf/modeling/_sklearn_compat.py:40-45`; `modeling/_tuning/params.py`; `modeling/_tuning/engine.py`) — Align the documented explicit-ratio precedence with penalty semantics across constructor and searcher set_params paths; keep nullable Elastic Net defaults OC-282 separate. | decision + small | ⬜ open — On sklearn 1.8.0, penalty=l2 and l1_ratio=0.5 produce maximum coefficient difference 0.10500864 between normalized construction and search-style set_params; l1 control differs by 0.81086563, while elasticnet matches. Public grid/halving_grid score the same single l2 candidate and stratified folds at -0.33013530/-0.27716182 log-loss scores, then return identical final coefficients. Reproduction and limitations are in the latest archive Log. |


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
| OC-306 | 🟡 | **Monitoring upload parsing masks size-limit errors and allows a very large eager read** (`backend/monitoring/router.py:262-282`) — Qwen #42. Preserve HTTP 413 and enforce an appropriate bounded upload/read policy before parsing. | small | ⬜ open — Seventeen bytes against a sixteen-byte limit becomes HTTP 400 Failed to parse uploaded file instead of 413; the default read requests 10 GiB plus one byte, but an actual huge allocation or OOM was not attempted. |
| OC-307 | 🟡 | **A direct Preview request ending at Data Preview succeeds without executing upstream work** (`backend/ml_pipeline/_internal/_routers/preview.py:357-359,445-448`) — Qwen #43. Make terminal-sink handling in the API execute the intended upstream graph or reject an unsupported request explicitly. | small | ⬜ open — Posting loader-to-data_preview directly returns 200/success with zero executed nodes; the current Canvas toolbar removes the sink first and its normal Preview path works. |
| OC-308 | 🟡 | **Invalid cyclic Preview graphs are recorded as critical server failures** (`backend/ml_pipeline/_internal/_routers/preview.py:919-923`) — Qwen #44. Validate cyclic input as a client error before graph execution and avoid recording it as an internal critical incident. | small | ⬜ open — A direct API request with two scalers in a cycle returns 500 and creates a real ErrorEvent classified critical; current Canvas connection and submit validation block that graph. |
| OC-309 | 🟡 | **Sampling and job-list endpoints accept invalid or excessive pagination bounds** (`backend/data_ingestion/connectors/file.py:192-215`) — Qwen #45. Validate non-negative offsets and bounded positive limits before eager reads or large metric serialization. | medium | ⬜ open — limit=-1 eagerly reads a 25-row source and returns 24 rows; job endpoints accept large limits/negative skip, while the combined route has a skip cap; 243 KB times 1000 is about 243 MB, not 24 GB. |
| OC-310 | 🟡 | **Failed job submission leaks entries in the per-key lock registry** (`backend/ml_pipeline/_internal/_routers/run_pipeline.py:190-212`) — Qwen #46. Validate job types and clean up reservation failures without evicting locks still owned or awaited; coordinate with OC-277. | small | ⬜ open — Four actual run requests with schema-accepted unknown job types return 500 and leave four unlocked registry entries; the UI does not submit those values and the route has an explicit 20/min rate limit. |
| OC-311 | 🟡 | **Legacy deployment artifact resolution disagrees with the default permitted root** (`backend/ml_pipeline/deployment/service.py:101-104,142-144,224-239`) — Qwen #47. Resolve supported legacy artifact references consistently with configured storage while preserving path containment checks. | small | ⬜ open — A legacy abstract URI resolves under exports/models and raises PermissionError with the default uploads/models root; current successful jobs store real URIs, and export-root configuration or TESTING changes the outcome. |


### Remaining — direct-audit modules

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-91 | 🟡 | Three public `core/` seams (263 lines) have zero call sites; one duplicates a differently-shaped backend class name | small | ⬜ open |
| OC-111 | 🟡 | A profiling recommendation branch is unreachable | small | ⬜ open |
| OC-102 | ⚪ | Five tunable models return an empty search space from the live `/defaults` endpoint (`hyperparameters/_registry.py`) | small | ⬜ open |

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
| OC-11 | ⚪ | Mega smoke test silently skips nodes with empty params (`tests/unit/test_all_nodes_smoke.py`) | small | ⬜ open |

### Remaining — encoding / cleaning / imputation / scaling / drop / resampling

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-298 | 🟡 | **Boolean alias conversion changes unmatched numeric values to strings on Polars** (`skyulf-core/skyulf/preprocessing/cleaning/alias.py:55,65,103,110`) — Qwen #31. Define consistent handling for explicitly selected non-text columns and preserve unmatched values under the supported alias contract. | small | ⬜ open — Explicit numeric input 2,3,4 remains integer on pandas but becomes strings on Polars; ordinary automatic text selection does not demonstrate this defect. |
| OC-301 | 🟡 | **Advanced resampler settings are saved but not forwarded to sampler construction** (`skyulf-core/skyulf/preprocessing/resampling.py:205,213,298,303`) — Qwen #34. Pass supported SVC/KMeans/n_jobs settings to the appropriate sampler with capability-aware validation. | small | ⬜ open — Actual SVMSMOTE receives svm_estimator=None and KMeansSMOTE receives kmeans_estimator=None/n_jobs=None despite configured artifacts; plain SMOTE in this installation does not accept n_jobs. |


### Remaining — feature generation / selection / vectorization / transformations

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-313 | 🟡 | **SentenceEmbedder model caching permits duplicate concurrent loads** (`skyulf-core/skyulf/preprocessing/vectorization/sentence_embedder.py:33,41,53,159,209`) — Qwen #49. Coordinate same-key model construction within a process and handle loading failures without leaving waiters stuck. | medium | ⬜ open — Four threads using one model name create four model objects through a synchronized fake constructor but leave one cache entry; real model download, GPU OOM and cross-process cache sharing were not tested. |


### Remaining — profiling (outside the OC-39–46 cluster)

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-47 | 🟡 | Common-column dtype drift can silently disappear or be erased by a successful lossy cast to the reference dtype (`profiling/drift.py:192`) | small | ⬜ open — besides the original uncastable-to-null case, integer reference `[0]*50+[1]*50` against fractional current `[0.9]*50+[1.9]*50 reports all distances zero; a Float64 reference control detects drift with normalized Wasserstein `1.8`. |
| OC-50 | 🟡 | Binary targets miss class-balance advice or flip to regression by sample size (`recommendations.py:147-152`) | small | ⬜ open |
| OC-259 | 🟡 | Decimal columns are classified as Text and abort the entire EDA profile when batched string aggregates run (`profiling/_analyzer/_utils.py:64`, `profiling/analyzer.py:278`) | small | ⬜ open — a two-row Decimal amount column raises SchemaError from `str.len_bytes()`; Time/List columns hit the same unsupported-dtype fallback, while Float64 succeeds. |
| OC-260 | 🟡 | Time-series plotting builds unequal timestamp/value arrays when metrics have different missingness (`profiling/visualizer.py:823-827,846`) | small | ⬜ open — a native profile with 1000 timestamps and only 500 observed values for one metric makes public `EDAVisualizer.plot()` raise an x/y dimension mismatch. |
| OC-274 | 🟡 | **Causal feature selection spends its cap on constant columns** (`skyulf-core/skyulf/profiling/_analyzer/causal.py:25-30`) — Qwen #5. Handle non-finite ranking values deterministically and retain eligible variable features before applying the cap. | small | ⬜ open — Moving five constant columns from the start to the end of the schema changes selected variable features from 9 to 14; the reported 20/70 edge counts were not reproduced. |
| OC-287 | 🟡 | **One constant column suppresses VIF diagnostics for all other features** (`skyulf-core/skyulf/profiling/_analyzer/numeric.py:54-59`) — Qwen #19. Retain useful diagnostics for variable columns and explain excluded constants or unavailable calculations. | small | ⬜ open — Correlated features produce about 1.34 million VIF and two alerts, but adding an integer constant yields vif=None and zero alerts; an existing test intentionally expects the guard, not a useful explanation. |
| OC-288 | 🟡 | **Temporal decomposition buckets cannot be used as drill-down filters** (`skyulf-core/skyulf/profiling/_analyzer/decomposition.py:72-83,127,153-158`) — Qwen #21. Round-trip serialized date/time bucket values through dtype-aware filtering and verify reachable Date/Datetime UI paths. | medium | ⬜ open — All six returned Date/Datetime/Time bucket values fail when reused as filters, while String/int controls pass; HTTP error mapping is source-traced and the full Time profiling path also has separate OC-259. |
| OC-302 | 🟡 | **Outlier results omit the sample population behind their counts and percentages** (`skyulf-core/skyulf/profiling/_analyzer/multivariate.py:383-410`) — Qwen #37. Expose the sampled row count and make UI labels distinguish sampled results from the entire dataset. | medium | ⬜ open — A 200000-row dataset yields 2500 outliers and 5% from a 50000-row sample without that denominator in the payload/UI; a full-data count of 10000 or exact fourfold undercount was not established. |
| OC-303 | 🟡 | **Correlation truncation is hidden from the normal frontend warning path** (`skyulf-core/skyulf/profiling/correlations.py:31-45`) — Qwen #38. Include omission metadata with capped matrices and render it even when the returned matrix is within the cap. | small | ⬜ open — A real 25-column profile returns a 20-column matrix, leaving the frontend greater-than-20 check false; a server warning exists but does not reach the user. |
| OC-304 | 🟡 | **Reused EDAAnalyzer loses metadata for filters still applied to its data** (`skyulf-core/skyulf/profiling/analyzer.py:128,143-154,646,672`) — Qwen #39. Keep active-filter metadata consistent with the documented stateful analyzer behavior across repeated analyze calls. | small | ⬜ open — analyze(x>=5) on 0..9 returns five rows and a filter; a subsequent analyze() still returns five rows but active_filters is empty; normal backend requests construct fresh analyzers. |


### Remaining — core / engines / pipeline

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-64 | 🟠 | **F-14 only partially fixed** — engine registry global still an unlocked race (`engines/registry.py:60,86-91`) | small | ⬜ open Qwen recheck: ordered threads reproduce the shared-default race, but no production set_active_engine call was found; current live-request impact is unproven. Qwen #57. |
| OC-65 | 🟡 | polars `to_numpy()` zero-width "parity fix" does not achieve parity (`engines/polars_engine.py`) | small | ⬜ open Qwen recheck: a three-row, zero-column selection reaches conversion as pandas (3,0) versus Polars (0,0), so Polars has already lost its height. Qwen #50. |
| OC-312 | 🟡 | **The sklearn bridge returns zero-dimensional object arrays for unsupported containers** (`skyulf-core/skyulf/engines/sklearn_bridge.py:28,44`) — Qwen #48. Reject unsupported input containers clearly at the public adapter boundary while preserving valid mixed-engine X/y conversion. | small | ⬜ open — object, dict and SplitDataset inputs warn and become 0-D object arrays; pandas X with Polars y converts correctly and no normal production misuse of SplitDataset was found. |


### Remaining — outliers / casting / binning / timeseries / geo

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-255 | 🟡 | DateFeatures uses local offsets on pandas and UTC on Polars; mixed DST offsets make the pandas `.dt` access fail (`preprocessing/time_series/date_features.py:64-66,106-109`) | small | ⬜ open — identical `+02:00` strings produce different days/hours; mixed `+02:00`/`+03:00` strings raise AttributeError on pandas, so choose and enforce one timezone contract. |


### Remaining — modeling / tuning

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-251 | 🟡 | Fold-aware Halving/Optuna scoring keeps original validation labels after preprocessing filters prediction rows (`modeling/_tuning/fold_pipeline.py:168-171`) | medium | ⬜ open — IQR yields 120 held-out labels / 110 predictions; actual halving_grid and Optuna searches fail all trials while grid scores the same chain at R2 `0.999998`. |
| OC-253 | 🟡 | F1 tuning and evaluation/threshold tuning disagree on the positive class for labels `{1,2}` (`modeling/_tuning/metrics.py:235`, `modeling/_evaluation/classification.py:84`) | decision + small | ⬜ open — the same predictions score `0.909091` for class 1 in tuning and `0.8` for class 2 in evaluation; reconcile the documented stock-scorer exception with a shared positive-class contract. Qwen recheck: the same {1,2} predictions score 0.75 for class 1 and 0.5 for class 2; retain the documented scorer-contract decision. Qwen #1. |
| OC-269 | 🟡 | Optuna constructs the selected pruner but never enables pruning on OptunaSearchCV (`modeling/_tuning/strategies/optuna.py:240-252`) | medium | ⬜ open — Hyperband with incremental-fit-capable SGD still has `enable_pruning=False` and no intermediate trial values; respect estimator capabilities when implementing the advertised early stopping. |


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
| OC-314 | 🟡 | **Keyboard paste mutates a read-only Canvas** (`frontend/ml-canvas/src/core/hooks/useClipboard.ts:43-76`) — Qwen #51. Enforce the effective read-only state in clipboard mutation paths while retaining permitted copy behavior. | small | ⬜ open — The actual store and clipboard hook with readOnlyOverride enabled allow Ctrl+C/Ctrl+V to increase the node count from one to two; this is a UI editing contract, not an authorization boundary. |
| OC-315 | 🟡 | **Edge selection adds structural undo-history entries** (`frontend/ml-canvas/src/core/store/graphStore/historyEquality.ts:15-16`) — Qwen #52. Ignore selection-only edge changes when comparing graph history while preserving real edits; cover this alongside the separate missing-drag snapshot in OC-227. | small | ⬜ open — The public onEdgesChange(select=true) call increases undo history from zero to one without a structural edit; the open OC-227 concerns a completed node drag creating no entry. |
| OC-316 | 🟡 | **Canvas source links duplicate datasets already present in the graph** (`frontend/ml-canvas/src/pages/CanvasPage.tsx:153-157`) — Qwen #53. Recognize dataset nodes through their registered definition and source identity before automatically inserting from source_id. | small | ⬜ open — Opening CanvasPage with the same source_id and an existing custom/dataset_node creates two dataset nodes because the check reads n.type instead of data.definitionType. |


### Remaining — tests / packaging / CI (outside the Ongoing tier)

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-80 | 🟡 | 3 weakest-covered modules untested exactly where silence is dangerous (`_sklearn_compat.py`, `value_replacement.py`, `config_validation.py`) | ~1 day | ⬜ open |
| OC-213 | ⚪ | Leakage examples produce 22 ty diagnostics: 16 in the notebook and six in the Python script, caused by heterogeneous configuration inference and un-narrowed `SplitDataset` slots (`skyulf-core/examples/09_leakage_safety.ipynb`, `09_leakage_safety.py`) | small | ⬜ open — both examples execute successfully; type information needs correction |
| OC-246 | 🟡 | Local-only full-inference smoke returns a passing result before inference when the current model artifact is a tuple; remaining checks only print success/failure (`tests/integration/test_full_inference_pipeline.py:180-187`) | small | ⬜ open — a path-only workspace probe prints “Model artifact is not a dict: class tuple” then passes without reaching inference; the original external-workspace script is unchanged and excluded from final EDA verification. See the EDA review's follow-up evidence. |

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
