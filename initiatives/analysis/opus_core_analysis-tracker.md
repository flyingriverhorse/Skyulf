# Opus core audit — fix tracker

**Source audit:** [`opus_core_analysis.md`](opus_core_analysis.md) (master report) +
[`opus_core_analysis/README.md`](opus_core_analysis/README.md) (index of the 19 per-area
report files `00`–`18`).
**Baseline:** commit `93d7719e` (master), audit run 2026-08-31 → 09-01 by 15 parallel
read-only agents (Claude Opus 5). 116 findings: 5 🔴 / 45 🟠 / 44 🟡 / 22 ⚪.
27 agent-phase findings were re-verified by execution (25 stand, 4 worse than filed:
OC-12/18/40/42; 2 corrected: OC-01, OC-46). OC-100 was retracted as a false positive
and is not counted.

The queue below follows the master report's suggested fix order (4 tiers), then the
remaining findings grouped by domain. R1 (the systemic core↔frontend contract fix)
retires 8 findings as a class and is tracked separately.

**Status key:** ⬜ open · 🟨 in progress · ✅ done · ⏭️ parked

---

## Corrections & re-verification (from the audit's own validation pass)

| ID | What happened |
|---|---|
| OC-100 | **Retracted** — false positive (binning out-of-range behaviour is intentional); not counted in the 116 |
| OC-01 | **Corrected** — not "always stale": a stale `0.5.8` dist-info shadows the real `0.8.8` path-order-dependently |
| OC-46 | **Downgraded** 🟠→🟡 — non-finite floats are silently coerced to `null` by orjson; only stdlib-json paths emit invalid JSON |
| OC-12, OC-18, OC-40, OC-42 | **Worse than filed** on execution re-verification |
| F-02, F-04, F-06, F-13, F-15 | Prior findings **re-verified fixed** |
| F-14 | **Reopened** as OC-64 — engine-registry global race only partially closed |

---

## Live — fix queue

Ordered by the master report's suggested fix order: **Now** (silent wrongness
reaching users), **Next** (wrong results in realistic configs), **Then** (decide
deployment model), **Ongoing** (remove the hiding conditions). Remaining findings
follow, grouped by domain.

### Now — silent wrongness reaching users

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-75 | 🔴 | Dev polars 1.40.1 below declared floor ≥1.43.2 — 10 tests, every notebook, a benchmark broken; prerequisite for trusting any polars result | 1 line | ✅ done |
| OC-12 | 🔴 | Row-dropping desyncs `X` and `y` on non-unique pandas indexes (`drop_rows.py:60-67`, `deduplicate.py:44-47`); polars path already correct | small | ✅ fixed 2026-09-03 |
| OC-58 | 🔴 | Numeric→boolean cast on polars treats any nonzero as `True` (`casting.py:143-178`) | small | ✅ fixed 2026-09-03 |
| OC-62 | 🔴 | `fingerprint()` not reproducible for any artifact holding an object-dtype array (`pipeline/seal.py:57-59`) | small | ✅ fixed 2026-09-03 |

### Next — wrong results in realistic configs

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-13 | 🟠 | Drop-Rows UI settings ignored; every canvas run becomes "drop any missing" (`pipelineConverter.ts:249-253`) | small | ✅ fixed 2026-09-03 |
| OC-14 | 🟠 | Iterative Imputer UI estimator choices silently fall back to BayesianRidge (`imputation/_common.py:103-111`) | small | ✅ fixed 2026-09-03 |
| OC-15 | 🟠 | MinMax/Robust scaler range controls in UI ignored (`scaling/minmax.py:96-100`, `robust.py:116-123`) | small | ✅ fixed 2026-09-03 |
| OC-19 | 🟡 | Alias Replacement exposes `punctuation` mode that does nothing (`cleaning/alias.py:45-52`) | small | ✅ fixed 2026-09-04 |
| OC-20 | 🟡 | Value Replacement's "empty columns = all columns" UI promise is false (`cleaning/value_replacement.py:163-180`) | small | ✅ fixed 2026-09-04 |
| OC-53 | 🟡 | `select_from_model`'s `max_features` is Python-only, UI-unreachable | small | ✅ fixed 2026-09-04 |
| OC-61 | ⚪ | `BinningNode`'s "Precision (Decimals)" UI field never sent to backend (`BinningNode.tsx`) | small | ⬜ open — R1 candidate |
| OC-66 | 🟠 | `CalibratedClassifierCV`'s user-selected base estimator silently discarded during tuning (`classification.py:206-282` vs `_tuning/engine.py:495-499`) | small | ⬜ open — R1 candidate |
| OC-16 | 🟠 | KNN/Iterative imputers crash on all-missing fitted columns (`imputation/knn.py:64-76`, `iterative.py:68-84`) | small | ⬜ open |
| OC-17 | 🟠 | SimpleImputer polars mean/median crashes on all-null columns (engine divergence, `imputation/_common.py:32-37`) | small | ⬜ open |
| OC-69 | 🟠 | Engine trusts `config.nodes` list order, never verifies topological sort (`_schema_graph.py:49-70`); `_kahn_topological_order` already exists — wiring fix | small | ⬜ open |
| OC-35 | 🟠 | Multiclass splits missing a class emit binary-only metrics + null curve points (`metrics.py:217-237,361-363`) | small | ⬜ open |
| OC-36 | 🟠 | F1 threshold tuning picks pathological threshold on single-class validation (`thresholds.py:101-111`) | small | ⬜ open |
| OC-39 | 🟠 | NaN-bearing numeric columns publish `nan` stats and leak non-finite JSON (`profiling/analyzer.py:215-224`) | small | ⬜ open — profiling cluster |
| OC-40 | 🟠 | PCA/clustering "mean imputation" actually replaces NaN with `0.0` (`multivariate.py:46-60`) | small | ⬜ open — profiling cluster |
| OC-41 | 🟠 | Quartiles use nearest-rank, not linear interpolation (disagrees with pandas) (`analyzer.py:221-222`) | small | ⬜ open — profiling cluster |
| OC-42 | 🟠 | Skewness/kurtosis use biased estimators, breaking the hardcoded threshold rule (`analyzer.py:223-224`) | small | ⬜ open — profiling cluster |
| OC-43 | 🟠 | Correlation drops valid columns/rows instead of the defined missing-data policy (`correlations.py:41-44,100-110`) | small | ⬜ open — profiling cluster |
| OC-44 | 🟠 | Wasserstein drift thresholds normalized value but reports raw one (`drift.py:181-195`) | small | ⬜ open — profiling cluster |
| OC-45 | 🟠 | Schema drift computed but never counted or rendered as drift (`drift.py:76-98`) | small | ⬜ open — profiling cluster |
| OC-46 | 🟡 | Non-finite floats reach public payloads; only stdlib-json paths emit invalid JSON (`schemas.py:7-17,263-302`) | small | ⬜ open — profiling cluster |

### Then — decide deployment model first

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-71 | 🟠 | **No authentication or authorization anywhere on the API** (`main.py:373-395`, `database/models.py:151-159`) — **confirm intent first**: single-tenant self-hosted → documentation task; multi-tenant → highest-priority item in the entire report (scaffolded `User` model + dead `AUTH_FALLBACK_*` settings suggest the latter was intended) | decision + ~1 week | ⬜ open — confirm intent |
| OC-72 | 🟡 | Insecure-by-default config: unset `FASTAPI_ENV` fails open to wildcard CORS + credentials (`config/factory.py:26`, `main.py:359-366`) | small | ⬜ open — with OC-71 |
| OC-73 | ⚪ | `DataSource.credentials` documented encrypted, stored plaintext JSON (`database/models.py:107`) | small | ⬜ open — with OC-71 |

### Ongoing — remove the hiding conditions

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-76 | 🟠 | Cross-engine parity tests cover 9 of 100 nodes and never compare applied output — directly caused OC-04/23/24/58 to go unnoticed | ~3 days | ⬜ open |
| OC-77 | 🟠 | `--maxfail=1` hides real failure count; `--cov-fail-under=45` vs 98.4% actual (two flag changes, `.github/workflows/skyulf-core-tests.yml:82-87`) | mechanical | ⬜ open |
| OC-01 | 🟠 | `skyulf.__version__` ambiguous: stale `0.5.8` dist-info shadows real `0.8.8` (path-order dependent) — packaging-integrity cluster | small | ⬜ open |
| OC-02 | 🟠 | Dev editable install dangling; `import skyulf` fails outside repo — packaging-integrity cluster | small | ⬜ open |
| OC-78 | 🟡 | `py.typed` declared in packaging metadata but file does not exist — packaging-integrity cluster | 1 line | ⬜ open |
| OC-79 | 🟡 | `joblib` imported at module scope but not in `install_requires` — packaging-integrity cluster | 1 line | ⬜ open |
| OC-81 | ⚪ | No `License ::` classifier / SPDX field — packaging-integrity cluster | 1 line | ⬜ open |
| OC-03 | 🟠 | Systemic `infer_output_schema` int→float misprediction across 22 nodes — one sweep + parametrized test (predicted schema == actual schema for every node) | ~1 day | ⬜ open |
| OC-09 | 🟡 | Narrow `ruff select` hides ~500 missing docstrings + 84 unused args — **last**, widening first would bury the signal | half day | ⬜ open |

### Remaining — evaluation & explainability

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-146 | 🔴 | Binary `pr_auc` scored against wrong class on `{1,n}` labels — reports 0.32 vs true 0.97, no warning (`metrics.py:324-326`) | small | ⬜ open |
| OC-149 | 🟠 | Clustering evaluation crashes on polars when a numeric feature is all-null within one cluster (`clustering.py:83-88`) | small | ⬜ open |
| OC-37 | 🟡 | Binary PR-AUC dropped for string-labeled classifiers (`metrics.py:324-327`) | small | ⬜ open |
| OC-148 | 🟡 | PII detector flags ordinary 7+ digit numeric ID columns as "Email/Phone" (`profiling/_analyzer/text.py:107-128`) | small | ⬜ open |
| OC-147 | ⚪ | `optimize_thresholds` returns a dict shape that bypasses its own documented binary rule, flipping `>=` to `>` on exact ties (`thresholds.py:66-88`) | small | ⬜ open |
| OC-38 | ⚪ | Clustering metrics treat DBSCAN `-1` noise as a real cluster (`metrics.py:432-459`) | small | ⬜ open |

### Remaining — backend infrastructure

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-68 | 🟠 | Model alias map task-unaware — direct API caller silently trains the wrong estimator family (`_execution/engine/_node_runners.py:1157-1183`) | small | ⬜ open |
| OC-70 | 🟡 | Leakage validator checks for *a* splitter globally, not that *this* branch is protected (`_execution/_leakage_validation.py:189-267`) | small | ⬜ open |
| OC-130 | 🟠 | Typo in `FASTAPI_ENV` silently disables the entire production security posture (wildcard CORS w/ credentials, DEBUG=True, no SECRET_KEY check) (`config/factory.py:27-32`) | small | ⬜ open |
| OC-150 | 🟠 | S3 error "sanitiser" matches credential key names case-sensitively — S3 403 bodies + replayable presigned URLs logged verbatim; duplicated in two files (`connectors/s3.py:31-37`, `artifacts/s3.py:67-73`) | small | ⬜ open |
| OC-153 | 🟠 | Multi-input merge silently switches column-wise→row-wise when a branch changes row count — 5-row set + filtered branch yields 8 rows, 3 duplicates, zero UI warnings (`_merge.py:338-348`) | small | ⬜ open |
| OC-154 | 🟠 | Serving-time feature-order reindex (fix F-02) fails open on column mismatch — returned 213.00 where truth is 321.00 (`deployment/service.py:438-442`) | small | ⬜ open |
| OC-155 | 🟠 | Legacy predict path zero-fills missing features and returns a prediction normally; caller never sees a warning (`deployment/service.py:457-462`) | small | ⬜ open |
| OC-145 | 🟡 | Crashed cross-validation returns the same `{}` sentinel as a disabled one — job reports success with missing `cv_*` metrics (`_node_runners.py:871-907`) | small | ⬜ open |
| OC-151 | 🟡 | Trial-buffer `clear_*` hooks documented but never called — 110.9 MB retained for process lifetime (`realtime/trial_buffer.py:56-59,103-106`) | small | ⬜ open |
| OC-156 | 🟡 | `roc_auc` threshold-tuning objective scores hard predictions — bit-identical to `balanced_accuracy` (`threshold_tuning_service.py:77-92`) | small | ⬜ open |
| OC-158 | 🟡 | Sync/async JSON serializers disagree: sync nulls 8 of 15 legitimate strings (`"nan"`, `"NaT"`, `"<NA>"`, `"inf"`…), async nulls none; 603-line module production-dead but test-covered (`serialization.py:369,435-446`) | half day | ⬜ open |
| OC-131 | ⚪ | Diagnostics fail open — PSI returns `0.0` on any numeric failure (`profiling/drift.py:474-476`) | 1 line | ⬜ open |
| OC-132 | ⚪ | Dead `dropped_features` branch (key appears exactly once in repo) (`graph_utils.py:534-537`) | 1 line | ⬜ open |
| OC-152 | ⚪ | Two raw-SQL executors accept unconstrained query strings, zero callers — latent injection sink (`async_connection_manager.py:243-268`) | small | ⬜ open |
| OC-157 | ⚪ | `first_wins` merge strategy reverses output column order, contradicting its docstring (`_merge.py:221-236`) | small | ⬜ open |
| OC-159 | ⚪ | Empty filter dict compiles to WHERE-less `DELETE FROM data_sources`/`UPDATE`; dead call path today (`async_sqlite_queries.py:129-146`) | 1 line | ⬜ open |

### Remaining — direct-audit modules

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-110 | 🟠 | Semantic-type inference misclassifies small categorical columns as `Text`, so task type never inferred (`profiling/_analyzer/column.py`, `analyzer.py:502`) | small | ⬜ open |
| OC-113 | 🟠 | Near-perfect multicollinearity silently reports VIF = 1.0 — `max(1.0, …)` clamps numerical garbage (`numeric.py:32-63`) | small | ⬜ open |
| OC-120 | 🟠 | `Decimal` columns silently skipped by every auto-numeric node; crash pandas when selected explicitly (`engines/__init__.py`, `preprocessing/_helpers.py`) | small | ⬜ open |
| OC-91 | 🟡 | Three public `core/` seams (263 lines) have zero call sites; one duplicates a differently-shaped backend class name | small | ⬜ open |
| OC-101 | 🟡 | `calibrated_classifier`'s `random_state` no-op for two independent reasons (estimator rejects it AND factories hardcode the seed) | small | ⬜ open |
| OC-111 | 🟡 | A profiling recommendation branch is unreachable | small | ⬜ open |
| OC-114 | 🟡 | All-null tracked column yields 30 `NaN` autocorrelation lags as real analysis (≥1000-row datasets) (`temporal.py:167-191`) | small | ⬜ open |
| OC-102 | ⚪ | Five tunable models return an empty search space from the live `/defaults` endpoint (`hyperparameters/_registry.py`) | small | ⬜ open |
| OC-112 | ⚪ | Comment and code disagree about the applied threshold | 1 line | ⬜ open |
| OC-121 | ⚪ | polars `Enum` columns invisible to text auto-detection, diverging from pandas `Categorical` (`_helpers.py:148-157`) | small | ⬜ open |
| OC-122 | ⚪ | `TextCleaning` silently ignores unrecognised operation name (`cleaning/text.py:151-153`) | small | ⬜ open |
| OC-90 | ⚪ | Unknown split config keys silently dropped instead of rejected (`preprocessing/split.py`) | small | ⬜ open |

### Remaining — file-coverage closure

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-140 | 🟠 | `InvalidValueReplacement` diverges across engines on non-numeric columns (pandas silently NaNs, polars raises) | small | ⬜ open |
| OC-142 | 🟠 | EDA correlation ratio η exceeds 1.0 with nulls; null-heavy columns rank as strongest association | small | ⬜ open |
| OC-143 | 🟠 | RFE ignores the UI's `k`, silently selecting half the features | small | ⬜ open |
| OC-141 | ⚪ | `invalid_values` param declared in `node_meta` with zero consumers | 1 line | ⬜ open |
| OC-144 | ⚪ | Geo distance column named `_km` even when the unit is miles | 1 line | ⬜ open |

### Remaining — cross-cutting & packaging

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-04 | 🟡 | Cross-engine dtype divergence in 3 nodes (int64 vs int8/uint32) (`encoding/dummy.py`, `bucketing.py`) | small | ⬜ open |
| OC-05 | 🟡 | `PowerTransformer` triggers a pandas deprecation that will become an error (`transformations/power.py:101`) | 1 line | ⬜ open |
| OC-06 | 🟡 | 6 registered nodes unreachable from the UI (incl. all of `geo/`) — `registry.py` vs `frontend/` | small | ⬜ open — R1 step 3 catches this class |
| OC-07 | 🟡 | Node-id naming split 55 PascalCase / 45 snake_case + redundant aliases (`registry.py`) | half day | ⬜ open |
| OC-08 | 🟡 | Public-API name collision: `DatasetProfile` means two things (`skyulf/__init__.py:32-46`) | small | ⬜ open |
| OC-10 | ⚪ | 4 dead `infer_output_schema` overrides that only `return None` (`vectorization/*`) | mechanical | ⬜ open |
| OC-11 | ⚪ | Mega smoke test silently skips nodes with empty params (`tests/unit/test_all_nodes_smoke.py`) | small | ⬜ open |

### Remaining — encoding / cleaning / imputation / scaling / drop

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-18 | 🟡 | One-hot/dummy generated names can collide with existing columns (`encoding/one_hot.py:68-92`, `dummy.py:76-99`) | small | ⬜ open |
| OC-21 | 🟡 | WOE additive smoothing not normalized over categories (`encoding/woe.py:130-145`) | small | ⬜ open |
| OC-22 | ⚪ | `TargetEncoder.infer_output_schema` checks an impossible `regression` value (`encoding/target.py:340-360`) | 1 line | ⬜ open |

### Remaining — feature generation / selection / vectorization / transformations

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-23 | 🟠 | Polars `ratio` flips the sign of near-zero negative denominators (`feature_generation/_polars_ops.py:97-112`) | small | ⬜ open |
| OC-24 | 🟠 | Polars group aggregates treat null group keys differently from pandas (`_polars_ops.py:222-234`) | small | ⬜ open |
| OC-25 | 🟠 | RFE "K" chosen in UI ignored by backend (`feature_selection/_common.py:236-240`) | small | ⬜ open |
| OC-26 | 🟠 | `HashingVectorizer` UI "none" norm is an invalid sklearn value → crash (`hashing_vectorizer.py:59`) | small | ⬜ open |
| OC-27 | 🟠 | `GeneralTransformation` ignores the UI `standardize` toggle (`transformations/general.py:34-39,138-139`) | small | ⬜ open |
| OC-28 | 🟠 | Box-Cox transform failures silently return untransformed data (`transformations/power.py:97-104`) | small | ⬜ open |
| OC-29 | 🟡 | `FeatureGeneration` advertises `polynomial` but silently skips it (`feature_generation/_common.py:24-31`) | small | ⬜ open |
| OC-30 | 🟡 | Datetime extraction ignores the UI output name, overwrites collisions (`_pandas_ops.py:173-184`) | small | ⬜ open |
| OC-31 | 🟡 | Frontend wrongly requires a target for unsupervised CorrelationThreshold (`FeatureSelectionNode.tsx:564-566`) | small | ⬜ open |
| OC-32 | 🟡 | `VarianceThreshold` crashes when all candidates are constant (`feature_selection/variance.py:38-47`) | small | ⬜ open |
| OC-33 | 🟡 | `FeatureInteraction` cannot generate single-column self-products (`feature_generation/interaction.py:173-178`) | small | ⬜ open |
| OC-34 | 🟡 | Count/TF-IDF vectorizers crash on empty or stop-word-only corpora (`count_vectorizer.py:79-80`) | small | ⬜ open |

### Remaining — profiling (outside the OC-39–46 cluster)

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-47 | 🟡 | Common-column dtype drift can silently disappear (`profiling/drift.py:136-153`) | small | ⬜ open |
| OC-48 | 🟡 | Expectations pass vacuously on empty frames (`profiling/expect.py:92-209`) | small | ⬜ open |
| OC-49 | 🟡 | Valid partially-unlabelled PCA payloads crash plotting (`profiling/visualizer.py:716-737`) | small | ⬜ open |
| OC-50 | 🟡 | Binary targets miss class-balance advice or flip to regression by sample size (`recommendations.py:147-152`) | small | ⬜ open |
| OC-51 | 🟡 | Transform advice can be mathematically invalid and self-contradictory (`recommendations.py:66-78,129-139`) | small | ⬜ open |
| OC-52 | ⚪ | Categorical colour mapping is process-nondeterministic (`visualizer.py:710-713`) | small | ⬜ open |

### Remaining — core / engines / pipeline

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-63 | 🟠 | `artifact_digest` raises `RecursionError` instead of the documented `TypeError` on cyclic graphs (`pipeline/seal.py`) | small | ⬜ open |
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
| OC-67 | 🟡 | Tuning metrics `pr_auc`/`pr_auc_weighted`/`g_score` crash the entire search (`modeling/_tuning/metrics.py:19-36,127-146`) | small | ⬜ open |

### Remaining — frontend

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-54 | 🟡 | `DebugNode` is dead code that would silently no-op if wired up (`nodes/DebugNode.tsx`) | small | ⬜ open |
| OC-55 | 🟡 | `tsc --noEmit` fails: `mermaid` declared but not installed (`frontend/ml-canvas/package.json`) | 1 line | ⬜ open |
| OC-56 | ⚪ | `useSchemaPreview` does not cancel in-flight requests on unmount (`hooks/useSchemaPreview.ts`) | small | ⬜ open |
| OC-57 | ⚪ | `any`-typed chart props bypass type safety in EDA components (`modules/eda/`) | small | ⬜ open |

### Remaining — tests / packaging / CI (outside the Ongoing tier)

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-80 | 🟡 | 3 weakest-covered modules untested exactly where silence is dangerous (`_sklearn_compat.py`, `value_replacement.py`, `config_validation.py`) | ~1 day | ⬜ open |

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

## Log

### 2026-09-04 — OC-53 fixed: `select_from_model`'s `max_features` now reachable from the canvas
OC-53 closed. Root cause: the backend (`feature_selection/_common.py` `_build_model_selector`) reads `config.get("max_features")` and passes it to sklearn's `SelectFromModel`, and `@node_meta` declares it — but the UI's `select_from_model` branch (`FeatureSelectionNode.tsx`) rendered only a `threshold` field, so the cap was Python-only and unreachable from the canvas. Fix (frontend-only): added `max_features?: number` to the `FeatureSelectionConfig` interface and an optional "Max Features" numeric input in the `select_from_model` branch (mirroring the `k` field pattern; empty = no cap, matching the backend's `None` default). No converter change: the `feature_selection` branch already passes `node.data` through unchanged, so the field flows to the backend automatically. Added 2 vitest cases to `pipelineConverter.test.ts` (cap forwarded; omitted when unset). Verified: 40/40 `pipelineConverter.test.ts` pass, `npm run lint` clean, `npm run build` clean.

### 2026-09-04 — OC-20 fixed: Value Replacement UI help text now matches empty-columns behavior
OC-20 closed. Root cause: the UI (`ValueReplacementSettings.tsx`) help text promised "If empty, applies to all compatible columns," but the backend no-ops on an empty `columns` list — which is the intended repo convention (`user_picked_no_columns` in `skyulf/utils.py`: "When every box is unchecked, the user's intent is unambiguously 'do nothing for this node'"). The fix aligns the UI text with actual behavior rather than changing the backend to apply-to-all (which would contradict the documented convention). Fix (frontend-only): the help text now reads "Select columns to apply replacements to. If no columns are selected, this node does nothing." No backend change: `value_replacement.py` already no-ops correctly on empty columns. Added a backend contract-locking test `test_apply_empty_columns_is_noop` (parametrized pandas/polars) asserting empty `columns` + a mapping leaves data unchanged. Verified: 25/25 `test_value_replacement.py` pass, `ruff check`/`ruff format`/`ty check` clean, frontend `npm run build`/`lint` clean.

### 2026-09-04 — OC-19 fixed: Alias Replacement `punctuation` mode now strips punctuation
OC-19 closed. Root cause: the UI (`AliasReplacementNode.tsx`) offers a `punctuation` mode ("Removes common punctuation characters from text"), but the backend's `_apply_polars`/`_apply_pandas` in `cleaning/alias.py` only had branches for the alias-mapping modes — `punctuation` fell through to the mapping path, whose resolved mapping is `{}` for this mode, so the applier was a silent no-op. Fix (backend, both engine paths): a dedicated `punctuation` branch that strips `string.punctuation` only — case and spaces are preserved, matching the UI wording (unlike the mapping modes, which fully normalise). Polars: `str.replace_all` with the escaped punctuation class (nulls pass through unchanged). Pandas: `str.translate(ALIAS_PUNCTUATION_TABLE)` with the original NaN restored after `astype(str)`. No converter/UI change: `mode`/`alias_type` already flow through verbatim. Added 3 JSON cases (`type_resolution` pass-through, `resolve_mapping` empty, `applier_value_lists` strip case) + a direct pandas/polars NaN-passthrough parity test. Verified: 45/45 `test_cleaning_alias.py` pass, `ruff check`/`ruff format`/`ty check` clean.

### 2026-09-03 — OC-15 fixed: MinMax/Robust scaler range controls honored on canvas
OC-15 closed. Root cause: the UI (`ScalingNode.tsx`) stores scaler ranges as scalar fields (`feature_range_min`/`feature_range_max` for minmax, `quantile_range_min`/`quantile_range_max` for robust), but the converter's `scale_numeric_features` branch did `params = config` — forwarding the scalars verbatim while the backend reads tuple keys (`feature_range` in `scaling/minmax.py`, `quantile_range` in `scaling/robust.py`) — so every canvas run silently used the defaults (0/1 and 25/75). Fix (converter-only, `pipelineConverter.ts`): the branch now assembles `feature_range` (minmax, defaults 0/1) and `quantile_range` (robust, defaults 25/75) from the scalar fields; scalar keys are left in the payload (harmless — the backend ignores unknown keys). No backend change: the contract was already correct and covered by `test_scaling.py`. No doc change: `docs/reference/preprocessing_nodes.md` already documents the tuple keys. Added 4 vitest cases to `pipelineConverter.test.ts` (minmax custom range, robust custom range, defaults when absent, absent for standard/maxabs). Verified: 865/865 vitest pass, `npm run lint` clean, `npm run build` clean.

### 2026-09-03 — OC-14 fixed: Iterative Imputer canvas estimator choices honored
OC-14 closed. Root cause: the UI (`ImputationNode.tsx`) emits lowercase aliases (`bayesian_ridge`, `decision_tree`, `extra_trees`, `knn`) and the converter forwards `estimator` verbatim, but `_build_iterative_estimator` in `imputation/_common.py` matched only the exact documented strings (`DecisionTree`, `ExtraTrees`, `KNeighbors`) — so every canvas run silently fell back to `BayesianRidge` regardless of the user's choice. Fix (backend normalization, single owner of the mapping): the alias is now lowercased and stripped of non-alphanumerics before dispatch, so both the UI values and the documented aliases resolve to the same regressor; unknown names still fall back to `BayesianRidge`. No frontend or converter change needed. Added 4 JSON-driven cases to `iterative_estimator_aliases` (`ui_decision_tree`, `ui_extra_trees`, `ui_knn`, `ui_bayesian_ridge`). Verified: 74/74 `test_imputation_common_knn_iterative_simple.py` pass, `ruff check`/`ruff format`/`ty check` clean.

### 2026-09-03 — OC-13 fixed: Drop-Rows percentage threshold now reaches the backend
OC-13 closed. Root cause: the UI (`DropRowsNode.tsx`) stores `{drop_if_any_missing, missing_threshold}` (a 0–100 **percentage** slider, default 50), but the converter sent those keys verbatim while the backend node only reads `subset`/`how`/`threshold` (absolute non-missing count) — so every canvas run silently ran as `how="any"` (drop any missing). A frontend-only fix is impossible: percentage→absolute conversion needs the column count, unknown at conversion time. Fix (backend percentage mode, mirroring `DropMissingColumns`): `drop_rows.py` gains a `missing_threshold` param — new `_min_non_na_for_percentage` helper, percentage branch in both `_polars_dropna_filter` and `_drop_missing_rows_apply_pandas` (keep rows with `non_na >= (1 - X/100) * n_cols`, i.e. drop rows missing **more than** X% — exactly the UI wording; a row at exactly X% is kept), exposed in `@node_meta` params and `fit()`; `DropMissingRowsArtifact` gains `missing_threshold: float | None`; converter now maps checkbox/null/≤0 → `{how: "any"}`, else `{missing_threshold: X}`. No leakage change needed: the node is `learns_from_data=False` and the threshold is a fixed user setting, not learned data. Added 7 tests (fit preservation/default, pandas percentage drop, boundary keep-at-exact-share, subset respect, tuple X/y sync, polars parity) + a `percentage_threshold` round-trip case in `drop_rows.json`. Verified: 29/29 `test_drop_rows.py` pass, `ruff check`/`ruff format`/`ty check` clean, frontend `npm run build`/`lint`/861 vitest pass.

### 2026-09-03 — OC-62 fixed: object-dtype arrays digested by value, not by pointer
OC-62 closed. Root cause: `_feed_canonical` in `pipeline/seal.py` digested `np.ndarray` via `arr.tobytes()`; for `dtype=object` arrays that serialises raw `PyObject*` pointers, which are allocator/ASLR dependent — so `fingerprint()` of any artifact holding an object-dtype array (OneHotEncoder/LabelEncoder/Ordinal/TargetEncoder `categories_`) was noise that changed across processes. Fix: the ndarray branch now detects `arr.dtype == object` and digests the shape plus each element recursively via `_feed_canonical`, so the digest reflects values. Added three regression tests in `tests/unit/test_pipeline_coverage.py` (value-vs-pointer stability incl. non-interned strings, shape sensitivity); verified the new tests FAIL on the pre-fix code (`b'\xca' != b'\xf5'`) and pass with the fix. Verified: 30/30 `test_pipeline_coverage.py` pass, `ruff check`/`ruff format`/`ty check` clean.

### 2026-09-03 — OC-58 fixed: polars numeric→bool cast mirrors pandas 0/1 semantics
OC-58 closed. Root cause: `_build_polars_cast_exprs` only special-cased string/categorical→bool; numeric→bool fell through to the generic `pl.col(col).cast(pl.Boolean, strict=...)`, which is C-style truthiness (`x != 0`) and never raises — so `2.0` silently became `True` on polars while pandas `astype("boolean")` produced `<NA>` (and raised `TypeError` in strict mode). Fix in `skyulf-core/skyulf/preprocessing/casting.py`: new `_bool_expr_from_numeric_col_polars` helper builds `pl.when(col == 0).then(False).when(col == 1).then(True).otherwise(None)`, so only exact 0/1 values map to booleans and everything else (including non-integer floats) becomes null; the column is tracked in the bool-cast list and validated in strict mode by the renamed `_validate_polars_bool_casts` (raises `ValueError` on newly-null values, matching the existing string→bool strict behavior). Added four regression tests in `tests/integration/test_casting.py` (coerce nulls, strict raise, pure 0/1, engine parity). Verified: 85/85 `test_casting.py` pass, `ruff check`/`ruff format`/`ty check` clean.

### 2026-09-03 — OC-12 fixed: positional keep-mask for pandas X/y desync
OC-12 closed. Root cause: the pandas paths of `DropMissingRows` and `Deduplicate` selected `y` by label (`y.loc[X_clean.index]`); with duplicate index labels `.loc` returns *all* matching rows, so `y` came back longer than `X` with wrong labels — silent X/y desync. Fix mirrors the already-correct polars paths: compute a positional keep mask (`notna` threshold / `duplicated`), take `kept_positions = np.flatnonzero(mask)`, select `X.iloc[kept_positions]`, and filter `y` positionally via the new `_pandas_filter_y_by_kept_positions` helper in `_common.py` (`.iloc`, `None` passthrough). Note: `X.index.get_indexer(X_clean.index)` was rejected as a recovery path — it returns the *first* occurrence for duplicate labels. Added two duplicate-index regression tests (`test_drop_rows.py`, `test_drop_and_missing_gaps.py`). Verified: 62/62 targeted tests pass, `ruff check`/`ruff format`/`ty check` clean.

### 2026-09-02 — OC-75 fixed: stale nested `uv.lock` removed, benchmark guarded
OC-75 closed. Root cause of the stale lockfile: the repo is a **uv workspace** (root `pyproject.toml` declares `[tool.uv.workspace] members = ["skyulf-core"]`), so the **root `uv.lock` is the single source of truth** (already pins polars 1.44.1) and `skyulf-core/uv.lock` was a redundant pre-workspace leftover — `uv lock` from inside `skyulf-core/` rewrites the *root* lockfile, never the nested one, so the audit's "regenerate `skyulf-core/uv.lock`" instruction is not a normal uv operation in a workspace. Fix: `git rm skyulf-core/uv.lock` (CI never uses it — all workflows install via `uv pip install -r requirements-ci.txt`; only dependency-review/labeler reference `**/uv.lock`). Also applied the audit's secondary fix: `benchmarks/bench_roundtrip_removal.py`'s per-bench loop now uses the same try/except-and-skip pattern as `bench_engine_comparison.py` (a failing node prints `SKIP (Type: msg)` instead of crashing the table). Verified: `uv lock --check` exit 0, 47/47 `test_split.py` pass, benchmark runs clean (3 nodes, no crash), ruff clean.

### 2026-09-02 — OC-75 re-verified: venv already fixed, `uv.lock` still stale
Re-checked OC-75 before starting: the venv already has polars **1.43.2** (floor met) and all 47 `skyulf-core/tests/integration/test_split.py` tests pass — the "10 failed" state from the audit is gone in the venv. But `skyulf-core/uv.lock` still pins **polars 1.36.1** (below the `>=1.43.2` floor in `setup.py`/`pyproject.toml`/`requirements-ci.txt`), and `uv lock --check` reports the lockfile stale. Remaining work: `uv lock` to regenerate, re-run the 10 polars split tests, commit.

### 2026-09-01 — Tracker created

- Tracker created from the master report's 116-finding inventory and its 4-tier
  suggested fix order. All items ⬜ open; no code changed.
- Corrections carried over: OC-100 retracted (false positive), OC-01 corrected,
  OC-46 downgraded 🟠→🟡, OC-12/18/40/42 worse than filed on execution re-verification.
- OC-71 (no authn/authz) is gated on a deployment-model decision before any fix
  work is scoped.
