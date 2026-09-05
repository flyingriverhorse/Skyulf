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
| OC-39 | **Worse than filed** — the median is not merely `nan`, it is silently *wrong* (`[1, 2, nan, 4]` → `3.0` where pandas gives `2.0`), and the histogram builder raises `ComputeError: breaks cannot be NaN` |
| OC-43 | **Corrected** — the claim that pandas' pairwise deletion returns all-`1.0` for the sparse frame is wrong; pandas also returns all-`NaN` there, so `None` was already the right answer |
| OC-45 | **Partially retracted** — schema drift *was* already rendered (`SchemaDriftPanel`); only `drifted_columns_count` omitted it |
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
| OC-61 | ⚪ | `BinningNode`'s "Precision (Decimals)" UI field never sent to backend (`BinningNode.tsx`) | small | ✅ fixed 2026-09-04 |
| OC-66 | 🟠 | `CalibratedClassifierCV`'s user-selected base estimator silently discarded during tuning (`classification.py:206-282` vs `_tuning/engine.py:495-499`) | small | ✅ fixed 2026-09-04 |
| OC-16 | 🟠 | KNN/Iterative imputers crash on all-missing fitted columns (`imputation/knn.py:64-76`, `iterative.py:68-84`) | small | ✅ fixed 2026-09-04 |
| OC-17 | 🟠 | SimpleImputer polars mean/median crashes on all-null columns (engine divergence, `imputation/_common.py:32-37`) | small | ✅ fixed 2026-09-04 |
| OC-69 | 🟠 | Engine trusts `config.nodes` list order, never verifies topological sort (`_schema_graph.py:49-70`); `_kahn_topological_order` already exists — wiring fix | small | ✅ fixed 2026-09-04 |
| OC-35 | 🟠 | Multiclass splits missing a class emit binary-only metrics + null curve points (`metrics.py:217-237,361-363`) | small | ✅ fixed 2026-09-04 |
| OC-36 | 🟠 | F1 threshold tuning picks pathological threshold on single-class validation (`thresholds.py:101-111`) | small | ✅ fixed 2026-09-05 |
| OC-39 | 🟠 | NaN-bearing numeric columns publish `nan` stats and leak non-finite JSON (`profiling/analyzer.py:215-224`) | small | ✅ fixed 2026-09-05 |
| OC-40 | 🟠 | PCA/clustering "mean imputation" actually replaces NaN with `0.0` (`multivariate.py:46-60`) | small | ✅ fixed 2026-09-05 |
| OC-41 | 🟠 | Quartiles use nearest-rank, not linear interpolation (disagrees with pandas) (`analyzer.py:221-222`) | small | ✅ fixed 2026-09-05 |
| OC-42 | 🟠 | Skewness/kurtosis use biased estimators, breaking the hardcoded threshold rule (`analyzer.py:223-224`) | small | ✅ fixed 2026-09-05 |
| OC-43 | 🟠 | Correlation drops valid columns/rows instead of the defined missing-data policy (`correlations.py:41-44,100-110`) | small | ✅ fixed 2026-09-05 |
| OC-44 | 🟠 | Wasserstein drift thresholds normalized value but reports raw one (`drift.py:181-195`) | small | ✅ fixed 2026-09-05 |
| OC-45 | 🟠 | Schema drift computed but never counted or rendered as drift (`drift.py:76-98`) | small | ✅ fixed 2026-09-05 |
| OC-46 | 🟡 | Non-finite floats reach public payloads; only stdlib-json paths emit invalid JSON (`schemas.py:7-17,263-302`) | small | ✅ fixed 2026-09-05 |

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
| OC-146 | 🔴 | Binary `pr_auc` scored against wrong class on `{1,n}` labels — reports 0.32 vs true 0.97, no warning (`metrics.py:324-326`) | small | ✅ fixed 2026-09-05 |
| OC-149 | 🟠 | Clustering evaluation crashes on polars when a numeric feature is all-null within one cluster (`clustering.py:83-88`) | small | ⬜ open |
| OC-37 | 🟡 | Binary PR-AUC dropped for string-labeled classifiers (`metrics.py:324-327`) | small | ✅ fixed 2026-09-05 — same one-arg fix as OC-146 |
| OC-148 | 🟡 | PII detector flags ordinary 7+ digit numeric ID columns as "Email/Phone" (`profiling/_analyzer/text.py:107-128`) | small | ⬜ open |
| OC-147 | ⚪ | `optimize_thresholds` returns a dict shape that bypasses its own documented binary rule, flipping `>=` to `>` on exact ties (`thresholds.py:66-88`) | small | ✅ fixed 2026-09-05 — with OC-36 |
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

### 2026-09-05 — OC-39/40/41/42/43/44/45/46 fixed: the profiling cluster closes, four findings from one root cause
The whole cluster is two root causes and one design decision. **Root cause 1 — polars keeps NaN distinct from null** (OC-39, OC-40, OC-43): `fill_null` is a no-op on NaN, `drop_nulls()` keeps NaN rows, aggregations *propagate* NaN instead of skipping it, `std()` of a NaN-bearing column is NaN, and `nan == nan` is False. Every repro had to be built with `pl.DataFrame({"x": [1.0, 2.0, float("nan"), 4.0]})` directly — `pl.from_pandas` converts NaN→null and therefore masks all three bugs, which is why the existing suites never caught them. **OC-39** was worse than filed: not just NaN stats, but a silently *wrong* median (`3.0` where pandas gives `2.0` on `[1, 2, nan, 4]`) and a `ComputeError: breaks cannot be NaN` in the histogram builder, whose `min_val == max_val` guard cannot see that NaN never equals itself. Fixed once at the boundary instead of at six call sites: `_nan_to_null` in `EDAAnalyzer.__init__` rewrites NaN→null in every float column. **OC-40**: `_impute_matrix`'s `fill_null(strategy="mean")` no-oped, so values fell through to `np.nan_to_num(nan=0.0)` — PCA/clustering were fitted with `0.0` exactly where `SimpleImputer` would have put the column mean, which for a mean-centered feature is the most distorting value available. Now mirrors its sibling `_impute_matrix_drop_empty`. **OC-43**: `drop_nulls()` + `DataFrame.corr()` is *listwise*, so one surviving null made the entire matrix NaN, the broad `except` swallowed it, and the profile lost its correlation section. Now pairwise via `pl.corr` (which already does pairwise deletion, matching pandas `.corr()` — no hand-rolled covariance math needed). The audit's supporting claim here was **wrong**: it asserted pandas' pairwise deletion returns all-`1.0` for the sparse frame, but pandas also returns all-`NaN`, so `None` was already correct for that input. `correlations.py` deliberately *duplicates* the 3-line `_nan_to_null` helper rather than importing `_analyzer._utils` — that import executes `_analyzer/__init__.py` and drags the sklearn/scipy/statsmodels mixins into a leaf module. **Root cause 2 — polars' estimator defaults are not pandas'** (OC-41, OC-42): `Expr.quantile()` defaults to `interpolation="nearest"` (pandas/NumPy use linear), and `skew()`/`kurtosis()` default to `bias=True` with non-Fisher kurtosis (pandas reports bias-corrected, and `.kurt()` is Fisher excess). OC-42 was user-visible, not merely numerical: `[1, 2, 3, 4, 10]` reports `1.30` biased against `1.70` unbiased, so the "High skewness — consider a transform" recommendation stayed silent on a column that clears the 1.5 threshold; post-fix it emits `('x', 'Transform', 'High skewness (1.70)')` where previously only `(None, 'Keep')` appeared. Pinned at both quantile sites (`_numeric_advanced_aggs` and `_compute_boxplot_stats`, which disagreed with `NumericStats` about the same column's hinges). **OC-44** was a design decision, not just a patch: the drift verdict has always been made on the std-normalized Wasserstein distance while `value` carried the raw one, so a large-scale column that barely moved reported `value=50.0, threshold=0.1, has_drift=False` — measured `0.017318` normalized against `50.0000` raw. Rather than teach each consumer a metric-specific exception (which is what an added `normalized_value` field would have required, and which the UI would have to keep re-implementing), the invariant now lives in the schema: `value` is the number `threshold` applies to, and the untransformed distance moved to a new `raw_value`. Evidence that this is the scale every consumer already assumed: `tests/unit/test_drift.py:25/46` assert `value < 0.1` / `> 0.1`, `docs/user_guide/drift_monitoring.md` documented the threshold as "0.1 (normalized)", and the published example notebook shows the failure outright — `784.7832 (Thresh: 0.1) [PASS]`. Option B fixed `DriftAlertModal`, `csvExport`, `_build_drift_column_summary` and the persisted `drift_check_results.summary` with **zero edits** to any of them; no backend model change was needed either, since `EnrichedDriftReport.column_drifts` is `dict[str, Any]`. A constant reference has no scale to normalize by and falls back to the raw distance rather than emitting `inf` (verified finite). The one documented exception is `ks_test_p_value` — diagnostics only, borrows the KS statistic's threshold (F-12). **OC-45**: `drifted_columns_count` was rebuilt from per-column metric flags alone, so a vanished feature left it at `0` while `_classify_drift_severity` called that same report `"critical"` and `DriftStatusSummary` counted drifted jobs by that field — a critical job reported as having no drift. The finding's "never rendered" half was **already handled** by `SchemaDriftPanel` (`DataDriftPage.tsx:168`); only the count was missing, so a banner first added to `DriftTable` was removed as a duplicate of it. `_classify_drift_severity` needed no change: it returns `"critical"` early on any schema drift, so the `count / len(column_drifts)` ratio path never sees a schema-inflated numerator. **OC-46**: the ten optional float fields of `NumericStats` now run through a `FiniteFloat` annotated type mapping any non-finite float to `None` before validation — `backend/eda/tasks.py` persists via `model_dump(mode="json")`, which retains a Python `nan`, and stdlib `json.dumps` then emits a bare `NaN` token that the browser's `JSON.parse` rejects. `BoxPlotStats` left alone (required fields; making them optional would be a real contract change, not a fix). **Tests**: 22 added — 13 profiling (6 analyzer parity, 4 correlations, 2 multivariate, 1 target boxplot), 5 core drift, 1 backend cross-layer, 3 frontend hook. Proven genuine by reverting only the fixed source to `HEAD` and re-running: 10 of the 13 profiling tests failed pre-fix with the exact pathological values, 2 passed by design (guards pinning behaviour that must not change — all-missing column zero-fill, all-missing column dropped from the matrix), all 5 core drift tests failed, the backend one failed (`assert 0 == 1`), and 2 of the 3 frontend ones failed (`expected +0 to be 1`). The third frontend test (Wasserstein verdict) passes either way by construction — the fix moved into the backend, so what it pins is the contract, not a regression. One test was strengthened after it initially passed pre-fix: the `value > threshold` sweep needed a large-scale column, because on unit-scale data raw and normalized coincide and the invariant holds by accident. **Gates**: core **3661 passed / 70 skipped**, backend **1547 passed**, `ruff check` + `ruff format --check` (661 files) clean, `ty check` clean, frontend `eslint` clean, **872 tests passed**, `npm run build` succeeded (which caught a `noUncheckedIndexedAccess` error vitest does not typecheck). Docs: `drift_monitoring.md` metric table corrected (Wasserstein normalization, KS statistic not p-value) plus a new `value`/`threshold`/`raw_value` contract note; `eda_profiling.md` metric descriptions and schema-drift counting corrected. Also fixed a **doc bug found in passing**: the custom-thresholds example passed `"ks": 0.01`, but the calculator reads `ks_statistic` — unknown keys are merged and never read, so the documented override silently did nothing. `mkdocs build --strict` NOT run — mkdocs is not installed in this venv. **Residuals left open deliberately**: (1) `CorrelationMatrix.values` is `list[list[float]]` with no "unknown" cell, so pairs with fewer than `MIN_PAIRWISE_OVERLAP = 3` overlapping observations are reported as `0.0` with one aggregated warning — at n=2 a Pearson r is always exactly ±1.0, and representing "not computable" properly needs a schema change; (2) OC-42 surfaced a cross-layer inconsistency that is a product decision, not a code fix — core `recommendations.py` uses `SKEWNESS_TRANSFORM_THRESHOLD = 1.5` while `backend/ml_pipeline/_internal/_advisor.py:195` applies `1.0` to pandas unbiased skew, so two rules govern one concept; (3) persisted `drift_check_results.summary["wasserstein"]` rows written before this change hold raw distances and new rows hold normalized ones, so history is not scale-comparable across the boundary — left un-migrated because the old values were the misleading ones.

### 2026-09-05 — OC-146 fixed (last 🔴 closed): binary `pr_auc` now scores the class the model treats as positive; OC-37 closed by the same change
OC-146 and OC-37 closed together — one argument, two failure modes. Root cause: `_add_roc_pr_auc_metrics`'s binary branch called `average_precision_score(y_arr, proba[:, 1])` with no `pos_label`. That function's default is `pos_label=1`, unlike `roc_auc_score`, which infers the positive class from the sorted uniques (which is why `roc_auc` was correct in all nine of the audit's label encodings and `pr_auc` was not). With labels `{1,2}`/`{1,5}` the literal `1` names the *negative* class while `proba[:, 1]` is `P(classes_[1])`, so PR-AUC was computed for the inverted problem — re-verified locally on 400 rows of learnable signal: **0.3123 reported vs 0.9718 true**, no exception, no warning. For every other non-`{0,1}` label set (`{2,3}`, `{0,2}`, `{10,20}`, `{"no","yes"}`) sklearn raised `pos_label=1 is not a valid label` and `_try_add_metric` dropped the key, which is OC-37. The report also contradicted itself on screen: `classification.py:99` builds the PR *curve* with `pos_label=classes[1]`, so one `CurveData` carried a curve drawn for class 2 with an AUC computed for class 1. Fix (core-only, `metrics.py`): resolve `pos_label` from `model.classes_` — the pattern `_add_binary_unweighted_metrics` (30 lines above) and the curve builder already used; the `else 1` fallback only preserves today's behaviour for stub models that expose no `classes_` (any real sklearn classifier's `classes_` length equals `proba.shape[1]`). Deliberately *not* `pos_label=None`: sklearn 1.8 rejects it with `InvalidParameterError`, which `_try_add_metric` would swallow and drop the metric. Verified post-fix: `pr_auc` is bit-identical (`0.9845354656027469`) and equal to the `pos_label=classes_[1]` ground truth across `{0,1}`, `{1,2}`, `{1,5}`, `{-1,1}`, `{"no","yes"}`. Added 7 tests to `test_evaluation_metrics.py` (5-encoding parametrized ground-truth invariance + cross-encoding agreement). Confirmed 4 of them FAIL on pre-fix code (`assert 'pr_auc' in {...}` for strings, comparison failures for `{1,2}`/`{1,5}`). Full core suite: **3643 passed, 70 skipped**; `ruff check` / `ruff format --check` / `ty check` clean.

### 2026-09-05 — OC-36 fixed: threshold tuning no longer returns a pathological cutoff; OC-147 closed alongside (search and apply now share one decision rule)
OC-36 and OC-147 closed together — both live in the `_grid_search_binary` ↔ `apply_thresholds` pair and both come from the same habit of letting an arbitrary tie-break decide. **OC-36** root cause: `_grid_search_binary` kept a candidate only on a strict `score > best_score`, so any tie left the *first-scanned* candidate standing. On a validation split holding one class every candidate scores identically (F1 = 0.0 throughout), so tuning returned the first grid point — re-verified: `{0: 0.9901960784313726, 1: 0.00980392156862745}`, predicting **49/50 rows positive at F1 0.0**, then persisted as a tuned threshold. Because F1 is piecewise constant in the cutoff, tied plateaus are routine on healthy splits too, where the same rule pinned whichever plateau edge came first. Fix (two layers): (1) core — `_grid_search_binary` now warns and returns the neutral `{0.5, 0.5}` when `len(np.unique(y_true)) < 2`, and breaks score ties toward the `0.5` cut (NaN scores from a caller-supplied metric still fail both comparisons and are skipped, preserving the existing fall-back-to-0.5 tolerance); (2) tuning — `_tune_decision_thresholds` gains a sixth gate alongside its five existing ones, so a degenerate split leaves `decision_thresholds=None` and `predict()` keeps the model's default rule instead of the UI reporting a threshold nothing was tuned against. Post-fix the same repro returns `{0.5, 0.5}` with a logged warning and 24/50 positives — identical to the default rule. **OC-147** root cause: `apply_thresholds` documents "predicts the positive class when `y_proba[:, 1] >= threshold`" and special-cased only a bare float and a *one*-entry dict, but `_grid_search_binary` returns a *two*-entry dict — which fell through to the multiclass scaled argmax, where `np.argmax` breaks exact ties toward the first column, silently turning `>=` into `>`. The search scores candidates with `>=` (line 106), so at a tie the tuned score it reported was not the score apply-time produced; reachable because the grid includes `0.5` and trees routinely emit `p1` of exactly `0.5`. Fix: the two-class case now compares `scaled[:, 1] >= scaled[:, 0]` directly. Chosen over the audit's alternative (return a one-entry dict) for two verified reasons: backend `_validate_save_payload` requires threshold keys to cover *every* class, and a user can save a non-complementary binary pair — the naive "just read `thresholds[classes[1]]`" fix would silently ignore that entry and create a fresh OC-13-class "UI setting ignored" bug. Measured: 0 differing rows out of 4000 for each of `(0.5,0.5)`, `(0.99,0.01)`, `(1.0,0.1)`, `(3.0,0.25)`, and 0/100k for search-vs-apply at `t=0.3/0.5/0.7`; only the exact tie flips (`p1==t==0.5`: argmax `[0]` → now `[1]`, matching the bare-float form). Docs: `docs/user_guide/threshold_tuning.md` gains the new gate bullet and the tie-break semantics. Added 6 tests (4 in `test_evaluation_thresholds.py`, 1 gate test in `test_tuning_engine.py`, plus a non-complementary-pair guard that pins both entries still being honored); confirmed 4 FAIL pre-fix with the exact pathological values above. Verification: 613 passed on the evaluation/tuning/classification/threshold selection, **3643 passed** full core suite, **1546 passed** full backend suite (threshold + deployment paths included), `ruff check` / `ruff format --check` / `ty check` clean. `mkdocs build --strict` NOT run — mkdocs is not installed in this venv.

### 2026-09-04 — OC-35 fixed: multiclass splits missing a class no longer emit binary-only metrics or null curve points
OC-35 closed. Root cause: three places decided binary-vs-multiclass from the labels present in `y_true` instead of the model's trained label set, so a 3-class model evaluated on a split containing only two classes was misclassified as binary. (1) `_add_binary_unweighted_metrics` (`metrics.py`) gated on `len(np.unique(y_arr)) != 2`, so such a split gained unweighted `precision`/`recall`/`f1` keys that don't belong to a multiclass model. (2) `_add_probability_based_metrics` (`metrics.py`) called `log_loss(y_arr, proba)` with no `labels=`, so sklearn raised "Number of classes in y_true not equal to columns in y_score (2 vs 3)" and the metric was silently dropped. (3) `evaluate_classification_model`'s multiclass loop (`classification.py`) ran `roc_curve`/`precision_recall_curve` on the all-zero one-vs-rest target of the absent class, which returns NaN points that serialize as `null` curve coordinates. Fix (core-only): (1) the binary gate now resolves `classes_ = getattr(model, "classes_", None)` and is binary iff `len(classes_) == 2` (falling back to the unique-label count only when the model exposes no `classes_`, preserving the `model=None` test path); (2) `log_loss` is now called with `labels=classes` where `classes` is `model.classes_` when its length matches the proba column count, else `np.arange(class_count)`; (3) the per-class loop skips any class whose binarized target has fewer than 2 unique values (absent from the split) with a `logger.debug`. Added 3 regression tests to `test_evaluation_metrics.py` (missing-class split keeps multiclass-only keys, `log_loss` present and equal to sklearn ground truth with `labels=[0,1,2]`, and `evaluate_classification_model` emits no non-finite curve points and no curve for the absent class). Verified: 32/32 `test_evaluation_metrics.py` pass, full evaluation suites (7 files) 148 passed, `ruff check` / `ruff format --check` / `ty check` clean.

### 2026-09-04 — OC-69 fixed: engine and schema predictor no longer trust `config.nodes` list order
OC-69 closed. Root cause: both `predict_schemas` (`_schema_graph.py`) and `_run_node_loop` (`engine/__init__.py`) iterated `for node in config.nodes:` assuming the list was already topologically sorted, but `validate_no_cycles()` only detects cycles — it never verifies or restores order. The frontend's `pipelineConverter.ts` BFS enqueues a merge node when *any* parent is dequeued (not all), so the UI can emit an acyclic-but-misordered list (e.g. diamond merge fed by unequal-depth branches), which then produced a cryptic "Artifact not found" engine failure and silent `None` schema degradation. Fix (backend-only): new public `topological_order(nodes)` in `graph_utils.py` reusing the existing private `_build_in_degree_and_children` + `_kahn_topological_order` helpers (no duplication), called at both consumer sites before iteration. Added 5 regression tests in `tests/integration/test_topological_order.py` (misordered-diamond sort, already-sorted preservation, idempotence, `predict_schemas` on misordered diamond, full engine e2e run of a misordered diamond). Verified: 5/5 new tests pass, related suites (schema graph, merge input order, parallel partitioning, execution, engine recording) 49 passed, `ruff check` / `ruff format --check` / `ty check` clean.

### 2026-09-04 — OC-17 fixed: SimpleImputer polars mean/median no longer crashes on all-null columns
OC-17 closed. Root cause: when a `SimpleImputer` is fit with `mean`/`median` on a column that is entirely null, the polars fit path stores `fill_values[col] = None` (polars `mean()`/`median()` over an all-null column returns `None`), and `_apply_polars` then called `pl.col(col).fill_null(None)` — which raises `ValueError: must specify either a fill value or strategy`. The pandas path already guarded this with `if val is None: continue`, so the two engines diverged (pandas left the column all-null; polars crashed). Fix (core-only, `imputation/simple.py` `_apply_polars`): extract `val = fill_values[col]` and, when `val is None`, append `pl.col(col)` as a passthrough (column stays all-null) and `continue`, mirroring `_apply_pandas`. The restore branch (`pl.lit(fill_values[col]).alias(col)` for fit-time-missing columns) is already parity-correct — `pl.lit(None)` yields an all-Null column — so no change there. Added 3 regression tests to `test_imputation_common_knn_iterative_simple.py` (polars apply no-crash for mean, no-crash for median, and a pandas/polars engine-parity test asserting the all-null column stays all-null in both engines; the polars frame uses an explicit `Float64` all-null column since `pl.from_pandas` infers the pandas object column as `String`). Verified: 82/82 imputation tests pass, `ruff check` / `ruff format --check` / `ty check` clean.

### 2026-09-04 — OC-16 fixed: KNN/Iterative imputers no longer crash on all-missing fitted columns
OC-16 closed. Root cause: sklearn's `KNNImputer`/`IterativeImputer` silently drop all-missing columns from `transform()` output when the column was all-missing **at fit time**, desyncing the artifact's `columns` list from the imputer's width and crashing `_sklearn_transform_subset` (IndexError in the polars branch, ValueError in the pandas branch). Fix (core-only, `imputation/_common.py`): new `drop_all_missing_columns` helper called from both `KNNImputerCalculator.fit` and `IterativeImputerCalculator.fit` — drops all-missing columns from the fit matrix so the artifact's `columns` stays in lockstep with the imputer's width, logs a warning naming the dropped columns, and `fit` returns `{}` when every configured column is all-missing (appliers already pass empty artifacts through as a no-op). Added 5 regression tests to `test_imputation_common_knn_iterative_simple.py` (fit drops all-missing column + warning, all-columns-all-missing empty artifact, helper unit test; parametrized over KNN + Iterative, pandas + polars). Verified: 79/79 imputation tests pass, `ruff check` / `ruff format --check` / `ty check` clean.

### 2026-09-04 — OC-66 fixed: `CalibratedClassifierCV`'s base estimator now survives tuning
OC-66 closed. Root cause: the tuning engine builds the meta-estimator from `model_calculator.default_params` (`_tuning/engine.py:496`, `refit.py:44`, `grid_random.py:139`), and `CalibratedClassifierCalculator.default_params` hardcoded `estimator=LogisticRegression` — so the user's `base_estimator` selection (read only by `fit` via `_resolve_base_estimator`) was silently discarded whenever the node was tuned. Fix (core-only): routed the selection through the established structural-tuning hook (the same mechanism `_BaseEnsembleCalculator` uses) — `CalibratedClassifierCalculator` now declares `STRUCTURAL_TUNING_KEYS = ("base_estimator",)`, captures the selection in `prepare_tuning_params` (flat or nested `params` config shape), and its `default_params` override resolves it via the `BASE_ESTIMATORS` factory into `estimator` (unknown keys warn + fall back to `logistic_regression`). No backend change: both the fixed-run and tuned paths in `_node_runners.py` already call `prepare_tuning_params` and exclude `STRUCTURAL_TUNING_KEYS` from the search space. Added unit tests in `test_modeling_classification_gaps.py` (flat/nested capture, non-structural key exclusion, `default_params` resolution, unknown-key fallback, no-prepare default) plus an integration test in `test_tuning.py` asserting the tuned pipeline's fitted model is a `RandomForestClassifier` inside `CalibratedClassifierCV`. Verified: `ruff check`/`ty check` clean, OC-66 suites 20 passed, broader classification+tuning suites (6 files) 173 passed.

### 2026-09-04 — OC-61 fixed: `BinningNode`'s "Precision (Decimals)" now reaches the backend
OC-61 closed. Root cause: the backend (`bucketing.py`) reads `config.get("precision", 3)` and the canvas `BinningNode` rendered a "Precision (Decimals)" input, but `pipelineConverter.ts`'s `BinningNode` branch listed its params explicitly and omitted `precision`, so the value was silently dropped before reaching the backend. Fix (frontend-only): added `precision: node.data.precision` to the `GeneralBinning` params object. Added 2 vitest cases to `pipelineConverter.test.ts` (precision forwarded; omitted when unset). Verified: 42/42 `pipelineConverter.test.ts` pass, `npm run lint` clean, `npm run build` clean.

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
