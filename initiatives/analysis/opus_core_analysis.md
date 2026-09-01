# Skyulf Core — Deep Audit (Opus)

**Date:** 2026-08-31 → 2026-09-01
**Auditor:** GitHub Copilot CLI (Claude Opus 5) — 15 parallel read-only audit agents, then a hands-on verification pass
**Scope:** `skyulf-core/` (33,735 lines of `skyulf/` source, every file, every line), extended on request to `backend/` (27,564 lines) and `frontend/ml-canvas/src` (71,666 lines)
**Baseline commit:** `93d7719e` (master)
**Status of repo at audit time:** `ruff check` clean, `ruff format --check` clean (356 files), `ty check` clean. **No repository files were modified by this audit.**

**Result:** **116 findings** — 5 🔴 Critical, 45 🟠 High, 44 🟡 Medium, 22 ⚪ Low —
plus one systemic recommendation, [R1](#r1), that retires eight of them as a class.

81 came from the agent phase (`OC-01 … OC-81`); the other 12 from five modules
the lead auditor [audited directly with no agent
input](#modules-audited-directly-by-me-with-no-agent-input). **27 of the
agent-phase findings were then re-verified by execution** — 2 needed correction —
and one of my *own* findings (OC-100) was retracted as a false positive. See [`00-validation-log.md`](./opus_core_analysis/00-validation-log.md) and
[Verification status](#verification-status).

Detail for the first five domains is in this file; the rest are in
[`opus_core_analysis/`](./opus_core_analysis/README.md).

---

## Severity key

| Symbol | Meaning |
|---|---|
| 🔴 Critical | Silent data corruption, wrong model output, or security exposure reaching users |
| 🟠 High | Wrong results / crashes in realistic configurations; user-visible contract broken |
| 🟡 Medium | Incorrect in edge cases, misleading output, or meaningful maintainability risk |
| ⚪ Low | Cosmetic, dead code, or hardening opportunity |

Finding IDs use the `OC-` prefix to avoid collision with `F-01..F-31` in
[`skyulf-core-findings.md`](./skyulf-core-findings.md).

---

## The headline

The single largest source of real, shipped bugs in this repository is **not** the
core algorithms — those are largely sound and well tested. It is the
**hand-duplicated contract between `skyulf-core` and `frontend/ml-canvas`**.

`frontend/ml-canvas/src/core/utils/pipelineConverter.ts` forwards `node.data`
to the backend **byte-for-byte with no validation, renaming, or schema check**.
Every node's parameter names, enum values, and defaults are retyped by hand in
TypeScript. This audit found **at least 11 distinct places** where the UI emits a
parameter name or enum value the Python side never reads — each one a silent
no-op or silent fallback, none of which raises an error, and none of which any
existing test catches.

The user-visible symptom is always the same and always invisible: **the canvas
says one thing, the pipeline does another.**

This is a systemic architectural issue, not eleven independent bugs. See
[Recommendation R1](#r1) for the structural fix.

---

## Summary of findings

### Cross-cutting / packaging (verified directly by me)

| ID | Sev | Title | Location |
|---|---|---|---|
| [OC-01](#oc-01) | 🟠 High | ⚠️ **corrected** — `skyulf.__version__` is ambiguous: a stale `0.5.8` dist-info shadows the real `0.8.8` (path-order dependent, *not* "always stale") | `skyulf/__init__.py:24-30` |
| [OC-02](#oc-02) | 🟠 High | Dev editable install is dangling; `import skyulf` fails outside the repo | venv `skyulf-core` dist-info |
| [OC-03](#oc-03) | 🟠 High | Systemic `infer_output_schema` int→float misprediction across 22 nodes | `preprocessing/*` + `backend/.../_schema_graph.py:104` |
| [OC-04](#oc-04) | 🟡 Medium | Cross-engine dtype divergence in 3 nodes (int64 vs int8/uint32) | `encoding/dummy.py`, `bucketing.py` |
| [OC-05](#oc-05) | 🟡 Medium | `PowerTransformer` triggers a pandas deprecation that will become an error | `transformations/power.py:101` |
| [OC-06](#oc-06) | 🟡 Medium | 6 registered nodes are unreachable from the UI (incl. all of `geo/`) | `registry.py` vs `frontend/` |
| [OC-07](#oc-07) | 🟡 Medium | Node-id naming is split 55 PascalCase / 45 snake_case + redundant aliases | `registry.py` |
| [OC-08](#oc-08) | 🟡 Medium | Public-API name collision: `DatasetProfile` means two different things | `skyulf/__init__.py:32-46` |
| [OC-09](#oc-09) | 🟡 Medium | Narrow `ruff select` hides ~500 missing docstrings and 84 unused args | `pyproject.toml:137-226` |
| [OC-10](#oc-10) | ⚪ Low | 4 dead `infer_output_schema` overrides that only `return None` | `vectorization/*` |
| [OC-11](#oc-11) | ⚪ Low | Mega smoke test silently skips nodes with empty params | `tests/unit/test_all_nodes_smoke.py` |

### Encoding, cleaning, imputation, scaling, drop/missing

| ID | Sev | Title | Location |
|---|---|---|---|
| [OC-12](#oc-12) | 🔴 Critical | Row-dropping desyncs `X` and `y` on non-unique pandas indexes | `drop_and_missing/drop_rows.py:60-67`, `deduplicate.py:44-47` |
| [OC-13](#oc-13) | 🟠 High | Drop-Rows UI settings ignored; every canvas run becomes "drop any missing" | `pipelineConverter.ts:249-253` |
| [OC-14](#oc-14) | 🟠 High | Iterative Imputer UI estimator choices silently fall back to BayesianRidge | `imputation/_common.py:103-111` |
| [OC-15](#oc-15) | 🟠 High | MinMax/Robust scaler range controls in the UI are ignored | `scaling/minmax.py:96-100`, `robust.py:116-123` |
| [OC-16](#oc-16) | 🟠 High | KNN/Iterative imputers crash on all-missing fitted columns | `imputation/knn.py:64-76`, `iterative.py:68-84` |
| [OC-17](#oc-17) | 🟠 High | SimpleImputer polars mean/median crashes on all-null columns (engine divergence) | `imputation/_common.py:32-37` |
| [OC-18](#oc-18) | 🟡 Medium | One-hot/dummy generated names can collide with existing columns | `encoding/one_hot.py:68-92`, `dummy.py:76-99` |
| [OC-19](#oc-19) | 🟡 Medium | Alias Replacement exposes a `punctuation` mode that does nothing | `cleaning/alias.py:45-52` |
| [OC-20](#oc-20) | 🟡 Medium | Value Replacement's "empty columns = all columns" UI promise is false | `cleaning/value_replacement.py:163-180` |
| [OC-21](#oc-21) | 🟡 Medium | WOE additive smoothing is not normalized over categories | `encoding/woe.py:130-145` |
| [OC-22](#oc-22) | ⚪ Low | `TargetEncoder.infer_output_schema` checks an impossible `regression` value | `encoding/target.py:340-360` |

### Feature generation, selection, vectorization, transformations

| ID | Sev | Title | Location |
|---|---|---|---|
| [OC-23](#oc-23) | 🟠 High | Polars `ratio` flips the sign of near-zero negative denominators | `feature_generation/_polars_ops.py:97-112` |
| [OC-24](#oc-24) | 🟠 High | Polars group aggregates treat null group keys differently from pandas | `feature_generation/_polars_ops.py:222-234` |
| [OC-25](#oc-25) | 🟠 High | RFE "K" chosen in the UI is ignored by the backend | `feature_selection/_common.py:236-240` |
| [OC-26](#oc-26) | 🟠 High | `HashingVectorizer` UI "none" norm is an invalid sklearn value → crash | `vectorization/hashing_vectorizer.py:59` |
| [OC-27](#oc-27) | 🟠 High | `GeneralTransformation` ignores the UI `standardize` toggle | `transformations/general.py:34-39,138-139` |
| [OC-28](#oc-28) | 🟠 High | Box-Cox transform failures silently return untransformed data | `transformations/power.py:97-104` |
| [OC-29](#oc-29) | 🟡 Medium | `FeatureGeneration` advertises `polynomial` but silently skips it | `feature_generation/_common.py:24-31` |
| [OC-30](#oc-30) | 🟡 Medium | Datetime extraction ignores the UI output name and overwrites collisions | `feature_generation/_pandas_ops.py:173-184` |
| [OC-31](#oc-31) | 🟡 Medium | Frontend wrongly requires a target for the unsupervised CorrelationThreshold | `FeatureSelectionNode.tsx:564-566` |
| [OC-32](#oc-32) | 🟡 Medium | `VarianceThreshold` crashes when all candidates are constant | `feature_selection/variance.py:38-47` |
| [OC-33](#oc-33) | 🟡 Medium | `FeatureInteraction` cannot generate single-column self-products | `feature_generation/interaction.py:173-178` |
| [OC-34](#oc-34) | 🟡 Medium | Count/TF-IDF vectorizers crash on empty or stop-word-only corpora | `vectorization/count_vectorizer.py:79-80` |

### Evaluation, thresholds, explainability

| ID | Sev | Title | Location |
|---|---|---|---|
| [OC-35](#oc-35) | 🟠 High | Multiclass splits missing a class emit binary-only metrics and null curve points | `modeling/_evaluation/metrics.py:217-237,361-363` |
| [OC-36](#oc-36) | 🟠 High | F1 threshold tuning picks a pathological threshold on single-class validation | `modeling/_evaluation/thresholds.py:101-111` |
| [OC-37](#oc-37) | 🟡 Medium | Binary PR-AUC is dropped for string-labeled classifiers | `modeling/_evaluation/metrics.py:324-327` |
| [OC-38](#oc-38) | ⚪ Low | Clustering metrics treat DBSCAN `-1` noise as a real cluster | `modeling/_evaluation/metrics.py:432-459` |

### Profiling — analyzer, statistics, drift, expectations, visualisation

| ID | Sev | Title | Location |
|---|---|---|---|
| [OC-39](#oc-39) | 🟠 High | NaN-bearing numeric columns publish `nan` stats and leak non-finite JSON | `profiling/analyzer.py:215-224` |
| [OC-40](#oc-40) | 🟠 High | PCA/clustering "mean imputation" actually replaces NaN with `0.0` | `profiling/_analyzer/multivariate.py:46-60` |
| [OC-41](#oc-41) | 🟠 High | Quartiles use nearest-rank, not linear interpolation (disagrees with pandas) | `profiling/analyzer.py:221-222` |
| [OC-42](#oc-42) | 🟠 High | Skewness/kurtosis use biased estimators, breaking a hardcoded threshold rule | `profiling/analyzer.py:223-224` |
| [OC-43](#oc-43) | 🟠 High | Correlation drops valid columns/rows instead of a defined missing-data policy | `profiling/correlations.py:41-44,100-110` |
| [OC-44](#oc-44) | 🟠 High | Wasserstein drift thresholds a normalized value but reports the raw one | `profiling/drift.py:181-195` |
| [OC-45](#oc-45) | 🟠 High | Schema drift is computed but never counted or rendered as drift | `profiling/drift.py:76-98` |
| [OC-46](#oc-46) | 🟡 Medium | ⚠️ **corrected, downgraded** — non-finite floats reach public payloads and are silently coerced to `null` by orjson; only stdlib-json paths emit invalid JSON | `profiling/schemas.py:7-17,263-302` |
| [OC-47](#oc-47) | 🟡 Medium | Common-column dtype drift can silently disappear | `profiling/drift.py:136-153` |
| [OC-48](#oc-48) | 🟡 Medium | Expectations pass vacuously on empty frames | `profiling/expect.py:92-209` |
| [OC-49](#oc-49) | 🟡 Medium | Valid partially-unlabelled PCA payloads crash plotting | `profiling/visualizer.py:716-737` |
| [OC-50](#oc-50) | 🟡 Medium | Binary targets miss class-balance advice or flip to regression by sample size | `profiling/_analyzer/recommendations.py:147-152` |
| [OC-51](#oc-51) | 🟡 Medium | Transform advice can be mathematically invalid and self-contradictory | `profiling/_analyzer/recommendations.py:66-78,129-139` |
| [OC-52](#oc-52) | ⚪ Low | Categorical colour mapping is process-nondeterministic | `profiling/visualizer.py:710-713` |

### Core, engines, data & pipeline → [report 06](./opus_core_analysis/06-core-engines-pipeline.md)

| ID | Sev | Title | Location |
|---|---|---|---|
| [OC-62](./opus_core_analysis/06-core-engines-pipeline.md#oc-62) | 🔴 Critical | `fingerprint()` is not reproducible for any artifact holding an object-dtype array | `pipeline/seal.py:57-59` |
| [OC-63](./opus_core_analysis/06-core-engines-pipeline.md#oc-63) | 🟠 High | `artifact_digest` raises `RecursionError` instead of the documented `TypeError` on cyclic graphs | `pipeline/seal.py` |
| [OC-64](./opus_core_analysis/06-core-engines-pipeline.md#oc-64) | 🟠 High | **F-14 only partially fixed** — the engine registry global is still an unlocked race | `engines/registry.py:60,86-91` |
| [OC-65](./opus_core_analysis/06-core-engines-pipeline.md#oc-65) | 🟡 Medium | The polars `to_numpy()` zero-width "parity fix" does not achieve parity | `engines/polars_engine.py` |
| [OC-74](./opus_core_analysis/06-core-engines-pipeline.md#oc-74) | 🟡 Medium | `NodeRegistry.list_models()` hides all 4 Ensemble models; its `category` arg is dead | `registry.py:101-108` |

### Outliers, casting, binning, time series & geo → [report 07](./opus_core_analysis/07-outliers-timeseries-geo.md)

| ID | Sev | Title | Location |
|---|---|---|---|
| [OC-58](./opus_core_analysis/07-outliers-timeseries-geo.md#oc-58) | 🔴 Critical | Numeric→boolean casting silently reinterprets any nonzero value as `True` on polars | `preprocessing/casting.py:143-178` |
| [OC-59](./opus_core_analysis/07-outliers-timeseries-geo.md#oc-59) | 🟠 High | `DatasetProfile` numeric-column coverage is completely different between engines | `preprocessing/inspection/` |
| [OC-60](./opus_core_analysis/07-outliers-timeseries-geo.md#oc-60) | 🟠 High | `GeneralBinning`'s `missing_strategy: "label"` is a silent no-op on polars | `preprocessing/bucketing.py` |
| [OC-61](./opus_core_analysis/07-outliers-timeseries-geo.md#oc-61) | 🟡 Medium | `BinningNode`'s "Precision (Decimals)" UI field is never sent to the backend | `BinningNode.tsx` |

### Modeling: estimators, CV & tuning → [report 08](./opus_core_analysis/08-modeling-tuning.md)

| ID | Sev | Title | Location |
|---|---|---|---|
| [OC-66](./opus_core_analysis/08-modeling-tuning.md#oc-66) | 🟠 High | `CalibratedClassifierCV`'s user-selected base estimator is silently discarded during tuning | `modeling/classification.py:206-282` vs `_tuning/engine.py:495-499` |
| [OC-67](./opus_core_analysis/08-modeling-tuning.md#oc-67) | 🟡 Medium | Tuning metrics `pr_auc`/`pr_auc_weighted`/`g_score` crash the entire search | `modeling/_tuning/metrics.py:19-36,127-146` |

> **Leakage audit: clean.** All 7 fit/apply boundaries in `skyulf-core` were
> traced; **none leaks**. 389 (model, param, extreme) combinations were executed;
> **every declared hyperparameter range is valid.** See report 08.

### Backend: execution, API & services → [report 09](./opus_core_analysis/09-backend.md)

| ID | Sev | Title | Location |
|---|---|---|---|
| [OC-68](./opus_core_analysis/09-backend.md#oc-68) | 🟠 High | Model alias map is task-unaware — a direct API caller silently trains the wrong estimator family | `_execution/engine/_node_runners.py:1157-1183` |
| [OC-69](./opus_core_analysis/09-backend.md#oc-69) | 🟠 High | The engine trusts `config.nodes` list order and never verifies it is topologically sorted | `_execution/_schema_graph.py:49-70`, `engine/__init__.py:135-171` |
| [OC-70](./opus_core_analysis/09-backend.md#oc-70) | 🟡 Medium | The leakage validator checks for *a* splitter globally, not that *this* branch is protected | `_execution/_leakage_validation.py:189-267` |
| [OC-71](./opus_core_analysis/09-backend.md#oc-71) | 🟠 High | **No authentication or authorization anywhere on the API** | `main.py:373-395`, `database/models.py:151-159` |
| [OC-72](./opus_core_analysis/09-backend.md#oc-72) | 🟡 Medium | Insecure-by-default config: unset `FASTAPI_ENV` fails *open* to wildcard CORS + credentials | `config/factory.py:26`, `main.py:359-366` |
| [OC-73](./opus_core_analysis/09-backend.md#oc-73) | ⚪ Low | `DataSource.credentials` documented as encrypted, stored as plaintext JSON | `database/models.py:107` |

> Path traversal, SQLi, SSRF, unsafe deserialization, upload limits, rate
> limiting and async-blocking were all checked and found **sound**. See the
> security checklist in report 09.

### Frontend node-config layer → [report 10](./opus_core_analysis/10-frontend.md)

| ID | Sev | Title | Location |
|---|---|---|---|
| [OC-53](./opus_core_analysis/10-frontend.md#oc-53) | 🟡 Medium | `select_from_model`'s `max_features` is a Python-only, UI-unreachable param | `feature_selection/` vs `FeatureSelectionNode.tsx` |
| [OC-54](./opus_core_analysis/10-frontend.md#oc-54) | 🟡 Medium | `DebugNode` is dead code that would silently no-op if ever wired up | `nodes/DebugNode.tsx` |
| [OC-55](./opus_core_analysis/10-frontend.md#oc-55) | 🟡 Medium | `tsc --noEmit` fails: `mermaid` declared but not installed | `frontend/ml-canvas/package.json` |
| [OC-56](./opus_core_analysis/10-frontend.md#oc-56) | ⚪ Low | `useSchemaPreview` does not cancel in-flight requests on unmount | `hooks/useSchemaPreview.ts` |
| [OC-57](./opus_core_analysis/10-frontend.md#oc-57) | ⚪ Low | `any`-typed chart props bypass type safety in EDA components | `modules/eda/` |

> Report 10 also carries the **full ~40-row node parity matrix** (registry id ↔
> node component ↔ params ↔ verdict) — the raw evidence behind the headline.

### Tests, benchmarks, packaging & CI → [report 11](./opus_core_analysis/11-tests-packaging-ci.md)

| ID | Sev | Title | Location |
|---|---|---|---|
| [OC-75](./opus_core_analysis/11-tests-packaging-ci.md#oc-75) | 🔴 Critical | Dev env polars (1.40.1) is below the declared floor (≥1.43.2) — 10 tests, **every notebook** and a benchmark all broken | `setup.py:27`, `uv.lock:1160`, `preprocessing/split.py:283-351` |
| [OC-76](./opus_core_analysis/11-tests-packaging-ci.md#oc-76) | 🟠 High | Cross-engine parity tests cover 9 of 100 nodes and never compare applied output | `tests/unit/test_engine_parity.py` |
| [OC-77](./opus_core_analysis/11-tests-packaging-ci.md#oc-77) | 🟠 High | `--maxfail=1` hides the real failure count; `--cov-fail-under=45` vs 98.4% actual | `.github/workflows/skyulf-core-tests.yml:82-87` |
| [OC-78](./opus_core_analysis/11-tests-packaging-ci.md#oc-78) | 🟡 Medium | `py.typed` declared in packaging metadata but the file does not exist | `setup.py:20-21`, `MANIFEST.in:3` |
| [OC-79](./opus_core_analysis/11-tests-packaging-ci.md#oc-79) | 🟡 Medium | `joblib` imported at module scope but not in `install_requires` | `core/serialization.py:22`, `setup.py:22-29` |
| [OC-80](./opus_core_analysis/11-tests-packaging-ci.md#oc-80) | 🟡 Medium | The 3 weakest-covered modules are untested exactly where silence is dangerous | `_sklearn_compat.py`, `value_replacement.py`, `config_validation.py` |
| [OC-81](./opus_core_analysis/11-tests-packaging-ci.md#oc-81) | ⚪ Low | No `License ::` classifier / SPDX field | `setup.py:81-85` |

> **The test suite itself is excellent**: 3,670 tests, **98.40% coverage**, zero
> modules under 50%, fully deterministic across repeated runs and
> `PYTHONHASHSEED` changes. Every problem above is *around* the suite — the
> environment it runs in, the gates that guard it, the metadata it ships.

---

### Modules audited directly by me, with no agent input

These five areas were in no agent's assigned scope, or were re-done from scratch
after the verification pass. Every finding below was found *and* reproduced by
the lead auditor.

| ID | Sev | Title | Location | Report |
|---|---|---|---|---|
| [OC-110](./opus_core_analysis/15-profiling-analyzers.md#oc-110) | 🟠 High | Semantic-type inference misclassifies small categorical columns as `Text`, so task type is never inferred | `profiling/_analyzer/column.py`, `analyzer.py:502` | [15](./opus_core_analysis/15-profiling-analyzers.md) |
| [OC-113](./opus_core_analysis/15-profiling-analyzers.md#oc-113) | 🟠 High | Near-perfect multicollinearity silently reports **VIF = 1.0** (no multicollinearity) — `max(1.0, …)` clamps numerical garbage | `profiling/_analyzer/numeric.py:32-63` | [15](./opus_core_analysis/15-profiling-analyzers.md) |
| [OC-114](./opus_core_analysis/15-profiling-analyzers.md#oc-114) | 🟡 Medium | An all-null tracked column yields 30 `NaN` autocorrelation lags presented as real analysis (≥1000-row datasets only) | `profiling/_analyzer/temporal.py:167-191` | [15](./opus_core_analysis/15-profiling-analyzers.md) |
| [OC-120](./opus_core_analysis/16-dtype-coverage.md#oc-120) | 🟠 High | `Decimal` columns silently skipped by every auto-numeric node; crash pandas when selected explicitly | `engines/__init__.py`, `preprocessing/_helpers.py` | [16](./opus_core_analysis/16-dtype-coverage.md) |
| [OC-91](./opus_core_analysis/13-core-internals.md#oc-91) | 🟡 Medium | Three public `core/` seams (263 lines) have zero call sites; one duplicates a differently-shaped backend class name | `core/deprecation.py`, `core/model_registry.py`, `core/serialization.py` | [13](./opus_core_analysis/13-core-internals.md) |
| [OC-101](./opus_core_analysis/14-hyperparameters.md#oc-101) | 🟡 Medium | `calibrated_classifier`'s `random_state` field is a no-op for **two** independent reasons: the estimator rejects it, *and* the base-estimator factories hardcode the seed | `hyperparameters/_calibration.py`, `modeling/classification.py:217-226` | [14](./opus_core_analysis/14-hyperparameters.md) |
| [OC-102](./opus_core_analysis/14-hyperparameters.md#oc-102) | ⚪ Low | Five tunable models return an empty search space from the live `/defaults` endpoint | `hyperparameters/_registry.py` | [14](./opus_core_analysis/14-hyperparameters.md) |
| [OC-111](./opus_core_analysis/15-profiling-analyzers.md#oc-111) | 🟡 Medium | A profiling recommendation branch is unreachable | `profiling/_analyzer/` | [15](./opus_core_analysis/15-profiling-analyzers.md) |
| [OC-90](./opus_core_analysis/12-splitters.md#oc-90) | ⚪ Low | Unknown split config keys are silently dropped instead of rejected | `preprocessing/split.py` | [12](./opus_core_analysis/12-splitters.md) |
| [OC-112](./opus_core_analysis/15-profiling-analyzers.md#oc-112) | ⚪ Low | Comment and code disagree about the applied threshold | `profiling/_analyzer/` | [15](./opus_core_analysis/15-profiling-analyzers.md) |
| [OC-121](./opus_core_analysis/16-dtype-coverage.md#oc-121) | ⚪ Low | polars `Enum` columns invisible to text auto-detection, diverging from pandas `Categorical` | `preprocessing/_helpers.py:148-157` | [16](./opus_core_analysis/16-dtype-coverage.md) |
| [OC-122](./opus_core_analysis/16-dtype-coverage.md#oc-122) | ⚪ Low | `TextCleaning` silently ignores an unrecognised operation name | `preprocessing/cleaning/text.py:151-153` | [16](./opus_core_analysis/16-dtype-coverage.md) |

> **[Splitting is the one module I audited and found entirely sound](./opus_core_analysis/12-splitters.md)** —
> exact cross-engine row-identity parity, zero partition overlap, exact
> stratification, correct `validation_size/(1-test_size)` renormalisation, clean
> errors on impossible ratios, and graceful degradation on singleton classes.
> OC-90 is a hardening nit, not a bug.

> ❌ **OC-100 was retracted.** I filed "the search-space dicts are dead code",
> and it was **wrong** — the dicts are live via `get_default_search_space()` →
> a mounted HTTP route → `TrainingSettings.tsx`. My grep searched for the dict
> *names* in `.py` files, which could never have found a wrapper-function
> consumer whose caller is TypeScript. Full account, including the broken probe
> that falsely "confirmed" it, in
> [the validation log](./opus_core_analysis/00-validation-log.md#oc-100).
> The one durable part is re-filed as OC-102.

---

### Backend infrastructure (verified directly by me after an agent sweep)

| ID | Sev | Title | Location | Report |
|---|---|---|---|---|
| [OC-130](./opus_core_analysis/09-backend.md#oc-130) | 🟠 High | A typo in `FASTAPI_ENV` silently disables the entire production security posture — wildcard CORS **with credentials**, `DEBUG=True`, no `SECRET_KEY` check, no security headers | `config/factory.py:27-32`, `config/base.py:188-189`, `main.py:361-363` | [09](./opus_core_analysis/09-backend.md) |
| [OC-131](./opus_core_analysis/09-backend.md#oc-131) | ⚪ Low | Diagnostics *fail open* — PSI returns `0.0` ("no drift") on any numeric failure, mirroring OC-113's `VIF = 1.0` | `skyulf/profiling/drift.py:474-476` | [09](./opus_core_analysis/09-backend.md) |
| [OC-132](./opus_core_analysis/09-backend.md#oc-132) | ⚪ Low | Dead `dropped_features` branch — the key appears exactly once in the whole repo (the read); the live path uses `dropped_columns` from job metrics | `ml_pipeline/_execution/graph_utils.py:534-537` | [09](./opus_core_analysis/09-backend.md) |
| [OC-145](./opus_core_analysis/09-backend.md#oc-145) | 🟡 Medium | A crashed cross-validation returns the same `{}` sentinel as a *disabled* one, so the job reports success with silently missing `cv_*` metrics | `ml_pipeline/_execution/engine/_node_runners.py:871-907` | [09](./opus_core_analysis/09-backend.md) |
| [OC-150](./opus_core_analysis/09-backend.md#oc-150) | 🟠 High | S3 error "sanitiser" matches credential *key names*, case-sensitively — S3 403 bodies and **replayable presigned URLs** are logged verbatim; duplicated in two files | `data_ingestion/connectors/s3.py:31-37`, `ml_pipeline/artifacts/s3.py:67-73` | [09](./opus_core_analysis/09-backend.md) |
| [OC-151](./opus_core_analysis/09-backend.md#oc-151) | 🟡 Medium | Trial-buffer `clear_*` hooks are documented as lifecycle cleanup but **never called** — measured **110.9 MB** retained for the process lifetime | `realtime/trial_buffer.py:56-59,103-106` | [09](./opus_core_analysis/09-backend.md) |
| [OC-152](./opus_core_analysis/09-backend.md#oc-152) | ⚪ Low | Two raw-SQL executors accept unconstrained query strings and have zero callers — a latent injection sink in a shared manager | `database/async_connection_manager.py:243-268` | [09](./opus_core_analysis/09-backend.md) |
| [OC-153](./opus_core_analysis/09-backend.md#oc-153) | 🟠 High | Multi-input merge **silently switches from column-wise to row-wise** when a branch changes the row count — a 5-row set merged with its own filtered branch yields **8 rows with 3 duplicates** and **zero UI warnings** | `ml_pipeline/_execution/engine/_merge.py:338-348` | [09](./opus_core_analysis/09-backend.md) |
| [OC-154](./opus_core_analysis/09-backend.md#oc-154) | 🟠 High | Serving-time feature-order reindex (fix F-02) **fails open** when the column set doesn't match — a numpy-fitted model silently returned **213.00 where truth is 321.00** | `ml_pipeline/deployment/service.py:438-442` | [09](./opus_core_analysis/09-backend.md) |
| [OC-155](./opus_core_analysis/09-backend.md#oc-155) | 🟠 High | Legacy predict path **zero-fills missing features** and returns the prediction as normal; caller never sees the server-side warning | `ml_pipeline/deployment/service.py:457-462` | [09](./opus_core_analysis/09-backend.md) |
| [OC-156](./opus_core_analysis/09-backend.md#oc-156) | 🟡 Medium | `roc_auc` threshold-tuning objective scores **hard predictions**, making it bit-identical to `balanced_accuracy` across all trials — two UI options, one metric | `ml_pipeline/_services/threshold_tuning_service.py:77-92` | [09](./opus_core_analysis/09-backend.md) |
| [OC-157](./opus_core_analysis/09-backend.md#oc-157) | ⚪ Low | `first_wins` merge strategy **reverses output column order** (`['a','b','c','d']` → `['c','d','a','b']`), contradicting its own docstring | `ml_pipeline/_execution/engine/_merge.py:221-236` | [09](./opus_core_analysis/09-backend.md) |
| [OC-158](./opus_core_analysis/09-backend.md#oc-158) | 🟡 Medium | The sync and async JSON serializers **disagree**: the sync one silently nulls **8 of 15** legitimate strings (`"nan"`, `"NaT"`, `"<NA>"`, `"inf"`…), the async one nulls none. 603-line module is **production-dead but test-covered** | `data_ingestion/serialization.py:369,435-446` | [09](./opus_core_analysis/09-backend.md) |
| [OC-159](./opus_core_analysis/09-backend.md#oc-159) | ⚪ Low | An empty filter dict compiles to a **WHERE-less `DELETE FROM data_sources`** / `UPDATE` — `_normalize_filter` maps `None`→`{}` with no guard. Dead call path today | `database/data_sources/async_sqlite_queries.py:129-146` | [09](./opus_core_analysis/09-backend.md) |

### Final coverage-gap closure — evaluation, thresholds, clustering, PII

Report [17](./opus_core_analysis/17-file-coverage.md) named 16 files that had
never been read. Working down that list by its own risk ranking produced four
more findings — including the **only 🔴 Critical found after the agent phase**.
The top-ranked file, `modeling/_evaluation/metrics.py`, was flagged there as
*"metric averaging bugs are severe and silent."* That prediction was correct.

| ID | Sev | Title | Location | Report |
|---|---|---|---|---|
| [OC-146](./opus_core_analysis/04-evaluation-explainability.md#oc-146) | 🔴 Critical | Binary `pr_auc` is scored against the wrong class on `{1, n}` labels — reports **0.32** for a model whose true PR-AUC is **0.97**, with no warning; the PR chart beside it is drawn correctly, so the report contradicts itself | `modeling/_evaluation/metrics.py:324-326` | [04](./opus_core_analysis/04-evaluation-explainability.md) |
| [OC-149](./opus_core_analysis/04-evaluation-explainability.md#oc-149) | 🟠 High | Clustering evaluation **crashes on polars** when a numeric feature is all-null within one cluster; pandas returns `nan` — `mean` is unguarded while `std` one line below guards `None` | `modeling/_evaluation/clustering.py:83-88` | [04](./opus_core_analysis/04-evaluation-explainability.md) |
| [OC-148](./opus_core_analysis/15-profiling-analyzers.md#oc-148) | 🟡 Medium | PII detector flags ordinary 7+ digit numeric ID columns as "Email/Phone"; the guard comment's stated rationale is false | `profiling/_analyzer/text.py:107-128` | [15](./opus_core_analysis/15-profiling-analyzers.md) |
| [OC-147](./opus_core_analysis/04-evaluation-explainability.md#oc-147) | ⚪ Low | `optimize_thresholds` returns a dict shape that bypasses its own documented binary rule, flipping `>=` to `>` on exact ties | `modeling/_evaluation/thresholds.py:66-88` | [04](./opus_core_analysis/04-evaluation-explainability.md) |

> ❌ **Not reproduced (my own hypothesis, rejected):** I suspected the
> unconditional `tracemalloc.start()` in `preprocessing/base.py:165` imposed a
> large per-step penalty and was blind to polars' Rust-side allocations.
> Measured overhead was only **1.11x**, and the polars figure I first read as
> confirmation turned out to match the size of the *numpy input arrays* I had
> built — a coincidence, not evidence. Neither claim survived, so neither was
> filed.

> ❌ **Not reproduced:** an agent's claim that `monitoring/router.py` emits bare
> `NaN`/`Infinity` tokens. Those sites are pydantic `model_dump()`, not stdlib
> `json`; FastAPI *raises* rather than emitting invalid JSON; and PSI/KS/
> Wasserstein stay finite on degenerate inputs. [Recorded in report 09](./opus_core_analysis/09-backend.md#not-reproduced)
> so it is not re-filed.

---

## Totals

| Severity | Count |
|---|---|
| 🔴 Critical | **5** — [OC-12](#oc-12), [OC-58](./opus_core_analysis/07-outliers-timeseries-geo.md#oc-58), [OC-62](./opus_core_analysis/06-core-engines-pipeline.md#oc-62), [OC-75](./opus_core_analysis/11-tests-packaging-ci.md#oc-75), [OC-146](./opus_core_analysis/04-evaluation-explainability.md#oc-146) — **all 5 independently re-verified** |
| 🟠 High | 45 |
| 🟡 Medium | 44 |
| ⚪ Low | 22 |
| **Total** | **116** — `OC-01 … OC-81` (agent phase) + `OC-90/91`, `OC-101/102`, `OC-110…114`, `OC-120…122`, `OC-130…132`, `OC-140…159` (direct). **OC-100 retracted**, not counted. |

### Per-file coverage of core

A mechanical, per-file audit of all 188 `.py` files in `skyulf-core/skyulf/` is
in [17-file-coverage.md](./opus_core_analysis/17-file-coverage.md): 147 files
(78%) read or explicitly analysed, 16 covered by behaviour only, and **16 never
examined** — each listed by name with its risk. Reading 7 previously-unexamined
files during that pass produced 5 new findings (OC-140…144), three of them
user-facing correctness bugs, which is why the remaining 16 are flagged as the
highest-value next step.

| ID | Sev | Issue |
|---|---|---|
| [OC-140](./opus_core_analysis/17-file-coverage.md#oc-140) | 🟠 High | `InvalidValueReplacement` diverges across engines on non-numeric columns (pandas silently NaNs, polars raises) |
| [OC-141](./opus_core_analysis/17-file-coverage.md#oc-141) | ⚪ Low | `invalid_values` param declared in `node_meta` with zero consumers |
| [OC-142](./opus_core_analysis/17-file-coverage.md#oc-142) | 🟠 High | EDA correlation ratio η exceeds 1.0 with nulls; null-heavy columns rank as strongest association |
| [OC-143](./opus_core_analysis/17-file-coverage.md#oc-143) | 🟠 High | RFE ignores the UI's `k`, silently selecting half the features |
| [OC-144](./opus_core_analysis/17-file-coverage.md#oc-144) | ⚪ Low | Geo distance column named `_km` even when unit is miles |

### Verification status

**27 of these findings have been independently re-verified by execution** — see
[00-validation-log.md](./opus_core_analysis/00-validation-log.md). Of those, 25
stand (4 are *worse* than filed: [OC-12](#oc-12), [OC-18](#oc-18),
[OC-40](#oc-40), [OC-42](#oc-42)) and **2 required correction**:
[OC-01](#oc-01) (claim overstated) and [OC-46](#oc-46) (claim wrong on the real
ship path — **downgraded 🟠 High → 🟡 Medium**, so the agent-phase split is 33
High / 35 Medium rather than the 34/34 originally filed; the Totals above then
add the 10 directly-audited findings on top). The remaining agent-phase findings
rest on their originating auditor's pasted evidence and should be treated as
*probable but unconfirmed* until re-run.

Everything in [OC-90 … OC-122](#modules-audited-directly-by-me-with-no-agent-input)
was found and reproduced by the lead auditor, so it needs no separate
verification pass.

A further 7 findings surfaced by more than one auditor were **merged into the
existing entry rather than re-filed** (each carries a
"*Merged, not re-filed*" note pointing at the owning id), and one — an apparent
binning out-of-range bug — was **dropped as a false positive** after
`bucketing.py:41-50` proved the behaviour intentional and documented.

---

## Cross-cutting findings (verified directly)

### OC-01
### 🟠 High — `skyulf.__version__` always reports a stale, wrong version

> ⚠️ **Corrected after independent re-verification** — see
> [00-validation-log.md](./opus_core_analysis/00-validation-log.md#oc-01).
> The "always" is wrong. There are **two** visible `skyulf-core` distributions
> (in-repo `egg-info` at `0.8.8`, installed `dist-info` at `0.5.8`), and which
> wins is purely `sys.path` order — in the normal repo-dev setup it resolves
> **correctly** to `0.8.8`. The real defect is **stale duplicate-distribution
> shadowing**, so the fix is to remove the stale `0.5.8` dist-info / reinstall
> cleanly, *not* to change the version-reading code. Severity stays 🟠 High.

**File:** `skyulf-core/skyulf/__init__.py:24-30`

`__version__` is resolved at import time via
`importlib.metadata.version("skyulf-core")`. The in-tree comment explains this
is deliberate, to avoid "a second copy that can drift out of sync". In practice
the mechanism drifts **silently and always**:

| Source | Version |
|---|---|
| `setup.py` (the real code version) | `0.8.8` |
| in-repo `skyulf_core.egg-info/PKG-INFO` | `0.8.5` |
| installed `skyulf_core-0.5.8.dist-info` | `0.5.8` |
| **what `skyulf.__version__` actually returns** | **`0.8.5`** |

```console
$ cd skyulf-core && python -c "import skyulf; print(skyulf.__version__)"
0.8.5
```

The value resolves to whichever stale metadata directory is found first — never
the version in `setup.py`. Because `importlib.metadata` succeeds, there is no
error to notice.

**Impact:** Any artifact, model card, run log, or reproducibility record that
stamps `skyulf.__version__` records a version that does not correspond to the
code that produced it. This defeats the purpose of version stamping.

**Fix:** Adopt a single source of truth. Either (a) put `__version__` in
`skyulf/__init__.py` and have `setup.py` read it, or (b) keep
`importlib.metadata` but add a build-time consistency assertion and delete the
stale `.egg-info` from version control. Add a test asserting
`skyulf.__version__ == <setup.py version>`.

---

### OC-02
### 🟠 High — Dev editable install is dangling; `import skyulf` fails outside the repo

`pip show skyulf-core` reports:

```
Editable project location: /private/tmp/skyulf-078k/skyulf-core
```

That directory **does not exist**. Consequently:

```console
$ cd ~ && python -c "import skyulf"
ModuleNotFoundError: No module named 'skyulf'
```

The test suite only passes because pytest inserts the rootdir onto `sys.path`.
Any script, notebook, or Celery worker run from a different working directory
cannot import the library.

**Impact:** The development environment does not reflect a real installation.
Import-time regressions, packaging errors, and missing-`__init__` problems are
invisible locally and only surface in deployment.

**Fix:** Re-install cleanly: `pip install -e ./skyulf-core`. Add a CI smoke step
that runs `cd /tmp && python -c "import skyulf; print(skyulf.__version__)"` so a
broken install fails the build.

---

### OC-03
### 🟠 High — Systemic `infer_output_schema` int→float misprediction across 22 nodes

**Files:** 22 preprocessing nodes using the `return input_schema` pass-through
pattern; consumed by `backend/ml_pipeline/_execution/_schema_graph.py:104`

`infer_output_schema` powers the frontend's downstream schema preview. 22 nodes
implement it as `return input_schema`, i.e. "I don't change the schema." For any
node that scales or transforms an integer column, that is false — the column is
promoted to `float64`.

I tested 6 of the 22 and **all 6 mispredicted**:

| Node | Predicted | Actual |
|---|---|---|
| `StandardScaler` | `{'a': 'int64'}` | `{'a': 'float64'}` |
| `MinMaxScaler` | `{'a': 'int64'}` | `{'a': 'float64'}` |
| `RobustScaler` | `{'a': 'int64'}` | `{'a': 'float64'}` |
| `MaxAbsScaler` | `{'a': 'int64'}` | `{'a': 'float64'}` |
| `Winsorize` | `{'a': 'int64'}` | `{'a': 'float64'}` |
| `PowerTransformer` | `{'a': 'int64'}` | `{'a': 'float64'}` |

`GeneralTransformation` (`transformations/general.py:160-165`) has the same
defect.

**Impact:** The canvas shows users an incorrect dtype for every downstream node
after a scaler. Any backend logic that plans or validates on the predicted
schema (type-compatibility checks, merge-conflict detection) is reasoning about
a schema that will never exist.

Note that returning `None` from `infer_output_schema` is a **documented,
intentional** "unknown / data-dependent" signal
(`preprocessing/base.py:97-113`). So the correct fix is cheap: these nodes
should promote the dtype, not return `None`.

**Fix:** Add a shared helper, e.g.
`schema.promote_to_float(cols)`, and use it in every node that produces
floating-point output from numeric input. Add a parametrised test asserting
`infer_output_schema(s) == SkyulfSchema.from_dataframe(apply(fit(df)))` for
every node that returns non-`None`.

---

### OC-04
### 🟡 Medium — Cross-engine dtype divergence in 3 nodes

I built an all-node cross-engine parity harness (fit + apply each registered
preprocessing node on both pandas and polars, then diff column sets, order,
shape, dtype, and values). Result: **26 match, 3 diverge, 0 error, 5 not
exercised, of 34 nodes.**

| Node | pandas dtype | polars dtype |
|---|---|---|
| `DummyEncoder` | `int64` (11 dummy cols) | `int8` |
| `GeneralBinning` | `int64` | `uint32` |
| `KBinsDiscretizer` | `int64` | `uint32` |

Values are identical; only the dtype differs. This matters because the existing
parity test (`tests/unit/test_engine_parity.py`) compares **artifacts only**,
never applied output frames, so it cannot catch this class of divergence.

The `uint32` case deserves attention: an unsigned bin index cannot represent a
negative sentinel. If a future change introduces `-1` for "out of range" (a
very common convention), the polars path would wrap to `4294967295` while
pandas gives `-1`.

**Impact:** A model trained through the pandas path and served through the
polars path receives different input dtypes. Strict schema validation would
reject; some estimators change behaviour on integer width.

**Fix:** Normalise output dtypes explicitly at the end of each dual-engine
applier. Extend `test_engine_parity.py` to compare **applied output frames**
(columns, order, dtypes, values) for every registered node, not just artifacts.

> **Checked and cleared:** I initially suspected that out-of-range values at
> transform time were silently becoming `NaN` in the binning nodes. They are —
> but this is **intentional and documented** at `bucketing.py:41-50`, and it is
> consistent across both engines. Not a bug. It is still worth surfacing a
> count of out-of-range rows to the user (see improvements).

---

### OC-05
### 🟡 Medium — `PowerTransformer` triggers a pandas deprecation that will become an error

**File:** `skyulf-core/skyulf/preprocessing/transformations/power.py:101`

```python
df_out.loc[:, valid_cols] = np.asarray(X_trans)
```

When applied to an integer column this emits:

```
FutureWarning: Setting an item of incompatible dtype is deprecated and will
raise in a future error of pandas.
```

I scanned the whole package; this is the **only** node with this pattern.

**Impact:** A future pandas release turns this warning into an exception,
breaking `PowerTransformer` on integer input.

**Fix:** Build the transformed block and assign with an explicit dtype-safe
construction, e.g. `df_out[valid_cols] = pd.DataFrame(X_trans, index=df_out.index, columns=valid_cols)`.

---

### OC-06
### 🟡 Medium — 6 registered nodes are unreachable from the UI

> **Corrected after the frontend audit.** This finding originally listed **9**
> nodes, derived from a registry-id-vs-quoted-string diff. That method produced
> three false positives: `SegmentationNode.tsx` builds its dropdown dynamically
> and *does* offer `birch`, `gaussian_mixture` and `minibatch_kmeans`. The
> corrected count is **6**. See the parity matrix in
> [report 10](./opus_core_analysis/10-frontend.md).

These 6 registered, implemented, tested nodes have no frontend affordance:

`CustomBinning`, `DataSnapshot`, `DatasetProfile`, `FeatureGeneration`,
`GeoDistance`, `H3Index`

Notably, the whole `geo/` package (348 lines + tests, fully implemented) is
unreachable. `backend/ml_pipeline/_execution/_leakage_validation.py:27-28` even
documents `GeoDistance`/`H3Index` behaviour.

A related, opposite-direction gap exists inside the registry itself:
[OC-74](./opus_core_analysis/06-core-engines-pipeline.md#oc-74) —
`NodeRegistry.list_models()` hides all 4 Ensemble models from any
registry-driven consumer.

**Impact:** Dead-but-maintained surface area. Contributors pay the cost of
keeping these nodes green with zero user benefit.

**Fix:** Decide per node — either expose it in the canvas or deprecate and
remove it. Add a CI check that every registry id is either referenced in the
frontend or on an explicit `INTENTIONALLY_HEADLESS` allow-list.

---

### OC-07
### 🟡 Medium — Node-id naming is split 55 PascalCase / 45 snake_case + redundant aliases

The single registry mixes `StandardScaler` with `random_forest_classifier`.
There is no rule a contributor can follow.

There is also redundant triple-aliasing:
- `FeatureGeneration` / `FeatureMath` / `FeatureGenerationNode` → the same applier (`feature_generation/generation.py:24-26`)
- `PolynomialFeatures` / `PolynomialFeaturesNode` → the same applier (`polynomial.py:88-89`)

**Fix:** Document the convention (preprocessing = PascalCase, estimators =
snake_case appears to be the de-facto rule) and enforce it in the `@node_meta`
decorator. Collapse the aliases behind `_DEPRECATED_ALIASES`, which already
exists for `Split → TrainTestSplitter`.

---

### OC-08
### 🟡 Medium — Public-API name collision: `DatasetProfile` means two different things

- `skyulf.DatasetProfile` → `skyulf.profiling.schemas.DatasetProfile` (a dataclass)
- registry node id `DatasetProfile` → `skyulf.preprocessing.inspection.DatasetProfileCalculator`

Same string, two unrelated concepts, both public.

**Fix:** Rename the node id to `DatasetProfileInspection` (keeping the old id in
`_DEPRECATED_ALIASES`), or rename the schema export.

---

### OC-09
### 🟡 Medium — Narrow `ruff select` hides substantial debt behind a green CI

**File:** `pyproject.toml:137-226`

```toml
select = ["E9","F63","F7","F82","I","UP","B","C4","SIM","PERF","BLE","PLC0415"]
```

This omits `D`, `ARG`, `S`, `PLR`, `C90`, `DTZ`, `N`, `RUF`, `PD`, `TRY`.
Running `ruff check --select ALL` on `skyulf-core/skyulf` reveals:

| Rule | Count | Why it matters here |
|---|---:|---|
| `ARG001`/`ARG002` unused arguments | **84** | This is the exact signature of "parameter accepted but silently ignored" — the dominant bug class in this audit |
| `D101`/`D102`/`D107` missing docstrings | **495** | Directly violates the repo's own `AGENTS.md` docstring rule |
| `PLR2004` magic values | 120 | Hardcoded thresholds like the `abs(skew) > 1.5` in OC-42 |
| `C901`/complexity | 15 | Worst: `pipeline/seal.py:34` `_feed_canonical` (C901=26, 67 statements, 17 returns); `_tuning/engine.py:458` `tune` (C901=16) |
| `ERA001` commented-out code | 7 | |
| `T201` `print()` in library code | 3 | `profiling/visualizer.py:29,496,516` |
| `S110` try-except-pass | 2 | `profiling/_analyzer/dates.py:92,103` |
| `TRY004` | 3 | Consistent with still-open `F-30` |

**Impact:** "CI is green" is not evidence of health. The 84 unused arguments in
particular are a mechanical detector for the very class of bug that dominates
this report.

**Fix:** Enable `ARG` immediately (highest signal-to-noise, directly finds
silent-no-op bugs). Then adopt `D` and `PLR2004` incrementally with
`per-file-ignores` for the existing backlog so new code is held to the standard.

---

### OC-10
### ⚪ Low — 4 dead `infer_output_schema` overrides

`vectorization/count_vectorizer.py:128`, `tfidf_vectorizer.py:122`,
`tokenizer.py:157`, `sentence_embedder.py:178` each override
`infer_output_schema` only to `return None`, which is already the base-class
default. Dead code that implies intent where none exists.

**Fix:** Delete the four overrides.

---

### OC-11
### ⚪ Low — Mega smoke test silently skips nodes with empty params

**File:** `skyulf-core/tests/unit/test_all_nodes_smoke.py`

The harness contains `if not params: return`. A node that regresses to a silent
no-op (returning an empty artifact) therefore **passes**. The test is also
pandas-only and asserts no semantics.

My parity harness found 5 nodes producing an empty fit under their own declared
default params: `AliasReplacement`, `InvalidValueReplacement`, `TextCleaning`,
`TargetEncoder`, `WOEEncoder`. For `TargetEncoder`/`WOEEncoder` this is
legitimate (they require `y`). For the other three it means the mega smoke test
never actually exercises them — and OC-19/OC-20 show that two of those three do
in fact have silent no-op bugs in production configurations.

**Fix:** Replace `return` with an explicit `pytest.skip(reason)` so skips are
visible in test output, and add per-node minimal configs that produce a
non-empty artifact.

**Verified sound:** determinism. I re-ran every preprocessing node twice
in-process and across `PYTHONHASHSEED=0/1`. **34/34 nodes produced
byte-identical output**, confirming no reliance on Python's salted `hash()`
(`HashEncoder` correctly uses `hashlib.blake2b`).

---

## Encoding, cleaning, imputation, scaling, drop/missing

### OC-12
### 🔴 Critical — Row-dropping desyncs `X` and `y` on non-unique pandas indexes

> ✅ **Independently re-verified — and it is worse than filed.** Running
> `_drop_missing_rows_apply_pandas` on a 4-row frame with index `[0,0,1,2]`:
> `X_out=3 rows, y_out=4 rows`, and `y_out` still contains `20` — **the label of
> the row that was dropped**. So labels are not merely misaligned, they are
> wrong. Evidence in
> [00-validation-log.md](./opus_core_analysis/00-validation-log.md#oc-12).

**Files:** `preprocessing/drop_and_missing/drop_rows.py:60-67`,
`preprocessing/drop_and_missing/deduplicate.py:44-47`

Both pandas paths filter the target with `y.loc[X_clean.index]`. With duplicate
index labels, `.loc` returns **all rows matching each label**, not the
corresponding positions. `y` can come back longer than `X`:

```text
dropmissing  X index [0, 1]              shape (2, 2)
dropmissing  y index [0, 0, 1]  values ['row0','row1','row2']  len 3

dedup        X index [0, 1]              shape (2, 2)
dedup        y index [0, 0, 1]  values ['row0','row1','row2']  len 3
```

The polars paths do this correctly by tracking row positions.

**Impact:** This is the most severe finding in the audit. Downstream training
either fails on a length mismatch or — if lengths happen to align — **trains on
silently misaligned labels**, producing a model that is wrong with no error
anywhere. Non-unique indexes arise routinely from `concat`, `explode`, and
resampling upstream.

**Fix:** Mirror the polars implementation: capture positional row numbers before
dropping and select `y` with `.iloc`, never `.loc`. Add a regression test using
a duplicate-index frame.

---

### OC-13
### 🟠 High — Drop-Rows UI settings ignored; every canvas run becomes "drop any missing"

**Files:** `frontend/ml-canvas/src/core/utils/pipelineConverter.ts:249-253`,
`preprocessing/drop_and_missing/drop_rows.py:38-41,97-103`

The frontend sends `missing_threshold` and `drop_if_any_missing`. Python reads
only `subset`, `how`, and `threshold`. The UI default
`{drop_if_any_missing: false, missing_threshold: 50}` therefore fits as
`{how: "any", threshold: None}`:

```text
drop rows params: {'type':'drop_missing_rows','subset':None,'how':'any','threshold':None}
drop rows kept index: [0]
```

**Impact:** A user who sets "drop rows with more than 50% missing" gets "drop
rows with *any* missing value". On wide data this can delete almost the entire
dataset, and `y` is filtered to that unintended subset.

**Fix:** Translate in `pipelineConverter.ts`: `drop_if_any_missing === true →
how: "any"`; otherwise convert the percentage into an absolute `threshold`.
Better: add a backend `missing_threshold` percentage mode so the UI and core
speak the same language.

---

### OC-14
### 🟠 High — Iterative Imputer UI estimator choices silently fall back to BayesianRidge

**File:** `preprocessing/imputation/_common.py:103-111`

The UI emits `decision_tree`, `extra_trees`, `knn`. `_build_iterative_estimator()`
matches only `DecisionTree`, `ExtraTrees`, `KNeighbors`.

```text
iter estimator decision_tree -> BayesianRidge
iter estimator extra_trees   -> BayesianRidge
iter estimator knn           -> BayesianRidge
iter estimator DecisionTree  -> DecisionTreeRegressor
```

**Impact:** Every MICE imputation configured from the canvas uses Bayesian Ridge
regardless of what the user selected. Results are plausible, so nobody notices.

**Fix:** Normalise aliases case-insensitively and accept both spellings.

---

### OC-15
### 🟠 High — MinMax/Robust scaler range controls in the UI are ignored

**Files:** `preprocessing/scaling/minmax.py:96-100`, `robust.py:116-123`

The UI stores `feature_range_min`/`feature_range_max` and
`quantile_range_min`/`quantile_range_max`. Python expects `feature_range` and
`quantile_range` tuples. Given UI config `feature_range_min=-1, feature_range_max=1`
and `quantile_range_min=10, quantile_range_max=90`:

```text
minmax feature_range: (0, 1)      # user asked for (-1, 1)
robust quantile_range: (25.0, 75.0)  # user asked for (10, 90)
```

**Impact:** Scaling output is silently wrong whenever a user configures a
non-default range — a common action for neural-network inputs needing `[-1, 1]`.

**Fix:** Map in `pipelineConverter.ts`:
`feature_range: [min ?? 0, max ?? 1]`, `quantile_range: [low ?? 25, high ?? 75]`.

---

### OC-16
### 🟠 High — KNN/Iterative imputers crash on all-missing fitted columns

**Files:** `preprocessing/imputation/knn.py:64-76`, `iterative.py:68-84`,
`_common.py:73-99`

The fit artifact retains every requested column, but sklearn drops
all-missing features during `transform()`. `_sklearn_transform_subset()` assumes
the transformed width equals `len(cols)` and writes by index:

```text
KNNImputerCalculator       pd ValueError  Columns must be same length as key
KNNImputerCalculator       pl IndexError  index 1 out of bounds for axis 1 with size 1
IterativeImputerCalculator pd ValueError  Columns must be same length as key
IterativeImputerCalculator pl IndexError  index 1 out of bounds for axis 1 with size 1
```

**Impact:** An all-null column — extremely common in real data — makes
multivariate imputation unusable instead of skipping the column.

**Fix:** Pass `keep_empty_features=True` where sklearn supports it, or drop
empty columns from the artifact at fit time and emit a node warning.

---

### OC-17
### 🟠 High — SimpleImputer polars mean/median crashes on all-null columns

**File:** `preprocessing/imputation/_common.py:32-37,54-56`

Polars `mean()`/`median()` over an all-null column yields `None`, which is stored
in `fill_values`; apply then calls `fill_null(None)` and raises. Pandas skips
the column instead:

```text
pd {'fill_values': {'b': 1.5}, 'columns': ['b'], ...}
pl {'fill_values': {'a': None, 'b': 1.5}, 'columns': ['a','b'], ...}
pl ValueError: must specify either a fill `value` or `strategy`
```

**Impact:** Artifacts diverge between engines, and polars pipelines fail at apply
time on sparse numeric input.

**Fix:** Filter out columns whose computed statistic is `None`/`NaN`, matching
the pandas behaviour, or adopt an explicit documented `keep_empty_features`
policy for both engines.

---

### OC-18
### 🟡 Medium — One-hot/dummy generated names can collide with existing columns

**Files:** `preprocessing/encoding/one_hot.py:68-92`, `dummy.py:76-99`

Generated names such as `city_a` are never checked against existing columns.
Each engine fails differently:

```text
onehot pandas columns: ['city_a','city_a','city_b']   duplicate? True
onehot polars: DuplicateError  column 'city_a' has more than one occurrence
dummy  pandas columns: ['city_a','city_a','city_b']   duplicate? True
dummy  polars columns: ['city_a','city_b']            # silently overwrote the original
```

**Impact:** Three different wrong behaviours for one input — duplicate labels,
a hard crash, and silent data loss.

**Fix:** Precompute output names at fit time; reject or deterministically rename
collisions. Apply the same policy to `MissingIndicator` and multiclass
`TargetEncoder`.

---

### OC-19
### 🟡 Medium — Alias Replacement exposes a `punctuation` mode that does nothing

**File:** `preprocessing/cleaning/alias.py:21-28,45-52,100-114`

The UI offers `mode: 'punctuation'` and describes it as removing punctuation.
Python stores `alias_type: 'punctuation'`, `_resolve_alias_mapping()` returns
`{}`, and apply coalesces every non-match back to the original value:

```text
alias params: {'alias_type': 'punctuation', 'custom_map': {}}
alias out: ['A.B!', 'yes']    # unchanged
```

**Fix:** Either implement the mode or remove the UI option and point users to
`TextCleaning.remove_special`.

---

### OC-20
### 🟡 Medium — Value Replacement's "empty columns = all columns" UI promise is false

**Files:** `frontend/.../ValueReplacementSettings.tsx:175-182`,
`preprocessing/cleaning/value_replacement.py:163-180,208-221`

The UI states that leaving the column selection empty applies replacements to
all compatible columns. Python calls `resolve_columns(X, config)` with no
default selector, gets `[]`, and both apply paths return unchanged data:

```text
value repl params: {'columns': [], 'mapping': {-999: 0}, ...}
value repl out:    {'a': [-999, 1], 'b': [-999, 2]}    # unchanged
```

**Fix:** Align the contract — either require ≥1 column in the UI, or make the
node treat empty `columns` as "all compatible columns".

---

### OC-21
### 🟡 Medium — WOE additive smoothing is not normalized over categories

**File:** `preprocessing/encoding/woe.py:130-145`

`_column_woe()` adds `reg` to each category count but divides by
`total_pos + reg` / `total_neg + reg` instead of `total + reg * n_categories`.
The results are therefore not probability distributions over bins:

```text
woe actual:              {'a': -0.788457, 'b': 0.820981, 'c': -0.277632}
woe laplace-normalized:  {'a': -0.619039, 'b': 0.990399, 'c': -0.108214}
```

**Impact:** WOE values and IV feature-importance scores are shifted for
imbalanced targets with more than two categories.

**Fix:** Use denominators `total_pos + reg * n_bins` and `total_neg + reg * n_bins`.

---

### OC-22
### ⚪ Low — `TargetEncoder.infer_output_schema` checks an impossible value

**File:** `preprocessing/encoding/target.py:340-360`

```python
if config.get("target_type", "auto") not in ("binary", "regression"):
    return None
```

The frontend and sklearn both use `"continuous"` for regression target
encoding; `"regression"` is never produced. Valid continuous configs therefore
always lose schema prediction.

**Fix:** Replace `"regression"` with `"continuous"`.

---

## Feature generation, selection, vectorization, transformations

### OC-23
### 🟠 High — Polars `ratio` flips the sign of near-zero negative denominators

**File:** `preprocessing/feature_generation/_polars_ops.py:97-112`

The pandas path uses `_safe_divide`, which preserves the denominator's sign when
clamping to epsilon. The polars `ratio` path always substitutes positive
`epsilon`. (Polars `divide` gets this right — `ratio` is the outlier.)

```text
ratio pandas = [0.1, -1999999999.9999998, 0.0]
ratio polars = [0.1,  1999999999.9999998, 0.0]
```

**Impact:** The same pipeline produces **opposite-signed** features depending on
the engine.

**Fix:** Reuse the signed-epsilon logic from `_polars_divide()` in `_polars_ratio()`.

---

### OC-24
### 🟠 High — Polars group aggregates treat null group keys differently from pandas

**File:** `preprocessing/feature_generation/_polars_ops.py:222-234`

Pandas `groupby(...).transform()` drops null group keys, returning `NaN`. Polars
`over()` groups nulls together and emits real aggregates.

```text
group_count pandas = [1.0, nan, 1.0]      polars = [1, 1, 1]
group_mean  pandas = [1.0, nan, 1.0]      polars = [1.0, 2.0, 1.0]
```

**Fix:** Choose a contract and enforce it. For pandas parity, wrap the polars
output in `when(col(group_col).is_null()).then(None).otherwise(...)`.

---

### OC-25
### 🟠 High — RFE "K" chosen in the UI is ignored by the backend

**File:** `preprocessing/feature_selection/_common.py:236-240`

The UI exposes `k`; sklearn's `RFE` takes `n_features_to_select`. Nothing maps
between them, so RFE silently uses its default (half the features):

```text
config: {'method':'rfe', 'k':2, 6 input features}
selected: ['c','d','e']   count 3      # user asked for 2
```

**Fix:** Map `k → n_features_to_select` in the converter, or accept `k` as a
backend alias.

---

### OC-26
### 🟠 High — `HashingVectorizer` UI "none" norm is an invalid sklearn value

**File:** `preprocessing/vectorization/hashing_vectorizer.py:59`

```python
norm = config.get("norm", "l2") or None
```

The UI sends the **string** `"none"`, which is truthy, so it reaches sklearn:

```text
InvalidParameterError: The 'norm' parameter of normalize must be a str among
{'l1','l2','max'}. Got 'none' instead.
```

**Impact:** Selecting "None" normalization in the canvas is a guaranteed runtime
failure. Unlike the other frontend-sync bugs this one is at least loud.

**Fix:** Normalise the string `"none"` to `None` in the backend (and have the UI
store `null`).

---

### OC-27
### 🟠 High — `GeneralTransformation` ignores the UI `standardize` toggle

**File:** `preprocessing/transformations/general.py:34-39,138-139`

The UI exposes `standardize` for Yeo-Johnson/Box-Cox and the converter forwards
it, but the backend always fits and applies `PowerTransformer(standardize=True)`:

```text
config standardize=False
output mean -0.0  std 1.0     # standardized anyway
```

The dedicated `PowerTransformer` node honours `standardize=False` correctly —
the bug is confined to `GeneralTransformation`.

**Fix:** Store `standardize` in the artifact and pass it to both fit and the
apply-time reconstruction.

---

### OC-28
### 🟠 High — Box-Cox transform failures silently return untransformed data

**File:** `preprocessing/transformations/power.py:97-104`

Fit selects Box-Cox columns using **training** positivity. If transform-time data
contains zero or negative values, sklearn raises, the applier logs, and returns
the **original dataframe unchanged**:

```text
train a=[1,2,3], test a=[0,4,5]
PowerTransformer (Pandas) application failed: Box-Cox requires strictly positive data
output: [0.0, 4.0, 5.0]      # untransformed
```

**Impact:** Classic train/serve skew. The model was trained on Box-Cox-transformed
features and is scored on raw ones, with only a log line to indicate it.

**Fix:** Validate transform-time positivity and raise a clear error, or null out
the invalid rows while still transforming unaffected columns. Failing open is
the wrong default for a fitted transform.

---

### OC-29
### 🟡 Medium — `FeatureGeneration` advertises `polynomial` but silently skips it

**File:** `preprocessing/feature_generation/_common.py:24-31`

`FEATURE_MATH_ALLOWED_TYPES` includes `"polynomial"`, but neither engine's
handler dict implements it, so it no-ops without error.

**Fix:** Remove it from the allow-list, or route it to `PolynomialFeatures`.

---

### OC-30
### 🟡 Medium — Datetime extraction ignores the UI output name and overwrites collisions

**Files:** `feature_generation/_pandas_ops.py:173-184`, `_polars_ops.py:181-205`

The UI shows "Output Column Name" for every operation, but datetime extraction
always writes `{source}_{feature}` and never consults `output_column` or
`_resolve_output_col()`. Existing columns are overwritten even when
`allow_overwrite=False`:

```text
input columns: ['dt','dt_year'],  output_column='custom_year'
pandas: {'dt': [...], 'dt_year': [2024]}    # custom name ignored, dt_year clobbered
```

**Fix:** Support `output_prefix`/per-feature names with collision resolution, or
hide the output-name control for multi-output datetime operations.

---

### OC-31
### 🟡 Medium — Frontend wrongly requires a target for unsupervised CorrelationThreshold

**File:** `frontend/.../FeatureSelectionNode.tsx:564-566`

Validation requires `target_column` for every method except
`variance_threshold`, but `CorrelationThresholdCalculator.fit()` ignores `_y`
entirely and is unsupervised.

**Impact:** Valid pipelines are blocked in the canvas unless the user picks a
meaningless target. (Note this is the mirror image of the other findings — here
the frontend is *stricter* than the backend.)

**Fix:** Exempt `correlation_threshold` from the target requirement.

---

### OC-32
### 🟡 Medium — `VarianceThreshold` crashes when all candidates are constant

**File:** `preprocessing/feature_selection/variance.py:38-47`

```text
all_constant  ValueError: No feature in X meets the variance threshold 0.00000
all_null      ValueError: No feature in X meets the variance threshold 0.00000
```

**Impact:** "Remove all zero-variance columns" — a legitimate and common
request — aborts the pipeline exactly when it would be most useful.

**Fix:** Catch the sklearn no-feature `ValueError` and return
`selected_columns=[]` with `candidate_columns=cols`.

---

### OC-33
### 🟡 Medium — `FeatureInteraction` cannot generate single-column self-products

**File:** `preprocessing/feature_generation/interaction.py:173-178`

`_resolve_combinations()` supports self-products via
`combinations_with_replacement` when `interaction_only=False`, but `fit()` skips
everything unless `len(cols) >= degree`, so a single column with `degree=2`
produces nothing.

**Fix:** Only require `len(cols) >= degree` when `interaction_only=True`.

---

### OC-34
### 🟡 Medium — Count/TF-IDF vectorizers crash on empty or stop-word-only corpora

**Files:** `vectorization/count_vectorizer.py:79-80`, `tfidf_vectorizer.py:73-74`

```text
CountVectorizerCalculator.fit(pd.DataFrame({'txt': ['', None]}), {'columns': ['txt']})
ValueError: empty vocabulary; perhaps the documents only contain stop words
```

**Fix:** Catch the empty-vocabulary error and return an empty artifact with a
node warning.

---

## Evaluation, thresholds, explainability

Metric correctness was verified against sklearn ground truth for **30 metrics**;
all standard paths matched exactly. The findings below are edge cases.

### OC-35
### 🟠 High — Multiclass splits missing a class emit binary-only metrics and null curve points

**File:** `modeling/_evaluation/metrics.py:217-237,361-363`,
`_evaluation/classification.py:109-127`

A 3-class model evaluated on a split whose `y_true` contains only two classes is
treated as **binary**: it gains unweighted `precision`/`recall`/`f1` keys, loses
a computable `log_loss`, and emits ROC points with `null` coordinates.

```text
keys: ['f1','f1_weighted','precision','precision_weighted','recall','recall_weighted','roc_auc_ovr']
binary_precision_added 1.0
Metric 'log_loss' failed ... 2 vs 3. Please provide labels.
expected log_loss with labels=[0,1,2] = 0.5736542308362172
ROC (Class 2) points: [(0.0, nan), (1.0, nan)]  →  JSON "y": null
```

**Impact:** The metric contract varies by split composition, so the frontend
receives a different key set than it expects and renders invalid curve points.

**Fix:** Determine binary-vs-multiclass from `model.classes_` (or the probability
column count), not from `y_true`. Pass `labels=classes` to `log_loss`. Skip or
explicitly mark per-class curves whose one-vs-rest target has a single class.

---

### OC-36
### 🟠 High — F1 threshold tuning picks a pathological threshold on single-class validation

**File:** `modeling/_evaluation/thresholds.py:101-111`, `_tuning/refit.py:167-199`

`tune_decision_thresholds` verifies `model.classes_` has two classes but never
checks that the **validation labels** contain both. With no positives, F1 is
tied at 0 for every candidate, and the strict `>` tie-break keeps the first grid
point — a near-zero threshold:

```text
y_val=[0,0,0,0], positive probabilities=[.01,.20,.40,.49]
f1_pos1            threshold 0.009804  pred [1,1,1,1]  pos_rate 1.0
balanced_accuracy  threshold 0.490196  pred [0,0,0,0]  pos_rate 0.0
```

**Impact:** With `tune_threshold=True` on a small or imbalanced split, the model
persists a threshold that classifies nearly everything positive.

**Fix:** Require `np.unique(y_val)` to cover both classes before tuning;
otherwise skip. Tie-break toward the default 0.5 rule.

---

### OC-37
### 🟡 Medium — Binary PR-AUC is dropped for string-labeled classifiers

**File:** `modeling/_evaluation/metrics.py:324-327`

`average_precision_score(y_arr, proba[:, 1])` is called without `pos_label`, so
sklearn defaults to `pos_label=1` and fails on `"no"`/`"yes"` labels:

```text
Metric 'pr_auc' failed: pos_label=1 is not a valid label. It should be one of ['no','yes']
expected pr_auc (pos='yes') = 0.9166666666666665
```

**Fix:** Pass `pos_label=model.classes_[1]`.

---

### OC-38
### ⚪ Low — Clustering metrics treat DBSCAN `-1` noise as a real cluster

**File:** `modeling/_evaluation/metrics.py:432-459`

```text
labels=[0,0,1,1,-1,-1]
with_noise:    n_clusters=3.0  silhouette=0.856246
without_noise: n_clusters=2.0  silhouette=0.858586
```

Currently latent — no shipped clustering model emits `-1` — but it will bite as
soon as density clustering is added.

**Fix:** Add `exclude_noise` support, or document that all labels count.

---

## Profiling — analyzer, statistics, drift, expectations, visualisation

### OC-39
### 🟠 High — NaN-bearing numeric columns publish `nan` stats and leak non-finite JSON

**Files:** `profiling/analyzer.py:215-224`, `backend/eda/tasks.py:221-222`

NaN is counted as missing, but the aggregations still run on the raw float
column, so mean/std/variance/skew/kurtosis all become `nan`:

```text
EDAAnalyzer([1.0, 2.0, nan, 4.0]).x stats:
  {'mean': nan, 'std': nan, 'variance': nan, 'skewness': nan}
pandas skipna equivalent:
  {'mean': 2.3333, 'std': 1.5275, 'skew': 0.9352}
```

The backend then persists `profile.model_dump(mode="json")` — which still
contains Python `nan`.

**Impact:** A single NaN blanks the entire statistics panel, and strict JSON
consumers or DB JSON encoders can reject the payload outright.

**Fix:** Normalise float NaN to null before aggregation (`fill_nan(None)`), and
sanitize `DatasetProfile` with a shared finite-float normaliser before dump.

---

### OC-40
### 🟠 High — PCA/clustering "mean imputation" actually replaces NaN with `0.0`

**File:** `profiling/_analyzer/multivariate.py:46-60`

`_impute_matrix()` claims mean imputation but only fills **polars nulls**. Float
NaNs survive into NumPy and are then zeroed by `np.nan_to_num`:

```text
Xdf = {'a': [10.0, 20.0, nan], 'b': [1.0, 2.0, 3.0]}
_impute_matrix →  [[10.0,1.0], [20.0,2.0], [0.0,3.0]]
expected mean-imputed a → [10.0, 20.0, 15.0]
```

**Impact:** Every PCA projection, component loading, cluster centre, and cluster
assignment is wrong whenever the data contains NaN. `0.0` is not a neutral value
for unscaled data — it is an extreme outlier.

**Fix:** Mirror the sibling `_impute_matrix_drop_empty()`, which already does
this correctly (cast → `fill_nan(None)` → fill nulls with column means).

---

### OC-41
### 🟠 High — Quartiles use nearest-rank, not linear interpolation

**Files:** `profiling/analyzer.py:221-222`, `_analyzer/target.py:154-156`

`pl.Expr.quantile()` is called without `interpolation=`, so polars defaults to
nearest-rank. Pandas/NumPy default to linear:

```text
data = [1, 2, 3, 10]
analyzer q25/q75:  2.0  / 3.0
pandas   q25/q75:  1.75 / 4.75
```

**Impact:** Boxplot hinges and IQR outlier fences are materially wrong on
small/medium datasets, and disagree with what a user computes in pandas.

**Fix:** Pass `interpolation="linear"` consistently.

---

### OC-42
> ✅ **Independently re-verified — worse than filed.** `SKEWNESS_TRANSFORM_THRESHOLD = 1.5`
> (`recommendations.py:7`). For `[1,2,3,4,10]` the biased value actually used is
> **1.1384** (rule does *not* fire) while the correct unbiased value is
> **1.6971** (rule *should* fire) — so the transform recommendation silently
> fails on genuinely skewed data. Kurtosis differs even in sign (biased
> `-0.212` vs unbiased `3.152`). Note also that
> `backend/ml_pipeline/_internal/_advisor.py:195` applies threshold `1.0` to
> **pandas unbiased** skew — two different rules for the same concept.
### 🟠 High — Skewness/kurtosis use biased estimators, breaking a hardcoded threshold

**Files:** `profiling/analyzer.py:223-224`, `_analyzer/recommendations.py:66-78`

Polars `skew()`/`kurtosis()` default to the biased (SciPy `bias=True`)
estimators; pandas reports bias-corrected values:

```text
data = [1, 2, 3, 10]
analyzer:         skew 1.0182  kurt -0.7696
pandas:           skew 1.7636  kurt  3.2280
scipy bias=True:  skew 1.0182  kurt -0.7696
scipy bias=False: skew 1.7636  kurt  3.2280
```

The recommendation engine applies a hardcoded `abs(skewness) > 1.5` rule to the
**biased** value, so this example (pandas skew 1.76, clearly skewed) is **not**
flagged.

**Fix:** Pick an explicit public convention. For pandas parity use bias-corrected
estimators; otherwise rename the fields and re-tune the threshold.

---

### OC-43
### 🟠 High — Correlation drops valid columns/rows instead of a defined missing policy

**File:** `profiling/correlations.py:41-44,100-110`

Two compounding problems: a column containing any NaN gets `std() == nan`, fails
the `> 1e-9` check, and is discarded **as if constant**; and the matrix uses
listwise deletion across all selected columns, whereas pandas is pairwise.

```text
x=[1,2,nan,4], y=[1,2,3,4]
pandas corr_xy:          1.0
calculate_correlations:  None

sparse 3-col frame — complete rows after drop_nulls: 0
pandas pairwise corr: all 1.0
calculate_correlations: None
```

**Impact:** The correlation panel and leakage hints vanish entirely on ordinary
partially-missing data.

**Fix:** Convert NaN → null, then compute pairwise with a documented minimum
overlap count.

---

### OC-44
### 🟠 High — Wasserstein drift thresholds a normalized value but reports the raw one

**Files:** `profiling/drift.py:181-195`,
`frontend/.../drift/_hooks/useDriftReport.ts:77-78`

The backend decides drift with `norm_wd = wd / std_ref` but serializes
`value=float(wd)` alongside the **normalized** threshold. The frontend then
re-applies the threshold to the raw value:

```text
reference_std=1004.771  raw_wd=50.000  normalized_wd=0.049763  threshold=0.1
backend has_drift=False        value > threshold = True
```

**Impact:** A report shows `wasserstein_distance=50.0, threshold=0.1,
has_drift=false` — self-contradictory on its face. Touching the threshold slider
flips the same column to "drifted" without any data changing.

**Fix:** Emit both `raw_value` and `normalized_value`, and have the frontend
compare the normalized field.

---

### OC-45
### 🟠 High — Schema drift is computed but never counted or rendered

**Files:** `profiling/drift.py:76-98`, `frontend/.../drift/DriftTable.tsx:281-288`

`missing_columns` and `new_columns` are computed, but `drifted_columns_count`
counts only per-common-column metric drift, and the UI never references either
field:

```text
missing_columns = ['dropped']
new_columns     = ['new_col']
drifted_columns_count = 0
drift_detected = False
UI empty state: "No drifted columns found — all features are stable."
```

**Impact:** Dropping a training feature in production — one of the most serious
drift events possible — is reported as "all features are stable".

**Fix:** Add first-class schema-drift records; count and render them.

---

### OC-46
### 🟡 Medium — Non-finite floats enter public profile payloads (corrected: they do *not* break the EDA route)

> ⚠️ **Corrected and downgraded 🟠 High → 🟡 Medium after independent
> re-verification** — see
> [00-validation-log.md](./opus_core_analysis/00-validation-log.md#oc-46).
> The EDA profile actually ships through `orjson.dumps`
> (`backend/eda/router.py:237,251`), and **orjson coerces non-finite to `null`**
> (`b'{"mean":null,"std":null}'`) — valid JSON. Pydantic's own
> `model_dump_json()` does the same. The "breaks strict JSON" outcome only
> applies to stdlib-`json` paths such as the monitoring persistence at
> `backend/monitoring/router.py:449,569`. The residual real problem is the
> silent `NaN`→`null` ambiguity (a failed statistic becomes indistinguishable
> from a missing one), which is why this stays a finding at all.

**File:** `profiling/schemas.py:7-17,263-302`

Public schema fields are plain `float`, so `NaN`/`Infinity` pass validation:

```text
json.dumps(profile.model_dump(mode='json'), allow_nan=False)
→ ValueError: Out of range float values are not JSON compliant: inf
```

**Fix:** Sanitize all profile floats to finite values (or `None`) before model
construction. Add a `json.dumps(..., allow_nan=False)` test.

---

### OC-47
### 🟡 Medium — Common-column dtype drift can silently disappear

**File:** `profiling/drift.py:136-153`

A reference numeric column that becomes non-numeric in production is cast with
`strict=False`; all values become null, the column returns `None`, and it is
omitted from `column_drifts`:

```text
reference: a=[1,2,3]      current: a=['x','y','z']
missing=[]  new=[]  column_drifts={}  drifted_count=0
```

**Fix:** Compare dtypes before metric calculation and emit an explicit
`type_drift` alert when a cast fails.

---

### OC-48
### 🟡 Medium — Expectations pass vacuously on empty frames

**File:** `profiling/expect.py:92-209`

`expect_no_nulls`, `expect_value_range`, and `expect_unique` all **pass** on an
empty frame.

**Impact:** A failed ingestion producing zero rows sails through the data-quality
gate.

**Fix:** Add `expect_non_empty`, or default `allow_empty=False`.

---

### OC-49
### 🟡 Medium — Valid partially-unlabelled PCA payloads crash plotting

**Files:** `profiling/schemas.py:153-157`, `visualizer.py:716-737`

`PCAPoint.label` is optional, but `_pca_color_values()` filters out `None`,
producing a colour vector shorter than `x`/`y`:

```text
pca labels=[None,'a','b']
ValueError: 'c' argument has 2 elements, inconsistent with 'x' and 'y' with size 3
```

**Fix:** Emit one colour per point using a sentinel for missing labels.

---

### OC-50
### 🟡 Medium — Binary targets miss class-balance advice or flip to regression by sample size

**Files:** `_analyzer/recommendations.py:147-152`, `_analyzer/_utils.py:39-42`

Balance recommendations accept only `"Categorical"`, excluding real Boolean
targets. And integer 0/1 targets are typed by the ratio
`n_unique / row_count < 0.05`, so the **same target** changes task type with
sample size:

```text
bool target 95 False / 5 True  → dtype Boolean, target recs []
int [0,1]*10  n=20  → Numeric,    task Regression,     balance_recs []
int [0,1]*50  n=100 → Categorical, task Classification, balance_recs [Resample]
```

**Impact:** Severe imbalance goes unreported, and rule discovery can train a
*regressor* for a classification target on small data.

**Fix:** Treat Boolean as categorical; detect binary integer targets explicitly
when `target_col` is supplied, independent of row count.

---

### OC-51
### 🟡 Medium — Transform advice can be invalid and self-contradictory

**File:** `_analyzer/recommendations.py:66-78,129-139`

High skew always suggests "Log or Box-Cox" without checking `min`, `zeros_count`,
or `negatives_count` — both are invalid for non-positive data. Worse, the same
profile also reports it is ready for modeling, because `Transform` is not
counted as an issue:

```text
x = [-100,1,2,3,4,5,6,7]   skew=-2.2544  min=-100  negatives=1
recs = ["Apply Log or Box-Cox transformation to 'x'.",
        "No missing values or constant columns found. Data is ready for modeling!"]
```

**Fix:** Recommend Yeo-Johnson for non-positive data; count any Transform /
Encode / Resample recommendation as "not fully ready".

---

### OC-52
### ⚪ Low — Categorical colour mapping is process-nondeterministic

**File:** `profiling/visualizer.py:710-713`

`_label_color_map()` builds labels from a `set`:

```text
PYTHONHASHSEED=1: {'gold':0, 'silver':1, 'bronze':2}
PYTHONHASHSEED=2: {'bronze':0, 'gold':1, 'silver':2}
```

**Fix:** Use `sorted(...)` or `dict.fromkeys(...)` to preserve first-seen order.

---

<!-- APPEND-POINT: remaining domains -->

## Findings in the remaining six domains

Full detail for OC-53 … OC-81 lives in the per-domain reports, linked from each
summary row above and indexed in
[`opus_core_analysis/README.md`](./opus_core_analysis/README.md). They are not
duplicated here so the two cannot drift.

| Report | Findings |
|---|---|
| [06 — Core, engines, data & pipeline](./opus_core_analysis/06-core-engines-pipeline.md) | OC-62 … OC-65, OC-74 |
| [07 — Outliers, casting, binning, time series, geo](./opus_core_analysis/07-outliers-timeseries-geo.md) | OC-58 … OC-61 |
| [08 — Modeling: estimators, CV & tuning](./opus_core_analysis/08-modeling-tuning.md) | OC-66, OC-67 |
| [09 — Backend: execution, API & services](./opus_core_analysis/09-backend.md) | OC-68 … OC-73 |
| [10 — Frontend node-config layer](./opus_core_analysis/10-frontend.md) | OC-53 … OC-57 |
| [11 — Tests, benchmarks, packaging & CI](./opus_core_analysis/11-tests-packaging-ci.md) | OC-75 … OC-81 |

---

<a id="r1"></a>

## Recommendation R1 — Generate the frontend contract from `@node_meta`

**This is the single highest-leverage change in this report.** It closes the
entire bug class described in [The headline](#the-headline), and it retires
OC-13, OC-14, OC-15, OC-19, OC-20, OC-53, OC-61 and OC-66 *as a category* rather
than one at a time.

### The problem, stated precisely

There are three layers and **zero** shared schema between them:

```text
 FeatureGenerationNode.tsx          pipelineConverter.ts            skyulf-core
 ─────────────────────────          ────────────────────            ───────────
 hand-typed DATE_METHODS[]  ──────► params = node.data      ──────► ALLOWED_DATETIME_FEATURES
 hand-typed dropdown values         (byte-for-byte,                 config.get("rule")
 hand-typed defaults                 no validation,                 if rule == "..."
 hand-typed tooltips                 no renaming)                   else: silently ignored
```

Every arrow is a hand-maintained copy. Nothing fails loudly when they disagree —
a param name Python never reads is simply dropped on the floor, and the run
proceeds happily with defaults.

The eleven confirmed instances of this in the audit are not eleven bugs. They are
eleven *samples* from a population whose size is bounded only by how carefully
each future PR is reviewed.

### Why the obvious fixes don't work

- **"Just add validation in `pipelineConverter.ts`"** — that only moves the
  hand-maintained copy. The TS allow-list still drifts from the Python one.
- **"Just write tests"** — a test can only assert the contract someone already
  wrote down by hand. The contract is the thing that's missing.
- **"Reject unknown params in the backend"** — necessary (see step 3) but not
  sufficient: it catches *misspelled* keys, not a key whose **value** is an enum
  member Python doesn't handle (OC-19's `punctuation`, OC-60's `label`).

### The fix

`@node_meta` already carries the authoritative `params` dict. Make it the source
of truth and generate everything downstream from it.

**Step 1 — Enrich `@node_meta` so it fully describes each param.**

```python
@node_meta(
    node_id="AliasReplacement",
    category="Cleaning",
    params={
        "mode": ParamSpec(
            type="enum",
            choices=("exact", "contains", "regex"),   # <- the real, handled set
            default="exact",
            label="Match mode",
            help="How alias keys are matched against cell values.",
        ),
        ...
    },
)
```

The `choices` tuple must be the *same object* the implementation branches on —
not a copy — so a new branch cannot be added without appearing in the contract.

**Step 2 — Emit the contract as a build artifact.** A tiny script walks the
registry and writes `node-contract.json` + a generated `nodeContract.ts`
(literal-union types, defaults, choices, labels, help text). Commit the generated
file; add a CI check that regenerating it produces no diff — the same pattern
that already guards lockfiles.

**Step 3 — Make drift fail loudly at every layer:**

| Layer | Change | Catches |
|---|---|---|
| TypeScript | Node components import their union types from `nodeContract.ts` instead of hand-typing arrays | Drift at **compile time** — `tsc` fails |
| `pipelineConverter.ts` | Validate `node.data` against the contract before send | Drift at **canvas time**, with a real error |
| Backend | Reject unknown param keys and out-of-`choices` values instead of ignoring them | Drift from **any** client, including direct API |
| CI | Assert every registry id is either present in `nodeContract.ts` or on an explicit `INTENTIONALLY_HEADLESS` allow-list | [OC-06](#oc-06)-class gaps |

**Step 4 — Generate the tooltip/help metadata too.** OC-61 and the
`DATE_METHOD_META` gap are both *metadata* drift, not functional drift; they only
stay fixed if the help text is generated as well.

### Sequencing

Steps 1–2 are additive and can land without touching any node. Step 3's backend
strictness should ship behind a warn-only flag for one release (log every
rejected key) so the real drift surface is measured before it starts failing
builds — that log is also the fastest way to find the instances this audit
missed.

---

## Suggested fix order

Ordered by *risk removed per unit of effort*, not by severity alone.

### Now — silent wrongness reaching users

| # | Finding | Why first |
|---|---|---|
| 1 | [OC-75](./opus_core_analysis/11-tests-packaging-ci.md#oc-75) 🔴 polars version drift | One-line env/lockfile fix that un-breaks 10 tests, **every example notebook**, and a benchmark. Also a prerequisite for trusting any polars result in this report. |
| 2 | [OC-12](#oc-12) 🔴 `X`/`y` desync on duplicate pandas indexes | Silently misaligned training labels — the worst possible failure mode. Polars path is already correct, so the fix is well-specified. |
| 3 | [OC-58](./opus_core_analysis/07-outliers-timeseries-geo.md#oc-58) 🔴 numeric→bool truthiness on polars | Every user hits it; `coerce_on_error` isn't even exposed in the UI. Silent data corruption. |
| 4 | [OC-62](./opus_core_analysis/06-core-engines-pipeline.md#oc-62) 🔴 non-reproducible `fingerprint()` | Defeats the entire purpose of pipeline sealing for any pipeline containing a string-category encoder. |

### Next — wrong results in realistic configurations

| # | Finding | Note |
|---|---|---|
| 5 | [OC-13](#oc-13), [OC-14](#oc-14), [OC-15](#oc-15), [OC-19](#oc-19), [OC-20](#oc-20), [OC-53](./opus_core_analysis/10-frontend.md#oc-53), [OC-61](./opus_core_analysis/07-outliers-timeseries-geo.md#oc-61), [OC-66](./opus_core_analysis/08-modeling-tuning.md#oc-66) | Fix the individual no-ops, then land **[R1](#r1)** so the class cannot recur. Doing R1 *first* is tempting but leaves users on broken behaviour longer. |
| 6 | [OC-16](#oc-16), [OC-17](#oc-17) 🟠 imputer crashes on all-missing columns | Crash, not corruption — loud, but trivially reachable on real data. |
| 7 | [OC-69](./opus_core_analysis/09-backend.md#oc-69) 🟠 unsorted `config.nodes` | Reachable from the shipped UI's own BFS ordering. `_kahn_topological_order` already exists — this is a wiring fix. |
| 8 | [OC-35](#oc-35), [OC-36](#oc-36) 🟠 evaluation edge cases | Wrong numbers on a metrics dashboard are believed. |
| 9 | [OC-39](#oc-39)–[OC-46](#oc-46) 🟠 profiling correctness cluster | Eight related findings in one subsystem; fix as one workstream, not eight tickets. |

### Then — decide the deployment model, then act

| # | Finding | Note |
|---|---|---|
| 10 | [OC-71](./opus_core_analysis/09-backend.md#oc-71) 🟠 no authn/authz | **Confirm intent first.** If Skyulf is single-tenant self-hosted behind a trusted network, this is a documentation task. If it is multi-tenant — or ever will be — it is the highest-priority item in this entire report. The scaffolded `User` model and dead `AUTH_FALLBACK_*` settings suggest the latter was intended. |
| 11 | [OC-72](./opus_core_analysis/09-backend.md#oc-72), [OC-73](./opus_core_analysis/09-backend.md#oc-73) | Fix alongside 10; both are cheap and both become severe the moment auth exists. |

### Ongoing — remove the conditions that let these hide

| # | Finding | Note |
|---|---|---|
| 12 | [OC-76](./opus_core_analysis/11-tests-packaging-ci.md#oc-76) 🟠 parity tests cover 9/100 nodes, never compare output | **Directly caused OC-04, OC-23, OC-24 and OC-58 to go unnoticed.** My ad-hoc harness is a working proof this is cheap. |
| 13 | [OC-77](./opus_core_analysis/11-tests-packaging-ci.md#oc-77) 🟠 `--maxfail=1` + a 45% gate on a 98.4% suite | Two flag changes. Makes every future red run legible. |
| 14 | [OC-01](#oc-01), [OC-02](#oc-02), [OC-78](./opus_core_analysis/11-tests-packaging-ci.md#oc-78), [OC-79](./opus_core_analysis/11-tests-packaging-ci.md#oc-79), [OC-81](./opus_core_analysis/11-tests-packaging-ci.md#oc-81) | The packaging-integrity cluster — small, mechanical, and collectively the difference between "a script in a repo" and "a library". |
| 15 | [OC-03](#oc-03) 🟠 systemic schema misprediction across 22 nodes | Large but mechanical; best done as one sweep with a parametrized test asserting predicted schema == actual schema for every node. |
| 16 | [OC-09](#oc-09) 🟡 narrow `ruff select` | Widening the rule set will surface real debt — do it after the above, not before, or it will bury the signal. |

---

## What I checked and found sound

An audit that only reports defects misrepresents the codebase. These are the
things I deliberately went looking for and did **not** find.

### Verified by execution

| Area | Result |
|---|---|
| **Data leakage in `skyulf-core`** | **Clean.** All 7 fit/apply boundaries traced — per-fold preprocessing, holdout tuning, threshold tuning, final refit, nested CV. Every one re-fits on train only. See [report 08](./opus_core_analysis/08-modeling-tuning.md#leakage-audit-table--all-clean). |
| **Hyperparameter ranges** | **All valid.** 389 (model, param, extreme) combinations constructed and fitted across 30 calculators. Zero invalid declared ranges. |
| **Cross-engine parity** | 34 nodes exercised on both engines: **26 MATCH / 3 DIVERGE / 0 ERROR / 5 no-op.** The 3 divergences are [OC-04](#oc-04), [OC-23](#oc-23), [OC-24](#oc-24). |
| **Determinism** | **34/34 nodes byte-identical** across `PYTHONHASHSEED=0` and `=1`. The only nondeterminism found in the whole audit is cosmetic ([OC-52](#oc-52)). |
| **Test suite determinism** | 3,670 tests, byte-identical pass/fail across repeated runs and under `PYTHONHASHSEED=1`. No flaky, network-dependent or dict-ordering-dependent tests. |
| **Coverage** | **98.40%**, 188 files, **zero** modules below 50%. |
| **Packaging** | sdist and wheel both build cleanly, install standalone into a fresh venv, and import correctly. No `.DS_Store` leakage, no committed `.egg-info`. |
| **Static analysis baseline** | `ruff check`, `ruff format --check` (356 files) and `ty check` all clean at `93d7719e`. |

### Verified by reading and tracing

- **Security (backend)** — path traversal, SQL injection, SSRF, unsafe
  deserialization, upload limits, rate limiting and async-blocking were each
  specifically hunted and each came back clean. The S3 connector's
  `_apply_trusted_endpoint` in particular is a deliberate, correct
  metadata-endpoint SSRF defence. Full checklist in
  [report 09](./opus_core_analysis/09-backend.md#security-checklist).
- **`validate_no_cycles`** — a correct Kahn's-algorithm implementation that
  precisely names loop members and prunes merely-downstream nodes. (Its
  limitation, OC-69, is that it validates *only* acyclicity — not that it is
  wrong at what it does.)
- **`ModelRegistryService.get_next_version`** — atomic `UPDATE … RETURNING` with
  retry-on-`IntegrityError`; the classic read-then-write version race is
  genuinely closed.
- **Artifact path containment** — every local path construction goes through a
  `realpath` + containment check; artifact names are sanitized against `.`/`..`.
- **Optional heavy dependencies** — `shap`, `sentence-transformers`,
  `imbalanced-learn`, `xgboost`, `lightgbm` and `optuna` are all lazily imported
  with clean, actionable error messages, correctly gated behind extras.
- **`infer_output_schema` returning `None`** — explicitly means "unknown /
  data-dependent" (`preprocessing/base.py:97-113`). It is a documented contract,
  not a stub. I checked before flagging.
- **Binning out-of-range behaviour** — investigated as a suspected bug,
  confirmed **intentional and documented** at `bucketing.py:41-50`, and
  **dropped as a false positive**.
- **The 161-file "no dedicated test" list** — a naming artefact, not a coverage
  gap. Spot-checked files show 96–100% coverage via integration tests.

### Prior findings re-verified as genuinely fixed

`F-02` (tuning ignores `random_state`), `F-04` (no class-imbalance handling),
`F-06` (all-folds-failed still returns a model), `F-13` (threshold tuning never
wired), `F-15` (per-fold preprocessing) — all confirmed fixed by execution, not
by reading the changelog.

One prior finding is **reopened**:
[`F-14` → OC-64](./opus_core_analysis/06-core-engines-pipeline.md#oc-64) — the
engine-registry global race was only partially closed.

---

## Method & limitations

**Method.** 15 read-only audit agents partitioned so that no two reviewed the
same files, each required to produce *executed* output for every claim — not
inferred behaviour. On top of that: my own import-graph analysis,
registry-vs-frontend coverage diff, lint-debt survey, packaging/version integrity
check, an all-node cross-engine parity harness, and a determinism harness.
Findings raised by more than one auditor were merged into a single id.

**Limitations, stated plainly.**

1. **The polars results carry an asterisk.** Everything ran in a venv with polars
   1.40.1, below the package's declared floor of 1.43.2
   ([OC-75](./opus_core_analysis/11-tests-packaging-ci.md#oc-75)). Preprocessing
   parity results stand — splitters were out of their scope — but **any polars
   splitter behaviour in this report should be re-verified on ≥ 1.43.2.**
2. **Coverage of `frontend/` is by sampling, not exhaustive.** 71,666 lines were
   reviewed for the node-config contract specifically. Application-level React
   concerns (state management, rendering performance, accessibility) were not
   systematically audited.
3. **Confidence scores are honest, not decorative.** Anything at 6–7/10 is a
   judgement call worth a second opinion before acting — particularly
   [OC-71](./opus_core_analysis/09-backend.md#oc-71), where the *facts* are
   certain (9/10) but the *intent* is not.
4. **This audit changed nothing.** No repository file was created, modified or
   deleted; every repro script lived in `/tmp` and was removed. The only
   artefacts are this report and its per-domain files.
