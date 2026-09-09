# Opus core audit — fix tracker

> **Archived half, split 2026-09-06.** This file holds the closed findings, the
> corrections pass and the fix Log. The 100 findings still open, R1, and their
> reproduction evidence moved to
> [`opus_core_analysis-open_queue.md`](opus_core_analysis-open_queue.md).

**Source audit:** [`opus_core_analysis.md`](opus_core_analysis.md) (master report) +
[`opus_core_analysis/README.md`](opus_core_analysis/README.md) (index of the 19 per-area
report files `00`–`18`).
**Baseline:** commit `93d7719e` (master), audit run 2026-08-31 → 09-01 by 15 parallel
read-only agents (Claude Opus 5). 116 findings: 5 🔴 / 45 🟠 / 44 🟡 / 22 ⚪.
27 agent-phase findings were re-verified by execution (25 stand, 4 worse than filed:
OC-12/18/40/42; 2 corrected: OC-01, OC-46). OC-100 was retracted as a false positive
and is not counted.

**Supplemental review (2026-09-05):** OC-163–168 add six execution-reproduced
findings (2 🟠 / 4 🟡) outside the source audit. OC-169 (1 🟡) was filed the same
day out of the OC-150 fix pass. They are tracked in the queue — closed rows below,
open ones in the live file — with reproduction evidence alongside; the
historical baseline counts above are unchanged.

**File-by-file follow-up (2026-09-05):** OC-170–176 add seven execution-reproduced
findings (7 🟡): three from the preceding source pass and four from the bounded
10-file outliers/time-series batch. Coverage is recorded in
[`core_source_review_2026-09-05.md`](core_source_review_2026-09-05.md).

**Cleaning/encoding follow-up (2026-09-06):** OC-177–182 add six more
execution-reproduced findings (1 🟠 / 5 🟡) from another bounded 10-file batch.
Historical baseline counts remain unchanged.

**Docstring-pass by-product (2026-09-06):** OC-183–186 add four backend findings
(3 🟠 / 1 🟡) surfaced while writing docstrings against the code during the OC-09
pass. Unlike the batches above these were verified by reading the call sites and
grepping for consumers, **not** reproduced by execution. Each changes behaviour
rather than documentation, so all four were filed open for a later session;
OC-186 has since been fixed and pinned by a test (see the Log), leaving
OC-183–185 open. Historical baseline counts remain unchanged.

**Remaining-source continuation (2026-09-06):** OC-187–206 add 20 executed
findings (5 🟠 / 14 🟡 / 1 ⚪). As of 2026-09-09, sixteen are fixed —
OC-188–190, OC-193–198, OC-200–206 — and four remain open. OC-200/201/202/203/205 closed in one pass with
OC-67 from an earlier batch (see the Log). Rows and reproduction evidence are
grouped by domain — 8
profiling, 9 modeling/tuning, 3 evaluation & explainability — with the open ones
in the live queue and the closed ones below.
The original core-source ledger now records
**188/188 files read**; the 45 selected modeling/profiling test files passed
**909 tests** (142 warnings). The exact command is recorded in
[`core_source_review_2026-09-05.md`](core_source_review_2026-09-05.md).
Historical baseline counts remain unchanged; no implementation fixes were made
by this review.

**Fix-pass by-product (2026-09-06):** OC-207 (1 🟠) was filed out of the OC-164
fix the way OC-169 came out of OC-150 — reading `get_fitted_split()`'s callers
to pick a fix strategy showed three doc sites prescribing an input
`optimize_thresholds` transforms a second time. Closed 2026-09-07 by a contract
decision — see the Log; historical baseline counts unchanged.

Both files follow the master report's suggested fix order (4 tiers), then the
remaining findings grouped by domain. R1 (the systemic core↔frontend contract
fix) retires 8 findings as a class and is tracked separately, in the live queue.

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

## Closed findings — by tier and domain

The closed half of the queue, kept in the tier and domain grouping the live queue
uses, so a fixed finding stays where it was filed.

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
| OC-207 | 🟠 | `optimize_thresholds()` transforms its `X_val` internally (`pipeline/_pipeline.py:325`), but `docs/user_guide/threshold_tuning.md:32`, `skyulf-core/README.md:205` and the method's own docstring (`:280`) all tell callers to feed it `get_fitted_split()` output — which is **already** preprocessed. The documented workflow therefore tunes thresholds on double-transformed probabilities, and `predict(use_tuned_thresholds=True)` then applies them to singly-transformed ones (`:365`), so the cutoffs are fitted against a distribution inference never reproduces. Needs a contract decision (fix the three docs, or accept pre-transformed input) before code | decision + small | ✅ fixed 2026-09-07 — the contract decision was "keep the code, fix the prose": `optimize_thresholds` and `predict` already both took raw input and transformed exactly once, so the three narrative sites that pointed callers at `get_fitted_split()` were the defect. See the log entry |
| OC-177 | 🟠 | Pandas `DummyEncoder` changes a known category's encoding with batch composition: after fitting `[1.0,2.0]`, `1.0` encodes as known alone but all-zero when accompanied by `2.5` (`preprocessing/encoding/dummy.py:60-64`) | small | ✅ fixed 2026-09-06 — **broader than filed**: the two engines also learned different category *strings* from the same float data (`["1","2"]` vs `["1.0","2.0"]`), so they emitted differently named indicator columns; both symptoms were one batch-dependent renderer, replaced by a per-value rule shared by the engines. See the log entry |
| OC-164 | 🟠 | `get_fitted_split()` on new data replaces a trained pipeline's preprocessing while retaining its old model — the same input's prediction changed from 50 to −950 (`pipeline/_pipeline.py:234`) | small | ✅ fixed 2026-09-06 — isolation, not invalidation: a throwaway `FeatureEngineer` over the same steps leaves the pipeline's fitted state alone, so predictions are identical before and after. See the log entry |
| OC-163 | 🟠 | `LagFeatures` / `RollingAggregate` sort X without reordering tuple y on both engines — `[3,1,2]` times become `[1,2,3]` while targets remain `[300,100,200]`, silently training on wrong labels (`preprocessing/time_series/lag.py:45,81`, `rolling.py:63,119`) | small | ✅ fixed 2026-09-06 — both engines now derive one positional permutation and hand it to X and y alike, which retired OC-165 and OC-166 in the same pass. See the log entry |
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

### Ongoing — remove the hiding conditions

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-03 | 🟠 | Systemic `infer_output_schema` int→float misprediction across 22 nodes — one sweep + parametrized test (predicted schema == actual schema for every node) | ~1 day | ✅ done - runtime dtype parity covered by parametrized tests; completed row moved from the live queue on 2026-09-09. |
| OC-77 | 🟠 | `--maxfail=1` hides real failure count; `--cov-fail-under=45` vs 98.4% actual (two flag changes, `.github/workflows/skyulf-core-tests.yml:82-87`) | mechanical | ✅ fixed 2026-09-06 — floor raised 45 → 90 against a measured 96% (CI run 34032836551), and `--maxfail=1` removed here **and** in `backend-tests.yml`, which carried the same flag unfilled. See the log entry |
| OC-01 | 🟠 | `skyulf.__version__` ambiguous: stale `0.5.8` dist-info shadows real `0.8.8` (path-order dependent) — packaging-integrity cluster — re-verified 2026-09-05: the venv holds exactly one dist-info (`skyulf_core-0.8.13`), the stale `0.5.8` is gone, and `skyulf.__version__` reports `0.8.13` | small | ✅ resolved by the 0.8.13 install refresh |
| OC-02 | 🟠 | Dev editable install dangling; `import skyulf` fails outside repo — packaging-integrity cluster — re-verified 2026-09-05 by importing from a CWD outside the repo: resolves to `skyulf-core/skyulf/__init__.py` at `0.8.13` | small | ✅ resolved by the 0.8.13 install refresh |
| OC-78 | 🟡 | `py.typed` declared in packaging metadata but file does not exist — packaging-integrity cluster | 1 line | ✅ fixed 2026-09-06 — created `skyulf-core/skyulf/py.typed`, verified first through setuptools' own `build_py` and then inside an actually built wheel. See the OC-05/22/78/79/112/132/141 log entry |
| OC-79 | 🟡 | `joblib` imported at module scope but not in `install_requires` — packaging-integrity cluster | 1 line | ✅ fixed 2026-09-06 — `joblib>=1.3.0` declared in core `install_requires`, and in root `pyproject.toml`/`requirements.txt` for the backend's identical undeclared-import gap. See the log entry |
| OC-81 | ⚪ | No `License ::` classifier / SPDX field — packaging-integrity cluster | 1 line | ✅ fixed 2026-09-06 — owner decided `skyulf-core` = Apache-2.0 with backend + frontend staying AGPLv3, declared **statically** in `skyulf-core/pyproject.toml` so it emits PEP 639 `License-Expression:` rather than the deprecated free-text field, and the three files that contradicted the decision reconciled to it. See the OC-81 log entry |
| OC-09 | 🟡 | Narrow `ruff select` hides ~500 missing docstrings + 84 unused args — **last**, widening first would bury the signal | done — `ARG` declined by decision (121 in-scope sites measured) | ✅ fixed 2026-09-06 — `F401`/`F841`/the `D` family now enforced on `skyulf-core/skyulf/` + `backend/`, 904 docstrings hand-written (82 of them invisible to ruff because `D1xx` is privacy-gated on the whole dotted module path), and `ARG` declined by decision; closed by the owner as “fixed as much as we did, no need to continue”. See the 2026-09-05 and 2026-09-06 OC-09 log entries |
| OC-76 | 🟠 | Cross-engine parity tests originally covered 9 of 100 nodes and never compared applied output | ~3 days | ✅ fixed 2026-09-07 — registry-wide checks now compare applied pandas/Polars outputs and dtypes for 51 comparable registered nodes; structured splitter outputs remain outside the frame comparator |
| OC-04 | 🟡 | Cross-engine dtype divergence in binning outputs (`encoding/dummy.py`, `bucketing.py`) | small | ✅ fixed 2026-09-07 — ordinal/bin-index bucketing now emits signed `int64` on both pandas and Polars; DummyEncoder and MissingIndicator were aligned in the OC-76 pass |
| OC-149 | 🟠 | Clustering evaluation crashes on Polars when a numeric feature is all-null within one cluster | small | ✅ fixed 2026-09-07 — cluster-local Polars means now preserve pandas-compatible `NaN` values |
| OC-196 | 🟡 | GaussianMixture probability prediction omitted the feature/reference filtering used for fit and ordinary prediction | small | ✅ fixed 2026-09-07 — shared clustering prediction preparation now filters both labels and probabilities identically |
| OC-195 | 🟡 | Clustering numeric-feature selection skipped `SkyulfPandasWrapper` | small | ✅ fixed 2026-09-07 — wrapped pandas frames now use native pandas numeric selection and preserve the wrapper contract |
| OC-197 | 🟡 | Polars clustering reference crosstabs crashed when reference columns were named `count` or `__skyulf_cluster__` | small | ✅ fixed 2026-09-07 — crosstab internals now use dedicated collision-proof helper names |

### Remaining — evaluation & explainability

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-38 | ⚪ | Clustering metrics treat DBSCAN `-1` noise as a real cluster (`metrics.py:432-459`) | small | ✅ fixed 2026-09-07 — DBSCAN noise rows are excluded from cluster counts and quality scores; regression coverage added |
| OC-146 | 🔴 | Binary `pr_auc` scored against wrong class on `{1,n}` labels — reports 0.32 vs true 0.97, no warning (`metrics.py:324-326`) | small | ✅ fixed 2026-09-05 |
| OC-37 | 🟡 | Binary PR-AUC dropped for string-labeled classifiers (`metrics.py:324-327`) | small | ✅ fixed 2026-09-05 — same one-arg fix as OC-146 |
| OC-147 | ⚪ | `optimize_thresholds` returns a dict shape that bypasses its own documented binary rule, flipping `>=` to `>` on exact ties (`thresholds.py:66-88`) | small | ✅ fixed 2026-09-05 — with OC-36 |

### Remaining — backend infrastructure

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-68 | 🟠 | Model alias map task-unaware — direct API caller silently trains the wrong estimator family (`_execution/engine/_node_runners.py:1157-1183`) | small | ✅ fixed 2026-09-07 — ambiguous aliases are task-aware and mismatched model/task combinations fail clearly |
| OC-70 | 🟡 | Leakage validator checks for *a* splitter globally, not that *this* branch is protected (`_execution/_leakage_validation.py:189-267`) | small | ✅ fixed 2026-09-08 — every training branch now needs its own splitter or explicit CV; data-dependent ancestors on unprotected branches are reported |
| OC-145 | 🟡 | Crashed cross-validation returns the same `{}` sentinel as a disabled one — job reports success with missing `cv_*` metrics (`_node_runners.py:871-907`) | small | ✅ fixed 2026-09-08 — post-tuning CV exceptions now fail the training node and pipeline; regression coverage added |
| OC-130 | 🟠 | Typo in `FASTAPI_ENV` silently disables the entire production security posture (wildcard CORS w/ credentials, DEBUG=True, no SECRET_KEY check) (`config/factory.py:27-32`) — **worse than filed**: a second, unfiled channel — `FASTAPI_ENV` is not a `Settings` field and pydantic-settings never exports dotenv values into `os.environ`, so the bare `os.getenv` could not see a `.env`-only `production` either; both now fail closed through `resolve_environment()` | small | ✅ fixed 2026-09-05 |
| OC-150 | 🟠 | S3 error "sanitiser" matches credential key names case-sensitively — S3 403 bodies + replayable presigned URLs logged verbatim; duplicated in two files (`connectors/s3.py:31-37`, `artifacts/s3.py:67-73`) — **worse than filed**: executed against real shapes the old helper was a *no-op* on all three leaks and exposed the **secret access key** (the audit only ever demonstrated key IDs and signatures), while separately destroying benign text (`key=reports/2026/q3.csv` → `redacted sensitive S3 error`); both copies deleted in favour of one shape-based `redact_credentials()` | small | ✅ fixed 2026-09-05 |
| OC-153 | 🟠 | Multi-input merge silently switches column-wise→row-wise when a branch changes row count — 5-row set + filtered branch yields 8 rows, 3 duplicates, zero UI warnings (`_merge.py:338-348`) — repro came out **9 rows / 4 duplicates**; fixed by warning, not raising, so appending datasets still works | small | ✅ fixed 2026-09-05 |
| OC-154 | 🟠 | Serving-time feature-order reindex (fix F-02) fails open on column mismatch — returned 213.00 where truth is 321.00 (`deployment/service.py:438-442`) | small | ✅ fixed 2026-09-05 |
| OC-155 | 🟠 | Legacy predict path zero-fills missing features and returns a prediction normally; caller never sees a warning (`deployment/service.py:457-462`) — **worse than filed**: the zero-fill also mutated the caller's DataFrame in place | small | ✅ fixed 2026-09-05 |
| OC-131 | ⚪ | Diagnostics fail open — PSI returns `0.0` on any numeric failure (`profiling/drift.py:474-476`) | 1 line | ✅ fixed 2026-09-06 — `drift.py` was the only module under `profiling/` with no logger, so all three fail-open paths (PSI, KL, and the uncastable-column drop the finding missed) now warn; the finite `0.0` contract is kept and documented, since `None` is a three-layer change and `inf` cannot survive `JSONResponse`'s `allow_nan=False`. See the log entry |
| OC-132 | ⚪ | Dead `dropped_features` branch (key appears exactly once in repo) (`graph_utils.py:534-537`) | 1 line | ✅ fixed 2026-09-06 — branch deleted after confirming no test, fixture or writer references the key; the superseding runtime path is recorded in a docstring at the site so the branch is not re-added. See the log entry |
| OC-152 | ⚪ | Two raw-SQL executors accept unconstrained query strings, zero callers — latent injection sink (`async_connection_manager.py:243-268`) — **broader than filed**: `AsyncSQLiteConnectionManager` carried a byte-identical pair, so four dead sinks were deleted, not two | small | ✅ fixed 2026-09-05 |
| OC-157 | ⚪ | `first_wins` merge strategy reverses output column order, contradicting its docstring (`_merge.py:221-236`) — fixed by dropping the reversed iteration, so order is strategy-independent by construction | small | ✅ fixed 2026-09-05 — with OC-153 |
| OC-159 | ⚪ | Empty filter dict compiles to WHERE-less `DELETE FROM data_sources`/`UPDATE`; dead call path today (`async_sqlite_queries.py:129-146`) | 1 line | ✅ fixed 2026-09-06 — all four sites (sqlite + postgres × delete + update) raise `ValueError` before opening a session; the path is dead today, but `_normalize_filter(None) → {}` means the signature itself accepts the table-wiping input. See the log entry |
| OC-186 | 🟠 | `S3Catalog.exists` skips the option-name mapping that every sibling method applies. `catalog.py:521` builds a throwaway `s3fs.S3FileSystem(**self.storage_options)` from the raw instance options, while `__init__`:275, `load`:452 and `save`:491 all pass through `_prepare_s3fs_options`, which maps `aws_access_key_id`→`key` and `aws_secret_access_key`→`secret` and moves region into `client_kwargs['region_name']`. With AWS-named credentials `exists()` therefore authenticates differently from the methods it is supposed to agree with, and reports `False` for (or errors on) an object `load()` reads fine — so callers that gate on `exists` before `load` take the wrong branch | 1 line | ✅ fixed 2026-09-06 — `S3Catalog.exists` goes through `_prepare_s3fs_options` like the methods it must agree with, which also brings it under the SSRF guard. See the log entry |

### Remaining — direct-audit modules

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-114 | 🟡 | All-null tracked column yields 30 `NaN` autocorrelation lags as real analysis (≥1000-row datasets) (`temporal.py:167-191`) | small | ✅ fixed 2026-09-09 - undefined temporal diagnostics are omitted unless sufficient finite varying observations and finite results support them. |
| OC-112 | ⚪ | Comment and code disagree in the categorical profiler — the comment promises a rendered missing-value marker, the code `continue`s and discards the null category (`profiling/_analyzer/categorical.py:22-30`). *Filed as "disagree about the applied threshold"; the real subject is the null-category marker* | 1 line | ✅ fixed 2026-09-06 — comment-only, no behaviour change; the reasoning for why dropping the null category is correct now lives in the code comment it rewrote. See the log entry |
| OC-148 | 🟡 | PII detector flags ordinary 7+ digit numeric ID columns as "Email/Phone" (`profiling/_analyzer/text.py:107-128`) | small | ✅ fixed 2026-09-07 — phone detection now requires positive format evidence and repeated sample evidence; plain IDs, ZIP+4, and isolated phone-shaped IDs are excluded |
| OC-122 | ð¡ | `TextCleaning` silently ignores unrecognised operation name (`cleaning/text.py:151-153`) | small | â fixed 2026-09-08 |

### Remaining — file-coverage closure

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-143 | 🟠 | RFE ignores the UI's `k`, silently selecting half the features — **duplicate of OC-25**, same file and line; one fix retires both | small | ✅ fixed 2026-09-05 — with OC-25 |
| OC-141 | ⚪ | `invalid_values` param declared in `node_meta` with zero consumers | 1 line | ✅ fixed 2026-09-06 — key deleted from `node_meta` after re-verifying zero consumers across all three layers and the `.ambr` snapshots, behaviour-neutral because `user_picked_no_columns` keys off `columns`. The other half of the divergence (the params the calculator really reads are still undeclared) is left to **R1 step 1**. See the log entry |

### Remaining — cross-cutting & packaging

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-05 | 🟡 | `PowerTransformer` triggers a pandas deprecation that will become an error (`transformations/power.py:101`) | 1 line | ✅ fixed 2026-09-06 — **worse than filed**: casting each destination column to `float64` before the `.loc` write removes a per-column pandas FutureWarning that the surrounding bare `except` would otherwise swallow into a silent no-op, i.e. OC-28's failure mode arriving through OC-05. See the log entry |

### Remaining — encoding / cleaning / imputation / scaling / drop

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-172 | 🟡 | `StandardScaler` crashes on mixed pandas nullable numeric columns containing `pd.NA`; native sklearn and equivalent Polars input succeed (`preprocessing/scaling/standard.py:144,154`, `engines/sklearn_bridge.py:52`) | small | ✅ fixed 2026-09-09 - nullable numeric missing sentinels become NumPy NaN without rounding observed integers; StandardScaler applies numeric arithmetic safely. |
| OC-178 | 🟡 | `HashEncoder` hashes the same missing value into different buckets across Polars, pandas object, and pandas nullable string inputs, even with one shared fitted artifact (`preprocessing/encoding/hash.py:45,76`) | small | ✅ already fixed - verified 2026-09-09: one shared hash artifact gives identical missing-value buckets across pandas object/string and Polars. |
| OC-171 | 🟡 | Pandas `SimpleImputer` silently excludes explicitly selected constant/binary numeric columns for mean/median, leaving missing values unfilled; Polars honors the selection (`preprocessing/imputation/simple.py:173-177`) | small | ✅ already fixed - verified 2026-09-09: explicit constant/binary mean and median imputation fills missing values in both engines. |
| OC-179 | ð¡ | `DummyEncoder(drop_first=True)` retains a single-category indicator on Polars but removes it on pandas, changing feature width across engines (`preprocessing/encoding/dummy.py:33`) | small | â fixed 2026-09-08 |
| OC-180 | ð¡ | Pandas `TextCleaning(normalize_slash_dates)` crashes on `pd.NA` in a nullable string column; equivalent Polars input preserves the missing value (`preprocessing/cleaning/text.py:35-37,116`) | small | â fixed 2026-09-08 |
| OC-181 | ð¡ | `ValueReplacement` coerces every unrecognized boolean mapping key to `False`: mapping `{"banana": true}` changes `[true,false]` to `[true,true]` on both engines (`preprocessing/cleaning/value_replacement.py:31-32`) | small | â fixed 2026-09-08 |
| OC-182 | ð¡ | Encoder auto-detection ignores pandas `StringDtype` columns: Dummy/Hash encoding silently leaves strings untouched unless columns are selected explicitly (`preprocessing/encoding/_common.py:140`) | small | â fixed 2026-09-08 |
| OC-22 | ⚪ | `TargetEncoder.infer_output_schema` checks an impossible `regression` value (`encoding/target.py:340-360`) | 1 line | ✅ fixed 2026-09-06 — the `("binary", "regression")` passthrough was pinned by a test asserting a prediction for a config sklearn 1.8 rejects outright; changed to `"continuous"` and confirmed it really encodes rather than merely being reachable. See the log entry |

### Remaining — feature generation / selection / vectorization / transformations

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-211 | 🟡 | Pandas datetime features depend on prediction-batch composition: prepending a different valid date format makes the original rows' year/month/day missing in both `FeatureGeneration.datetime_extract` and `DateFeatures` (`feature_generation/_pandas_ops.py:179`, `time_series/date_features.py:64`) | small | ✅ already fixed - verified 2026-09-09: mixed-format batch companions preserve calendar features across both engines and aliases. |
| OC-212 | 🟡 | Similarity generation silently omits its output column for duplicate pandas indexes: label-based `.at[i]` returns Series to a scalar helper and the operation exception is swallowed (`feature_generation/_common.py:137-139`) | small | ✅ fixed 2026-09-08 — similarity now reads and assigns by row position, preserving duplicate indexes and individual scores; see the Log entry. |
| OC-25 | 🟠 | RFE "K" chosen in UI ignored by backend (`feature_selection/_common.py:236-240`) | small | ✅ fixed 2026-09-05 — closes OC-143 too |
| OC-28 | 🟠 | Box-Cox transform failures silently return untransformed data (`transformations/power.py:97-104`) | small | ✅ fixed 2026-09-06 — the silent path was the `valid_cols` filter, not the `except` (which has logged since the node was created); both engines now share `_fitted_columns_present`, which names the fitted columns the frame lacks, and fail-open is kept by decision. See the log entry |

### Remaining — profiling (outside the OC-39–46 cluster)

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-193 | 🟡 | A single missing timestamp removes time-series analysis at the 1,000-row resampling boundary: dynamic grouping receives null date keys and the exception is swallowed (`profiling/_analyzer/temporal.py:232,243`) | small | ✅ fixed 2026-09-09 - null timestamps are excluded from temporal calculations without changing the existing resampling threshold. |
| OC-217 | 🟠 | Repeated profiling on one analyzer returns previously excluded columns in sample data and frame statistics because only newly excluded columns trigger the narrowed frame (`profiling/analyzer.py:analyze`) | small | ✅ fixed 2026-09-09 - persistent column exclusions remain effective for sample rows and frame statistics across repeated calls. |
| OC-216 | 🟡 | Rule feature conditions include unrelated labels from Polars shared categorical dictionaries even when those labels never occur in the feature (`profiling/_analyzer/rules.py:_build_feature_matrix`) | small | ✅ fixed 2026-09-09 - feature-local codes and label lists exclude unrelated categories from displayed rule conditions. |
| OC-198 | 🟠 | Profiling a string target overwrites an existing `<target>_encoded` feature, then duplicate selection prevents correlation and causal analysis (`profiling/analyzer.py:341-348`) | small | ✅ fixed 2026-09-09 - encoded targets use an unoccupied column name and temporary analyzer state is restored after analysis. |
| OC-190 | 🟡 | A categorical column named `count` crashes profiling and categorical drift because `value_counts()` generates the same column name (`profiling/analyzer.py:286-290`, `profiling/drift.py:380-381`) | small | ✅ fixed 2026-09-09 - profiling, categorical drift and rule target counts use distinct value/count names. |
| OC-189 | 🟡 | Classification rule text reports `Samples: 1` for leaves containing multiple rows: it sums sklearn's normalized class proportions instead of using the leaf sample count (`profiling/_analyzer/rules.py:299-301`) | small | ✅ fixed 2026-09-09 - classification rule support uses the fitted tree row count while confidence retains class proportions. |
| OC-188 | 🟠 | Rule discovery decodes sklearn class positions against Polars' shared category dictionary, publishing labels absent from the target while reporting perfect accuracy (`profiling/_analyzer/rules.py:169-170,196-198,295-298`) | small | ✅ fixed 2026-09-09 - target-local codes map every rule prediction to the actual observed class label. |

### Remaining — core / engines / pipeline

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-167 | 🟡 | Ambiguous string boundaries in artifact serialization give different fitted label encoders identical pipeline fingerprints, despite encoding the same input as 0 vs −1 (`pipeline/seal.py:52,64`) — distinct from OC-62's pointer instability | small | ✅ fixed 2026-09-09 - typed length framing and canonical unordered entries distinguish different fitted values deterministically. |
| OC-63 | 🟠 | `artifact_digest` raises `RecursionError` instead of the documented `TypeError` on cyclic graphs (`pipeline/seal.py`) | small | ✅ fixed 2026-09-09 - active-path cycle detection raises TypeError and still accepts shared acyclic state. |
| OC-170 | 🟡 | `validate_leakage_safety()` rejects registered stateless nodes before the split as unknown/data-dependent, including `TextCleaning`, `DateFeatures`, `Casting`, and `feature_target_split` (`leakage.py:140-156`) | small | ✅ already fixed - verified 2026-09-09: all four filed stateless nodes are accepted before a split. |
| OC-208 | 🟡 | Failed pipeline refit leaves new preprocessing attached to the previous model (`pipeline/_pipeline.py`, `modeling/base.py`) | half day | ✅ fixed 2026-09-09 — any failed fit clears partial fitted state and blocks prediction until a successful refit. |
| OC-161 | 🟡 | Polars clustering evaluation overwrites a numeric feature named `__skyulf_cluster__`, then fails while computing centroids (`modeling/_evaluation/clustering.py`) | small | ✅ fixed 2026-09-09 — centroid subsets use positional label masks without creating a helper column. |
| OC-160 | 🟡 | DropMissingRows and Deduplicate collide with valid `__idx__` feature or target columns (`preprocessing/drop_and_missing/`) | small | ✅ fixed 2026-09-09 — collision-free X row-position names and direct target gathering preserve user columns and positional alignment. |
| OC-162 | 🟡 | Polars time-series CV overwrites feature columns with temporary or real target names (`modeling/cross_validation.py`) | small | ✅ fixed 2026-09-09 — a shared positional permutation sorts X/y separately without materializing target columns. |

### Remaining — outliers / casting / binning / timeseries / geo

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-176 | 🟡 | Polars `LagFeatures(drop_na=True)` removes nulls but retains float NaN in source/lag columns; equivalent pandas input drops those rows (`preprocessing/time_series/lag.py:54-59`) — independent of OC-165's y desynchronization | small | ✅ fixed 2026-09-09 - Polars lag filtering removes both null and floating NaN rows with one positional selection shared by X and y. |
| OC-175 | 🟡 | Polars `RollingAggregate` propagates float NaN through windows instead of ignoring missing observations like pandas — `[1,NaN,3]` with window 2 yields mean `[1,NaN,NaN]` vs `[1,1,3]` (`preprocessing/time_series/rolling.py:48`) | small | ✅ fixed 2026-09-09 - floating NaN is treated as missing inside Polars rolling expressions, preserving source values and grouped-window semantics. |
| OC-174 | 🟡 | Polars `DateFeatures` crashes on an entirely invalid string date column despite `strict=False`; pandas produces nullable calendar features (`preprocessing/time_series/date_features.py:102`) | small | ✅ already fixed - verified 2026-09-09: wholly invalid date strings produce nullable calendar features in both engines. |
| OC-173 | 🟡 | Duplicate pandas indexes reintroduce missing rows during EllipticEnvelope prediction, disabling outlier filtering (`preprocessing/outliers/elliptic.py`) | small | ✅ fixed 2026-09-08 in `f12dde9f8`; reverified 2026-09-09 — finite-row selection and prediction scatter are positional, preserving indexes and X/y alignment. |
| OC-165 | 🟡 | Pandas `LagFeatures(drop_na=True)` removes X rows but leaves tuple y untouched — 3 rows become 2 features / 3 targets even with a unique index (`preprocessing/time_series/lag.py:85-87`) | small | ✅ fixed 2026-09-06 — with OC-163; `drop_na` now filters y through the same positional keep-mask as X, duplicate-index case included. See the log entry |
| OC-166 | 🟡 | Polars `IQR`, `ZScore`, and `ManualBounds` filter X but leave NumPy y untouched — 5 rows become 4 features / 5 targets; Polars Series y works (`preprocessing/outliers/_common.py:9-15`) | small | ✅ fixed 2026-09-06 — with OC-163, and **broader than filed**: a fourth copy of the same silent pass-through sat inline in `EllipticEnvelope`, and list targets failed too (crashing on pandas, no-opping on polars). See the log entry |

### Remaining — modeling / tuning

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-206 | ⚪ | Ensemble configuration resolution shallow-copies nested base-model parameters, so fitting mutates the caller's configuration (`modeling/ensemble.py:473,484`) | small | ✅ fixed 2026-09-09 - inner base-model parameter maps are copied before temporary overrides, preserving caller settings and later refits. |
| OC-204 | 🟡 | `fit_predict` drops an embedded target during training but keeps it in held-out tuple features when explicit y is also supplied, causing prediction to fail (`modeling/base.py:317-324`) | small | ✅ fixed 2026-09-09 - held-out tuples use training target extraction, excluding embedded targets while preserving explicit-y precedence. |
| OC-168 | 🟡 | Pipeline refitting retains decision thresholds from the previous model (`pipeline/_pipeline.py`) | small | ✅ fixed 2026-09-09 — fitting clears thresholds and requires fresh optimization for both unchanged and new class labels. |
| OC-209 | 🟡 | Time-series tuning drops its time column only from training, corrupting explicit validation concatenation (`modeling/_tuning/engine.py`, `_tuning/splitters.py`) | half day | ✅ fixed 2026-09-09 — named and array validation payloads mirror training's removed time columns without reordering held-out rows. |
| OC-194 | 🟠 | Pandas time-series CV interprets missing-date argsort sentinels as row positions, duplicating/dropping observations (`modeling/cross_validation.py`) | small | ✅ fixed 2026-09-09 — stable positional sorting keeps every row once, with missing dates last and X/y pairing intact. |
| OC-210 | 🟡 | Public `SkyulfPipeline.optimize_thresholds()` rejects hyperparameter-tuned classifiers because it reads `classes_` from the `(fitted_model, TuningResult)` tuple instead of the underlying classifier (`pipeline/_pipeline.py:511`) | small | ✅ fixed 2026-09-08 — threshold search and thresholded prediction resolve classes from the fitted model through the existing unwrap helper; see the Log entry. |
| OC-200 | 🟠 | Halving search accepts an all-NaN score set as a successful best result and refits a model; grid search correctly fails on identical folds (`modeling/_tuning/strategies/runner.py:110-127`) | small | ✅ fixed 2026-09-06 — with OC-205: a search left with no fully-scored candidate now fails with grid's "All trials failed" instead of returning `nan` and refitting. See the log entry |
| OC-205 | 🟠 | Grid/random tuning discards failed folds from each candidate's average, allowing a partially failed candidate to win with an apparently valid score and no failure count in the result (`modeling/_tuning/grid_random.py:91-92`) | small | ✅ fixed 2026-09-06 — with OC-200: a candidate is eligible only if every fold scored, and a partly-failed one is logged as disqualified. See the log entry |
| OC-201 | 🟡 | Optuna skips search-space normalization: `max_depth=['none']` works in grid search but fails every Optuna trial (`modeling/_tuning/strategies/optuna.py:199`) | small | ✅ fixed 2026-09-06 — Optuna now runs `clean_search_space` like grid/random and halving already did. See the log entry |
| OC-202 | 🟡 | Fold-aware tuning wrapper omits `decision_function` and unconditionally advertises `predict_proba`, breaking ROC-AUC scoring for SVC without probability support (`modeling/_tuning/fold_pipeline.py:164-172`) | small | ✅ fixed 2026-09-06 — both response methods are gated by `available_if`, so the wrapper advertises exactly what the wrapped model can do. See the log entry |
| OC-203 | 🟡 | Optuna CMA-ES treats Boolean candidates as integers, turning valid `fit_intercept=[True,False]` into invalid sklearn parameter values (`modeling/_tuning/strategies/optuna.py:113-140`) | small | ✅ fixed 2026-09-06 — one `_is_number` predicate excludes `bool`, so Boolean lists stay categorical under CMA-ES. See the log entry |
| OC-67 | 🟡 | Tuning metrics `pr_auc`/`pr_auc_weighted`/`g_score` crash the entire search (`modeling/_tuning/metrics.py:19-36,127-146`) | small | ✅ fixed 2026-09-06 — `pr_auc` now aliases to `average_precision` and the two names sklearn has no scorer for are built locally. See the log entry |

### Remaining — frontend

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-55 | 🟡 | `tsc --noEmit` fails: `mermaid` declared but not installed (`frontend/ml-canvas/package.json`) | 1 line | ✅ verified stale 2026-09-06 — `mermaid@11.17.2` is in `dependencies`, in the lockfile, installed and lazy-imported into its own chunk; the exact CI `tsc --noEmit` exits 0, `npm run build` succeeds, and the 5 real-parser tests pass. No change needed |

---

## New findings

### 2026-09-06 — remaining-source continuation (findings added as verified)

Filed with 20 findings; 15 of them still open — those blocks and this
batch's context are in [the live queue](opus_core_analysis-open_queue.md).

**OC-205 — failed folds improve candidate eligibility.** Executed public
grid tuning of `KNeighborsRegressor` on five rows (`x=range(5)`, all-zero y),
with two unshuffled folds, metric `mse`, and `n_neighbors=[3]`. Fold 1 fails
because its training set has two rows; fold 2 scores **0.0**. The search
returns a fitted model, `best_score=0.0`, and a trial containing only the
successful mean. The custom loop removes `-inf` failure sentinels before
averaging, so candidates can be compared over different surviving subsets.
**Fix/verification target:** define consistent candidate failure semantics
and expose fold failures; do not present a partial-fold mean as a successful
complete-CV result. Separate from OC-200's all-NaN halving acceptance.

**OC-200 — invalid halving scores still produce a fitted result.** Executed
`TuningCalculator(SklearnCalculator(Ridge,{},'regression')).fit` on
`X=DataFrame({'x':range(4)})`, `y=Series(range(4))`, with metric `r2`, four
folds, `search_space={'alpha':[1.]}` and `min_resources=4`. Grid search raises
`All trials failed` because each validation fold has only one row. Changing
the strategy to `halving_grid` returns a fitted model, `best_score=nan` and
a trial score of `nan`. **Fix/verification target:** require a finite winning
score before logging completion or refitting, consistently across strategies;
preserve actionable scorer failures. This is independent of the invalid
metric-name lookup already tracked by OC-67.

**OC-201 — strategy-dependent null parameter handling.** On
`X=DataFrame({'x':range(40)})`, `y=Series(arange(40)%2)`, tune a
`DecisionTreeClassifier` with two folds, one trial and
`search_space={'max_depth':['none']}`. Executed grid search succeeds with
`best_params={'max_depth':None}`. Optuna instead fails all trials because it
constructs distributions directly from the uncleaned string. **Fix/verification
target:** normalize once before strategy dispatch and pin equivalent candidate
values across grid, random, halving and Optuna.

**OC-202 — fold-aware scoring loses estimator response methods.** Executed
ROC-AUC scoring of `SVC(probability=False)` on 40 alternating-label rows:
the fitted native model returns **0.525**. Wrap the same estimator in
`FoldAwareModelStep` with identity `fit_transform`/`transform`, and the same
scorer raises `AttributeError: This 'SVC' has no attribute 'predict_proba'`.
The wrapper exposes that method even when the estimator cannot implement it,
and does not forward the working `decision_function`. A halving search with
identity preprocessing also reproduces invalid scores, interacting with
OC-200. **Fix/verification target:** delegate response methods conditionally
and preserve decision scores through preprocessing and label mapping.

**OC-203 — Boolean CMA-ES choices become integer candidates.** Executed
`build_optuna_distributions({'fit_intercept':[True,False]}, True)` yields an
`IntDistribution` spanning 0–1 because `isinstance(True,int)` is true. Native
`Ridge(fit_intercept=True)` fits, while integer `fit_intercept=1` raises
`InvalidParameterError`. Public tuning with the Boolean search space, Optuna
and `strategy_params={'sampler':'cmaes'}` fails every trial on the same
40-row dataset. **Fix/verification target:** keep Boolean lists categorical
before numeric-range detection, including a successful end-to-end Boolean
parameter search. The source's comment already promises this behavior.

### 2026-09-05 — OC-163–168 filed: supplemental core review, six additional reproduced bugs

All six findings are now closed. OC-164 and OC-168 are recorded in their
respective fix logs; OC-167 closed with canonical artifact framing on 2026-09-09.

**OC-163 — time-series sort loses X/y alignment (🟠).** With tuple input `X = {time: [3,1,2], value: [30,10,20]}` and `y = [300,100,200]`, fit/apply `LagFeatures` with `columns=["value"], lags=[1], sort_by="time"`, or `RollingAggregate` with `columns=["value"], window=2, sort_by="time"`. Both pandas and Polars return times `[1,2,3]` but targets `[300,100,200]`; the correct targets are `[100,200,300]`. The engine branches sort only X and return the original y. Downstream conversion to NumPy consumes these mismatched rows positionally, silently corrupting supervised training. Locations: `skyulf-core/skyulf/preprocessing/time_series/lag.py:45,81` and `rolling.py:63,119`. **Fix/verification target:** apply the same positional permutation to X and y; cover both nodes, both engines, and sorting combined with lag row removal. This is separate from OC-162's reserved-column collision in cross-validation and OC-165's filtering-only failure.

**OC-165 — pandas lag filtering leaves y unfiltered (🟡).** Fit/apply `LagFeatures` to pandas `X = {value: [10,20,30]}`, `y = [100,200,300]` with `columns=["value"], lags=[1], drop_na=True` and no sorting. Output X has **2 rows**, while y still has **3**; expected y is `[200,300]`. `lag.py:85-87` drops missing rows from the feature frame and returns the original target. The equivalent Polars Series probe correctly returns 2/2 rows, providing an engine control. This is independent of OC-163 and distinct from OC-12, which concerned duplicate-index expansion in `DropMissingRows` / `Deduplicate`. **Fix/verification target:** filter y with the same positional keep-mask as X, including duplicate-index coverage; sorting and filtering must compose correctly.

**OC-166 — Polars outlier helpers skip NumPy targets (🟡).** With Polars `X = {x: [1.,2.,3.,4.,100.]}` and NumPy `y = [10,20,30,40,1000]`, fit/apply `IQR(columns=["x"])`, `ZScore(columns=["x"], threshold=1)`, or `ManualBounds(bounds={"x": {"lower": 0, "upper": 10}})`. Each removes x=100 but returns all five targets: **4 X rows / 5 y rows**. Repeating each probe with Polars Series y returns the correct four targets. The dispatcher accepts engine-neutral NumPy targets, but `_filter_y_polars` in `preprocessing/outliers/_common.py:9-15` filters only Polars Series/DataFrames and silently returns other types. **Fix/verification target:** preserve positional alignment for supported array-like targets, with NumPy and native-Polars controls across the affected nodes. No reserved helper-column name is involved, so this does not duplicate OC-160.

---

## Log

### 2026-09-09 - frontend complexity refactor batch 2: verified

Plan: [`frontend_ccn_refactor_batch2_2026-09-09.md`](frontend_ccn_refactor_batch2_2026-09-09.md).
Baseline `4bd55065`: 212 functions above CCN 8 across 128 of 521 files.
The user authorized multiple Astra 6 agents for this batch; separate owners
handle training settings, preview results, encoding and canvas edges, with
primary-controlled branch coloring, integration and independent review.

- Training settings: entry **72 -> 2**, extracted production helpers at most
  **8**. Original 13 tests plus six new characterization cases pass both before
  and after extraction (**19 tests**); model switches, target synchronization,
  strategy boundaries, CV state, zero seeds and threshold options are covered.
- Branch colors: entry **70 -> 1**, extracted production helpers at most **8**.
  Original 10 tests plus four new characterization cases pass before and after
  extraction (**14 tests**). A temporary differential probe compared ordered
  edge maps for **2,000 varied graphs** against the original with no differences;
  temporary test/reference files were removed. Path ordering, cycles, shared
  edges, multi-handle grouping and target passthrough are preserved.
- Preview results: entry **58 -> 8**, new presentation helpers at most **8**.
  **21 focused tests** pass before and after extraction, covering branch/split
  selection, totals, persistent panes/advisories, confirmation, read-only
  navigation and keyboard resizing. Original hooks/effects remain in the entry.
- Encoding settings: entry **58 -> 6**, extracted production helpers at most
  **6**. **18 tests** pass against both original and final extracted code.
  Coverage includes all seven methods, numeric zero/empty input behavior,
  schema choices, recommendations, metrics and control state across switches.
- Review caught and corrected an introduced encoding dispatch issue for imported
  method names matching inherited object keys. `__proto__`, `constructor` and
  `toString` reproduced three failures in the extraction; own-property guards
  restore the original unsupported-method fallback. All four unknown-method
  cases pass against the original and fixed code. No released audit issue is
  introduced or closed by this review correction.

- Canvas edges: entry **53 -> 2**, six new helpers at most **6**. Original nine
  tests plus nine new characterizations pass before and after extraction
  (**18 tests**), covering path thresholds, grouped split geometry, branch/merge
  styling, hover cleanup, measured controls and stale/invalid measurement fallback.

Three independent Astra 6 reviews found no further regressions. They ran
**40**, **32** and **90** focused tests plus strict CCN 8 checks; UI literals,
effect lifetimes, graph ordering, method dispatch, edge geometry and public
contracts were inspected against the base. A type-only review observation was
addressed: branch labels explicitly allow the existing nullable FlowCanvas value.

Full verification passed **1,712 Vitest tests** across 149 files (**43 new
behavior cases**), **98 Playwright tests**, full ESLint, the expanded strict
CCN 8 gate, TypeScript/production build and all bundle budgets. Main bundle:
**316.3 KiB gzip / 325 KiB budget**; no limits raised. Workflow/package scope,
whitespace and removal of temporary differential/compiler outputs were checked.
Existing diagnostic logs and circular/empty vendor-chunk warnings remain.
The user also confirmed the frontend works after manual testing and requested
a commit of this batch.

The full report now has **204 functions** above CCN 8 in **123 of 555 files**
(previously 212 in 128 of 521); its highest CCN fell from **72 to 52**.
All five entries and 32 new production helper modules are enforced by CI.
Next report-only hotspots: `EnsembleSettings.tsx` (**52**), `VariableRow.tsx`
(**51**), `EDASidebar.tsx` (**47**), `FeatureSelectionNode.tsx` (**46**) and
`EvaluationView.tsx` (**43**). The v0.8.18 notes were updated. No audit finding
is closed by this maintenance; the live queue remains **57 open, 4 parked**.

### 2026-09-09 - frontend complexity refactor batch: verified

Plan: [`frontend_ccn_refactor_2026-09-09.md`](frontend_ccn_refactor_2026-09-09.md).
The five measured hotspots are pipeline conversion, job details, the toolbar,
canvas node cards and inference. Existing API payloads and UI behavior are the
contract; extracted helpers are included in the CCN 8 target.

- Pipeline conversion: main function **138 -> 6**, extracted helpers at most
  **8**. Existing payload tests/snapshots plus six characterization cases pass
  (**53 tests**). A temporary differential probe compared **1,345 payloads**
  with HEAD across dispatch/default/ensemble/graph cases without differences;
  temporary reference/probe files were removed.
- Toolbar: main component **86 -> 7**, all toolbar helpers at most **8**.
  **55 focused tests** pass, including eight new behavior cases; shortcuts,
  read-only restrictions, menus, exports and submission guards are preserved.
- Job details: main component **112 -> 5**, all extracted helpers at most **8**.
  **43 focused tests** pass, including seven new characterization cases verified
  against both versions. Chart/log state stays mounted across tab changes.
- The first three areas pass **151 tests** together and a combined strict CCN 8
  ESLint check. Existing jsdom network-error logging remains in job-detail tests.
- Inference: main component **73 -> 1**, extracted helpers at most **8**.
  **11 focused tests** pass, including three new cases verified before/after;
  sample projection, schema acknowledgement and manual thresholds are covered.
- Canvas cards: main component **75 -> 6**, extracted helpers at most **7**.
  **21 focused tests** pass, covering telemetry, validation timing, branch
  ordering/fallback labels and in-flight job summaries.
- Three independent reviewers found no material regressions across all five
  refactors. The strict frontend CI command now includes their entry files and
  complete helper folders in addition to the original clean Core scope.

Integration caught a main-bundle increase to **326.1 KiB gzip**, above the
existing **325 KiB** limit. Inference now loads its code on first visit while
remaining mounted across view switches. A new browser test failed on eager
loading before the change and passed afterward, also checking unsaved schema
acknowledgement. The final main bundle is **313.3 KiB** and inference is
**13.7 KiB**, with a separate **20 KiB** budget. The loading change also passed
independent review; no existing budget was raised.

Final verification passed **1,669 Vitest tests** (147 files), **98 Playwright
tests**, ESLint, the expanded CCN 8 gate, TypeScript/production build, all bundle
budgets, workflow YAML checks and `git diff --check`. There are **27 new unit
cases** and **one new browser case**. Existing jsdom diagnostic logs and Vite
circular/empty vendor-chunk warnings remain. Production assets were rebuilt.

The full report now has **212 functions** above CCN 8 in **128 of 521 files**
(previously 230 in 134 of 478); its highest CCN fell from **138 to 72**.
`TrainingSettings.tsx` (72) and `useBranchColors.ts` (70) lead the remaining
report-only hotspots. This maintenance closes no audit finding; the live queue
remains **57 open** and **4 parked**. The v0.8.18 notes were updated.

### 2026-09-09 - frontend complexity workflow

ESLint now enforces CCN 8 in `src/core/{api,constants,contexts,factories,perf,
realtime,registry,theme,types}` through `npm run complexity:check`. The frontend
CI gate runs this command before downstream build and browser-test jobs.
`npm run complexity:report` reports all TypeScript/TSX sources without blocking
CI on legacy complexity: the initial inventory found **230 functions** above
CCN 8 in **134 of 478 files**. Other folders remain report-only until cleaned.

Temporary TS and TSX probes verified that CCN 8 passes and CCN 9 exits with an
error; the probes were removed. ESLint, the scoped complexity gate, workflow
YAML checks, **1,642 frontend tests**, production build and bundle-size checks
passed. The full report exited successfully with 230 warnings and no errors.
Vite reported circular and empty vendor chunks during the successful build.
The v0.8.18 note was updated. This CI addition closes no audit finding, so the
live queue remains unchanged at **57 open** and **4 parked** rows.

### 2026-09-09 - OC-172 follow-up: sklearn bridge complexity gate

Lizard reproduced `_convert_single` at CCN 12 against the CI limit of 8.
Nullable numeric normalization now lives in a focused helper, preserving the
conversion predicates, missing sentinels and exact integer-category values.
`_convert_single` is CCN 6 and the new helper is CCN 7.

The full CI Lizard scope (core/engines/scaling/transformations) passed with
`--CCN 8 -w`. Existing bridge, scaling, parity, modeling-wrapper, encoding and
imputation checks passed **313 tests** (12 existing warnings). Scoped Ruff,
formatting, the CI Ty scope and `git diff --check` passed. The v0.8.18 note was
updated; no live issue was reopened, leaving **57 open** and **4 parked** rows.

### 2026-09-09 - profiling/ensemble batch: final verification

The preceding nullable/time-series/tuple batch was committed as `9f44d6f3`
with DCO sign-off and all applicable pre-commit hooks passing. This continuation
closes seven existing findings (OC-114/188/189/190/193/198/206) and two defects
reproduced during repair (OC-216/217), each recorded before its implementation.
Three parallel implementers, root-owned fixes and independent peer review
covered the final code. The batch adds **83 regression cases**.

The final complete Core run passed **6,809 tests**, with **72 skipped**, **412
warnings** and **3 snapshots passed**. Backend profiling, drift, EDA API/router
and task checks passed **79 tests** (30 warnings). Repository-wide Ruff, the
CI Ty scope, formatting for all ten changed Python files and `git diff --check`
passed. Core tests used offline model-cache settings; both test runs used
unique writable temporary directories.

The live queue contains **57 open** and **4 parked** rows. All nine closed rows
appear exactly once in the archive, with their live rows/reproductions removed
and concise v0.8.18 release notes added. OC-71/72/73/185 remain parked. This
continuation is verified in the working tree and is not included in `9f44d6f3`.

### 2026-09-09 - OC-193/114 fixed: temporal missingness and finite diagnostics

The original 999/1000-row probe with one missing date now retains temporal
analysis on both sides of the resampling boundary (OC-193). Null timestamps
are excluded from the local temporal view while the existing total-row-count
threshold and trend aggregation remain unchanged. All-missing date columns
produce an empty temporal result instead of aborting the profile.

ACF requires at least 11 finite varying trend observations and ADF requires 21;
only then are remaining gaps mean-filled (OC-114). All-missing, constant,
insufficient or overflowed series omit diagnostics instead of publishing NaN
lags. Finite ACF remains available when ADF itself returns an undefined result;
both returned ADF statistic and p-value must be finite before publication.

The initial **26 public cases gave 10 failures / 16 passing controls**. Two
large-finite-value controls separately reproduced NaN ADF output, and primary
review found a tiny linear series (`arange(1000) * 1e-140`) returning an infinite
ADF statistic despite finite variance. That public regression also failed before
return-value validation. The final **29 new cases** plus existing temporal tests
passed **48 tests**. Six warnings originate from unrelated sklearn outlier checks
on intentionally extreme inputs. Scoped Ruff, Ty and formatting passed.

### 2026-09-09 - OC-188/189/216 fixed: rule labels, conditions and support

Keeping unrelated Polars categories alive reproduced target labels absent from
the data at reported accuracy 1.0 (OC-188), and absent feature categories in
IF conditions (OC-216). Targets and categorical features now use their own
contiguous codes and matching observed-label lists. Existing Missing and top-ten
plus Other grouping policies remain in place. Classification text uses the tree
leaf's row count (OC-189), matching structured nodes; class proportions still
supply confidence, including impure leaves.

The first rule regressions reported **10 failures / 2 controls passed**, including
the OC-190 high-cardinality target named `count`. Five additional OC-216 cases
failed before feature-local encoding. All **17 new cases** now pass; the complete
rule suite passed **28 tests**, and rules plus analyzer passed **65 tests**.
Independent peer review verified class/feature mappings and regression controls.

### 2026-09-09 - OC-190/198 fixed: profiling helper names preserve user data

A categorical `count` column previously collided with the count field generated
by Polars. Profiling now isolates value/count names inside its aggregate struct;
drift uses a distinct count-column name, and rule discovery aliases its target
before top-ten counting. Exact category counts and shifted/unchanged categorical
PSI are covered, including internal-looking column names.

String or Boolean target encoding now chooses an unoccupied name against every
source column, including excluded columns. The original analyzer frame and lazy
view are restored with try/finally, so repeated calls or later analysis failures
do not leave an encoded helper behind. Real `<target>_encoded` features retain
their values, correlation results and causal graph nodes.

Initial public-path tests reported **8 failures / 9 controls passed**. An
independent failure-path regression also failed against the original analyzer.
All **18 collision cases** now pass, including repeated analysis with excluded
occupied names. Related analyzer/drift/target/correlation/column suites passed
**141 tests** before the separate OC-217 repair; final combined verification is
recorded above. Scoped Ruff, Ty and formatting passed.

### 2026-09-09 - OC-217 fixed: repeated profiling retains exclusions

The same analyzer previously returned `private` in its second sample after
`analyze(exclude_cols=['private'])`, while still omitting that column's profile.
The exclusion helper now reports every column outside the persistent active
selection, so both sample data and frame statistics keep using the narrowed
frame. Exclusion metadata also reflects that active selection on every call.

Three public regressions failed before the fix and pass afterward, covering a
repeated exclusion, omitted later exclusions and unknown later names. They pin
sample keys, missing-cell percentage, duplicate counts, selected-column metadata
and source preservation. Combined repeated-exclusion, name-collision and analyzer
checks passed **58 tests**. Independent review found no material regression.

### 2026-09-09 - OC-217 filed during target-collision review

The collision-fix implementer reproduced excluded data returning in sample rows
on the second call to the same analyzer, even without a target. Root confirmed
that `_apply_column_exclusions` reports only newly removed columns while the
selection itself persists. OC-217 records the sample/statistics exposure before
root applies the bounded active-selection correction.

### 2026-09-09 - OC-216 filed during rule-label repair

Independent execution of categorical rule features with unrelated live Polars
categories reproduced labels absent from the feature in the displayed IF
conditions. This is separate from OC-188's target-class position mismatch.
Recorded as OC-216 in the live queue with its reproduction before implementing
the same-file feature-local encoding correction.

### 2026-09-09 - OC-206 fixed: ensemble overrides preserve reusable settings

A real VotingClassifier fit with a decision-tree base and
`decision_tree__min_samples_leaf=3` previously wrote that temporary setting into
`base_estimator_params['decision_tree']`. Resolution copied only the outer map.
It now copies each inner parameter dictionary before merging overrides; the
existing parameter values and configuration shapes retain their behavior.

All **16 new public-fit regressions failed before the change** and passed after
it. The cases cover voting/stacking classifiers and regressors, flat/nested
payloads, direct training and captured tuning settings. They assert requested
base/final overrides reach the fitted estimators, input dictionaries remain
unchanged, and later fits without overrides recover the original parameters.
The new tests plus existing ensemble integration and unit suites passed
**80 tests**. Production changes are confined to the shared ensemble resolver.

### 2026-09-09 - nullable/time-series/tuple batch: final verification

After committing the preceding batch as `a5e5d991` with DCO sign-off and passing
pre-commit hooks, three parallel implementers fixed OC-172/175/176/204. Primary
and independent peer review caught and resolved integer-category rounding and
empty-frame lag filtering regressions. The four fixes add **173 test cases**.
OC-173 was separately verified as already fixed and archived without a duplicate
implementation or release note.

The final complete core run passed **6,726 tests**, with **72 skipped**, **421
warnings** and **3 snapshots passed**, using offline model-cache settings and a
writable temporary directory. Backend scaling, numeric guards, fold replay,
split-identity and modeling checks passed **109 tests** (two deprecation warnings).
Repository-wide Ruff, the CI Ty scope, formatting for all ten changed Python
files and `git diff --check` passed. This batch is verified in the working tree;
it is not part of the preceding commit.

The live queue now contains **64 open** and **4 parked** rows. All five closed
rows appear exactly once in the archive and none remains in the live table or
its reproduction blocks. Each new fix has a short v0.8.18 entry; the parked
OC-71/72/73/185 rows retain their status.

### 2026-09-09 - OC-172 fixed: nullable numeric scaling and exact category values

Mixed pandas Int64/Float64 columns `[1, missing, 3]` / `[2, missing, 4]`
previously raised an NAType conversion error in StandardScaler fit. The shared
sklearn bridge now normalizes nullable numeric missing sentinels to NumPy NaN
without casting observed values to float. StandardScaler's pandas apply performs
its own floating arithmetic; disabling both scaling flags preserves input dtypes.

The first regression run reported **11 failures and 7 passing controls**.
Follow-up tests pinned clean nullable holdouts, disabled flags and exact integer
targets. Peer review then caught a draft float cast collapsing category IDs
`2**53` and `2**53 + 1` when another numeric column contained NaN. Four real
OneHotEncoder fit/apply cases failed before removing that cast and now pass,
covering native/wrapped pandas and native/nullable missing companions. They assert
held-out indicator values, unknown categories and caller-input immutability.

The final related bridge, scaling, parity, modeling-wrapper, encoding and
imputation run passed **313 tests** (12 existing warnings). This change adds
**35 cases**; mixed categorical inputs and nonmissing integer labels remain
unchanged. The OC-170-176 source-review batch is now fully archived; its coverage
ledger and the per-finding fix logs retain the review and reproduction evidence.

### 2026-09-09 - OC-204 fixed: held-out tuple feature extraction

The public `StatefulEstimator.fit_predict` probe fitted one feature from
`(X, X.target)` but passed two features to prediction when X also contained the
target column. Held-out tuple features now go through the same extraction helper
as training: embedded targets are excluded even with an explicit y, and the
explicit target retains precedence. Caller inputs and row order are unchanged.

The new suite first reported **16 failures and 16 passing controls**, then all
**32 cases passed**. Real plain/tuned classifiers cover independent test and
validation splits, pandas/Polars native/wrapped frames, conflicting explicit and
embedded labels, duplicate indexes and input immutability. Related modeling and
tuning tests passed **107 tests**. Independent peer review found no material issue.

### 2026-09-09 - OC-175/176 fixed: missing values in time-series features

The original rolling probe `[1, NaN, 3]` now gives mean `[1, 1, 3]` for window 2
and minimum count 1 on both engines. Polars floating NaN is converted to null only
inside generated rolling expressions, so the original source column is retained.
Grouped windows, six aggregations and minimum-count rules follow pandas.

Lag filtering now removes null and floating NaN across source/generated columns,
using the same positional selection for X and tuple targets. The original
`[1, NaN, 3]`, lag 1, `drop_na=True` probe retains zero rows on both engines.
Nonfloating columns use null checks only. The initial regression suite reported
**30 failures and 74 passing controls**; all **104 cases then passed**, with
**320 related tests passing**. Coverage includes Float32/Float64, grouped/sorted
windows, native/wrapped engines, dates, strings and multiple target containers.

Peer review caught an empty Polars frame reaching an empty horizontal mask;
filtering now skips frames without columns. Two native/wrapped public regressions
failed before that guard and pass afterward: **106 new cases** and **322 related
tests** now pass. Primary review confirmed the guard preserves the prior no-op.

### 2026-09-09 — OC-173 reconciled: positional EllipticEnvelope filtering already fixed

Commit `f12dde9f8` replaced duplicate-label selection/scatter with finite row
positions on 2026-09-08. The exact filed probe now passes with duplicate and
unique indexes: fitting [-2,-1,-0.5,0,0.5,1,2,100] and applying [0,100,missing,1]
removes 100, retains [0,missing,1], and selects targets [10,30,40]. Primary rerun
confirmed both controls and seven existing regression tests.

Independent verification also passed eight multicolumn pandas/Polars checks with
list, NumPy, Series and DataFrame targets, plus 76 related EllipticEnvelope tests.
Current feature values, index metadata, paired targets and caller-owned inputs
are preserved. No new implementation or duplicate release-note claim was needed;
the stale row and reproduction were moved from the live queue to this archive.

### 2026-09-09 — parallel core fixes: final verification

Closed OC-160/161/162/194/167/63 with 158 new regression cases across row
filtering, clustering, time-series CV/tuning and artifact seals. Three independent
implementers and primary/peer review covered separate file groups. Review-found
pandas target truncation, deep recursion and structured-array hashing issues were
fixed and rechecked before completion.

The final combined core suite passed **6,553 tests**, with **72 skipped** and
**3 snapshots passed**. Backend fold-replay, split-identity, cross-validation and
modeling integration checks passed **169 tests**. Repository-wide Ruff, scoped
Ty, formatting for all ten changed Python files and `git diff --check` passed.
Tests used writable temporary directories and offline model-cache settings.

The live queue now has **69 open** and **4 parked** rows. In addition to the six
new fixes, five previously fixed findings were verified and archived, and OC-03's
already-completed row was moved out of the queue. All twelve rows appear exactly
once in the archive; each new fix has a short v0.8.18 release note.

### 2026-09-09 - OC-167/63 fixed: unambiguous semantic artifact seals

OC-167 reproduced identical digests for object arrays ['a', 'bstr:c'] and
['astr:b', 'c'], plus analogous list/tuple/bytes/dict boundaries. Public fitted
LabelEncoder pipelines shared a fingerprint while transforming 'a' to 0 versus
-1. Typed byte lengths, container counts and canonical child digests now preserve
value boundaries. Unordered values no longer depend on repr/hash-seed ordering,
and dataclass identity includes its defining module.

OC-63 reproduced RecursionError on cyclic list, dict, tuple, object, dataclass,
object-array and fitted-estimator state. Active recursion-path detection now
raises TypeError and removes each visited identity on exit, so shared acyclic
values still hash like independent equal copies.

Primary review also identified pointer-byte hashing for structured arrays with
object fields. Three regressions failed before an explicit unsupported-dtype
guard; these arrays now raise TypeError. Plain object arrays still hash values,
and numeric structured arrays and fitted random-forest trees remain supported.

Peer review also caught reduced recursion capacity and numeric record-padding
bytes entering hashes. The recursive feeder now keeps one frame per list level;
600-level acyclic data succeeds, cycles raise TypeError, and excessive depth
fails with an explicit TypeError explanation. Numeric structured arrays now
traverse field values, retaining dtype/shape while ignoring direct, nested and
subarray padding.

The original 24 cases produced 17 failures and seven passing controls. The
object-containing structured-array follow-up produced three failures and one
passing numeric control; all 28 then passed on primary rerun. Seven depth/padding
cases reproduced six failures and one control before the final adjustment.
The final 35-case seal suite and broader pipeline/serialization checks passed
89 tests. Cross-process checks use distinct PYTHONHASHSEED values. Scoped Ruff,
formatting and Ty checks passed; independent re-review confirmed the corrections.

Compatibility: recompute stored artifact digests and fitted pipeline fingerprints
after upgrading to the corrected encoding. Unfitted topology-only fingerprints
are unchanged; object-containing structured NumPy arrays now fail explicitly.

### 2026-09-09 — OC-162/194 fixed: chronological sorting preserves paired observations

Public CV and tuner regressions reproduced two independent failures: Polars
replaced/dropped `__cv_y__` or a feature matching y's name; pandas used missing-date
`argsort()` sentinels as negative row positions, duplicating and losing rows.
Both engines now derive a stable positional permutation and apply it separately
to X/y. Missing dates sort last, ties preserve input order, and only the time
feature is dropped. Both engines reject unequal X/y lengths; pandas also accepts
list/tuple targets without trying to index a list with a NumPy permutation.

All initial 32 regression cases failed before the change. Peer review caught
silent truncation of extra pandas list targets; both engines now validate lengths
before sorting. Sixteen public CV mismatch cases cover short/long list/tuple
targets across all engines/wrappers (eight pandas failures and eight Polars
controls before the follow-up). All 48 integrity cases now pass, covering public
CV folds and final tuner refit, target-name collisions, duplicate dates/index
labels and entirely missing dates. The final focused CV/integrity/OC-209 holdout
suites passed 104 tests; earlier broader CV/tuning checks passed 253. Scoped Ruff,
formatting and Ty checks passed. Peer re-review found no remaining material issue.

### 2026-09-09 — OC-160 fixed: row filtering preserves user column names

Public DropMissingRows/Deduplicate fit/apply calls reproduced `DuplicateError`
when X or a multi-output y contained `__idx__`. X now receives a collision-free
position column, while DataFrame targets are gathered directly by retained
positions. The real `__idx__` feature still participates in missingness checks
and explicit deduplication keys; helper columns never reach the output.

The new 69-case suite covers pandas duplicate-index controls, native/wrapped
Polars, missingness policies, dedup keep modes, empty outputs, subset keys,
successive helper-name collisions and multi-output targets. Before the fix,
46 cases failed and 23 pandas controls passed. All 69 passed on independent
rerun; the implementer's broader checks passed 444 tests with two skips.
Scoped Ruff, formatting and Ty checks passed; primary review found no issue.

### 2026-09-09 - reconcile previously fixed queue entries

Fresh public-API probes confirmed five filed defects were already fixed by prior
changes; this reconciliation adds no new implementation or release-note claims:

- OC-170: TextCleaning, DateFeatures, Casting and feature_target_split all pass
  leakage validation before TrainTestSplitter, instead of being rejected as unknown.
- OC-171: explicit mean/median imputation fills constant [1, missing, 1] and binary
  [0, missing, 1] inputs with 1.0 and 0.5, respectively; all eight engine/strategy
  combinations passed instead of retaining the missing cell.
- OC-174: DateFeatures on entirely invalid strings returns missing year/month/day
  fields on both engines instead of raising a Polars format-inference error.
- OC-178: one newly fitted HashEncoder artifact gives the same [7704, 4494, 2274]
  buckets for ['a', missing, 'b'] across pandas object, pandas nullable string and
  Polars inputs, with n_features=10007.
- OC-211: the existing calendar batch-composition regression passes all eight
  cases for DateFeatures and FeatureGeneration aliases across pandas/Polars.
  Adding '04/05/2024' leaves the original ISO dates' year/month/day intact.

Moved these five rows to their archive domains and removed their obsolete live
reproductions. Also moved OC-03's already-completed row out of the open queue;
its status was already done before this reconciliation.

### 2026-09-09 — OC-161 fixed: clustering preserves internal-looking feature names

Public `evaluate_clustering_model()` reproduced `ColumnNotFoundError` when a
numeric feature was named `__skyulf_cluster__`: attaching labels replaced the
feature, and dropping the helper removed it. Centroid calculation now filters
the original numeric frame using the existing label mask, preserving all input
columns and values without an intermediate helper column.

Six regression cases cover pandas, native/wrapped Polars, and ordinary/noise
cluster labels; they compare full evaluation reports and caller-owned data.
Before the fix, four Polars cases failed and both pandas controls passed.
After the fix, all 93 clustering evaluation/integration tests passed; scoped
Ruff and formatting checks passed.

### 2026-09-09 — OC-208/168/209 final verification

The final implementation passed the complete core suite: **6,395 passed,
72 skipped, 3 snapshots passed**. Tests used a writable temporary directory
and offline model-cache settings. Backend fold-replay, split-identity,
cross-validation and threshold service/router regressions passed **198 tests**.
Repository-wide Ruff, scoped Ty, formatting checks for all five changed Python
files and `git diff --check` passed. Independent review confirmed the failed-fit
lifecycle, threshold reset and final validation alignment without remaining
material findings.

### 2026-09-09 — OC-209 fixed: time-series holdout feature parity

Reproduced Ridge tuning on clean `x/time/target` data split into 18 training,
6 validation and 6 test rows: ordinary tuning succeeded, while time-series
tuning introduced NaNs by concatenating training without `time` and validation
with it. The tuner now applies the training time-column removal to both
`validation_data` and raw `validation_frames` before conversion/search. Named
frames and NumPy arrays are supported; already-selected validation features,
held-out row/target order and caller-owned frames are preserved.

Added 18 direct-tuner cases and four pipeline holdout cases, covering
pandas/Polars, explicit numeric and inferred datetime keys, and already-selected
validation. The initial reproduction had 12 failing cases and eight passing
controls. Review added a one-hot encoding case where raw and prepared NumPy
payloads have the same width: positional selection incorrectly removed a real
feature and skipped threshold tuning. That regression failed while its named
control passed; arrays supplied alongside preprocessing and raw validation
frames now keep their prepared features. All 72 focused holdout, lifecycle,
core-pipeline and fold-refit tests passed; scoped Ruff and Ty checks passed.

### 2026-09-09 — OC-168 fixed: decision thresholds belong to the current fit

Reproduced stale thresholds after refitting a logistic classifier: unchanged
labels reused the previous cutoffs, while new string labels raised a missing-class
threshold error. Starting a fit now clears `_tuned_thresholds`; thresholded
prediction directs callers to optimize again for the current fitted model.
Validation rejected before learning still preserves the prior fitted state.

Four new lifecycle cases cover ordinary and tuned classifiers with unchanged
and relabeled targets, plus successful optimization and prediction after refit.
All four failed before the reset. The lifecycle, threshold, leakage-validation
and core-tuning suites passed 66 tests; scoped Ruff and Ty checks passed.

### 2026-09-09 — OC-208 fixed: failed refits invalidate partial fitted state

Reproduced the original StandardScaler/LinearRegression case: prediction at
`x=25` changed from 50 to -150 after replacing the training data with shifted
features and all-missing targets failed. `fit()` now invalidates prior model
metadata before learning and clears partial preprocessing/model state on failure,
including errors after the replacement model has fitted. Prediction checks fitted
state before applying preprocessing. A successful refit restores normal use.

Added 12 real regression cases in `test_pipeline_fit_lifecycle.py`, covering
pandas/Polars, plain models/tuners, failures during preprocessing, model training
and held-out processing, and successful recovery. All 12 failed before the fix.
The lifecycle, threshold, leakage-validation and core-tuning suites then passed
62 tests; scoped Ruff and Ty checks passed. OC-168 threshold reset remains a
separate lifecycle fix.

### 2026-09-08 — OC-210 fixed: threshold optimization after hyperparameter tuning

**Reproduction:** fit a logistic-regression grid tuner with
`search_space={"C":[1.0]}`, `metric="accuracy"`, and `cv_folds=3` on
`x=arange(30, dtype=float)`, `target=[0,1]*15`, using train rows `[:24]` and test
rows `[24:]`. Fitting succeeded, but
`pipeline.optimize_thresholds(raw_X, y, accuracy_score)` rejected the classifier
because it read `classes_` from `(fitted_model, TuningResult)`. Thresholded
prediction independently read `classes_` from the same tuple.

Both sites now use `StatefulEstimator._unwrap_tuned_model()`, the existing
model-resolution path used by evaluation. Probability prediction continues
through the tuning applier with the complete artifact. No signatures or
threshold-selection rules change.

**TDD/verification:** the new end-to-end regression failed in all three class
configurations before the fix: numeric binary, string binary, and string
multiclass. All now optimize thresholds and reproduce predictions from the
fitted classifier's probabilities while retaining default predictions. A
tuned-regressor control still rejects threshold optimization. The complete
pipeline threshold module passes **11 tests**. Joint verification with OC-212:
**5,492 core tests passed, 70 skipped** (benchmarks explicitly skipped),
**40 targeted backend tests passed**, `ruff check .` and the configured full
`ty check` passed. The pre-existing two ty diagnostics in the repeated-split
test were resolved by asserting the actual DataFrame/Series type before
comparison; that updated test passed all **4 cases**. Changed Python files
pass `ruff format --check`. Independent review found no actionable issues.

This closes the standalone core post-training API issue, separate from automatic
tuning thresholds, backend job threshold endpoints, and the OC-168 lifecycle issue.

### 2026-09-08 — OC-212 fixed: similarity preserves repeated row labels

**Reproduction:** configure `FeatureGeneration` with a `similarity` operation,
`input_columns=["a"]`, `secondary_columns=["b"]`, `output_column="score"`.
For `a=["apple","pear"]`, `b=["apple","pear"]`, the unique-index result was
`score=[100.0,100.0]`. Changing only the index to `[7,7]` removed `score`
entirely and logged `The truth value of a Series is ambiguous`. Label-based
`.at[i]` returned Series to the scalar scorer, and the dispatcher skipped the
failed operation.

`_vectorised_similarity()` now selects pairs and stores scores by position.
The original index, row order, null/empty handling, and input columns remain
intact. The public fit/apply regression covers raw and wrapped pandas with
repeated nonmonotonic labels, different scores sharing one label, missing values
and empty strings. Both parametrizations failed at the missing output-column
assertion before the fix and pass afterward. The four feature-generation
integration suites pass **197 tests**. Independent checks also passed for
MultiIndex, missing index labels, and the fallback scorer. Joint full-suite and
lint/type results are recorded in the OC-210 entry above.

### 2026-09-08 - OC-70 fixed

The backend leakage gate now checks each leaf training/tuning branch
independently. A data-dependent ancestor such as `StandardScaler` on a branch
without its own splitter is reported, even when another branch has a valid
splitter. Explicit CV on the training node remains an accepted protection.
Regression coverage includes the unprotected branch and both fixed-mode and
tuned-mode CV configuration.

### 2026-09-08 - OC-145 fixed

Post-tuning cross-validation failures were caught by `_run_tuned_cv`, logged,
and converted to the same empty metrics mapping used when CV was disabled. The
training node therefore completed successfully without `cv_*` metrics. The
exception now propagates through the engine's existing node-failure path, so
both the node and pipeline report `failed` while retaining the original error.
`test_run_tuned_cv_exception_is_caught` now reproduces the crash and asserts the
failure status and error text; the 57-test node-runner integration file passes,
along with targeted Ruff and ty checks.

### 2026-09-08 - OC-122, OC-179, OC-180, OC-181, OC-182 fixed

`TextCleaning` now rejects unknown operation names so invalid configs fail fast.
`DummyEncoder(drop_first=True)` behavior is now parity-stable for single-category
columns in both pandas and Polars; nullable `pd.NA` no longer crashes
`normalize_slash_dates`; `ValueReplacement` only applies declared boolean
mapping keys instead of coercing unknown ones; and encoder auto-detection now
treats pandas `StringDtype` columns as textual inputs for auto-selection.
Core parity and unit tests for all five IDs were updated and pass.

### 2026-09-07 — OC-68 fixed: ambiguous model aliases now respect the task

The backend model factory no longer maps the ambiguous `random_forest` alias to
the classifier unconditionally. Callers can provide `task_type` (also accepted
as the legacy `task` or `problem_type` parameter), which selects the classifier
or regressor registry entry; omitting it now fails closed. The factory also
checks explicit model IDs against the requested task, preventing a successful
but nonsensical classifier-on-regression run. Regression coverage pins all
three cases.

### 2026-09-07 — OC-38 fixed: DBSCAN noise is excluded from clustering metrics

`calculate_clustering_metrics()` now removes rows labelled `-1` before counting
clusters or calculating silhouette, Calinski–Harabasz, and Davies–Bouldin
scores. This matches DBSCAN's contract that `-1` means noise, not a learned
cluster. A regression test verifies both the reported cluster count and the
feature/label arrays passed to the quality metrics; the no-noise path remains
allocation-free for the existing large-label memory guard.

### 2026-09-07 — OC-148 fixed: phone PII detection no longer flags ordinary identifiers

The previous phone heuristic treated any 7–20 character numeric/separator
string with at least seven digits as a phone number, so customer IDs, order
references, and ZIP+4 values produced false `PII` alerts. It also flagged a
whole column from one matching sample. The detector now requires a `+` prefix
or phone-style separators plus 10–11 digits, and requires two matching phone
samples when the sample contains at least two values. Email detection remains
unchanged. Regression coverage includes long numeric IDs, ZIP+4 values, an
isolated phone-shaped ID, real phone formats, email, and plain text; all 7 PII
tests pass, with Ruff and Ty clean.

### 2026-09-07 — OC-76 fixed: applied pandas/Polars output parity is now enforced

Added registry-wide applied-output parity coverage for 51 comparable non-modeling
nodes, including output values, columns, row alignment, target passthrough, and
dtypes. The new checks exposed and fixed the `DummyEncoder` indicator dtype
divergence (`int8`) and `MissingIndicator` flag dtype divergence (`int64`).
Structured splitter results are intentionally excluded from the frame comparator
because they return `SplitDataset` objects; their existing artifact and wrapper
parity checks remain active. The registry contract suite passed 229 tests, with
Ruff and Ty checks also passing. The remaining bucketing dtype issue stays open
as OC-04.

### 2026-09-07 — OC-04 fixed: ordinal bucketing dtype is aligned

The applied Polars bucketing path cast ordinal and bin-index categories to
`UInt32`, while pandas emitted `int64` for complete inputs. The cast now uses
signed `Int64`, matching pandas and preserving the existing nullable behaviour
when missing values are present. A no-null cross-engine dtype regression test was
added; the bucketing suite, registry contract suite, Ruff, and Ty all pass.

### 2026-09-07 — OC-149 fixed: cluster-local all-null means match pandas

Polars returns `None` for the mean of a feature that is entirely null within
one cluster, while pandas returns `NaN`. The Polars centroid path now converts
that `None` to `float("nan")`, preventing `float(None)` from aborting the whole
clustering report and preserving the pandas-compatible centroid/profile input.
The clustering evaluation and integration suites, Ruff, and Ty all pass.

### 2026-09-07 — OC-196 fixed: GaussianMixture probabilities use fitted features

`predict_proba()` inherited the generic sklearn applier and bypassed the
clustering-specific reference-column and numeric-only filtering used by
`predict()`. A shared preparation helper now feeds both methods, so a fitted
`reference_column` or extra text column cannot change the feature count passed
to GaussianMixture. Clustering integration/modeling tests, Ruff, and Ty all pass.

### 2026-09-07 — OC-195 fixed: wrapped pandas clustering keeps numeric features

`SkyulfPandasWrapper` delegated `.columns` and `.dtypes`, so the explicit
Polars branch misclassified it and dropped every feature. `_select_numeric_features`
now recognizes the wrapper, selects numeric columns from its native pandas frame,
and returns a wrapped pandas subset. The clustering/modeling suites, Ruff, and
Ty all pass.

### 2026-09-07 — OC-197 fixed: Polars crosstab helper names are collision-proof

The Polars reference crosstab used `__skyulf_cluster__` for its temporary
cluster key and `count` for its aggregation output. Either name could already
be the user-selected reference column, causing a `DuplicateError`. The path now
uses dedicated internal cluster, reference, and count names independent of user
columns. Clustering evaluation/integration tests, Ruff, and Ty all pass.

### 2026-09-07 — OC-207 fixed: the threshold-tuning contract was right and the three docs describing it were wrong — plus the silent degenerate search the follow-up measurement exposed

Reproduced against `HEAD` before any edit, on the auditor's own 200-row setup
(`x ~ N(0,1)`, `target = (x + N(0,0.4) > 0)`, `StandardScaler` +
`TrainTestSplitter(test_size=0.25, random_state=42)` + `logistic_regression`).
`get_fitted_split()` returns an `X_val` with mean `0.0766` / std `0.9774` —
already standardized — and feeding it to `optimize_thresholds()` standardizes it
again at `pipeline/_pipeline.py:325`. On identical rows and one fitted pipeline,
the only difference being whether `transform()` runs a second time, `p(class=1)`
for the first six rows moved from
`[0.9526, 0.0888, 0.0404, 0.0241, 0.1377, 0.0826]` to
`[0.9560, 0.0783, 0.0343, 0.0200, 0.1244, 0.0725]` (largest shift `0.0161`), one
of 50 rows flipped class, and measured accuracy on the fold went `0.8800` →
`0.9000`. That last number is why the defect stays invisible: the double
transform is not merely different, it scored *better* on the tuning fold, so
nothing looks wrong from the metric alone.

**The harm is the mismatch, not the flip.** `predict(use_tuned_thresholds=True)`
transforms its caller's raw input exactly once (`:373`) before
`apply_thresholds`, so cutoffs fitted on twice-transformed probabilities get
applied to singly-transformed ones — systematically, and with no error.

**Decision (contract): keep the code, fix the prose.** The code was already
self-consistent — `optimize_thresholds` and `predict` both take raw input and
transform exactly once, the `Args:` entry at `:292` already said "*not* yet
transformed", and `FeatureEngineer.transform` skips splitters
(`preprocessing/pipeline.py:77-84`), so a raw holdout flows through correctly.
Three *narrative* sites contradicted that one signature. The auditor's other
option — dropping the internal `transform()` so `get_fitted_split()` output
becomes valid input — was rejected: it makes tuning asymmetric with inference
and contradicts every existing test. A second reason to reject it surfaced while
reading `get_fitted_split()`: it fits a **throwaway** chain over the configured
steps, so its frames are not guaranteed to be the pipeline's *own* fitted
transform at all. A raw holdout passed through the pipeline's own chain is the
only input that reproduces what `predict()` sees.

**Correction, measured after the fix landed — the first pass at this entry got
the severity backwards.** It called the threshold *dict* "a poor observable",
because on the reproduction above tuning on the raw holdout and on the
preprocessed one returned the same `{0: 0.3627, 1: 0.6373}` (`accuracy_score` is
piecewise constant in the cutoff and both landed on the same plateau). That was
one lucky dataset out of many: sweeping **32** configurations (8 seeds × four
size/feature-offset combinations), the raw and pre-transformed dicts **diverge
in 27 of them**. The filed reproduction was among the 5 coincidences, so it
understated the defect rather than characterising it. With feature *magnitude*
in the mix the pre-transformed path is far worse than the filed numbers suggest
— at `x ~ N(20,1)` the double transform saturates the probabilities (largest
shift **0.9994**), flips **32 of 50** hard predictions, and collapses the
returned dict to the untuned default `{0: 0.5, 1: 0.5}`. The dict is therefore a
*sensitive* observable; the probability distribution the regression test pins is
simply sensitive in 32/32 configurations rather than 27/32, which is why the
test pins that and not the dict.

**Unfiled, found while measuring that: the search gave up in silence.** The
collapse to `{0: 0.5, 1: 0.5}` is the grid's tie-break doing exactly what OC-36
designed — but it emitted nothing. `_grid_search_binary`
(`modeling/_evaluation/thresholds.py:132-146`) warned on the *single-class*
degenerate case and stayed mute when both classes were present yet every
candidate cutoff scored identically (saturated probabilities, or a metric that
ignores its predictions). Verified in isolation: `y_true` covering both classes,
all 101 cutoffs scoring `1.0`, `{0: 0.5, 1: 0.5}` returned, **zero** log
records. A caller could not distinguish "tuning gave up" from "0.5 really is
optimal" — and that silence is precisely what let OC-207's worst symptom pass
for a successful tune. **Fixed:** the loop now tracks the worst score beside the
best and warns when they coincide, mirroring the single-class message. An
all-NaN sweep still falls through silently, because both sentinels keep their
infinities and never compare equal — the existing
`test_optimize_thresholds_nan_scores_fall_back_to_default` behaviour is
untouched. End-to-end on the `x ~ N(20,1)` case: the raw holdout tunes to
`{0: 0.4902, 1: 0.5098}` with no warning, the pre-transformed one returns
`{0: 0.5, 1: 0.5}` **and** logs "every candidate cutoff scores 0.6000 on the
validation split". Two tests pin it —
`test_optimize_thresholds_all_tied_candidates_warn`, plus a guard-the-guard
`test_optimize_thresholds_stays_quiet_when_the_grid_discriminates` that first
asserts its own premise (more than one distinct score across the grid) so it
cannot pass vacuously and prove only that the warning is noisy. No new OC number
was minted: this is a by-product of the OC-207 measurement, logged here the way
the OC-177 entry logs its own unfiled engine-divergence find.

**Changes.** `docs/user_guide/threshold_tuning.md` — the Pipeline-usage example
now carves a raw holdout with `train_test_split` before `fit()`, explains that
the single internal transform is what makes tuning and inference agree, carries
an explicit "do not tune on `get_fitted_split()` output" paragraph, and states
the nested-split consequence when the config already holds a
`TrainTestSplitter`. Its `get_fitted_split()` link was repointed from
`validation_vs_sklearn.md` — which never mentions the helper — to
`reference/api/pipeline.md`, which renders its docstring.
`skyulf-core/README.md` — the Threshold-tuning paragraph drops the
`get_fitted_split()` recommendation and names it as not a valid source.
`_pipeline.py` — the `optimize_thresholds` narrative now agrees with its own
`Args:`, and `get_fitted_split()`'s `Returns:` warns that its frames come back
preprocessed and belong to a raw sklearn-style estimator, not back into this
pipeline.

**Deliberately kept.** No behaviour change anywhere; `_pipeline.py` is
docstring-only. `get_fitted_split()` itself is unchanged and still correct for
its documented purpose.

**Tests.** Three tests in
`skyulf-core/tests/unit/test_pipeline_threshold_tuning.py` sourced `X_val` from
`get_fitted_split()` and so encoded the broken workflow; they now use a shared
`_raw_holdout()` carve, with their assertions unchanged in intent. Added
`test_optimize_thresholds_sees_the_same_probabilities_as_predict`, which spies on
the per-pipeline applier's `predict_proba` (instantiated fresh at
`_pipeline.py:89-90`, so patching the instance stays contained) and asserts
tuning and `predict(use_tuned_thresholds=True)` each make exactly one call with
identical probability arrays for the same raw rows. It carries a teeth check:
the same rows handed over pre-transformed must produce a *different*
distribution. That check needed a `StandardScaler` config (`_scaling_config`),
because the existing mean-imputer chain is idempotent — imputing an
already-imputed frame changes nothing, so a double transform is unobservable
through it.

**Mutation-checked:** replacing `transform(X_val)` with `X_val` at `:325` fails
all four tuning tests, so the suite detects a contract break in both directions.

**Gates.** Re-run after the `thresholds.py` change: `skyulf-core/tests` **3813
passed, 56 skipped** (140s — 3811 before the two new tests); root `tests/unit`
plus the threshold-router, threshold-tuning-service and pipeline-config-snapshot
suites **957 passed**, 7 snapshots passed; `ruff check .` and
`ruff format --check .` clean (673 files); `ty check` clean after narrowing
`model_estimator` (typed `StatefulEstimator | None`) behind an explicit
precondition assert. Changelog bullets added to the still-open `v0.8.15` section
— no `v0.8.15` tag exists yet.

### 2026-09-06 — OC-177 + OC-164 fixed: the Next tier's last two rows, and both were one root cause rather than the filed symptom

Both were reproduced against `HEAD` before any edit. Both turned out to have a
single clean root cause, and both fixes came out broader than the queue row.

**OC-177 — `DummyEncoder`'s category rendering was a function of the batch, not
the value.** Fitting `pd.DataFrame({"x": [1.0, 2.0]})` learned `["1","2"]`;
applying `[1.0]` set `x_1 = 1`, applying `[1.0, 2.5]` set **nothing** — one
value, two answers, decided by whether its neighbours were integral.
`_pandas_col_to_str` asked `(non_null % 1 == 0).all()` and cast the *whole
column* to `Int64` only when that held, so a single fractional value switched
the renderer for every row. **Unfiled, found while reproducing:** the polars fit
path had no counterpart rule at all — `cast(pl.Utf8)` on the same data learned
`["1.0","2.0"]`, so the two engines emitted differently *named* indicator
columns (`x_1` vs `x_1.0`) and neither artifact encoded correctly on the other
engine. **Decision: one per-value rule, implemented once per engine** —
stringify, then drop a trailing `.0` from *float-dtype columns only*
(`_INTEGRAL_FLOAT_SUFFIX`, `_pandas_col_to_str`, `_polars_col_to_str_expr`).
Chosen over the alternatives after measuring the renderings side by side: a
per-column dtype rule cannot work, because the null-upcast case (`[1, 2, None]`
→ `float64`) and a genuine float column (`[1.0, 2.0]`) are the *same* dtype in
pandas but need different strings, and re-testing the values makes it
batch-dependent again. The measurement also showed the old `Int64` route was the
worse parity choice on top of being unstable — `1e20` rendered as
`"100000000000000000000"` on pandas against polars' `"1e+20"`, and `-0.0` as
`"0"` against `"-0"`, where the strip rule agrees with polars on both.
**Deliberately kept:** the null-upcast normalization the old heuristic existed
for, since an integral float still renders `"1"` — both pre-existing regression
tests pass unchanged — and the dtype gate, so a string column holding the
literal `"1.0"` keeps it. **Compatibility note:** an artifact fitted on a float
column before this change carries `".0"`-suffixed categories and no longer
matches. Blast radius is an explicit `columns=` selection only:
`detect_categorical_columns` matches `object`/`category` on pandas and
`Utf8`/`Categorical`/`Object` on polars, so it never auto-selects a float
column. Five tests added (batch composition, cross-engine category and
column-name parity, cross-engine artifact reuse, fractional values unrounded,
string `"1.0"` untouched). The wrapped-polars path was verified by a separate
probe, because `test_wrapped_frame_parity` reaches `DummyEncoder` with no
categorical column and so never enters the renderer.

**OC-164 — `get_fitted_split()` refitted the pipeline it was called on.**
`predict(x=5)` returned `50.0`; after `get_fitted_split()` on data shifted by
`+100` the identical call returned `-950.0`, with no error. `_pipeline.py:233`
ran the *live* `self.feature_engineer.fit_transform(data)`, which resets
`fitted_steps` and refits every step, while `model_estimator.model` stayed
fitted against the previous scaler. **Decision: isolate, not invalidate.** The
queue offered both; invalidating would break the *recommended* workflow, because
`optimize_thresholds()`' own docstring tells callers to get a clean holdout via
`get_fitted_split()` first — a helper documented as the preparation step for the
pipeline's next call cannot be the thing that disables it. Isolation is also
free of shared state here: `FeatureEngineer` keeps its fitted state in the
instance attribute `fitted_steps` and only ever *reads* `steps_config`, so a
throwaway `FeatureEngineer(self.preprocessing_steps, _validated=True)` shares
nothing mutable with the pipeline's own chain. **Deliberately kept:** the helper
still fits on the data handed to it (all five pre-existing tests call it on a
pipeline that was never fitted) and still raises when the chain produces no
split. One test added, pinning the prediction as exactly equal across the call
and reading provenance off the *unscaled* target — the returned X is
standardized, so it cannot show which dataset the throwaway chain fitted; that
assertion was written against X first and failed for exactly that reason.

**Filed out of this pass — OC-207 (🟠), the same way OC-150 exposed OC-169.**
Reading `get_fitted_split()`'s callers to choose between isolation and
invalidation turned up a contract contradiction the fix does not touch:
`optimize_thresholds` transforms its `X_val` internally (`_pipeline.py:325`) and
its `Args:` entry says so ("*not* yet transformed"), but its own docstring two
paragraphs earlier (`:280`), `docs/user_guide/threshold_tuning.md:32` and
`skyulf-core/README.md:205` all tell callers to hand it `get_fitted_split()`
output, which is *already* preprocessed. Executed: the documented ordering
standardizes the validation fold twice and returns `{0: 0.3627, 1: 0.6373}`,
where `predict(use_tuned_thresholds=True)` transforms raw input exactly once
(`:365`) — so the cutoffs are fitted against a probability distribution
inference never reproduces. Measured on identical rows and one fitted pipeline,
the double transform moves every probability, flips 1 of 50 predicted classes,
and *raises* fold accuracy `0.8800` → `0.9000`, which is why nothing looks wrong
from the metric. Left unfixed deliberately: it needs a contract decision (fix
the three doc sites, or accept pre-transformed input), not a code patch, and
OC-164's isolation fix is correct either way. Row and full reproduction are in
the live queue's **Next** tier.

**Verification:** `.venv/Scripts/python.exe -m pytest skyulf-core/tests -q
--no-cov -o addopts=''` → **3810 passed, 56 skipped** in 153.92s, exit 0 (the
2026-09-05 baseline was 3680 passed / 1 failed / 2 errors, all three
environmental). `ruff check .` clean repo-wide, `ruff format` clean on all four
touched files, `ty check backend skyulf-core/skyulf skyulf-core/tests
run_skyulf.py celery_worker.py` → all checks passed. **The Next tier emptied and
refilled in one session** — OC-177 and OC-164 were its last two filed rows, and
OC-207 above is now the only one left, so the master report's order still stops
at Next rather than reaching the parked OC-71 deployment-model decision.

### 2026-09-06 — OC-200/205/201/203/202/67 fixed: modeling/_tuning closed as one pass

Two of the six were one disagreement seen from both sides. Grid averaged whichever
folds survived, so a candidate erroring on one of two folds **won** with
`best_score=0.0` (5 rows, `KNeighborsRegressor(n_neighbors=3)`, unshuffled);
halving/optuna are built with `error_score=np.nan`, so an all-failed search reported
`nan` as the winning score and the caller refit a model and logged a completion where
grid raised (4 rows / 4 folds / Ridge — both measured pre-fix). The rule now shared:
**a candidate is eligible only if every fold scored, and a search left with no eligible
candidate fails with grid's existing actionable "All trials failed"** — sklearn's own
`error_score=nan` semantics. `evaluate_candidate_cv` disqualifies on `n_failed` and logs
`disqualified: 1/2 CV folds failed`; `extract_best_result` rejects a non-finite winner
through the same `_all_trials_failed(...)` builder the "No trials are completed yet"
path uses. A partly-failed candidate still loses to a healthy one instead of taking the
search down (`n_neighbors=2` wins, `3` records `-inf`).

OC-201: `build_optuna_searcher` was the only strategy converting `config.search_space`
without `clean_search_space` (`grid_random.py:27` and `halving.py:42,67,80` all call
it), so `max_depth=['none']` reached the estimator as a string — grid returned
`{'max_depth': None}`, Optuna failed every trial. Fixed at the call site rather than in
the dispatcher: each strategy already owns its normalization, and a second
`replace(config, ...)` in `engine.py` would have created one more config copy for the
halving-builder spies to disagree with. OC-203: `isinstance(x, (int, float))` is true
for `bool`, so `fit_intercept=[True,False]` became `IntDistribution(high=1, low=0)` and
every trial passed `1` where sklearn fits `True` (both confirmed directly against
sklearn). One `_is_number` predicate excludes `bool`, so Boolean lists stay
`CategoricalDistribution(choices=(True, False))` as the comment above the call already
promised, while integer lists still span a range (`max_iter=[50,100,200]` →
`IntDistribution(high=200, low=50)`) — CMA-ES behaviour unchanged.

OC-202 reproduced exactly as filed, run against the pristine pre-fix module:
`SVC(probability=False)` on 40 alternating-label rows scores **0.525** unwrapped and
raised `AttributeError: This 'SVC' has no attribute 'predict_proba'` wrapped. Both
response methods are now gated by `available_if`, whose predicate reads the fitted
`model_` when present and the constructor argument otherwise, so `hasattr` also answers
correctly on the unfitted clones a searcher makes. Neither needs label remapping: their
columns follow the fitted model's class order, which the label map preserves when it
maps `classes_` back. Wrapped roc_auc now equals the native 0.525.

OC-67: the three names sat in `INVALID_REGRESSION_METRICS` — the module knew them — but
not in `METRIC_ALIAS_MAP`, so `get_scorer` raised `'pr_auc' is not a valid scoring
value`, and the strategies failed *differently*: grid swallowed it per fold and reported
the misleading "All trials failed", while halving/optuna call `resolve_scorer` outside
any try (`engine.py:624`) and crashed the search outright. `pr_auc` now aliases to
`average_precision` and joins `BINARY_POS_LABEL_METRICS` (its `pos_label=1` default is
the same trap as `f1`); a multiclass target switches it to `pr_auc_weighted` through its
own branch in `weight_metric_for_multiclass`, because suffixing the alias would ask for
`average_precision_weighted`, which does not exist; the two names sklearn has no scorer
for are built locally behind `CUSTOM_SCORER_BUILDERS`, whose `g_score` builder refuses
up front when imblearn is missing so that failure stays a configuration error rather
than an all-fold one. Two traps fell out of verification, both documented where they are
avoided: sklearn hands a `predict_proba` scorer **only the positive column** for a
binary target (measured — the 1-D response equals `predict_proba(X)[:, 1]`), so indexing
it as a matrix raised `IndexError`; and `make_scorer` injects a `pos_label` into any
score function whose signature declares one, which is why `geometric_mean_score` raised
`pos_label=1 is not a valid label` on string labels — a two-argument `_g_score` wrapper
stops the injection, where passing an explicit `pos_label` only traded it for imblearn's
"pos_label is ignored when average != 'binary'" warning. Verified over 5 strategies ×
{binary, string, multiclass} × 3 metrics = 45 runs, all finite, grid/halving/optuna
agreeing to the digit (0.4758 / 0.4699 / 0.3431); the scorers pickle, which matters
because `n_jobs > 1` sends them to workers. Core 3790 (+21 tests), backend 1646 (+8),
ruff/format/ty clean. No open finding remains in `modeling/_tuning/`; the modeling rows
still open (OC-187/194/204/206/168) all sit outside that package.

### 2026-09-06 — OC-186 fixed, plus two CI follow-ups on the previous session's commits

`S3Catalog.exists` built its throwaway filesystem from the raw instance options while
`__init__`, `load` and `save` all went through `_prepare_s3fs_options`, so with
AWS-named credentials it authenticated differently from the methods it is supposed to
agree with. One line routes it through the same mapping, which also brings it under
`_apply_s3_endpoint`'s SSRF guard (caller-supplied `endpoint_url` dropped, only
`AWS_ENDPOINT_URL` honoured). The new test failed pre-fix with `KeyError: 'key'` —
`exists` was handing s3fs `aws_access_key_id` verbatim — and now pins
`key`/`secret`/`client_kwargs['region_name']`, the absence of the AWS-named keys, and
the `s3://bucket/id` path. OC-183–185 stay open.

**Codecov** put `data_sources/_common.py` — extracted the previous session for the
byte-identical sqlite/postgres writers — at **34.14%** patch coverage, 26 lines missing,
because only the empty-filter guard had tests. `test_data_sources_common_writers.py` now
runs both writers against an in-memory SQLite `data_sources` table: statement building,
the commit, `affected_rows`, a multi-key filter's AND semantics, and rollback-and-reraise
on a bogus column → **100%**. The two bare `raise` lines resisted the real-database
tests, and turned out to be a coverage-tracer loss across SQLAlchemy's `greenlet_spawn`
await — three scratch probes showed the identical shape reporting 100% without greenlets
— so a mock-session pair asserts the handler's contract directly
(`rollback.assert_awaited_once()` / `commit.assert_not_awaited()`), which is also the
only test able to distinguish "rolled back" from "never applied".

**SonarCloud** filed a MEDIUM "SQL Injection — possible SQL injection vector through
string-based query construction" on `preprocessing/_helpers.py:104`, the `TypeError`
message the OC-163/165/166 fix added. False positive: Bandit's B608 matches a
SELECT…FROM word pair *inside a string literal*, and "Cannot select rows from y of
type…" contains "select rows from y". Reworded to name the function instead of the
operation (`Unsupported y type for select_rows_by_position: dict`) with a comment saying
why, rather than a `# nosec`/`NOSONAR` that would hide the next real one.

### 2026-09-06 — OC-163/165/166 fixed: five improvised y-selections replaced by one positional helper, and three unfiled copies of the same bug fell out

Filed as three findings across two tiers, this was one root cause. Five call sites
each hand-rolled "adjust y to match X's rows", and **four of them silently returned
y untouched for a shape they did not recognise** — which is the bug, since a helper
that no-ops on an unrecognised target is exactly how X loses rows while y keeps them.
The fix is a single leaf helper, `select_rows_by_position(y, positions)` in
`preprocessing/_helpers.py`: one integer-position value serves *both* row-changing
operations (the argsort of a sort, the kept indices of a filter), so X and y agree
**by construction** rather than by two independently-correct-looking selections.
It handles every shape the dispatcher accepts — polars Series/DataFrame via `gather`,
pandas via `.iloc`, numpy, list — preserves the input's type (the tuple path returns
`(X, y)` verbatim through `pack_pipeline_output`), and raises `TypeError` on anything
else instead of passing it through, following `drop_and_missing/_common.py`'s OC-12
precedent. Positions are always positional, never label-based: `.loc`/`get_indexer`
on a duplicated index returns every matching row or the first occurrence, the same
defect in a different hat.

`time_series/_common.py` gains `sort_with_positions_pandas` / `_polars` and keeps
`sort_pandas` as a delegating wrapper — two JSON-driven test files pin it, and one
sort implementation means nothing can drift. The pandas positions are read off a
RangeIndex'd copy of the sort key run through pandas' own `sort_values`, so the order
is identical by construction rather than by a second sort I have to keep in sync; the
polars side uses `pl.arg_sort_by`, the expression form of `DataFrame.sort` taking the
same `nulls_last`/`maintain_order` flags, so `X.gather(order)` *is* the `X.sort(...)`
it replaces. **No new reserved helper-column name was introduced** — OC-160 is still
open, and materialising `__pos__` into a user frame would be a fresh instance of that
collision class. Proved by running the whole matrix against a frame with a column
literally named `__pos__`.

Three defects this pass exposed that were never filed, all the same silent
pass-through and all now routed through the helper: (1) **`EllipticEnvelopeApplier`
had a fourth inline copy of OC-166** — `y.filter(mask) if hasattr(y, "filter") else y`
— so the finding's "IQR, ZScore, ManualBounds" list was one node short;
(2) polars `LagFeatures(drop_na=True)` called `.filter` **directly** on y, raising
`AttributeError: 'numpy.ndarray' object has no attribute 'filter'` for the very
targets `_check_xy_engine_parity` documents as engine-neutral and accepts;
(3) the pandas outlier path did `y[mask]`, raising `TypeError: list indices must be
integers or slices, not Series` on a list y. So OC-166 was broader than filed twice
over: list targets failed as well as numpy, and the two engines failed *differently*
— a crash on pandas, a silent no-op on polars. The silent variant is the dangerous
one; the crash at least announces itself.

A test had pinned the bug as the contract.
`test_polars_tuple_xy_with_non_polars_y_passthrough` asserted `y_out is y`, with a
docstring naming "the `_filter_y_polars` fallback branch" as the behaviour to
preserve — that branch *is* OC-166. Rewritten (not deleted) as a parametrized
list/numpy case asserting y is filtered in sync, with y mirroring the `val` column so
it pins *which* rows survived, not merely how many: a y filtered through the wrong
mask would still have the right length.

Deliberately left open: **OC-173** is a different defect in the same file
(`_elliptic_filter_pandas` reselecting valid values by duplicated index labels), and
I verified no interaction — `X_pd[mask]` and `X_pd.iloc[keep]` are identical on
duplicate labels. `winsorize.py` was checked and never filters rows. **OC-160** is
untouched, and this fix adds no reserved names for it to collide with.

Verification: new `tests/integration/test_xy_row_alignment.py` (127 tests)
parametrizes the full engine × y-shape product — 2 engines × list/numpy/Series/
DataFrame = 8 ids — across `LagFeatures`, `RollingAggregate`, `IQR`, `ZScore`,
`ManualBounds` and `EllipticEnvelope`, plus sorting composed with `drop_na`, and
helper-level tests pinning the sort positions against `sort_values`/`DataFrame.sort`
on ties, nulls, dates and duplicated indexes. **Genuineness proved:** reverting only
the four node files to HEAD (helpers left present-but-unused so imports still
resolve, making every failure behavioural rather than a collection error) gives
**45 failed, 82 passed** — and the 82 passes are precisely the controls the findings
themselves named: polars Series y already filtered correctly, X-only sorting
unchanged. Restored with `md5sum -c`, all four OK. One incidental change: the pandas
`drop_na` positions moved off `~df.isna().any(axis=1).to_numpy()` because ty types
`DataFrame.any(axis=1)` as `Series | bool`; the replacement
`df.notna().to_numpy().all(axis=1)` is the same predicate numpy-side and matches
`df.dropna()` on 11 edge cases (zero-column, all-null column, object dtype, nullable
`Int64`, duplicated and non-monotonic indexes). Core 3672 → **3783**, backend
**1637** unchanged, ruff check/format and ty clean.

### 2026-09-06 — OC-159/131/28/77 fixed, OC-55 verified stale: three fail-open paths and a coverage floor 51 points below reality

Empty `filter_dict` compiled to a WHERE-less DELETE/UPDATE; four sites (sqlite +
postgres × delete + update) now raise before the session opens. `drift.py` had no
logger at all, so PSI/KL's finite `0.0` fallback was invisible — all three fail-open
paths log now, including the one the finding missed: an uncastable column is dropped
from the report whole, and its absence reads as "no drift". OC-28's silent path was
the `valid_cols` filter, not the `except` (which has logged since `2d605197`); both
engines now share `_fitted_columns_present`, and stayed fail-open by decision — 25
such sites under `preprocessing/`, and `apply` runs at inference where a raise turns
a degraded prediction into a 500 — so the decision is recorded on the docstring.
Coverage floor 45 → 90 (CI measures 96%), `--maxfail=1` gone from both workflows
though the finding filed only core. OC-55 does not reproduce. Core 3672, backend
1637, ruff/ty clean. Standing convention from here: status cells in the tables are
one sentence, and the detail lives in this section.

### 2026-09-06 — OC-81 closed: the license decision arrived, and "1 line" turned out to be five files

Owner decided: `skyulf-core` is Apache-2.0, backend + frontend stay AGPLv3 — so
`COMMERCIAL-LICENSE.md` was the one correct file of the four that disagreed.
Declared as a **static** `license` in `skyulf-core/pyproject.toml`: adding it to
`dynamic` (how every other field there defers to `setup.py`) is accepted with no
warning but emits the deprecated free-text `License:` instead of PEP 639's
`License-Expression:`, the only field PyPI indexes for license filtering. No
`License ::` trove classifier — setuptools >= 77 deprecates those once an SPDX
expression is present. Verified by building a real wheel: `Metadata-Version: 2.4`,
`License-Expression: Apache-2.0`, Apache text bundled, zero warnings, and OC-78's
`py.typed` confirmed inside the artifact PyPI receives. Three files then reconciled,
which is where "1 line" went: `COPYRIGHT.md` stated the split **backwards** and cited
a frontend license path that does not exist; `skyulf-core/README.md` declared AGPLv3+
while linking the Apache file beside it — and that README *is* the PyPI
`long_description`, so the wrong license sat on the package's front page; its shields
badge used `github/license`, which resolves the **repository root** `LICENSE`, so it
rendered AGPLv3 on an Apache package. Version synced to 0.8.15 root + frontend.

### 2026-09-06 — OC-05/22/78/79/112/132/141 fixed: the "1 line" tier closed, and two of the seven were not what they claimed

The whole `1 line`/`mechanical` tier: seven closed, OC-81 blocked on the license
decision, OC-10 and OC-144 re-scoped and left open. Two of the seven were mis-filed
and worse than their ⚪/🟡. **OC-05** was not cosmetic: the pandas path wrote float
results into `int64` columns via `.loc`, which pandas 2.3.2 warns about per column
and says will become an error — and the bare `except` around it means the promoted
error would silently skip the transform and return the original data, i.e. OC-28's
failure mode arriving through OC-05. The regression test **records** warnings rather
than promoting them, because a promoted one is swallowed by that same `except` and
the test would pass while the transform no-ops. **OC-22** had a test pinning the bug:
`test_infer_output_schema.py:207` asserted passthrough for `("binary", "regression")`,
but `"regression"` is not a sklearn `target_type` and `_build_target_encoder`
forwards it verbatim, so it locked in a prediction for a config that can never fit
while the legitimate `"continuous"` case fell through to `None`. The rest: `py.typed`
created and proven via setuptools' own `build_py`; `joblib>=1.3.0` declared in core
*and* root, since the backend had the identical undeclared-import gap; OC-112
comment-only; OC-132 deleted a branch reading a key nothing writes; OC-141 deleted an
unconsumed `node_meta` field. Left open: OC-10 is five dead overrides whose comments
OC-03's sweep will need, and OC-144 is 10 assertions plus two unreachable fallbacks.
Core 3670 / backend 1630, both exact baselines. Trap: piping a gate through `tail`
reports *`tail`'s* exit code — read the output, not the status.

### 2026-09-06 — OC-183–186 filed: four backend defects the OC-09 docstring pass surfaced, deferred by decision

Four backend findings the OC-09 docstring pass surfaced, filed in full in the tables
above and **deferred by decision**: three need a product call before a fix is
meaningful (OC-183 which config name is canonical, OC-185 which authz model), and
OC-184 is a behaviour change — a security-headers middleware in
`main.py::_add_middleware`, where order is load-bearing and CORS must stay outermost
— not the config fix its effort estimate implies. OC-183 is OC-130's root cause
repeating: a bare `os.getenv` for a name that is not a `Settings` field, which
pydantic-settings never exports from the dotenv.

### 2026-09-06 — OC-09 closed: 904 hand-written docstrings, and a ruff privacy gate that hid 82 of them from every check we run

904 hand-written docstrings. Measured **3,350** `D` sites, not the audit's ~500;
2,528 sat in tests/examples/benchmarks, now waived per-file, leaving the enforced
paths at 0. **The discovery:** `D1xx` is privacy-gated on the *whole dotted module
path* — a leading underscore on the module **or on any enclosing package** exempts
everything beneath it, so 82 sites across 112 of 327 in-scope modules
(`ml_pipeline/_internal`, `_execution`, `_services`, `modeling/_tuning`) were
invisible to every gate we run, including the earlier batches that looked complete.
`D2xx`/`D4xx` are *not* gated, which is why those batches seemed to finish the files.
Recovered by copying each private file to a public name outside its package and
linting the copy — byte-identical, so line numbers carry straight back. All **210**
modified files AST-proven against `HEAD`: 199 docstring-only, 11 real changes. Six
genuine defects surfaced and were fixed, the loudest being `backend.eda.tasks` never
imported by `celery_worker.py`. **`ARG` declined by decision:** 121 in-scope sites,
~90 of them Calculator/Applier or framework contract signatures that cannot be
renamed without breaking keyword callers — ~90 permanent waivers to surface ~4
defects, and those were fixed directly instead. Full reasoning now lives in
`AGENTS.md`. Core 3668 / backend 1630, both exact baselines.

### 2026-09-06 — OC-177–182 filed: 10-file cleaning/encoding continuation

Six reproduced findings from the 10-file cleaning/encoding continuation; the
source-read ledger is now 79/188 files. All six are cross-engine or batch-composition
divergences, stated in the tables above, and the numbers that make them concrete:
OC-177 a float category encoding as *known* alone but all-zero when accompanied by
`2.5`; OC-178 one shared fitted artifact hashing the same missing value to
`[928,171]` / `[928,915]` / `[928,870]` across polars, pandas object and pandas
nullable string; OC-179 a single-category indicator retained on polars but dropped on
pandas, changing feature width; OC-180 `pd.NA` reaching `re.Pattern.sub`; OC-181
`{"banana": true}` turning `[true,false]` into `[true,true]`; OC-182 `StringDtype`
columns left silently unencoded.

### 2026-09-05 — OC-09 half closed: `F401` enabled repo-wide, after two orphans inside my own security fix proved the gap was not cosmetic

`F401` enabled repo-wide; the rest of OC-09 stayed open at the time. **The finding
earned its place by catching me, not by argument:** the OC-152 raw-SQL deletion
orphaned `from typing import cast`, and `KNOWN_ENVIRONMENTS` in `config/factory.py`
was referenced only inside a comment — both passed `ruff check`, `ruff format
--check`, `ty check` **and** 1626 tests; only an external pyflakes pass caught them
(`ty` does not report unused imports at all). All **27** sites were classified
*before* automating, because pre-commit runs ruff with `--fix` and enabling `F401`
makes auto-stripping permanent: 22 dead, **4 intentional availability probes** waived
per-site (each exists purely so the `except ImportError` branch runs), and 1
redundant `import skyulf` dropped only after confirming `skyulf/__init__.py:22`
imports `.registry`, so the registration side effect still runs. Trap: `# noqa` must
sit on the **import statement** line, where a `ty: ignore` may go on the line above —
different placement rules. Core 3669 / backend 1626.

### 2026-09-05 — OC-130 + OC-150 + OC-152 fixed: the fail-open security batch; OC-169 filed from what OC-150 exposed

The fail-open security batch, backend-only, each reproduced against `HEAD` before any
edit; all three broader than filed. **OC-130**: two independent exact-match readers
of `FASTAPI_ENV` — `factory.py`'s `.get(env, DevelopmentSettings)` and `base.py`'s
own `os.getenv` production guard — so `prod` yielded dev settings, `DEBUG=True`,
`CORS_ORIGINS=['*']`. The unfiled half is worse: `FASTAPI_ENV` is not a `Settings`
field and pydantic-settings never exports the dotenv into `os.environ`, so `.env` was
dead for exactly this one setting. That matters beyond the profile choice, because
`main.py` hardcodes `allow_credentials=True` and Starlette with `allow_origins=["*"]`
**reflects the caller's origin**, so any origin could make credentialed requests.
**Decision: fail closed (raise), not fail loud (warn)** — a warning is written by the
same process that is about to open the port. `resolve_environment()` now normalizes
and raises naming every accepted value, *including empty* (what an unset CI variable
renders to, and precisely the silent fallback being removed), reading through a
`_EnvironmentSelector(BaseSettings)` that shares `Settings`' `SettingsConfigDict` —
which is what makes the dotenv channel live. `_ENV_SETTINGS_MAP` was hoisted to
module scope for the test asserting it matches `KNOWN_ENVIRONMENTS`; hoisted because
constructing a settings *subclass* calls `setup_logging()`, which strips every
handler off the root logger for the rest of the session. **OC-150**: filed as
case-sensitivity; execution showed `_sanitize_error` was a **no-op** on all three
real shapes — an S3 403 XML body, a SigV4 presigned URL, and an s3fs options-dict
repr exposing the **secret access key** itself, which the audit missed — while
*destroying* benign messages (`key=reports/2026/q3.csv not found` → `redacted
sensitive S3 error`). One design error both ways: match the setting *name*, then
discard the whole *message*. Replaced with `redact_credentials()` beside the existing
`sanitize_for_log`, both byte-identical copies deleted: a key-ID shape regex, a
name→separator→value scrub that re-emits name and separator so the diagnostic
survives, and an XML-tag scrub. Idempotent by construction, which matters because a
message may pass through both a call site and a wrapper. Presigned URLs are **bearer
credentials**, so the constructor's `logger.info` redacts its `path` too. **OC-169
filed, not fixed**: `ErrorHandlerMiddleware` logs `{exc}`, `traceback.format_exc()`
**and** `exc_info=True` unredacted, so any uncaught exception leaks regardless of the
call site — app-wide, not an S3 defect. **OC-152**: filed as two dead raw-SQL
executors; SQLite carried a byte-identical pair, so **four** went, zero callers
grepped first. Deleted rather than parameterised — a helper whose contract is "accept
an unconstrained query string" cannot be made safe by editing its body. **46 new
tests**, proven genuine by reverting only the seven source files. That pass found a
**fixture bug**: the OC-130 guard tests had been passing vacuously wherever
`SECRET_KEY` happened to be exported (as this shell does), so the production guard
correctly stood down; `isolated_env` now scrubs every `Settings.model_fields` name.
Backend 1626 (was 1580). **Residuals**: raised S3 messages stay unredacted by design
(they carry the caller's own input back), and `redact_credentials` covers AWS shapes
only — widening it risks eating benign text, the exact failure this fix removed.

### 2026-09-05 — OC-153 + OC-157 fixed: merge no longer stacks rows in silence, and column order no longer depends on the tiebreak strategy — plus the 0.8.13 bump

Two findings, one file (`_execution/engine/_merge.py`). **OC-157**: `first_wins` was
implemented by reversing the iteration, and `result_cols` is a plain dict whose
insertion order *becomes* the merged frame's column order — so `A(a,b) + B(c,d)` gave
`['a','b','c','d']` under `last_wins` but `['c','d','a','b']` under `first_wins`. Did
not take the audit's two-pass reorder (it can drift from the ownership pass); deleted
the reversed iteration instead, so both strategies walk inputs in their own order and
the strategy only decides whether a later input may overwrite a claimed column. Order
is first-appearance **by construction**, since re-assigning a dict key keeps its
position. **OC-153**: `_merge_frames_rowwise` warned only when column sets
*differed*, so the likeliest accident — identical sets, one branch filtered — stacked
`(5,2) + (4,2)` into `(9,2)` with 4 duplicated rows and `merge_warnings == []`.
**Decision: warn, not raise**, departing from the OC-154/155 precedent set the same
day: row stacking is a *supported* feature (`row_concat_drop`'s own UI copy says so),
and a merge runs where a human is looking at the canvas, unlike serving time where a
wrong number reaches an API caller with a 200. The bug was never the row-wise path;
it was its silence. **Frontend work was required**: `MergeWarningsBanner.tsx` ends in
a fall-through default rendering "No column overlap — all columns from all branches
are kept" plus a "Chain instead" rewire button, so a new kind with no branch would
have shown text actively wrong for a row stack and offered a rewire that cannot fix
one. Deliberately did *not* compute `merged.duplicated().sum()` for the message — an
O(rows × width) hash on every merge, when the row counts alone let the user infer it.
7 tests, proven genuine by reverting only `_merge.py`: **5 failed, 2 passed**, and the
2 are precisely the controls — the wanted signature. Nothing exercised the
row-count-mismatch path before this (zero `rowwise` hits outside the file), which is
how a whole advisory condition could be wrong and unnoticed. **Also 0.8.13 bumped**,
and the versioning instructions turned out to omit `skyulf-core/setup.py` — the file
`release.yml` greps to decide whether to publish — with no CI gate comparing it to
root, so following the documented steps exactly would silently skip the core release.
That file is gitignored, so the correction is machine-local and the gap stays open.
Backend 1580 (was 1573), frontend 873 + build.

### 2026-09-05 — OC-25 + OC-143 fixed: RFE now selects the number of features the user asked for — and the two findings turn out to be one

**The same bug filed twice**, by two audit passes pointing at the same line, so the
report's 116-finding inventory double-counts by one (distinct total 115).
`_build_model_selector`'s `rfe` branch read `config.get("n_features_to_select")` — a
key **nothing in the repository writes** — so it was always `None` and sklearn's
`RFE` fell back to keeping *half* the candidates. The UI writes `k`
(`FeatureSelectionNode.tsx:34,406,571`). It hid because `step`, RFE's other field,
*does* match, so the panel looks fully wired; and the fixture encoded the bug as the
contract — its `rfe` case passed `n_features_to_select: 2`, leaving the `k` path the
UI actually sends with **zero coverage**. Fixed in **core, not the converter**: the
converter already passes `node.data` through, so repairing the reader fixes canvas,
API callers and notebooks in one move. The audit's `config.get(a, config.get(b))`
one-liner was rejected — a `.get` default only fires when the key is *absent*, so a
cleared field serialising to `null` would still yield `None`. Reproduced on 6
candidates: `rfe k=2` → **3** pre-fix. Gotcha: fixture test IDs come from parameter
*values*, not the JSON case name, so `-k <case name>` silently matches nothing and
looks like a passing suite. Residual: `ModelBasedSelection`'s `@node_meta` declares
no `method`/`step`/`k`, so the whole RFE path is undeclared in the metadata R1 step 1
wants as the source of truth — that belongs to the contract work. Core 3669 /
backend 1573.

### 2026-09-05 — OC-154/OC-155 fixed: both serving paths now fail closed instead of returning a wrong number as a normal result

One hazard class, two sites: the deployment predict path degraded silently wherever
it could not guarantee feature alignment, so a wrong number of arbitrary magnitude
reached the API caller with a 200. **OC-154**: F-02's reindex guarded itself with
`if not missing:` — it disabled the protection precisely when alignment could not be
confirmed, the exact state F-02 exists to prevent. A bundle trained on `a,b,c` whose
feature engineer emits `a,b,d` skipped the reindex and handed a positional model
`[1,2,99]` as if it were `[a,b,c]` under weights `1/10/100`, returning **`9921.0`**
with no error and no warning. Now raises naming *both* column sets plus the specific
missing ones, so "your request omitted a column" stays distinguishable from "this
bundle's transform no longer matches its training" — different owners, different
remedies. **OC-155**: the legacy path imputed absent features with the literal `0`,
an extreme out-of-distribution input for anything not centred on zero — trained on
`income,age,price`, request omits `price` → **`50030.0`** behind a server-side
warning. **Worse than filed**: the loop was `df[c] = 0`, mutating the *caller's*
frame, so the fabricated column outlived the request. Fix reuses the
`_validate_required_columns` helper the bundled path already called 15 lines earlier,
removing the zero-fill and the mutation in one move. **Why raise rather than warn**: a
warning reproduces OC-155's own complaint, while `api.py:129` already maps
`ValueError` → 400 with `detail=str(e)` and `client.ts` already normalizes `detail`
into `error.message` — the column names reach the browser with **zero frontend
work**. Residual: the bare-numpy legacy case is unfixable by construction — no
`feature_names_in_` means no record of the expected columns, so the guard stays
`hasattr`.

### 2026-09-05 — OC-39/40/41/42/43/44/45/46 fixed: the profiling cluster closes, four findings from one root cause

The profiling cluster closes: eight findings, two root causes, one design decision.
**Root cause 1 — polars keeps NaN distinct from null** (OC-39/40/43): `fill_null` is
a no-op on NaN, `drop_nulls()` keeps NaN rows, aggregations *propagate* it, and
`nan == nan` is False. Every repro had to be built with `pl.DataFrame` directly —
`pl.from_pandas` converts NaN→null and therefore **masks all three bugs**, which is
why the existing suites never caught them. Fixed once at the boundary (`_nan_to_null`
in `EDAAnalyzer.__init__`) instead of at six call sites. OC-39 was worse than filed:
a silently *wrong* median (`3.0` where pandas gives `2.0` on `[1,2,nan,4]`) plus
`ComputeError: breaks cannot be NaN` in the histogram builder, whose
`min_val == max_val` guard cannot see that NaN never equals itself. OC-40's no-oped
`fill_null(strategy="mean")` let values fall through to `nan_to_num(nan=0.0)`, so
PCA/clustering were fitted with `0.0` exactly where `SimpleImputer` would put the
mean — the most distorting value available for a mean-centered feature. OC-43:
`drop_nulls()` + `DataFrame.corr()` is *listwise*, so one surviving null NaN'd the
whole matrix, the broad `except` swallowed it, and the profile lost its correlation
section; now pairwise via `pl.corr`. The audit's supporting claim was **wrong** here
— it asserted pandas returns all-`1.0` for the sparse frame, but pandas returns
all-`NaN` too. **Root cause 2 — polars' estimator defaults are not pandas'**
(OC-41/42): `Expr.quantile()` defaults to `interpolation="nearest"`, `skew()`/
`kurtosis()` to `bias=True` non-Fisher. OC-42 was user-visible: `[1,2,3,4,10]`
reports `1.30` biased against `1.70` unbiased, so the "consider a transform" hint
stayed silent on a column clearing the 1.5 threshold. **OC-44 was a design
decision**: the verdict has always used the std-normalized Wasserstein distance while
`value` carried the raw one, so a large-scale column that barely moved reported
`value=50.0, threshold=0.1, has_drift=False`. Rather than teach every consumer a
metric-specific exception, the invariant now lives in the schema — `value` is the
number `threshold` applies to, raw moved to `raw_value`. That was the scale every
consumer already assumed (`test_drift.py:25/46`, the docs' "0.1 (normalized)", the
published notebook showing `784.7832 (Thresh: 0.1) [PASS]`), and it fixed the alert
modal, CSV export, column summary and persisted summary with **zero edits** to any of
them. **OC-45**: `drifted_columns_count` came from per-column flags alone, so a
vanished feature left it `0` while `_classify_drift_severity` called the same report
`"critical"`. The "never rendered" half was already handled by `SchemaDriftPanel`, so
a banner first added to `DriftTable` was removed as a duplicate. **OC-46**:
`NumericStats`' optional floats now map non-finite → `None` before validation, because
`model_dump(mode="json")` retains a Python `nan` and stdlib `json.dumps` emits a bare
`NaN` token the browser's `JSON.parse` rejects. **22 tests**, proven genuine by
reverting only the fixed source: 10 of 13 profiling tests failed with the exact
pathological values, 2 passed *by design* (guards pinning behaviour that must not
change). One was strengthened after passing pre-fix — the sweep needed a large-scale
column, since on unit-scale data raw and normalized coincide and the invariant holds
by accident. Also fixed a doc bug in passing: the custom-thresholds example passed
`"ks": 0.01` but the calculator reads `ks_statistic`, and unknown keys are merged and
never read, so the documented override silently did nothing. Core 3661 / backend 1547
/ frontend 872 + build (which caught a `noUncheckedIndexedAccess` error vitest does
not typecheck). **Residuals**: no "unknown" cell in `CorrelationMatrix.values`, so
pairs below `MIN_PAIRWISE_OVERLAP = 3` report `0.0`; core's skew threshold is 1.5
while `_advisor.py:195` applies 1.0 (product decision); persisted wasserstein
summaries are not scale-comparable across this change, left un-migrated because the
old values were the misleading ones.

### 2026-09-05 — OC-146 fixed (last 🔴 closed): binary `pr_auc` now scores the class the model treats as positive; OC-37 closed by the same change

Last 🔴 closed, and OC-37 with it — one argument, two failure modes.
`_add_roc_pr_auc_metrics`'s binary branch called `average_precision_score` with no
`pos_label`; that function defaults to `pos_label=1`, unlike `roc_auc_score`, which
infers the positive class from the sorted uniques — which is why `roc_auc` was correct
in all nine of the audit's label encodings and `pr_auc` was not. With labels
`{1,2}`/`{1,5}` the literal `1` names the *negative* class while `proba[:, 1]` is
`P(classes_[1])`, so PR-AUC was computed for the inverted problem: **0.3123 reported
vs 0.9718 true** on 400 rows of learnable signal, no exception, no warning. Every
other non-`{0,1}` set raised `pos_label=1 is not a valid label` and `_try_add_metric`
dropped the key — that is OC-37. The report also contradicted itself on screen:
`classification.py:99` builds the PR *curve* with `pos_label=classes[1]`, so one
`CurveData` carried a curve drawn for class 2 beside an AUC computed for class 1. Fix
resolves `pos_label` from `model.classes_` — the pattern the function 30 lines above
and the curve builder already used. Deliberately *not* `pos_label=None`: sklearn 1.8
rejects it, and `_try_add_metric` would swallow that and drop the metric. Verified
bit-identical to ground truth across `{0,1}`, `{1,2}`, `{1,5}`, `{-1,1}`,
`{"no","yes"}`; 4 of the 7 new tests confirmed failing pre-fix. Core 3643.

### 2026-09-05 — OC-36 fixed: threshold tuning no longer returns a pathological cutoff; OC-147 closed alongside (search and apply now share one decision rule)

OC-36 and OC-147 closed together — both in the `_grid_search_binary` ↔
`apply_thresholds` pair, both from letting an arbitrary tie-break decide. **OC-36**:
the search kept a candidate only on a strict `score > best_score`, so a tie left the
*first-scanned* one standing. On a split holding one class every candidate scores
F1 0.0, so tuning returned the first grid point — re-verified as `{0: 0.990,
1: 0.0098}`, predicting **49/50 rows positive at F1 0.0** and persisting it as a
tuned threshold. Because F1 is piecewise constant in the cutoff, tied plateaus are
routine on healthy splits too. Fixed in two layers: the search warns and returns the
neutral `{0.5, 0.5}` when `len(np.unique(y_true)) < 2` and breaks ties toward `0.5`;
`_tune_decision_thresholds` gains a sixth gate, so a degenerate split leaves
`decision_thresholds=None` and `predict()` keeps the model's own rule instead of the
UI reporting a threshold nothing was tuned against. **OC-147**: `apply_thresholds`
documents `>=` and special-cased a bare float and a *one*-entry dict, but the search
returns a *two*-entry dict — which fell through to the multiclass scaled argmax,
where `np.argmax` breaks exact ties toward the first column, silently turning `>=`
into `>`. Reachable because the grid includes `0.5` and trees routinely emit exactly
`0.5`. Now compares `scaled[:, 1] >= scaled[:, 0]` directly; the audit's "return a
one-entry dict" was rejected because backend `_validate_save_payload` requires keys
for *every* class and a user can save a non-complementary pair, which the naive fix
would silently ignore — a fresh OC-13-class "UI setting ignored" bug. Measured 0
differing rows out of 4000 across four threshold pairs; only the exact tie flips. 4
of 6 new tests confirmed failing pre-fix. Core 3643 / backend 1546.

### 2026-09-04 — OC-35 fixed: multiclass splits missing a class no longer emit binary-only metrics or null curve points

Three places decided binary-vs-multiclass from the labels present in `y_true` instead
of the model's trained label set, so a 3-class model evaluated on a two-class split
was misclassified as binary: it gained unweighted `precision`/`recall`/`f1` keys that
do not belong to it; `log_loss` was called with no `labels=`, so sklearn raised
"2 vs 3" and the metric was silently dropped; and the per-class curve loop ran on the
all-zero one-vs-rest target of the absent class, returning NaN points that serialize
as `null` coordinates. Fix: the binary gate resolves `model.classes_` (falling back
to the unique-label count only when the model exposes none, preserving the
`model=None` test path), `log_loss` gets `labels=`, and the loop skips any class
whose binarized target has fewer than 2 unique values. 3 tests; 148 passed across the
7 evaluation files.

### 2026-09-04 — OC-69 fixed: engine and schema predictor no longer trust `config.nodes` list order

`predict_schemas` and `_run_node_loop` both iterated `for node in config.nodes:`
assuming the list was already topologically sorted, but `validate_no_cycles()` only
detects cycles — it never verifies or restores order. The frontend's
`pipelineConverter.ts` BFS enqueues a merge node when *any* parent is dequeued (not
all), so the UI can emit an acyclic-but-misordered list (a diamond merge fed by
unequal-depth branches), producing a cryptic "Artifact not found" engine failure and
silent `None` schema degradation. Fix: a public `topological_order(nodes)` in
`graph_utils.py` reusing the existing private Kahn helpers, called at both consumer
sites. 5 tests; 49 passed across the related suites.

### 2026-09-04 — OC-17 fixed: SimpleImputer polars mean/median no longer crashes on all-null columns

Fitting `SimpleImputer` with `mean`/`median` on an entirely-null column stores
`fill_values[col] = None` (polars returns `None` for both), and `_apply_polars` then
called `pl.col(col).fill_null(None)`, which raises `ValueError: must specify either a
fill value or strategy`. The pandas path already guarded with `if val is None:
continue`, so the engines diverged — pandas left the column all-null, polars crashed.
Fix mirrors pandas: pass the column through unchanged. The restore branch was already
parity-correct, since `pl.lit(None)` yields an all-Null column. 3 tests including
engine parity; `pl.from_pandas` infers a pandas object column as `String`, so the
fixture needs an explicit `Float64` all-null column.

### 2026-09-04 — OC-16 fixed: KNN/Iterative imputers no longer crash on all-missing fitted columns

sklearn's `KNNImputer`/`IterativeImputer` silently drop all-missing columns from
`transform()` output when the column was all-missing **at fit time**, desyncing the
artifact's `columns` list from the imputer's width and crashing
`_sklearn_transform_subset` (IndexError on polars, ValueError on pandas). Fix: a
`drop_all_missing_columns` helper called from both `fit`s, so the artifact stays in
lockstep, the dropped columns are named in a warning, and `fit` returns `{}` when
every configured column is all-missing (appliers already pass empty artifacts through
as a no-op). 5 tests parametrized over both imputers × both engines.

### 2026-09-04 — OC-66 fixed: `CalibratedClassifierCV`'s base estimator now survives tuning

The tuning engine builds the meta-estimator from `default_params` (`engine.py:496`,
`refit.py:44`, `grid_random.py:139`), and
`CalibratedClassifierCalculator.default_params` hardcoded
`estimator=LogisticRegression` — so the user's `base_estimator` selection, read only
by `fit`, was silently discarded whenever the node was tuned. Fix routes it through
the structural-tuning hook `_BaseEnsembleCalculator` already uses: declare
`STRUCTURAL_TUNING_KEYS`, capture the selection in `prepare_tuning_params` (flat or
nested config shape), resolve it in `default_params` via the `BASE_ESTIMATORS`
factory. No backend change — `_node_runners.py` already calls `prepare_tuning_params`
and excludes structural keys from the search space. An integration test asserts the
tuned model is a `RandomForestClassifier` *inside* `CalibratedClassifierCV`; 173
passed across the classification + tuning suites.

### 2026-09-04 — OC-61 fixed: `BinningNode`'s "Precision (Decimals)" now reaches the backend

The backend reads `config.get("precision", 3)` and the canvas rendered a "Precision
(Decimals)" input, but `pipelineConverter.ts`'s `BinningNode` branch lists its params
explicitly and omitted `precision`, so the value was dropped before reaching the
backend. One line in the converter plus 2 vitest cases; 42/42 pass.

### 2026-09-04 — OC-53 fixed: `select_from_model`'s `max_features` now reachable from the canvas

`_build_model_selector` reads `config.get("max_features")` and `@node_meta` declares
it, but the UI's `select_from_model` branch rendered only a `threshold` field, so the
cap was Python-only and unreachable from the canvas. Fix adds the field and an
optional input mirroring the `k` pattern (empty = no cap, matching the backend's
`None` default). **No converter change**: the `feature_selection` branch already
passes `node.data` through unchanged. 2 vitest cases; 40/40 pass.

### 2026-09-04 — OC-20 fixed: Value Replacement UI help text now matches empty-columns behavior

The UI promised "If empty, applies to all compatible columns," but the backend no-ops
on an empty `columns` list — the intended convention (`user_picked_no_columns` in
`skyulf/utils.py`: every box unchecked means "do nothing for this node"). Fixed the
text to match the behaviour rather than the backend to match the text, since
apply-to-all would contradict that convention. Added a contract-locking test
(parametrized pandas/polars) so the no-op cannot silently become an apply-to-all
later.

### 2026-09-04 — OC-19 fixed: Alias Replacement `punctuation` mode now strips punctuation

The UI offers a `punctuation` mode, but `cleaning/alias.py`'s appliers had branches
only for the alias-mapping modes — `punctuation` fell through to the mapping path,
whose resolved mapping is `{}` for this mode, so it was a silent no-op. Fix adds a
dedicated branch stripping `string.punctuation` only: case and spaces are preserved,
matching the UI wording, unlike the mapping modes which fully normalise. Polars uses
`str.replace_all` with an escaped class (nulls pass through); pandas uses
`str.translate` with the original NaN restored after `astype(str)`. 3 JSON cases plus
a NaN-passthrough parity test; 45/45 pass.

### 2026-09-03 — OC-15 fixed: MinMax/Robust scaler range controls honored on canvas

The UI stores scaler ranges as scalars (`feature_range_min`/`_max`,
`quantile_range_min`/`_max`) but the converter's `scale_numeric_features` branch did
`params = config`, forwarding them verbatim while the backend reads tuple keys
(`feature_range`, `quantile_range`) — so every canvas run silently used the defaults
0/1 and 25/75. Fix assembles the tuples in the converter; the scalars are left in the
payload, harmless because the backend ignores unknown keys. No backend or doc change
— both were already correct. 4 vitest cases; 865/865 pass.

### 2026-09-03 — OC-14 fixed: Iterative Imputer canvas estimator choices honored

The UI emits lowercase aliases (`bayesian_ridge`, `decision_tree`, `extra_trees`,
`knn`) and the converter forwards `estimator` verbatim, but
`_build_iterative_estimator` matched only the documented spellings (`DecisionTree`,
`ExtraTrees`, `KNeighbors`) — so every canvas run silently fell back to
`BayesianRidge` regardless of the user's choice. Fixed by normalizing in the backend,
the single owner of the mapping: lowercase and strip non-alphanumerics before
dispatch, so both spellings resolve to the same regressor and unknown names still
fall back. No frontend change. 4 JSON-driven alias cases; 74/74 pass.

### 2026-09-03 — OC-13 fixed: Drop-Rows percentage threshold now reaches the backend

The UI stores `missing_threshold` as a 0–100 **percentage** slider (default 50), but
the converter sent it verbatim while the backend only read `subset`/`how`/`threshold`
(an absolute non-missing count) — so every canvas run silently ran as `how="any"`. A
frontend-only fix is impossible: percentage→absolute needs the column count, unknown
at conversion time. Fix adds a percentage mode to `drop_rows.py` mirroring
`DropMissingColumns` — keep rows with `non_na >= (1 - X/100) * n_cols`, i.e. drop rows
missing **more than** X%, exactly the UI wording, with a row at exactly X% kept — in
both engine paths, exposed in `@node_meta` and `fit()`, carried on
`DropMissingRowsArtifact`, and mapped by the converter (checkbox/null/≤0 →
`how: "any"`). No leakage change: the node is `learns_from_data=False` and the
threshold is a fixed user setting. 7 tests including the boundary and X/y tuple sync;
29/29 pass.

### 2026-09-03 — OC-62 fixed: object-dtype arrays digested by value, not by pointer

`_feed_canonical` in `pipeline/seal.py` digested `np.ndarray` via `arr.tobytes()`; for
`dtype=object` arrays that serialises raw `PyObject*` pointers, which are
allocator/ASLR dependent — so `fingerprint()` of any artifact holding one
(OneHotEncoder/LabelEncoder/Ordinal/TargetEncoder `categories_`) was noise that
changed across processes, defeating the point of the F-15 semantic seal. Fix detects
`dtype == object` and digests the shape plus each element recursively, so the digest
reflects values. 3 tests including non-interned strings; confirmed failing pre-fix
(`b'\xca' != b'\xf5'`).

### 2026-09-03 — OC-58 fixed: polars numeric→bool cast mirrors pandas 0/1 semantics

`_build_polars_cast_exprs` special-cased only string/categorical→bool; numeric→bool
fell through to the generic `cast(pl.Boolean)`, which is C-style truthiness (`x != 0`)
and never raises — so `2.0` silently became `True` on polars while pandas
`astype("boolean")` produced `<NA>` (and raised in strict mode). Fix builds
`when(col == 0).then(False).when(col == 1).then(True).otherwise(None)`, so only exact
0/1 map and everything else, including non-integer floats, becomes null — tracked and
validated in strict mode by the renamed `_validate_polars_bool_casts`. 4 tests
including engine parity; 85/85 pass.

### 2026-09-03 — OC-12 fixed: positional keep-mask for pandas X/y desync

The pandas paths of `DropMissingRows` and `Deduplicate` selected `y` by label
(`y.loc[X_clean.index]`); with duplicate index labels `.loc` returns *all* matching
rows, so `y` came back longer than `X` with wrong labels — silent X/y desync. Fix
mirrors the already-correct polars paths: a positional keep mask, `np.flatnonzero`,
`X.iloc[...]`, and `y` filtered positionally through a new shared helper.
`X.index.get_indexer(X_clean.index)` was rejected as a recovery path — it returns the
*first* occurrence for duplicate labels, the same bug in a different hat. 2
duplicate-index regression tests; 62/62 pass.

### 2026-09-02 — OC-75 fixed: stale nested `uv.lock` removed, benchmark guarded

The repo is a **uv workspace** (root `pyproject.toml` declares
`members = ["skyulf-core"]`), so the root `uv.lock` is the single source of truth and
`skyulf-core/uv.lock` was a pre-workspace leftover — `uv lock` from inside
`skyulf-core/` rewrites the *root* lockfile, never the nested one, so the audit's
"regenerate it" instruction is not a normal uv operation here. `git rm`'d instead (CI
never reads it — every workflow installs via `requirements-ci.txt`; only
dependency-review/labeler glob `**/uv.lock`). Also applied the secondary fix:
`bench_roundtrip_removal.py` now try/excepts per bench and prints `SKIP (Type: msg)`
instead of crashing the table. `uv lock --check` exit 0, 47/47 split tests pass.

### 2026-09-02 — OC-75 re-verified: venv already fixed, `uv.lock` still stale

Pre-flight re-check: the venv already had polars **1.43.2** (floor met) and all 47
`test_split.py` tests passed, so the audit's "10 failed" state was gone — but
`skyulf-core/uv.lock` still pinned **1.36.1**, below the `>=1.43.2` floor declared in
three places, and `uv lock --check` reported stale. That is what the entry above
resolved by deletion rather than regeneration.

### 2026-09-01 — Tracker created

- Tracker created from the master report's 116-finding inventory and its 4-tier
  suggested fix order. All items ⬜ open; no code changed.
- Corrections carried over: OC-100 retracted (false positive), OC-01 corrected,
  OC-46 downgraded 🟠→🟡, OC-12/18/40/42 worse than filed on execution re-verification.
- OC-71 (no authn/authz) is gated on a deployment-model decision before any fix
  work is scoped.
