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

**Qwen follow-up (2026-09-12):** the verified findings
filed 48 additional records, OC-271–318, after deduplication and scope
correction. Nineteen have since closed. The live queue now has
**58 open / 4 parked**; details
and exclusions are in the latest Log entry. Historical baseline counts
below are unchanged.

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
OC-185 parked; OC-183–184 closed on 2026-09-10. Historical baseline counts remain unchanged.

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
| OC-250 | 🟠 | Tuning bypasses the calculator's nonnative `class_weight` to `sample_weight` conversion while reporting `class_weight="balanced"` in the selected parameters (`modeling/_tuning/grid_random.py:150-154`, `refit.py:77`, `params.py:60`) | medium | ✅ fixed 2026-09-12 — share direct-fit weighting policy across all five tuning strategies and final refit, computing nonnative sample weights only from each fit's transformed training labels. |
| OC-248 | 🟠 | Merged fold preprocessing joins different observations: `LagFeatures`/`RollingAggregate` can sort a branch, and `LagFeatures(drop_na=True)`/`ManualBounds` can filter it, but the merge resets indexes, combines columns positionally and takes the first branch's target (`preprocessing/fold_adapter.py:28-34,52-54,181`) | medium | ✅ fixed 2026-09-12 — enforce the existing row-preserving positional merge contract for configured sorting/filtering in Core and backend, isolate validated configs and reject runtime row-count violations. |
| OC-249 | 🟠 | Public threshold optimization transforms only `X_val`, leaving `y_val` in its original order when preprocessing sorts or filters rows (`pipeline/_pipeline.py:543-549`) | small | ✅ fixed 2026-09-12 — validate raw row counts and carry labels through fitted preprocessing sorting/filtering before threshold scoring; both engines and failure-state preservation are covered. |
| OC-75 | 🔴 | Dev polars 1.40.1 below declared floor ≥1.43.2 — 10 tests, every notebook, a benchmark broken; prerequisite for trusting any polars result | 1 line | ✅ done |
| OC-12 | 🔴 | Row-dropping desyncs `X` and `y` on non-unique pandas indexes (`drop_rows.py:60-67`, `deduplicate.py:44-47`); polars path already correct | small | ✅ fixed 2026-09-03 |
| OC-58 | 🔴 | Numeric→boolean cast on polars treats any nonzero as `True` (`casting.py:143-178`) | small | ✅ fixed 2026-09-03 |
| OC-62 | 🔴 | `fingerprint()` not reproducible for any artifact holding an object-dtype array (`pipeline/seal.py:57-59`) | small | ✅ fixed 2026-09-03 |
| OC-271 | 🟠 | **DummyEncoder bool/date categories do not survive engine changes** (`skyulf-core/skyulf/preprocessing/encoding/dummy.py:78,92,183`) — Qwen #2. Use one category-key contract when fitting and replaying across engines; cover the Polars-training/pandas-serving path. | medium | ✅ fixed — 2026-09-12: Versioned Dummy artifacts preserve bool/date/datetime categories and saved predictions across engines, including Float32 controls; legacy artifacts retain their names and require encoder/model refit for portable replay. |
| OC-273 | 🟠 | **Mixed valid date formats become missing values during profiling** (`skyulf-core/skyulf/profiling/_analyzer/dates.py:31,143-159`) — Qwen #4. Preserve valid mixed-format dates or report parsing uncertainty without inventing missing data and a Drop recommendation. | medium | ✅ fixed — 2026-09-12: Automatic date conversion now requires every non-null value to parse, retaining original strings otherwise; the 50/950 mixed-format profile has zero fabricated missing values and no false Drop. |
| OC-276 | 🟠 | **Binning interval labels differ between training and replay engines** (`skyulf-core/skyulf/preprocessing/bucketing.py:55,67,170,179`) — Qwen #7. Canonicalize integral bin-edge labels across engines and verify an actual downstream DummyEncoder replay. | medium | ✅ fixed — 2026-09-12: New range artifacts persist identical labels for both engines, using exact edge text when rounding merges names; serialized binning-to-Dummy replay preserves all four indicators, with legacy formatting retained until coordinated refit. |
| OC-283 | 🟠 | **Binning can overwrite an existing output column or delete its source** (`skyulf-core/skyulf/preprocessing/bucketing.py:98,138,266,279`) — Qwen #15. Validate generated-name collisions and define safe same-name output/drop behavior before mutating either engine. | small | ✅ fixed — 2026-09-12: Fit and inference reject retained-name collisions and build outputs from original source values, so empty-suffix replacement preserves its binned output and other dropped source names can be reused safely. |
| OC-297 | 🟠 | **Replacing infinity can silently round large Polars integers** (`skyulf-core/skyulf/preprocessing/cleaning/invalid_value.py:67,182,213`) — Qwen #30. Preserve exact integer values and avoid widening integer columns merely to inspect or replace floating-point infinities. | small | ✅ fixed — 2026-09-12: Infinity-only cleanup skips impossible integer matches, preserving exact signed/unsigned values and dtypes at fit/apply boundaries while configured numeric rules and float replacements remain effective. |
| OC-290 | 🟠 | **Tokenizer and vectorizer outputs collide with existing columns** (`skyulf-core/skyulf/preprocessing/vectorization/tokenizer.py:63,98`) — Qwen #23. Apply collision-safe output handling to Tokenizer and the shared Count/TF-IDF/Hashing append path. | medium | ✅ fixed — 2026-09-12: Tokenizer and all four vectorizer consumers validate retained-column collisions at fit/apply; Tokenizer computes from original sources before dropping them, preserving safe name reuse and engine/wrapper/index behavior. |
| OC-293 | 🟠 | **Inference CSV pairs saved predictions with subsequently edited input** (`frontend/ml-canvas/src/components/pages/inference/useInferenceController.tsx:429-447`) — Qwen #26. Export the input snapshot belonging to the completed inference result and handle edits or reruns without mixing generations. | small | ✅ fixed — 2026-09-12: The results table and CSV use the displayed completed run's saved input through edits, reruns and history restoration; clearing or retiring a request blocks late success/error/finally writes. |
| OC-317 | 🟠 | **Drift summary cards ignore available categorical PSI metrics** (`frontend/ml-canvas/src/pages/drift/SummaryCards.tsx:16-27`) — Qwen #54. Include the appropriate categorical metric when calculating overall drift and selecting the most-drifted feature. | small | ✅ fixed — 2026-09-12: Summary cards include finite numeric and categorical PSI once per column; the 0.01/5 example shows average 2.5050 and category as most drifted, while unavailable PSI remains distinct from measured zero. |


### Next — wrong results in realistic configs

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-46 | 🟡 | Constant numeric features still publish non-finite PCA explained-variance ratios in the public profile (`profiling/_analyzer/multivariate.py:161`, `profiling/schemas.py:PCAComponent`) | small | ✅ fixed 2026-09-12 — return the existing unavailable PCA state for constant prepared features, keeping strict JSON finite while retaining PCA for varying inputs. |
| OC-35 | 🟠 | Binary evaluation still builds a ROC curve when the held-out partition contains only one class, publishing NaN coordinates (`modeling/_evaluation/classification.py:88-94`) | small | ✅ fixed 2026-09-12 — omit undefined binary ROC curves for single-class holdouts while preserving finite PR output and the earlier multiclass guard. |
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
| OC-66 | 🟠 | `CalibratedClassifierCV`'s user-selected base estimator silently discarded during tuning (`classification.py` vs `_tuning/engine.py`) | small | ✅ completed 2026-09-12 — the earlier flat-config fix missed Canvas search-space candidates; every strategy now resolves the selected base through trial fitting and final refitting, including fold pipelines. |
| OC-16 | 🟠 | KNN/Iterative imputers crash on all-missing fitted columns (`imputation/knn.py:64-76`, `iterative.py:68-84`) | small | ✅ fixed 2026-09-04 |
| OC-17 | 🟠 | SimpleImputer polars mean/median crashes on all-null columns (engine divergence, `imputation/_common.py:32-37`) | small | ✅ fixed 2026-09-04 |
| OC-69 | 🟠 | Engine trusts `config.nodes` list order, never verifies topological sort (`_schema_graph.py:49-70`); `_kahn_topological_order` already exists — wiring fix | small | ✅ fixed 2026-09-04 |
| OC-36 | 🟠 | F1 threshold tuning picks pathological threshold on single-class validation (`thresholds.py:101-111`) | small | ✅ fixed 2026-09-05 |
| OC-39 | 🟠 | NaN-bearing numeric columns publish `nan` stats and leak non-finite JSON (`profiling/analyzer.py:215-224`) | small | ✅ fixed 2026-09-05 |
| OC-40 | 🟠 | PCA/clustering "mean imputation" actually replaces NaN with `0.0` (`multivariate.py:46-60`) | small | ✅ fixed 2026-09-05 |
| OC-41 | 🟠 | Quartiles use nearest-rank, not linear interpolation (disagrees with pandas) (`analyzer.py:221-222`) | small | ✅ fixed 2026-09-05 |
| OC-42 | 🟠 | Skewness/kurtosis use biased estimators, breaking the hardcoded threshold rule (`analyzer.py:223-224`) | small | ✅ fixed 2026-09-05 |
| OC-43 | 🟠 | Correlation drops valid columns/rows instead of the defined missing-data policy (`correlations.py:41-44,100-110`) | small | ✅ fixed 2026-09-05 |
| OC-44 | 🟠 | Wasserstein drift thresholds normalized value but reports raw one (`drift.py:181-195`) | small | ✅ fixed 2026-09-05 |
| OC-45 | 🟠 | Schema drift computed but never counted or rendered as drift (`drift.py:76-98`) | small | ✅ fixed 2026-09-05 |
| OC-291 | 🟡 | **Out-of-order polling responses restore stale non-terminal job state** (`frontend/ml-canvas/src/core/hooks/useJobPolling.ts:161-197`) — Qwen #24. Scope response writes to the active polling generation and keep terminal state consistent with scheduling. | small | ✅ fixed — 2026-09-12: Polling ignores responses older than its newest applied snapshot and stops queued/socket refreshes after terminal results; slow APIs still publish progress and stale failures cannot corrupt retry accounting. |

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
| OC-219 | 🟡 | Threshold toggle accepts unsaved previews and returns HTTP 400; Save help incorrectly says saving leaves thresholds inactive | small | ✅ fixed 2026-09-09 — track saved state separately, disable the toggle until persistence, and explain that Save also enables predictions. |
| OC-38 | ⚪ | Clustering metrics treat DBSCAN `-1` noise as a real cluster (`metrics.py:432-459`) | small | ✅ fixed 2026-09-07 — DBSCAN noise rows are excluded from cluster counts and quality scores; regression coverage added |
| OC-146 | 🔴 | Binary `pr_auc` scored against wrong class on `{1,n}` labels — reports 0.32 vs true 0.97, no warning (`metrics.py:324-326`) | small | ✅ fixed 2026-09-05 |
| OC-37 | 🟡 | Binary PR-AUC dropped for string-labeled classifiers (`metrics.py:324-327`) | small | ✅ fixed 2026-09-05 — same one-arg fix as OC-146 |
| OC-147 | ⚪ | `optimize_thresholds` returns a dict shape that bypasses its own documented binary rule, flipping `>=` to `>` on exact ties (`thresholds.py:66-88`) | small | ✅ fixed 2026-09-05 — with OC-36 |

### Remaining — backend infrastructure

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-151 | 🟡 | Trial-buffer `clear_*` hooks documented but never called — 110.9 MB retained for process lifetime (`realtime/trial_buffer.py:56-59,103-106`) | small | ✅ fixed 2026-09-10 — execution cleanup releases both chart buffers; cancellation clears them after commit, and successful jobs retain persisted chart history. |
| OC-156 | 🟡 | `roc_auc` threshold-tuning objective scores hard predictions — bit-identical to `balanced_accuracy` (`threshold_tuning_service.py:77-92`) | small | ✅ fixed 2026-09-10 — remove ROC AUC from new threshold preview/save requests and UI choices; preserve existing saved cutoffs and recommend Balanced Accuracy. |
| OC-158 | 🟡 | Sync/async JSON serializers disagree: sync nulls 8 of 15 legitimate strings (`"nan"`, `"NaT"`, `"<NA>"`, `"inf"`…), async nulls none; 603-line module production-dead but test-covered (`serialization.py:369,435-446`) | half day | ✅ fixed 2026-09-10 — remove string-based missing detection, preserve real missing scalars, and pin text parity through public serializer entry points. |
| OC-169 | 🟡 | Uncaught request errors re-log raw messages and traceback chains, bypassing S3 call-site credential redaction (`middleware/error_handler.py:53-65`) | small | ✅ fixed 2026-09-10 — mask application request/error log surfaces and fallback-handler persistence; retain formatted diagnostics without raw `exc_info`. |
| OC-183 | 🟠 | `SmartCatalog` ignores dotenv-only bucket configuration and disagrees with the documented settings name (`data/catalog.py:556`, `config/mixins/aws.py:12`) | small | ✅ fixed 2026-09-10 — read canonical `AWS_BUCKET_NAME` through Settings, accept legacy `S3_BUCKET_NAME`, and align the docs with tested source precedence. |
| OC-184 | 🟠 | Production security headers are configured but never sent (`config/environments.py:84`, `main.py::_add_middleware`) | half day | ✅ fixed 2026-09-10 — apply the configured policy to application HTTP responses, preserving CORS/streaming and browser-tested Canvas/API-docs compatibility. |
| OC-224 | 🟠 | Drift compared a transformed splitter reference with a raw upload, reporting severe drift for the same file | small | ✅ fixed 2026-09-10 — resolve the selected model's unique saved raw loader snapshot; the user's existing job now reports 0/4 drift and PSI 0 without retraining. |
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
| OC-90 | ⚪ | Unknown split config keys silently dropped (`preprocessing/split.py`) | small | ✅ fixed 2026-09-12 — warn once about ignored public keys and list supported settings, preserving existing split semantics and quiet routing metadata; Canvas emits only supported settings. |
| OC-101 | 🟡 | `calibrated_classifier`'s `random_state` is dropped and its factories hardcode the seed | small | ✅ fixed 2026-09-12 — a cloneable calibration wrapper seeds supported base estimators during training, tuning and final refitting; defaults, explicit 0/None, unshuffled integer CV and saved predictions are covered. |
| OC-121 | ⚪ | Polars Enum columns omitted by text auto-selection, unlike pandas Categorical (`preprocessing/_helpers.py`) | small | ✅ fixed 2026-09-12 — recognize parameterized Enum dtypes in the shared selector; TextCleaning and AliasReplacement now include them while preserving nulls, explicit selections and replay behavior. |
| OC-120 | 🟠 | Decimal columns skipped by numeric auto-selection and mishandled during explicit numeric processing (`utils.py`, `preprocessing/_helpers.py` and numeric nodes) | small | ✅ fixed 2026-09-12 — recognize parameterized Polars and pandas object Decimals, normalize selected values at numeric boundaries, and preserve missingness, source values and target alignment. |
| OC-110 | 🟠 | Semantic-type inference misclassifies small categorical columns as `Text`, so task type never inferred (`profiling/_analyzer/_utils.py`, `analyzer.py`) | small | ✅ fixed 2026-09-12 — infer repeated small string categories from non-null counts in both profiling paths, restoring statistics and classification target analysis. |
| OC-113 | 🟠 | Near-perfect multicollinearity silently reports VIF = 1.0 — `max(1.0, …)` clamps numerical garbage (`profiling/_analyzer/numeric.py`) | small | ✅ fixed 2026-09-12 — use per-feature regression residuals for unstable correlation inversion, preserving high-VIF warnings without falsely flagging unrelated columns. |
| OC-114 | 🟡 | All-null tracked column yields 30 `NaN` autocorrelation lags as real analysis (≥1000-row datasets) (`temporal.py:167-191`) | small | ✅ fixed 2026-09-09 - undefined temporal diagnostics are omitted unless sufficient finite varying observations and finite results support them. |
| OC-112 | ⚪ | Comment and code disagree in the categorical profiler — the comment promises a rendered missing-value marker, the code `continue`s and discards the null category (`profiling/_analyzer/categorical.py:22-30`). *Filed as "disagree about the applied threshold"; the real subject is the null-category marker* | 1 line | ✅ fixed 2026-09-06 — comment-only, no behaviour change; the reasoning for why dropping the null category is correct now lives in the code comment it rewrote. See the log entry |
| OC-148 | 🟡 | PII detector flags ordinary 7+ digit numeric ID columns as "Email/Phone" (`profiling/_analyzer/text.py:107-128`) | small | ✅ fixed 2026-09-07 — phone detection now requires positive format evidence and repeated sample evidence; plain IDs, ZIP+4, and isolated phone-shaped IDs are excluded |
| OC-122 | ð¡ | `TextCleaning` silently ignores unrecognised operation name (`cleaning/text.py:151-153`) | small | â fixed 2026-09-08 |

### Remaining — file-coverage closure

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-140 | 🟠 | `InvalidValueReplacement` silently coerces nonnumeric values on pandas but fails on Polars | small | ✅ fixed 2026-09-12 — reject nonnumeric selections consistently before active operations at fit/apply, exclude durations from automatic selection, and preserve inactive operations, source values, targets and existing wrapper output conventions. |
| OC-142 | 🟠 | EDA correlation ratio η exceeds 1.0 with nulls; null-heavy columns rank as strongest association | small | ✅ fixed 2026-09-11 - use complete target-feature pairs for all group counts, means and sums of squares; preserve source rows and correct report ranking. |
| OC-144 | ⚪ | Geo distance column named `_km` even when the unit is miles | small | ✅ fixed 2026-09-11 - resolve automatic names from the selected unit; preserve explicit configuration and saved artifact names. |
| OC-143 | 🟠 | RFE ignores the UI's `k`, silently selecting half the features — **duplicate of OC-25**, same file and line; one fix retires both | small | ✅ fixed 2026-09-05 — with OC-25 |
| OC-141 | ⚪ | `invalid_values` param declared in `node_meta` with zero consumers | 1 line | ✅ fixed 2026-09-06 — key deleted from `node_meta` after re-verifying zero consumers across all three layers and the `.ambr` snapshots, behaviour-neutral because `user_picked_no_columns` keys off `columns`. The other half of the divergence (the params the calculator really reads are still undeclared) is left to **R1 step 1**. See the log entry |

### Remaining — cross-cutting & packaging

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-05 | 🟡 | `PowerTransformer` triggers a pandas deprecation that will become an error (`transformations/power.py:101`) | 1 line | ✅ fixed 2026-09-06 — **worse than filed**: casting each destination column to `float64` before the `.loc` write removes a per-column pandas FutureWarning that the surrounding bare `except` would otherwise swallow into a silent no-op, i.e. OC-28's failure mode arriving through OC-05. See the log entry |

### Remaining — encoding / cleaning / imputation / scaling / drop

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-267 | 🟡 | SMOTE+Tomek stores but ignores the configured `k_neighbors`, leaving its inner SMOTE at the default five neighbors (`preprocessing/resampling.py:221-222,292`) | small | ✅ fixed 2026-09-12 — configure inner SMOTE neighbors, strategy and seed, and expose/validate k Neighbors for SMOTE + Tomek in Canvas. |
| OC-254 | 🟡 | LabelEncoder fits supported NumPy/list targets but assumes native Series methods during apply (`preprocessing/encoding/label.py:69,113-115`) | small | ✅ fixed 2026-09-12 — normalize encoded list/NumPy targets with the fitted string-label policy, preserve native Series metadata and pass feature-only targets through unchanged. |
| OC-18 | 🟡 | One-hot/dummy generated names can collide with existing columns (`encoding/one_hot.py`, `dummy.py`) | small | ✅ fixed 2026-09-12 — reject conflicting generated names in one-hot, dummy, missing-indicator and multiclass target encoding at fit/apply, while preserving legitimate reuse of dropped source names. |
| OC-21 | 🟡 | WOE additive smoothing not normalized over categories (`encoding/woe.py`) | small | ✅ fixed 2026-09-11 — normalize class totals by all observed-category pseudocounts in full fits and training complements, correcting WOE/IV without rewriting saved mappings. |
| OC-172 | 🟡 | `StandardScaler` crashes on mixed pandas nullable numeric columns containing `pd.NA`; native sklearn and equivalent Polars input succeed (`preprocessing/scaling/standard.py:144,154`, `engines/sklearn_bridge.py:52`) | small | ✅ fixed 2026-09-09 - nullable numeric missing sentinels become NumPy NaN without rounding observed integers; StandardScaler applies numeric arithmetic safely. |
| OC-178 | 🟡 | `HashEncoder` hashes the same missing value into different buckets across Polars, pandas object, and pandas nullable string inputs, even with one shared fitted artifact (`preprocessing/encoding/hash.py:45,76`) | small | ✅ already fixed - verified 2026-09-09: one shared hash artifact gives identical missing-value buckets across pandas object/string and Polars. |
| OC-171 | 🟡 | Pandas `SimpleImputer` silently excludes explicitly selected constant/binary numeric columns for mean/median, leaving missing values unfilled; Polars honors the selection (`preprocessing/imputation/simple.py:173-177`) | small | ✅ already fixed - verified 2026-09-09: explicit constant/binary mean and median imputation fills missing values in both engines. |
| OC-179 | ð¡ | `DummyEncoder(drop_first=True)` retains a single-category indicator on Polars but removes it on pandas, changing feature width across engines (`preprocessing/encoding/dummy.py:33`) | small | â fixed 2026-09-08 |
| OC-180 | ð¡ | Pandas `TextCleaning(normalize_slash_dates)` crashes on `pd.NA` in a nullable string column; equivalent Polars input preserves the missing value (`preprocessing/cleaning/text.py:35-37,116`) | small | â fixed 2026-09-08 |
| OC-181 | ð¡ | `ValueReplacement` coerces every unrecognized boolean mapping key to `False`: mapping `{"banana": true}` changes `[true,false]` to `[true,true]` on both engines (`preprocessing/cleaning/value_replacement.py:31-32`) | small | â fixed 2026-09-08 |
| OC-182 | ð¡ | Encoder auto-detection ignores pandas `StringDtype` columns: Dummy/Hash encoding silently leaves strings untouched unless columns are selected explicitly (`preprocessing/encoding/_common.py:140`) | small | â fixed 2026-09-08 |
| OC-22 | ⚪ | `TargetEncoder.infer_output_schema` checks an impossible `regression` value (`encoding/target.py:340-360`) | 1 line | ✅ fixed 2026-09-06 — the `("binary", "regression")` passthrough was pinned by a test asserting a prediction for a config sklearn 1.8 rejects outright; changed to `"continuous"` and confirmed it really encodes rather than merely being reachable. See the log entry |
| OC-299 | 🟡 | **Automatic encoder selection skips native Enum columns** (`skyulf-core/skyulf/preprocessing/encoding/_common.py:131,146`) — Qwen #32. Include supported categorical Enum types in the common selector used by all six encoders. | small | ✅ fixed — 2026-09-12: The shared categorical selector now includes native Enum for all six encoder consumers; auto-selection matches explicit encoding and persisted held-out replay, with pipeline refit needed for previously skipped columns. |
| OC-296 | 🟡 | **MissingIndicator schema predicts output columns that execution will not create** (`skyulf-core/skyulf/preprocessing/drop_and_missing/missing_indicator.py:103,117,157`) — Qwen #29. Make inferred output schema match missing-column handling and validate downstream references against the actual result. | small | ✅ fixed — 2026-09-12: MissingIndicator predicts flags only for sources present in its original input schema; real schema-graph propagation now exposes downstream phantom references and matches both engines' runtime columns. |
| OC-285 | 🟡 | **DropMissingRows accepts invalid how values with different semantics** (`skyulf-core/skyulf/preprocessing/drop_and_missing/drop_rows.py:56,94,158`) — Qwen #17. Reject unsupported how values at the public configuration boundary before engine dispatch. | small | ✅ fixed — 2026-09-12: DropMissingRows validates how at fit and saved-artifact replay before engine dispatch, rejecting unsupported/null values while retaining omitted-any, valid any/all and threshold precedence with aligned targets. |
| OC-294 | 🟡 | **Numeric imputation handles explicitly selected text columns inconsistently** (`skyulf-core/skyulf/preprocessing/imputation/simple.py:169,187`) — Qwen #27. Validate or consistently filter incompatible explicit column selections before mean/median calculation. | small | ✅ fixed — 2026-09-12: Shared numeric validation rejects explicit text selections for mean/median before dispatch, preserving automatic selection and explicit binary/constant/Decimal numeric support plus most-frequent/constant text behavior. |
| OC-295 | 🟡 | **Automatic text cleaning converts pandas Decimal values into strings** (`skyulf-core/skyulf/preprocessing/_helpers.py:283`) — Qwen #28. Exclude semantic numeric Decimal columns from automatic text selection while retaining legitimate string cleaning. | small | ✅ fixed — 2026-09-12: Automatic text detection excludes semantic Decimal object columns on pandas, preserving their values and dtype in TextCleaning and AliasReplacement; explicit selections and existing fitted artifacts keep their prior replay contract. |


### Remaining — feature generation / selection / vectorization / transformations

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-256 | 🟡 | Feature-selection task inference treats pandas StringDtype/Categorical targets with more than ten classes as regression (`preprocessing/feature_selection/_common.py:54-69`) | small | ✅ fixed 2026-09-12 — recognize StringDtype and string categorical targets before the numeric cardinality heuristic. |
| OC-30 | 🟡 | Datetime extraction ignores the UI output name and overwrites collisions (`feature_generation/_pandas_ops.py`, `_polars_ops.py`) | small | ✅ fixed 2026-09-12 — honor exact/prefix output naming and the shared numbered collision policy on both engines, including fitted operation replay. |
| OC-24 | 🟠 | Polars group aggregates treat missing keys differently from pandas (`feature_generation/_polars_ops.py`) | small | ✅ already fixed — verified 2026-09-12: public fitted aggregates share null/NaN training groups across engines and wrappers; unseen groups remain missing, while the public artifact guard excludes divergent private unfitted helpers. |
| OC-230 | 🟡 | Native Polars NaN inputs propagate through feature ratios while pandas treats them as missing (`feature_generation/_polars_ops.py:_polars_ratio`, `_pandas_ops.py:_pandas_ratio`) | small | ✅ fixed 2026-09-11 - normalize native NaN and null ratio operands to zero before summing, preserving other operands, input columns and signed epsilon. |
| OC-23 | 🟠 | Polars `ratio` flips the sign of near-zero negative denominators (`feature_generation/_polars_ops.py:97-112`) | small | ✅ fixed 2026-09-11 - preserve the sign when clamping a near-zero ratio denominator to epsilon on Polars, matching pandas. |
| OC-211 | 🟡 | Pandas datetime features depend on prediction-batch composition: prepending a different valid date format makes the original rows' year/month/day missing in both `FeatureGeneration.datetime_extract` and `DateFeatures` (`feature_generation/_pandas_ops.py:179`, `time_series/date_features.py:64`) | small | ✅ already fixed - verified 2026-09-09: mixed-format batch companions preserve calendar features across both engines and aliases. |
| OC-212 | 🟡 | Similarity generation silently omits its output column for duplicate pandas indexes: label-based `.at[i]` returns Series to a scalar helper and the operation exception is swallowed (`feature_generation/_common.py:137-139`) | small | ✅ fixed 2026-09-08 — similarity now reads and assigns by row position, preserving duplicate indexes and individual scores; see the Log entry. |
| OC-25 | 🟠 | RFE "K" chosen in UI ignored by backend (`feature_selection/_common.py:236-240`) | small | ✅ fixed 2026-09-05 — closes OC-143 too |
| OC-26 | 🟠 | `HashingVectorizer` UI "none" norm is an invalid sklearn value → crash (`hashing_vectorizer.py`) | small | ✅ fixed 2026-09-11 — normalize the Canvas value to Python `None`, preserving unnormalized token counts and existing L1/L2 behavior. |
| OC-33 | 🟡 | `FeatureInteraction` cannot generate single-column self-products (`feature_generation/interaction.py`) | small | ✅ fixed 2026-09-11 — generate repeated-column combinations when `interaction_only=False`, including fewer columns than the degree; align Canvas validation. |
| OC-32 | 🟡 | `VarianceThreshold` crashes when all candidates are constant (`feature_selection/variance.py`) | small | ✅ fixed 2026-09-11 — record an empty selection when every candidate fails the threshold, preserving candidate variances, replay and invalid-input errors. |
| OC-31 | 🟡 | Frontend wrongly requires a target for unsupervised CorrelationThreshold (`FeatureSelectionNode.tsx`) | small | ✅ fixed 2026-09-11 — exempt correlation selection from target validation, retaining target requirements for all eight supervised methods. |
| OC-29 | 🟡 | `FeatureGeneration` advertises `polynomial` but silently skips it (`feature_generation/_common.py:24-31`) | small | ✅ fixed 2026-09-11 - reject unsupported operation types during fit and replay; polynomial requests point to the separate PolynomialFeatures node. |
| OC-34 | 🟡 | Count/TF-IDF vectorizers crash on empty or stop-word-only corpora (`vectorization/count_vectorizer.py`, `tfidf_vectorizer.py`) | small | ✅ fixed 2026-09-11 — warn and return an empty artifact for an empty vocabulary; preserve source data and replay, while other settings/pruning errors still raise. |
| OC-27 | 🟠 | `GeneralTransformation` ignores the UI `standardize` toggle (`transformations/general.py`) | small | ✅ fixed 2026-09-11 — retain each power rule's standardization choice through fitting and replay; older artifacts keep their previous default. |
| OC-28 | 🟠 | Box-Cox transform failures silently return untransformed data (`transformations/power.py:97-104`) | small | ✅ fixed 2026-09-06 — the silent path was the `valid_cols` filter, not the `except` (which has logged since the node was created); both engines now share `_fitted_columns_present`, which names the fitted columns the frame lacks, and fail-open is kept by decision. See the log entry |
| OC-300 | 🟡 | **Polars arithmetic feature fillna leaves floating-point NaN untouched** (`skyulf-core/skyulf/preprocessing/feature_generation/_polars_ops.py:30`) — Qwen #33. Apply the configured missing-value policy to both null and NaN before arithmetic operations. | small | ✅ fixed — 2026-09-12: Polars arithmetic fills both NaN and null using the configured replacement before add/subtract/multiply/divide, preserving source values and integer/null-only controls. |
| OC-264 | 🟡 | GeneralTransformation fits each power rule against the original input although apply executes same-column rules sequentially (`preprocessing/transformations/general.py:225-228`) | small | ✅ fixed — 2026-09-12: Power rules now learn from preceding same-column transformations, matching replay; log then standardized Yeo-Johnson has mean about 4e-16 and standard deviation one on both engines, with old artifacts retaining saved statistics until refit. |

### Remaining — profiling (outside the OC-39–46 cluster)

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-258 | 🟡 | Decomposition grouping columns named `value` or `ratio` collide with generated aggregation fields (`profiling/_analyzer/decomposition.py:140-148`) | small | ✅ fixed 2026-09-12 — separate group labels from aggregation output fields and retain native numeric measures, including when grouping and measuring the same column. |
| OC-192 | 🟡 | Decomposition's categorical null bucket loses rows when drilled into (`profiling/_analyzer/decomposition.py`) | small | ✅ fixed 2026-09-12 — preserve nullable group identity through response/filter payloads and tree navigation, keeping actual `Unknown` categories separate. |
| OC-48 | 🟡 | Expectations pass vacuously on empty frames (`profiling/expect.py`) | small | ✅ fixed 2026-09-12 — require at least one row by default in null/range/uniqueness checks, with explicit `allow_empty=True`, preserved schema validation and safe all-null range checks. |
| OC-51 | 🟡 | Transform advice can be mathematically invalid and contradict the clean-dataset message | small | ✅ fixed 2026-09-12 — use domain/skew-aware transform advice and evaluate all preparation recommendations, including target resampling, before emitting a limited clean-dataset message. |
| OC-247 | 🟡 | Missing categorical values inflate unique/rare-label counts and consume a top-10 frequency slot (`profiling/analyzer.py`, `_analyzer/categorical.py`) | small | ✅ filed and fixed 2026-09-12 — exclude nulls before frequency ranking and rare-label aggregation, and remove the null entry from the distinct-label count. |
| OC-235 | 🟠 | Directed causal edges are serialized backwards (`profiling/_analyzer/causal.py`) | small | ✅ fixed 2026-09-11 — preserve causal-learn endpoint order; real graph objects and public collider regressions verify both orientations. |
| OC-236 | 🟠 | Nominal target codes feed Pearson/Fisher-Z as numeric magnitudes, making results depend on category order (`profiling/analyzer.py`) | design + medium | ✅ fixed 2026-09-11 — omit categorical targets from numeric analysis while preserving category associations; numeric task overrides and omission metadata are explicit. |
| OC-237 | 🟡 | The causal graph cap guesses the target from column names and can omit the actual selection (`profiling/_analyzer/causal.py`) | small | ✅ fixed 2026-09-11 — retain the explicitly selected eligible numeric target and record all/target-correlation/variance selection. |
| OC-238 | ⚪ | Temporary target names become public graph and matrix labels (`profiling/analyzer.py`) | small | ✅ fixed 2026-09-11 — no temporary nominal target codes are created; original feature identities remain intact and omitted targets are explained. |
| OC-239 | 🟡 | An excluded target still drives rules and inferred task type (`profiling/analyzer.py`) | small | ✅ fixed 2026-09-11 — rule discovery requires an active target, including across repeated analysis; backend serialization regression verifies exclusion precedence. |
| OC-240 | 🟡 | Target/grouping names collide with aggregation aliases, silently removing statistics (`profiling/_analyzer/target.py`) | small | ✅ fixed 2026-09-11 — isolate group keys from statistic names; test both target directions, nulls, constants and every affected alias. |
| OC-241 | 🟠 | Sampled outlier offsets identify the wrong source rows (`profiling/_analyzer/multivariate.py`) | small | ✅ fixed 2026-09-11 — carry positions separately from model features and publish zero-based positions in the filtered input before sampling; scores and sampled rows remain unchanged. |
| OC-245 | 🟡 | Target name `count` collides with recommendation class-count output and aborts public analysis (`profiling/_analyzer/recommendations.py:_target_class_counts`) | small | ✅ filed and fixed 2026-09-11 — isolate the group key; numeric/string labels, literal count categories, missing labels and imbalance ratios are covered. |
| OC-49 | 🟡 | Valid partially-unlabelled PCA payloads crash plotting (`profiling/visualizer.py:716-737`) | small | ✅ fixed 2026-09-11 - render missing-label PCA points as a separate gray series; preserve every coordinate and omit the colorbar when all labels are absent. |
| OC-52 | ⚪ | Categorical colour mapping is process-nondeterministic (`visualizer.py:710-713`) | small | ✅ fixed 2026-09-11 - sort the shared categorical label set for reproducible PCA and geospatial colors across row order and Python hash seeds. |
| OC-191 | 🟡 | All-null and Polars Enum columns are classified as text and sent to string-only aggregates, aborting the whole profile (`profiling/analyzer.py`, `_analyzer/column.py`) | small | ✅ fixed 2026-09-11 - profile Null dtype as Unknown with missing-value reporting and native Enum as Categorical, preserving the rest of the report. |
| OC-199 | 🟡 | Explicit latitude/longitude selections bypass `exclude_cols`, returning coordinates for columns excluded from the profile (`profiling/_analyzer/geo.py:58-59`) | small | ✅ fixed 2026-09-11 - resolve explicit coordinates only within the persistent active column selection; excluded coordinates omit geospatial output. |
| OC-193 | 🟡 | A single missing timestamp removes time-series analysis at the 1,000-row resampling boundary: dynamic grouping receives null date keys and the exception is swallowed (`profiling/_analyzer/temporal.py:232,243`) | small | ✅ fixed 2026-09-09 - null timestamps are excluded from temporal calculations without changing the existing resampling threshold. |
| OC-217 | 🟠 | Repeated profiling on one analyzer returns previously excluded columns in sample data and frame statistics because only newly excluded columns trigger the narrowed frame (`profiling/analyzer.py:analyze`) | small | ✅ fixed 2026-09-09 - persistent column exclusions remain effective for sample rows and frame statistics across repeated calls. |
| OC-216 | 🟡 | Rule feature conditions include unrelated labels from Polars shared categorical dictionaries even when those labels never occur in the feature (`profiling/_analyzer/rules.py:_build_feature_matrix`) | small | ✅ fixed 2026-09-09 - feature-local codes and label lists exclude unrelated categories from displayed rule conditions. |
| OC-198 | 🟠 | Profiling a string target overwrites an existing `<target>_encoded` feature, then duplicate selection prevents correlation and causal analysis (`profiling/analyzer.py:341-348`) | small | ✅ fixed 2026-09-09 - encoded targets use an unoccupied column name and temporary analyzer state is restored after analysis. |
| OC-190 | 🟡 | A categorical column named `count` crashes profiling and categorical drift because `value_counts()` generates the same column name (`profiling/analyzer.py:286-290`, `profiling/drift.py:380-381`) | small | ✅ fixed 2026-09-09 - profiling, categorical drift and rule target counts use distinct value/count names. |
| OC-189 | 🟡 | Classification rule text reports `Samples: 1` for leaves containing multiple rows: it sums sklearn's normalized class proportions instead of using the leaf sample count (`profiling/_analyzer/rules.py:299-301`) | small | ✅ fixed 2026-09-09 - classification rule support uses the fitted tree row count while confidence retains class proportions. |
| OC-188 | 🟠 | Rule discovery decodes sklearn class positions against Polars' shared category dictionary, publishing labels absent from the target while reporting perfect accuracy (`profiling/_analyzer/rules.py:169-170,196-198,295-298`) | small | ✅ fixed 2026-09-09 - target-local codes map every rule prediction to the actual observed class label. |
| OC-305 | 🟡 | **All-null temporal profile bounds serialize as the string None** (`skyulf-core/skyulf/profiling/_analyzer/dates.py:164-171`) — Qwen #40. Return actual nullable bounds in the profile schema instead of stringifying absent temporal extrema. | small | ✅ fixed — 2026-09-12: Missing Date/Datetime extrema remain actual nulls in profile dictionaries and JSON while native units, timezones and nonmissing bounds retain their existing behavior. |
| OC-257 | 🟡 | Native Enum columns bypass categorical drift dispatch and disappear from the report (`profiling/drift.py:154-162`) | small | ✅ fixed — 2026-09-12: Native Enum uses categorical drift dispatch; the 100 a to 100 b shift yields PSI 10.543651559430593 and one drifted column, matching String/Categorical controls. |


### Remaining — core / engines / pipeline

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-261 | 🟡 | Pipeline fingerprints omit stored decision thresholds, allowing identical fingerprints for different predictions under the same `use_tuned_thresholds=True` call (`pipeline/_pipeline.py:695-701`) | small | ✅ fixed 2026-09-12 — include stored decision thresholds in the semantic pipeline fingerprint while preserving untuned and legacy identities. |
| OC-262 | 🟡 | Both dataframe wrappers recurse through missing `_df` during pickle/joblib restoration (`engines/pandas_engine.py:100-102`, `engines/polars_engine.py:121-123`) | small | ✅ fixed 2026-09-12 — resolve wrapper state without recursive delegation during restoration; pickle/joblib round trips preserve both native frames and their methods. |
| OC-74 | 🟡 | `NodeRegistry.list_models()` hides all 4 Ensemble models; `category` arg dead (`registry.py:101-108`) | small | ✅ fixed 2026-09-11 - include Modeling and Ensemble in model discovery and exclude both from transformer discovery; preserve exact category filters and registration order. |
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
| OC-266 | 🟡 | Polars categorical-to-boolean casting invokes string methods on categorical expressions; Enum inputs fall through to an unsupported generic cast (`preprocessing/casting.py:109,177-179,221`) | small | ✅ fixed 2026-09-12 — cast categorical and Enum labels through the existing boolean string alias table. |
| OC-265 | 🟡 | Polars string-to-datetime casting uses a generic cast without parsing and silently replaces valid ISO dates with nulls under default coercion (`preprocessing/casting.py:221`) | small | ✅ fixed 2026-09-12 — parse textual dates with the shared mixed-format parser, retaining coercion, strict errors and the resolved engine dtype. |
| OC-263 | 🟡 | Haversine roundoff can put the intermediate outside `[0,1]`, producing NaN for valid antipodal coordinates on both engines (`preprocessing/geo/distance.py:53-54,108-109`) | small | ✅ fixed 2026-09-12 — clamp the Haversine intermediary to its valid range on both engines, preserving missingness and finite antipodal distances. |
| OC-59 | 🟠 | DatasetProfile selects different numeric-column sets on pandas and Polars (`preprocessing/inspection.py`) | small | ✅ fixed 2026-09-12 — use shared dtype-based selection, include supported small/unsigned integers and binary/constant/empty numeric columns, and exclude temporal columns on both engines. |
| OC-60 | 🟠 | GeneralBinning ignores missing_strategy=label on Polars (`preprocessing/bucketing.py`) | small | ✅ fixed 2026-09-12 — fill missing and out-of-range bins with the configured label; labeled Polars outputs use a stable String dtype and the default keep strategy is preserved. |
| OC-176 | 🟡 | Polars `LagFeatures(drop_na=True)` removes nulls but retains float NaN in source/lag columns; equivalent pandas input drops those rows (`preprocessing/time_series/lag.py:54-59`) — independent of OC-165's y desynchronization | small | ✅ fixed 2026-09-09 - Polars lag filtering removes both null and floating NaN rows with one positional selection shared by X and y. |
| OC-175 | 🟡 | Polars `RollingAggregate` propagates float NaN through windows instead of ignoring missing observations like pandas — `[1,NaN,3]` with window 2 yields mean `[1,NaN,NaN]` vs `[1,1,3]` (`preprocessing/time_series/rolling.py:48`) | small | ✅ fixed 2026-09-09 - floating NaN is treated as missing inside Polars rolling expressions, preserving source values and grouped-window semantics. |
| OC-174 | 🟡 | Polars `DateFeatures` crashes on an entirely invalid string date column despite `strict=False`; pandas produces nullable calendar features (`preprocessing/time_series/date_features.py:102`) | small | ✅ already fixed - verified 2026-09-09: wholly invalid date strings produce nullable calendar features in both engines. |
| OC-173 | 🟡 | Duplicate pandas indexes reintroduce missing rows during EllipticEnvelope prediction, disabling outlier filtering (`preprocessing/outliers/elliptic.py`) | small | ✅ fixed 2026-09-08 in `f12dde9f8`; reverified 2026-09-09 — finite-row selection and prediction scatter are positional, preserving indexes and X/y alignment. |
| OC-165 | 🟡 | Pandas `LagFeatures(drop_na=True)` removes X rows but leaves tuple y untouched — 3 rows become 2 features / 3 targets even with a unique index (`preprocessing/time_series/lag.py:85-87`) | small | ✅ fixed 2026-09-06 — with OC-163; `drop_na` now filters y through the same positional keep-mask as X, duplicate-index case included. See the log entry |
| OC-166 | 🟡 | Polars `IQR`, `ZScore`, and `ManualBounds` filter X but leave NumPy y untouched — 5 rows become 4 features / 5 targets; Polars Series y works (`preprocessing/outliers/_common.py:9-15`) | small | ✅ fixed 2026-09-06 — with OC-163, and **broader than filed**: a fourth copy of the same silent pass-through sat inline in `EllipticEnvelope`, and list targets failed too (crashing on pandas, no-opping on polars). See the log entry |
| OC-284 | 🟡 | **Duplicate bin edges produce incompatible results and discard custom labels** (`skyulf-core/skyulf/preprocessing/bucketing.py:88,162,224,234,267`) — Qwen #16. Validate or normalize duplicate edges and their labels together with one explicit policy for both engines. | small | ✅ fixed — 2026-09-12: Duplicate-edge labels must match distinct intervals on both engines, preserving unambiguous labels and rejecting labels for zero-width bins while retaining established unique-edge fallbacks. |


### Remaining — modeling / tuning

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-270 | 🟡 | Tuning's hardcoded missing-value allowlist rejects tree estimators that natively accept NaN in the installed sklearn version (`modeling/_tuning/engine.py:318,404-416`) | small | ✅ fixed 2026-09-12 — use configured estimator capabilities for missing-feature admission, preserving search overrides, required structure and legacy sklearn tags. |
| OC-268 | 🟡 | Calibrated ensemble fitting applies nested calibration-estimator parameters before creating the calibration wrapper, losing selected base-model settings during subsequent fit/CV (`modeling/ensemble.py:299-300,468-473`) | small | ✅ fixed 2026-09-12 — apply tuned nested parameters after calibration wrappers are constructed so refits and CV preserve selected values. |
| OC-252 | 🟡 | Weighted PR-AUC tuning derives its class axis from the holdout instead of the trained model (`modeling/_tuning/metrics.py:84-90`) | small | ✅ fixed 2026-09-12 — derive the probability class axis from the fitted estimator across all five search strategies, preserving the trained binary positive class. |
| OC-187 | 🟡 | LightGBM's subsample control and search dimension have no effect with default frequency zero | small | ✅ fixed 2026-09-12 — resolve automatic bagging after candidate parameters, enabling ordinary row sampling while preserving GOSS, explicit frequencies, native aliases and existing artifacts. |
| OC-218 | 🟡 | Connected model CV seed `0` is replaced by the existing ensemble seed during frontend settings synchronization | small | ✅ fixed 2026-09-09 - explicit zero now survives synchronization and both fixed/tuned request conversion; missing seeds retain the ensemble default. |
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
| OC-275 | 🟡 | **Temporal CV misses pandas datetime.date columns** (`skyulf-core/skyulf/modeling/cross_validation.py:283-291,361,399`) — Qwen #6. Recognize native date objects consistently and preserve chronological folds through the public CV and fold-preprocessing path. | medium | ✅ fixed — 2026-09-12: Temporal CV recognizes homogeneous native pandas date objects, preserving stable target-aligned sorting and missing dates last; public fold boundaries now match Polars at 3/4, 6/7 and 9/10 instead of 8/2, 9/4 and 11/5. |
| OC-282 | 🟡 | **Elasticnet with a missing l1_ratio silently behaves like L2** (`skyulf-core/skyulf/modeling/_sklearn_compat.py:43-44`) — Qwen #14. Validate or resolve the elasticnet ratio before estimator construction and cover the real pipeline/search-space contract. | small | ✅ fixed — 2026-09-12: Omitted/null Elastic Net ratios resolve to 0.5 for direct fits, all five searches and refits, preserving numeric and Optuna distribution choices; mixed-penalty searches with unspecified ratios fail clearly. The separate explicit-ratio precedence mismatch is filed as OC-319. |

### Remaining — frontend

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-234 | 🟡 | 3D scatter legends promise triangle/star markers that Plotly silently renders as circles (`chartMarkerShapes.ts`, `ThreeDScatterPlot.tsx`) | small | ✅ fixed 2026-09-11 - use five shapes supported by both scatter engines and render matching plus/X legend symbols; real browser regressions verify resolved Plotly markers. |
| OC-242 | 🟠 | Causal graph ID sanitization merges distinct column names (`components/eda/CausalGraph.tsx`) | small | ✅ fixed 2026-09-11 — map original identities to distinct opaque IDs and preserve edge endpoints; Chromium covers spaces, punctuation, Unicode and prototype-like names. |
| OC-243 | 🟡 | Bidirected causal edges render only one arrowhead (`components/eda/CausalGraph.tsx`) | small | ✅ fixed 2026-09-11 — render both endpoint markers for bidirected edges and retain directed/undirected behavior. |
| OC-244 | 🟡 | Loading an empty historical graph preserves old results (`components/eda/CausalGraph.tsx`, `tabs/CausalTab.tsx`) | small | ✅ fixed 2026-09-11 — clear graph state and render an unavailable state with the current target explanation; desktop/mobile history regressions verify removal. |
| OC-231 | ⚪ | EDA scatter and geospatial category colors depend on first appearance; reordering identical points changes the color of the same category (`scatterGrouping.ts:32-35`, `GeospatialTab.tsx:60-67`) | small | ✅ fixed 2026-09-11 - Sort observed category labels and share color metadata across 2D, 3D, map and legends; row reordering preserves category colors. |
| OC-232 | 🟡 | Missing PCA/scatter labels are merged with genuine `Other` labels, making unlabeled and observed categories indistinguishable (`scatterGrouping.ts:22-23`) | small | ✅ fixed 2026-09-11 - Keep missing values in a neutral group with collision-safe captions, preserving every point and showing all-missing legends. |
| OC-233 | 🟡 | Valid scatter labels such as `__proto__`, `constructor` and `toString` crash grouping because inherited object properties are treated as arrays (`scatterGrouping.ts:20-24`) | small | ✅ fixed 2026-09-11 - Use Map-backed grouping so prototype-named categories render normally without losing points or crashing. |
| OC-214 | 🟠 | Frontend PostCSS retains vulnerable `nanoid@3.3.17` (CVE-2026-67213) | small | ✅ fixed 2026-09-10 — lockfile and installed tree use compatible 3.3.18; npm audit no longer reports the package, and frontend coverage/tests, lint, CCN and build pass. |
| OC-215 | 🟡 | Frontend Tailwind/PostCSS retains vulnerable `postcss-selector-parser@6.1.2` (CVE-2026-9358) | small | ✅ fixed 2026-09-10 — both parent paths resolve to compatible 6.1.4; npm audit no longer reports the package, and frontend coverage/tests, lint, CCN and build pass. |
| OC-228 | 🟡 | Add Casting Rule overwrites the first column's existing type with `float` when every available column already has a rule | small | ✅ fixed 2026-09-10 — Add is disabled and its handler returns when no unassigned usable column remains; unit and browser regressions preserve types and cover removal/re-addition and Preview payloads. |
| OC-229 | 🟡 | A late inspector response replaces the details of a more recently selected node | small | ✅ fixed 2026-09-10 — request generations guard data, error and loading writes; cleanup invalidates old requests on selection changes, close and unmount, with unit/browser coverage of reordered responses and retries. |
| OC-222 | ⚪ | Audit Log's loaded-page filter hint contradicts its server-side actor/kind/time filtering | small | ✅ fixed 2026-09-10 — the hint now explains full-history filtering before the page limit; API behavior is unchanged. |
| OC-221 | 🟡 | Error Log applies an older search response after a newer response, displaying rows that disagree with the current search | small | ✅ fixed 2026-09-10 — request generations guard HTTP/pipeline results, errors and loading; refresh and effect cleanup invalidate obsolete work. |
| OC-220 | 🟡 | Resampling Target Column native suggestions open away from the input in the user's browser | small | ✅ fixed 2026-09-09 — use an anchored editable listbox; docked/expanded browser geometry and keyboard selection are covered. |
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

### 2026-09-12 — 0.8.21 version alignment and six-fix commit verification

Prepared OC-264/275/282/285/294/295, their regression tests and documentation
for the requested commit. The app's root `pyproject.toml`, Core's `setup.py`,
Canvas `package.json` and both root version fields in its `package-lock.json`,
and the app entry in `uv.lock` now agree on **0.8.21**. Refreshed the local
editable Core installation without changing dependencies and verified that
`skyulf.__version__` reports `0.8.21`.

Fresh verification passes **1,372 Core tests / three snapshots** and
**53 backend tests / seven snapshots** in separate runs. Core Ruff, global
Ty, frontend version checks, ESLint, complexity, TypeScript/Vite build and all
11 bundle budgets pass. The rebuilt frontend assets match the committed
outputs; the package and lockfile diffs contain version changes only.
Queue status remains **58 open / 4 parked**, including the separate OC-319
follow-up. The user's unreviewed Qwen report and local investigation artifacts
are outside this commit's scope.

### 2026-09-12 — OC-264/275/282/285/294/295: six fixes; OC-319 filed separately

Committed the preceding seven-finding batch as `c9de8e7a` with DCO sign-off.
Fresh pre-commit verification passed 207 Core regressions, 28 backend tests
and 48 frontend tests; every applicable commit hook passed. The commit includes
the already verified frontend assets. No push was requested or performed.
Continued with six unparked findings using three implementation agents and
independent reviews. This new batch remains uncommitted for review.

- **OC-264:** GeneralTransformation now fits each power rule against preceding
  transformations on that same column. The original log then standardized
  Yeo-Johnson example gives mean `4.07081775696e-16` and standard deviation
  one on both engines. Sixteen failing regressions preceded the repair;
  eighteen cases cover repeated/interleaved rules, Box-Cox eligibility,
  standardization, JSON replay, FeatureEngineer and old-artifact controls.
  Artifact layout and legacy replay stay unchanged; refit affected steps and
  downstream models to replace previously incorrect learned statistics.
- **OC-275:** native pandas `datetime.date` object columns now participate in
  temporal auto-detection. Public CV with real fold preprocessing and a fitted
  model now records train-max/validation-min days `3/4, 6/7, 9/10`, replacing
  `8/2, 9/4, 11/5`. Twenty-one focused cases preserve stable sorting, missing
  dates last, target alignment, wrapper/native engines and first-column order.
  Text, mixed date/datetime objects and entirely missing objects remain
  outside automatic detection. Normalize such columns explicitly; rerun
  temporal CV/tuning that previously used unordered native date objects.
- **OC-282:** null Elastic Net ratios now resolve to the existing omitted
  default `0.5`. A shared search-config normalization applies that same value
  before Grid, Random, Halving Grid, Halving Random and Optuna dispatch, with
  and without fold preprocessing; final refits retain the resolved value.
  Explicit numeric ratios and caller settings are preserved. A search that
  mixes Elastic Net with other penalties while leaving the ratio unspecified
  fails with instructions to search Elastic Net separately. Independent review
  caught a regression with supplied Optuna distributions: four additional
  tests now protect numeric distributions, categorical penalty/ratio choices
  and their categorical sampling semantics. All 66 focused cases pass,
  including actual pandas/Polars SkyulfPipeline fits. The different explicit
  ratio/penalty precedence problem below remains separate.
- **OC-285:** DropMissingRows validates `how` before fit and artifact replay,
  preventing invalid settings from selecting different filtering policies on
  pandas and Polars. Unsupported values and explicit null now raise clear
  `ValueError`, even under a threshold; omitted `how` remains `any`. Valid
  policies preserve their row counts and target alignment. The first run
  captured 32 invalid-policy failures alongside eight valid controls.
- **OC-294:** shared SimpleImputer validation rejects explicit nonnumeric
  mean/median selections before engine dispatch. Previously pandas silently
  discarded text selections while Polars failed on reduction. Numeric-looking
  strings remain text until explicitly cast; binary, constant, Decimal and
  typed all-null numeric controls remain supported. The existing all-null
  parity fixture now declares numeric dtype on both engines. Text
  most-frequent/constant strategies and held-out replay remain covered.
- **OC-295:** automatic text detection excludes pandas Decimal object columns
  using the existing semantic numeric predicate. TextCleaning now preserves
  values such as `Decimal("12.34")` and their dtype, matching Polars; the shared
  AliasReplacement selector also excludes them. Combined OC-294/295 tests
  reproduced 28 failures before fixing and pass 56 focused cases afterward.
  Explicit selection and old artifacts keep their stored replay behavior;
  refit automatic text-cleaning selections and rerun from original data.

**OC-319 — new, still open:** explicit L1/L2 penalty and nonmatching numeric
ratio have inconsistent constructor/searcher semantics. On sklearn 1.8.0,
use `make_classification(n_samples=80, n_features=6, random_state=4)` and
`solver="saga", l1_ratio=0.5, C=0.1, max_iter=2000, random_state=2`.
Compare `instantiate_model(LogisticRegression, params).fit(X, y)` with an
estimator constructed from the calculator defaults, followed by
`set_params(**params).fit(X, y)`. Maximum coefficient differences are
`0.10500863958311868` for `penalty="l2"`, `0.8108656276940928` for `"l1"`,
and zero for `"elasticnet"`. Public single-candidate Grid and Halving Grid
with stratified two-fold CV, CV seed 3, seed 2, metric `neg_log_loss`, and
halving `min_resources=80, factor=2` score the same candidate at
`-0.3301352967281843` and `-0.27716181751699975`, then return identical final
coefficients. Both an independent reviewer and the root reproduced this.
Resolve the existing documented explicit-ratio precedence against the public
penalty contract and apply one rule to candidate fits and refitting. This
reproduction concerns supported Core calls; frontend reachability and other
installed sklearn versions were not established.

Final verification: **1,372 tests and three snapshots passed** across 31
affected Core suites covering transformations, imputation, cleaning, missing
rows, CV, tuning, pipeline integration and artifacts. **53 tests and seven
snapshots passed** in the separate backend config, fold-preprocessing,
calibrated-tuning and feature-node suites. Scoped formatting, Core Ruff and
global `ty check` pass; the additional model type assertion was rerun in both
pipeline engine cases. Existing dependency warnings remain. Updated reference
docs, temporal CV/tuning guides, and the v0.8.21 changelog. This batch changes
Core behavior and documentation; frontend source and build assets stay at the
preceding verified commit. Six rows closed and one new row added:
**58 open / 4 parked** remain. R1, DRIFT-01 and the four parked items are preserved.

### 2026-09-12 — OC-257/290/291/293/296/300/317: seven verified fixes

Committed the preceding eight-finding Core batch as `8053a84c` with DCO
sign-off after fresh regression tests and passing commit hooks. Continued
with seven unparked findings using three agents and independent diff reviews.
This new batch remains uncommitted for review; no push was requested or made.

- **OC-257:** native Enum columns now enter categorical drift calculation.
  Replacing 100 `a` values with 100 `b` values produces categorical PSI
  `10.543651559430593` and one drifted column, matching String/Categorical
  controls. Stable data, unused Enum labels and custom thresholds are covered.
  The separate lossy dtype-cast finding OC-47 remains open.
- **OC-290:** Tokenizer, Count, TF-IDF, Hashing and Sentence Embedder validate
  generated names against the full retained input schema at fit and replay.
  Retained-name collisions now raise `ValueError` on both engines. Tokenizer
  computes all outputs from original source values before dropping/concatenating,
  so names of dropped sources can be reused without corrupting later inputs.
  Tests cover pipeline replay, native/wrapped engines, labels, indices, missing
  sources and vocabulary controls. Sentence Embedder's external model loader
  is stubbed; no model download or real embedding quality claim is involved.
- **OC-291:** effect-local polling generations protect status and failure
  counters from out-of-order completions. Terminal results stop pending timer
  and socket refreshes. Review reproduced starvation in an initial guard when
  every request exceeded the polling interval; the final guard compares against
  the newest applied request, allowing slow progress while newer work is pending.
- **OC-293:** result-table rows and downloaded CSV use the displayed completed
  run's saved input, including persisted history and results kept during a rerun.
  Predictions `10/20` for input `1/2` still export `1,10` and `2,20` after the
  editor changes to `9/8/7` or invalid JSON. Clear, undeploy, deployment changes
  and unmount retire obsolete requests; stale success/error/finally handlers
  cannot overwrite newer state. Unmount retains interrupted-run provenance.
- **OC-296:** schema prediction checks MissingIndicator sources against the
  original input schema, matching runtime filtering. Input `x` with selected
  `ghost` no longer invents `ghost_missing`. Actual backend schema propagation
  and reference validation detect both the original missing source and the
  downstream phantom flag; mixed selections, custom suffixes and cascading
  flag names are covered on both engines. Automatic selection stays unknown.
- **OC-300:** Polars arithmetic replaces both NaN and null before all four
  arithmetic operations, using the configured `fillna` or default zero. The
  `[1, NaN, 3] + 2` example with replacement `10` now gives `[3, 12, 5]`.
  Primary/secondary operands, source preservation and integer/null-only
  controls agree with pandas; the separate ratio behavior is unchanged.
- **OC-317:** the real summary cards use finite `psi` and `psi_categorical`
  measurements once per column. Numeric `0.01` plus categorical `5` now shows
  average `2.5050`, Significant drift, and category as Most Drifted. Missing or
  non-finite PSI is unavailable, while measured zero remains valid. Scope is
  the summary cards; compact history summaries and CSV metric selection were
  not changed by this item.

Verification: failing regressions were captured before production changes.
The final affected Core suites passed **703 tests** (365 feature/schema/missing
tests plus 338 vectorization/drift tests), and the separate backend schema
graph/preview/reference suites passed **28 tests**. Full frontend Vitest passed
**2,522 tests in 195 files**. Four Playwright cases passed at desktop 1440px
and mobile 390px using the real pages with mocked API boundaries; inference
checks inspect actual downloaded CSV contents, and drift screenshots were
visually checked. The CSV test's ambiguous upload/download selector was fixed
before its successful rerun. Scoped Ruff/format, global `ty check`, full ESLint,
CCN, explicit E2E lint, TypeScript/Vite production build and bundle size gates
passed. Generated `static/ml_canvas` assets were rebuilt.

Updated preprocessing, text-NLP and drift user documentation plus the v0.8.21
changelog. Existing conflicting text inputs must be renamed; rerun affected
transformations from original data and refit downstream models where needed.
Regenerate old Enum drift checks to recover omitted metrics. The seven rows
are archived with their evidence: **63 open / 4 parked** remain. The user's
deleted standalone Qwen reports and unrelated `tmp_eda_shots/` stay untouched.

### 2026-09-12 — OC-271/273/276/283/284/297/299/305: eight Core fixes

Continued the user-approved queue with three implementation agents owning
binning, date profiling and categorical encoding; the primary agent handled
integer precision and integration. A separate read-only reviewer inspected
the combined changes, and the date agent independently checked integer cleanup.

- **OC-271/299, categorical encoding:** newly fitted Dummy artifacts record
  category-key version 1. Booleans and date/datetime instants render consistently
  across engines, retaining nanoseconds and normalizing aware instants to UTC;
  literal strings and dtype-aware numeric rendering are preserved. A persisted
  linear pipeline that returned `[20,20,20,20]` on the other engine now retains
  `[10,30,10,30]`. The common selector includes native Enum for all six encoder
  consumers, with persisted artifacts and held-out vocabulary controls.
- **OC-276, range replay:** newly fitted range artifacts store their label text.
  Serialized binning-to-Dummy pipelines preserve all four active indicators in
  both engine directions, including wrappers. If requested rounding duplicates
  labels for distinct bins, exact edge text is stored instead.
- **OC-283/284, binning validation:** validate against the complete original
  schema during fit and apply; build all outputs before dropping source names.
  Retained-column collisions raise, and an empty suffix with source dropping
  keeps the binned output. Duplicate edges accept labels for distinct intervals;
  `[0,5,5,10]` with three labels now raises while two labels remain intact.
- **OC-273/305, date profiling:** only commit an inferred cast when the chosen
  existing parser accepts the entire non-null column. Mixed formats retain
  their original strings and accurate missing counts, without false Drop or
  temporal metadata. The 50 ISO/950 other-format example no longer invents
  950 missing values. All-null Date/Datetime bounds serialize as actual nulls;
  native temporal units/timezones and real missingness remain covered.
- **OC-297, integer precision:** skip infinity comparisons on integer columns
  rather than widening them to the replacement sentinel's float dtype. Values
  such as `9007199254740993` and UInt64 maxima remain exact with nullable/native/
  wrapped inputs and inference dtype changes. Float infinity replacement and
  separately configured integer rules still execute.

Compatibility: unversioned Dummy and range-binning artifacts retain their
previous category names and rendering. Refit affected encoders/binning and
dependent models together to obtain portable replay before changing engines;
also refit automatically selected pipelines to encode previously skipped Enum
columns. Regenerate saved profiles, and normalize retained mixed date formats
explicitly when temporal analysis is needed. Reprocess integer data from its
original source if an earlier run rounded it. User guidance, source docstrings
and the v0.8.21 release notes record these behaviors; no Canvas change is needed.

Verification:

- Failing-first results: binning **82 failed / 18 passed**; dates **10 failed /
  5 passed**; initial encoding **22 failed / 26 passed**; integer precision
  **8 failed / 10 passed**. Follow-up tests reproduced eight tiny-range failures.
- Independent review caught a new Float32 key expansion introduced by the
  first encoding patch. Seven of thirteen added controls failed before the
  correction. The final renderer retains original non-temporal dtype formatting;
  the reviewer independently verified Float32/nullable/categorical replay in
  both directions and found no remaining actionable issue in the bounded review.
- Final integrated run across **33 Core files: 1,574 passed**, including
  **233 new regression cases**, registry/artifact contracts, persisted pipelines,
  preprocessing dispatch, semantic seals, encoding and profiling controls.
  Its 89 warnings are dependency deprecations and expected edge-data diagnostics.
- Root project binning suite ran separately: **7 passed**, one existing Windows
  physical-core-count warning. The integer reviewer also checked 32 additional
  signed/unsigned-width, null, empty-frame and replacement combinations.
- Full-repository Ruff and backend/Core/tests type checks passed. All ten
  changed/new Python files pass format checks, and `git diff --check` is clean.

The user confirmed removing the standalone Qwen source/report after filing;
dead links were removed and the measured evidence stays in each queue/archive
row. These eight completed rows moved to the archive: **70 open / 4 parked**
remain, with OC-71/72/73/185 still parked and R1/DRIFT-01 unchanged.

### 2026-09-12 — verified Qwen review filed as OC-271–318

At the user's request, filed the actionable scope of all 57 Qwen claims after
verification. The original verdicts remain
**41 confirmed / 12 partial / 3 not established as runtime bugs / 1 already fixed**.
The production-code baseline is DCO-signed **b0fd13b8**, containing the earlier
seven Core fixes and the reported Canvas path/display-name repair.

- Added **48 open records, OC-271–318**, with trigger, affected source,
  measured evidence, repair scope and the individual Qwen claim number.
- Reused **OC-253/65/64** for #1/#50/#57 and added the new evidence to those
  existing rows; #13 remains closed as **OC-268**.
- Combined #18/#41 into **OC-286** for the non-finite JSON persistence contract.
  Both EDA profile writes and background preview/job metrics must be verified
  before closure; a repair of just one path is incomplete.
- Retained policy-only #36 outside the OC defect count. #20/#35/#56 do not
  acquire standalone runtime-bug records, with their disposition preserved
  in the queue's decision notes.
- **30 + 48 = 78 open / 4 parked** after filing; parked OC-71/72/73/185,
  historical closures, R1 and DRIFT-01 remain unchanged.

This filing was a documentation/queue update, not a product fix or a new
execution review. Filing verification checked ID uniqueness, all 57 claim
dispositions, existing-row preservation, report links/anchors, counts and
Markdown whitespace. The user subsequently removed the standalone Qwen
source/report files after their actionable content was transferred to the
queue. Dead links were removed; measured evidence, scope limitations and claim
numbers remain in the OC records, with completed repairs documented in this Log.

### 2026-09-12 — reported Canvas/Preview path-letter and display-name mismatch

The user reported a Resampling connection labeled Path L while Preview Results
used Path M and showed Imputation under Path L. Reproduced the exact letter
shift with thirteen preprocessing leaves and a separate Data Preview sink:
Preview submission removes that sink, but Canvas previously counted its
upstream node as consumed and omitted that result branch. One shared graph
filter now drives submission and Canvas branch presentation. The submitted
configuration retains its existing behavior. Tests cover the sink attached to
Imputation and, following the user's clarification, to a different earlier leaf.

The user also requested **Resampling**, without **Node**, in the path label.
Canvas now consults the registered display name before formatting a technical
type name, preserving custom labels/titles and model names. A failing test
confirmed `Path A · Resampling Node` versus the already-correct submitted
`Resampling` name before this repair.

Verification:

- Path-letter regression failed with actual **Path L · Resampling** versus
  expected **Path M · Resampling** before repair. Final focused coverage:
  **19 passed**, including both sink positions and custom name overrides.
- Real HTTP Preview plus PipelineEngine: Path L Imputation returns six rows
  with its missing value filled to -9; Path M Resampling returns eight balanced
  rows, retaining the original missing value. Branch membership, totals and
  bulk node-inspection receipts match those outputs. Relevant backend suites:
  **119 passed** (three existing deprecation warnings).
- Full frontend suite: **2,498 passed in 193 files**. The additional alternate
  sink-position case passed in the subsequent focused run above.
- Desktop 1440px and mobile 390px Playwright checks verify visible connection
  labels, matching Preview buttons, distinct rows when switching branches and
  omission of the inspection sink from the actual submitted request. The two
  new browser cases and existing Preview smoke test passed (**3 total**).
- Frontend lint, CCN, TypeScript/production build, bundle-size limits, targeted
  Python Ruff/format/ty and scoped diff checks passed. Rebuilt static Canvas assets.

This is an additional user-reported fix; the audit queue remains **30 open /
4 parked**. The user subsequently approved committing this fix with the seven
Core fixes below. Delivery checks repeated **118 Core regressions, 56 focused
frontend tests and the real Preview branch-output test**, all passing.

### 2026-09-12 — OC-256/261/265/266/267/268/270: seven verified fixes with parallel agents

After the previous five findings were committed as DCO-signed `61d62726`, the
user requested more than five open findings using subagents. Three agents owned
casting, selection/resampling and modeling; root owned threshold fingerprints,
shared documentation and full verification. Independent reviews checked the
selection/resampling and fingerprint changes. The user reviewed these seven
fixes and approved a DCO-signed commit together with the Canvas path fix above;
no push was requested or performed.

- **OC-265/266:** reproduced valid ISO dates becoming null, categorical boolean
  strings raising SchemaError and Enum strings raising ComputeError. The new
  regression module initially had **15 failures / 21 passing controls**. Only
  selected textual date columns reuse the existing pandas mixed-format parser;
  the resolved Polars datetime dtype, strict/coerce policy, valid dates and
  missingness remain intact. Categorical and Enum boolean labels reuse the
  existing alias table through String expressions. **36 new cases** and the
  existing casting coverage passed (**121 total**). DateFeatures timezone
  semantics (OC-255) remain separate and unchanged.
- **OC-256:** eleven string labels stored as StringDtype or string Categorical
  were inferred as regression, breaking both feature selectors. The dtype
  inference now recognizes those representations before the numeric heuristic.
  Python/Arrow string targets match object targets; numeric categoricals retain
  the ten-value cutoff and explicit problem types remain authoritative. An
  independent review also verified missing string labels and numeric-category
  boundaries. No new target encoding policy was introduced.
- **OC-267:** SMOTE+Tomek ignored `k_neighbors=1` and failed with two minority
  observations because its inner SMOTE used five neighbors. The inner sampler
  now receives the configured neighbors, seed and strategy. Tomek cleanup is
  unchanged: an independent Polars/string-target check exactly matched separate
  SMOTE then TomekLinks(all), reducing 104 generated rows to 92. The two
  selection/resampling regression modules went from **16 failures / 29 controls**
  to **45 passing cases**; related Core coverage passed **266 tests**.
  Canvas had also hidden the neighbor input and accepted zero for this method.
  Its existing controls and validator now include SMOTE+Tomek. UI regressions
  went from **3 failures / 34 controls** to **37 passing tests**, including a
  changed value of 1 reaching the Oversampling payload and zero being rejected.
- **OC-268:** real tuned voting/stacking ensembles returned selected nested
  parameters, but calculator refit and post-tuning CV rebuilt calibrated bases
  with fixed `C=2.0` instead of selected `C=0.01`. Apply tuned base overrides
  after constructing calibration wrappers, preserving fixed base settings,
  seeds and caller-owned mappings. Four public tuning/refit/CV regressions
  prove predictions and log loss match the tuned model, including a selected
  calibration method of isotonic and both flat/nested config shapes.
- **OC-270:** tree/forest tuning rejected valid missing-feature data before
  search although the installed estimators could fit it directly. Admission
  now reads estimator tags instead of a class-name allowlist. A bounded
  baseline check avoids pinning defaults replaced by search axes and retains
  required structural constructor arguments. Candidate fits still enforce
  criterion/monotonic restrictions; numeric infinity and missing targets remain
  rejected, and fold imputation remains available for nonnative models. The
  standalone sklearn >=1.4 contract is retained through the legacy tag API;
  that fallback was tested by simulation, not a separate old sklearn install.
  **23 new missing-feature cases** plus the four ensemble cases passed.
  Actual XGBoost, LightGBM and HistGradientBoosting classifier/regressor smoke
  checks all returned finite tuned scores and predictions. Related modeling
  coverage passed 312 tests before the final capability guards; the final full
  Core run below includes every guard.
- **OC-261:** two copies of one fitted classifier learned positive thresholds
  **0.4803921568627451** and **0.5196078431372549**, disagreed on four of eight
  predictions and still shared a fingerprint. The seal now includes stored
  thresholds through the existing canonical artifact digest. Four regressions
  failed before repair; **10 final cases** pass, covering numeric/string labels,
  refit clearing, dictionary order, pickle protocols and legacy absent state.
  Related pipeline/seal/threshold coverage passed **101 tests**. Untuned
  fingerprints are unchanged; stored fingerprints for pipelines with tuned
  thresholds need recomputing, without retraining the model.

Final root verification: **8,222 Core tests passed / 80 skipped**, three
snapshots passed, 499 existing dependency/degenerate-data/legacy warnings.
Separate relevant backend tests passed **920 cases** and seven snapshots
(121 warnings), covering casting, resampling, feature selection, calibrated
tuning, fold replay, graph/submission leakage and config schemas. Frontend
Vitest passed **2,494 tests in 193 files**. Four Playwright checks passed,
including real control edits, keyboard input, blocked zero-neighbor Preview and
the submitted neighbor value at desktop and mobile widths; mobile editing was
enabled through its existing read-only toggle. Earlier browser failures were
test expectations about mobile mode and the Preview issue-panel presentation.
Full Ruff, configured backend/Core ty, frontend ESLint/CCN, production build,
bundle-size checks and strict MkDocs passed. Generated static assets were rebuilt.
User docs, docstrings, changelog and both audit records describe the changes.

The queue decreases from **37 to 30 open**, with four parked findings unchanged
(OC-71/72/73/185). OC-264 and OC-269 remain open from the latest audit additions.
The unrelated `tmp_eda_shots/` directory and ignored investigation logs are
outside the delivery scope.

### 2026-09-12 — OC-46/248/250/254/263: merged folds, class weights and numeric/container edge cases

At the user's request, verified concrete outputs from the previous five-fix
batch again (**97 regressions passed**) and committed it with DCO sign-off as
`d470b871`; all commit hooks passed. Continued with five more findings using
three implementation agents and root-owned merged-fold work, then independent
cross-reviews. The user reviewed these fixes and requested a signed commit.

- **OC-248:** re-executed the merged LagFeatures/StandardScaler example and
  observed times `[3,1,2]` paired with targets `[100,200,300]` instead of
  `[300,100,200]`. The adapter's existing contract rejects branches that cannot
  preserve positional observations; its static list missed configured sorting,
  LagFeatures `drop_na=True` and ManualBounds filtering. A shared configured
  admission check now rejects these cases in Core and backend reconstruction.
  Safe unsorted/unfiltered lag and rolling branches remain usable. Validated
  configs are copied in full so caller edits cannot introduce unsafe sorting;
  runtime row-count checks reject malformed inputs/branch outputs before
  publishing a merge or replacing the last successful fitted state. Single
  branch adapters still allow sorting/filtering and now flag conditional row
  changes accurately. **21 Core cases and all 12 real backend graph cases
  failed before repair**; five existing-behavior controls passed. Final Core
  coverage includes four further branch-output/state regressions, for **30
  cases**, and all 12 backend cases pass on both engines. Default learned-graph
  scoring fails explicitly; existing `on_leakage='warn'/'ignore'` legacy
  fallback retains its diagnostic code. This enforces the existing merged-fold
  scope; it does not add row-identity joins or change global merge semantics.
- **OC-250:** direct fits already converted nonnative class weights into
  sample weights, but tuning trials and final refits bypassed that policy.
  The shared policy now covers grid, random, both halving variants and Optuna,
  computing weights only from the current fit's post-preprocessing training
  labels. Native class weights remain on their estimator; the serving artifact
  remains the native fitted model. **25 tests failed and five native controls
  passed before repair**; all **65 new cases** pass, including resampling,
  holdout isolation, dictionaries, None sentinels, defaults and unsupported
  estimators. Tuned probabilities match direct balanced training exactly:
  maximum difference **0.0** for GradientBoosting and XGBoost. Differences from
  current unweighted controls are **0.5755483623828201** and **0.66135305**;
  the older audit's XGBoost number used earlier defaults. XGBoost still streams
  ten final-refit callbacks and detaches them afterward. Independent LightGBM
  and calibrated-classifier controls across all five strategies also match
  direct fits exactly, retain native estimator types and unprefixed parameters.
  No explicit sample-weight API was added. Existing models need retraining to
  apply weights previously ignored during tuning.
- **OC-46:** constant numeric features, including constants after imputation,
  published `[NaN, NaN]` PCA ratios and failed strict stdlib JSON serialization.
  The prepared-matrix variance guard now returns the existing unavailable
  contract, `pca_data=None` and `pca_components=None`. The current PCA tab
  already handles that state; no schema or frontend change is needed. A varying
  feature alongside a constant companion still returns ratios `[1.0, 0.0]`.
  **Two public analyzer/strict-JSON cases failed before repair**, then passed
  alongside the varying-input control and existing multivariate tests.
- **OC-254:** LabelEncoder accepted list/NumPy targets at fit but applied
  pandas `.map` or Polars `.clone` methods to those containers. Reuse the
  existing fitted string-label conversion and return native integer Series
  only when encoding the target. `['yes','no','yes']` becomes `[1,0,1]` on both
  engines; native names and pandas indexes survive. Feature-only targets pass
  through unchanged, and missing/unknown semantics plus the mixed-native-engine
  guard remain intact. **14 cases failed and ten controls passed before repair**;
  all **30 final regressions** pass. Independent nullable/datetime/categorical
  target and metadata controls found no remaining issue.
- **OC-263:** the valid antipodal pair from the audit produced Haversine
  intermediary `1.0000000000000002`, causing `sqrt(1-a)` to return NaN.
  Clamp that intermediary to `[0,1]` in both native engines. The reproduced
  distance is now **20015.114442035923 km / 12436.8155228673 mi**. **Four cases
  failed before repair and pass afterward**, covering both units and engines;
  reverse antipodes, quarter-circumference, zero distance, missing inputs and
  source immutability are checked within those cases.

Final validation: full Core pytest passed **8,104 tests / 80 skips**, with
**three snapshots passing** and 499 dependency/degenerate-data/legacy warnings.
Separate backend fold-refit, execution replay and submission/leakage suites
passed **894 tests** (121 warnings). Focused selections passed 267 modeling/
tuning tests, 637 encoding/dispatcher/pipeline tests and 179 profiling/geo tests
(nine optional H3 skips); these selections overlap and are not additive totals.
Full-repository Ruff, configured backend/Core ty, formatting of the 17 changed
Python files, strict MkDocs and diff whitespace checks passed. No model
downloads were needed; offline Hugging Face flags were set for the Core suite.
Independent reviews found no remaining blocking issue. User docs, method
docstrings, changelog and both audit files record the resulting behavior.

The queue decreases from **42 to 37 open**, with **four parked** findings
unchanged (OC-71/72/73/185). The Now and Next tiers are empty again; all seven
OC-264–270 additions remain open. No push was requested or performed.

### 2026-09-12 — OC-35/249/252/258/262: validation alignment, metrics, grouping and restoration

Committed the preceding OC-90/187/192 batch as DCO-signed `f479e196` after
fresh Core/frontend/HTTP regressions, strict docs and all commit hooks passed.
Continued with five independent findings using three implementation agents
and root-owned decomposition work, followed by cross-reviews. Concurrent
audit additions OC-264–270 remain recorded and open; they were not overwritten
by this repair batch.

- **OC-249:** reproduced the reversed RollingAggregate validation case: raw
  labels scored against sorted probabilities selected `0.4117647`, turning
  eight correct predictions into a constant class. Threshold optimization now
  passes both features and labels through the fitted preprocessing and uses
  the transformed labels; the same case selects `0.5` and predicts four zeros
  then four ones. Raw length mismatches are rejected before sorting can hide
  them. **Fourteen cases failed and four passed before repair.** Cross-review
  then caught a compatibility regression for pandas labels with Polars
  features and the reverse. Two added cases failed before normalizing labels
  to NumPy at the paired transform boundary; all **20 cases now pass**.
  Sorting, LagFeatures/ManualBounds filtering, series/list/NumPy targets,
  positional pandas indexes and preserving saved thresholds on failure are
  covered. Independent controls also exercised native/wrapped inputs, fitted
  splitters and feature-target separation. Existing saved cutoffs are not
  changed automatically; rerun optimization for an affected pipeline.
- **OC-35:** binary single-class holdouts emitted non-finite ROC coordinates
  despite the earlier multiclass guard. Omit the undefined ROC curve while
  retaining finite PR output; scalar ROC AUC was already sanitized away.
  **Four regressions failed before repair and pass afterward**, covering
  positive-only/negative-only holdouts with numeric and string labels. Rerun
  evaluation to refresh previously stored reports.
- **OC-252:** weighted PR-AUC inferred its probability class axis from the
  holdout, so a three-class estimator and a two-class validation partition
  failed every trial. The scorer now receives the estimator and aligns
  targets to its trained classes; all five search strategies and the
  multiclass `pr_auc` alias score the reproduced case at **1.0**. A binary
  negative-only holdout now scores **0.0**, keeping the trained positive class
  instead of relabeling the remaining class as positive. **Seventeen failed
  and six passed before repair; all 23 pass afterward.** Missing first,
  middle, last and multiple classes are covered. Independent grid/halving/
  Optuna controls with fold-fitted StandardScaler passed both inner CV and
  explicit sparse holdouts. OC-251 filtering and OC-253's stock F1 policy
  remain separate open findings.
- **OC-258:** grouping by `value` raised DuplicateError and grouping by
  `ratio` replaced category identities with aggregate fractions. A label
  expression now uses a separate name in the grouped result without casting
  the source measure. Numeric grouping columns can also be their own measure.
  **Fifteen regressions failed before repair; all 30 now pass**, covering all
  aggregate modes, reserved/control names, null filters, ratios and unchanged
  source data. Three real HTTP cases preserve null and literal `Unknown`
  drill-down for `group`, `value` and `ratio`. Independent review passed an
  additional 25-case split/measure-name cross-product. OC-192 remains intact;
  the response shape and Canvas contract are unchanged.
- **OC-262:** pickle and Core JoblibModelSerializer dumped wrappers but hit
  RecursionError during restoration because `__getattr__` recursively looked
  up absent `_df`. Direct object lookup now terminates those pre-state probes
  with AttributeError and retains native delegation once state exists.
  **All 20 regressions failed before repair and pass afterward**, including
  both engines, pickle protocols 4/5, nested containers, schema/null/index
  fidelity, native methods and shallow copies. Four artifacts created before
  repair loaded afterward without rewriting. Delegated deep-copy behavior is
  unchanged. Independent review found no remaining issue.

The latest audit also identified an old recommendation test fixture missing
the NumericStats fields required after OC-51. Reproduced its AttributeError
and replaced the incomplete SimpleNamespace with real ColumnProfile/
NumericStats values, retaining both threshold branches and asserting the
domain-valid suggestion. This is test maintenance, not a sixth product fix.

Focused verification passed **227 pipeline/threshold tests**, **442 evaluation/
tuning tests**, **146 engine/serialization tests**, **73 decomposition/fixture
tests**, and **132 separate backend tests**. These selections overlap; they
are not an additive total. User docs, method docstrings, changelog and both
audit files are updated. The frontend implementation and response shape do
not change in this batch.

Final verification after the mixed-label compatibility repair: full Core
pytest passed **7,972 tests / 80 skips**, with **three snapshots passing**
and 491 dependency/degenerate-metric/legacy-alias warnings. The earlier failing
recommendation fixture now passes in the full suite. Full-repository Ruff,
configured backend/Core ty, formatting for all 13 changed Python files, strict
MkDocs and diff whitespace checks passed. Independent review confirmed the
mixed-label blocker resolved and found no remaining actionable blockers.
No network model downloads were needed (`HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`). The user subsequently authorized committing this
batch after reviewing concrete output examples; its 97 regression cases
passed again before staging.

The queue has **42 open / 4 parked**: the concurrent review grew it from 40 to
47, then this batch closed five. OC-71/72/73/185 remain parked.

### 2026-09-12 — OC-264–270 filed: seven more reproduced Core defects

The continuation adds seven P2 findings to the
[live queue](opus_core_analysis-open_queue.md), bringing its recorded status to
**47 open / 4 parked**. Existing parked decisions and prior fixes are unchanged.
All seven symptoms were executed against current source and independently
reproduced by the main reviewer. Paths below are relative to `skyulf-core/skyulf/`.

- **OC-264 — sequential transformation fitting:** use
  `GeneralTransformationCalculator` and its applier on
  `x=[1.,2.,4.,8.,16.,32.]`, with rules
  `{'column':'x','method':'log'}` then
  `{'column':'x','method':'yeo-johnson','standardize':True}`.
  Fitting/applying both rules together yields mean `-0.9953748450991124`;
  fitting the second rule on the first rule's applied output yields
  `4.070817756958907e-16`. Both pandas and Polars reproduce.
  `preprocessing/transformations/general.py:225-228` fits the power rule on
  original `X`, although apply receives the intermediate logged values.
  Preserve sequential fitting semantics for overlapping rules; test equivalence
  to separate nodes. OC-27's standardization-setting repair remains valid.
- **OC-265 — datetime cast loses valid dates:** fit/apply `CastingCalculator`
  and `CastingApplier` with `columns=['x'], target_type='datetime'` on
  `x=['2024-01-01','2024-06-15']`. Pandas produces both expected timestamps;
  Polars produces `[None,None]` under default coercion.
  `preprocessing/casting.py:221` uses a generic String-to-Datetime cast instead
  of parsing. Cover valid strings and invalid-value coercion/strict behavior;
  these ordinary timezone-free inputs are separate from OC-255.
- **OC-266 — categorical boolean cast fails:** apply Casting with
  `columns=['x'], target_type='boolean'` to a Polars Series containing
  `['true','false']`. String succeeds with `[True,False]`, Categorical raises
  `SchemaError: expected String, got cat`, and
  `pl.Enum(['true','false'])` raises a categorical-to-Boolean `ComputeError`.
  Pandas categorical input succeeds. The explicit categorical branch at
  `preprocessing/casting.py:177-179` calls string operations at line 109 without
  first converting the expression to String; Enum misses that branch entirely.
  Preserve the existing text-to-boolean vocabulary and coercion contract while
  covering both categorical representations. OC-58's numeric boolean repair
  concerns a different input family.
- **OC-267 — combined SMOTE ignores neighbor count:** fit/apply
  `OversamplingCalculator` and its applier to
  `X.x=[0.,1.,2.,3.,4.,10.,11.]`, `y=[0,0,0,0,0,1,1]`, with
  `k_neighbors=1, random_state=42`. Method `smote` produces ten rows and
  `{0:5,1:5}`; method `smote_tomek` raises
  `Expected n_neighbors <= n_samples_fit, but n_neighbors = 6,
  n_samples_fit = 2, n_samples = 2`.
  The control `SMOTETomek(smote=SMOTE(k_neighbors=1,random_state=42),
  random_state=42)` succeeds with the same balanced counts.
  `preprocessing/resampling.py:221-222` omits the configured inner SMOTE,
  although the calculator stores the setting at line 292.
- **OC-268 — calibrated ensemble loses tuned parameters on replay:** build
  pandas inputs from `make_classification(n_samples=100,n_features=4,
  random_state=7)`. Prepare `VotingClassifierCalculator` with
  `base_estimators=['logistic_regression'], calibrate_base_models=True,
  calibration_cv=2, voting='soft', n_jobs=1`. Run `TuningCalculator.fit` with
  `strategy='grid', cv_folds=2` and the single-value search space
  `{'logistic_regression__estimator__C':[0.01]}`. The returned tuned estimator
  has `C=0.01`; `calc.fit(X,y,{'params':result.best_params})` restores `C=1.0`.
  Predictions differ by up to `0.4102064767291098` in probability.
  The invalid `estimator` parameter warning arises because
  `modeling/ensemble.py:299-300,468-473` routes nested parameters to the bare
  base estimator before wrapping it for calibration. Preserve the selected
  configuration across tuning, subsequent fit and CV; this is separate from
  OC-206's shared-configuration mutation.
- **OC-269 — pruning is never activated:** build the Optuna searcher with
  `TuningConfig(strategy='optuna', n_trials=2, search_space={'alpha':[0.001]},
  strategy_params={'pruner':'hyperband'})`, `SGDClassifier(random_state=7)`,
  `KFold(2)` and scoring `accuracy`. Fit the same 100-row classification input
  used for OC-268. The study uses `HyperbandPruner`, but
  `search.enable_pruning` is `False`, and trial intermediate values are
  `[{},{}]`. `modeling/_tuning/strategies/optuna.py:240-252` omits the enabling
  flag. SGD supports `partial_fit`, so this also fails for a compatible model.
  Implement capability-aware pruning; universally enabling it would break
  estimators without incremental fitting.
- **OC-270 — tuning rejects natively supported NaN:** use
  `X.x=[0.,1.,NaN,3.,4.,5.,6.,7.,8.,9.,10.,11.]`,
  `X.z=[0.,0.,0.,0.,1.,1.,1.,1.,0.,0.,1.,1.]`, and
  `y=[0,0,0,0,1,1,1,1,0,0,1,1]`. A direct
  `RandomForestClassifierCalculator.fit` with
  `params={'max_depth':2,'n_estimators':5,'n_jobs':1}` succeeds and predicts
  all twelve labels correctly. A two-fold grid search over single-value lists
  of those same parameters raises an imputer-required ValueError before
  dispatch, despite the estimator's `input_tags.allow_nan=True`.
  ExtraTrees and DecisionTree reproduce with their supported parameters.
  `modeling/_tuning/engine.py:318,404-416` relies on a class-name allowlist
  that omits these supported estimators; validate estimator capabilities while
  retaining rejection for models that cannot accept missing values.

Verification: inline reproductions above plus the following existing focused
tests, **263 passed, 1 warning**:

```powershell
.venv\Scripts\python.exe -m pytest `
  skyulf-core/tests/integration/test_transformations_general.py `
  skyulf-core/tests/integration/test_casting.py `
  skyulf-core/tests/unit/test_resampling.py `
  skyulf-core/tests/integration/test_sentence_embedder.py `
  skyulf-core/tests/unit/test_modeling_ensemble_gaps.py `
  skyulf-core/tests/unit/test_tuning_engine_failure_branches.py `
  -q --tb=short -p no:cacheprovider `
  --basetemp="$env:TEMP\skyulf-core-audit-followup-20260912"
```

This continuation sampled transformations, casting, resampling, scaling,
outliers, splitting, calibrated ensembles, nested CV, Optuna and vectorization.
It is not proof that all remaining behavior is correct. A reproduced empty-batch
SentenceEmbedder failure was not filed because its expected empty-input contract
was not established. H3 was unavailable locally and was not execution-verified.
This entry records findings and evidence; it does not claim implementation fixes
or another full-suite run.

### 2026-09-12 — OC-248–263 filed; OC-35/46 reopened and OC-47 extended

At the user's request, recorded the preceding read-only Core review in the
[live queue](opus_core_analysis-open_queue.md). The review inventoried 190 Python
files and covered all main module groups, with deeper execution checks on
preprocessing, modeling/tuning, evaluation, profiling and pipeline/engine seams.
H3, sentence embedding, some model defaults and several tuning internals received
more limited inspection; this was not a line-by-line certification of every file.

**Filing:** sixteen new findings, OC-248–263 (3 🟠 / 13 🟡), two reopened IDs,
and additional evidence on already-open OC-47. Open rows for OC-35/46 move back
out of this archive; their earlier fix entries remain intact. The current
OC-90/187/192 repairs remain closed. Counts change from **22 open / 4 parked**
to **40 open / 4 parked**, excluding R1 and DRIFT-01 planning work. No
implementation, regression-test or release-note changes belong to this filing.

All findings below were reproduced by executing inline Python against the
working source with `.venv/Scripts/python.exe`; no model downloads were needed.
The queue rows retain the affected files, current scope, effort and status.

- **OC-248 — merged branch alignment:** use times `[3,1,2]`, values
  `[30,10,20]`, targets `[300,100,200]`, one sorted LagFeatures/RollingAggregate
  branch and one StandardScaler branch. Merged times remain `[3,1,2]` but the
  first branch supplies sorted targets `[100,200,300]`. Index resets discard
  observation identity. LagFeatures with `drop_na=True` and ManualBounds also
  evade the unsafe-branch screen and yield unequal feature/target counts.
  Individual applier alignment repairs in OC-163/165 remain valid.
- **OC-249 — threshold label alignment:** fit LogisticRegression with `C=0.001`
  on times/values `0..19`, targets ten zeros then ten ones, after RollingAggregate
  with window 2, `min_periods=1`, sorted by time. Tune on reversed times `13..6`
  and labels four ones then four zeros. The public method uses unsorted labels
  against sorted probabilities and selects `0.411765`; explicitly aligning the
  labels selects `0.5`. Tuned predictions become eight ones instead of four
  zeros then four ones in output order.
- **OC-250 — ignored class weights:** generate 200 classification rows with five
  features, three informative, 90/10 class weights, no label noise and seed 17.
  Compare direct plain/balanced fits with single-candidate grid tuning using
  ten trees, depth 2, seed 7 and two folds. GradientBoosting's tuned probabilities
  differ from the plain fit by exactly zero and from the balanced fit by up to
  `0.5755483623828201`; XGBoost controls give zero and `0.63141084` respectively.
  The result still lists `class_weight="balanced"`. Candidate construction and
  final refit bypass the calculator's sample-weight conversion.
- **OC-251 — filtered fold scoring:** repeat `[0,1,2,3,4,5,6,7,8,9,10,100]` ten
  times as feature x, with target `2*x+1`, and use an IQR fold adapter. Ridge
  tuning with alpha 1, three folds and R2 succeeds under grid at
  `0.9999981339595826`; halving_grid and Optuna fail all trials. The fitted
  FoldAwareModelStep returns 110 predictions for 120 labels; its prediction
  path transforms X without carrying the scorer's y through the filter.
- **OC-252 — holdout PR-AUC classes:** fit a three-class logistic model on
  x=`0..59` and 20 observations per class, then use the first 40 rows as explicit
  validation data with `metric="pr_auc_weighted"`. Three probability columns
  meet a target binarized from only the two observed holdout classes, failing
  all trials. Binarizing against the model's classes succeeds with weighted
  average precision `1.0`; the tuning scorer needs that class axis.
- **OC-253 — F1 positive-class consistency:** train on x=`0..39`, class 1 for
  the first 30 rows and class 2 for the remainder. Holdout x values
  `[0,10,20,25,29,30,35,39]` with labels `[1,1,1,1,1,2,1,2]` give tuning F1
  `0.9090909090909091`, matching `pos_label=1`; `pos_label=2` gives `0.8`.
  Evaluation and threshold tuning use the latter class. The tuning exception
  that retains sklearn's default when label 1 exists is documented, so agree
  on the common contract before changing it.
- **OC-254 — LabelEncoder target containers:** fit/apply `(X, y)` with three
  numeric feature rows and NumPy or list labels such as `["yes","no","yes"]`.
  Fit accepts them; pandas apply assumes `.astype(...).map(...)` and Polars
  apply assumes `.clone()`. Polars clones y even when only features are encoded.
- **OC-255 — timezone/DST extraction:** the same `+02:00` ISO timestamps yield
  local day/hour features on pandas and UTC features on Polars. A two-row
  sample across `2024-03-30T12:00:00+02:00` and
  `2024-03-31T12:00:00+03:00` raises pandas' non-datetimelike `.dt` error;
  Polars succeeds. Pick a shared timezone policy and cover mixed offsets.
- **OC-256 — categorical selector inference:** repeat 11 string class labels
  four times and run UnivariateSelection/ModelBasedSelection with automatic
  problem type. Object labels work, but pandas StringDtype/Categorical labels
  enter regression and fail with division/conversion errors. Explicit
  `problem_type="classification"` avoids the faulty inference.
- **OC-257 — Enum drift:** an Enum with allowed labels a/b changes from 100 a
  observations to 100 b observations. Both inputs keep the same dtype, yet
  the report has zero drift and no column metrics. String controls report
  categorical PSI `10.543651559430593`; this is not OC-47's dtype-change case.
- **OC-258 — decomposition aliases:** group `["a","a","b"]` and sum amounts
  `[1,2,3]`. A grouping column named `value` raises DuplicateError, while
  `ratio` overwrites both names and drill-down filter values with `0.5`.
  A grouping column named `group` works. Reread and reproduced against the
  concurrent OC-192 missing-group repair; its null-identity fix remains valid.
- **OC-259 — unsupported dtype fallback:** public EDA analysis of a two-row
  Decimal amount column raises SchemaError when the fallback Text path applies
  `str.len_bytes()`. Time/List columns hit the same path; Float64 controls work.
  The earlier Null/Enum semantic-type repair in OC-191 does not cover these.
- **OC-260 — time-series plot alignment:** analyze 1000 hourly timestamps, one
  complete metric and another with only its first 500 values present. Native
  profile construction succeeds, but public `EDAVisualizer.plot()` raises an
  x/y mismatch for shapes `(1000,)` and `(500,)`. Every date is appended while
  missing metric values are omitted. This differs from OC-193's missing dates.
- **OC-261 — fingerprint thresholds:** copy one fitted logistic pipeline twice
  and optimize against two different labelings of the same eight validation
  features. Stored positive thresholds `0.480392` and `0.519608` produce
  different predictions on four rows under `use_tuned_thresholds=True`, but
  fingerprints stay equal because the stored thresholds are not hashed.
- **OC-262 — wrapper restoration:** wrap a two-row pandas or Polars frame with
  EngineRegistry, dump through JoblibModelSerializer and load it again. Both
  raise RecursionError through `__getattr__` before `_df` exists. Direct
  `pickle.loads(pickle.dumps(wrapper))` reproduces the same failure; this
  evidence concerns dataframe wrappers, not every saved model.
- **OC-263 — antipodal distance:** both engines return NaN for the valid points
  `(-89.91000888888888,0)` and `(89.91000888888888,180)`, instead of the mean-Earth
  half-circumference `20015.114442035923 km`. Haversine roundoff makes the
  intermediate slightly greater than one before `sqrt(1-a)`.

**Existing IDs with current evidence:**

- **OC-35 reopened:** the binary model from OC-253 evaluated on the first three
  holdout rows sees only class 1 and publishes ROC coordinates
  `[(0.0, NaN), (0.3333333333333333, NaN), (1.0, NaN)]`. The multiclass guard
  repaired in the earlier entry does not protect this binary path.
- **OC-46 reopened:** public analysis of ten rows with x=1 and z=2 publishes
  PCA explained-variance ratios `[NaN, NaN]`.
  `json.dumps(profile.model_dump(mode="json"), allow_nan=False)` raises
  ValueError. Preserve earlier finite-value repairs and cover PCA output too.
- **OC-47 extended:** reference integers `[0]*50+[1]*50` versus current floats
  `[0.9]*50+[1.9]*50` report zero distances and zero drift. Casting current
  values to the integer reference loses their fractional parts. Keeping the
  reference Float64 yields drift with normalized Wasserstein `1.8`. This
  supplements the original all-null-after-casting disappearance in the same ID.

**Verification from the preceding audit, not a new full-suite run for this
documentation update:** Core pytest returned **7796 passed / 80 skipped /
1 failed**, with three snapshots passing. The single failing test,
`test_patch_coverage_core_round5.py::TestBalanceRecommendationDirections::test_skewness_threshold_both_directions`,
also failed in isolation: its SimpleNamespace fixture lacks `min` and other
NumericStats fields used by the updated skewness recommendation. A real
ColumnProfile/NumericStats payload succeeds, so record this as a test-maintenance
follow-up rather than another demonstrated product bug. Current decomposition
missing-group tests passed **12/12**, and Core Ruff/Ty checks passed. Inline
reproductions establish the findings above independently of those green gates.

### 2026-09-12 — OC-90/187/192: split warnings, LightGBM sampling and missing-group navigation

Continued the unparked queue after signed commit `81f35f07` closed the preceding
OC-18/30/48 batch. Each repair began with a public-path reproduction and checked
the actual Canvas payload before editing another layer.

- **OC-90:** a node config using `stratify_col="target"` silently dropped that
  key. With 200 rows, 20% positives, 20% test/validation and seed 42, both
  engines produced class proportions `[0.2, 0.175, 0.225]`; the supported
  `stratify=True` setting produced `[0.2, 0.2, 0.2]`. The source audit explicitly
  recommended a warning, so fitting now names ignored public keys and lists
  the six supported settings without changing the artifact or splitting
  behavior. `type` and private underscore-prefixed routing metadata remain
  quiet. The current Canvas serializer also forwarded `definitionType`,
  labels and titles; it now selects the six supported fields before shared
  routing metadata is attached. **Eight Core regressions and two frontend
  payload tests failed before the fix**; all ten Core cases and both frontend
  cases pass afterward.
- **OC-187:** reproduced the original 300-row, six-feature, seed-7 LightGBM
  case with 20 trees and one worker. Changing `subsample` from 1.0 to 0.4
  produced identical predictions; explicit frequency 1 produced differences
  of **0.16583687788133306** for classifier probabilities and
  **85.66120232221425** for regression. Grid-search scores were also identical
  before repair. A blanket frequency of 1 breaks GOSS, so new estimator
  subclasses resolve the automatic `None` frequency at the native parameter
  handoff: 1 for ordinary bagging, 0 for effective GOSS. Constructor state stays
  unchanged across clone/set_params, and explicit frequencies/native aliases
  keep LightGBM's precedence. Uppercase GOSS values are supported too. Native
  missing-feature support remains admitted by the tuning validator; imports
  still succeed with LightGBM unavailable. **56 regressions** cover both model
  tasks, all five search strategies, default-space boosting modes, actual
  training effects, final refit, explicit overrides, cloning and saved models.
  Existing fitted models retain predictions; retrain to apply previously
  ignored fractions, or use explicit frequency 0 to keep the old policy.
- **OC-192:** the original `group=['a', None, 'b']`, `v=[1,2,3]` sum split
  displayed an `Unknown` bucket worth 2, but filtering `group == 'Unknown'`
  returned 0. Further reproduction showed missing rows and a literal
  `Unknown` category merging to 9 rather than separate groups worth 6 and 3.
  Grouped results now retain null identity in an additive `filter_value`
  field; `== null` / `!= null` select/exclude missing rows. The tree uses this
  identity for requests, selection, connection anchors and cached navigation,
  labels missing groups explicitly, and falls back to names for older rows.
  Numeric legacy `Unknown` filtering and float-NaN normalization remain.
  Nullable filters are scoped to decomposition; other EDA filter contracts
  stay unchanged. **Nine Core cases and the UI regression failed before repair**;
  real HTTP and browser roundtrips now preserve distinct null/string values.

Validation: split/schema/pipeline/leakage/decomposition Core suites passed
**1,313 tests** (two existing skips); modeling/tuning suites passed **308 tests**.
Separate backend split, EDA HTTP, schema-preview and graph-leakage suites passed
**68 tests**. Full Vitest passed **2,493 tests in 193 files**; after the build
caught an unsupported test-query option and an overly broad nullable filter
type, the corrected decomposition/sidebar selection passed **17 tests** and
TypeScript passed. Desktop/mobile Playwright passed **2 tests**, including
keyboard Enter activation, selected-state distinction and real JSON request
inspection. Full frontend lint and CCN gates, full-repository Ruff, configured
ty, Python formatting, frontend production build and bundle-size gates passed.
Generated `static/ml_canvas` assets, reference/API docs and changelog are updated.
Independent reviews found no actionable blockers. Existing dependency/legacy
alias warnings remain; no parked work was resumed.

The queue now has **22 open / 4 parked**; OC-71/72/73/185 stay parked.

### 2026-09-12 — OC-18/30/48: generated names, datetime outputs and empty-data checks

Continued the unparked queue after signed commit `7947d5af`; the calibrated
Base Estimator multiselect stays unchanged. Reproductions used public
Calculator/Applier and expectation APIs with native and wrapped Pandas/Polars
frames. Canvas encoding and datetime payloads already forwarded the relevant
settings correctly, so this batch changes Core behavior without frontend edits.

- **OC-18:** one-hot/dummy generated `city_a` names duplicated an existing
  Pandas column; Polars one-hot raised `DuplicateError`, while dummy encoding
  silently replaced the retained values. MissingIndicator and multiclass
  TargetEncoder also overwrote retained output names, including cross-fitted
  target-encoder training. A shared validator now rejects names duplicated
  within generated outputs or shared with retained inputs at fit/apply. Names
  belonging only to dropped sources remain available. One-hot and dummy
  construction reads all original sources before output assembly, fixing lost
  indicators and corrupted secondary source values in those valid cases.
  The dedicated regressions went from **78 failed / 46 passed** to **124 passed**.
- **OC-30:** a Canvas-shaped datetime operation with
  `output_column="custom_year"` ignored that name and replaced an existing
  `dt_year=99` with `2024` despite overwrite being disabled. Both engines now
  honor exact single-result names, feature/source prefixes for multiple
  results, optional default-name prefixes and the existing numbered collision
  policy. The configured source count keeps remaining names stable when an
  inference source is absent; suffixes still depend on the current input
  schema, as for other Feature Generation operations. The node's existing
  schema-prediction fallback remains `None`. **68 regressions** cover native
  and wrapped inputs, replay, overwrite, repeated features, runtime columns,
  and downstream arithmetic/fitted group aggregation.
- **OC-48:** all three row-level expectations returned success for zero-row
  datasets on both engines. They now raise `ExpectationError` by default, with
  keyword-only `allow_empty=True` for intentionally empty data. Requested
  columns are still validated first; column-existence checks stay schema-only.
  An additional probe found empty/all-null Polars `Null` series invoking an
  unsupported comparison. Range checks now return after null removal when no
  observed values remain, preserving their documented null-ignore behavior.
  **81 regressions** distinguish empty rows, absent columns, all-null values,
  invalid nonempty data, zero-width Pandas frames and native Polars execution.

Validation: the combined encoding, feature-generation, expectation, artifact,
schema, preprocessing-pipeline and leakage Core selection passed **2,080 tests**
and **3 snapshots** (11 existing warnings). Separate backend encoding,
preprocessing, schema-preview/graph and operation-leakage suites passed
**925 tests** (two existing deprecations); the two existing root feature-generation
suites also passed **3 tests**. Full-repository Ruff, all 13 changed Python files'
format checks, and the complete configured ty gate passed. Independent reviews
found no actionable issues. Reference docs, changelog and source findings record
the behavior changes: rename conflicting encoder inputs, refit pipelines affected
by datetime naming/overwrite changes, and explicitly allow expected empty data.

The queue now has **25 open / 4 parked**. OC-71/72/73/185 remain parked.

### 2026-09-12 — OC-66 follow-up: Canvas calibrated base selection

The user reported Optuna rejecting `model__estimator__base_estimator` after
selecting Random Forest for a calibrated classifier. This reopened an uncovered
path of OC-66: the 2026-09-04 fix handled a flat structural selection, and the
OC-101 seed regression manually prepared that flat shape. Neither covered the
Canvas multiselect's `tuning_config.search_space.base_estimator` list.

Reproduced with the real backend `_prepare_tuning_config`, a StandardScaler
fold adapter and one Optuna trial: `method="isotonic"`, `cv=3` and
`base_estimator="random_forest"` reached the nested estimator's `set_params`,
which rejected the last key. Grid search instead silently filtered that key
and trained logistic regression. Thus adding scaling cannot repair this
parameter-routing error.

The calibration-specific estimator now retains `base_estimator` as a cloneable
symbolic parameter. Each fit resolves the selected factory before applying the
seed, so searcher clones, fold pipelines and the winning refit all use the
requested model family. The factory is shared with the existing basic and flat
tuning paths, preserving their choices and fallback behavior. Multiple selected
bases remain valid search candidates; trial and best-parameter reports retain
JSON-safe names without internal pipeline prefixes. No frontend or backend
production change was needed.

Added **12 Core regressions**, all observed failing before the repair: every
search strategy with and without fold preprocessing, observed trial/refit model
types and seeds, multi-base comparisons, cloning, changing candidates, source
configuration preservation and pickle round trips. Added **two backend
regressions** using Canvas-shaped node config and real Grid/Optuna training;
both failed before the fix and now pass. Together with the OC-101 checks, the
calibration module passes 33 cases.

Final related verification: **395 Core tests and 171 backend tests passed**,
covering calibration, hyperparameters, all tuning strategies, fold replay and
node runners. Scoped Ruff lint/format, the configured full ty check and an
independent review passed. Updated model docs, dynamic field help, changelog
and both source-audit notes. OC-66 keeps its existing archive row, now marked
completed for the actual Canvas path; queue remains **28 open / 4 parked**.

### 2026-09-12 — OC-51/101/140: recommendations, calibration seeds and numeric rules

Committed OC-24/59/60 as `9663132a` after 131 fresh focused tests, a strict
documentation build and all applicable commit hooks passed. Continued with
three independent findings, two implementation agents and a separate review
of each change.

**OC-51 — fixed.** Public analysis of `[-100, 1, 2, 3, 4, 5, 6, 7]` and
zero-containing skewed data recommended Log/Box-Cox and simultaneously claimed
modeling readiness. An imbalanced string target also produced both Keep and
Resample. Log/Box-Cox advice now requires known positive values and right skew;
non-positive values, left skew and unknown domain evidence use Yeo-Johnson.
The clean message is evaluated after target advice and is suppressed by Drop,
Impute, Transform, Encode or Resample. Its text describes these checks rather
than certifying readiness. New public-profile and serialized-payload tests
cover those actions, signed/zero/positive domains, incomplete/nonfinite stats,
thresholds, source preservation and the allowed Info+Keep case: **20 failed
before the repair, 22 passed afterward**. The existing Insights tab displays
the supplied recommendation text; no frontend change was required.

**OC-101 — fixed.** A configured calibrated random forest seed of 7 was
dropped as unsupported, and its base estimator still used 42. The tuning path
constructs and clones `model_class` directly, so a calculator-only change would
leave tuning broken. A small calibration-specific sklearn subclass accepts the
seed and applies it to a cloned supported base estimator at fit. It covers
ordinary training, search clones and final refitting without changing generic
wrappers. Default 42, explicit 0 and None, deterministic bases and integer CV
semantics are preserved. Dynamic hyperparameter text now describes the actual
control. **17 initial regression failures** led to **21 passing new cases**,
including fitted fold seeds, repeatable/different forest predictions, caller
preservation on success/failure, clone/set_params, public tuning and pickle
round trips. Existing fitted artifacts keep their predictions; retraining
applies previously ignored custom seeds.

**OC-140 — fixed.** With Canvas-shaped `mode="negative_to_nan"` on
`["hello", "-3", "2"]`, pandas produced `[NaN, NaN, 2.0]` while Polars raised
a numeric comparison error. Active numeric operations now reject selected
nonnumeric columns consistently at fit and apply, naming the columns and
requiring explicit conversion. Apply validation also covers inference dtype
changes. A local numeric selector excludes pandas durations from automatic
selection, matching Polars. Inactive operations remain no-ops. Four native/
wrapped engine shapes cover rules, infinity flags, temporal/text/category/bool
inputs, Decimal/nullable numeric values, empty selections and source/target
preservation. **59 original failures**, two duration failures and five wrapper
return-type failures were observed during development; the final new module
passes **116 cases**. The wrapper regressions ensured no-op handling preserves
the dispatcher's native pandas output and wrapped Polars output. Canvas already
restricts selection to numeric columns and passes the config through.

Final combined verification: **1,566 Core tests passed** across profiling,
classification, tuning, cleaning, registry and artifact/schema coverage;
**79 backend tests passed** across EDA, frontend-node contracts and cleaning/
preprocessing integration. Scoped Ruff lint and format checks, the configured
full ty check, and independent reviews of all three changes passed. User docs,
changelog and source findings were updated. Queue: **28 open / 4 parked**.

### 2026-09-12 — OC-24/59/60: parallel preprocessing audit closure

After committing OC-121 as `2ba4c67c`, investigated three independent findings
with two implementation agents and a separate review of the combined change.

**OC-59 — fixed.** Reproduced differing numeric profile coverage: pandas
reported Int8/Int16/unsigned columns but excluded binary, constant and entirely
missing columns; Polars did the reverse for those cases. Both paths now use
shared dtype-based numeric selection, because these columns remain useful in a
diagnostic profile regardless of cardinality. The pandas path excludes timedeltas
from its number selector, matching the Polars temporal boundary. Decimal
conversion and the read-only applier remain intact. Six new public regressions
failed before the repair and pass afterward, covering all ten existing integer/
float dtypes, Decimal, nullable/empty inputs and native/wrapped frames.

**OC-60 — fixed.** With fitted edges `[0, 5, 10]`, Polars left missing and
out-of-range bins null even when `missing_strategy="label"` selected a sentinel;
pandas emitted the configured label. The Polars expression now widens labeled
outputs to String and fills missing bins, including null, NaN and held-out values
outside the fitted edges. String output applies even to complete/empty batches
so concatenation keeps a stable schema. Pandas retains its existing numeric/text
object values, and default `keep` behavior is unchanged. The 67-case regression
matrix went from **25 failed / 42 passed** to all passing; it covers ordinal,
bin-index, range/custom labels, boundary inclusion, empty/default/existing labels,
drop/source/target preservation and wrappers. Canvas exposes neither missing
option, so the existing Core configuration needed no frontend change.

**OC-24 — verified already fixed.** The public repair dates to `f12dde9f8`
(2026-09-08): fitting includes missing keys as a training group, replay uses its
stored aggregate and normalizes Polars null/NaN keys, and the public applier
rejects artifacts without a fitted mapping. Training keys `[1, null, NaN]` with
values `[1, 2, 5]` produce `[1, 3.5, 3.5]` on both engines. Held-out keys
`[null, NaN, 1, 2]` reuse `[3.5, 3.5, 1, missing]` independently of their values.
Missing keys absent from training remain missing during inference. Forty added
cases cover all seven aggregates, native/wrapped inputs and the unfitted-artifact
guard. The retained private unfitted compatibility helpers still differ, but
their direct callers are tests; the public path cannot reach that behavior.
They were left unchanged. The editor/resolver supports one grouping column,
so this closure does not introduce multi-key grouping.

Combined verification: **2,169 Core tests passed / 2 skipped**, plus **65 backend
catalog, data-service and preprocessing tests passed**. Ruff, formatting and full
configured `ty check` pass; independent review found no blockers. Docs and
changelog describe the profile coverage, missing-label dtype choice and existing
fitted group semantics. The live queue now has **31 open / 4 parked** findings.

### 2026-09-12 — OC-121 fixed: Enum text auto-selection

After committing OC-120 as `8425f56f`, reproduced the adjacent dtype finding
with `status = [" YES ", "No!", null, "Maybe"]`. Polars Enum auto-detection
returned no status column; both TextCleaning and AliasReplacement left it
unchanged. Equivalent pandas Categorical input was selected and transformed.
Explicitly selecting status already worked on both engines.

The shared text selector now recognizes `isinstance(dtype, pl.Enum)` alongside
the existing text dtypes. No node transformation or frontend changes were needed.
Enum selection uses the declared dtype, so it also includes empty/all-null
columns. Text cleaning and alias mapping keep their existing string output,
null preservation, target exclusions and explicit-empty-selection no-op.

Verification:

- Before the fix, the new matrix had **10 failed / 26 passed**, isolating the
  Polars native/wrapped automatic-selection failures from the working explicit
  and pandas controls.
- **341 tests passed**, including all 36 new Enum cases and the existing text,
  alias, helper, pipeline, engine-parity and Decimal regressions.
- Tests exercise fit/apply and held-out Enum vocabularies, preserve source frames
  and untouched columns, and verify empty/all-null dtype discovery.
- Ruff, formatting and full configured `ty check` pass; focused review found no
  concrete defects or scope concerns.

Docs and changelog explain the new selection and refitting previously saved
artifacts. OC-121 moves into the archive; the live queue is **34 open / 4 parked**.

### 2026-09-12 — OC-120 fixed: Decimal numeric selection and processing

After committing OC-113 as `4fea1cca`, reproduced OC-120 with Decimal prices
on native and wrapped pandas/Polars frames. Both numeric selectors omitted
prices. The original explicit StandardScaler crash was already repaired;
Robust/MinMax/MaxAbs still raised Decimal/float arithmetic errors on pandas.
Winsorize fitted bounds but skipped Decimal values on both engines, Polars
IQR/ZScore called unsupported Decimal `is_nan`, KNN/Iterative fitting failed
on object NumPy arrays, and binning could silently omit the output column.

Numeric detection now recognizes parameterized Polars Decimal dtypes and
homogeneous non-missing pandas Decimal objects without accepting numeric
strings or mixed string/Decimal columns. Binary/constant/all-missing exclusions,
target exclusion and explicit empty scaler selections remain intact. Numeric
boundaries convert only selected Decimal columns to float64, including pandas
`pd.NA`; original frames and unselected columns are unchanged. Mean/median
imputation remains numeric for downstream nodes, outlier filters preserve null
rows and matching targets, and DatasetProfile emits numeric Decimal statistics.
Schema promotion also recognizes Decimal dtype labels.

Verification:

- The initial public-path regression matrix reproduced **67 failures / 10
  passes**. Added missing-sentinel cases reproduced seven further failures
  before repairing fit/apply conversion boundaries.
- **110 Decimal regressions pass**, covering four scalers, power transforms,
  imputers, outliers, binning, feature selection/generation, profile statistics,
  native/wrapped inputs, Arrow Decimal missingness and precision/scale variants.
- Broader Core preprocessing, utility, engine and schema suites: **1,924 passed,
  2 skipped**. Backend catalog/data-service/preprocessing suites: **65 passed**.
- The real FileSystemCatalog Parquet path feeds automatic SimpleImputer then
  MinMaxScaler through FeatureEngineer on both engines; fit and replay produce
  `[0.0, 0.5, 1.0]` while retaining the source Decimals and label column.
- Ruff and full configured `ty check` pass. Focused review found no blockers.

Docs and changelog explain float64 precision and refitting older pipelines whose
artifacts omitted Decimal columns. The live queue has **35 open / 4 parked**;
OC-121 remains open and parked findings remain parked.

### 2026-09-12 — OC-113 fixed: stable per-feature VIF diagnostics

After committing OC-110/247 as DCO-signed `529e7c9b` with passing hooks,
the next unparked finding was reproduced through public `EDAAnalyzer.analyze`.
For 100 standard-normal rows from `default_rng(2)`, `b = 2*a + 1e-9*noise`
and an independent third column, the correlation condition number was
`1.811e16`. Both near-duplicate columns reported VIF **1.0**, and no
multicollinearity alerts were emitted. Seeds 3 and 4 reproduced the same
false all-clear. An exact duplicate pair also exposed the old singular fallback
incorrectly assigning 999 to an independent feature.

Well-conditioned inputs retain the inverse-correlation calculation. Above a
condition number of `1e10`, on inversion failure, or with invalid inverse
diagonal entries, VIF is computed from each feature's regression on the other
centered, normalized features. Direct residual sums avoid the cancellation in
`1 - R²`; the existing finite 999 warning marker applies only to a feature whose
residual variance ratio is at or below float64 epsilon. Unrelated features
retain their own VIF. The seed-2 report now returns **999 / 999 / 1.020194**
and two high-VIF alerts. Constants, missing-row handling and the minimum-row
guard are preserved.

The approach follows NumPy's documented
[inverse-conditioning caveat](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html)
and [least-squares residual definition](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html).
Unlike a blanket 999 fallback or a correlation pseudoinverse, it does not
publish a low VIF for dependent features or blame every feature for a singular
pair. No dependency or wire-format change is needed.

Tests first produced **9 expected failures / 6 passing controls**, followed
by four failing invalid-inverse cases. All **19 focused tests** now pass.
An orthogonal design has the independently derived VIF `1 + 4/noise²`, tested
at noise 0.5, 1e-5 and 1e-7, including shifted/scaled columns. Other cases cover
exact dependence with an independent feature, three original false-negative
seeds, negative/sub-one/non-finite inverse outputs, public warnings and backend
JSON serialization with `allow_nan=False`.

Final verification: **558 tests passed / 133 existing dependency and numeric
edge-case warnings** across every `test_profiling_*.py` module and the seven
backend EDA suites (the prior six plus `test_eda_vif.py`). The run used
`-q --tb=short -p no:cacheprovider
--basetemp=tmp_repro_artifacts/oc113-final-pytest`; output is recorded in ignored
`tmp_repro_artifacts/oc113-final-tests.log`. Repository-wide Ruff, the configured
backend/Core Ty scope, changed Python formatting and diff checks pass.
Independent review found no remaining issue. User documentation and v0.8.20
release notes explain the warning marker and rerunning saved analyses.
OC-113 moves to the archive; the live queue has **36 open / 4 parked** findings.

### 2026-09-12 — OC-110/247 fixed: observed string categories and counts

Public `EDAAnalyzer.analyze(target_col="label")` reproduced the filed failure
for 50 rows / 3 labels and 100 rows / 5 or 10 labels: the target was `Text`,
`task_type` and `rule_tree` were `None`, categorical statistics were absent,
and target associations were empty. The new regressions and updated customer
sample expectation produced **13 expected failures / 8 passing controls**
before the production change.

Both the batched and per-series inference paths now pass their column counts
to the same helper. Strings are categorical when fewer than 5% of observed
values are distinct, or when at most 20 distinct observed values include a
repetition. Nulls are excluded from both the distinct count and denominator.
Unlike the source audit's blanket small-vocabulary suggestion, the repetition
guard preserves all-unique strings as Text. All-null strings remain Text
regardless of frame size. Integer inference and native Categorical/Enum
behavior are unchanged; OC-50 and OC-111 remain open.

Regressions cover the 5% boundary, 20/21 distinct labels, nulls at the small
vocabulary boundary, sparse strings, constant labels, all-unique text and
large low-ratio vocabularies. Full target analysis verifies categorical
statistics, classification, rules, associations and balance advice. The
backend analysis/JSON serialization regression verifies the saved report
contains these results without altering source rows. The checked-in customer
sample now correctly profiles repeated city labels as Categorical.

Independent review also identified **OC-247**, reproduced for both strings
and native categoricals: `["a", "a", None]` reported two distinct and two rare
labels; 99 copies of `"a"` plus null still reported two distinct labels and
one rare label. A null-heavy column with ten observed categories displayed
only nine because null occupied a top-ten slot before being removed.
Eight new regressions failed before this correction. Category frequencies now
drop nulls before ranking/counting, and the distinct-label count removes the
null entry. Tests cover String/Categorical/Enum, low/high frequency labels,
the top-ten boundary and all-null native categories. Missing counts remain
available independently; source rows and native semantic types are preserved.
OC-247 is filed closed here; it does not increase the live queue count.

Final verification: **543 tests passed** across all `test_profiling_*.py` modules
and the six backend EDA suites (`test_eda_api`, `test_eda_router_extra`,
`test_eda_tasks_extra`, `test_eda_profiling_dtypes`, `test_eda_target_contract`,
`test_eda_target_association_missing_values`). The run reports 133 dependency,
numeric edge-case and plotting warnings, with no failures. Command: repository
`.venv/Scripts/python.exe -m pytest` with those paths, `-q --tb=short
-p no:cacheprovider --basetemp=tmp_repro_artifacts/oc110-oc247-final-pytest`.
The log is `tmp_repro_artifacts/oc110-oc247-final-tests.log` (ignored).
Repository-wide Ruff, the configured backend/Core Ty scope and formatting of
all seven touched Python files pass. Follow-up review found no remaining issue
in the correction. The EDA user guide and v0.8.20 release notes
describe the heuristic and rerunning saved analyses. OC-110 moves to the
archive; the live queue now contains **37 open / 4 parked** findings.

Commit verification repeated the **543-test** run and passed Ruff/Ty again.
`mkdocs build --strict` also passes after adding the missing docstring type
for the existing `FeatureEngineer.fit_transform(node_id_prefix)` parameter;
this one-line documentation repair does not change its signature or behavior.

### 2026-09-11 — Web EDA label sorting review

The repeated scanner finding referred to the single bare `.sort()` in
`scatterGrouping.ts`. It now uses `String.localeCompare` with the fixed `en`
locale and a lexical tie-breaker for distinct Unicode spellings that collate
equally. Mixed-case/accented labels sort alphabetically without making color
and marker assignments depend on browser language or input row order.

The mixed-case regression failed before the change; all **112 EDA component
tests** and **6 desktop/mobile browser cases** now pass. The browser fixture's
expected legend order was updated to the intended alphabetical order. Frontend
lint/CCN, explicit browser-spec lint, TypeScript/Vite build, size budgets and
diff checks pass. Docs and rebuilt assets are updated; OC counts are unchanged.

### 2026-09-11 — OC-235–245 fixed: target semantics, causal graphs and profiling identity

The user authorized all ten reviewed repairs with subagents. Three agents
implemented causal conversion/selection, target aggregation/outlier provenance,
and frontend graph rendering; the parent integrated categorical target
eligibility, exclusions, schemas, backend payload coverage, documentation and
the production build. Independent review found the adjacent recommendation
`count` collision (OC-245), which was also reproduced and repaired.

The resulting contract is documented in
[`eda_target_causal_review_2026-09-11.md`](eda_target_causal_review_2026-09-11.md)
and the EDA user guide. Categorical targets retain eta associations and original
values; Pearson/Fisher-Z no longer treat arbitrary category codes as numeric
measurements. Numeric target eligibility respects explicit task choice. New
optional report metadata explains omissions, even without a graph, and the
actual selection method; legacy fields remain readable. Both causal and
correlation notices are reachable when no numeric matrix is available.

Directed edges match the real PC output. Graph caps use the actual selected
numeric target. Exclusions override rule discovery; aliases cannot overwrite
group keys; outlier indices retain zero-based positions in the filtered input
before sampling without changing the sampled model features or scores. Web
graphs preserve distinct IDs, render both bidirected arrowheads, and remove
old results when history loads an empty graph. Saved analyses must be rerun
for numerical/result changes; reloading the UI applies rendering corrections.

All repairs have failing-before/passing-after focused regressions. Final gates:
**7,054 Core tests passed / 80 skipped** (cached NLP model in offline mode),
**3,612 backend tests passed / 1 legacy test deselected**, **2,487 frontend
tests passed**, and **8 Chromium desktop/mobile cases passed**. All ten Python
snapshots passed. Repository Ruff, backend/Core Ty, changed Python formatting,
frontend ESLint/CCN, production build, size budgets and diff checks passed.
Commands and environmental limits are recorded in the linked review.
No production changes were needed in the backend; its configuration and
serialization path is covered by the new target-contract regression.

The broad backend run also exposed test-environment limitations. Artifact
routing now uses pytest's temporary directory instead of `/tmp/artifacts`.
The separate local-only inference smoke (OC-246) is unchanged: its external
workspace path is inaccessible in the sandbox, and a safe path-only probe
showed it returning early on a tuple model artifact while claiming a pass.
It is excluded from the final backend run and filed as a separate test repair,
not counted as inference validation.

OC-235–244 move out of the live queue, OC-245 is filed closed, and OC-246 is
filed open: **38 open / 4 parked**. Historical baseline and parked items are
unchanged. No commit was requested for this repair pass.

### 2026-09-11 — OC-235–244 filed: EDA target and causal graph review

The reported `species_encoded` node was reproduced with Iris. It is a
temporary Core target representation surfaced as a public label; the source
`species` column is unchanged. The bounded follow-up found **10 new issues
(4 🟠 / 5 🟡 / 1 ⚪)**, recorded open with evidence in
[`eda_target_causal_review_2026-09-11.md`](eda_target_causal_review_2026-09-11.md).

Seven public `EDAAnalyzer.analyze` reproductions establish reversed causal
arrows, arbitrary category-code-dependent Pearson/PC results, omission of the
selected target under the graph cap, internal target labels, rules using an
excluded target, target-name aggregation collisions, and incorrect sampled
outlier row indices. The real causal-learn collider returns `A → C ← B`, while
Skyulf emits `C → A` and `C → B`. The same Iris observations reordered by class
change encoded-target Pearson from 0.9565 to 0.5804 and the PC graph from four
to five edges; fixed-code and categorical-association controls remain stable.

Three original-source Chromium probes reproduce colliding causal node IDs,
one-ended bidirected edges, and stale graph state after loading an empty
historical graph. These use mocked schema-supported report/history responses;
the audit does not claim current PC regularly emits bidirected or empty graph
payloads. Changing the target selector before Analyze was not filed as a bug.

The existing three focused profiling suites still pass **83 tests / 41
warnings** and miss these cases. Production source and committed tests were
unchanged; only the review, live queue, and this log were updated. OC-235–244
remain open, taking the live queue from **37 to 47 open / 4 parked**. Existing
parked items and historical baseline counts are unchanged.

### 2026-09-11 - OC-234 fixed: matching scatter marker shapes and legends

The user requested the separately filed 3D marker repair. Two new component
regressions failed on the old shape cycle and diagonal cross legend. Four
real-browser PCA cases also failed before source changes: on desktop and
mobile, Plotly coerced triangle-up/star to circles, while the all-missing
legend drew X for a resolved cross (+) marker.

The shared cycle now uses circle, square, diamond, cross (+) and x. Chart.js
maps these to circle/rect/rectRot/cross/crossRot; scatter3d accepts their common
names unchanged. The legend draws the matching plus and diagonal X. The five
shapes repeat for larger category sets. This shared configuration covers PCA
and Bivariate in both dimensions, preserving category colors, row grouping
and missing-value identity. Maps continue using their circle legend.

Verification:

- **99 focused EDA tests pass**, including both new regressions.
- **2,476 frontend tests pass** across 190 files.
- **7 browser tests pass**: the six desktop/mobile label cases plus the
  existing real Plotly PNG export test. The strengthened checks compare
  `_fullData[].marker.symbol` with requested symbols and the legend's actual
  SVG geometry through 2D -> 3D -> 2D and reversed rows. All five symbols are
  preserved; gray missing markers and their plus legend also match.
- Lint, CCN10, TypeScript/Vite build, all eleven size limits and diff checks
  pass. Production assets rebuilt at v0.8.20. Independent review found no
  actionable issue and confirmed Chart.js marker geometry with a drawing probe.

The EDA guide and v0.8.20 changelog describe the common marker cycle. OC-234
moves to the closed rows; the live queue has **37 open / 4 parked** findings.

Original pre-fix reproduction evidence, moved from the live queue:

### 2026-09-11 - OC-234: Plotly 3D marker shapes differ from the legend

Found while verifying OC-231-233 in the real EDA page. The existing shared
adapter supplies triangle-up and star to scatter3d, but Plotly coerces both
to circle. The external legend still shows a triangle/star. This predates
the grouping changes; `chartMarkerShapes.ts` is unchanged by this fix batch.

Executed `npm.cmd run test:e2e -- e2e/eda-label-groups.spec.ts --workers=1
--reporter=line`: six tests pass for category colors, coordinates and label
identity. The PCA cases also record requested `data[].marker.symbol` and
resolved `_fullData[].marker.symbol` in `pca-marker-symbols.json` attachments.
Both desktop and mobile record triangle-up -> circle for the real Unlabeled
category and star -> circle for constructor. Circle, square, diamond and
cross remain unchanged. These color/point tests do not claim symbol parity.

Fix target: use renderer-supported shapes and align each chart's visible
legend with its actual markers; pin resolved Plotly symbols as well as input
trace props.

### 2026-09-11 - v0.8.20 versions and OC-21 artifact snapshot correction

The user supplied full-suite CI failures for the WOE artifact snapshot. The
failure reproduced locally: the OC-21 normalization fix correctly returns
IV `1.739533`, while the active unit snapshot still expected `2.126096`.
For its x/y/z fixture, positive proportions are `[7, 3, 1] / 11` and negative
proportions are `[1, 5, 5] / 11`; both sum to one. Independent calculation gives
IV `1.7395330719636604`. The old denominator reproduces `2.1260959768444736`.

The existing test now checks this hand-derived IV before comparing the whole
artifact. Syrupy regenerated the active snapshot with `--snapshot-update`;
the only snapshot change is the city IV scalar. WOE mappings and production
calculation are unchanged. This closes the snapshot omission from the earlier
OC-21 verification, which did not include `test_artifact_snapshots.py`.

Root app/Core versions, the uv lock record and frontend package/lock mirrors
are now **0.8.20**. Dependency payloads are unchanged. The local Core editable
install was refreshed offline without dependencies; `skyulf.__version__`
also reports **0.8.20**. Frontend assets were rebuilt from the current tree.

Verification: **172 related tests pass**, including all three artifact
snapshots; the snapshot module also passes from the `skyulf-core` directory.
Scoped Ruff, format and Ty checks, version sync/check, production frontend
build and all eleven bundle-size limits pass. Independent review found no
actionable issue. Changelog updated; queue remains **38 open / 4 parked**.

### 2026-09-11 - OC-231-233 fixed: web EDA category identity

Shared grouping now uses Map keys with null reserved for missing values and
sorted observed labels for repeatable color/shape assignment. Missing groups
keep a neutral gray color and a collision-safe caption, including when real
categories contain Other, Unlabeled or its missing suffix. 2D datasets, 3D
traces and maps consume the same category color metadata. Colors are stable
for the same observed category set; this is not a global category palette.

All-missing PCA/scatter legends remain visible. Map legends use circles to
match Leaflet markers and retain category colors beyond twenty groups, with
the existing long-list filter. No-target maps preserve blue coordinate-only
markers/popups. Chart.js receives the category color for fill and stroke, so
missing crosses are actually gray. Raw table/CSV labels are preserved. The
guide and changelog explain the behavior and reuse of existing saved profiles.

Verification:

- New helper tests first reproduced row-order color changes and prototype-key
  crashes; component tests caught the old map color differences. Two review
  regressions then reproduced a missing Chart.js stroke color and an unwanted
  no-target map legend, and passed after correction.
- Focused EDA suite: **97 tests passed**. Final full frontend suite:
  **2,474 tests passed** across 190 files (`npm.cmd test -- --maxWorkers=4
  --reporter=dot --silent`). Existing deliberate error-boundary traces remain.
- Six new real-browser tests pass on desktop and mobile across PCA 2D/3D,
  Leaflet points/popups, row reversal, missing/colliding labels and all-missing
  targets (`npm.cmd run test:e2e -- e2e/eda-label-groups.spec.ts --workers=1
  --reporter=line`). Unit regressions supplied the source-red evidence;
  browser tests were first run after implementation.
- Final lint, CCN10, TypeScript/Vite production build and bundle-size checks
  pass. Generated `static/ml_canvas` assets were rebuilt. Read-only review
  found no remaining scoped issue and verified the actual Chart.js stroke.

The renderer-specific Plotly shape coercion found during browser validation
is separately filed as OC-234; overall 2D/3D shape parity is not claimed.
The live queue has **38 open / 4 parked** findings after these three closures
and the new finding. No commit was requested for this frontend fix turn.

Original pre-fix reproduction evidence, moved from the live queue:

### 2026-09-11 - OC-231-233: frontend EDA label review

Review requested after the OC-49/52 Python visualizer examples. These frontend
charts consume profile JSON using Chart.js, Plotly and Leaflet; they do not
use `EDAVisualizer.plot()`. Executed the current `CanvasScatterPlot`,
`ThreeDScatterPlot`, `GeospatialTab` and shared grouping helpers through React
server rendering, capturing chart-library props while keeping the actual
component/grouping code. This is runtime payload validation, not a browser
pixel/rendering check. Reproduction script:
`tmp_repro_artifacts/frontend_eda_label_audit.cjs` (ignored local artifact).

**Controls:** partial labels `["a",null,"b"]`, all-null labels, and text labels
`["nan",null,"inf","-inf","1e309","1"]` retain all coordinates in both 2D
Chart.js datasets and 3D Plotly traces. Labels are not converted into numeric
colors, so the Python non-finite-color masking defect is not reproduced here.

**OC-231:** render labels `["Gold","Silver","Bronze"]`, then reverse their
order. Gold changes from `#8884d8` to `#ffc658` in scatter datasets. Reordering
geospatial points produces the same color change. The palette index comes
from first-seen group order. **Fix/verification target:** deterministic category
identity for the same label set across row order, matching chart/legend colors
and marker shapes; use the same rule for 2D, 3D and the map.

**OC-232:** labels `[null,"Other","a"]` produce a single Other group with
**2 points**, shared by both scatter engines and the legend. No coordinates
are dropped, but missing and genuine category identity are conflated.
**Fix/verification target:** keep missing labels separate from real strings,
with an explicit missing-label presentation that cannot collide with an
observed category of the same displayed name.

**OC-233:** each label `__proto__`, `constructor` and `toString` throws
`groups[label].push is not a function` through the current scatter component.
The empty `{}` grouping object inherits these keys, and `??=` does not replace
the non-null inherited value with an array. **Fix/verification target:** group
arbitrary labels without prototype-key collisions and keep 2D/3D/legend
consumers consistent. This is a reproduced rendering failure, not a claim of
an application security exploit.

### 2026-09-11 - OC-231-233 filed: frontend EDA needs separate label fixes

The user asked whether the Python visualizer defects also affect frontend EDA.
Runtime component/renderer-boundary probes confirmed three separate frontend
findings, now open with reproduction evidence in the live queue: row-order
color changes (OC-231), missing/Other category collision (OC-232), and
prototype-named label crashes (OC-233).

Partial/all-missing labels and numeric-looking text labels retain their point
coordinates in both 2D and 3D chart payloads. Python and frontend rendering
implementations are independent. The earlier "no Canvas change is required"
statement applies only to integrating the Python correction; it was not a
validated claim that frontend label handling had no defects. This review
corrects that overbroad interpretation. No frontend implementation was changed
in this review, and no browser pixel-level check is claimed.
The existing OC-230/49/52 Python fixes remain valid and pending commit.
The live queue is now **40 open / 4 parked** findings.

### 2026-09-11 - OC-230, OC-49 and OC-52 fixed: missing ratios and plot labels

Committed OC-23, OC-191 and OC-199 as `b28f90fc` with DCO sign-off after
**574 Core / 59 backend tests passed** and all applicable hooks passed.
No push was performed. Reproduced each item in the next three-finding batch
before implementing its correction.

**OC-230:** native Polars NaNs survived operand normalization and poisoned
ratio sums, unlike pandas' existing missing-as-zero behavior. Normalize each
ratio operand's NaN and null values to zero before summing. An observed value
in another operand still contributes; signed epsilon clamping is unchanged.
Direct native frames cover numerator/denominator NaNs, mixed/all-missing sums,
negative small denominators, default/custom epsilon, integer/null-only inputs,
infinity controls and unchanged source columns. Red: **2 failures / 6 controls
passed**; both Polars cases had 9 incorrect rows out of 11. The new cases and
OC-23 sign regressions then passed **12 tests**. No fitted artifact migration
is needed; existing ratio artifacts use the corrected applier.

**OC-49:** missing labels shortened the PCA color vector without shortening
the coordinates. Real plotting reproduced the color/coordinate size error.
PCA now renders labeled points with target colors and missing-label points as
gray crosses with an Unlabeled legend. There is no numeric missing-label
sentinel, so real labels such as -1 and 0 retain their existing meaning.
All-missing labels produce a complete plot without a target colorbar.
Independent review also reproduced masked points for valid text labels such
as "nan" and "inf". Numeric colors now require finite conversions; otherwise
the complete label set uses categorical colors. Four real-artist regressions
failed before this correction, then passed, including "-inf" and "1e309";
a finite-numeric scale control preserves continuous target coloring.

**OC-52:** enumerating a set assigned different category codes across Python
hash seeds. The shared helper now sorts the non-null category strings before
enumerating, stabilizing PCA/geospatial colors for an unchanged category set
regardless of row order or process. Changing the category set can still
renumber codes; persistent category palettes are outside this fix.

Visualizer red phase: **8 failures / 2 controls passed**, including actual Agg
rendering and different PYTHONHASHSEED subprocesses. The final **15 new tests**
exercise public `plot()` for eleven label combinations, inspect real artists
for visible/unmasked coordinates and missing-label presentation, and cover
both categorical consumers and subprocess stability. A mixed-label PCA image
was also generated and visually checked. The existing visualizer tests remain
unchanged and pass alongside these regressions.

Combined verification: **597 Core tests pass**, covering all profiling and
feature-generation test modules; 121 existing warnings concern numerical
edge cases, library deprecations and the headless/Windows environment. Updated
the EDA guide, Feature Generation reference, relevant docstrings and changelog.
These are Python library behavior and plotting corrections; no Canvas change
is required. Existing saved profiles can be replotted without rerunning EDA.
Repository Ruff/Ty, scoped formatting, all applicable pre-commit hooks and
`git diff --check` pass. Independent review confirmed the non-finite-label
follow-up resolves its finding and reported no remaining actionable issues.
The live queue now has **37 open / 4 parked** findings.

Original pre-fix OC-230 reproduction, moved from the live queue:

Reproduced through public `FeatureGenerationCalculator.fit` and
`FeatureGenerationApplier.apply` while verifying OC-23. Select ratio inputs
`input_columns=["n"]`, `secondary_columns=["d"]`, `output_column="r"` and use
the default epsilon. On `{"n": [1.0], "d": [float("nan")]}`, a native pandas
DataFrame produces approximately **1e9**, while a native Polars DataFrame
produces **NaN**. Pandas' horizontal sum treats missing values as zero;
Polars' sum preserves native NaN. Constructing the Polars frame with
`pl.from_pandas` normalizes NaN to null and hides the difference.

**Fix/verification target:** define and apply consistent missing-value
semantics to native ratio inputs, with direct engine-native frames covering
NaN numerators, denominators and sums. OC-23 addresses only the sign of
finite near-zero denominators and does not close this separate behavior.

### 2026-09-11 - OC-23, OC-191 and OC-199 fixed: ratios and profiling inputs

Committed OC-74 as `acefc59f` with DCO sign-off, **337 Core tests / 3 backend
registry tests passing**, two optional H3 skips and all applicable hooks
passing. No push was performed. This continuation handles three independently
reproduced findings.

**OC-23:** the Polars ratio expression clamped every denominator with
`abs(denominator) < epsilon` to positive epsilon. Negative near-zero sums
therefore inverted the output sign. It now uses the denominator's sign, as
the pandas path already does. Public calculator/applier regressions cover
both engines, default/custom epsilon, positive and negative numerators,
threshold boundaries, zero, normalized missing values, and multi-column sums.
Red: **2 failures / 2 controls passed**. Focused feature-generation suites:
**171 passed**. Saved epsilon settings and other operations are unchanged.

**OC-191:** native Null and Enum fell through to Text and failed batched
string-length aggregation, aborting the whole profile. The shared dtype
resolver now maps Null to Unknown and native Enum to Categorical. Unknown
retains counts, percentages, samples and high-missing alerts without invalid
type-specific statistics. Typed all-null columns retain their known dtype.
Red: **5 failures / 3 controls passed**. The focused analyzer/column suites
pass **62 tests**. Tests also cover Enum category frequencies and classification
target inference; an additional real backend analysis/JSON serialization test
preserves both new profile cases for API consumers. The profile dtype field
already accepts strings, and Canvas uses a fallback badge and optional stats,
so no frontend implementation or request change is required.

**OC-199:** explicit coordinate names bypassed the active selection and read
excluded data from the analyzer's retained source frame. Resolution now
requires each supplied coordinate to belong to `self.columns`, including
partially explicit pairs. An excluded coordinate yields no geospatial section.
Red: **6 failures / 6 controls passed**. Focused geo and repeated-exclusion
suites pass **29 tests**, covering either/both exclusions, reuse of one analyzer,
auto-detection and allowed custom numeric/string coordinate names.

Combined verification: **574 Core tests / 59 backend tests pass** across all
profiling and feature-generation modules and backend EDA helpers/routes.
The 122 warnings come from existing numerical edge cases, optional visualization
behavior, Windows CPU detection and library deprecations. Updated the EDA guide,
feature-generation reference, schema comment, helper docstrings and changelog.
Repository Ruff/Ty, scoped formatting and every applicable pre-commit hook
pass; independent read-only review found no actionable issue.
`git diff --check` is clean.

During OC-23 verification, direct native Polars NaN input exposed a separate
missing-value parity issue, now recorded as **OC-230** in the live queue with
executed evidence. The finite-denominator sign fix does not claim to close it.
After closing these three findings and filing OC-230, the live queue contains
**40 open / 4 parked** findings. Existing parked work is untouched.

Original executed evidence for the supplemental profiling findings:

**OC-191 — unsupported string aggregation on valid dtypes.** Executed
`EDAAnalyzer(pl.DataFrame({'x': [None,None]})).analyze()` raises
`SchemaError: expected String, got null`; using
`pl.Series(['a','b'], dtype=pl.Enum(['a','b']))` raises the equivalent Enum
error. **Fix/verification target:** handle null-only columns and recognize or
normalize Enum before text aggregates. OC-121 concerns preprocessing
auto-selection; this finding concerns profiling aborting completely.

**OC-199 — explicitly selected coordinates survive exclusion.** Executed
`EDAAnalyzer(pl.DataFrame({'lat':[1.,2.,3.], 'lon':[10.,20.,30.],
'x':[1.,2.,3.]})).analyze(exclude_cols=['lat','lon'],lat_col='lat',lon_col='lon')`.
The result still contains all three coordinate pairs in
`geospatial.sample_points`, plus their bounds and centroid, although the
per-column profile excludes them. **Fix/verification target:** apply the
exclusion policy consistently before explicit geospatial selection.

### 2026-09-11 - OC-74 fixed: ensemble-aware model discovery

Committed OC-29, OC-144 and OC-142 together as `23ea9a2d`, with DCO sign-off,
**798 fresh tests passing / 11 optional H3 skips** and all applicable
pre-commit hooks passing. No push was performed.

Reproduced OC-74 against the real registered nodes before editing code:
`voting_classifier`, `stacking_classifier`, `voting_regressor` and
`stacking_regressor` were all registered but absent from `list_models()` and
`list_models(category="Ensemble")`. All four were also incorrectly returned
by `list_transformers()`. The category argument was not literally unused;
the hardcoded Modeling check prevented it from selecting any Ensemble node.
The actual backend `_build_node_registry()` already returned the four models
because it uses `get_all_metadata()`, so the current Canvas is unaffected.

A shared immutable set of model categories now defines complementary model
and transformer lists. Unfiltered models include both Modeling and Ensemble;
exact category filtering keeps those groups distinct. Transformer discovery
excludes both, while custom non-model categories and unknown-category empty
results retain their existing behavior. Node IDs, metadata and registration
order are unchanged.

Red phase: **6 failures / 5 controls passed**, using a category-filter matrix
and all four real ensemble nodes. The two focused modules then passed all
**52 tests**. Broader verification passes **337 Core tests / 3 backend registry
API tests**, covering model discovery, ensemble fitting/prediction, existing
classifiers, text-node listing and preprocessing registry/replay behavior.
Two optional H3 tests skip; the 22 combined warnings are existing deprecations.
Repository Ruff/Ty, scoped formatting and `git diff --check` pass. Independent
review found no actionable issue. Updated the discovery docstrings, SDK README,
modeling reference and changelog. No frontend implementation change is needed.
The live queue now contains **42 open / 4 parked** findings.

### 2026-09-11 - OC-142 fixed: complete-pair target association statistics

Validated the current implementation before fixing it. A numeric feature with
four observed values `[0, 2, 2, 4]` in two target groups has eta **0.707107**.
Appending six missing feature values to one group inflated the reported score
to **1.118034**, and sixty inflated it to **2.828427**. Null and NaN fixtures
both reproduced this because the analyzer already normalizes NaN to null.
Unlabeled rows also polluted the global mean and total sum of squares: adding
two unlabeled outliers changed a perfectly separated feature from **1.0** to
**0.577351**. The old null-target test only checked the broad [0, 1] bound and
missed this incorrect value; it now asserts the analytical score.

The selected target and feature are filtered to complete pairs once, and all
group sizes, means and sums of squares use that same subset. Filtering is
specific to each feature and leaves the analyzer's source frame unchanged.
No complete pairs means the feature is omitted from associations; an observed
constant feature still reports zero. Scores are corrected by consistent input
rows, without clipping an inflated result to one.

Red phase: **12 failures / 18 passes**, including two actual backend
`_run_eda_analyzer` cases where the serialized report incorrectly ranked a
sparse feature above a perfectly associated feature. Both Boolean and
Categorical target reports now put `strong` (1.0) before `sparse` (0.707107),
while retaining row counts and missing-value counts. Additional regressions
cover empty/all-null/disjoint pairs, feature-specific missingness, unchanged
source data, and complete-data controls.

Verification: **168 tests pass**, spanning target associations, full analyzer,
correlations/distributions, recommendations, exclusions, column-name handling,
and backend EDA analyzer/task/API/router paths. The 47 warnings are existing
deprecations, degenerate-statistic warnings and Windows CPU-discovery fallback.
Repository Ruff/Ty, scoped formatting and `git diff --check` pass. Independent
review found no actionable issue. Updated the calculation docstring, EDA user
guide and changelog. The UI already uses the returned scores and ordering,
so no frontend implementation change is needed. Saved EDA reports must be
rerun to refresh their stored scores. The queue now contains **43 open /
4 parked** findings.

### 2026-09-11 - OC-144 and OC-29 fixed: distance names and supported feature operations

Committed the preceding OC-34 fix as `2a64d449`, with DCO sign-off,
**485 fresh tests** and all applicable pre-commit hooks passing.

OC-144 was reproduced before changing production code: **16 failures / 8
controls passed** across direct/default-metadata fits, missing/empty output
names, and both existing engine paths. Distances were already calculated in
the correct unit; their automatic names were wrong. A shared resolver now
chooses `geo_distance_km` or `geo_distance_mi`, and the registry's empty default
means automatic. Fitting saves the resolved name; direct apply uses the same
fallback. Explicit names, including old miles artifacts named
`geo_distance_km`, are preserved. The reference documents how to retain an old
column name when refitting or update downstream references to the new name.

Re-verification corrected part of the queue's earlier scope note: the existing
miles-conversion test already supplies `geo_distance_mi` and correctly compares
it with a kilometer baseline. The stale kilometer-name expectation was instead
in the behavioral re-audit's unit matrix; its four miles cases now expect
`geo_distance_mi`. Also, a module-level H3 `importorskip` was hiding all distance
tests when the optional package was missing. An H3-only fixture now limits
that skip to the two H3 test classes, keeping distance regressions executable.

OC-29 was reproduced with **11 failures / 1 control passed**. The public
allow-list advertised `polynomial`, while all three Feature Generation aliases
accepted it and both appliers silently skipped it. Remove that advertised
value and validate operation types at both public fit/apply boundaries, before
running any operation. Polynomial requests receive an error directing callers
to `PolynomialFeatures`; other unknown types list supported choices. Existing
artifacts containing unsupported operations now fail explicitly too. Omitted
operation types still default to arithmetic, and supported-operation failure
handling is unchanged. The old unknown-operation no-op fixture was replaced
by explicit error regressions. A real backend CSV-loader-to-feature-node run
confirms the error reaches the failed node response without an output artifact.

Verification: **628 Core / 2 backend tests pass**, including feature-generation,
GeoDistance, registry-contract, leakage, behavioral replay and preprocessing
pipeline suites. **11 optional H3 tests skip** because H3 is not installed;
31 warnings are existing deprecations. Repository Ruff/Ty, scoped formatting
and `git diff --check` pass. Calculator/applier docstrings, the preprocessing
reference and changelog are updated. No frontend implementation change is
needed: its Feature Generation choices already contain only supported types,
and GeoDistance has no Canvas settings component. Independent review found
no actionable issue. The live queue contains **44 open / 4 parked** findings.

### 2026-09-11 - OC-34 fixed: empty text vocabularies warn and preserve input

The preceding OC-31 work was committed as `f485ebdf`, with DCO sign-off,
241 fresh focused frontend tests, the Canvas browser regression and all
applicable hooks passing.

Reproduced the current Count and TF-IDF failures before editing production
code. Blank/missing text, English stop words, single-character/punctuation-only
input, unavailable n-grams and zero-row corpora all raised sklearn's
`empty vocabulary` error. The new Core regressions first reported **20 failures
/ 10 controls passed**, and both real backend pipeline cases failed at the
vectorizer. This confirms the reported issue at both calculator and engine level.

Each builder now catches only the specific empty-vocabulary error, emits a
warning with guidance to check text/stop words/n-grams, and returns `{}`.
Existing appliers then preserve X, y, row order and source columns, including
when `drop_original=True`. A saved empty artifact remains a no-op on later
nonempty text, so prediction cannot learn a replacement vocabulary. Refit
after correcting training text/settings to produce numeric features. Invalid
parameters, contradictory frequency limits and pruning that removes all
learned terms still raise their original errors.

The backend regression runs the actual CSV loader and transformer through
`PipelineEngine.run`, checks unchanged persisted data, a warning tagged with
the correct node id/type, and replay through the saved pipeline artifact.
The test selects pandas through the cached settings object used by the catalog;
changing only the environment after settings initialization did not select it.

Verification: **483 Core tests / 2 backend tests pass** across vectorization,
encoding audit, registry-contract and leakage suites. The ten warnings in the
broader contract run are existing sklearn/deprecation warnings. The existing
frontend notification hook's **3 tests pass**; it already forwards node warnings
to the notification center, so no frontend implementation change is needed.
Repository Ruff/Ty, scoped formatting and `git diff --check` pass. Calculator
docstrings, the Text & NLP guide and changelog are updated. Independent review
found no actionable issue. The live queue now
contains **46 open / 4 parked** findings.

### 2026-09-11 - OC-31 fixed: target-free correlation selection in Canvas

Committed the preceding OC-32 work as `fc9dece3` with DCO sign-off, after
**594 fresh Core tests** and all applicable pre-commit hooks passed.

Revalidated the Core behavior and frontend rejection before editing production
code. The public feature-selection facade, without a target, drops `b` from
`a=[1,2,3,4]`, `b=[2,4,6,8]`, `other=[1,0,0,1]` at Pearson threshold `0.9`,
preserving `a` and `other`. Canvas validation nevertheless required a target
for correlation selection, while its settings panel correctly hid that field.

The validator now exempts `correlation_threshold` alongside `variance_threshold`.
The eight supervised selection methods still reject absent/empty targets and
accept a configured target. The new regression first failed with the original
target-required error; the other **25 tests passed**. An old assertion preserved
the incorrect rejection, and a hidden-field test artificially requested that
invalid target error for correlation; both obsolete expectations were removed.

Verification: **241 focused tests pass**, covering settings, validation reveal
and pipeline serialization. The new Playwright test passes keyboard method
changes, disappearance of the target error/control, an actual Preview POST with
the correlation settings and no `target_column`, mocked result rendering,
mobile read-only layout, retained desktop settings and restoration of the
supervised target requirement. Core numerical behavior was verified separately.

All **2,461 frontend tests pass across 188 files**. Frontend lint, complexity,
TypeScript/production build, all bundle-size budgets and `git diff --check`
pass. Independent review found no actionable issue. Updated the user reference,
changelog and generated `static/ml_canvas` assets. No backend/Core change is
needed. The live queue now contains **47 open / 4 parked** findings.

### 2026-09-11 - OC-32 fixed: variance selection may reject every candidate

Committed the preceding OC-21/33 work as `5b39c28d` with DCO sign-off.
Fresh combined verification passed **471 Core tests / 67 frontend tests**;
all applicable pre-commit hooks passed, including Ruff, Ty and frontend lint.

Revalidated OC-32 before changing production code: constant values `[7,7,7]`,
all-NaN values and `[0,1,0]` with threshold `1` each raised sklearn's
`No feature in X meets the variance threshold` error. The library had already
computed variances, but the exception prevented an artifact from reaching the
existing candidate-column removal logic.

The calculator now handles only that specific no-feature error and records
`selected_columns=[]` with the normal candidate list, threshold, drop flag and
variance values. Non-finite computed variances are accepted only for entirely
missing columns, preserving numerical overflow failures. Other input and
configuration errors still propagate. Applying
the artifact removes only candidate columns; unselected columns, target values
and `drop_columns=False` behavior remain intact. Both facade aliases retain
the fitted empty selection when prediction values later vary. If every model
feature is removed, the user must adjust selection before downstream training.

New regressions first reported **11 failures / 6 controls passed**. They cover
constant and all-null candidates, exact-threshold equality, one-row inputs,
duplicate indexes and target alignment, auto-selection of all numeric columns,
both drop modes, both facade aliases and inference replay. Controls preserve
errors for negative/NaN/text thresholds, infinite values, empty rows and
explicitly selected non-numeric values.

Independent review reproduced finite `[1e200,2e200]` overflowing its variance
to infinity, and `[1e308]*4 + [-1e308]*4` overflowing to NaN, at threshold `1`.
Both produce the same sklearn error prefix. Each additional regression failed
before its correction. The guard checks whether a column actually has observed
values, so computational overflow remains an error while all-missing columns
can legitimately have undefined variance.

Verification: **594 related tests pass** across the feature-selection, shared
helper, leakage and registry-contract suites. The 32 warnings include four
sklearn all-null warnings and three numerical warnings exposed by the new
cases; those calculations remain visible. Repository Ruff/Ty, scoped
formatting and `git diff --check` pass. User reference and changelog updated;
the existing Canvas variance controls require no frontend change.
Final independent review verified both overflow controls and found no remaining
actionable issue.
The live queue now contains **48 open / 4 parked** findings.

### 2026-09-11 - OC-33 fixed: self-products with fewer columns than the degree

Revalidated in Core and Canvas before changing production code. Core skipped
combination generation whenever the selected-column count was below the degree,
even with `interaction_only=False`. Canvas independently rejected the same valid
configuration. This affected single-column squares, cubes and fourth powers, as
well as two-column degree-three products. The existing combination resolver
already handles repeated columns correctly; fit now calls it directly. Canvas
requires at least `degree` columns only when distinct-column interactions are
enabled (including the existing omitted-setting default).

The new Core regressions first reported **4 failures / 8 controls passed**.
They pin hand-calculated powers and product names, missing values, duplicate
indexes, target alignment, input preservation, empty selections and optional
bias columns. Frontend regressions first reported **5 failures / 14 controls
passed**, including an invalid degree whose error previously targeted columns.
The related Core interaction, registry-contract and feature-operation leakage
suites now pass **302 tests**, with nine existing warnings. Focused frontend
validation and serialization checks pass **124 tests**.

The new Playwright test passes through the actual Canvas control and Preview
submission: keyboard toggling clears the validation issue, the request retains
one column with degree four and `interaction_only=False`, and a mocked response
renders the generated column. It also checks mobile read-only layout and retained
settings after returning to desktop. Numerical results are covered by Core tests.

Verification: **2,452 frontend tests pass across 188 files**, and the browser
regression passes. Repository Ruff/Ty, scoped Python formatting, frontend lint,
complexity, production build, bundle-size checks and `git diff --check` pass.
Independent review found no actionable issue. The node reference, calculator
docstrings, changelog and generated `static/ml_canvas` assets are updated.
The live queue now contains **49 open / 4 parked** findings.

### 2026-09-11 - OC-21 fixed: normalize WOE smoothing over observed categories

Revalidated before changing production code. The effect is broader than the
original report's "more than two categories": two equally sized categories,
each with one positive and two negative targets, produced WOE `-0.076961`
for both and IV `0.006841`, where both should be zero. Each category received
a regularization pseudocount, but the class totals included only one such
pseudocount, so neither class's category probabilities summed to one.

`_column_woe` now adds `regularization * n_categories` to each class total.
The shared calculation covers the full-training artifact and each training
complement independently. Missing feature values count as an observed category;
categories seen only in held-out rows do not inflate the training denominator
and continue to receive the existing zero fallback. Stored mappings are applied
unchanged, so existing fitted models retain their inference values; refit to
use the corrected WOE/IV calculation.

The seven new regressions all failed against the original code, while two
existing controls passed. They cover two/three equally sized categories with
identical target rates, two regularization values, hand-derived WOE and IV for
an imbalanced three-category example (including a missing category), and the
actual cross-fitting training hook with three categories per complement versus
four globally, unseen categories and a single-class complement. Two existing
test helpers copied the incorrect production formula; those expectations now
use hand-derived literals, including the deterministic held-out-fold results.

Verification: **169 related tests pass**, with two existing OneHotEncoder
unknown-category warnings. Repository Ruff/Ty, scoped formatting and
`git diff --check` pass. Related test command:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  skyulf-core/tests/integration/test_encoding_woe.py `
  skyulf-core/tests/integration/test_woe_and_calibration.py `
  skyulf-core/tests/unit/test_encoding_operation_leakage.py `
  skyulf-core/tests/unit/test_encoding_text_deep_audit_20260908.py `
  -q --tb=short --basetemp=.pytest-tmp-oc21-green -o cache_dir=.pytest-tmp-oc21-cache
```

The calculator docstring and preprocessing placement guide explain the
normalization and saved-artifact behavior; release notes are under **v0.8.20**.
Independent review found no actionable issue. OC-21 moves to the archive,
leaving **50 open / 4 parked** findings.

### 2026-09-11 - OC-27 fixed: preserve each power rule's standardization setting

Revalidated the current Canvas-to-Core path before changing code. The UI sends
`standardize=False`, and the converter includes it in each flattened column
rule. GeneralTransformation discarded that setting and hardcoded `True` when
fitting and reconstructing PowerTransformer. On `[1, 2, 4, 8, 16]`, Box-Cox
returned approximately `[-1.414, -0.707, 0, 0.707, 1.414]` instead of the
requested unstandardized `[0, 0.693, 1.386, 2.079, 2.773]`. Yeo-Johnson likewise
returned zero-mean, unit-standard-deviation output with the option disabled.

The new regressions initially produced **2 failures / 2 passing controls**.
For both methods, mixed column rules now compare `False`, `True` and omitted
settings against independently fitted sklearn transformers on training and
unseen data. Yeo-Johnson includes negative and zero inputs. The tests also
check stored choices, absence of scaler statistics when disabled, duplicate
held-out indexes and the output of historical artifacts without the flag.

Fitting now passes each rule's setting into PowerTransformer and stores it
beside the learned lambda. Both existing apply paths read the saved flag;
missing flags still default to `True`, preserving previously saved artifacts.
The calculator docstring and preprocessing reference explain the behavior;
the release note is under **v0.8.20**. Frontend code already emits the setting
correctly and needs no change.

Verification: **182 related tests pass**, with one existing Polars polynomial
concatenation deprecation warning. The four new regressions also pass after
the final test lint adjustment. Related suite command:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  skyulf-core/tests/integration/test_transformations_general.py `
  skyulf-core/tests/integration/test_transformations_power_simple.py `
  skyulf-core/tests/unit/test_feature_operation_leakage.py `
  skyulf-core/tests/unit/test_artifact_shapes.py `
  -q --tb=short --basetemp=.pytest-tmp-oc27-green -o cache_dir=.pytest-tmp-oc27-cache
```

Repository Ruff/Ty, scoped formatting and `git diff --check` pass. Independent
review found no actionable issue in the change. OC-27 moves to the archive,
leaving **51 open / 4 parked** findings.

### 2026-09-11 - OC-26 fixed: honor Hashing Vectorizer's None normalization option

Revalidated against the current code before changing production behavior.
The Canvas emits `norm="none"`; artifact construction retained that string,
and applying the artifact raised sklearn's `InvalidParameterError` in
`normalize`. The existing `norm=None` unit test only inspected the artifact
and never exercised the Canvas string or the resulting transformation.

The new registry-based regression initially produced **1 failure / 5 passing
controls**. A corpus with three `hello` and four `world` tokens must produce
bucket counts `[3, 4]` with normalization disabled, `[3/7, 4/7]` for L1 and
`[0.6, 0.8]` for L2. The test also covers Python `None`, the existing empty-string
alias, omitted/default L2, an empty document and reuse of the fitted artifact.

Artifact construction now translates only the Canvas string `"none"` to
Python `None` before building sklearn's vectorizer. The shared constructor
keeps the normalized value in both the artifact and vectorizer object.
The user guide and calculator docstring explain the accepted values; the
release note is under **v0.8.20**.

Verification: **233 related tests pass**, with two existing unknown-category
warnings from OneHotEncoder controls. Reproduce the test run with:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  skyulf-core/tests/integration/test_vectorization.py `
  skyulf-core/tests/unit/test_text_vectorization.py `
  skyulf-core/tests/unit/test_vectorization_gaps.py `
  skyulf-core/tests/integration/test_text_target_context.py `
  skyulf-core/tests/unit/test_encoding_operation_leakage.py `
  skyulf-core/tests/unit/test_encoding_text_deep_audit_20260908.py `
  -q --basetemp=.pytest-tmp-oc26-green -o cache_dir=.pytest-tmp-oc26-cache
```

Repository Ruff/Ty, scoped formatting and `git diff --check` pass. Independent
review found no actionable issue in the code, regression, guide or release
note. OC-26 moves to the archive, leaving **52 open / 4 parked** findings.

### 2026-09-10 - OC-184 fixed: send the configured production security headers

The existing production header dictionary had no consumer. The initial
integration regressions produced **7 failures / 4 passing controls** across
normal/error, redirect, static, streaming and custom-policy responses.
`SecurityHeadersMiddleware` now copies the configured map at startup and
updates only `http.response.start`, preserving bodies and other ASGI scopes.
It wraps the custom error middleware and sits inside outermost CORS. Empty
policies and development/testing profiles retain their previous behavior;
configured headers override conflicting endpoint values without collapsing
unrelated repeated headers such as cookies.

Applying the dormant CSP exposed real browser incompatibilities: Canvas PNG
export failed with blocked data/blob images; ReDoc then reported blocked blob
workers and its logo. The default policy now permits the actual chart/map
and enabled Swagger/ReDoc sources: data/blob images, OpenStreetMap tiles,
jsDelivr assets, the docs favicon, the specific ReDoc logo and blob workers.
Existing inline script/style allowances remain; `unsafe-eval` was not added.

Verification: **106 related backend tests pass**, including **15 security
header regressions**, request-error redaction, configuration and infrastructure
checks. Five existing dependency warnings remain. A local Chromium fixture
served the **actual built Canvas assets** through the production middleware:
Canvas boot, same-origin job WebSocket, real Plotly 3D rendering, PNG download,
map tiles and loaded Swagger/ReDoc all pass with **zero CSP violations**.
Browser fixture/scripts and result are under ignored
`tmp_repro_artifacts/oc184_*` / `oc184-browser-result.txt`; generated
`oc184-pca.png` confirms the export. No frontend source or bundle changed.

The configuration guide explains policy replacement and scope. CORS preflight
responses and fallback responses created by Starlette's outer server-error
handler bypass the inner middleware; normal route errors consumed by Skyulf's
error handler are covered. Release notes are under **v0.8.20**. OC-184 moves
to the archive, leaving **53 open / 4 parked** findings.

Repository Ruff/Ty, scoped formatting and `git diff --check` pass. Independent
review found no material issue within the documented scope.

### 2026-09-10 - OC-183 fixed: resolve the default S3 bucket through Settings

The raw `os.getenv("S3_BUCKET_NAME")` check ignored the Settings model's
canonical `AWS_BUCKET_NAME` and all dotenv-only bucket configuration. New
regressions produced **6 failures / 5 passing controls** before the fix.

`AWS_BUCKET_NAME` remains the canonical field. Pydantic `AliasChoices`
accepts the legacy `S3_BUCKET_NAME` name for existing configurations, and
`SmartCatalog` reads the resolved Settings value. Real dotenv/environment
tests pin both aliases, canonical precedence within a single source, and
environment precedence over dotenv even when the sources use different
aliases. Explicit catalog injection, local-only operation and optional-SDK
fallback are preserved. Credentials and S3 provider selection are unchanged.

Verification: **67 config/catalog/S3 tests pass**, including **11 new
regressions**. No S3 service is contacted: tests load real Settings and
replace the external catalog constructor. README and the backend
configuration guide now agree on the bucket name and precedence; the README
region example also uses the existing `AWS_DEFAULT_REGION` field. Release
notes are under **v0.8.20**. The queue now has **54 open / 4 parked** findings.

The final combined OC-158/169/183 check passes **286 tests**, including
Polars catalog ingestion and existing S3 security regressions (two existing
dependency warnings). Repository Ruff/Ty, scoped formatting and
`git diff --check` pass. Independent review found no material issue.

### 2026-09-10 - OC-169 fixed: keep request error logs redacted across wrappers

The production middleware stack reproduced the reported bypass with direct
exceptions, explicit causes, implicit context, groups, notes and source-line
credentials. The real S3 connector's `ConnectionError` wrapper still exposed
its cause. The adjacent `LoggingMiddleware` also wrote raw exception text and
request URL/user-agent metadata; the generic exception handler retained raw
logging/persistence and used ambient `format_exc()`. The initial HTTP/direct
handler regressions produced **10 failures / 1 passing control**.

Both application middleware layers now use the existing `redact_credentials`
policy before constructing messages, arguments or structured extras.
Tracebacks are formatted from the supplied exception, including the entire
chain, then redacted. Ordinary log output contains that safe traceback;
raw `exc_info` is omitted so a formatter cannot reconstruct the original.
URL/user-agent redaction runs before request-start logging and applies to
success and handled-error paths too. Request objects remain unchanged.
The generic handler also redacts its route/message/traceback before passing
them to `_record_error`, preserving type and status.

Verification: **196 tests pass** across the new **13 integration cases**,
existing S3 redaction, backend infrastructure, error monitoring, two backend
coverage suites and the OC-158 serializer/settings suites. The new cases
check formatted output and LogRecord fields, redacted source snippets and
notes, S3 wrapping, 200/422/500 metadata, fallback calls outside an active
exception, CORS, request IDs and normal response headers. Two existing
dependency deprecation warnings remain. Repository Ruff/Ty, scoped formatting
and `git diff --check` pass. Independent review found no material issue in the
stated scope. The configuration guide and **v0.8.20** release
notes describe the behavior. The queue now has **55 open / 4 parked** findings.

Scope: the shared redactor's recognized formats are unchanged. This is not a
global logging filter: third-party/server logging and other handled-error
persistence paths are outside this fix. In particular, Starlette can re-raise
after its outer generic handler; normal route errors consumed by the custom
error middleware do not reach that server fallback.

### 2026-09-10 - OC-158 fixed: preserve literal text in the sync JSON helper

The sync serializer's early `str(obj)` comparison erased eight literal tokens
and custom objects whose textual representation matched them. The async
helper preserved the same text. The new regressions initially produced
**11 failures / 18 passes**: eight token cases, both DataFrame export formats,
and a custom category object failed; ordinary strings and real missing-value
controls passed.

Removed `_handle_special_string_values` and handle `pd.NA` / `pd.NaT` by
identity alongside `None`. Existing numeric handlers retain real NaN/infinity
cleanup. Text now passes unchanged through nested dictionaries/lists and
DataFrame records/columns, matching the async helper for these inputs.
The class docstring and serialization guide document this distinction; the
obsolete direct-handler test explanation was corrected. No production file
imports this module, so this closes a compatibility-helper defect without
claiming a current HTTP response change. Broader serializer consolidation is
outside this fix.

Verification: **115 tests pass** across `test_serialization_values.py`,
`test_serialization_extra.py`, `test_pagination_and_thresholds_settings.py` and
`test_patch_coverage_backend.py`. Regressions cover all 15 text controls,
nested values, both DataFrame orientations, a custom text fallback and 11
actual missing/non-finite scalar values, including numpy and Decimal NaN.
Strict JSON round trips use `allow_nan=False`. Scoped Ruff, formatting,
repository Ty and `git diff --check` pass; the guide's example was executed
successfully. Independent review found no material issue. Release notes are
under **v0.8.20**. OC-158 moves to the archive,
leaving **56 open / 4 parked** findings.

### 2026-09-10 - OC-156 fixed: stop offering ROC AUC as a threshold objective

The backend scored thresholded class labels with `roc_auc_score`, which is
balanced accuracy for binary predictions, while the UI advertised ROC AUC.
The actual probability-ranking score is independent of the decision cutoff.
Before the fix, **6 backend regressions failed / 36 controls passed** and
**2 frontend regressions failed / 33 controls passed**, including binary
numeric/string labels, misleading multiclass guidance, both HTTP write paths,
the metric dropdown and hydration of a legacy saved `roc_auc` set.

New threshold preview/save requests now reject `roc_auc` with HTTP 400 and an
explanation pointing to `balanced_accuracy`. The unused hard-label AUC scorer
and binary-only exception are removed. Accuracy, F1, Precision, Recall and
Balanced Accuracy remain supported. Model-evaluation and hyperparameter-search
ROC AUC are unchanged; the core training-time threshold fallback already uses
Balanced Accuracy and needs no change.

The dropdown no longer offers ROC AUC, and saved legacy metadata cannot select
it during hydration. Existing saved cutoffs can still be read, enabled,
disabled or cleared. Their metadata and values are preserved; replacing them
requires a new preview with a supported objective. Both the threshold user
guide and API guide explain this compatibility policy. Release notes are under
**v0.8.20**, alongside OC-151.

Independent review identified an enabled Save button on legacy sets: although
the dropdown showed F1, Save still posted the preserved `roc_auc` metadata.
Two failing UI regressions confirmed it for `roc_auc` and training-only
`f1_weighted`. Hydration and Save now share the supported-metric list; Save
stays disabled with an instruction until a supported preview succeeds, while
using or clearing the stored cutoffs remains available.

Verification: **43 backend tests**, **43 focused frontend tests**, and the full
**2,444 frontend tests / 188 files** pass. **5 Chromium scenarios** pass,
including legacy thresholds and keyboard-selected Balanced Accuracy requests
at 1440px and 390px, plus preview/save/toggle/reload/clear. The mobile fixture
uses the existing Collapse Sidebar action before opening evaluation; no layout
change or forced click was needed. Frontend lint and source-wide CCN 10,
repository-wide Ruff and Ty, and changed Python formatting pass. The combined
OC-151/156 backend check passes **159 tests** (15 dependency deprecation
warnings). Follow-up independent review found no remaining issue. The live
queue now contains **57 open / 4 parked** findings.

Final TypeScript/production build and all **11** unchanged bundle budgets pass;
rebuilt assets use main entry `index-C9Juv0Fp.js`. The build caught a test-only
Playwright-style `exact` option in a Testing Library query; removing that option
preserved exact string matching and the focused **43 tests** pass again.
Explicit lint of the browser regression passes. Logs are under ignored
`tmp_repro_artifacts/oc156-*-final.log`; `git diff --check` passes.

### 2026-09-10 - OC-151 fixed: release finished-job chart buffers

The cleanup hooks were still unused in production. Before the fix, the new
real-SQLite lifecycle suite produced **18 expected failures / 2 passing
controls**: successful, failed, raised-error and cancelled executions all
retained live chart points, as did successful cancellation requests. The
controls confirm that failed cancellation commits must preserve the buffers.

`execute_pipeline` now clears trials and boosting iterations in its `finally`
block, after result persistence or failure handling. Both the single-job and
parallel-batch task entry points use this service. The shared cancellation
manager clears local buffers only after its database commit; execution cleanup
also removes points emitted by a fitting thread after cancellation. Unrelated
active jobs and the existing LRU bounds are preserved. Storage remains
process-local; this change does not add cross-process live history.

The regression suite covers fixed/tuned jobs through both task entry points,
all four outcomes, chart availability at the successful commit, persisted
chart scores read from a new session, unrelated active-job buffers, and
successful/failed cancellation commits. The focused backend run passes
**116 tests** (two dependency deprecation warnings). The API guide and buffer
docstrings describe cleanup and the completed-job metrics fallback; release
notes are under v0.8.20. OC-151 moves from the live queue to the archive.

Final verification: the same **116 tests** pass after documentation/test
cleanup; repository-wide `ruff check .`, the full configured `ty check`
scope, formatting of all five touched Python files, and `git diff --check`
pass. Independent review found no correctness or regression issue. The live
queue now contains **58 open / 4 parked** findings.

### 2026-09-10 - OC-229 fixed: obsolete inspector responses cannot replace current details

Continued after OC-228 on `6e887335`, preserving its pending changes. The old
inspector characterization expected the first response to replace the second;
it now asserts the intended behavior. Before the production fix, the expanded
modal suite had **9 expected failures / 8 passing controls**. Besides stale
data, it reproduced obsolete error and loading writes, job-to-pipeline target
changes, same-node close/reopen, stale retries and Strict Mode effect cleanup.

`NodeInspectorModal.tsx` follows the existing Error Log generation pattern.
Every fetch, including Retry, gets a new generation. Success, error and finally
blocks may write only while that generation is current; effect cleanup
invalidates it on selection change, close or unmount. HTTP methods and response
contracts are unchanged. An old transport request may finish, but its result
cannot change the current inspector. Existing node navigation, provenance,
not-found behavior and current-request retries remain intact.

Focused inspector/ModalShell verification passes **28 tests / 2 files**.
The new `e2e/node-inspector-races.spec.ts` passes **4 Chromium scenarios** at
1440px and 900px through the real Error Log node link. It holds a first
opening's HTTP request, closes and reopens the same inspector, then releases
the old success/error after current details are visible. It also verifies
upstream navigation, a failed request followed by Retry, Escape and restored
opener focus. Independent review found no blocking issue. Frontend lint,
source-wide CCN 10 and explicit lint of the new browser test pass.

Final combined-tree verification passes **2,440 tests / 188 files**,
TypeScript/production build and all **11** unchanged bundle budgets. Production
assets were rebuilt with both OC-228 and OC-229; the final browser rerun again
passes all four inspector scenarios.

The generation assignment uses the same explicit ref write as Error Log;
ESLint's effect-ref check rejected the initial increment shorthand. No lint
rule or complexity limit was relaxed. Local evidence is under ignored
`tmp_repro_artifacts/oc229-*`. The platform walkthrough documents historical
inspection and clarifies that OC-228 concerns the **+ icon beside Casting
Rules**, whose tooltip is Add Casting Rule, rather than adding a canvas node.
Release notes are under v0.8.20. OC-229 moves to the archive; the live queue
is now **59 open / 4 parked**.

### 2026-09-10 - OC-228 fixed: casting Add preserves existing rules

On `6e887335`, the public settings reproduction changed `{ age: 'int' }` to
`{ age: 'float' }` when Add was clicked with all available columns assigned.
The old characterization test intentionally pinned that behavior during the
CCN refactor; it is now a regression for the corrected contract. Before the
production fix, **3 regressions failed / 48 existing tests passed**, including
the exact unintended callback payload.

`CastTypeNode.tsx` derives the next unassigned column from its existing
schema/drop-filtered list. Both the handler and button use that value; there
is no fallback to the first assigned column. Missing schema still disables
Add, hidden saved rules remain intact, and removing a rule makes its column
available again with the existing Float default.

Focused settings/serialization/body-preview verification: **232 tests passed**.
The full suite passes **2,431 tests / 188 files**. **5 Chromium scenarios** pass
in `preprocessing-experiment-graphs.spec.ts`, including exhausted casting rules,
removal and keyboard re-addition at 1440px and 1100px, plus the actual Preview
payload `{ age: 'int', species: 'string', height: 'float' }`. The first browser
attempt correctly blocked Preview because the new fixture left its neighboring
replacement/binning nodes unconfigured; completing those settings through the
real UI resolved the fixture failure. No production workaround was needed.

Independent review found no blocking issue. Frontend lint and source-wide
CCN 10 pass. TypeScript/production build and all **11** unchanged bundle budgets
pass; rebuilt assets are included, with main entry `index-nNPBPCLt.js`.
User instructions are documented in the platform walkthrough,
and release notes are under v0.8.20. Local evidence is under ignored
`tmp_repro_artifacts/oc228-*`. OC-228 moves to this archive; the live queue is
now **60 open / 4 parked**, with the remaining priorities unchanged.

### 2026-09-10 - Frontend Plotly peer follow-up: npm audit reaches zero

The [second dependency pass](frontend_static_analysis_review_2026-09-10.md#plotly-peer-dependency-follow-up)
removes the two npm entries left by the compatible updates below. An npm alias
installs the existing official GL3D distribution as `plotly.js`, satisfying the
React wrapper's peer dependency while keeping its supported factory entry.
GL3D stays at **3.5.0**, with identical tarball URL/integrity and production
assets. The unused full-Plotly/MapLibre chain is removed: **253 fewer lock
entries**, no new entries and no forced major overrides or scanner exclusions.

Final main-tree `npm audit` exits 0 with **zero known vulnerable package
entries**; this is package/advisory matching, not a security audit of embedded
prebuilt code. Clean isolated `npm ci`, peer resolution, **2,429 Vitest tests**,
lint, source CCN 10, TypeScript/build and all **11** bundle budgets pass.
**18 Chromium tests** pass on a fresh development server, and the new 3D PCA /
PNG export regression also passes against the production preview. Independent
review found no blocking implementation issue.

The previous entry records the intermediate 14-to-2 result; the current npm
result is **14 -> 2 -> 0**. Codacy must rescan to establish its own finding's
status. OC-214/215 remain closed; queue totals remain **61 open / 4 parked**.
The frontend README documents the alias/factory contract. Release notes: v0.8.20.

### 2026-09-10 - OC-214/215 fixed: compatible frontend dependency updates

This is the first-pass record; the follow-up above closes its remaining two
npm entries.

Continued the [scanner report](frontend_static_analysis_review_2026-09-10.md#dependency-audit-continuation)
from `988b5a43` on `0820`. Fresh npm audit reproduced **14 affected package
entries**. Targeted compatible updates remove twelve entries; **2 critical
entries remain**, representing the same MapLibre advisory and its Plotly parent.
The exact Codacy lockfile advisory is still unavailable, so no Codacy closure
or application exploitability is inferred from npm's result.

OC-214 originally resolved `postcss@8.5.26 -> nanoid@3.3.17`; the parent range
permits patched 3.3.18, now present in both the lockfile and installed tree.
The original PostCSS call uses `nanoid/non-secure` with constant size 6, so
the advisory's zero-size-generator condition was not established in that path.
OC-215 originally resolved selector-parser 6.1.2 under Tailwind 3.4.18 and
postcss-nested 6.2.0. Both ranges now resolve to 6.1.4, which includes the
6.1.3 security fix. Neither package appears in the final npm audit.

The update also aligns Vitest/UI/coverage at 4.1.11 and refreshes vulnerable
Browserslist, baseline mapping, query-string/decoder, fflate and js-yaml.
Only **27 existing lock entries** change; no packages are added or removed.
Independent review verified all **52** dependency/peer constraints referencing
changed entries and found no blocking or important issue.

Verification: baseline and final **2,429 tests / 188 files passed**; the final
run also exercised V8 coverage. ESLint, source-wide CCN 10, TypeScript/build and
all **11** bundle budgets pass. A separate clean `npm ci --ignore-scripts`
installation preserves the same lockfile hash. Production artifacts are
unchanged; generated module evidence shows zero MapLibre modules and zero
rendered bytes from the full Plotly package. The installed critical dependency
remains open because patched MapLibre 6.4.1 exceeds Plotly's declared range.
See the scanner report for browser results and environment limitations.

OC-214/215 move from the live queue to the archive: **61 open / 4 parked**.
The remaining MapLibre item stays in the scanner follow-up; other audit IDs
and parked/deferred priorities are unchanged. Release notes: v0.8.20.

### 2026-09-10 - Codacy follow-up: remove temporary verification files

The user supplied 21 findings in `tmp_repro_artifacts/verify_security_followup.py`.
Commit `6a9dbbe5` had included seven local investigation scripts/reports from
that directory. The verifier checks one specific working-tree snapshot and
depends on local logs; it is not an application or CI entry point. Python's
optimized mode does remove its assertions, but this is a local verification
script, not a production security control. Its Git calls use argument lists
and fixed call sites; no user-controlled command execution was established.

The user requested deletion, so all seven files were deleted and their removals
staged. The directory was added to `.gitignore` and the root Codacy exclusions.
Existing Ruff and ty configuration already excludes this temporary directory.
The durable evidence summary remains in the
[scanner report](frontend_static_analysis_review_2026-09-10.md#codacy-temporary-artifact-follow-up).
No application code, tests, dependency versions or security rules were changed.
Verification confirmed seven deleted/untracked files, matching Git ignore rules,
valid Codacy YAML with only the new temporary exclusion, and clean diff checks.

The separately reported `frontend/ml-canvas/package-lock.json` finding has no
advisory details in the supplied output. A read-only npm audit reports 14 package
entries, but those cannot be equated with the single Codacy issue. Its exact
message/package/advisory is requested before selecting a dependency fix.
Existing OC rows and the **63 open / 4 parked** queue are unchanged.

### 2026-09-10 - job-log numeric regex performance

Reproduced the three reported super-linear numeric patterns in `JobLogs.tsx`:
when a long digit run lacked a duration, percentage or decimal suffix, the
unanchored expressions retried from each following digit. A fixed-width
negative lookbehind before the first digit now prevents those redundant starts.
Its placement after the optional minus preserves negative values following
digits. Rule order, colors, log text and control behavior are unchanged.

The 16,000-digit duration case measured **354.922 ms -> 0.555 ms** in the bounded
local comparison; this is an illustrative sample, not a latency guarantee.
Old/new full highlighting matched on **100,000** generated messages; independent
review found no blocking issue. Focused tests **61/2**, full Vitest **2,429/188**,
**7 Chromium tests** (including a 100,000-digit log), lint, strict CCN 10,
TypeScript/build and all **11** bundle budgets passed. The informational CCN 8
inventory remains **133/96**; no production function was split.

See the [scanner follow-up](frontend_static_analysis_review_2026-09-10.md#job-log-regex-performance-follow-up)
for evidence and scope limits. External scanner status still needs a fresh scan.
The existing OC queue remains **63 open / 4 parked**. Notes are under v0.8.19;
this change and the preceding lookup follow-up were committed in `6a9dbbe5`.

### 2026-09-10 - frontend scanner follow-up: explicit lookup registries

Reviewed the four reported dynamic-regex/callable-object warnings against
`5a63fe55`. The test regex interpolated a static array length; operational
URL parsing already checked a fixed kind whitelist; comparison rows supply
fixed field labels. No application command-injection path was established.
Direct helper calls with inherited names did reproduce invalid return values
or exceptions, so CV/tuning lookups now use Map registries with the normal `-`
fallback. Operational parser lookup also uses Map, retaining its mapped type
and all ten parser bodies; the test uses a literal string matcher.

Original regression selection: **107 passed / 4 expected new failures**;
final **111/4** (tests/files) passed. Full Vitest **2,411/187**, normal ESLint,
strict CCN 10, TypeScript/build, all **11** bundle budgets and **9 Chromium
tests** passed. Independent review found no blocking issue. The largest
functions in the three reported files remain **8, 3 and 10**; file-level
complexity deltas do not represent individual function CCNs. The global
informational report stays **133/96** at CCN 8 and the strict backlog stays zero.

See the [review and metric explanation](frontend_static_analysis_review_2026-09-10.md).
External scanner confirmation remains pending; no exemptions or limits changed.
The queue remains **63 open / 4 parked**, with existing OC rows and DRIFT-01
unchanged. Release notes are under v0.8.19; the follow-up was committed in `6a9dbbe5`.

### 2026-09-10 - frontend batch 11: complete the CCN 10 backlog

Base `73c0b7e7` on `0819`. Three disjoint Astra 6 implementers handled execution/
tuning/evaluation hooks, Jobs/Experiments/monitoring screens and remaining node
settings; the primary handled Navbar, notification details and ModalShell.
All **22 remaining violations across 20 original production files** are removed.
Extractions stay module-local, preserving public exports, defaults, state/effect/
request ownership, callbacks, numeric policies and DOM semantics. Fresh independent
spec and quality reviews passed for all four groups, with source/test diffs still
matching their reviewed packages. See the [batch 11 plan](frontend_ccn_refactor_batch11_2026-09-10.md).

The source-wide strict CCN 10 gate now **passes: 0 violations in 0 files**;
the largest source function is **10**, down from 24. The informational CCN 8
report remains unchanged in policy and lists **133 functions in 96 files**, all
optional 9/10 candidates (previously 139/101). The [inventory](frontend_ccn_remaining_2026-09-10.md)
records every optional function and its measured location. No thresholds,
exemptions, dependencies, source scope or bundle budgets changed.

Original/final characterization passed for each group: hooks **195 tests / 10
files**, pages **245/29**, settings **330/15**, shell and consumers **34/4**.
These selections overlap. Frozen-source full Vitest passed **2,402 tests / 186
files**. Normal ESLint, explicit new-browser-test lint with `--no-ignore`, project
`tsc --noEmit`, production build and all **11 bundle budgets** pass. Main bundle
`index-B8ImxiHA.js` is **324.5 KiB gzip / 325 KiB budget**; all **251** relative
built imports resolve. Expected error-path/jsdom stderr also occurs in original
consumer tests; the full suite has no failed tests.

The complete Chromium suite passed **133 tests**, without failures or retries.
Two new cases exercise Count, TF-IDF and Hashing controls at 1440/1100px: actual
schema filtering, independent node values, empty max features, panel expansion/
collapse, selection round trips and submitted Preview payloads. HTTP is mocked;
registry definitions, graph stores, controls and conversion are real. Expanded
panel screenshots were inspected. Existing suites cover job and experiment
navigation, training/tuning, thresholds, notifications, modals and accessibility.

Release notes are under v0.8.19; v0.8.18 and earlier entries are unchanged.
No additional functional finding was discovered or silently fixed. The audit
queue remains **63 open / 4 parked**; OC-223/225/226/227/228/229 remain open,
OC-71/72/73/185 stay parked, and DRIFT-01 stays deferred. Final integration
review and the manual frontend checklist are recorded in the plan. The user
authorized committing this verified batch on 2026-09-10.

### 2026-09-10 - frontend batch 10: canvas inspection, data screens and analysis

Base `c56d6cca` on `0819`. Three disjoint Astra 6 implementers simplified eight
canvas/inspection entries, seven dataset/model screens and eight drift/analysis
entries; the primary handled metric and ensemble formatting. All **34 selected
violations across 24 original files** are removed, including the new port helper.
Public contracts, state/effect ownership, DOM, callbacks, numeric fallbacks and
submitted payloads are preserved. All four groups passed fresh independent spec
and quality reviews; their final source/test diffs match the reviewed packages.
See the [batch 10 plan](frontend_ccn_refactor_batch10_2026-09-10.md).

Strict backlog: **56 -> 22 functions**, **44 -> 20 files**, **maximum 26 -> 24**.
Informational CCN 8: **147/106 -> 139/101** (functions/files), with **117** optional
functions at 9/10. Thresholds and source-wide scope are unchanged; strict exits 1
for the remaining backlog while all selected entries/helpers pass 10. The
[inventory](frontend_ccn_remaining_2026-09-10.md) was regenerated from both reports.

Original/final characterization passed for every group: canvas **145 tests / 13
files**, data **35/8**, analysis **39/10**, formatting and consumers **100/3**.
Frozen-source full Vitest: **2,330 tests / 176 files**. Normal ESLint, explicit
new-browser-test lint with `--no-ignore`, project `tsc --noEmit`, production build
and all **11 bundle budgets** pass. `index-BjHkPcSK.js` and all **251** relative
built imports resolve. Existing jsdom AggregateError stderr occurs in original
consumer tests too; no executed Vitest test failed.

Final complete Chromium run: **131 passed**, no failures or retries. Four new
cases at 1440/1100px cover real dataset preview/profile conversion and reopen,
drift upload payloads, histogram geometry, sort, filter, threshold re-evaluation
and actual CSV download. HTTP is mocked; stores, controls and charts are real.
Existing suites cover canvas connections, inspection, focus/resize, shortcuts,
settings, model operations and chart navigation. Screenshots were inspected.
The first full run had one blank-page timeout in an unchanged theme test; three
unchanged diagnostic repeats and the final full run passed. Its cause remains
unconfirmed; no production change or weaker assertion was made for that timeout.

Concise release notes are under v0.8.19; older releases are unchanged. The existing
inspector response race was filed separately as **OC-229**, leaving **63 open /
4 parked**. OC-223/225/226/227/228 stay open, OC-71/72/73/185 stay parked and
DRIFT-01 stays deferred. Final integration review is recorded in the plan;
the user reported the frontend checks working and authorized the batch commit.

### 2026-09-10 - OC-229 filed: an earlier inspector request replaces newer node details

Batch 10 characterization reproduced this against original `c56d6cca`
`NodeInspectorModal`: open node `first`, select `second`, resolve the second
request and then the first. The modal replaces **Second response** with
**First response** although the selection and canvas link identify the newer
node. `fetchNode` writes data, errors and loading without a request identity
guard; its original and final source at lines 59-78 is unchanged.

Evidence: `frontend/ml-canvas/src/components/shared/NodeInspectorModal.test.tsx:99`,
case `currently allows an earlier node response to replace a newer one`.
Both original and refactored sources pass the same **145 tests / 13 files**;
logs and original copies are under `tmp_repro_artifacts/ccn10-task1/`
(`original-extended-tests.log`, `final-tests.log`). Independent review confirms
that the race predates this refactor; no matching prior tracker item exists.

Filed separately as **OC-229**, preserving behavior during CCN extraction.
Guard success, error and loading updates by the current request and modal
lifetime; cover reversed responses, retries, closing and node navigation.
The queue is now **63 open / 4 parked**. OC-71/72/73/185 stay parked and
DRIFT-01 remains deferred.

### 2026-09-10 - frontend batch 9: preprocessing, graph utilities and experiment charts

Base `45c3f142` on `0819`. Three disjoint Astra 6 implementers handled ten
preprocessing settings entries, seven graph rule/diff/layout/export utilities,
and seven experiment chart/summary entries; the primary handled VariableCard.
All **31 original CCN violations in 25 files** are removed. Helpers retain
cohesive responsibilities and the same public contracts, state ownership,
numeric fallbacks, DOM and submitted payloads. All four task groups passed
independent spec and quality review. See the
[batch 9 plan](frontend_ccn_refactor_batch9_2026-09-10.md) for the exact scope.

The measured strict backlog fell **87 -> 56 functions**, **69 -> 44 files**,
and **maximum 32 -> 26**. Informational CCN 8 changed **159/113 -> 147/106**
(functions/files), including **91** optional functions at 9/10. Thresholds and
source-wide scope remain unchanged; global strict exits 1 for the remaining
backlog while every selected entry/helper passes 10. The
[inventory](frontend_ccn_remaining_2026-09-10.md) was regenerated from both reports.

Original/final public characterization passed for every group. Final shared
Vitest: **2,268 tests / 168 files**; normal ESLint, explicit new-browser-test
lint, project `tsc --noEmit`, production build and all **11 size budgets** pass.
Generated `index-BYpfjj1i.js` and all **251** relative built imports resolve.
Existing jsdom AggregateError stderr also occurs in the original experiment
suite; the executed suite has no failed tests. Complete Chromium integration:
**127 passed**, with no failures or retries in the final run. Five new cases
cover responsive settings/Preview payloads, real PNG/SVG file downloads,
classification threshold and run/split changes, regression plots and cluster
charts/tables. Plot assertions wait for visible, stable SVG geometry. Existing
suites also cover connection rejection, leakage feedback and Pipeline Diff.
The primary inspected settings and rendered-chart screenshots; final independent
integration review passed. Browser tests use mocked HTTP endpoints, while
production stores, controls and chart libraries remain real.

Concise release notes are under v0.8.19; older releases are unchanged.
VariableCard has no current production consumers and is verified through its
public component tests, not an invented browser route. The existing casting
overwrite was separately filed as OC-228 below, leaving **62 open / 4 parked**.
OC-223/225/226/227 remain open; OC-71/72/73/185 stay parked and DRIFT-01 deferred.
The user reported the frontend checks working and authorized the batch commit.

### 2026-09-10 - OC-228 filed: adding a casting rule overwrites an existing rule

Batch 9 characterization reproduced this against the original `45c3f142`
`CastTypeNode` public settings: schema contains only `age`, its rule is `int`,
and clicking **Add Casting Rule** emits `{ column_types: { age: 'float' } }`.
The button is disabled only for an empty schema. When all columns are assigned,
the unchanged add handler falls back to the first available column and replaces
its type. The same test passes after extraction; independent review confirmed
the original behavior and found no matching existing tracker item.

Evidence: `InvalidValueReplacementNode.test.tsx`, case `preserves the existing
cast add fallback when every column has a rule`; original 49 tests and final
126 tests passed. Logs: `tmp_repro_artifacts/frontend_ccn_batch9_task1/`.
This CCN refactor preserves the behavior and records it separately as **OC-228**.
Fix by disabling/no-oping Add when no unassigned column remains; verify that
removing a rule permits adding it again without changing other types.
Queue becomes **62 open / 4 parked**; earlier parked work is unchanged.

### 2026-09-10 - frontend complexity refactor batch 8: verified

Committed the previously verified batch 7 with DCO sign-off as `6af9a600`, then
continued on `0819` with three independent Astra 6 owners and separate reviews.
The strict threshold remains 10 and the informational threshold remains 8.

| Scope | Selected functions before -> after |
|---|---|
| Graph store | validation collector 17 -> 2; connection confirmation 32 -> 6; history equality 12 -> 6 |
| Operational context | serialization 19 -> 5; reference parsing 28 -> 3; description 11 -> 2 |
| Scaling / Outlier | settings 27 -> 9 / 26 -> 9; scaling feedback callback 19 -> 3; outlier feedback 22 -> 4 and recommendations 15 -> 5 |

Every selected entry and extracted helper is at or below 10. The graph retains
validation ordering, synchronous confirmation/cancellation and existing history
semantics. Record codecs preserve identity types, optional values, query order
and accessible descriptions. Settings preserve defaults, numeric/empty values,
upstream column choice, feedback and recommendation ordering.

Original-source characterization passed **63**, **81** and **50** tests before
extraction; related final suites passed **355**, **138** and **129** tests.
Independent review strengthened the dragging-data history test to keep node
count fixed, and the Outlier traversal test to distinguish dataset IDs and
breadth-first/equal-depth precedence. Their follow-up suites passed **59** and
**52** tests. All three independent task reviews passed.

Final verification: **2181 Vitest tests / 160 files**, normal ESLint, TypeScript
noEmit, production build and all **11 bundle budgets** pass. The complete
Chromium confirmation passed **122/122**. The new five browser cases cover
native fan-in cancel/accept and undo/redo, encoded Jobs links through reload/back,
and Scaling/Outlier method/parameter fields in real Preview requests at compact
and desktop widths. After review, these five passed again with explicit checks
for completed width transitions and one/two-column settings layouts; refreshed
screenshots were inspected. Shared responsive behavior was not changed.

**Browser diagnostic:** the first complete run passed 121 tests and timed out
once before Add Dataset appeared in an unchanged validation-navigation test.
Its screenshot showed a blank startup page. The unchanged navigation suite then
passed **15/15** with traces, followed by the **122/122** complete confirmation.
No root cause or product fix is claimed. Logs are under `tmp_repro_artifacts/`:
`ccn8-browser-all.log`, `ccn8-browser-navigation-repeat.log`,
`ccn8-browser-all-confirmation.log` and `ccn8-browser-review-final.log`.

The measured strict backlog fell **98 -> 87 functions** and **73 -> 69 files**;
maximum remains **32**. The informational report changed **166/111 -> 159/113**
(functions/files), with **72** optional functions at 9/10. The global strict
command still exits 1 for the remaining backlog; selected scopes pass.
See [the inventory](frontend_ccn_remaining_2026-09-10.md) and
[batch 8 plan](frontend_ccn_refactor_batch8_2026-09-10.md).

Generated assets were rebuilt (`index-51eF5Jsl.js`); index entries and all 251
relative built JS/CSS imports resolve. Concise notes are under v0.8.19; older
release text is unchanged. The existing drag-end history gap was separately
filed as OC-227 below, leaving **61 open / 4 parked**. OC-223/225/226 remain open;
OC-71/72/73/185 stay parked and DRIFT-01 stays deferred. This new batch is
verified; the user confirmed frontend checks and authorized its commit.

### 2026-09-10 - OC-227 filed: completed node drags bypass undo history

During batch 8, the original `useGraphStore.ts` at `6af9a600` passed the public
characterization `currently ignores both in-progress drag positions and the
drag-end transition`. After clearing history, the test sends positions `(10,20)`
and `(30,40)` with `dragging: true`, then `(50,60)` with `dragging: false` through
`onNodesChange`. The position changes, but `pastStates` remains empty. A later
non-drag position update creates one entry; undo restores `(50,60)`, not the
position before dragging.

The equality rule ignores position differences whenever **either** node is
dragging, so the transition ending a drag is suppressed too. FlowCanvas passes
React Flow's changes directly to this action. Evidence is a public-store test,
not a newly executed pointer-drag browser reproduction. Batch 8 preserves this
behavior in `graphStore/historyEquality.ts`; it does not repair it silently.
Original characterization: 63 tests passed; extracted graph/consumer set:
355 tests passed. Logs: `tmp_repro_artifacts/batch8-task1-original-tests.log` and
`tmp_repro_artifacts/batch8-task1-final-tests.log`.

**Fix target:** retain one snapshot from before a drag and record the completed
single/group movement once, with real-pointer undo/redo coverage. Avoid adding
selection-only or per-frame history entries. Queue: **61 open / 4 parked**;
OC-71/72/73/185 remain parked. This is separate from CCN simplification.

### 2026-09-10 - frontend complexity refactor batch 7: verified

Plan: [`frontend_ccn_refactor_batch7_2026-09-10.md`](frontend_ccn_refactor_batch7_2026-09-10.md).
Prior batch 6, OC-224 and the deferred DRIFT-01 specification are signed commit
`4a6db08e`; the user subsequently authorized this next batch's signed commit.

Three independent Astra 6 owners characterized original behavior before
extracting cohesive state/presentation; fresh independent reviewers approved
each task for spec compliance and code quality. Entry CCN: Layout **29→5**,
JobsDrawer **28→5**, JobCard **24→1**, SegmentationSettings **28→10**. Every
entry/helper is <=10; readable 9/10 functions remain intact. State, effects,
subscriptions, public props, labels, CSS, payloads and ordering are preserved.

Original public tests: Layout **22**, JobsDrawer/JobCard **49**, Segmentation
**29**. Integrated verification: **2,045 Vitest tests / 158 files**, normal
ESLint, TypeScript/Vite build and all **11 bundle budgets** pass. Generated
frontend assets are rebuilt. The strict all-source CCN10 gate intentionally
still fails on **98 functions / 73 files / max32**, down from 105/77; the CCN8
informational report passes with **166 functions / 111 files**, including 68
optional functions at 9/10. The inventory is regenerated from both executed logs.

The complete Chromium confirmation passes **117/117** with mocked HTTP and real
UI/stores. The first four-worker run passed 116/117, with an unchecked threshold
checkbox after a click in an unchanged test. That suite passed three focused
repetitions (9/9), and the full two-worker confirmation passed without source
changes. A delayed-defaults diagnostic did not reproduce the proposed cause;
no threshold fix is claimed. Logs retain the initial failure and later evidence.
New browser cases cover mobile/desktop navigation, Jobs filtering/details and
compact/expanded Segmentation edits plus submission payloads; representative
screenshots were visually inspected. Live training remains a user-side check.

Two original defects found during characterization are recorded separately:
OC-225 and OC-226 below. Queue: **60 open / 4 parked**. DRIFT-01 remains deferred;
there are no new backend/Core/API/store/registry/dependency changes in this batch.
Final independent integrated review reports PASS with no introduced regression
or scope violation; the delivery audit confirms all 31 scoped text files, queue
counts, generated entry references, preserved older notes and unchanged CCN policy.

### 2026-09-10 - OC-226 filed: stale segmentation defaults overwrite newer config

The original `SegmentationSettings.tsx` model effect captured the entire config
and onChange callback, then called `onChange({ ...config, hyperparameters:
defaults })` when its request settled. It had no active-request or unmount guard.
The same logic now lives in `segmentationSettings/useSegmentationModels.ts`;
batch 7 deliberately preserves it rather than mixing a concurrency fix into
the extraction.

Executed characterization before production edits: the same-model case changes
reference_column from `old` to `new` and execution_mode from parallel to merge
while defaults are pending; completion calls the original callback with the old
values. The reversed-response case resolves a newer DBSCAN request, unmounts,
then resolves the older K-Means request; the old callback emits K-Means defaults
after the newer result. Both cases pass on original and extracted source in
`SegmentationSettings.test.tsx` (`keeps the original callback and config...` and
`retains out-of-order definitions...`). Logs:
`tmp_repro_artifacts/ccn7-segmentation-original-tests.log` and
`ccn7-segmentation-final-tests.log`.

Fix target: default seeding must preserve current non-parameter edits, ignore
superseded model requests, and avoid writes after settings unmount. Verify the
controlled parent/graph outcome, not only a callback spy. Update these deliberate
characterization assertions when implementing that separate fix. No matching
queue finding existed; queue is now **60 open / 4 parked**. DRIFT-01 remains a
deferred enhancement outside the audit counts.

### 2026-09-10 - OC-225 filed: job detail loses its dialog name

During batch 7's browser characterization, pressing Enter on a real JobCard
opens the existing detail view, but the resulting `role="dialog"` has no
accessible name. `JobsDrawer.tsx` at base `4a6db08e` references
`aria-labelledby="jobs-drawer-title"` on the persistent panel; the referenced
heading exists only in the unselected history branch. `jobs/jobDetails/JobHeader.tsx`
also renders its Back action as an unlabeled icon-only button. Both conditions
predate this refactor; no existing tracker/queue entry covered them.

Evidence: `tmp_repro_artifacts/ccn7-browser-probe2.log` reproduced the missing
named dialog at both 1440px and 1100px while the Job Details heading remained
visible. Source inspection confirms the dangling label reference and unlabeled
Back control. The browser scenario now locates the detail dialog by its heading
to continue checking current behavior; that test does not claim accessibility
is repaired. Keep the pure extraction unchanged and fix the dialog label plus
Back name in a separate regression-covered change. Queue: **59 open / 4 parked**;
OC-223 and the deferred DRIFT-01 enhancement retain their existing scope.

### 2026-09-10 - OC-224 fixed: compare raw uploads with saved raw source data

`backend/monitoring/drift_reference.py` resolves the selected training node's
ancestors from its saved graph and uses the unique loader's persisted snapshot.
Shared-loader branches resolve to one source; unrelated graph branches cannot
choose the baseline or exclude columns. The router excludes the target and
explicit drop-column configurations symmetrically, without intersecting raw
columns with encoded/derived model feature names. No upload fitting, pipeline
replay, threshold change, model retraining or artifact rewrite is involved.

The baseline contains the rows actually loaded, before preprocessing/splitting;
sampled loaders retain their sample. It can include validation/test rows, so
the user's model card remains 120 training rows while drift now compares
**150 reference / 150 current rows**. Graphless legacy jobs keep their previous
reference contract; its preprocessing stage cannot be independently established.
Missing/ambiguous source metadata for a graph-backed job produces an explicit
error and attempts to record a failed check rather than publishing drift scores.
Explicit drops follow the existing bundle convention, not full feature lineage.

TDD: **9 failed / 1 passed** before repair; **21 passed** after, including 11
existing target regressions. New coverage includes actual engine training with
log-before-split and shared branches, existing pandas/Polars source snapshots,
real distribution/schema changes, encoded categorical raw inputs, unrelated
branches, missing/ambiguous sources and graphless fallback. Wider relevant
backend/Core suites pass **214 tests** (493 dependency/runtime warnings).
Ruff check/format and scoped ty pass; the post-narrowing focused run again
passes 21. An independent Astra review found no outstanding issues.

The running local API was checked with job
`bf22d1e1-f61a-4434-a89c-77662d42cfc9` and the user's original CSV:
**HTTP 200, 0/4 drifted features, all four PSI values 0.0, no missing/new columns,
severity none**. This created history alert **27**; previous results were not
rewritten. Logs: `tmp_repro_artifacts/drift-reference-{red,green,integration,
confirm,types}.log` and `tmp_repro_artifacts/drift-bf22-live-api.log`.

The [drift guide](../../docs/user_guide/drift_monitoring.md) and v0.8.19 notes
explain the raw-source contract and existing-job compatibility. OC-224 moves to
the closed backend rows; queue returns to **58 open / 4 parked**. OC-223 stays
open. The user approved this fix and the preceding frontend batch for a signed
commit, including the deferred DRIFT-01 enhancement specification in the queue.

### 2026-09-10 - OC-224 filed: drift reference/upload preprocessing mismatch

The user trained `bf22d1e1-f61a-4434-a89c-77662d42cfc9` and checked the same
Iris CSV. Reading its stored graph/reference and executing the real calculator
reproduced PSI **8.30354, 8.32062, 5.77086, 3.18667** (mean **6.3954**), with
all four features marked drifted. Reference: 120 rows; current: 150 rows.
The graph applies `GeneralTransformation(method=log)` before the splitter.
For SepalLengthCm, the saved reference mean is **1.91567**, versus raw **5.84333**.

`_run_data_loader` initially saves a raw reference, but `_run_transformer`
overwrites it with the splitter's already transformed training partition.
`calculate_drift` parses the upload and compares it directly with that reference.
This combines different preprocessing stages and predates the batch 6 frontend
refactor. The saved graph also merges a dropped-Id branch and the log branch;
the fix must not assume a single linear preprocessing chain.

Evidence: `tmp_repro_artifacts/inspect_drift_bf22.py` and
`tmp_repro_artifacts/drift-bf22-inspection.log`, using the job's stored artifacts
and `uploads/data/3cfaca74-96d0-483b-bac5-76e5cde20057.csv`. No existing finding
in the tracker/queue describes this mismatch. Filed open before repair;
queue **59 open / 4 parked**. Scope: consistent reference/current data stages,
existing-job compatibility, and preserved detection of real distribution/schema
changes. Do not fit transformations on the upload or lower alert thresholds.

### 2026-09-10 - frontend complexity refactor batch 6: verified

Plan: [`frontend_ccn_refactor_batch6_2026-09-10.md`](frontend_ccn_refactor_batch6_2026-09-10.md).
Baseline `534d1e94`, branch `0819`; the prior strict-10/report-8 policy and
inventory are committed with DCO sign-off. Three independent Astra implementers
and separate reviewers covered these scopes:

| Scope | Previous maximum | Final maximum | Preserved behavior evidence |
|---|---:|---:|---|
| Audit Log and 6 helpers | 31 | 10 | 23 original/refactored page tests: server filters/facets, summaries, expansion, refresh and stale responses |
| Drift alert modal and 5 helpers | 31 | 8 | 27 original modal tests / 36 related final tests: evidence, links, actions, promises and state lifetime |
| Feature Generation and 12 helpers | 30 | 7 | 32 original settings tests / 102 related final tests: operations, columns, controlled updates, reveal, feedback and validation |

All three pure extractions passed independent specification and quality review.
The Audit page itself is now 6; two cohesive helpers remain at 10 intentionally.
OC-222 was then repaired separately with red/green evidence and independent
review (see below). OC-223 was reproduced on the original modal and remains
open for a separate repair of modal and parent request lifetimes.

Final integration: **157 Vitest files / 1,998 tests**, **111 Chromium tests**,
normal ESLint, TypeScript/Vite production build and all **11 bundle budgets
pass**. Main bundle: **320.1 KiB gzip / 325 KiB**. The six new browser cases
exercise actual audit/drift pages at 1440px/900px and editable feature settings
at 1440px/1100px, including keyboard focus and panel expansion. Routes are
mocked; these checks do not establish live-backend integration. Logs:
`tmp_repro_artifacts/ccn6-{vitest,lint,build,size,playwright}-final.log`.

All selected source files and helpers pass scoped CCN 10. The source-wide
strict gate still exits **1** on the remaining **105 functions / 77 files /
maximum 32**, down from 112 functions / 80 files. The informational CCN 8
report exits **0**, listing **167 functions / 109 files**, down from 172 / 110.
The 62 functions at 9 or 10 are optional improvements, not strict violations.
Evidence: `tmp_repro_artifacts/ccn6-{gate,report}-final.log`; the linked
[remaining-work inventory](frontend_ccn_remaining_2026-09-10.md) is refreshed.

Generated assets and concise v0.8.19 notes are updated. No backend, shared API,
store, registry, dependency or lockfile change. Queue: **58 open / 4 parked**
after closing OC-222 and filing OC-223; parked decisions are unchanged.
The user completed frontend checks and subsequently requested a signed commit
with the separately verified OC-224 drift correction.

### 2026-09-10 - OC-222 fixed: explain full-history audit filtering

After the extraction passed review, the stale loaded-page hint was replaced
with: "Actor, action kind and time filters apply across the full history before
the page limit." The paragraph and CSS are retained; only copy and its existing
public-page assertion changed. The route builds history, applies actor/kind/date
filters, reverses the matches and then caps the response, so the hint matches
the current backend behavior.

The corrected assertion first failed **1 of 23 tests**; all **23 then passed**.
Scoped ESLint/CCN 10 and whitespace checks pass. Independent review confirmed
the two-file diff, red/green logs and backend filtering order. Evidence:
`tmp_repro_artifacts/ccn6-task-5-{red,green,eslint,diff-check}.log`.
OC-222 moves to the closed frontend rows; OC-223 remains open. Release note:
v0.8.19. No API behavior change.

### 2026-09-10 - OC-223 filed during drift modal characterization

The original modal at `534d1e94` clears its current note whenever a pending
`onApplyDisposition` resolves truthy, even after `alertId` has changed. The test
`retains asynchronous truthy-result semantics across an alert switch` submits
`first` for one alert, rerenders with alert ID 8, types `second`, then resolves
the old callback. The current note becomes empty. Original-source characterization
passes **27 tests**, including this case; evidence:
`tmp_repro_artifacts/ccn6-task-2-baseline-expanded-tests.log`.

This predates the extraction and is preserved in its characterization, not
introduced by the refactor. No duplicate was found in the tracker/open queue.
The eventual fix should assess `useDriftAlertDetail` alongside modal note state:
the parent also applies asynchronous responses without request ownership checks,
but wider parent behavior has not yet been execution-reproduced in this finding.
Filed open for a separate request-lifetime repair; queue **59 open / 4 parked**
including the separately filed OC-222 wording defect.

### 2026-09-10 - OC-222 filed during Audit Log characterization

Original `AuditLogPage.tsx` at `534d1e94:495-496` says: "Filters apply only to
the loaded page; the backend currently supports dataset and page limit only."
The public-page test `states that filters span the whole history, not just the
page` renders this hint alongside the full-history footer; original-source
characterization passes **21 tests**. Other cases verify actor/kind/time arguments
are sent. The real route `backend/ml_pipeline/_internal/_routers/pipelines_io.py`
accepts these filters in `get_pipeline_audit_log`, confirming the hint is stale.

Evidence: `tmp_repro_artifacts/ccn6-task-1-original-tests.log`, subsequently
refreshed with **23 original-source tests** after adding two characterization
cases. No duplicate was found in the tracker/open queue. Filed separately from
the pure extraction; correct the hint after that review. Queue at filing:
**58 open / 4 parked**.

### 2026-09-10 - frontend policy: strict CCN 10, informational CCN 8

The accepted limit for subsequent frontend refactors is **CCN <= 10 per
function**. `complexity:check` enforces 10 across all `src` TypeScript; the
informational `complexity:report` stays at 8. No per-file exceptions or explicit
scope list. This supersedes the earlier source-wide CCN 8 policy below.

The strict command reports **112 errors in 80 files** (exit 1, maximum 32).
The informational report lists **172 warnings in 110 files** (exit 0), including
60 functions at CCN 9 or 10 that now pass the gate. This is a threshold change,
not 60 additional code fixes. Remaining required work is recorded with source
links, function labels and CCN values in
[`frontend_ccn_remaining_2026-09-10.md`](frontend_ccn_remaining_2026-09-10.md).
Prioritize complex responsibilities and the largest functions; avoid splitting
readable code solely to lower a number.

Normal ESLint and TypeScript/Vite build pass; production assets are unchanged.
Evidence: `tmp_repro_artifacts/frontend-ccn10-{check,lint,build}-2026-09-10.log`
and `frontend-ccn8-report-2026-09-10.log`. The workflow and v0.8.19 note match
these thresholds. No runtime code or audit status changed; **57 open / 4 parked**.

### 2026-09-10 - frontend CCN 8 gate expanded to all source files

The user confirmed the batch 5 frontend checks, requested a signed commit, and
chose source-wide enforcement while the remaining backlog is being addressed.
`complexity:check` now runs
`eslint src --ext ts,tsx --rule "complexity: [error, 8]" --max-warnings 0`.
The explicit file/folder list is removed; new TypeScript files under `src` are
automatically covered. The workflow retains the informational report and runs
the same strict command, with no increased limit or baseline exemption.

Verification: the new command exits **1 with 172 complexity errors**, as expected
from the existing backlog in 110 files (maximum 32). This supersedes the scoped
gate's passing status recorded below; the source-wide gate will fail until those
violations are resolved. Normal ESLint and a fresh TypeScript/Vite build pass;
the rebuilt main asset hash is unchanged. No runtime code changed in this
follow-up. Evidence: `tmp_repro_artifacts/ccn-global-{check,lint,build}-2026-09-10.log`.
The v0.8.19 CI note reflects the final policy; queue stays **57 open / 4 parked**.

### 2026-09-10 - frontend complexity refactor batch 5: verified

Plan: [`frontend_ccn_refactor_batch5_2026-09-09.md`](frontend_ccn_refactor_batch5_2026-09-09.md).
Baseline `15aa043a`, branch `0819`; the removed CCN 10 exception is not restored.
Independent Astra implementers and reviewers covered three separate scopes:

| Scope | Previous maximum | Final maximum | Preserved behavior evidence |
|---|---:|---:|---|
| Comparison table and 11 helpers | 34 | 8 | 11 original/refactored public-component tests: scoring groups, ties, config precedence, graph alignment and expansion |
| Pipeline diff and 5 helpers | 33 | 8 | 35 original/refactored related tests: snapshots, failures, cancellation, Swap, graph diff and labels |
| Error Log and 10 helpers | 32 | 8 | 28 original/refactored page tests: filters, diagnostics, actions, export, links and asynchronous results |

All three pure extractions passed independent spec/quality review. A separate
OC-221 repair then increased Error Log coverage to 38 cases (see below); its
independent review found no actionable issue. New actual-page browser coverage
passed four cases against both original and extracted source at 1440px/900px,
including keyboard Swap, focus retention, comparison folds and error filters.

Final integration: **156 Vitest files / 1,926 tests**, **105 Chromium tests**,
ESLint, strict CCN 8, TypeScript/Vite production build and all **11 bundle
budgets pass**. Main bundle: **319.7 KiB gzip / 325 KiB**. Exact logs are under
`tmp_repro_artifacts/ccn5-{vitest,lint,gate,build,size,playwright,report}-final.log`.
Browser tests use mocked API routes; no live-backend integration claim is made.

The full report falls from **181 functions / 113 files / maximum 34** to
**172 functions / 110 files / maximum 32** above CCN 8. The three entries and
all 26 helper modules are included in the strict gate without raised limits.
Next active hotspots include `useGraphStore.confirmConnection` (32),
`AuditLogPage`/`DriftAlertModal` (31) and `FeatureGenerationNode` (30).
Unreferenced `VariableCard` (32) is not selected merely to reduce the report.

Generated assets and concise v0.8.19 notes are updated. The v0.8.18 and older
sections still match branch base `784649d9`. Queue: **57 open, 4 parked** after
OC-221 closure; parked decisions remain unchanged. No backend, dependency or
lockfile change. The user confirmed the frontend checks and requested a signed
commit; the subsequent source-wide CCN policy change is recorded above.

### 2026-09-10 - OC-221 fixed: retain the latest Error Log request

After independent review of the behavior-preserving extraction, a separate fix
adds request generations to `errorLog/useErrorLogPage.ts`. Every filter load and
refresh supersedes older HTTP and pipeline requests. Success, rejection and
loading settlement check the same generation; effect cleanup also invalidates
pending work on filter change or unmount. HTTP results remain usable before
pipeline logs arrive. Requests are not aborted, and API arguments are unchanged.

The desired latest-request behavior failed **11 of 38 tests** against the reviewed
extraction; all **38 now pass**. Coverage includes both completion orders, stale
success/error/loading, same-filter refresh, delayed pipeline results/rejections
and unmount. Independent Astra review found no actionable issue. Scoped ESLint,
strict CCN 8 and project TypeScript pass; the hook's maximum remains 7.
Evidence: `tmp_repro_artifacts/ccn5-task-5-{red,green,ccn,types,metrics}.log` and
the batch 5 independent review. Full Vitest also passes **1,926 tests**.

OC-221 moves from the live queue to the closed frontend rows above. The queue
returns to **57 open and 4 parked**; OC-71/72/73/185 remain parked. Release note:
v0.8.19. No backend changes.

### 2026-09-09 - OC-221 filed during Error Log characterization

The original `ErrorLogPage.tsx` at `15aa043a` applies every completed `load()`
without checking whether its filter scope is still current. The new public-page
test `retains the current completion-order behavior when searches overlap`
starts searches `older` then `newer`, resolves the newer response first and the
older one last, and observes `older result` while the input still says `newer`.
It passes against original source, establishing this as a pre-existing defect.
Pipeline results and loading/error state use the same unguarded completion style;
a repair must cover stale completions without making HTTP results wait for logs.

Filed separately from the behavior-preserving CCN extraction. At filing the queue
had **58 open and 4 parked** findings, before the request-lifetime repair above.

### 2026-09-09 - branch 0819 release-note placement corrected

At the user's request, move all release notes added on branch `0819` since
`784649d9` (the `0818` base), including OC-218/219/220 and the frontend CCN
workflow/refactors, into v0.8.19. The v0.8.18 and older release contents match
the branch base. Related tracker and frontend CCN plan references now point to
v0.8.19. This documentation correction changes no implementation or queue status.

### 2026-09-09 - OC-220 fixed: anchor Resampling target suggestions

Replace native datalist rendering with `TargetColumnField`, using the existing
Radix Popover dependency to anchor an editable listbox to the input. Matching
suggestions retain upstream dropped-column filtering; selection still updates
the same target setting and arbitrary column names remain editable. Keyboard
selection, Escape dismissal, Tab and outside-click focus are preserved.

Both new unit regressions went from red to green; all **119 related settings and
accessibility tests** pass, including instance-local listbox IDs. Two Chromium
regressions pass for docked/expanded settings: the list follows input position
and width during viewport/panel resizing, with click and keyboard selection.
Independent review found no behavioral defect; its test-fixture lint finding
was corrected. The original native-popup offset remains user-reported, as
documented in the filing; replacement geometry is verified in the browser.

Full **1,869 unit tests** in 154 files, **101 Playwright tests**, ESLint, strict
CCN 8, TypeScript/Vite build and all 11 size budgets pass.
Main bundle: 318.9 KiB gzip / 325 KiB. Generated
assets and the v0.8.19 release note are updated. OC-220 is closed; the live queue
returns to **57 open and 4 parked**. No backend or dependency changes.

### 2026-09-09 - OC-220 filed from Resampling manual feedback

The user reports Target Column suggestions opening away from the input. Both
baseline `63745ca3` and the extracted control use native input/datalist markup;
the application has no DOM popup whose geometry follows the resizable settings
panel. The exact native-popup offset is not observable through Playwright DOM
geometry, so it is recorded as user-reported rather than locally reproduced.
Two unit cases fail before adding an application listbox (**2 failed / 34 passed**),
and docked/expanded browser cases fail because that listbox does not exist.
The repair will anchor editable suggestions to the input and verify bounds after
viewport/panel resizing, while preserving target values and dropped-column filtering.

### 2026-09-09 - frontend complexity refactor batch 4: verified

Plan and evidence:
[`frontend_ccn_refactor_batch4_2026-09-09.md`](frontend_ccn_refactor_batch4_2026-09-09.md).
Baseline `63745ca3` contains the signed batch 3 commit and OC-218/219 repairs.
Three Astra 6 agents and the primary handled four independent scopes:

- **Resampling:** settings 40 -> 7; two adjacent modules at most 8. All 34 new
  characterization cases pass against original and extracted code; related
  accessibility suite (117 tests) passes. Public config/defaults/validation/conversion,
  upstream target synchronization, parser behavior and last-run results remain.
- **EDA page:** page 30 -> 1, content renderer 40 -> 8; six modules at most 8.
  All 19 cases pass original/extracted source. All 25 API/store/query-key
  expressions are AST-equivalent; dataset URL selection, report polling,
  filter application, history and tab data guards retain their behavior.
- **Node inspection:** entry 38 -> 7; six modules at most 7. The 41 related
  original/extracted tests and five browser scenarios pass. Receipt selection, focus,
  Input/Output matching, complete-schema comparison and sample bounds remain.
- **Imputation:** settings 35 -> 7; four modules at most 8. All 17 new cases pass
  original/extracted source; 198 related tests pass. Public node definition,
  validation, defaults and preview are unchanged; method controls, column
  filtering, recommendation merging and execution feedback retain behavior.

Each scope passed independent spec and code-quality review. Final integration:
**1,867 unit tests** across 154 files, **99 Playwright tests**, full ESLint,
expanded strict CCN 8 gate, TypeScript/Vite build and all 11 size budgets pass.
Main: 318.2 KiB gzip / 325 KiB; EDA: 84.0 KiB / 140 KiB. Assets were rebuilt.
Final whole-batch Astra review also passed with no actionable finding.
CI now includes four more entry files and 18 helper modules; no limit or waiver
changed. Global report: **183 warnings across 115 files, maximum 34**, versus
189 / maximum 40 before this batch. Next largest: ComparisonTableView (34),
PipelineDiffView (33), VariableCard/useGraphStore/ErrorLogPage (32).

Release notes are under v0.8.19. This behavior-preserving maintenance closes no
audit finding; the live queue stays **57 open and 4 parked**, including the
unchanged parked OC-71/72/73/185 decisions. New batch 4 work is not committed.

### 2026-09-09 - OC-219 fixed: require saved thresholds before toggling

The evaluation hook now tracks saved-threshold availability separately from
the preview and enabled flag, resetting per job and hydrating behind the
existing stale-response guard. Successful Save sets availability and enables
predictions; successful Clear resets both. Preview leaves availability alone,
and a saved but disabled set can still be enabled again. The checkbox is
disabled without a saved set, with inline Preview/Save guidance and a corrected
Save tooltip. The user guide documents post-training tuning without retraining.

Regression evidence: **6 failed / 22 passed** before repair, then **33 focused
frontend tests passed**. The complete frontend suite passes **1,795 tests**;
unchanged backend service/router suites pass **37 tests**. The threshold browser
spec passes all **3 tests**, including real-page Save/Clear state transitions,
disable/re-enable, job switching and reload with a stateful API fixture.
Independent review found no introduced issue. Lint and the strict CCN gate pass.
Final production build, all 11 bundle budgets and all **99 browser tests** pass.
OC-219 is closed; the queue returns to **57 open and 4 parked**. Release note:
v0.8.19. No backend persistence semantics changed.

### 2026-09-09 - OC-219 filed from threshold tuning feedback

The user reproduced a successful Preview followed by toggle HTTP 400:
`Job has no saved tuned thresholds to toggle.` Backend service and route tests
confirm Preview does not persist and Save both persists and enables. The UI
allows the invalid toggle and its Save tooltip incorrectly promises inactivity.
Two new UI cases reproduce the enabled checkbox with no saved set, before and
after Preview; four hook cases also fail before saved-state tracking is added
(**6 failed, 22 passed**). The behavior predates the batch 3 extraction. Filed
in the live queue before production repair.

### 2026-09-09 - frontend complexity refactor batch 3: verified

Plan: [`frontend_ccn_refactor_batch3_2026-09-09.md`](frontend_ccn_refactor_batch3_2026-09-09.md).
Baseline `abe5ea7d`: 204 functions above CCN 8 in 123 of 555 files.
Three Astra 6 agents implemented separate Ensemble, EDA and Evaluation scopes;
primary implemented Feature Selection and owned shared verification/delivery.
Different owners reviewed each implementation and the final public interfaces.

- Ensemble settings: entry-file maximum **52 -> 6**, five helper modules at
  most **8**. The original 12 tests grew to **31** characterizations passing on
  both original and extracted code; implementer/reviewer differential probes
  matched **4,000** and **2,000** connected-model configurations respectively.
  The separately filed OC-218 repair brought this suite to **39** passing tests.
- EDA variable row: entry maximum **51 -> 7**; two helpers at most **7**.
  EDA sidebar: **47 -> 4**; five helpers at most **7**. **34 tests** pass against
  both original and extracted implementations. Independent review matched
  **84 DOM comparisons** with shared chart/UI stubs. Numeric formatting,
  export state, filter parsing, exclusions and navigation remain equivalent.
- Feature Selection: settings component **46 -> 7**; five helper modules at
  most **8**. The cohesive body preview remains **9** after merging equivalent
  K-count cases. This entry stays report-only; its helper folder is gated.
  **16 new** characterizations plus existing coverage passed **48 tests** on
  original and extracted code; independent review passed **131** related tests.
- Evaluation view: entry maximum **43 -> 1**; nine helper modules at most **8**.
  **21 tests** pass, including five new original-passing characterizations.
  Mutation lifetimes, retries, loading/error precedence, split selection and
  threshold/chart props were checked against the original implementation.

Final independent integration review found no introduced actionable issue:
all five public interfaces are unchanged and **99 helper imports** resolve.
The first full build found exact-optional-property and test typing errors;
these were repaired without widening public contracts. OC-218 is the only
intentional product behavior change in this batch and is documented separately.

Final verification passed **1,788 Vitest tests** across 152 files (**76 new**),
**98 Playwright tests**, full ESLint, the expanded strict CCN 8 gate,
TypeScript/production build and all 11 bundle budgets. Main bundle:
**317.3 KiB gzip / 325 KiB budget**; no limits raised. Browser tests stub backend
HTTP and cover UI wiring; actual data/model execution remains a useful manual
check. Existing jsdom diagnostics and vendor/proxy warnings remain.

The global report now has **189 functions** above CCN 8 in **119 of 584 files**;
highest CCN fell from **52 to 40**. All 26 new production helper modules are
gated, alongside four entry files. Next report-only hotspots: `ResamplingNode`
(**40**), `EDAPage` (**40**), `NodeInspectionPanel` (**38**), `ImputationNode`
(**35**) and `ComparisonTableView` (**34**). Release notes are under v0.8.19.
OC-218 was filed and fixed; the queue remains **57 open and 4 parked**.

### 2026-09-09 - OC-218 fixed: preserve connected ensemble CV seed zero

Reproduction against `abe5ea7d` and the initial extraction showed a connected
seed of `0` becoming `42`. A separate executable probe enabled shuffled CV and
confirmed both fixed and tuned parameter builders forwarded `42`; a source
seed of `7` was forwarded correctly. The finding was recorded before repair.

`ensembleSettings/connectedModels.ts` now uses a nullish fallback for
`cv_random_state`, preserving zero and retaining the local seed only when the
source value is absent. The existing synchronization characterization was
updated, and eight cases exercise the real `convertEnsembleNode` fixed/tuned
branches with seed `0`, seed `7`, and absent-source fallbacks to `42` and `0`.
Before repair: **3 failed, 36 passed**. After repair and independent review:
**39 passed**, strict CCN 8 lint and TypeScript passed.

Type repairs preserve the public interface: an undefined time-column candidate
is omitted (the prior comparison already discarded it), and the first model
has an explicit guard. **32 comparisons** verified unchanged time-column patch
semantics. Other connected-model defaults are unchanged. Parameter-only
inspector synchronization is retained: the converter rereads wired model
parameters, so the no-op does not establish a stale training-payload defect.

The closed row moved here and its evidence left the live queue; it returns to
**57 open and 4 parked** rows. The concise fix note is under v0.8.19.

### 2026-09-09 - OC-218 filed during frontend ensemble review

The original and extracted Ensemble settings both replace a connected model's
CV seed `0` with the current ensemble seed (`42` in the reproduction). A
separate executable probe with shuffled CV confirmed fixed and tuned parameter
builders both forward `42`; a source seed of `7` is forwarded correctly.
The truthiness fallback is the cause. Filed in the live queue before repair;
the fix will preserve explicit zero and verify both submission modes.

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
`EvaluationView.tsx` (**43**). The v0.8.19 notes were updated. No audit finding
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
remains **57 open** and **4 parked**. The v0.8.19 notes were updated.

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
The v0.8.19 note was updated. This CI addition closes no audit finding, so the
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
