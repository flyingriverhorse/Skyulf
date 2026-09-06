# Opus core audit — fix tracker

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
day out of the OC-150 fix pass. They are tracked below with reproduction evidence
in the log; the historical baseline counts above are unchanged.

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
grepping for consumers, **not** reproduced by execution, and none is fixed: each
changes behaviour rather than documentation, so all four are filed open by
decision for a later session. Historical baseline counts remain unchanged.

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
| OC-177 | 🟠 | Pandas `DummyEncoder` changes a known category's encoding with batch composition: after fitting `[1.0,2.0]`, `1.0` encodes as known alone but all-zero when accompanied by `2.5` (`preprocessing/encoding/dummy.py:60-64`) | small | ⬜ open |
| OC-163 | 🟠 | `LagFeatures` / `RollingAggregate` sort X without reordering tuple y on both engines — `[3,1,2]` times become `[1,2,3]` while targets remain `[300,100,200]`, silently training on wrong labels (`preprocessing/time_series/lag.py:45,81`, `rolling.py:63,119`) | small | ⬜ open |
| OC-164 | 🟠 | `get_fitted_split()` on new data replaces a trained pipeline's preprocessing while retaining its old model — the same input's prediction changed from 50 to −950 (`pipeline/_pipeline.py:234`) | small | ⬜ open |
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
| OC-71 | 🟠 | **No authentication or authorization anywhere on the API** (`main.py:373-395`, `database/models.py:151-159`) — **confirm intent first**: single-tenant self-hosted → documentation task; multi-tenant → highest-priority item in the entire report (scaffolded `User` model + dead `AUTH_FALLBACK_*` settings suggest the latter was intended) | decision + ~1 week | ⬜ open — confirm intent | PARKED!
| OC-72 | 🟡 | Insecure-by-default config: unset `FASTAPI_ENV` fails open to wildcard CORS + credentials (`config/factory.py:26`, `main.py:359-366`) | small | ⬜ open — with OC-71 |
| OC-73 | ⚪ | `DataSource.credentials` documented encrypted, stored plaintext JSON (`database/models.py:107`) | small | ⬜ open — with OC-71 |

### Ongoing — remove the hiding conditions

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-76 | 🟠 | Cross-engine parity tests cover 9 of 100 nodes and never compare applied output — directly caused OC-04/23/24/58 to go unnoticed | ~3 days | ⬜ open |
| OC-77 | 🟠 | `--maxfail=1` hides real failure count; `--cov-fail-under=45` vs 98.4% actual (two flag changes, `.github/workflows/skyulf-core-tests.yml:82-87`) | mechanical | ✅ fixed 2026-09-06 — floor raised 45 → 90 against a measured 96% (CI run 34032836551), and `--maxfail=1` removed here **and** in `backend-tests.yml`, which carried the same flag unfilled. See the log entry |
| OC-01 | 🟠 | `skyulf.__version__` ambiguous: stale `0.5.8` dist-info shadows real `0.8.8` (path-order dependent) — packaging-integrity cluster — re-verified 2026-09-05: the venv holds exactly one dist-info (`skyulf_core-0.8.13`), the stale `0.5.8` is gone, and `skyulf.__version__` reports `0.8.13` | small | ✅ resolved by the 0.8.13 install refresh |
| OC-02 | 🟠 | Dev editable install dangling; `import skyulf` fails outside repo — packaging-integrity cluster — re-verified 2026-09-05 by importing from a CWD outside the repo: resolves to `skyulf-core/skyulf/__init__.py` at `0.8.13` | small | ✅ resolved by the 0.8.13 install refresh |
| OC-78 | 🟡 | `py.typed` declared in packaging metadata but file does not exist — packaging-integrity cluster | 1 line | ✅ fixed 2026-09-06 — created `skyulf-core/skyulf/py.typed`, verified first through setuptools' own `build_py` and then inside an actually built wheel. See the OC-05/22/78/79/112/132/141 log entry |
| OC-79 | 🟡 | `joblib` imported at module scope but not in `install_requires` — packaging-integrity cluster | 1 line | ✅ fixed 2026-09-06 — `joblib>=1.3.0` declared in core `install_requires`, and in root `pyproject.toml`/`requirements.txt` for the backend's identical undeclared-import gap. See the log entry |
| OC-81 | ⚪ | No `License ::` classifier / SPDX field — packaging-integrity cluster | 1 line | ✅ fixed 2026-09-06 — owner decided `skyulf-core` = Apache-2.0 with backend + frontend staying AGPLv3, declared **statically** in `skyulf-core/pyproject.toml` so it emits PEP 639 `License-Expression:` rather than the deprecated free-text field, and the three files that contradicted the decision reconciled to it. See the OC-81 log entry |
| OC-03 | 🟠 | Systemic `infer_output_schema` int→float misprediction across 22 nodes — one sweep + parametrized test (predicted schema == actual schema for every node) | ~1 day | ⬜ open |
| OC-09 | 🟡 | Narrow `ruff select` hides ~500 missing docstrings + 84 unused args — **last**, widening first would bury the signal | done — `ARG` declined by decision (121 in-scope sites measured) | ✅ fixed 2026-09-06 — `F401`/`F841`/the `D` family now enforced on `skyulf-core/skyulf/` + `backend/`, 904 docstrings hand-written (82 of them invisible to ruff because `D1xx` is privacy-gated on the whole dotted module path), and `ARG` declined by decision; closed by the owner as “fixed as much as we did, no need to continue”. See the 2026-09-05 and 2026-09-06 OC-09 log entries |

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
| OC-130 | 🟠 | Typo in `FASTAPI_ENV` silently disables the entire production security posture (wildcard CORS w/ credentials, DEBUG=True, no SECRET_KEY check) (`config/factory.py:27-32`) — **worse than filed**: a second, unfiled channel — `FASTAPI_ENV` is not a `Settings` field and pydantic-settings never exports dotenv values into `os.environ`, so the bare `os.getenv` could not see a `.env`-only `production` either; both now fail closed through `resolve_environment()` | small | ✅ fixed 2026-09-05 |
| OC-150 | 🟠 | S3 error "sanitiser" matches credential key names case-sensitively — S3 403 bodies + replayable presigned URLs logged verbatim; duplicated in two files (`connectors/s3.py:31-37`, `artifacts/s3.py:67-73`) — **worse than filed**: executed against real shapes the old helper was a *no-op* on all three leaks and exposed the **secret access key** (the audit only ever demonstrated key IDs and signatures), while separately destroying benign text (`key=reports/2026/q3.csv` → `redacted sensitive S3 error`); both copies deleted in favour of one shape-based `redact_credentials()` | small | ✅ fixed 2026-09-05 |
| OC-153 | 🟠 | Multi-input merge silently switches column-wise→row-wise when a branch changes row count — 5-row set + filtered branch yields 8 rows, 3 duplicates, zero UI warnings (`_merge.py:338-348`) — repro came out **9 rows / 4 duplicates**; fixed by warning, not raising, so appending datasets still works | small | ✅ fixed 2026-09-05 |
| OC-154 | 🟠 | Serving-time feature-order reindex (fix F-02) fails open on column mismatch — returned 213.00 where truth is 321.00 (`deployment/service.py:438-442`) | small | ✅ fixed 2026-09-05 |
| OC-155 | 🟠 | Legacy predict path zero-fills missing features and returns a prediction normally; caller never sees a warning (`deployment/service.py:457-462`) — **worse than filed**: the zero-fill also mutated the caller's DataFrame in place | small | ✅ fixed 2026-09-05 |
| OC-145 | 🟡 | Crashed cross-validation returns the same `{}` sentinel as a disabled one — job reports success with missing `cv_*` metrics (`_node_runners.py:871-907`) | small | ⬜ open |
| OC-151 | 🟡 | Trial-buffer `clear_*` hooks documented but never called — 110.9 MB retained for process lifetime (`realtime/trial_buffer.py:56-59,103-106`) | small | ⬜ open |
| OC-156 | 🟡 | `roc_auc` threshold-tuning objective scores hard predictions — bit-identical to `balanced_accuracy` (`threshold_tuning_service.py:77-92`) | small | ⬜ open |
| OC-158 | 🟡 | Sync/async JSON serializers disagree: sync nulls 8 of 15 legitimate strings (`"nan"`, `"NaT"`, `"<NA>"`, `"inf"`…), async nulls none; 603-line module production-dead but test-covered (`serialization.py:369,435-446`) | half day | ⬜ open |
| OC-131 | ⚪ | Diagnostics fail open — PSI returns `0.0` on any numeric failure (`profiling/drift.py:474-476`) | 1 line | ✅ fixed 2026-09-06 — `drift.py` was the only module under `profiling/` with no logger, so all three fail-open paths (PSI, KL, and the uncastable-column drop the finding missed) now warn; the finite `0.0` contract is kept and documented, since `None` is a three-layer change and `inf` cannot survive `JSONResponse`'s `allow_nan=False`. See the log entry |
| OC-132 | ⚪ | Dead `dropped_features` branch (key appears exactly once in repo) (`graph_utils.py:534-537`) | 1 line | ✅ fixed 2026-09-06 — branch deleted after confirming no test, fixture or writer references the key; the superseding runtime path is recorded in a docstring at the site so the branch is not re-added. See the log entry |
| OC-152 | ⚪ | Two raw-SQL executors accept unconstrained query strings, zero callers — latent injection sink (`async_connection_manager.py:243-268`) — **broader than filed**: `AsyncSQLiteConnectionManager` carried a byte-identical pair, so four dead sinks were deleted, not two | small | ✅ fixed 2026-09-05 |
| OC-157 | ⚪ | `first_wins` merge strategy reverses output column order, contradicting its docstring (`_merge.py:221-236`) — fixed by dropping the reversed iteration, so order is strategy-independent by construction | small | ✅ fixed 2026-09-05 — with OC-153 |
| OC-159 | ⚪ | Empty filter dict compiles to WHERE-less `DELETE FROM data_sources`/`UPDATE`; dead call path today (`async_sqlite_queries.py:129-146`) | 1 line | ✅ fixed 2026-09-06 — all four sites (sqlite + postgres × delete + update) raise `ValueError` before opening a session; the path is dead today, but `_normalize_filter(None) → {}` means the signature itself accepts the table-wiping input. See the log entry |
| OC-169 | 🟡 | Filed while fixing OC-150 — the global `ErrorHandlerMiddleware` logs `{exc}`, `traceback.format_exc()` **and** `exc_info=True` with no redaction, so any *uncaught* exception whose message or frames carry a credential leaks it to the log regardless of call-site scrubbing; the S3 paths now redact their own `logger.error` but still `raise ConnectionError(...) from e`, leaving `e` reachable from the chained traceback (`middleware/error_handler.py:53-65`) | small | ⬜ open |
| OC-183 | 🟠 | `SmartCatalog` S3 auto-init is dead for `.env`-only config, and the two docs name different variables — **OC-130's root cause repeating**. `backend/data/catalog.py:556` reads `os.getenv("S3_BUCKET_NAME")`, but pydantic-settings loads the dotenv into the model and never exports it into `os.environ`, so a bucket configured only in `.env` is invisible and `s3_catalog` silently stays `None` (falling back to local disk with no error or warning). Worse, `S3_BUCKET_NAME` is **not a `Settings` field at all**: `config/mixins/aws.py:12` declares `AWS_BUCKET_NAME`, which is what `docs/guides/backend_configuration.md:146` documents, while `README.md:105` documents `S3_BUCKET_NAME` — so following the README sets a variable nothing reads. Needs a canonical-name decision before the one-line code fix | small | ⬜ open |
| OC-184 | 🟠 | `ProductionSettings.SECURITY_HEADERS` is declared and never sent. `_PROD_SECURITY_HEADERS` (HSTS, `X-Frame-Options: DENY`, CSP, …) is assigned at `config/environments.py:84` and referenced nowhere else in the repo — no middleware reads it — so a production boot logs "Running in PRODUCTION mode with enhanced security" while emitting none of those headers. Fixing means adding a security-headers middleware in `main.py::_add_middleware`, where order is load-bearing (CORS must stay outermost), i.e. a behaviour change and not a config fix | half day | ⬜ open |
| OC-185 | 🟡 | Authorization is stubbed in three mutually inconsistent pieces. `database/models.py:157 has_permission` is `return True  # Placeholder` with **zero callers**; `data_ingestion/dependencies.py:26,31 require_data_access`/`require_data_admin` are async no-ops wired to no route; and `data_ingestion/router.py:148,169` hardcode `user_id = 1` under an explicit `# KNOWN-GAP: Auth not implemented yet`, so every source belongs to one user and is visible to everyone. Nothing is exploitable *through* `has_permission` today precisely because nothing calls it — the risk is that the first caller gets an always-yes check shaped like a real API. Needs an authz decision before code | decision + ~1 week | ⬜ open |
| OC-186 | 🟠 | `S3Catalog.exists` skips the option-name mapping that every sibling method applies. `catalog.py:521` builds a throwaway `s3fs.S3FileSystem(**self.storage_options)` from the raw instance options, while `__init__`:275, `load`:452 and `save`:491 all pass through `_prepare_s3fs_options`, which maps `aws_access_key_id`→`key` and `aws_secret_access_key`→`secret` and moves region into `client_kwargs['region_name']`. With AWS-named credentials `exists()` therefore authenticates differently from the methods it is supposed to agree with, and reports `False` for (or errors on) an object `load()` reads fine — so callers that gate on `exists` before `load` take the wrong branch | 1 line | ⬜ open |
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
| OC-112 | ⚪ | Comment and code disagree in the categorical profiler — the comment promises a rendered missing-value marker, the code `continue`s and discards the null category (`profiling/_analyzer/categorical.py:22-30`). *Filed as "disagree about the applied threshold"; the real subject is the null-category marker* | 1 line | ✅ fixed 2026-09-06 — comment-only, no behaviour change; the reasoning for why dropping the null category is correct now lives in the code comment it rewrote. See the log entry |
| OC-121 | ⚪ | polars `Enum` columns invisible to text auto-detection, diverging from pandas `Categorical` (`_helpers.py:148-157`) | small | ⬜ open |
| OC-122 | ⚪ | `TextCleaning` silently ignores unrecognised operation name (`cleaning/text.py:151-153`) | small | ⬜ open |
| OC-90 | ⚪ | Unknown split config keys silently dropped instead of rejected (`preprocessing/split.py`) | small | ⬜ open |

### Remaining — file-coverage closure

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-140 | 🟠 | `InvalidValueReplacement` diverges across engines on non-numeric columns (pandas silently NaNs, polars raises) | small | ⬜ open |
| OC-142 | 🟠 | EDA correlation ratio η exceeds 1.0 with nulls; null-heavy columns rank as strongest association | small | ⬜ open |
| OC-143 | 🟠 | RFE ignores the UI's `k`, silently selecting half the features — **duplicate of OC-25**, same file and line; one fix retires both | small | ✅ fixed 2026-09-05 — with OC-25 |
| OC-141 | ⚪ | `invalid_values` param declared in `node_meta` with zero consumers | 1 line | ✅ fixed 2026-09-06 — key deleted from `node_meta` after re-verifying zero consumers across all three layers and the `.ambr` snapshots, behaviour-neutral because `user_picked_no_columns` keys off `columns`. The other half of the divergence (the params the calculator really reads are still undeclared) is left to **R1 step 1**. See the log entry |
| OC-144 | ⚪ | Geo distance column named `_km` even when the unit is miles | ~~1 line~~ **small, not 1 line** — scoped 2026-09-06 | ⬜ open — **not a one-liner; blast radius measured.** Four code sites (`geo/distance.py:83` pandas apply, `:112` polars apply, `:163` `node_meta` default, `:185` `fit`), **10** assertions in `tests/integration/test_geo_nodes.py` (incl. `:91`, which reads `result_km["geo_distance_km"]` while converting to miles — the mislabel the finding describes, baked into a test), and `docs/reference/preprocessing_nodes.md:630`. **The structural detail that decides the fix:** the two apply-path fallbacks are unreachable in the normal pipeline, because `fit` always writes `output_column` into the artifact — so the *declared* `node_meta` default is what really picks the name. `node_meta` params are a static dict and cannot be unit-dependent, so `f"geo_distance_{unit}"` has to be resolved in `fit` (declaring `""` = auto, or dropping the key), not patched at the four sites independently. Frontend impact is nil — all of `geo/` is UI-unreachable per OC-06 |

### Remaining — cross-cutting & packaging

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-04 | 🟡 | Cross-engine dtype divergence in 3 nodes (int64 vs int8/uint32) (`encoding/dummy.py`, `bucketing.py`) | small | ⬜ open |
| OC-05 | 🟡 | `PowerTransformer` triggers a pandas deprecation that will become an error (`transformations/power.py:101`) | 1 line | ✅ fixed 2026-09-06 — **worse than filed**: casting each destination column to `float64` before the `.loc` write removes a per-column pandas FutureWarning that the surrounding bare `except` would otherwise swallow into a silent no-op, i.e. OC-28's failure mode arriving through OC-05. See the log entry |
| OC-06 | 🟡 | 6 registered nodes unreachable from the UI (incl. all of `geo/`) — `registry.py` vs `frontend/` | small | ⬜ open — R1 step 3 catches this class |
| OC-07 | 🟡 | Node-id naming split 55 PascalCase / 45 snake_case + redundant aliases (`registry.py`) | half day | ⬜ open |
| OC-08 | 🟡 | Public-API name collision: `DatasetProfile` means two things (`skyulf/__init__.py:32-46`) | small | ⬜ open |
| OC-10 | ⚪ | 4 dead `infer_output_schema` overrides that only `return None` (`vectorization/*`) | mechanical | ⬜ open — **re-measured 2026-09-06: five, not four** (`count_vectorizer.py:147`, `hashing_vectorizer.py:135`, `tfidf_vectorizer.py:141`, `tokenizer.py:179`, `sentence_embedder.py:204`). `BaseCalculator.infer_output_schema` already ends in `return None` (`preprocessing/base.py:129`), so all five are behaviourally identical to inheriting. **Recommend folding into OC-03 rather than deleting standalone:** each override carries the per-node *reason* the schema is unknowable (learned vocabulary, model-loaded embedding width, data-dependent column survival), which is exactly the documentation OC-03's parametrized "predicted == actual for every node" test needs beside it, and OC-03 will touch these same five files |
| OC-11 | ⚪ | Mega smoke test silently skips nodes with empty params (`tests/unit/test_all_nodes_smoke.py`) | small | ⬜ open |

### Remaining — encoding / cleaning / imputation / scaling / drop

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-178 | 🟡 | `HashEncoder` hashes the same missing value into different buckets across Polars, pandas object, and pandas nullable string inputs, even with one shared fitted artifact (`preprocessing/encoding/hash.py:45,76`) | small | ⬜ open |
| OC-179 | 🟡 | `DummyEncoder(drop_first=True)` retains a single-category indicator on Polars but removes it on pandas, changing feature width across engines (`preprocessing/encoding/dummy.py:33`) | small | ⬜ open |
| OC-180 | 🟡 | Pandas `TextCleaning(normalize_slash_dates)` crashes on `pd.NA` in a nullable string column; equivalent Polars input preserves the missing value (`preprocessing/cleaning/text.py:35-37,116`) | small | ⬜ open |
| OC-181 | 🟡 | `ValueReplacement` coerces every unrecognized boolean mapping key to `False`: mapping `{"banana": true}` changes `[true,false]` to `[true,true]` on both engines (`preprocessing/cleaning/value_replacement.py:31-32`) | small | ⬜ open |
| OC-182 | 🟡 | Encoder auto-detection ignores pandas `StringDtype` columns: Dummy/Hash encoding silently leaves strings untouched unless columns are selected explicitly (`preprocessing/encoding/_common.py:140`) | small | ⬜ open |
| OC-171 | 🟡 | Pandas `SimpleImputer` silently excludes explicitly selected constant/binary numeric columns for mean/median, leaving missing values unfilled; Polars honors the selection (`preprocessing/imputation/simple.py:173-177`) | small | ⬜ open |
| OC-172 | 🟡 | `StandardScaler` crashes on mixed pandas nullable numeric columns containing `pd.NA`; native sklearn and equivalent Polars input succeed (`preprocessing/scaling/standard.py:144,154`, `engines/sklearn_bridge.py:52`) | small | ⬜ open |
| OC-18 | 🟡 | One-hot/dummy generated names can collide with existing columns (`encoding/one_hot.py:68-92`, `dummy.py:76-99`) | small | ⬜ open |
| OC-21 | 🟡 | WOE additive smoothing not normalized over categories (`encoding/woe.py:130-145`) | small | ⬜ open |
| OC-22 | ⚪ | `TargetEncoder.infer_output_schema` checks an impossible `regression` value (`encoding/target.py:340-360`) | 1 line | ✅ fixed 2026-09-06 — the `("binary", "regression")` passthrough was pinned by a test asserting a prediction for a config sklearn 1.8 rejects outright; changed to `"continuous"` and confirmed it really encodes rather than merely being reachable. See the log entry |

### Remaining — feature generation / selection / vectorization / transformations

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-23 | 🟠 | Polars `ratio` flips the sign of near-zero negative denominators (`feature_generation/_polars_ops.py:97-112`) | small | ⬜ open |
| OC-24 | 🟠 | Polars group aggregates treat null group keys differently from pandas (`_polars_ops.py:222-234`) | small | ⬜ open |
| OC-25 | 🟠 | RFE "K" chosen in UI ignored by backend (`feature_selection/_common.py:236-240`) | small | ✅ fixed 2026-09-05 — closes OC-143 too |
| OC-26 | 🟠 | `HashingVectorizer` UI "none" norm is an invalid sklearn value → crash (`hashing_vectorizer.py:59`) | small | ⬜ open |
| OC-27 | 🟠 | `GeneralTransformation` ignores the UI `standardize` toggle (`transformations/general.py:34-39,138-139`) | small | ⬜ open |
| OC-28 | 🟠 | Box-Cox transform failures silently return untransformed data (`transformations/power.py:97-104`) | small | ✅ fixed 2026-09-06 — the silent path was the `valid_cols` filter, not the `except` (which has logged since the node was created); both engines now share `_fitted_columns_present`, which names the fitted columns the frame lacks, and fail-open is kept by decision. See the log entry |
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
| OC-170 | 🟡 | `validate_leakage_safety()` rejects registered stateless nodes before the split as unknown/data-dependent, including `TextCleaning`, `DateFeatures`, `Casting`, and `feature_target_split` (`leakage.py:140-156`) | small | ⬜ open |
| OC-63 | 🟠 | `artifact_digest` raises `RecursionError` instead of the documented `TypeError` on cyclic graphs (`pipeline/seal.py`) | small | ⬜ open |
| OC-64 | 🟠 | **F-14 only partially fixed** — engine registry global still an unlocked race (`engines/registry.py:60,86-91`) | small | ⬜ open |
| OC-65 | 🟡 | polars `to_numpy()` zero-width "parity fix" does not achieve parity (`engines/polars_engine.py`) | small | ⬜ open |
| OC-74 | 🟡 | `NodeRegistry.list_models()` hides all 4 Ensemble models; `category` arg dead (`registry.py:101-108`) | small | ⬜ open |
| OC-160 | 🟡 | Polars row-filter helpers reserve `__idx__` without collision protection: a valid feature column named `__idx__` crashes `DropMissingRows`; a multi-output `y` DataFrame with that name crashes the X/y synchronisation path (`drop_and_missing/drop_rows.py:65`, `_common.py:19`, `deduplicate.py:40`) | small | ⬜ open |
| OC-161 | 🟡 | Polars clustering evaluation reserves `__skyulf_cluster__` without collision protection: a numeric feature with that name is overwritten by internal labels and then dropped, so centroid calculation crashes with `ColumnNotFoundError` (`modeling/_evaluation/clustering.py:92-101`) | small | ⬜ open |
| OC-162 | 🟡 | Polars time-series CV reserves `__cv_y__` for an unnamed/list target: an input feature with that name is overwritten and dropped before fitting, silently changing the feature matrix (`modeling/cross_validation.py:317-322`) | small | ⬜ open |
| OC-167 | 🟡 | Ambiguous string boundaries in artifact serialization give different fitted label encoders identical pipeline fingerprints, despite encoding the same input as 0 vs −1 (`pipeline/seal.py:52,64`) — distinct from OC-62's pointer instability | small | ⬜ open |

### Remaining — outliers / casting / binning / timeseries / geo

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-173 | 🟡 | Pandas `EllipticEnvelope` reselects valid values by duplicated index labels, can feed NaN back into prediction, then fails open and retains an outlier that a unique-index control removes (`preprocessing/outliers/elliptic.py:32-43`) | small | ⬜ open |
| OC-174 | 🟡 | Polars `DateFeatures` crashes on an entirely invalid string date column despite `strict=False`; pandas produces nullable calendar features (`preprocessing/time_series/date_features.py:102`) | small | ⬜ open |
| OC-175 | 🟡 | Polars `RollingAggregate` propagates float NaN through windows instead of ignoring missing observations like pandas — `[1,NaN,3]` with window 2 yields mean `[1,NaN,NaN]` vs `[1,1,3]` (`preprocessing/time_series/rolling.py:48`) | small | ⬜ open |
| OC-176 | 🟡 | Polars `LagFeatures(drop_na=True)` removes nulls but retains float NaN in source/lag columns; equivalent pandas input drops those rows (`preprocessing/time_series/lag.py:54-59`) — independent of OC-165's y desynchronization | small | ⬜ open |
| OC-59 | 🟠 | `DatasetProfile` numeric-column coverage completely different between engines (`preprocessing/inspection/`) | small | ⬜ open |
| OC-60 | 🟠 | `GeneralBinning`'s `missing_strategy: "label"` silent no-op on polars (`preprocessing/bucketing.py`) | small | ⬜ open |
| OC-165 | 🟡 | Pandas `LagFeatures(drop_na=True)` removes X rows but leaves tuple y untouched — 3 rows become 2 features / 3 targets even with a unique index (`preprocessing/time_series/lag.py:85-87`) | small | ⬜ open |
| OC-166 | 🟡 | Polars `IQR`, `ZScore`, and `ManualBounds` filter X but leave NumPy y untouched — 5 rows become 4 features / 5 targets; Polars Series y works (`preprocessing/outliers/_common.py:9-15`) | small | ⬜ open |

### Remaining — modeling / tuning

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-67 | 🟡 | Tuning metrics `pr_auc`/`pr_auc_weighted`/`g_score` crash the entire search (`modeling/_tuning/metrics.py:19-36,127-146`) | small | ⬜ open |
| OC-168 | 🟡 | `SkyulfPipeline.fit()` retains the previous model's tuned thresholds — refitting with new class labels makes thresholded prediction crash; unchanged labels reuse stale cutoffs (`pipeline/_pipeline.py:135,380-389`) | small | ⬜ open |

### Remaining — frontend

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-54 | 🟡 | `DebugNode` is dead code that would silently no-op if wired up (`nodes/DebugNode.tsx`) | small | ⬜ open |
| OC-55 | 🟡 | `tsc --noEmit` fails: `mermaid` declared but not installed (`frontend/ml-canvas/package.json`) | 1 line | ✅ verified stale 2026-09-06 — `mermaid@11.17.2` is in `dependencies`, in the lockfile, installed and lazy-imported into its own chunk; the exact CI `tsc --noEmit` exits 0, `npm run build` succeeds, and the 5 real-parser tests pass. No change needed |
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

## New findings;
### 2026-09-06 — remaining-source continuation (findings added as verified)

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-187 | 🟡 | LightGBM's advertised `subsample` control and default search dimension have no effect: both calculators retain native `subsample_freq=0`, disabling row bagging (`modeling/hyperparameters/_tree.py:576`, `_registry.py:298,309`; `classification.py:754`, `regression.py:547`) | small | ⬜ open |
| OC-188 | 🟠 | Rule discovery decodes sklearn class positions against Polars' shared category dictionary, publishing labels absent from the target while reporting perfect accuracy (`profiling/_analyzer/rules.py:169-170,196-198,295-298`) | small | ⬜ open |
| OC-189 | 🟡 | Classification rule text reports `Samples: 1` for leaves containing multiple rows: it sums sklearn's normalized class proportions instead of using the leaf sample count (`profiling/_analyzer/rules.py:299-301`) | small | ⬜ open |
| OC-190 | 🟡 | A categorical column named `count` crashes profiling and categorical drift because `value_counts()` generates the same column name (`profiling/analyzer.py:286-290`, `profiling/drift.py:380-381`) | small | ⬜ open |
| OC-191 | 🟡 | All-null and Polars Enum columns are classified as text and sent to string-only aggregates, aborting the whole profile (`profiling/analyzer.py`, `_analyzer/column.py`) | small | ⬜ open |
| OC-192 | 🟡 | Decomposition's categorical null bucket displays as `Unknown`, but drilling into it filters for the literal string and silently loses the bucket's rows (`profiling/_analyzer/decomposition.py:71-76`) | small | ⬜ open |
| OC-193 | 🟡 | A single missing timestamp removes time-series analysis at the 1,000-row resampling boundary: dynamic grouping receives null date keys and the exception is swallowed (`profiling/_analyzer/temporal.py:232,243`) | small | ⬜ open |

**OC-188/189 — wrong rule labels and support counts.** Keep
`held = pl.Series(['unrelated_1','unrelated_2']).cast(pl.Categorical)` alive,
then discover classification rules for `x=[0,0,0,1,1,1]` and
`target=['no','no','no','yes','yes','yes']`. Executed
`EDAAnalyzer(df)._discover_rules(['x'], 'target', 'classification')` returns
accuracy **1.0** but predicts **unrelated_1 / unrelated_2** in its nodes and
rule text. Both leaf nodes have `samples=3`, while the text says **Samples: 1**.
The first defect confuses encoded class values with sklearn class-array
positions; the second independently treats normalized proportions as counts.
**Fix/verification targets:** decode through fitted `clf.classes_` and the
matching category mapping; use the actual leaf count for textual support.
Cover non-contiguous category codes and leaves containing multiple samples.

**OC-190 — reserved count column.** Executed
`EDAAnalyzer(pl.DataFrame({'count': ['a']*99 + ['b']})).analyze()` raises
`DuplicateError: using value_counts on a column/series named 'count' would
lead to duplicate column names`. The categorical drift path reproduces the
same failure. **Fix/verification target:** choose collision-safe internal
count names and cover profile and drift entry points with user columns named
`count`. This is distinct from OC-161's clustering feature overwrite.

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

**OC-193 — nullable dates break large-frame time analysis.** Executed the
same daily-date/value construction with the final date missing: **999 rows**
yield **998** trend points, while **1,000 rows** yield `timeseries=None` and
log `null values in dynamic group_by not supported`. The larger-frame branch
resamples without removing null date keys. **Fix/verification target:** apply
an explicit missing-timestamp policy before resampling and test both sides of
the row-count boundary. OC-114 instead concerns all-null numeric ACF results.

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

### 2026-09-05 — OC-163–168 filed: supplemental core review, six additional reproduced bugs

All six were reproduced through executed Python probes against the working tree and checked against the existing tracker and relevant source-audit reports. IDs follow the review's reported order. Two high-severity findings enter **Next**; the four medium-severity findings enter their domain queues. All remain **open**. This filing changes only the tracker; no implementation fixes or regression tests were added.

**OC-163 — time-series sort loses X/y alignment (🟠).** With tuple input `X = {time: [3,1,2], value: [30,10,20]}` and `y = [300,100,200]`, fit/apply `LagFeatures` with `columns=["value"], lags=[1], sort_by="time"`, or `RollingAggregate` with `columns=["value"], window=2, sort_by="time"`. Both pandas and Polars return times `[1,2,3]` but targets `[300,100,200]`; the correct targets are `[100,200,300]`. The engine branches sort only X and return the original y. Downstream conversion to NumPy consumes these mismatched rows positionally, silently corrupting supervised training. Locations: `skyulf-core/skyulf/preprocessing/time_series/lag.py:45,81` and `rolling.py:63,119`. **Fix/verification target:** apply the same positional permutation to X and y; cover both nodes, both engines, and sorting combined with lag row removal. This is separate from OC-162's reserved-column collision in cross-validation and OC-165's filtering-only failure.

**OC-164 — split extraction invalidates an existing trained model (🟠).** Train a `SkyulfPipeline` containing `StandardScaler(columns=["x"])` and `linear_regression` on a `SplitDataset`: data has `x = arange(20)`, `target = 10*x`, first 15 rows train and last 5 test. `predict(x=5)` returns **50.0**. Call `get_fitted_split()` on the same split with x shifted by +100, then predict the original `x=5` again: **−949.9999999999998**, with no error. `pipeline/_pipeline.py:234` calls the live `feature_engineer.fit_transform(data)`, replacing its fitted scaler while retaining the model trained against the previous scaler. Refitting preprocessing is documented for this helper; the defect is leaving an already-fitted model usable with incompatible preprocessing. **Fix/verification target:** isolate split extraction from the trained pipeline's state, or explicitly invalidate the retained model when refitting preprocessing. Pin unchanged predictions for a non-mutating implementation, or a clear unfitted-state error if invalidation is chosen.

**OC-165 — pandas lag filtering leaves y unfiltered (🟡).** Fit/apply `LagFeatures` to pandas `X = {value: [10,20,30]}`, `y = [100,200,300]` with `columns=["value"], lags=[1], drop_na=True` and no sorting. Output X has **2 rows**, while y still has **3**; expected y is `[200,300]`. `lag.py:85-87` drops missing rows from the feature frame and returns the original target. The equivalent Polars Series probe correctly returns 2/2 rows, providing an engine control. This is independent of OC-163 and distinct from OC-12, which concerned duplicate-index expansion in `DropMissingRows` / `Deduplicate`. **Fix/verification target:** filter y with the same positional keep-mask as X, including duplicate-index coverage; sorting and filtering must compose correctly.

**OC-166 — Polars outlier helpers skip NumPy targets (🟡).** With Polars `X = {x: [1.,2.,3.,4.,100.]}` and NumPy `y = [10,20,30,40,1000]`, fit/apply `IQR(columns=["x"])`, `ZScore(columns=["x"], threshold=1)`, or `ManualBounds(bounds={"x": {"lower": 0, "upper": 10}})`. Each removes x=100 but returns all five targets: **4 X rows / 5 y rows**. Repeating each probe with Polars Series y returns the correct four targets. The dispatcher accepts engine-neutral NumPy targets, but `_filter_y_polars` in `preprocessing/outliers/_common.py:9-15` filters only Polars Series/DataFrames and silently returns other types. **Fix/verification target:** preserve positional alignment for supported array-like targets, with NumPy and native-Polars controls across the affected nodes. No reserved helper-column name is involved, so this does not duplicate OC-160.

**OC-167 — ambiguous canonical serialization creates fingerprint collisions (🟡).** `artifact_digest(np.array(["a", "bstr:c"], dtype=object))` equals the digest of `np.array(["astr:b", "c"], dtype=object)`: strings contribute `b"str:" + value` without a length prefix, and object-array elements have no boundary markers. Ordinary lists also collide: `["a", "b,str:c"]` versus `["a,str:b", "c"]`. Confirmed through the public pipeline API: two otherwise identical `LabelEncoder(columns=["x"])` pipelines fitted on the first pair of category lists return **identical `fingerprint()` values**, but transform input `"a"` to **0 versus −1**. Locations: `pipeline/seal.py:52,64` (and the list serialization branch). This is deterministic aliasing of distinct values, not OC-62's process-dependent pointer hashing, and not OC-63's cycle handling. **Fix/verification target:** make the canonical byte encoding unambiguous for strings/bytes and nested containers; regress both direct digest collisions and differing fitted pipeline behavior, while preserving process stability.

**OC-168 — refitting leaves old decision thresholds active (🟡).** Fit a logistic-regression pipeline on `x = arange(40)`, `target = (x >= 20).astype(int)`, using the frame as both train and test for this lifecycle probe. Tune on the same features/labels with `accuracy_score`, obtaining `{0: 0.5, 1: 0.5}`. Refit the same pipeline instance with labels mapped to `{0: "no", 1: "yes"}`. Normal prediction at x=25 returns `"yes"`, but `predict(..., use_tuned_thresholds=True)` raises `ValueError: thresholds is missing entries for classes: ['no', 'yes']`. `_tuned_thresholds` is initialized in `__init__` and assigned by optimization, but never reset by `fit()` (`pipeline/_pipeline.py:135,380-389`). With unchanged labels the same stale thresholds remain accepted, even though they belong to a previous model. **Fix/verification target:** invalidate tuned thresholds when retraining begins and require fresh tuning for the replacement model; cover both changed and unchanged label sets. This is lifecycle state retention, separate from OC-36's degenerate validation search and OC-147's tie comparison.

**Verification during the review:** full command `.venv/Scripts/python.exe -m pytest skyulf-core/tests -q --no-cov --tb=short -o addopts=''` produced **3680 passed, 56 skipped, 1 failed, 2 errors** in 130.11 seconds. The three unsuccessful tests were environmental: two serializer fixtures could not access pytest's default temporary directory, and the wrapped-Polars sentence-embedder test hit restricted network access while checking the model cache. All three passed on a targeted rerun with a writable temporary directory and `HF_HUB_OFFLINE=1` (cached model available): **3 passed**. These suite results are separate from the six successful bug reproductions; no fixes are implied by the rerun. Temporary verification files were removed after use.

### 2026-09-05 — OC-160/161/162 filed: internal Polars helper-column names collide with valid user columns
These are outside the Opus inventory. **OC-160:** the row-dropping implementation creates a physical `__idx__` column to retain X/y positional alignment. Polars rejects the operation if X (in `DropMissingRows`) or DataFrame-shaped y already has that perfectly valid name, so data-cleaning fails instead of returning the filtered frame. The failure was executed and reproduced as `polars.exceptions.DuplicateError`. It also affects the y-aware `Deduplicate` path, which creates the same temporary column.

**OC-161:** native Polars clustering evaluation appends labels as `__skyulf_cluster__`, then removes that column from each cluster subset. If a numeric input feature already uses that name, the append overwrites it and the removal deletes it. The centroid helper still iterates the original feature-name list, so selecting the missing feature raises `polars.exceptions.ColumnNotFoundError`. This was executed through the public `evaluate_clustering_model` entry point.

**OC-162:** the Polars time-sort helper uses `__cv_y__` whenever y is a list/array or an unnamed Series. `with_columns` replaces an existing feature of that name; the following `drop([y_name, sort_col])` removes the replacement, permanently excluding the real feature from cross-validation. The source path is deterministic and the current test suite covers only a named target (`target`), not this collision. Add regression coverage for all three names and ensure internal columns use collision-free names or avoid materialising them as user-visible columns.

### 2026-09-05 — OC-170–176 filed: source review plus bounded 10-file follow-up

No implementation changes. All seven findings below were reproduced against the
local source, including working controls where applicable. The final batch read
every line of five remaining `outliers/` files and all five `time_series/` files;
the coverage ledger lists the exact files and the remaining review scope.
Existing targeted suites passed **141 tests** (one pytest-cache permission warning),
so the additional probes expose gaps not covered by those passing suites.

**OC-170 — registered stateless nodes rejected by the leakage validator (🟡).**
Call `validate_leakage_safety({"preprocessing": [{"transformer": name, "params": {}},
{"transformer": "TrainTestSplitter", "params": {}}]})` for each of `TextCleaning`,
`DateFeatures`, `Casting`, and `feature_target_split`. All four raise `ValueError`
and claim the node is not known, although their registry metadata declares
`learns_from_data=False`. The validator constructs a set of learners, then treats
every node outside that set as unregistered unless one of four special-case
predicates accepts it. **Fix/verification target:** distinguish registered
stateless nodes from genuinely unknown nodes; keep learned-before-split and
unknown-node rejection tests. This is the core linear-config validator, not
OC-70's backend branch-protection issue. Location: `skyulf/leakage.py:140-156`.

**OC-171 — explicit mean/median imputation silently skipped (🟡).** Fit/apply
`SimpleImputer` with `columns=["x"]` and either `strategy="mean"` or `"median"`
on pandas `x=[1.0,None,1.0]` or `x=[0.0,None,1.0]`. The fitted artifact is `{}`
and the missing cell survives. Equivalent Polars inputs fill it with **1.0**
and **0.5**, respectively. The pandas safety filter calls
`detect_numeric_columns()` with its default constant/binary exclusions, discarding
the user's explicit selection. **Fix/verification target:** validate numeric
dtype without applying auto-selection exclusions to explicitly chosen columns;
test both strategies and both engines. Unlike OC-16/17, the columns contain
valid observations. Location: `preprocessing/imputation/simple.py:173-177`.

**OC-172 — nullable pandas scaling fails at the NumPy boundary (🟡).** Construct
`X=pd.DataFrame({"x": pd.Series([1,None,3], dtype="Int64"), "z":
pd.Series([2,None,4], dtype="Float64")})`. Calling
`StandardScalerCalculator().fit(X, {"columns":["x","z"]})` raises
`TypeError: float() argument must be a string or a real number, not 'NAType'`.
Native sklearn `StandardScaler().fit(X)` succeeds with means `[2,3]`, as does
the Skyulf calculator on `pl.from_pandas(X)`. The scaler's subset enters the
generic bridge without the nullable-to-float/NaN normalization already present
in `resolve_columns_then_to_numpy`. **Fix/verification target:** cover mixed
nullable numeric columns with missing cells; preserve missing values as `np.nan`
and verify other scaler callers of the same bridge. This reproduction requires
the mixed-column case; a single nullable integer column was not found broken.
Locations: `preprocessing/scaling/standard.py:144,154`,
`engines/sklearn_bridge.py:52`.

**OC-173 — duplicate pandas indexes disable EllipticEnvelope filtering (🟡).**
Fit `EllipticEnvelope` on `x=[-2,-1,-0.5,0,0.5,1,2,100]` with
`columns=["x"], contamination=0.125`. Apply to `x=[0,100,None,1]` with indexes
`[0,0,1,1]`: all four rows survive and a warning says prediction received NaN.
The same values with a unique index return `[0,NaN,1]`, correctly removing 100.
`series.dropna().index` followed by `series.loc[valid_idx]` expands duplicate
labels and reintroduces the missing row; the broad exception handler skips
that column's filtering. **Fix/verification target:** select and scatter by row
position, testing duplicate labels with and without missing values and X/y
alignment. This is not OC-12's already-fixed DropMissingRows/Deduplicate target
selection. Location: `preprocessing/outliers/elliptic.py:32-43`.

**OC-174 — wholly invalid date strings crash the Polars date node (🟡).**
Fit/apply `DateFeatures` with `columns=["d"], features=["year"]` to a Polars
String column `d=["bad","invalid"]`: `ComputeError: could not find an appropriate
format to parse dates, please define a format`. Pandas returns two nullable
missing years. Controls with `["2024-03-01","bad"]` and an all-null String
column succeed in both engines. `str.to_datetime(strict=False)` tolerates
individual parse failures but still requires an inferable format.
**Fix/verification target:** make the all-unparseable case follow the documented
invalid-date-to-null behavior; retain mixed-valid/invalid and all-null tests.
Location: `preprocessing/time_series/date_features.py:102`.

**OC-175 — rolling float NaN semantics diverge across engines (🟡).** Fit/apply
`RollingAggregate` to numeric `x=[1.0,float("nan"),3.0]` using `columns=["x"],
window=2, min_periods=1, aggregations=["mean"]`. Pandas emits `[1,1,3]`; a native
Polars Float64 column emits `[1,NaN,NaN]`. Sum/min/max/median show the same
divergence in the probe. The Polars expression passes NaN directly into rolling
operators; pandas treats it as a missing observation. **Fix/verification target:**
normalize numeric missing-value semantics before aggregation and cover actual
float NaN, not just Polars null, across aggregations and grouped windows.
This concerns generated feature values, not OC-163's sorting/target alignment.
Location: `preprocessing/time_series/rolling.py:48`.

**OC-176 — lag drop-na leaves float NaN rows on Polars (🟡).** Fit/apply
`LagFeatures` to numeric `x=[1.0,float("nan"),3.0]` with `columns=["x"], lags=[1],
drop_na=True`, with no target and no sorting. Pandas returns zero rows because
every row has a missing source or lag; Polars retains two rows, each with NaN
in one of those columns. Its filtering uses only `is_null()`/`drop_nulls()`.
**Fix/verification target:** treat NaN and null consistently for floating columns
without calling numeric-only checks on other dtypes; cover frame-only and tuple
input and apply any keep-mask identically to y. Unlike OC-165, this reproduces
without a target. Location: `preprocessing/time_series/lag.py:54-59`.

## Log

### 2026-09-06 — OC-159/131/28/77 fixed, OC-55 verified stale: three fail-open paths and a coverage floor 51 points below reality

**OC-159 — an empty filter is a table-wide write.** Both functions build their
WHERE clause one key at a time from `filter_dict`, so with no keys the statement
compiles to a bare `DELETE FROM data_sources` (or a WHERE-less `UPDATE`) that
empties or rewrites the table and still reports `affected_rows` as though that had
been requested. The tracker called it a dead call path and that is correct — all 8
exports of `backend/database/data_sources/` have zero external references, and the
app's real delete goes through the ORM (`data_ingestion/service.py:107-143`).
Fixed anyway, because `_normalize_filter(None) → {}`
(`async_data_sources_crud.py:161`) means the *signature itself* accepts the
table-wiping input. Guards raise `ValueError` **before** the session opens, on all
four sites (sqlite + postgres × delete + update). Note the crud layer wraps every
exception as `RuntimeError(...) from None`, so callers there see "Failed to delete
from primary database" — masking that predates this fix, pinned as-is.

**OC-131 — the fail-open was never the problem; the silence was.** `drift.py` was
the **only** module under `skyulf/profiling/` with no logger (13 siblings each have
one), so a column whose histogram blew up reported as perfectly stable with no
trace. Rejected alternatives: `None` needs `DriftMetric.value` optional, a contract
change across core + backend mirror + the frontend sliders that re-derive
`value > threshold`; `inf` fails closed but Starlette's `JSONResponse` uses
`allow_nan=False` and raises; NaN compares False, so it reads as no drift *and*
breaks the response. Kept the finite `0.0`, documented why, and made all three
fail-open paths observable — including the worst one the finding missed:
`_calculate_numeric_column_drift` returns `None` on an uncastable column, so the
**whole column disappears** and its absence reads as "no drift". The NaN poisoning
I expected was already closed upstream by F-13's `drop_nans()`.

**OC-28 — filed against the wrong line.** The `except` blocks have called
`logger.exception` since the node was created (`2d605197`). The unlogged path is
the `valid_cols` filter: a frame missing columns the transformer was fitted on
passes them through raw while the node reports success — train/serve skew, not a
no-op. Both engines now share `_fitted_columns_present`, which names the missing
columns, so they cannot drift apart. Did **not** switch to fail-closed:
`general.py` fails open the same way three times, there are 25 `except Exception`
sites under `preprocessing/`, and `apply` runs at inference where a raise turns a
degraded prediction into a 500 — that decision is now on the `apply` docstring.
Also fixed an engine asymmetry found on the way: pandas copied the frame *before*
the try, so a failure returned a copy already cast to `float64` while polars
returned the untouched original. The transform is now computed first and a failure
hands back the caller's own object on both engines, pinned with `out is df`.

**OC-77 — the floor was 51 points below reality.** CI run 34032836551 reports
`TOTAL 13982 346 4012 265 96%` (97% locally), so `--cov-fail-under=45` could never
fail. Raised to 90: headroom for optional-extras variance, since CI installs fewer
than a dev venv (`sentence_transformers`, `causallearn`, `torch`, `transformers`
are local-only — which is why CI measures a point lower). `--maxfail=1` removed, as
the suite finishes in ~2 minutes and bailing early only cost the real failure
count. **The same flag was in `backend-tests.yml:101` and is removed there too**,
though OC-77 filed only core. No backend floor: it measures 77% on both legs (run
34032836559, 2367 vs 2364 missed — so the existing "both legs produce the same
coverage" comment is right) and needs its own number chosen against 77.

**OC-55 — stale, no change.** `mermaid@11.17.2` is in `dependencies`, in the
lockfile, installed, and lazy-imported into its own chunk. The exact CI command
`npx tsc --project tsconfig.json --noEmit` exits 0, `npm run build` succeeds and
emits `mermaid.core-*.js`, and the 5 real-parser tests pass.

**Gates.** Core **3672 passed / 70 skipped** (3670 + 2 new), backend **1637 passed
/ 7 snapshots** (1630 + the OC-159 file's 7 cases). `ruff check`, `ruff format
--check`, `ty check backend skyulf-core/skyulf` clean. Both workflows parse, and
the new flags were proven by running them: `--collect-only` reports `FAIL Required
test coverage of 90% not reached. Total coverage: 26.32%` — the floor firing, not a
flag error. The tracked frontend build output (71 files under `static/ml_canvas/`)
came back byte-identical.

**Tracker hygiene, per the owner's instruction.** Status cells are now one sentence
with detail here, and the nine longest existing cells were trimmed the same way
(OC-09, OC-81, OC-22, OC-05, OC-141, OC-112, OC-132, OC-78, OC-79) — 7,115
characters out of the tables. Each was checked against its Log entry before
trimming; the two whose detail lived only in the cell were relocated, not lost —
OC-112's reasoning is verbatim in the comment the fix rewrote
(`profiling/_analyzer/categorical.py:22-30`), and OC-141's deferred half stays in
its cell as a live pointer to R1 step 1. OC-09's 3,064-character *Item* cell had
become a running fix log and is back to the finding as filed. OC-01/OC-02 were left
alone: already one-sentence status cells, and OC-02 has no Log entry, so trimming
its re-verification note would have destroyed the only record of it.

### 2026-09-06 — OC-81 closed: the license decision arrived, and "1 line" turned out to be five files

The previous entry left OC-81 **blocked** rather than open, on the grounds that
"add a `License ::` classifier" presumes the license is known and four files in
the repo disagreed about which one applies to `skyulf-core`. The owner's decision:
**`skyulf-core` is Apache-2.0; the backend and frontend stay AGPLv3.** That makes
`COMMERCIAL-LICENSE.md` the authoritative file of the four — it was the only one
already stating the split correctly.

**The fix is not in `setup.py`.** `skyulf-core/pyproject.toml` has a `[project]`
table that defers every other field to `setup.py` by listing it in `dynamic`, so
the obvious move is to add `"license"` there and put the value beside the version
literal. That **silently produces the wrong metadata**: setuptools accepts it
with no warning at all, but a dynamic license is emitted as the deprecated
free-text `License: Apache-2.0` field, whereas PEP 639's `License-Expression:` —
the only one PyPI indexes for license filtering — is produced *only* from a
static `[project] license`. Passing it to `setup()` instead is worse, and at
least loud: `SetuptoolsWarning: license overwritten by pyproject.toml`. Caught by
generating the metadata rather than by reading the config. The classifier form
the finding asked for was **not** used: setuptools >= 77 deprecates `License ::`
trove classifiers once an SPDX expression is present.

Verified by building a real wheel through `setuptools.build_meta.build_wheel`,
not by inspecting the config: `Metadata-Version: 2.4`,
`License-Expression: Apache-2.0`, `License-File: LICENSE` with the Apache text
bundled at `dist-info/licenses/LICENSE`, `Requires-Dist: joblib>=1.3.0` (OC-79),
`skyulf/py.typed` present in the zip (OC-78, now confirmed in the artifact PyPI
actually receives rather than only in a `build_py` scratch dir), and **zero**
setuptools warnings.

**Three files were then reconciled to the decision**, which is where "1 line"
went:

- `COPYRIGHT.md` had the split **backwards** — "Backend & Core: Apache License,
  Version 2.0" — pointed that Apache claim at the root `LICENSE`, which is AGPLv3
  text, and cited `frontend/feature-canvas/LICENSE`: a path that does not exist,
  since the frontend is `frontend/ml-canvas`, which has no license file of its
  own and inherits the root AGPLv3. Two of those three errors would have
  survived a fix that only added the classifier.
- `skyulf-core/README.md:410-413` declared AGPLv3+ while linking the Apache-2.0
  `LICENSE` sitting beside it. This one had the largest blast radius: that README
  *is* the PyPI `long_description`, so the wrong license was on the package's
  front page, on the one distribution whose entire selling point is that it can be
  vendored into proprietary work.
- Its shields.io badge read AGPLv3 for a subtler reason — the `github/license`
  endpoint resolves the **repository root** `LICENSE`, not the nearest one. Any
  subdirectory package in a split-license monorepo inherits the root's badge
  unless the badge is pinned. Replaced with a static Apache-2.0 badge.

**Version drift found on the way.** `skyulf-core/setup.py:19` was already at
`0.8.15` while the root `pyproject.toml` — which its own docstring names as the
thing to stay in step with — was at `0.8.14`, and the `v0.8.15` changelog section
existed as an empty placeholder. Synced root to `0.8.15` and ran
`npm run sync-version` for the frontend (diff is version-only, 3 lines across
`package.json` + `package-lock.json`). No rebuild is implied: nothing under
`frontend/ml-canvas/src` reads the package version and `dist/` is untracked.

**Gate.** `ruff check .` clean. Two traps worth keeping: a Git Bash `mktemp -d`
path (`/tmp/...`) is not resolvable by Windows Python, so a build probe pointed
at one fails with `error in 'egg_base' option: does not exist` — use
`tempfile.mkdtemp()` from inside Python instead; and `cd` persists between Bash
calls, so a `cd skyulf-core` in one command makes the next command's relative
pathspecs silently match nothing.

### 2026-09-06 — OC-05/22/78/79/112/132/141 fixed: the "1 line" tier closed, and two of the seven were not what they claimed

Took the whole `1 line` / `mechanical` effort tier that was still open. **Seven
closed**, one **blocked on a decision**, two investigated and **re-scoped but left
open**. Every claim below was verified by execution or by reading the call sites,
not inferred from the audit text.

**Two of the seven were mis-filed, and both were worse than the ⚪/🟡 they carried:**

- **OC-05 was not cosmetic.** The pandas apply path wrote a float transform
  result into `int64` columns via `.loc`, which pandas 2.3.2 warns about *per
  column* and has said will become an error. The escalation is the bare
  `except Exception` around it that only logs: once pandas raises, the transform
  is **silently skipped** and the original data returned — OC-28's failure mode
  arriving through OC-05. Fixed by casting the destination columns to `float64`
  first. Verified 0 warnings (was 2), values bit-identical, column order and
  index preserved, pandas↔polars parity intact.
  The regression test **records** warnings instead of promoting them to errors,
  because a promoted warning would be swallowed by that same `except` and the
  test would pass while the transform no-ops — so it also asserts the values
  really changed.
- **OC-22 had a test pinning the bug.** `test_infer_output_schema.py:207`
  asserted passthrough for `("binary", "regression")`. sklearn 1.8.0's
  `TargetEncoder` accepts `{auto, binary, multiclass, continuous}` and raises
  `InvalidParameterError` for `"regression"`, and `_build_target_encoder`
  forwards the config value verbatim — so that assertion locked in a prediction
  for a config that can never fit, while the legitimate `"continuous"` case fell
  through to `None`. Changed to `"continuous"`, and confirmed the prediction is
  *correct* rather than merely reachable: fitting+applying on a continuous `y`
  does encode in place. The existing test was the reason the defect survived.

**The five straightforward ones.** OC-78 created `skyulf-core/skyulf/py.typed`
(declared in both `setup.py:31` and `MANIFEST.in:3`, absent on disk) and was
proven by running setuptools' own `build_py` into a temp dir and finding the file
in the build lib — `build`/`wheel` are not installed, and installing them was not
mine to do. OC-79 declared `joblib>=1.3.0`; the filed finding was
`install_requires`, but the backend has the identical undeclared-direct-import
gap (`ml_pipeline/artifacts/local.py:8`, `s3.py:8`), so it went into root
`pyproject.toml` and `requirements.txt` as well. OC-112 was comment-only. OC-132
deleted a branch reading `dropped_features`, a key nothing in the repo writes,
and recorded the superseding runtime path in a docstring so it is not re-added.
OC-141 deleted the unconsumed `invalid_values` from `node_meta`, after
re-verifying zero consumers and that `user_picked_no_columns` keys off `columns`,
so the smoke test's short-circuit is unchanged.

**OC-81 is blocked, not open.** "Add a `License ::` classifier" assumes the
license is known. It is not — four files disagree about which license applies to
`skyulf-core`, and the classifier would publish that choice to PyPI:
`skyulf-core/LICENSE` is the full **Apache-2.0** text; `COMMERCIAL-LICENSE.md`
says *AGPLv3 backend+frontend / Apache-2.0 core*; `COPYRIGHT.md` says the
**opposite** split (*"Backend & Core: Apache 2.0"*, *"Frontend: AGPLv3"*) while
pointing at a root `LICENSE` that is **AGPLv3**; and `skyulf-core/README.md:410-413`
declares **AGPLv3+** while linking the Apache-2.0 file beside it. Needs a
decision (and then the contradicting docs need fixing regardless of which way it
goes). Marked ⛔ in the queue.

**Two left open on purpose, both re-scoped from what the tracker said:**

- **OC-10** is **five** overrides, not four, and `BaseCalculator.infer_output_schema`
  already ends in `return None` (`preprocessing/base.py:129`), so all five are
  behaviourally dead. Deferred because each carries the per-node *reason* the
  schema is unknowable, and OC-03's sweep will touch these same five files —
  deleting the rationale now would just cost OC-03 the documentation it needs.
- **OC-144** is not a one-liner: 4 code sites, **10** assertions in
  `test_geo_nodes.py` (one of which reads `geo_distance_km` while converting to
  miles — the mislabel baked into a test), and a docs line. The detail that
  decides the fix is that both apply-path fallbacks are unreachable, since `fit`
  always writes `output_column` into the artifact, so the *declared* `node_meta`
  default picks the name — and a static dict cannot be unit-dependent.

**Gates.** `ruff check .` clean; `ruff format --check` clean on all 11 touched
files; `ty check` clean. Core **3670 passed / 70 skipped** — the 3668 baseline
plus the two new tests. Backend **1630 passed / 7 snapshots passed** — exact
baseline. Two verification traps worth recording: `pytest --timeout=900` is not
supported here (no `pytest-timeout`), and piping a gate through `tail` reports
*`tail`'s* exit code, so a backgrounded `pytest … | tail` showed exit 0 for a run
that had actually aborted on argument parsing. Read the output, not the status.

Also noted while grepping: repo-wide `grep -r` matches `site/search/search_index.json`
(built docs, ~140 KB single line) and floods the context; exclude `site/` and
`.venv/`, and prefer the ripgrep-backed Grep tool with `--glob`.

### 2026-09-06 — OC-183–186 filed: four backend defects the OC-09 docstring pass surfaced, deferred by decision

All four were found while writing docstrings against the code, and all four were
*verified by grep and by reading the call sites* rather than inferred — but none
was fixed, because each changes behaviour rather than documentation. Filed for
the next session by decision; the OC-09 log entry carries the same list in prose.

**OC-183 — `SmartCatalog` S3 auto-init reads a variable no `Settings` field answers to (🟠).**
`backend/data/catalog.py:556` does `bucket = os.getenv("S3_BUCKET_NAME")` and builds
an `S3Catalog` only if it is truthy. Two independent problems. (1) It is the exact
failure mode OC-130 was filed for: pydantic-settings loads `.env` into the model and
never exports it into `os.environ`, so a bare `os.getenv` cannot see the documented
configuration file — a bucket set only in `.env` leaves `s3_catalog` as `None`, and
`SmartCatalog` silently routes everything to `FileSystemCatalog` instead. No error,
no warning. (2) The *name* is wrong too: `config/mixins/aws.py:12` declares
`AWS_BUCKET_NAME`, `docs/guides/backend_configuration.md:146` documents
`AWS_BUCKET_NAME`, and `README.md:105` documents `S3_BUCKET_NAME`. The two docs
contradict each other and the code reads the one that is not a field, so even a real
process environment variable only works by accident of naming. The fix is one line
plus a README correction, but it needs a canonical-name decision first — silently
switching to `AWS_BUCKET_NAME` would break anyone who set the README's name in a
real environment.

**OC-184 — `ProductionSettings.SECURITY_HEADERS` is configured and never emitted (🟠).**
`config/environments.py:39` defines `_PROD_SECURITY_HEADERS` and `:84` assigns it to
`ProductionSettings.SECURITY_HEADERS`. A repo-wide grep finds those two lines and the
docstring that describes them — **no consumer**. `main.py::_add_middleware` installs
TrustedHost, Logging, ErrorHandler and CORS and never reads the setting, so production
boots logging "Running in PRODUCTION mode with enhanced security" while sending none
of HSTS, `X-Frame-Options: DENY` or the CSP. This is a behaviour change to fix, not a
lint or config fix: a headers middleware has to be added, and its position matters
because `add_middleware` *wraps* — CORS must remain outermost or error responses lose
`Access-Control-Allow-Origin`. Two prior initiative docs noticed the declaration
(`initiatives/backend-and-core-review/README.md:57`,
`initiatives/enterprise-readiness/2026-08-11-data-governance-audit.md:183`) but it was
never filed here as a finding.

**OC-185 — authorization is stubbed three different ways at once (🟡).**
`database/models.py:157` `has_permission(self, permission) -> bool` is
`return True  # Placeholder` and has **zero callers** anywhere in the repo.
`data_ingestion/dependencies.py:26,31` define `require_data_access` and
`require_data_admin` as async no-ops, and no route declares them as a dependency.
`data_ingestion/router.py:148,169` hardcode `user_id = 1` under an explicit
`# KNOWN-GAP: Auth not implemented yet`, and the module docstring says the same, so
every data source is owned by one user and visible to every other. Filed at 🟡 rather
than 🟠 because the hardcoding is self-documented and `has_permission` is unreachable
today; the finding is that a permission API, two route guards and the routers disagree
with each other, so whichever one someone wires up first will look authoritative and
not be. Effort is recorded as `decision + ~1 week` because the real work is choosing
an authz model, not editing `return True`.

**OC-186 — `S3Catalog.exists` authenticates differently from `load` and `save` (🟠).**
`_prepare_s3fs_options` (`catalog.py:283`) is the single place that translates
AWS-named options into s3fs names: `aws_access_key_id`→`key`,
`aws_secret_access_key`→`secret`, region→`client_kwargs['region_name']`, plus the
endpoint handling. `__init__`:275, `load`:452 and `save`:491 all call it.
`exists`:521 does not — it builds `s3fs.S3FileSystem(**self.storage_options)` from the
raw instance dict. So for a catalog constructed with AWS-style credentials, `exists()`
runs with unmapped options while `load()` runs with mapped ones, and the two can
disagree about the same key: `exists` returns `False` (or raises on the unexpected
kwargs) for an object `load` reads successfully. Any caller that gates on `exists`
before `load` — the natural pattern — therefore takes the wrong branch. One-line fix:
pass `self._prepare_s3fs_options(self.storage_options)`.

**Related, recorded but not filed separately.** The same `exists`/`load` asymmetry
appears on the filesystem side and is already documented in the code by the OC-09 pass:
`FileSystemCatalog.exists`:228 goes through `_get_path` and so skips the
`.parquet`/`.csv` extension probe `_resolve_dataset_path` uses, reporting `False` for a
legacy id that `load` would find; and `FileSystemCatalog.save`:211-213 calls the pandas
writers (`to_csv`/`to_parquet(index=False)`) although `load` returns polars, so a polars
frame must be converted by the caller even when `SKYULF_ENGINE=polars`. Lower severity
than OC-186 because neither mis-authenticates; both are honest-contract limitations
rather than silent divergence.

### 2026-09-06 — OC-09 closed: 904 hand-written docstrings, and a ruff privacy gate that hid 82 of them from every check we run

**The audit's estimate was wrong by an order of magnitude, so the scope was decided by measurement.** OC-09 was filed as "half day, ~2h remain" for "~500 missing docstrings + 84 unused args". Measured instead: **3,350** `D` sites and **531** `ARG` sites repo-wide. Only 81 had an auto-fix available and every `D` fix ruff offers is marked *unsafe*, so `--fix` was never an option for the rest — ~4,000 human judgements, not a two-hour sweep. Two decisions followed. (1) **`D` is waived for `tests/**`, `skyulf-core/tests/**`, `benchmarks/`, `docs/examples/` and `skyulf-core/examples/`** via per-file-ignores: they held 2,528 of the 3,350 sites, and writing 1,852 "missing docstring" stubs into test files would have found no defects while burying the rules that matter. This is a *linting* decision, not a change to the standard — `coding_standards.instructions.md` §4 still asks for a docstring on every function, tests included, so `AGENTS.md` now says explicitly that new tests get a one-line docstring stating what breaks if the test fails, and that nothing will remind you. (2) The remaining sites in `skyulf-core/skyulf/`, `backend/` and the entry points were written by hand. `lint.ignore` is now `["E203", "UP042", "UP046", "UP047", "B008"]` — five entries, none of them docstring rules, with 34 `D` rules confirmed live in `linter.rules.enabled`. Pre-commit runs ruff with `args: [--fix]` (safe fixes only), so a future missing docstring fails the commit rather than being silently rewritten.

**The blind spot: `D1xx` is privacy-gated on the *whole dotted module path*, and 112 of our 327 in-scope modules are private.** Ruff's "missing docstring" rules fire only on *public* definitions, and publicity is derived from the module path — a leading underscore on the module **or on any enclosing package** makes everything beneath it private and exempt. This repo leans hard on underscore-prefixed packages for encapsulation (`ml_pipeline/_internal/`, `_execution/`, `modeling/_tuning/`, `modeling/hyperparameters/`), so **82 sites were never reported by any gate we run**: 64 in `backend/`, 18 in `skyulf-core/`. `D2xx`/`D4xx` *formatting* rules are **not** privacy-gated, which is why the earlier batches looked complete while the missing-docstring half of the same files stayed invisible. Adding an `__init__.py` to an underscore-named directory is enough to switch its whole subtree off — verified empirically. **The audit technique that recovers them:** copy the file to a public filename *outside* its package and lint the copy with a standalone `select = ["D"]` config; the copy is byte-identical, so reported line numbers carry straight back. Renaming the real file is not an option — the underscore is deliberate. All 82 are now written (64 backend across 16 files, 18 core across 7), and re-measurement reports **0**. Recorded in `pyproject.toml`, `AGENTS.md` and the `skyulf-codebase-map` skill, because the failure mode is silent: a clean `ruff check .` is not evidence that a private module has docstrings.

**Verification, because 200+ files were edited by parallel agents and "cosmetic only" had to be proved, not asserted.** Parse each file and its `HEAD` version, strip every leading docstring `Expr` from both, compare `ast.dump`. Identical ⇒ no statement, decorator, default, annotation or import moved. **210 modified Python files: 199 provably docstring-only, 11 real code changes, each one intended and itemised below.** Gates: `ruff check .` clean, `ruff format --check .` clean over 668 files, `ty check` clean, core **3668 passed / 70 skipped**, backend **1630 passed / 7 snapshots passed** — both exact baselines, so no test moved. A second, weaker gate also had to be closed by hand: `E501` is not in `select` and `ruff format` will not reflow prose, so an over-long docstring line passes everything. Eleven such lines were shortened across 8 core files; the pre-existing >100 *code* lines carrying `# pylint: disable=arguments-differ` in the same files were left alone, since reflowing them would be a code change.

**Two agent reports were wrong and only measurement caught them.** One claimed "124/124 cleared" while its own `bucketing.py` still held 7 live `D101`/`D102` sites. Another hit its turn limit and reported nothing, yet had in fact finished all 31 of its files. Lesson: for a pass this wide, re-derive the count from the linter after every batch and treat an agent's summary as a claim, not a result.

**Defects the pass surfaced, then fixed — a docstring is a contract statement, and writing one against the code broke the illusion in six places.**

- **`eda.generate_profile` was invisible to the Celery worker.** `celery_worker.py` imported `data_ingestion.tasks`, `ml_pipeline.tasks` and `monitoring.tasks` — but not `backend.eda.tasks`, and `celery_app.py` does no autodiscovery. With `USE_CELERY=True`, `eda/router.py` calls `generate_profile_celery.delay(...)` and the worker rejects `eda.generate_profile`, leaving the report PENDING forever. Hidden by the default `USE_CELERY=False` BackgroundTasks path. Fixed by adding the import, and pinned by `tests/unit/test_celery_worker_task_registration.py`, a static AST guard asserting every task module is imported by `celery_worker.py` (plus a guard-the-guard test, needed because `@shared_task` is a bare `ast.Name` while `@celery_app.task` is attribute access).
- **The two profilers disagreed on what a duplicate row is.** `GET /datasets/{dataset_id}/schema` returns `AnalysisProfile` from *either* branch: the cached profile reads the ingestion profiler's polars `is_duplicated().sum()` (every row in a duplicate group), while the fresh-sample branch used pandas' `duplicated()` default `keep='first'` (only the extras). The same file therefore reported a different duplicate count depending on whether its profile happened to be cached. Fixed at the outlier — `_advisor.py` now uses `duplicated(keep=False).sum()` — because core's polars analyzer and `expect_unique` already establish `keep=False` as the repo convention; changing the ingestion profiler instead would have created a backend/core divergence. TDD: failing test (`assert 2 == 3`) → one-line fix → `test_both_profilers_agree_on_duplicate_row_count`.
- **Dead parameters removed, after proving no caller passed them.** `BaseConnector.fetch_data(query=...)` was ignored by all three connectors and only ever served to *disable* the lazy path; `safe_delete_path(force_delete=...)` was never read while its docstring still advertised removed "backup options". Both dropped across the ABC, both implementations, `_try_lazy_head` and the affected tests. `DataIngestionService.upload_dir` hardcoded `"uploads/data"` instead of `settings.UPLOAD_DIR`, so a file written through the service was not readable back through `LocalFileConnector`, which resolves against the setting; it now defaults to `get_settings().UPLOAD_DIR` with `.expanduser()`, pinned by two new tests.
- **`MergeMixin.__doc__` was `None` at runtime.** Its description sat as a bare string literal *after* the type-only attribute stubs, so Python never treated it as a docstring — which is why `D101` fired. Proper docstring added at the top and the orphaned literal deleted, because it sat immediately after `artifact_store: Any` where mkdocstrings-style tooling would render it as *that attribute's* documentation.
- **`backend/data_ingestion/engine/` had no `__init__.py`** and was imported as `backend.data_ingestion.engine.profiler` from three places via implicit namespace package, while every sibling directory has one. Added, matching the `connectors/` and `schemas/` style.
- **Two copyright lines had silently lost their `©` and `—`** (`backend/dependencies.py`), and `backend/config/routes.py` had the copyright as its `D415` summary with the real description demoted to paragraph two — the reverse of the pattern the rest of the batch established. `git log -S` confirmed the characters were never present in tracked history, so this pre-dated the pass.

**Still open, deliberately not fixed inside a lint pass.** `SmartCatalog` reads `os.getenv("S3_BUCKET_NAME")` — a *repeat of OC-130*, since pydantic-settings does not export dotenv values into `os.environ`. Worse, the name is not a `Settings` field at all: `backend/config/mixins/aws.py` declares `AWS_BUCKET_NAME`, `docs/guides/backend_configuration.md` documents `AWS_BUCKET_NAME`, and `README.md:105` documents `S3_BUCKET_NAME` — the two docs contradict each other and the code reads the one that no `Settings` field answers to, so S3 auto-init is dead for any `.env`-only configuration. Needs a name decision before a code fix. `ProductionSettings.SECURITY_HEADERS` is declared but read by nothing — grep finds it only in `config/environments.py` — so production security headers are configured and never sent; fixing it means adding a middleware, i.e. a behaviour change, not a lint fix. Also open: `S3Catalog.exists` skips `_prepare_s3fs_options`, so AWS-named credentials and region are never mapped; `FileSystemCatalog.save` is pandas-only (`to_parquet(index=False)`) although `load` returns polars, and its `exists` skips the extension probe `load` uses; `models.py has_permission` is `return True  # Placeholder` with zero callers. The remainder are engine-parity divergences (`casting.py` strict-mode string→bool raises on pandas and succeeds on polars; `invalid_value.py` NaNs a non-numeric column on pandas and raises `ComputeError` on polars) and `node_meta` advertising params the code never reads (`hash.py` says `n_features: 8`, all three paths default to 10; `KBinsDiscretizer` advertises `encode`, `_fit_kbins` hardcodes `"ordinal"`; `casting.py` advertises `type_map` and its `infer_output_schema` and `fit` disagree on precedence, so the predicted schema can contradict the applied cast), plus dead surface (`ScatterSample` is a public model never constructed).

**`ARG` will not be enabled.** 121 in-scope sites across 73 files: 39 are an unused `config`, and ~90 are Calculator/Applier protocol or framework contract signatures (`fit(X, y, config)` on a transformer that needs no `y`, FastAPI's `request`, Celery's `sender`, sklearn-parity `progress_callback`). Enabling it would mean ~90 permanent `# noqa` waivers to police a rule class whose entire in-scope yield was the handful of defects above — all of which were fixable directly, and were. `AGENTS.md` now carries **"never rename an identifier to silence a linter"** as a hard rule, because a prior session prefixed an unused `config` → `_config`, broke keyword callers, and collected a wall of `ty` errors instead of a lint win. One further trap recorded there: **a pydantic model's docstring is wire format** — it becomes the `description` in `model_json_schema()` and reaches clients through OpenAPI, so writing docstrings on `NodeConfigModel`/`PipelineConfigModel` legitimately changed the served schema and the snapshot had to be updated on purpose (7 insertions, 0 deletions; descriptions only, no renamed field and no new required key).

### 2026-09-06 — OC-177–182 filed: 10-file cleaning/encoding continuation

Read all six `preprocessing/cleaning/` files and encoding `__init__.py`,
`_common.py`, `dummy.py`, and `hash.py`. All six findings were reproduced with
the public Calculator/Applier interfaces. Seven existing targeted suites passed
**218 tests** with pytest's cache disabled. No implementation changes were made;
the cumulative source-read ledger is now **79/188 files**, with 109 remaining.

**OC-177 — dummy encoding depends on unrelated prediction rows (🟠).** Fit
`DummyEncoderCalculator` on pandas `x=[1.0,2.0]` with `columns=["x"]` and default
`drop_first=False`; the artifact records categories `["1","2"]`. Applying that
same artifact to `x=[1.0]` gives `x_1=[1], x_2=[0]`. Applying it to
`x=[1.0,2.5]` gives `x_1=[0,0], x_2=[0,0]`: the identical first row is now encoded
as unseen. `_pandas_col_to_str` converts floats to nullable integers only when
**every** non-null value in the current batch is integral, so an unrelated
fractional value changes `1.0`'s string from `"1"` to `"1.0"`. This can change
model predictions depending on batching. **Fix/verification target:** use a
stable per-value representation or a fit-time conversion policy, and assert
that transforming concatenated batches equals concatenating their transforms.
Distinct from OC-18's generated-column collision. Location:
`preprocessing/encoding/dummy.py:60-64`.

**OC-178 — missing categories hash differently across engines/dtypes (🟡).** Fit
`HashEncoder` on Polars `x=["a",None]` with `columns=["x"], n_features=1000`.
Apply this single artifact to the equivalent Polars frame, pandas object frame,
and pandas `pd.Series(["a",None], dtype="string")` frame. Outputs are respectively
`[928,171]`, `[928,915]`, and `[928,870]`. The ordinary category is consistent;
the missing category is not. Polars fills nulls with `"nan"`, while pandas
`astype(str)` renders `None` as `"None"` and nullable missing as `"<NA>"`.
**Fix/verification target:** canonicalize missing inputs before hashing, testing
None, NaN, and pd.NA with a shared artifact and preserving ordinary strings.
The earlier F-11 shared-hash fix did not address this input-normalization gap;
this is not OC-26's separate HashingVectorizer norm failure. Locations:
`preprocessing/encoding/hash.py:45,76`.

**OC-179 — singleton drop-first produces different feature sets (🟡).** Fit/apply
`DummyEncoder` to `x=["a","a"], keep=[1,2]` with `columns=["x"], drop_first=True`.
Pandas returns only `keep`; Polars returns `keep` and `x_a`. The Polars helper
only drops the first category when `len(cats)>1`, whereas pandas drops it even
when it is the sole category. A retained numeric column makes the mismatch
observable without relying on zero-width-frame semantics. **Fix/verification
target:** match the documented pandas.get_dummies behavior for zero, one, and
multiple categories, including engine-crossing application of one artifact.
Distinct from OC-04's dummy integer-width difference and OC-18's name collision.
Location: `preprocessing/encoding/dummy.py:33`.

**OC-180 — slash-date normalization cannot handle nullable missing text (🟡).**
Construct pandas `x=pd.Series(["1/5/2024",None], dtype="string")`; fit/apply
`TextCleaning` with `columns=["x"], operations=[{"op":"regex",
"mode":"normalize_slash_dates"}]`. It raises `TypeError: expected string or
bytes-like object, got 'NAType'`. Equivalent Polars text input returns
`["2024-01-05",None]`. `_normalize_slash_dates_text` guards None and float NaN
but sends pd.NA to `re.Pattern.sub`. **Fix/verification target:** preserve all
supported missing-text representations before regex evaluation; cover object
and nullable string inputs. This is a recognized operation, unlike OC-122's
unknown-operation handling. Locations: `preprocessing/cleaning/text.py:35-37,116`.

**OC-181 — unrecognized boolean mapping keys overwrite False values (🟡).**
Fit/apply `ValueReplacement` to boolean `x=[True,False]` with `columns=["x"],
mapping={"banana":True}`. Both engines return `[True,True]` despite no input
value matching `"banana"`. `_coerce_key` returns `key.lower() in ("true","1")`
for *any* string key on a boolean column, silently treating every other token
as False. **Fix/verification target:** explicitly recognize accepted true/false
spellings; reject or leave unmatched other tokens rather than converting them
to an existing boolean value. Test valid and invalid string keys on both
engines and nullable booleans. Location:
`preprocessing/cleaning/value_replacement.py:31-32`.

**OC-182 — pandas nullable string columns invisible to encoder auto-selection (🟡).**
Construct pandas `x=pd.Series(["a","b"], dtype="string")`. Fit/apply DummyEncoder
or HashEncoder with config `{}`: both return the original text column, with
DummyEncoder recording `columns=[]` and HashEncoder returning `{}`. Repeating
with `columns=["x"]` successfully encodes it. Object-dtype pandas and Polars
String controls are detected. The shared detector selects only `object` and
`category`, omitting pandas StringDtype. **Fix/verification target:** include
supported pandas text extension dtypes; verify omitted selection still differs
from an explicit empty list (intentional no-op). This is a pandas encoder
detector issue, distinct from OC-121's Polars Enum text-helper exclusion.
Location: `preprocessing/encoding/_common.py:140`.

### 2026-09-05 — OC-09 half closed: `F401` enabled repo-wide, after two orphans inside my own security fix proved the gap was not cosmetic

**The finding earned its place by catching me, not by argument.** OC-09 reads as a style complaint — "narrow `ruff select` hides ~500 missing docstrings + 84 unused args" — and it was filed *last* on purpose, because widening the rule set before the codebase was clean would bury real signal in ~580 errors. What made it concrete is that the OC-152 raw-SQL deletion orphaned `from typing import cast` in `backend/database/async_connection_manager.py`, and `KNOWN_ENVIRONMENTS` in `config/factory.py` was imported but referenced only inside a comment. Both passed `ruff check`, `ruff format --check`, `ty check` **and** 1626 tests; an external pyflakes/Prospector pass flagged them post-commit. `ty` is not a substitute — it does not report unused imports at all. So the `F401` half is now enabled and the rest stays open.

**Classification before automation, because `.pre-commit-config.yaml` runs the ruff hook with `args: [--fix]`.** Enabling `F401` makes auto-stripping permanent for every future commit, so a blind `--fix` would silently delete imports that are load-bearing-but-unreferenced. All **27** sites were inspected first:

- **22 dead** — auto-fixed. Includes 4× `timezone` in `test_catalog.py`, 2× `asyncio`, `LocalArtifactStore`, `TargetEncoder{Applier,Calculator}`, `ValidationError`, `StepType`, `Path`, `DatasetProfileArtifact`, `DataSnapshotArtifact`, `TuningResult`, `logging`, and 3× `import numpy as np` in notebooks.
- **4 intentional probes** — waived per-site with `# noqa: F401 - <reason>`, never deleted: `skyulf-core/tests/unit/test_utils.py:441` (polars availability flag feeding 16 tests), `tests/integration/test_frontend_nodes.py:57` (h3), `skyulf-core/tests/integration/test_feature_generation_full.py:288` (polars), `skyulf-core/tests/integration/test_wrapped_polars_frames.py:105` (sentence_transformers). Each exists purely so the `except ImportError` branch runs; the bound name is never read. `test_feature_generation_full.py` also had the alias dropped — lines 63 and 504 genuinely use `pl` and were verified untouched before editing line 288.
- **1 redundant** — 2× `import skyulf` in `test_all_nodes_smoke.py` / `test_registry_contract.py`. This was the one that could have broken node registration, so it was checked rather than assumed: `skyulf/__init__.py` itself imports `.registry` (line 22), and both files still `from skyulf.registry import NodeRegistry`, so the package init — and with it the registration side effect — still runs. Core suite confirms it.

**Two traps hit.** (1) `# noqa` must sit on the **import statement line**; my first edit put it on a preceding comment line next to the `ty: ignore[unresolved-import]` directive and ruff ignored it entirely (the two directives have different placement rules — `ty: ignore` *may* go on the line above). Verified empirically afterwards: 22 errors remained, none at the 4 probe sites. (2) My first notebook safety check was `'numpy' in src or 'np.' in src` over *all* cells and reported numpy still present in two of them — a false alarm from **markdown prose**. Re-run over code cells only with `\bnp\.\w+`: zero uses in all three, plus `json.load` on each to prove the auto-fix didn't mangle the notebook structure. Lesson: when an auto-fix touches notebooks, check the cell type before believing the grep.

**Gates after enablement.** `ruff check .` → All checks passed (F401 now in scope, 0 remaining). `ruff format --check backend skyulf-core tests run_skyulf.py celery_worker.py` → 654 files formatted (one file needed a re-format: deleting `import asyncio` left a double blank line). `ty check` → clean. **skyulf-core: 3669 passed, 70 skipped** — the first core run of the session, and the one that mattered, since 7 core test files changed. **backend: 1626 passed**, exact baseline match.

**Deliberately left open.** `F841` (unused-variable), the docstring rules (~500) and unused-args (84) stay out of `select` — OC-09 remains ⬜ open with its own reasoning intact ("**last**, widening first would bury the signal"). `F401` was safe to add now precisely because its blast radius was 27 sites and 22 were mechanically dead; the same is not true of the docstring rules, which need ~580 human judgements. The remaining effort is now ~2h, not half a day.

### 2026-09-05 — OC-130 + OC-150 + OC-152 fixed: the fail-open security batch; OC-169 filed from what OC-150 exposed

Three findings, backend-only, deliberately **not** gated on the parked OC-71 deployment-model decision — each is a defect at its own call site and none of them requires choosing an auth model first. All three turned out to be **worse or broader than filed**, and each was reproduced by execution against `HEAD` before any edit.

**OC-130 — `FASTAPI_ENV` fails closed.** Root cause was two independent exact-match readers: `factory.py` mapped the value to a settings class with `.get(env, DevelopmentSettings)`, and `base.py`'s production `SECRET_KEY` guard did its own `os.getenv("FASTAPI_ENV", "development").lower()`. Reproduced: `FASTAPI_ENV=prod` → `DevelopmentSettings`, `DEBUG=True`, `CORS_ORIGINS=['*']`, `SECURITY_HEADERS` empty. **The second channel was never filed and is arguably the worse one:** `FASTAPI_ENV` is *not* a `Settings` field, and pydantic-settings loads the dotenv into the model without ever exporting it into `os.environ`, so a bare `os.getenv` cannot see `.env` at all — the documented configuration file was dead for exactly this one setting. Proven before the fix: `.env`-only `FASTAPI_ENV=production` → `DevelopmentSettings`, `DEBUG=True`, `CORS=['*']`. Note why that matters beyond the profile choice: `main.py::_add_middleware` hardcodes `allow_credentials=True`, and Starlette with `allow_origins=["*"]` **reflects the caller's origin** rather than sending a literal `*`, so the browser's normal refusal never fires — any origin can make credentialed requests.

**Decision — fail closed (raise), not fail loud (warn).** A warning is written to the log by the same process that is about to serve wildcard CORS with credentials; nobody reads it before the port opens, and the failure mode is silent-by-construction. `resolve_environment()` now normalizes (`.strip().lower()`, so `"production "` and `PRODUCTION` are accepted) and raises `ValueError` naming every accepted value for anything else, *including empty*. Empty must raise: `FASTAPI_ENV=` is what a YAML/CI variable renders to when unset, and treating it as "development" is precisely the silent fallback being removed. Confirmed `.env.example` has no `FASTAPI_ENV` line, so raising on empty cannot break the documented `cp .env.example .env` quickstart. The resolver reads through a `_EnvironmentSelector(BaseSettings)` with the *same* `SettingsConfigDict` as `Settings`, which is what makes the dotenv channel live; `_ENV_FILE = ".env"` is now a module constant shared by both so the two cannot drift. `KNOWN_ENVIRONMENTS` is the single accepted-value list, and `_ENV_SETTINGS_MAP` was hoisted to module scope so a test can assert `set(_ENV_SETTINGS_MAP) == set(KNOWN_ENVIRONMENTS)` — hoisted specifically because constructing a settings *subclass* calls `setup_logging()`, which strips every handler off the root logger for the rest of the pytest session. Rejected: a `Literal["development","production","testing"]` field on `Settings` itself, which would have moved the choice inside the model and lost the "resolve before you can construct anything" property the factory needs.

**OC-150 — credentials redacted by value shape, one copy.** Filed as a case-sensitivity bug in a key-name blocklist. Execution against `HEAD` showed something worse in *both* directions. Leaking: `_sanitize_error` was a **no-op** on all three real shapes — an S3 403 XML body, a SigV4 presigned URL and an s3fs options-dict repr all came back byte-identical. The dict repr is the finding the audit missed entirely: `{'key': 'AKIA…', 'secret': 'wJalr…'}` exposes the **secret access key** itself, not just a key ID and signature. Over-redacting: a benign `key=reports/2026/q3.csv not found` was replaced wholesale by `redacted sensitive S3 error`, destroying the only useful part of the message. Both failures come from the same design error — matching on the *setting name* and then discarding the *entire message*.

Replaced with `redact_credentials()` in `backend/utils/logging_utils.py`, next to the existing `sanitize_for_log` (CWE-117), and **both** byte-identical `_sanitize_error` copies deleted. Three passes: an AWS access-key-ID shape regex (4-char type prefix + 16 uppercase alnum, word-bounded), a name→separator→value assignment scrub, and an XML-tag scrub for the S3 403 body's `<SignatureProvided>`/`<StringToSign>`/`<AWSSecretAccessKey>`/`<SessionToken>`. Two properties were designed in rather than discovered: **in-place scrubbing** keeps the surrounding diagnostic (`name` and `sep` are re-emitted, only `value` becomes `[REDACTED]`), so the message stays useful; and the value charset excludes `[`/`]` so an already-redacted value cannot be re-matched — that makes the function **idempotent**, which matters because a message may pass through both a call site and a wrapper. It also excludes quotes/braces/parens so the scrub cannot eat into surrounding prose. `key` is in the name list because *both* S3 modules use it as the access-key-id option name (`key` ↔ `aws_access_key_id`). Presigned URLs are treated as **bearer credentials** — anyone holding the URL holds the object — so the constructor's `logger.info` now redacts its caller-supplied `path` too, not just errors.

**Deliberate non-decisions.** (1) The *raised* exception messages stay unredacted: they carry the caller's own input back to them, and redacting would make user-facing errors useless — recorded as a residual, and it is what forces OC-169 below. (2) `main.py`'s wildcard CORS with credentials was left alone: changing dev CORS is OC-72/OC-71 territory and would break local frontend dev. (3) `S3ArtifactStore`'s `path` values are built internally from server config (`s3://{bucket}/{prefix}/{key}`), so they were left unredacted.

**OC-169 filed, not fixed.** `ErrorHandlerMiddleware` logs `{exc}`, `traceback.format_exc()` **and** `exc_info=True` with no redaction, so any *uncaught* exception whose message or frames carry a credential leaks it regardless of how carefully the call site scrubs — and the S3 paths still `raise ConnectionError(...) from e`, leaving the original `e` reachable from the chained traceback. This is an app-wide property of every exception in the process, not a defect in the S3 sanitiser, so expanding OC-150 to cover it would have hidden a systemic issue inside a connector fix. Filed as its own 🟡.

**OC-152 — dead raw-SQL sinks deleted.** Filed as two executors on `AsyncPostgreSQLConnectionManager`; `AsyncSQLiteConnectionManager` carried a byte-identical pair, so **four** were removed. Zero callers — confirmed by grep before deleting. Deleting rather than parameterising: a helper whose entire contract is "accept an unconstrained query string" cannot be made safe by editing its body, and keeping it invites the next caller to rediscover it. `test_no_raw_sql_executor` pins absence by `hasattr` over both classes × both names, with a message pointing at SQLAlchemy constructs / parameterised `sa_text`.

**Correction — a claim in the first draft of this entry was false, and *why* it was false is the useful part.** I wrote that "`ruff check` stayed clean afterwards, which independently proves the `cast`/`Any` imports those methods used are still needed elsewhere." It proves nothing: this repo's `select` is `["E9","F63","F7","F82","I","UP","B","C4","SIM","PERF","BLE","PLC0415"]` and **`F401` is not in it** — only the `F63`/`F7`/`F82` subsets of pyflakes are enabled. Deleting the executors *did* orphan `from typing import cast` in `async_connection_manager.py` (its only two uses were the `cast(int, rowcount)` returns), and `KNOWN_ENVIRONMENTS` in `config/factory.py` was imported but referenced only inside a comment. Both sailed through `ruff check`, `ruff format --check`, `ty check` **and** the full 1626-test suite; an external pyflakes/Prospector pass flagged them after the commit. **Lesson: in this repo a clean `ruff check` is not evidence about unused imports — when a change deletes code, run `ruff check --select F401` on the touched files explicitly.** Scale measured while confirming: **27 F401 errors repo-wide, 24 auto-fixable**, all pre-existing. That is concrete evidence for OC-09, whose "narrow `ruff select`" complaint reads as stylistic right up until it hides a real orphan inside a security fix. *(Superseded later the same day — the lesson above was the workaround, not the fix. `F401` is now in `select`, so plain `ruff check` does cover unused imports and the explicit `--select F401` pass is no longer needed; see the OC-09 half-closed entry.)*

**Tests — 46 new across three files, all proven genuine by reverting only the seven source files to `HEAD` and re-running.** OC-152 gave the strongest signature: **4 real assertion failures** naming the exact methods. OC-130 and OC-150 failed at *import* (`KNOWN_ENVIRONMENTS`, `redact_credentials` do not exist at `HEAD`), which proves the symbols are new but not that the behaviour is — so both were also proven behaviourally with probes against the reverted tree, quoted above (`FASTAPI_ENV=prod` → `DevelopmentSettings`/`DEBUG=True`/`CORS=['*']`/no headers; `.env`-only `production` → same; `_sanitize_error` a no-op on three leak shapes and destructive on a benign one). **Test-fixture bug found by the genuineness pass, not by the suite:** the first two OC-130 guard tests failed with `DID NOT RAISE`, and the cause was the *tests*, not the guard — this shell exports `SECRET_KEY` into the process environment, so `is_field_set("SECRET_KEY")` was legitimately `True` and the production guard correctly stood down. The tests were passing vacuously in any environment that happened not to export it. `isolated_env` now scrubs every name in `Settings.model_fields` in addition to `chdir`-ing to `tmp_path`; a positive control (`SECRET_KEY` supplied ⇒ production boots) pins that the guard fires on *absence* rather than on production generally. Same fixture removed a `PytestCollectionWarning` — `from ... import TestingSettings` put a `Test*`-prefixed class in the module namespace where pytest's `python_classes` pattern tried to collect it; importing the `environments` module and reaching classes through it avoids the binding.

**Gates:** backend **1626 passed** (was 1580, +46), `ruff check` clean, `ruff format --check` clean (666 files), `ty check` clean over `backend skyulf-core/skyulf skyulf-core/tests run_skyulf.py celery_worker.py`. **No frontend work and no rebuild** — zero frontend files touched. Core suite not run — zero `skyulf-core/` files touched. **Residuals left open deliberately:** (1) OC-169, above; (2) raised S3 exception messages are unredacted by design, so a caller who logs the exception it receives re-creates the leak — the fix would be to redact at the *logging* boundary app-wide rather than at each raise site; (3) `redact_credentials` covers AWS shapes only, so a GitHub token, a private key block or a DSN password in an error message still passes through — widening it is a separate decision because each new shape risks eating benign text, which is exactly the failure this fix removed; (4) the remaining 🟠 in this backend cluster is now OC-68 (task-unaware model alias map).

### 2026-09-05 — OC-153 + OC-157 fixed: merge no longer stacks rows in silence, and column order no longer depends on the tiebreak strategy — plus the 0.8.13 bump
Two findings, one file (`_execution/engine/_merge.py`), one investigation pass. **OC-157** root cause: `ordered = indexed if strategy == "last_wins" else list(reversed(indexed))` combined with `result_cols` being a plain dict whose *insertion order becomes the merged frame's column order* — so reversing the iteration to implement `first_wins` also reversed the output columns. Reproduced exactly as filed: `A(a,b) + B(c,d)` → `['a','b','c','d']` under `last_wins`, `['c','d','a','b']` under `first_wins`, values correct under both. **Did not take the audit's suggested shape** ("resolve ownership in reverse but emit columns in forward input order"), which needs a second pass to reorder and can drift from the ownership pass; instead deleted the reversed iteration entirely — walk inputs in their own order under *both* strategies and let the strategy decide only whether a later input may overwrite an already-claimed column. Winners are unchanged because re-assigning an existing dict key keeps its position, so column order is first-appearance input order *by construction* rather than by a fixup. `contested` logging is unaffected (it is consumed as `sorted(set(...))`). **OC-153** root cause: `_merge_frames` picks its merge mode purely from `same_rows`, and `_merge_frames_rowwise` appended to `merge_warnings` only under `if any(common_cols != cs for cs in col_sets)` — so the *identical*-column-set case, which is the most likely accidental one (one branch filtered, the other did not), emitted nothing at all. Reproduced: raw `(5,2)` plus its own outlier-filtered branch `(4,2)` → `(9,2)` with **4 duplicated rows** and `merge_warnings == []` (the audit filed 8 rows / 3 duplicates; same class, my filtered branch kept 4 of 5). **Decision — warn, not raise, departing from both the audit's first option and the OC-154/155 precedent set earlier today.** The audit offered "either raise, or emit a `merge_warnings` entry of the same weight as `row_concat_drop`". Raising would have been *consistent* with the serving-time fixes but wrong here for two reasons: (1) row-wise stacking is a **supported feature**, not only a degradation — `row_concat_drop`'s own UI copy reads "Only columns present in every input are kept when row counts differ", and appending a second labelled dataset is a legitimate merge-node use that a raise would delete; (2) OC-154/155 were raised because they happen at *serving* time, where a wrong number reaches an API caller with a 200 and no human is in the loop, whereas a merge runs during a pipeline/preview where a human is looking at the canvas and the app already has a purpose-built advisory channel for exactly this class. The bug was never the row-wise path; it was its silence. **Frontend work was required, not optional:** `MergeWarningsBanner.tsx` has explicit branches for `row_concat_drop` and `upstream_drop_reapplied` followed by a **fall-through default** rendering the sibling-fan-in copy — "No column overlap — all columns from all branches are kept" plus a "Chain instead" rewire button — so a new kind with no branch would have shown text actively wrong for a row stack, and offered a rewire that cannot fix one. Added the branch and `row_counts?: number[]` to `MergeWarning` (`kind` is typed `string`, not a union, so no widening needed). `_dedup_merge_warnings` keys on `(node_id, kind, inputs, overlap_columns, dropped_columns, part)`, so the new kind dedupes correctly on `part` with no change. The message is tailored on a free signal: identical column sets ⇒ "one branch filtered rows … move that step after the merge"; differing column sets ⇒ "expected when appending separate datasets". Deliberately did *not* compute `merged.duplicated().sum()` for the message — that is an O(rows × width) hash over the whole frame on every merge, and the row counts alone let the user infer it. **Tests**: 7 new across two classes. Proven genuine by reverting only `_merge.py` to `HEAD` and re-running: **5 failed, 2 passed** — and the 2 that passed are precisely the controls (`test_disjoint_columns_keep_input_order[last_wins]`, `test_equal_row_counts_emit_no_mismatch_warning`), which is the wanted signature: controls pass pre-fix so they are discriminating rather than tautological, bug cases fail. **Gates**: backend **1580 passed** (was 1573, +7), `ruff check` + `ruff format --check` clean, `ty check` clean; frontend **873 passed / 104 files**, `tsc --noEmit` clean, eslint clean, `npm run build` done (frontend files touched ⇒ rebuild mandatory). Core suite not run — zero `skyulf-core/` files touched. **Gotcha**: *nothing* exercised the row-count-mismatch path before this — grepping `rowwise|row-wise` outside `_merge.py` returns zero hits and only `test_merge_everywhere.py` references `_merge_frames` at all, so a branch reachable in production had no test coverage, which is how an entire advisory condition could be wrong and go unnoticed. **Also this pass — 0.8.13 bumped**: root `pyproject.toml`, `skyulf-core/setup.py`, `package.json` + `package-lock.json` (via `npm run sync-version`, verified with `check-version`), and the editable install refreshed with `uv pip install -e ./skyulf-core --no-deps` so `skyulf.__version__` reports 0.8.13 instead of stale dist-info. Backend needed **no edit** — `APP_VERSION` comes from `importlib.metadata.version("skyulf")`. Found and closed a real doc gap while doing it: `.github/instructions/versioning.instructions.md`'s 3-step protocol omitted `skyulf-core/setup.py`, which hardcodes its own `version=` and is the file `release.yml` greps to decide whether to publish to PyPI — with **no CI gate comparing it to the root `pyproject.toml`**, so following the documented steps exactly would have silently skipped the core release. **Caveat: that file is gitignored** (`.gitignore:140`, whole `.github/instructions/` directory) and untracked, so the correction is machine-local and the gap stays open for every other contributor — the durable fix is a CI gate asserting `setup.py`'s version equals the root `pyproject.toml`'s, mirroring `npm run check-version`. The changelog bullet describing it was removed for exactly this reason: a gitignored edit is not a shipped, contributor-visible change. **Residuals left open deliberately**: (1) OC-153 is advisory-only by design, so a user who ignores the banner still trains on duplicated rows; if that shows up in practice the next step is a *hard* error specifically when the merge feeds a splitter, because that case is leakage (identical rows on both sides of the boundary) rather than mere reweighting; (2) `_merge_frames` still chooses column-wise vs row-wise purely on row counts, so two branches that coincidentally have the *same* row count but describe *different* observations still join column-wise and yield a plausible-looking wrong frame — equal counts are necessary but not sufficient for alignment, and nothing in the engine checks index or row identity; (3) the remaining 🟠s in this backend cluster are OC-68 (task-unaware model alias map), OC-130 (`FASTAPI_ENV` typo disabling the production security posture) and OC-150 (case-sensitive S3 credential sanitiser).

### 2026-09-05 — OC-25 + OC-143 fixed: RFE now selects the number of features the user asked for — and the two findings turn out to be one
**OC-25 and OC-143 are the same bug filed twice**, from two different audit passes (`03-feature-generation-selection-vectorization.md:48` and `17-file-coverage.md:127`), pointing at the same file and the same line. One fix retires both, so the report's 116-finding inventory double-counts by one here — the distinct total is 115. Root cause: `_build_model_selector`'s `rfe` branch read `config.get("n_features_to_select")`, a key that **nothing in the entire repository ever writes**. It was therefore always `None`, and sklearn's `RFE` fell back to its own default of keeping *half* the candidates. The UI writes `k`: `FeatureSelectionNode.tsx:34` declares `k?: number // SelectKBest, RFE`, line 406 renders that one input for both methods, and line 571 prints `rfe · k=${config.k}` in the node summary — the frontend contract was unambiguous and the backend was reading a different name. **Why it hid for so long:** `step`, RFE's other field, *does* match (`config.get("step", 1)`), so the panel looks fully wired up; and the fixture `feature_selection_common.json` encoded the bug as the intended contract — its `rfe` case passed `n_features_to_select: 2`, the internal spelling that already worked, leaving the `k` path the UI actually sends with **zero coverage**. A green suite coexisted with a broken feature, exactly as the audit diagnosed. Fix: a `_resolve_rfe_n_features` helper matching the file's existing alias-tolerance convention (`_normalize_univariate_method`, `_resolve_generic_param`, `_resolve_problem_type`). Placed in **core, not `pipelineConverter.ts`**, deliberately — the converter already passes `node.data` through untouched (`params = node.data || {}`), so fixing the reader repairs the canvas, direct API callers and notebooks in one move; same single-owner decision as OC-14. An explicit `n_features_to_select` wins over `k`, so the documented sklearn spelling keeps working and a config carrying both is not ambiguous. The audit's suggested one-liner `config.get("n_features_to_select", config.get("k"))` was rejected: a `.get` default only fires when the key is *absent*, so a client sending `{"n_features_to_select": null, "k": 3}` — what a cleared field serialises to — would still get `None`. **Reproduced** the audit's three-way proof on 6 candidate features: control `select_k_best k=3` → 3 (probe valid), `rfe k=2` → **3** pre-fix (half of 6; the user asked for 2), `rfe n_features_to_select=2` → 2 (proof of cause). Post-fix all four checks correct, precedence included (`k=5` + `n_features_to_select=2` → 2). **Tests**: 2 fixture cases added to the `build_model_selector` group (`rfe_accepts_ui_k_alias`, `rfe_explicit_name_wins_over_k`) plus an end-to-end node test in `test_feature_selection_gaps.py` parametrized over both spellings. Proven genuine by reverting only `_common.py` to `HEAD`: both `k`-path tests failed (`assert 3 == 2`, `where 3 = len(['x1', 'x2', 'x5'])`) while all three controls passed — the parametrization carries its own control, so a broken harness cannot masquerade as a regression. **Gotcha worth recording:** the fixture-driven test IDs are generated from parameter *values* (`test_build_model_selector[rfe-LogisticRegression-config4-RFE-n_features_to_select-3]`), not from the JSON case name, so `-k rfe_accepts_ui_k_alias` silently matches nothing and looks like a passing suite; filter on `TestBuildModelSelector` instead. **Gates**: core **3669 passed / 70 skipped**, backend **1573 passed**, `ruff check` + `ruff format --check` (663 files) clean, `ty check` clean. No frontend change and no rebuild — the UI already sends and displays `k` correctly; the defect was entirely on the reading side. `mkdocs build --strict` not run — no doc change. **Residual left open deliberately**: `ModelBasedSelection`'s `@node_meta` declares only `{"estimator", "threshold", "max_features"}` — no `method`, no `step`, no `k`/`n_features_to_select` — so the entire RFE path is undeclared in the metadata R1 step 1 wants as the single source of truth (same class as OC-06). Not touched here: adding keys to `node_meta.params` changes the node's advertised defaults and belongs to the R1 contract work, not a drive-by edit.

### 2026-09-05 — OC-154/OC-155 fixed: both serving paths now fail closed instead of returning a wrong number as a normal result
One hazard class, two sites: the deployment predict path *degraded silently* wherever it could not guarantee feature alignment, so an arbitrary-magnitude wrong prediction reached the API caller with a 200. **OC-154** root cause: F-02's reindex guarded itself with `if not missing:` — it disabled the protection precisely when alignment could not be confirmed, which is the exact state F-02 was introduced to prevent. Reproduced: a bundle trained on `a, b, c` whose feature engineer emits `a, b, d` skipped the reindex and handed a positional model (no `feature_names_in_`, so nothing downstream could object) the frame as-produced — consumed `[1, 2, 99]` as if it were `[a, b, c]` under weights `1/10/100` and returned **`9921.0`**, no error, no warning. Fix: raise a `ValueError` naming *both* the produced and the recorded column sets plus the specific missing ones, so the message distinguishes "your request omitted a column" from "this bundle's transform no longer matches what it was trained on" — the two have different owners and different remedies, and reusing `_validate_required_columns`' wording would have conflated them. **OC-155** root cause: the legacy path imputed absent features with the literal constant `0`, which for any feature not centred on zero (income, age, price, anything scaled) is an extreme out-of-distribution input. Reproduced: model trained on `income, age, price`, request omits `price` → returned **`50030.0`** as a normal result behind a server-side `logger.warning`. **Worse than filed** — the loop was `df[c] = 0`, which mutates the *caller's* DataFrame, so the fabricated column outlived the request and the input frame left the function wider than it entered (`['income','age']` → `['income','age','price']`); the audit only reported the bad prediction. Fix reuses the `_validate_required_columns` helper the bundled path already called 15 lines earlier, which removes the zero-fill and the mutation in one move and makes the two paths agree. **Why raise rather than warn:** a warning reproduces OC-155's own complaint — a log line the API caller never reads — while `api.py:129-130` already maps `ValueError` → HTTP 400 with `detail=str(e)`, and `client.ts`'s response interceptor already normalizes `detail` into `error.message`. So the column names reach the browser with **zero frontend work**; this extends an error mode the bundled path has emitted since F-03 rather than introducing a new one, which is why no rebuild was needed. The audit's fallback (impute from the artifact's *training* statistics) was not taken: silently inventing feature values is what made the bug invisible, and a genuinely-optional feature should be modelled as one, not papered over at serving time. **Tests**: renamed `test_predict_with_legacy_artifact_fills_missing_columns_and_reorders` → `..._reorders_columns_to_match_model` (it only ever exercised reordering — both columns present — so the old name advertised coverage the test never had), replaced `..._fills_zero_for_missing_column` with `..._raises_on_missing_column` (also asserts the caller's frame is no longer mutated), and added `test_predict_with_bundled_artifact_raises_when_transform_output_misaligns`. Both new tests were proven genuine by reverting only `service.py` to `HEAD` and re-running: both failed, and the OC-154 one failed by *reaching* `predict()` — the bug expressed directly as a test outcome rather than as a mismatched value. **Gates**: backend **1573 passed**, `ruff check` + `ruff format --check` clean, `ty check` clean. Core suite not run — backend-only change, zero `skyulf-core/` files touched; likewise zero frontend files touched. `mkdocs build --strict` not run — no doc change, and a grep of `docs/` found nothing describing the zero-fill. **Residuals left open deliberately**: (1) this is a behaviour change, not just a bug fix — a deployment whose persisted `feature_columns` no longer matches what its own bundled feature engineer emits (e.g. an artifact recorded before F-03) now returns a 400 where it used to predict. That is the intended trade, but the remedy for such a bundle is retrain/redeploy, not code, and none were enumerated here; (2) the bare-numpy legacy case is still unfixable by construction — an artifact with no `feature_names_in_` carries no record of its expected columns, so nothing can be validated or reordered, and the guard remains `hasattr(artifact, "feature_names_in_")`; (3) OC-157 (`first_wins` reversing output column order) was explicitly the same hazard class per the audit — column order reaching positional consumers — and was still open at the time of writing; it is fixed in the OC-153 + OC-157 entry above.

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
