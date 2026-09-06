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

**Remaining-source continuation (2026-09-06):** OC-187–206 add 20 executed
findings (5 🟠 / 14 🟡 / 1 ⚪), filed directly in the continuation table under
**New findings**. All remain open. The original core-source ledger now records
**188/188 files read**; the 45 selected modeling/profiling test files passed
**909 tests** (142 warnings). The exact command is recorded in
[`core_source_review_2026-09-05.md`](core_source_review_2026-09-05.md).
Historical baseline counts remain unchanged; no implementation fixes were made
by this review.

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
| OC-163 | 🟠 | `LagFeatures` / `RollingAggregate` sort X without reordering tuple y on both engines — `[3,1,2]` times become `[1,2,3]` while targets remain `[300,100,200]`, silently training on wrong labels (`preprocessing/time_series/lag.py:45,81`, `rolling.py:63,119`) | small | ✅ fixed 2026-09-06 — both engines now derive one positional permutation and hand it to X and y alike, which retired OC-165 and OC-166 in the same pass. See the log entry |
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
| OC-165 | 🟡 | Pandas `LagFeatures(drop_na=True)` removes X rows but leaves tuple y untouched — 3 rows become 2 features / 3 targets even with a unique index (`preprocessing/time_series/lag.py:85-87`) | small | ✅ fixed 2026-09-06 — with OC-163; `drop_na` now filters y through the same positional keep-mask as X, duplicate-index case included. See the log entry |
| OC-166 | 🟡 | Polars `IQR`, `ZScore`, and `ManualBounds` filter X but leave NumPy y untouched — 5 rows become 4 features / 5 targets; Polars Series y works (`preprocessing/outliers/_common.py:9-15`) | small | ✅ fixed 2026-09-06 — with OC-163, and **broader than filed**: a fourth copy of the same silent pass-through sat inline in `EllipticEnvelope`, and list targets failed too (crashing on pandas, no-opping on polars). See the log entry |

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

All entries below have executed reproduction evidence. Source paths are
relative to `skyulf-core/skyulf/`; line numbers refer to the source read during
the review and may move with concurrent edits. The main reviewer independently
reproduced the filed symptoms before completing the source ledger.

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-187 | 🟡 | LightGBM's advertised `subsample` control and default search dimension have no effect: both calculators retain native `subsample_freq=0`, disabling row bagging (`modeling/hyperparameters/_tree.py:576`, `_registry.py:298,309`; `classification.py:754`, `regression.py:547`) | small | ⬜ open |
| OC-188 | 🟠 | Rule discovery decodes sklearn class positions against Polars' shared category dictionary, publishing labels absent from the target while reporting perfect accuracy (`profiling/_analyzer/rules.py:169-170,196-198,295-298`) | small | ⬜ open |
| OC-189 | 🟡 | Classification rule text reports `Samples: 1` for leaves containing multiple rows: it sums sklearn's normalized class proportions instead of using the leaf sample count (`profiling/_analyzer/rules.py:299-301`) | small | ⬜ open |
| OC-190 | 🟡 | A categorical column named `count` crashes profiling and categorical drift because `value_counts()` generates the same column name (`profiling/analyzer.py:286-290`, `profiling/drift.py:380-381`) | small | ⬜ open |
| OC-191 | 🟡 | All-null and Polars Enum columns are classified as text and sent to string-only aggregates, aborting the whole profile (`profiling/analyzer.py`, `_analyzer/column.py`) | small | ⬜ open |
| OC-192 | 🟡 | Decomposition's categorical null bucket displays as `Unknown`, but drilling into it filters for the literal string and silently loses the bucket's rows (`profiling/_analyzer/decomposition.py:71-76`) | small | ⬜ open |
| OC-193 | 🟡 | A single missing timestamp removes time-series analysis at the 1,000-row resampling boundary: dynamic grouping receives null date keys and the exception is swallowed (`profiling/_analyzer/temporal.py:232,243`) | small | ⬜ open |
| OC-194 | 🟠 | Pandas time-series CV sorts with `Series.argsort()`'s `-1` missing-date sentinels as row positions, duplicating/dropping observations and destroying chronological order (`modeling/cross_validation.py:327-330`) | small | ⬜ open |
| OC-195 | 🟡 | Clustering numeric-feature selection skips `SkyulfPandasWrapper`, so wrapping a working pandas frame with text columns makes fitting fail (`modeling/clustering.py:40-53`) | small | ⬜ open |
| OC-196 | 🟡 | GaussianMixture probability prediction omits the feature/reference filtering used for fit and ordinary prediction, causing a feature-count mismatch on the same input (`modeling/clustering.py:83-89`, `modeling/sklearn_wrapper.py:266-279`) | small | ⬜ open |
| OC-197 | 🟡 | Polars clustering reference crosstabs crash for reference columns named `count` or `__skyulf_cluster__` (`modeling/_evaluation/clustering.py:137-146`) | small | ⬜ open |
| OC-198 | 🟠 | Profiling a string target overwrites an existing `<target>_encoded` feature, then duplicate selection prevents correlation and causal analysis (`profiling/analyzer.py:341-348`) | small | ⬜ open |
| OC-199 | 🟡 | Explicit latitude/longitude selections bypass `exclude_cols`, returning coordinates for columns excluded from the profile (`profiling/_analyzer/geo.py:58-59`) | small | ⬜ open |
| OC-200 | 🟠 | Halving search accepts an all-NaN score set as a successful best result and refits a model; grid search correctly fails on identical folds (`modeling/_tuning/strategies/runner.py:110-127`) | small | ⬜ open |
| OC-201 | 🟡 | Optuna skips search-space normalization: `max_depth=['none']` works in grid search but fails every Optuna trial (`modeling/_tuning/strategies/optuna.py:199`) | small | ⬜ open |
| OC-202 | 🟡 | Fold-aware tuning wrapper omits `decision_function` and unconditionally advertises `predict_proba`, breaking ROC-AUC scoring for SVC without probability support (`modeling/_tuning/fold_pipeline.py:164-172`) | small | ⬜ open |
| OC-203 | 🟡 | Optuna CMA-ES treats Boolean candidates as integers, turning valid `fit_intercept=[True,False]` into invalid sklearn parameter values (`modeling/_tuning/strategies/optuna.py:113-140`) | small | ⬜ open |
| OC-204 | 🟡 | `fit_predict` drops an embedded target during training but keeps it in held-out tuple features when explicit y is also supplied, causing prediction to fail (`modeling/base.py:317-324`) | small | ⬜ open |
| OC-205 | 🟠 | Grid/random tuning discards failed folds from each candidate's average, allowing a partially failed candidate to win with an apparently valid score and no failure count in the result (`modeling/_tuning/grid_random.py:91-92`) | small | ⬜ open |
| OC-206 | ⚪ | Ensemble configuration resolution shallow-copies nested base-model parameters, so fitting mutates the caller's configuration (`modeling/ensemble.py:473,484`) | small | ⬜ open |

**OC-206 — fitting an ensemble mutates caller configuration.** Executed
`VotingClassifierCalculator().fit` with one decision-tree base learner,
`base_estimator_params={'decision_tree':{'max_depth':2}}`, and
`decision_tree__min_samples_leaf=3`. After fitting, the caller's original
`base_estimator_params['decision_tree']` has gained `min_samples_leaf:3`.
Only the outer mapping is copied before nested keys are absorbed. Reusing the
configuration after removing a temporary override therefore retains it.
**Fix/verification target:** copy the nested parameter mappings before
normalization and pin non-mutation of caller-owned configuration.

**OC-204 — tuple target extraction differs between train and test.** Executed
`StatefulEstimator(LogisticRegressionCalculator(),LogisticRegressionApplier(),'probe')`
with `X=DataFrame({'x':range(10),'target':[0]*5+[1]*5})`, and a `SplitDataset`
whose train/test splits both contain `(X,X.target)`. `fit_predict(...,
'target',{})` fits one feature, then raises
`X has 2 features, but LogisticRegression is expecting 1 features as input`.
Changing tuple y to `None` succeeds for both splits. **Fix/verification target:**
use the same target-column exclusion contract for training, test and validation,
regardless of whether y is supplied separately.

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

**OC-194 — missing dates corrupt time-series CV rows.** Executed
`_sort_pandas_by_column` with dates
`['2024-01-03',None,'2024-01-01',None,'2024-01-02']`, row IDs
`[0,1,2,3,4]`, and targets `[100,101,102,103,104]`. It returns row IDs
**[1,4,2,4,0]** and targets **[101,104,102,104,100]**. Row 3 disappears and
row 4 occurs twice; the retained dated rows are not chronological. Pandas
also emits a warning about the missing-value `argsort` behavior. The helper
consumes sentinel positions as valid negative `iloc` positions before dropping
the date column. **Fix/verification target:** construct a genuine positional
sort permutation with an explicit missing-date policy; verify one-to-one row
preservation, chronological order, and X/y alignment. Separate from OC-162's
Polars temporary-column collision.

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

**OC-198 — profiling target encoding overwrites a real feature.** Executed
`EDAAnalyzer` on `x=[1,2,3]`, `target=['a','a','b']`,
`target_encoded=[100,200,300]`, then `analyze(target_col='target')`.
The analyzer's `target_encoded` values become **[0,0,1]**. Correlation and
causal discovery log duplicate-projection errors because the helper's name
is also appended to the feature list. **Fix/verification target:** avoid
overwriting user columns and duplicating feature names when materializing
an encoded target; preserve the original values across repeated analysis.

**OC-199 — explicitly selected coordinates survive exclusion.** Executed
`EDAAnalyzer(pl.DataFrame({'lat':[1.,2.,3.], 'lon':[10.,20.,30.],
'x':[1.,2.,3.]})).analyze(exclude_cols=['lat','lon'],lat_col='lat',lon_col='lon')`.
The result still contains all three coordinate pairs in
`geospatial.sample_points`, plus their bounds and centroid, although the
per-column profile excludes them. **Fix/verification target:** apply the
exclusion policy consistently before explicit geospatial selection.

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
