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
22 ⚪, plus OC-160–206 filed by later reviews. OC-100 was retracted as a false
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

The **Now** and **Next** tiers have no open findings left — those rows are in
the archive. **Next** closed its last two filed rows (OC-177, OC-164) on
2026-09-06, immediately re-opened with OC-207 filed *while* fixing OC-164, and
closed that on 2026-09-07.

### Then — decide deployment model first

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-71 | 🟠 | **No authentication or authorization anywhere on the API** (`main.py:373-395`, `database/models.py:151-159`) — **confirm intent first**: single-tenant self-hosted → documentation task; multi-tenant → highest-priority item in the entire report (scaffolded `User` model + dead `AUTH_FALLBACK_*` settings suggest the latter was intended) | decision + ~1 week | ⏭️ parked — user requested pause | PARKED!
| OC-72 | 🟡 | Insecure-by-default config: unset `FASTAPI_ENV` fails open to wildcard CORS + credentials (`config/factory.py:26`, `main.py:359-366`) | small | ⏭️ parked — with OC-71 |
| OC-73 | ⚪ | `DataSource.credentials` documented encrypted, stored plaintext JSON (`database/models.py:107`) | small | ⏭️ parked — with OC-71 |

### Ongoing — remove the hiding conditions

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-03 | 🟠 | Systemic `infer_output_schema` int→float misprediction across 22 nodes — one sweep + parametrized test (predicted schema == actual schema for every node) | ~1 day | ✅ done — runtime dtype parity covered by parametrized tests |

### Remaining — evaluation & explainability

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-148 | 🟡 | PII detector flags ordinary 7+ digit numeric ID columns as "Email/Phone" (`profiling/_analyzer/text.py:107-128`) | small | ⬜ open |
| OC-38 | ⚪ | Clustering metrics treat DBSCAN `-1` noise as a real cluster (`metrics.py:432-459`) | small | ⬜ open |

### Remaining — backend infrastructure

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-68 | 🟠 | Model alias map task-unaware — direct API caller silently trains the wrong estimator family (`_execution/engine/_node_runners.py:1157-1183`) | small | ⬜ open |
| OC-70 | 🟡 | Leakage validator checks for *a* splitter globally, not that *this* branch is protected (`_execution/_leakage_validation.py:189-267`) | small | ⬜ open |
| OC-145 | 🟡 | Crashed cross-validation returns the same `{}` sentinel as a disabled one — job reports success with missing `cv_*` metrics (`_node_runners.py:871-907`) | small | ⬜ open |
| OC-151 | 🟡 | Trial-buffer `clear_*` hooks documented but never called — 110.9 MB retained for process lifetime (`realtime/trial_buffer.py:56-59,103-106`) | small | ⬜ open |
| OC-156 | 🟡 | `roc_auc` threshold-tuning objective scores hard predictions — bit-identical to `balanced_accuracy` (`threshold_tuning_service.py:77-92`) | small | ⬜ open |
| OC-158 | 🟡 | Sync/async JSON serializers disagree: sync nulls 8 of 15 legitimate strings (`"nan"`, `"NaT"`, `"<NA>"`, `"inf"`…), async nulls none; 603-line module production-dead but test-covered (`serialization.py:369,435-446`) | half day | ⬜ open |
| OC-169 | 🟡 | Filed while fixing OC-150 — the global `ErrorHandlerMiddleware` logs `{exc}`, `traceback.format_exc()` **and** `exc_info=True` with no redaction, so any *uncaught* exception whose message or frames carry a credential leaks it to the log regardless of call-site scrubbing; the S3 paths now redact their own `logger.error` but still `raise ConnectionError(...) from e`, leaving `e` reachable from the chained traceback (`middleware/error_handler.py:53-65`) | small | ⬜ open |
| OC-183 | 🟠 | `SmartCatalog` S3 auto-init is dead for `.env`-only config, and the two docs name different variables — **OC-130's root cause repeating**. `backend/data/catalog.py:556` reads `os.getenv("S3_BUCKET_NAME")`, but pydantic-settings loads the dotenv into the model and never exports it into `os.environ`, so a bucket configured only in `.env` is invisible and `s3_catalog` silently stays `None` (falling back to local disk with no error or warning). Worse, `S3_BUCKET_NAME` is **not a `Settings` field at all**: `config/mixins/aws.py:12` declares `AWS_BUCKET_NAME`, which is what `docs/guides/backend_configuration.md:146` documents, while `README.md:105` documents `S3_BUCKET_NAME` — so following the README sets a variable nothing reads. Needs a canonical-name decision before the one-line code fix | small | ⬜ open |
| OC-184 | 🟠 | `ProductionSettings.SECURITY_HEADERS` is declared and never sent. `_PROD_SECURITY_HEADERS` (HSTS, `X-Frame-Options: DENY`, CSP, …) is assigned at `config/environments.py:84` and referenced nowhere else in the repo — no middleware reads it — so a production boot logs "Running in PRODUCTION mode with enhanced security" while emitting none of those headers. Fixing means adding a security-headers middleware in `main.py::_add_middleware`, where order is load-bearing (CORS must stay outermost), i.e. a behaviour change and not a config fix | half day | ⬜ open |
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
| OC-114 | 🟡 | All-null tracked column yields 30 `NaN` autocorrelation lags as real analysis (≥1000-row datasets) (`temporal.py:167-191`) | small | ⬜ open |
| OC-102 | ⚪ | Five tunable models return an empty search space from the live `/defaults` endpoint (`hyperparameters/_registry.py`) | small | ⬜ open |
| OC-121 | ⚪ | polars `Enum` columns invisible to text auto-detection, diverging from pandas `Categorical` (`_helpers.py:148-157`) | small | ⬜ open |
| OC-122 | ⚪ | `TextCleaning` silently ignores unrecognised operation name (`cleaning/text.py:151-153`) | small | ⬜ open |
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
| OC-178 | 🟡 | `HashEncoder` hashes the same missing value into different buckets across Polars, pandas object, and pandas nullable string inputs, even with one shared fitted artifact (`preprocessing/encoding/hash.py:45,76`) | small | ⬜ open |
| OC-179 | 🟡 | `DummyEncoder(drop_first=True)` retains a single-category indicator on Polars but removes it on pandas, changing feature width across engines (`preprocessing/encoding/dummy.py:33`) | small | ⬜ open |
| OC-180 | 🟡 | Pandas `TextCleaning(normalize_slash_dates)` crashes on `pd.NA` in a nullable string column; equivalent Polars input preserves the missing value (`preprocessing/cleaning/text.py:35-37,116`) | small | ⬜ open |
| OC-181 | 🟡 | `ValueReplacement` coerces every unrecognized boolean mapping key to `False`: mapping `{"banana": true}` changes `[true,false]` to `[true,true]` on both engines (`preprocessing/cleaning/value_replacement.py:31-32`) | small | ⬜ open |
| OC-182 | 🟡 | Encoder auto-detection ignores pandas `StringDtype` columns: Dummy/Hash encoding silently leaves strings untouched unless columns are selected explicitly (`preprocessing/encoding/_common.py:140`) | small | ⬜ open |
| OC-171 | 🟡 | Pandas `SimpleImputer` silently excludes explicitly selected constant/binary numeric columns for mean/median, leaving missing values unfilled; Polars honors the selection (`preprocessing/imputation/simple.py:173-177`) | small | ⬜ open |
| OC-172 | 🟡 | `StandardScaler` crashes on mixed pandas nullable numeric columns containing `pd.NA`; native sklearn and equivalent Polars input succeed (`preprocessing/scaling/standard.py:144,154`, `engines/sklearn_bridge.py:52`) | small | ⬜ open |
| OC-18 | 🟡 | One-hot/dummy generated names can collide with existing columns (`encoding/one_hot.py:68-92`, `dummy.py:76-99`) | small | ⬜ open |
| OC-21 | 🟡 | WOE additive smoothing not normalized over categories (`encoding/woe.py:130-145`) | small | ⬜ open |

### Remaining — feature generation / selection / vectorization / transformations

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-23 | 🟠 | Polars `ratio` flips the sign of near-zero negative denominators (`feature_generation/_polars_ops.py:97-112`) | small | ⬜ open |
| OC-24 | 🟠 | Polars group aggregates treat null group keys differently from pandas (`_polars_ops.py:222-234`) | small | ⬜ open |
| OC-26 | 🟠 | `HashingVectorizer` UI "none" norm is an invalid sklearn value → crash (`hashing_vectorizer.py:59`) | small | ⬜ open |
| OC-27 | 🟠 | `GeneralTransformation` ignores the UI `standardize` toggle (`transformations/general.py:34-39,138-139`) | small | ⬜ open |
| OC-29 | 🟡 | `FeatureGeneration` advertises `polynomial` but silently skips it (`feature_generation/_common.py:24-31`) | small | ⬜ open |
| OC-30 | 🟡 | Datetime extraction ignores the UI output name, overwrites collisions (`_pandas_ops.py:173-184`) | small | ⬜ open |
| OC-31 | 🟡 | Frontend wrongly requires a target for unsupervised CorrelationThreshold (`FeatureSelectionNode.tsx:564-566`) | small | ⬜ open |
| OC-32 | 🟡 | `VarianceThreshold` crashes when all candidates are constant (`feature_selection/variance.py:38-47`) | small | ⬜ open |
| OC-33 | 🟡 | `FeatureInteraction` cannot generate single-column self-products (`feature_generation/interaction.py:173-178`) | small | ⬜ open |
| OC-34 | 🟡 | Count/TF-IDF vectorizers crash on empty or stop-word-only corpora (`count_vectorizer.py:79-80`) | small | ⬜ open |

### Remaining — profiling (outside the OC-39–46 cluster)

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-188 | 🟠 | Rule discovery decodes sklearn class positions against Polars' shared category dictionary, publishing labels absent from the target while reporting perfect accuracy (`profiling/_analyzer/rules.py:169-170,196-198,295-298`) | small | ⬜ open |
| OC-198 | 🟠 | Profiling a string target overwrites an existing `<target>_encoded` feature, then duplicate selection prevents correlation and causal analysis (`profiling/analyzer.py:341-348`) | small | ⬜ open |
| OC-189 | 🟡 | Classification rule text reports `Samples: 1` for leaves containing multiple rows: it sums sklearn's normalized class proportions instead of using the leaf sample count (`profiling/_analyzer/rules.py:299-301`) | small | ⬜ open |
| OC-190 | 🟡 | A categorical column named `count` crashes profiling and categorical drift because `value_counts()` generates the same column name (`profiling/analyzer.py:286-290`, `profiling/drift.py:380-381`) | small | ⬜ open |
| OC-191 | 🟡 | All-null and Polars Enum columns are classified as text and sent to string-only aggregates, aborting the whole profile (`profiling/analyzer.py`, `_analyzer/column.py`) | small | ⬜ open |
| OC-192 | 🟡 | Decomposition's categorical null bucket displays as `Unknown`, but drilling into it filters for the literal string and silently loses the bucket's rows (`profiling/_analyzer/decomposition.py:71-76`) | small | ⬜ open |
| OC-193 | 🟡 | A single missing timestamp removes time-series analysis at the 1,000-row resampling boundary: dynamic grouping receives null date keys and the exception is swallowed (`profiling/_analyzer/temporal.py:232,243`) | small | ⬜ open |
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

### Remaining — modeling / tuning

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-194 | 🟠 | Pandas time-series CV sorts with `Series.argsort()`'s `-1` missing-date sentinels as row positions, duplicating/dropping observations and destroying chronological order (`modeling/cross_validation.py:327-330`) | small | ⬜ open |
| OC-187 | 🟡 | LightGBM's advertised `subsample` control and default search dimension have no effect: both calculators retain native `subsample_freq=0`, disabling row bagging (`modeling/hyperparameters/_tree.py:576`, `_registry.py:298,309`; `classification.py:754`, `regression.py:547`) | small | ⬜ open |
| OC-204 | 🟡 | `fit_predict` drops an embedded target during training but keeps it in held-out tuple features when explicit y is also supplied, causing prediction to fail (`modeling/base.py:317-324`) | small | ⬜ open |
| OC-206 | ⚪ | Ensemble configuration resolution shallow-copies nested base-model parameters, so fitting mutates the caller's configuration (`modeling/ensemble.py:473,484`) | small | ⬜ open |
| OC-168 | 🟡 | `SkyulfPipeline.fit()` retains the previous model's tuned thresholds — refitting with new class labels makes thresholded prediction crash; unchanged labels reuse stale cutoffs (`pipeline/_pipeline.py:135,380-389`) | small | ⬜ open |

### Remaining — frontend

| ID | Sev | Item | Effort | Status |
|---|---|---|---|---|
| OC-54 | 🟡 | `DebugNode` is dead code that would silently no-op if wired up (`nodes/DebugNode.tsx`) | small | ⬜ open |
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

All six were reproduced through executed Python probes against the working tree and checked against the existing tracker and relevant source-audit reports. IDs follow the review's reported order. **Four have since closed** — OC-163/165/166 on 2026-09-06 with the shared positional y-selection helper, and OC-164 the same day — leaving OC-167 and OC-168 below, both still open; their reproduction detail and the fixes' Log entries are in the archive.

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
