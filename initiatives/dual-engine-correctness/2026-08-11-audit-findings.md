# Dual-Engine (Polars / Pandas) Correctness Audit — Findings

**Date:** 2026-08-11
**Branch:** `078`
**Baseline:** `master` @ `acfd9b6d`
**Versions at audit time:** `skyulf-core` 0.5.8 · backend 0.7.8 · frontend 0.7.8
**Method:** 4 parallel Opus-5 audit agents (engine-parity, pandas-purity, leakage, inference/experiments)
plus an adversarial rubber-duck pass. All findings below were **reproduced with executable probes**,
not inferred from reading code. Probes ran against throwaway worktrees; repo source untouched.

> **Coverage denominators — read before quoting any number from this document.**
> The registry contains **100 nodes** (verified live: appliers, calculators and metadata all
> return 100). Different agents probed different **subsets**, and those subset figures must not be
> read as full coverage:
>
> | Figure | What it actually means |
> |---|---|
> | **100** | Total registered nodes. Engine-capability and pandas-purity checks covered all 100. |
> | **61** | Non-modeling transformers probed for cross-engine output parity. |
> | **38** | Models probed for cross-engine training/prediction parity. |
> | **41** | Transformers probed for cross-engine *artifact replay* (fit on one engine, apply on the other). |
>
> An earlier draft of this document said "99 nodes". That was an error; the correct total is 100.

---

## 0. Executive summary

**Is it bad?** No — the architecture is sound and the pandas path is solid. But the Polars path
carries six real correctness bugs, and there are three **engine-independent** inference bugs that
affect pandas users too. Nothing here is unfixable; most are one- to ten-line fixes. The honest
position today is: *Polars is capable end-to-end but not yet at parity, and inference has a
column-ordering bug that must be fixed before anyone deploys.*

| Layer | Verdict |
|---|---|
| **Engine / data path** | ✅ Sound. Of **100 registered nodes**, 0 are Polars-incapable and 0 silently downgrade Polars→pandas. Pandas users stay 100% pandas (all 100 verified). |
| **Preprocessing nodes** | ⚠️ 54 of **61 transformers probed** match across engines. **7 diverge**, 1 hard-breaks Polars training. |
| **Modeling / numpy handoff** | ✅ **Provably correct.** Column order and values identical across engines; predictions bit-identical; CV/tuning/evaluation JSON identical (one 1-ULP `log_loss` float-summation difference). 37 of **38 models probed** match. |
| **Persistence** | ✅ joblib round-trip clean. Artifacts are plain `list`/`dict`/`str`/`bool` — no engine-specific objects. 40/41 transformers replay safely cross-engine (HashEncoder is the exception). |
| **Inference / deployment** | 🔴 **3 CRITICAL bugs, 2 of them engine-independent.** This is the weakest layer. |
| **Monitoring / drift** | ⚠️ One live bug (NaN → silently reports "no drift"). |
| **Experiments** | 🔴 **Worst-affected layer. 16 findings, 12 LIVE today.** Mostly *not* engine bugs — metric-semantics and UI-state defects that make users read the wrong number and act on it. Includes a CRITICAL race showing another job's evaluation, and "Recall" tuning that is literally accuracy and *reduces* recall. See §6. |
| **Leakage** | ⚠️ The per-node fit/apply discipline is **genuinely solid** — the core claim is true. But enforcement covers only one graph pattern in one execution path, WOE lacks cross-fitting, and CV does not re-fit preprocessing per fold. |

**Answering the specific questions asked:**

- *Does modeling work with Polars?* Yes — and this part is proven, not assumed. The
  Polars→numpy→sklearn handoff produces bit-identical predictions to the pandas path.
- *Does inference work?* Only if your pipeline has no column-adding transformer, and only if
  callers happen to send JSON keys in training order. Both are unacceptable — see F-02, F-03.
- *Do experiments work?* **This is the weakest layer, and it was nearly missed.** 16 findings, 12
  LIVE today without any Polars involvement: the evaluation panel can show another job's data,
  "Recall" threshold tuning is accuracy in disguise and *worsens* recall, "Best Score" compares
  incomparable metrics, and SHAP/feature importance are permanently blank for 6 of 11 classifier
  families while the UI blames a stale run. See §6.
- *Is everything else in place?* The 58-file diff on this branch (31 source, 20 test, 7
  version/changelog) is legitimate and reviewed. The bugs below are **pre-existing**, not
  regressions from that work.

**Two corrections to earlier statements made during this work:**

1. An earlier round reported the `is_null()`-misses-NaN fix as applied to `iqr.py` and `zscore.py`.
   **It was not.** `grep -rn is_nan skyulf/` returns only `drop_rows.py`, `missing_indicator.py`,
   `profiling/expect.py`. Recorded below as **F-06**.
2. An audit agent reported drift/SHAP as dead for the Polars engine (HIGH). That is a real code
   defect but is **not reachable in production** — the backend catalog reads exclusively via
   `pd.read_csv`/`pd.read_parquet`, and commit `7485ade6` added a `TypeError` guard at the single
   data-entry point. Downgraded to LOW/latent (**F-30**, **F-31**).

---

## 0.5 Triage — 49 findings is not 49 emergencies

49 is a large number and it invites panic. It shouldn't. The findings split cleanly by **who is
affected today**, and that split changes the priority order completely.

**The key fact:** the hosted backend runs **pandas only** (`backend/data/catalog.py` reads
exclusively via `pd.read_csv`/`pd.read_parquet`; `_node_runners.py:154` rejects anything else).
So the Polars parity bugs — the largest single group — **do not affect canvas users today**.

| Who | Count | What it means |
|---|---|---|
| **LIVE for canvas users** (pandas, today) | **~24** | Experiments (14), inference (F-02/F-03), leakage (4), drift NaN, the pandas nullable-`Int64` crashes. **Fix these first.** |
| **LIVE only for `skyulf-core` SDK users running Polars** | **~17** | Every engine-parity finding (F-01, F-04…F-11, F-43). Real, but a narrower audience. Also the **blocking prerequisite** for the backend Polars migration. |
| **Latent / minor** | **~8** | F-31, F-32 (unreachable while the catalog is pandas-only), dead code, dtype cosmetics, packaging. |

**Three practical consequences:**

1. **The scariest-sounding group is the least urgent right now.** "DummyEncoder breaks Polars
   training" is severe *in isolation*, but no canvas user can hit it today. It becomes urgent the
   moment the backend Polars migration starts — which is exactly why that migration is gated on
   fixing it first (see the migration plan).
2. **The most urgent group is the one nobody was looking at.** The experiments subsystem was
   initially reported as "nothing to audit". It turned out to hold **12 LIVE bugs**, mostly not
   engine-related at all — they make users read the wrong number and act on it. A user who trusts
   a wrong evaluation panel is worse off than one who hits a crash.
3. **Nothing here is a regression.** Every finding is pre-existing and passes the current
   195-test parity suite plus 226 deploy/predict tests. This is **test coverage debt**, not
   recently-broken code. The suite never exercised float `NaN` (only nulls) or wrapped frames.

**Suggested first commit — 7 fixes, all engine-independent, all LIVE:**
F-02, F-03 (deployment is broken for most real pipelines), F-33 (evaluation panel shows another
job's data), F-34, F-35 ("Recall" tuning is accuracy and *worsens* recall), F-37 (~2 lines restores
SHAP for 6 model families), plus F-01 to unblock the migration track.

That single tier removes every CRITICAL and the highest-impact HIGHs. The remaining ~40 are then a
normal, schedulable backlog rather than an emergency.

---

## 1. The single highest-yield bug pattern

**Polars `is_null()` does not match float `NaN`; pandas `isna()` matches both.** Polars stores
`null` and `NaN` as genuinely distinct values. Three independent audit agents converged on this
same root cause from different directions. It underlies F-04, F-06, F-07, F-13, F-19, F-20, F-21.

The corollary applies to every Polars null-adjacent API:
`fill_null()` does not fill NaN · `drop_nulls()` does not drop NaN · `null_count()` does not count
NaN · `null == x` evaluates to `null`, not `False` · `.sort()` places nulls **first**, so
`.mode().sort().first()` can select null as the mode.

**Test debt (cross-cutting):** the existing 195-test parity suite passes clean against
*every single bug in this document*. Parity tests exercise **nulls but never float NaN**, and
**raw frames but never wrapped frames**. Fixing the tests is as important as fixing the code.

---

## 2. Findings

Severity: 🔴 CRITICAL (silent wrong results or hard break) · 🟠 HIGH · 🟡 MED · ⚪ LOW

### Tier 1 — CRITICAL

#### F-01 🔴 `DummyEncoder` emits null dummies on Polars, breaking training
`skyulf-core/skyulf/preprocessing/encoding/dummy.py:79`

`(pl.col(c).cast(Utf8) == str(cat)).cast(Int8)` — for a null input, `null == x` is `null`, so
rows with a null category get **null dummies**. Pandas `get_dummies` gives `0`.
Downstream `LogisticRegression` raises `ValueError: Input X contains NaN` on Polars while the
identical pandas pipeline trains to 0.99 accuracy. Fix: wrap in `.fill_null(0)` after the cast,
or guard with `pl.when(pl.col(c).is_null()).then(0)`.

#### F-02 🔴 Inference trusts positional column order — silently wrong predictions
`backend/ml_pipeline/deployment/service.py:450` and `:356` · **engine-independent**

**Status: fixed in `4e95f170` (0.7.9).** `_predict_with_bundled_artifact` reindexes `X_transformed` to
the recorded training `feature_columns` order after `_transform_bundled_features` (guarded: only
when all recorded columns are present). Covered by
`tests/unit/test_deployment_service_extra.py::test_predict_with_bundled_artifact_reorders_columns_to_match_training`.

`SklearnBridge` (`skyulf/engines/sklearn_bridge.py:37`) hands sklearn a bare numpy array, so
sklearn never records `feature_names_in_` and cannot validate or reorder at predict time.
At serve time `df = pd.DataFrame(data)` takes column order from **the JSON key order of the
caller's first record**, `_validate_required_columns` (`:371-385`) checks *presence only and never
reorders*, then `estimator.predict()` consumes positionally.

Reproduced against the real `_predict_with_bundled_artifact`, and over live HTTP via FastAPI
`TestClient` (pydantic's `list[dict[str, Any]]` preserves key order verbatim):

```
[canonical]  cols=['age','income','score'] -> preds=[1, 0]
[reordered]  cols=['income','score','age'] -> preds=[1, 1]   # SAME DATA
SKEW DETECTED: True    (identical for pandas- and polars-trained models)
```

sklearn emits `UserWarning: X has feature names, but LogisticRegression was fitted without feature
names` — swallowed to stderr, never surfaced to the caller.
**Aggravating:** the *legacy* path (`:420-428`) does reorder via `df = df[model_cols]`. The modern
bundled path regressed. Fix: reindex to the recorded training feature order after
`_transform_bundled_features`.

#### F-03 🔴 `feature_columns` recorded post-transform, validated pre-transform
recorded `_node_runners.py:254-265` · validated `service.py:403` · schema `service.py:562-564`

**Status: fixed in `4e95f170` (0.7.9).** Validation now uses `_extract_features_from_engineer`
(pre-transform input columns) and the API/UI input schema uses
`_extract_features_from_bundled_artifact` (engineer → `feature_columns` → `feature_names_in_`
fallback). Frontend sync check passed: `InferencePage.tsx`/`deployment.ts` consume `input_schema`
generically (`{name, type}[]`), no post-transform assumption. Covered by
`tests/unit/test_deployment_service_extra.py::test_predict_with_bundled_artifact_validates_pre_transform_columns`
and the `_extract_features_from_*` / `_extract_input_features` tests.

The training node records feature names from its *input* — i.e. **after** feature engineering —
but deployment validates them against the **raw request frame**, before `feature_engineer.transform`.
The same wrong list also drives the API/UI input schema.

Reproduced on OneHotEncoder + StandardScaler, **both engines**:

```
recorded feature_columns   : ['age','income','color_blue','color_green','color_red']
API-advertised input_schema: ['age','income','color_blue','color_green','color_red']  # user has 'color'!
request {'age':75,'income':50000,'color':'red'}
  -> ValueError: Missing required column(s): ['color_blue','color_green','color_red']
```

**Impact:** any pipeline containing a column-*adding* transformer (OneHot, DateFeatures,
Polynomial, MissingIndicator, FeatureGeneration, any vectorizer) is **undeployable**, and the UI
asks users for fields that don't exist. It slipped through because only strictly in-place
pipelines (scalers/imputers) were ever deployed in tests.

### Tier 2 — HIGH, silent data corruption

#### F-04 🟠 `SimpleImputer` silently no-ops on NaN under Polars
`preprocessing/imputation/simple.py:45`, `_common.py:43-61`

`fill_null()` does not fill NaN. `[1,2,3,4,NaN]` with `strategy="mean"` →
pandas `[1,2,3,4,2.5]` vs polars `[1,2,3,4,NaN]`. The imputer reports success. Model then either
crashes or trains on NaN.

#### F-05 🟠 `SimpleImputer` `most_frequent` picks **null** as the mode on Polars
`preprocessing/imputation/_common.py:39`

`.mode().sort().first()` — Polars sorts nulls first, so a column whose most common value is null
yields `fill_value=None` → `ValueError` at apply time.

#### F-06 🟠 IQR / ZScore / ManualBounds silently delete rows containing NaN
`outliers/iqr.py:40`, `zscore.py:43`, `manual_bounds.py:27`
**⚠️ Previously reported as fixed. The fix was never applied — verified by grep.**

`col_mask | pl.col(col).is_null()` preserves nulls but **not NaN**, so NaN rows fail the bound
check and are dropped. `[1,2,3,4,5,100,NaN,None]` → pandas keeps 7 rows, polars keeps 6.
Silent row loss in an outlier step is exactly the failure mode users will never notice.
Fix: `| pl.col(col).is_null() | pl.col(col).is_nan()` (as already done in `drop_rows.py:23`).

#### F-07 🟠 `OrdinalEncoder` produces three different answers for the same data
`encoding/ordinal.py:37, 190, 290`

Missing the `.fill_null("nan")` normalisation that `label.py` applies at `:51, 78, 156, 174`.
Result: pandas-with-NaN, pandas-with-None-object, and polars each yield different codes.

#### F-08 🟠 `KBinsDiscretizer` silently clips out-of-range test values on Polars
Polars clips to the outer bin; pandas yields NaN. Test-time out-of-range values are therefore
silently absorbed on one engine and flagged on the other.

#### F-09 🟠 `SkyulfPolarsWrapper` crashes 7 of 61 nodes
`vectorization/_common.py:47`, `encoding/one_hot.py:70`, `feature_generation/polynomial.py:63`

The wrapper is a **documented public input type**. Passing a wrapped Polars frame crashes
OneHotEncoder, PolynomialFeatures and the 4 vectorizers. Wrapped *pandas*: 0 failures.
Asymmetry, not an inherent limitation.

#### F-10 🟠 Four pandas-only crashes on nullable `Int64`
`imputation/simple.py:73`, `knn.py:70`, `iterative.py:76`, `outliers/winsorize.py:57`

Polars handles all four correctly; pandas raises. Reachable via `.convert_dtypes()` or
Arrow-backed input. Note this is the one bug class where **pandas is the broken engine.**

#### F-11 🟠 `HashEncoder` produces different buckets per engine — artifact not portable
`encoding/hash.py:34` (polars native `.hash()`) vs `:53` (`blake2b`)

The file *documents* this as an accepted limitation — but `service.py:450` **guarantees** engine
crossing for every Polars-trained pipeline (serving always builds a `pd.DataFrame`). So the
"accepted limitation" is unconditionally triggered in production:

```
fitted on polars: train mapping a->3, b->3, c->5
   SERVE pandas -> [8, 9, 8]   # wrong buckets, no error, no warning
   SERVE polars -> [3, 4, 3]   # correct
```

Fix: use one hash algorithm for both engines. This is the **only** apply-time engine-divergent
transformer out of 41 tested.

#### F-12 🟠 Polars clustering deployment is completely broken
`_node_runners.py:260` — `if numeric_only and hasattr(train_frame, "select_dtypes")`

`select_dtypes` is a pandas-only API; `pl.DataFrame` and `SkyulfPolarsWrapper` lack it, so the
numeric filter is silently skipped and the artifact records non-numeric columns the model was
never fit on (`modeling/clustering.py:43-54` *does* drop them at fit).

```
[pandas] KMeans n_features_in_=2, feature_columns=['amount','freq']              -> predict OK
[polars] KMeans n_features_in_=2, feature_columns=['customer_id','amount','freq']
   send what the API advertises -> ValueError: could not convert string to float: 'c1'
   send only the real features  -> ValueError: Missing required column(s): ['customer_id']
```

Both routes fail. `DeploymentService` also bypasses `_NumericOnlyClusteringApplier` entirely, so
the fit-time numeric drop is never mirrored at serve time.

#### F-13 🟠 Drift detection silently reports "no drift" when NaN is present
`skyulf/profiling/drift.py:141-142` — `drop_nulls()` doesn't drop NaN

NaN propagates into the statistics; `nan > threshold` is `False`, so `has_drift=False`.
**Live today** via any uploaded CSV containing a literal `NaN` token (`pl.read_csv` produces NaN,
not null):

```
clean CSV                : wasserstein=8.0   ks_p=3.29e-45   -> drift flagged
CSV with one 'NaN' token : wasserstein=nan   ks_p=nan        -> both metrics vote "no drift"
```

A monitoring system that fails *silently closed* is worse than no monitoring.
Fix: `.drop_nans().drop_nulls()`.

#### F-14 🟠 `WOEEncoder` has no cross-fitting hook — measurable target leakage
`encoding/woe.py:218-231` (has only `fit`; `target.py:323` has `fit_transform_train`)

Quantified on a **pure-noise target** (ground truth AUC 0.500):

```
WOEEncoder     CV ROC-AUC = 0.791   <-- leaking
TargetEncoder  CV ROC-AUC = 0.503   <-- correct
```

#### F-15 🟠 Preprocessing is never re-fit inside CV folds — all CV/tuning scores optimistic
`skyulf/modeling/cross_validation.py`, `_tuning/engine.py:271-357, 663`

Zero references to the preprocessing layer; only the *model* is re-fit per fold. Preprocessing is
fitted once on the full training set, so every fold's validation data has already influenced the
imputation means, scaler statistics and encodings. **This is architectural**, not a patch — see
§4.

### Tier 3 — MED

| ID | Finding | Location |
|---|---|---|
| F-16 🟡 | 10 stateful nodes bypass the pre-split leakage gate (`MissingIndicator`, `DropMissingColumns`, Over/Undersampling, `HashEncoder`, `Deduplicate`, …). The comment wrongly calls the first two "stateless"; both provably learn columns from data. **Status:** ✅ Fixed (T3 leakage enforcement on `081`) — every node now declares a required `learns_from_data` on `@node_meta`; both gates derive their lists from the registry (duplicate backend frozenset deleted); the six stateful nodes are reclassified and gated; unknown transformers fail closed. ⚠️ Breaking: `on_leakage` defaults to `"raise"`. | `_leakage_validation.py:30-38` |
| F-17 🟡 | Pipelines with **no splitter node** get zero leakage protection — the gate is keyed on finding a split. **Status:** ✅ Fixed (T3 leakage enforcement on `081`) — both gates now emit an explicit advisory diagnostic ("the leakage guarantee does not apply") instead of returning nothing; silence removed. | `leakage.py:53-54` |
| F-18 🟡 | Row-dropping steps (`Deduplicate`, `DropMissingRows`) still execute at inference; 4 requested rows → 2 predictions, and `PredictionResponse.predictions: list[Any]` carries no row keys, so the caller cannot tell which inputs vanished. **Status:** ✅ Fixed (T3 parity batch on `081`) — `FeatureEngineer.transform()` now skips row-dropping step types; a null row surfaces as a visible model error instead of a silent misalignment. | `preprocessing/pipeline.py:64-70` |
| F-19 🟡 | `DropMissingColumns` `null_count()` misses NaN → different columns dropped per engine. **Status:** ✅ Fixed (T3 parity batch on `081`) — Polars path counts `is_null() + is_nan()` on float columns, matching pandas `isna()`; parity test covers a 50%-NaN frame on both engines. | `drop_and_missing/drop_columns.py:44` |
| F-20 🟡 | `DatasetProfile` on Polars reports `missing=0` while `mean=NaN` for NaN-bearing columns — self-contradictory profile. **Status:** ✅ Fixed (T3 parity batch on `081`) — both `_compute_frame_stats` and `_compute_basic_stats` count NaN as missing on float columns. | `profiling/` |
| F-21 🟡 | `lag.py` `drop_nulls()` vs pandas `dropna()` → different row counts. **Status:** ✅ Already resolved by the Polars migration (verified on `081`). | `time_series/lag.py:54` |
| F-22 🟡 | `DropMissingColumns` registers param `threshold` but reads `missing_threshold` — the UI value is ignored. **Status:** ✅ Fixed (T3 parity batch on `081`) — the `@node_meta` declaration now reads `missing_threshold`, matching what fit() consumes and what the frontend sends; declaration-only change, no user-facing param rename. | `drop_columns.py:91` |
| F-23 🟡 | `DummyEncoder` null rows: `0/0` int (pandas) vs `NaN/NaN` float (polars) — dtype divergence distinct from F-01. **Status:** ✅ Already resolved by the Polars migration (verified on `081`). | `encoding/dummy.py` |
| F-24 🟡 | `most_frequent` on an all-unique column: pandas silently no-ops, polars raises. **Status:** ✅ Already resolved by the Polars migration (verified on `081`). | `imputation/simple.py:66` |
| F-25 🟡 | **No engine identity recorded in the deployment bundle.** Nothing detects "trained on Polars, serving on pandas." Adding `"train_engine": "polars"\|"pandas"` and warning on mismatch would have caught F-11, F-12, F-30, F-31 at deploy time. **Status:** ✅ Fixed (T3 parity batch on `081`) — the bundle already records `engine` via `_resolve_train_engine`; serving now logs a warning when a non-pandas-trained bundle receives pandas input. | deployment bundle |
| F-26 🟡 | Polars is a **hard dependency** (`pipeline.py:12`; `setup.py:27` `install_requires`), so `polars_engine.py`'s `HAS_POLARS` fallback is unreachable dead code, tested only via monkeypatch. It advertises an optionality that has never worked. Matches packaging, so not a broken promise — but delete the dead branch or make it real. **Status:** ✅ Already resolved by the Polars migration (verified on `081` during T4). | `polars_engine.py` |

### Tier 4 — LOW

| ID | Finding | Location |
|---|---|---|
| F-27 ⚪ | Mixed-engine row filters raise `AttributeError` instead of a clear `TypeError`. **Status:** ✅ Fixed (T4 cleanup on `081`) — all three dispatcher entry points now reject mixed-engine `(X, y)` pairs with a clear `TypeError`; engine-neutral y (lists/numpy) unaffected. | `drop_rows.py:69`, `outliers/_common.py:25` |
| F-28 ⚪ | WOE artifact null key is `None` on one engine, `nan` on the other. **Status:** ✅ Fixed (T4 cleanup on `081`) — pandas fit/apply/cross-fit now normalise nulls to the `"nan"` key via `_string_keys_with_nan`, matching the Polars path. ⚠️ Pre-release pandas-fitted artifacts keep their `"None"` key until re-fit. | `woe.py:167-186` |
| F-29 ⚪ | `SimpleImputer` median dtype: `Int64` vs `Float64`. **Status:** ✅ Already resolved by the Polars migration (verified on `081`: both engines record a Python `float`). | `imputation/simple.py` |
| F-30 ⚪ | `_pretty_dtype` is pandas-only → Polars deployments show `unknown` for Date, Duration, Categorical. **Status:** ✅ Already resolved (verified on `081`: Polars dtype names recognised). | `service.py:613-638` |
| F-31 ⚪ | `_normalize_train_frame` returns `None` for Polars (drift reference + SHAP frame). **Status:** ✅ Already resolved by the Polars migration (verified on `081`: Polars frames and `(X, y)` tuples handled; `_shap_input_frame` converts at the boundary). | `_artifacts.py:154-178` |
| F-32 ⚪ | `_feature_names_for_importance` pandas-only → `feature_importances = None` on Polars. **Status:** ✅ Already resolved by the Polars migration (verified on `081`: accepts `pl.DataFrame` on all data shapes). | `_artifacts.py:38-58` |

---

## 3. What is proven correct

Stated explicitly because it is as important as the bug list, and because these were *measured*:

- **Polars runs end-to-end.** Of the **100 registered nodes**, **zero** are Polars-incapable and
  **zero** silently downgrade Polars → pandas. (Registry composition: 34 Modeling, 30
  Preprocessing, 9 Feature Engineering, 6 Cleaning, 5 Data Operations, 5 Feature Selection,
  5 Text, 4 Ensemble, 2 Inspection.)
- **Pandas users stay on pandas.** All 100 nodes verified — no silent Polars conversion anywhere.
- **The numpy handoff is correct.** Column order and values identical across engines. Predictions
  bit-identical. CV, tuning and evaluation JSON identical, apart from a 1-ULP `log_loss` difference
  caused by float summation order (`0.0406001434` on both to 10 s.f.). 37/38 models match.
- **Persistence is engine-clean.** Fitted artifacts are plain `list`/`dict`/`str`/`bool` — no
  Polars objects, no pandas `Index`, no engine-specific dtype. Save → load → predict is stable, and
  pandas-trained and polars-trained bundles agree on identical rows.
- **40 of 41 transformers replay safely cross-engine** (HashEncoder is the sole exception, F-11).
  StandardScaler showed a 1-ULP difference — floating-point noise, not a bug.
- **`DeploymentService` URI resolution, threshold reconciliation and label decoding** are
  engine-agnostic (they operate on numpy and dicts).
- **The core leakage discipline holds.** The `fit()` (Calculator) / `transform()` (Applier)
  separation is genuinely enforced per node. The adversarial "poisoned test set" proof in
  `docs/examples/leakage_proof.md` is valid *for the steps it covers*.
- **226 existing deploy/predict/artifact/drift tests pass.** Every bug above is a **coverage gap,
  not a regression.**
- The 58-file diff on `078` vs `master` (31 source, 20 test, 7 version/changelog) was reviewed
  file-by-file and is legitimate.

**Explicitly not verified:** `S3ArtifactStore` (no credentials; delegates to joblib so it should
mirror local). There is **no batch-prediction path** in the codebase (`grep batch` finds only Celery
pipeline batching) — nothing to audit there.

**Previously mis-reported as "nothing to audit":** the **experiments subsystem**. See §6.

---

## 4. The architectural question: per-fold preprocessing refit (F-15)

Flagged separately because it is a **design decision, not a bug fix**, and the user has asked for
investigation rather than implementation.

**The problem.** `cross_validation.py` and `_tuning/engine.py` re-fit only the estimator per fold.
Preprocessing is fitted once, on the full training set, before CV begins. Every fold's validation
rows therefore contributed to the imputation means, scaler statistics and encoder mappings used to
transform them. All reported CV and tuning scores are systematically optimistic.

**Why it is not a quick patch.**
- It changes the meaning of every score Skyulf has ever reported. Existing users will see their
  numbers move *down* after upgrade. That needs a release note, not a silent fix.
- It requires the preprocessing pipeline to be re-fittable per fold, which means a clean
  clone/reset contract on every calculator — currently not guaranteed.
- It multiplies preprocessing cost by `n_splits`; for KNN/Iterative imputation that is significant.
- Tuning (`_tuning/engine.py:271-357`) would need the same treatment, or tuning and CV will
  disagree.

**Suggested investigation output** (before writing any code): a short design note covering the
refit contract, the performance budget, the migration/communication plan for changed scores, and
whether to offer it as opt-in first (`refit_preprocessing_per_fold=True`) then flip the default in
a minor release.

---

## 5. Recommended fix order and versioning

Commits are grouped by tier; each tier is one commit with its own tests.

| Tier | Contents | Version | Release type |
|---|---|---|---|
| **T1** | F-01, F-02, F-03, **F-33, F-34, F-35, F-37** | core **0.5.9** · backend/frontend **0.7.9** | **Patch — ship first, alone.** All engine-independent and LIVE. F-02/F-03 block deployments; F-33 shows the wrong job's evaluation; F-35 makes "Recall" tuning actively harmful; F-37 is a ~2-line SHAP fix restoring 6 model families. |
| **T2** | F-04 … F-14 (excl. F-15) | core **0.6.0** · backend/frontend **0.8.0** | **Minor.** Behaviour changes: rows previously dropped are now kept (F-06), imputers that previously no-opped now impute (F-04), HashEncoder buckets change (F-11 — artifacts fitted before this release will not reproduce, needs a release note). |
| **T2b** | F-36, F-38, F-39, F-40, F-41, F-42, F-43, F-44 | backend/frontend ~~0.8.0~~ → **0.8.1** | **Done on `080` 2026-08-22; slipped 0.8.0, ships in 0.8.1** (F-43 already shipped via core 0.6.0). Experiments correctness: metric comparability, diff rendering, threshold validation. F-36 changes what the comparison table displays — called out in the 0.8.1 changelog. |
| **T3** | F-16 … F-26 + leakage enforcement | core **0.6.1** · backend/frontend **0.8.1** | **All findings fixed on `081` 2026-08-22.** Parity batch (F-18/F-19/F-20/F-22/F-25) and leakage enforcement (F-16/F-17, ⚠️ breaking `on_leakage="raise"` default) landed red-green; F-21/F-23/F-24/F-26 already resolved by the Polars migration. Remaining T3 work: pure-noise encoder regression test + doc-site updates. |
| **T4** | F-27 … F-32, F-45 … F-48 + dead-code cleanup (F-26 fallback, F-48 `refit()`) | core **0.6.1** · backend/frontend **0.8.1** | **Done on `081` 2026-08-22; folded into the upcoming 0.8.1 release.** F-26 and F-29–F-32 were already resolved by the Polars migration; F-27/F-28/F-45–F-48 fixed red-green. |
| **T5** | F-15 per-fold refit | core **0.7.0** | **Minor/major — separate initiative.** Changes reported scores. Design note first. |

**Cross-cutting, do in T1:** add engine-parity tests that use **float NaN** (not only nulls) and
**wrapped** frames. Without this the suite will keep passing while broken. Every fix below must be
verified red-green — write the failing test first, confirm it fails, then fix.

**Frontend sync check required for:** F-03 (the deployment input-schema drives the UI form).
F-22 needed no frontend change: `DropColumnsNode.tsx` already sends `missing_threshold`; only
the core `@node_meta` declaration was corrected. Per repo policy, backend param/enum changes must
be mirrored in `frontend/ml-canvas/src/modules/nodes/`.

**Docs to update alongside the code:**
- `docs/examples/leakage_proof.md:459` — scope the "leakage-free by design" conclusion (see the
  companion enforcement plan).
- `docs/index.md:99`, `docs/user_guide/validation_vs_sklearn.md:290` — same.
- `changelog/0.7.x.md` and `docs/` engine-parity notes — record the T2 behaviour changes,
  especially the HashEncoder artifact break.
- Any doc asserting Polars/pandas parity should not make that claim until T2 ships.

---

## 6. Experiments subsystem

Initially mis-reported. An agent grepped for MLflow and Weights & Biases, found neither, and
concluded *"No experiment tracker exists — nothing to audit."* That was wrong: Skyulf has a
substantial **native, job-based** experiments subsystem. A dedicated Opus-5 audit was run to close
the gap; findings F-33 … F-48 below.

**Process lesson:** an agent reporting "this subsystem does not exist" must be checked against the
frontend routes and API surface before the claim is accepted. Absence of a well-known third-party
integration is not evidence of absence of the capability.

**What exists:** `ExperimentsPage.tsx` + `ExperimentsPage/` (comparison table, metrics chart,
evaluation, SHAP, feature importance, segmentation, threshold tuning, branch comparison, job list)
and `experiments/` (pipeline diff). Backend: `/pipeline/jobs/{job_id}/evaluation`
(`_routers/jobs.py:194` → `_services/evaluation_service.py`), `/jobs/{job_id}/thresholds{,/preview,/save}`
(`ThresholdTuningService`), `model_registry/`, promote/unpromote.

**Headline:** this subsystem is in worse shape than the engine. Of 16 findings, **12 are LIVE
today** — reachable by ordinary use, no Polars required. Most are not engine bugs at all; they are
metric-semantics and UI-state bugs that cause users to **read the wrong number and act on it**.

### F-33 🔴 CRITICAL · LIVE — Evaluation panel silently shows another job's data
`frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx:227-255`

**Status:** ✅ Fixed (0.7.9 wave) — evaluation fetch + threshold-tuning state extracted into the `useEvaluationFetch` hook with a monotonic request-sequence guard that discards late/stale responses; 5 new hook tests, full frontend suite green.

`fetchEvaluationData` sets `evalJobId` synchronously, then `await`s the fetch and applies the
response **unconditionally** — no request-id or `AbortController` guard. Verified at source: there
is no staleness check between the `await` and `setEvaluationData(res.data)`.

```
Click job A (slow), then job B (fast)
FINAL STATE: evalJobId="B", evaluationData="evaluation-data-for-A"
```

Clicking between runs is the *normal* way to use the page. The header says B while the confusion
matrix, ROC/PR curves, per-class metrics and threshold tuner are all A's. No error, no spinner.
Fix: capture a monotonic request id before the `await`, discard if it is no longer current.

### F-34 🟠 HIGH · LIVE — "ROC AUC" threshold tuning 500s on string class labels
`backend/ml_pipeline/_services/threshold_tuning_service.py:46`; router `_routers/jobs.py:228-240`
catches only `ThresholdTuningError`

**Status:** ✅ Fixed (0.7.9 wave) — metric scorers are now built per request via
`_build_scorer()` in `threshold_tuning_service.py`; the `roc_auc` scorer maps
raw labels (string or numeric) to 0/1 positive-indicator arrays before
`roc_auc_score`, so string targets no longer 500. Rank-preserving for numeric
labels, so existing 0/1 behavior is unchanged; regression test
`test_preview_roc_auc_works_with_string_labels` green.

Train on a CSV with a `yes`/`no` target (the engine never requires label encoding), open
Evaluation → Threshold Tuning → "ROC AUC":

```
accuracy, f1, precision, recall, balanced_accuracy  -> HTTP 200
roc_auc                                             -> HTTP 500 "Internal server error"
  ValueError: dtype='numeric' is not compatible with arrays of bytes/strings.
```

5 of 6 options work, so it reads as a server fault rather than a data-shape limit.
Fix: validate before selecting the scorer and raise `ThresholdTuningError` (→400), or use
`LabelBinarizer`.

### F-35 🟠 HIGH · LIVE — "Recall" tuning is literally Accuracy, and makes recall *worse*
`backend/ml_pipeline/_services/threshold_tuning_service.py:36-47`

**Status:** ✅ Fixed (0.7.9 wave) — `_build_scorer()` now returns
`average="binary"`, `pos_label=classes[1]` scorers for `f1`/`precision`/
`recall` on 2-class jobs (multiclass keeps the weighted scorers), so
tuning "Recall" optimizes positive-class recall again; regression test
`test_preview_recall_uses_positive_class_not_class_mixture` fails with
`thresholds["yes"] == 0.6078` (accuracy optimum) before the fix and passes
after.

Every scorer in `_METRIC_SCORERS` uses `average="weighted"`. Weighted-average recall is
**identical to accuracy by definition** — independently reproduced:

```
acc=0.5200000000  weighted-recall=0.5200000000  equal=True
acc=0.5133333333  weighted-recall=0.5133333333  equal=True
```

Effect on a real binary problem (class balance 491/109):

```
option              thr(pos)  pos-recall  pos-precision  accuracy
(no tuning, 0.5)      0.5000      0.4587         0.6667    0.8600
recall                0.5980      0.4037         0.8000    0.8733   <- identical to accuracy
f1                    0.3529      0.6422         0.6306    0.8667
balanced_accuracy     0.1961      0.8257         0.5056    0.8217
```

The entire point of threshold tuning is "catch more positives". A user selecting **Recall** gets
positive-class recall **0.4587 → 0.4037 — worse than not tuning at all**, reported as success.
Precision and F1 are mislabelled the same way.
Fix: for 2-class problems use `average="binary"` with `pos_label=classes[1]`; otherwise rename the
options "Recall (weighted)" etc. and add explicit binary variants.

### F-36 🟠 HIGH · LIVE — "Best Score" compares different metrics under one label
`ComparisonTableView.tsx:370-371`, `MetricsComparisonChart.tsx:39,141`, `BranchComparisonCard.tsx:101-102`

**Status:** ✅ Fixed (Experiments batch 2) — `best_score` is now expanded
into one row per scoring metric via `groupJobsByScoringMetric` in both
starring surfaces (`ComparisonTableView`, `BranchComparisonCard`): each row
is labelled from its own metric ("Best Score (F1 Weighted)"), values are
masked to the jobs that optimised that metric, and `pickBestIndex` stars
only within the group. `MetricsComparisonChart` was checked and carries no
`selectedJobs[0]`-derived label — its bar is split-labelled ("CV mean"), so
no change was needed there. Helper covered by 4 unit tests; red-green.

```
basic run (run_mode=fixed) : best_score=0.9   scoring_metric=accuracy
tuned run (run_mode=tuned) : best_score=0.92  scoring_metric=f1_weighted

Row rendered: "Best Score (accuracy)"   values compared: [0.9, 0.92]
Reversing the selection order relabels it "Best Score (f1_weighted)" — same numbers.
```

The label comes from `selectedJobs[0]`. A regression case showed `Best Score (rmse)` comparing
`[-2.5, 0.42]` and starring the R² job as "best". A **Basic Training** run silently carries
`scoring_metric=accuracy` — an internal default the user never chose and never sees — so mixing a
basic and a tuned run is the *default* experience.
Note: the sign handling is **correct** (`_tuning/engine.py:506-522` sign-corrects via `neg_*`
scorers); the bug is magnitude incomparability plus the label source.
Fix: group `best_score` by each job's own `scoring_metric`, one row per metric; never star across
metrics.

### F-37 🟠 HIGH · LIVE — SHAP/feature importance impossible for 6 of 11 classifier families
`_execution/engine/_artifacts.py:60-70` (reads only `feature_importances_`/`coef_`);
`skyulf-core/skyulf/modeling/_explainability/shap_explanation.py:37-78` (passes the **estimator
object** to `shap.Explainer`); message from `utils/artifactCoverage.ts:35-36,80-84`

**Status:** ✅ Fixed (0.7.9 wave) — `_build_explainer` in
`shap_explanation.py` now falls back to the estimator's `predict_proba`
(then `predict`) when the estimator is not callable, so all six families
(SVC-RBF, KNN, GaussianNB, MLP, Voting, Stacking) get SHAP; 3 new unit
tests, 25/25 pass. The Experiments page artifact-coverage text now says
*not supported for this model type* instead of blaming an older run.
Permutation feature importance for these families remains `None`
(follow-up).

```
random_forest, decision_tree, gradient_boosting, logistic_regression, ridge_classifier -> FI YES, shap ok
svc_rbf, knn, gaussian_nb, mlp, voting(soft), stacking                                 -> FI None, shap none
  TypeError: The passed model is not callable and cannot be analyzed directly...
  shap.Explainer(model.predict_proba, masker) -> WORKS, shape=(5, 3, 2)
```

All six are user-selectable. Their Explainability and Feature Importance tabs are permanently
blank, and the UI text blames *"an older run, or the trainer skipped it"* — sending users to
re-train something that can never work. SHAP is recoverable in roughly two lines.
Fix: fall back to `predict_proba` (then `predict`) when the estimator isn't callable; add
permutation importance; change the UI text to *"not supported for this model type"*.

### F-38 🟠 HIGH · LIVE — `f1_macro` conflated with binary/weighted F1
`ExperimentsPage/utils/jobMeta.ts:41`

**Status:** ✅ Fixed (Experiments batch 2) — `mapJobMetricToDropdown` now
maps only exact `accuracy`/`f1_weighted`/`f1` (plus non-macro/micro
precision/recall prefixes); `f1_macro` and other averaged or
threshold-independent variants fall back to the documented `f1_weighted`
default instead of being routed onto the binary positive-class F1 scan.
5 unit tests; red-green.

```
f1_macro    = 0.7957   <- what the job was tuned on
f1_weighted = 0.9232
f1(binary)  = 0.6364
TS computes = 0.6364 (or 0.9232 after normalizeThresholdMetric)
```

Up to **0.29 absolute** away from the metric the run actually optimised. The user compares runs on
a number that does not exist in the job.

### F-39 🟠 HIGH · LIVE — Pipeline diff duplicates every renamed-and-modified node
`experiments/PipelineDiffView.tsx:224`; `core/utils/graphDiff.ts:210-227` — `registerPair` stores
the same `NodeDiff` object under both `left.id` and `right.id`. Node-id drift between runs is the
**normal** case, so nearly every real diff double-lists every change, plus a React duplicate-key
warning that can mis-reconcile rows. Fix: dedupe by object identity before rendering.

**Status:** ✅ Fixed (Experiments batch 2) — new `uniqueNodeDiffs` helper
in `graphDiff.ts` dedupes by object identity; `PipelineDiffView` renders
`uniqueNodeDiffs(diff.nodes)` before filtering to modified nodes. Unit
test covers a renamed-and-modified node appearing once; red-green.

### F-40 … F-48

| ID | Sev | Live? | Finding | Location |
|---|---|---|---|---|
| F-40 | 🟡 | ✅ Fixed (batch 2) | `thresholds/save` validates nothing; garbage persists and is silently discarded at predict time. **Fix:** `save()` now validates metric ∈ supported set, non-empty classes, threshold keys exactly matching the stringified classes, finite values, and split ∈ {validation, test}; 7 rejection cases + a preview→save round-trip acceptance test + 1 HTTP-400 integration case; red-green. | `threshold_tuning_service.py:169-190`, `deployment/service.py:304-328` |
| F-41 | 🟡 | ✅ Fixed (batch 2) | "Show CV metrics" checkbox does not hide `best_score`. **Fix:** `metricKeys` is filtered through new `splitOfMetric`/`filterMetricKeysBySplitVisibility` helpers (`metricMeta.ts`), so `best_score` follows the CV checkbox; 7 unit tests; red-green. | `ExperimentsPage.tsx` |
| F-42 | 🟡 | ✅ Fixed (batch 2) | Task `'other'` always reports "unsupported" even when the artifact exists. **Fix:** `artifactCoverage.ts` precedence reordered to failed > not-terminal > has-artifact (`available`) > unsupported > not_computed, and `task === 'other'` no longer counts as unsupported; 3 unit tests; red-green. | `utils/artifactCoverage.ts` |
| F-43 | 🟡 | **LIVE in the published `skyulf-core` SDK**, latent in the app | Polars reference crosstab invents a `"nan"` segment — `is_not_null()` misses float NaN, inflating segment counts. Docstring wrongly claims pandas parity. Same root cause as §1. | `skyulf-core/skyulf/modeling/_evaluation/clustering.py` |
| F-44 | 🟡 | ✅ Fixed (batch 2) | `changeDescriptions` renders `"v: 5 → 5"` for a real int/str coercion, so users dismiss a genuine config change. Detection is correct; only the rendering is wrong. **Fix:** `describeValue` quotes strings, so the coercion now renders `5 → "5"`; unit test plus the PipelineDiffView swap test updated to the quoted rendering; red-green. | `graphDiff.ts:93-99` |
| F-45 | ⚪ | ✅ Fixed (T4 cleanup on `081`) | `stableStringify` collapsed `NaN` and `null` → false "unchanged". **Fix:** `stableStringify` emits a distinct `NaN` token; unit test covers a NaN↔null diff. | `graphDiff.ts` |
| F-46 | ⚪ | ✅ Fixed (T4 cleanup on `081`) | Numeric class labels sorted lexicographically when `y_proba` is absent. **Fix:** `classLabelComparator` sorts all-numeric labels by value; unit tests cover numeric and string labels. | experiments |
| F-47 | ⚪ | ✅ Fixed (T4 cleanup on `081`) | `shortRunId` 8-char collisions. **Fix:** widened to 10 chars. | `utils/jobMeta.ts` |
| F-48 | ⚪ | ✅ Fixed (T4 cleanup on `081`) | `StatefulEstimator.refit()` was dead code — zero production callers. **Fix:** method and its four test call sites removed. | `skyulf-core/skyulf/modeling/base.py` |

### Proven correct in this subsystem

- `EvaluationService` has **zero** pandas/polars/numpy references — genuinely engine-agnostic.
- The execution engine is pandas-only **by assertion**, not merely convention:
  `_node_runners.py:154-160` raises `TypeError` at the single data-entry point (probe confirmed it
  firing). This is what keeps F-31/F-32/F-43 latent rather than live.
- **Binary threshold preview ↔ inference parity is exact** — 0 disagreements on 400 real rows;
  1 of 999 grid points differs, and only at the exact `>=`/`>` tie. Preview and inference
  weighted-F1 matched to 6 dp.
- Multiclass Nelder–Mead tuning genuinely improves its objective (weighted-F1 0.3506 → 0.4094).
- `best_score` **sign handling is correct** — the "regression sign is flipped" hypothesis was
  explicitly disproved.
- `_refit_best_model` refits on **train only** (`_tuning/engine.py:355`); validation is used solely
  for hyperparameter selection via `PredefinedSplit`. No train-on-eval contamination.
- Clustering evaluation has full pandas/polars parity on well-formed data — identical silhouette,
  Calinski-Harabasz, Davies-Bouldin, centroids, profiles and crosstab. F-43 (float NaN) is the only
  divergence reproducible.
- `graphDiff` core matching yields **no false negatives**; F-39/F-44/F-45 are presentational.
- `pipelineDiffLayout` is cycle-safe; `findBestThreshold` handles degenerate inputs;
  `runSelection.ts` is correct; registry and threshold API shapes agree with their routers.

### Not audited in this subsystem

Model-registry HTTP endpoints (promote/unpromote traced but not exercised); SHAP chart rendering
math for the families where data *is* present; `ClassificationChartsForSplit` /
`PerClassConfusionMatrix` prop paths; `ExperimentsPage` effect-dependency exhaustiveness beyond
F-33; job-list polling against in-flight mutations.

**Unverified, needs a decision:** `promote_job` (`_execution/jobs.py:373`) accepts only
`status == "completed"`, while `JobStatus.SUCCEEDED = "succeeded"` is treated as terminal in four
other modules. No site was found that *writes* `"succeeded"`. This is either harmless dead enum
drift or a latent "cannot promote" bug; proving which needs a state-changing probe.
