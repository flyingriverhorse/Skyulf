# Dual-Engine (Polars / Pandas) Correctness Audit — Findings

**Date:** 2026-08-11
**Branch:** `078`
**Baseline:** `master` @ `acfd9b6d`
**Versions at audit time:** `skyulf-core` 0.5.8 · backend 0.7.8 · frontend 0.7.8
**Method:** 4 parallel Opus-5 audit agents (engine-parity, pandas-purity, leakage, inference/experiments)
plus an adversarial rubber-duck pass. All findings below were **reproduced with executable probes**,
not inferred from reading code. Probes ran against throwaway worktrees; repo source untouched.

---

## 0. Executive summary

**Is it bad?** No — the architecture is sound and the pandas path is solid. But the Polars path
carries six real correctness bugs, and there are three **engine-independent** inference bugs that
affect pandas users too. Nothing here is unfixable; most are one- to ten-line fixes. The honest
position today is: *Polars is capable end-to-end but not yet at parity, and inference has a
column-ordering bug that must be fixed before anyone deploys.*

| Layer | Verdict |
|---|---|
| **Engine / data path** | ✅ Sound. 0 nodes are Polars-incapable. 0 nodes silently downgrade Polars→pandas. Pandas users stay 100% pandas (100/100 nodes verified). |
| **Preprocessing nodes** | ⚠️ 54/61 transformers match across engines. **7 diverge**, 1 hard-breaks Polars training. |
| **Modeling / numpy handoff** | ✅ **Provably correct.** Column order and values identical across engines; predictions bit-identical; CV/tuning/evaluation JSON identical (one 1-ULP `log_loss` float-summation difference). 37/38 models match. |
| **Persistence** | ✅ joblib round-trip clean. Artifacts are plain `list`/`dict`/`str`/`bool` — no engine-specific objects. 40/41 transformers replay safely cross-engine (HashEncoder is the exception). |
| **Inference / deployment** | 🔴 **3 CRITICAL bugs, 2 of them engine-independent.** This is the weakest layer. |
| **Monitoring / drift** | ⚠️ One live bug (NaN → silently reports "no drift"). |
| **Experiments** | ➖ No experiment tracker exists (no MLflow/W&B). `model_registry` is DB metadata only — no dataframes, so no engine risk. Nothing to fix; nothing to claim either. |
| **Leakage** | ⚠️ The per-node fit/apply discipline is **genuinely solid** — the core claim is true. But enforcement covers only one graph pattern in one execution path, WOE lacks cross-fitting, and CV does not re-fit preprocessing per fold. |

**Answering the specific questions asked:**

- *Does modeling work with Polars?* Yes — and this part is proven, not assumed. The
  Polars→numpy→sklearn handoff produces bit-identical predictions to the pandas path.
- *Does inference work?* Only if your pipeline has no column-adding transformer, and only if
  callers happen to send JSON keys in training order. Both are unacceptable — see F-02, F-03.
- *Do experiments work?* There is no experiment-tracking subsystem to break.
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
| F-16 🟡 | 10 stateful nodes bypass the pre-split leakage gate (`MissingIndicator`, `DropMissingColumns`, Over/Undersampling, `HashEncoder`, `Deduplicate`, …). The comment wrongly calls the first two "stateless"; both provably learn columns from data. | `_leakage_validation.py:30-38` |
| F-17 🟡 | Pipelines with **no splitter node** get zero leakage protection — the gate is keyed on finding a split. | `leakage.py:53-54` |
| F-18 🟡 | Row-dropping steps (`Deduplicate`, `DropMissingRows`) still execute at inference; 4 requested rows → 2 predictions, and `PredictionResponse.predictions: list[Any]` carries no row keys, so the caller cannot tell which inputs vanished. | `preprocessing/pipeline.py:64-70` |
| F-19 🟡 | `DropMissingColumns` `null_count()` misses NaN → different columns dropped per engine. | `drop_and_missing/drop_columns.py:44` |
| F-20 🟡 | `DatasetProfile` on Polars reports `missing=0` while `mean=NaN` for NaN-bearing columns — self-contradictory profile. | `profiling/` |
| F-21 🟡 | `lag.py` `drop_nulls()` vs pandas `dropna()` → different row counts. | `time_series/lag.py:54` |
| F-22 🟡 | `DropMissingColumns` registers param `threshold` but reads `missing_threshold` — the UI value is ignored. | `drop_columns.py:91` |
| F-23 🟡 | `DummyEncoder` null rows: `0/0` int (pandas) vs `NaN/NaN` float (polars) — dtype divergence distinct from F-01. | `encoding/dummy.py` |
| F-24 🟡 | `most_frequent` on an all-unique column: pandas silently no-ops, polars raises. | `imputation/simple.py:66` |
| F-25 🟡 | **No engine identity recorded in the deployment bundle.** Nothing detects "trained on Polars, serving on pandas." Adding `"train_engine": "polars"\|"pandas"` and warning on mismatch would have caught F-11, F-12, F-30, F-31 at deploy time. | deployment bundle |
| F-26 🟡 | Polars is a **hard dependency** (`pipeline.py:12`; `setup.py:27` `install_requires`), so `polars_engine.py`'s `HAS_POLARS` fallback is unreachable dead code, tested only via monkeypatch. It advertises an optionality that has never worked. Matches packaging, so not a broken promise — but delete the dead branch or make it real. | `polars_engine.py` |

### Tier 4 — LOW

| ID | Finding | Location |
|---|---|---|
| F-27 ⚪ | Mixed-engine row filters raise `AttributeError` instead of a clear `TypeError`. | `drop_rows.py:69`, `outliers/_common.py:25` |
| F-28 ⚪ | WOE artifact null key is `None` on one engine, `nan` on the other. | `woe.py:167-186` |
| F-29 ⚪ | `SimpleImputer` median dtype: `Int64` vs `Float64`. | `imputation/simple.py` |
| F-30 ⚪ | `_pretty_dtype` is pandas-only → Polars deployments show `unknown` for Date, Duration, Categorical. | `service.py:613-638` |
| F-31 ⚪ | `_normalize_train_frame` returns `None` for Polars (drift reference + SHAP frame). **Not reachable** — backend catalog is pandas-only and `7485ade6` added a guard. Latent only. | `_artifacts.py:154-178` |
| F-32 ⚪ | `_feature_names_for_importance` pandas-only → `feature_importances = None` on Polars. Same reachability caveat as F-31. | `_artifacts.py:38-58` |

---

## 3. What is proven correct

Stated explicitly because it is as important as the bug list, and because these were *measured*:

- **Polars runs end-to-end.** All 99 registered nodes accept a raw `pl.DataFrame`. **Zero** nodes
  are Polars-incapable. **Zero** nodes silently downgrade Polars → pandas.
- **Pandas users stay on pandas.** 100/100 nodes verified — no silent Polars conversion anywhere.
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
mirror local). There is **no batch-prediction path** and **no experiment tracker** in the codebase
— nothing to audit, and nothing to claim.

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
| **T1** | F-01, F-02, F-03 | core **0.5.9** · backend/frontend **0.7.9** | **Patch — ship first, alone.** F-02 and F-03 are engine-independent and block real deployments. |
| **T2** | F-04 … F-14 (excl. F-15) | core **0.6.0** · backend/frontend **0.8.0** | **Minor.** Behaviour changes: rows previously dropped are now kept (F-06), imputers that previously no-opped now impute (F-04), HashEncoder buckets change (F-11 — artifacts fitted before this release will not reproduce, needs a release note). |
| **T3** | F-16 … F-26 + leakage enforcement | core **0.6.1** · backend/frontend **0.8.1** | Patch/minor. F-22 changes a param name users may rely on. |
| **T4** | F-27 … F-32 + dead-code cleanup (F-26 fallback) | core **0.6.2** · backend/frontend **0.8.2** | Patch. Cosmetic and latent-only. |
| **T5** | F-15 per-fold refit | core **0.7.0** | **Minor/major — separate initiative.** Changes reported scores. Design note first. |

**Cross-cutting, do in T1:** add engine-parity tests that use **float NaN** (not only nulls) and
**wrapped** frames. Without this the suite will keep passing while broken. Every fix below must be
verified red-green — write the failing test first, confirm it fails, then fix.

**Frontend sync check required for:** F-22 (`DropMissingColumns` param rename) and F-03 (the
deployment input-schema drives the UI form). Per repo policy, backend param/enum changes must be
mirrored in `frontend/ml-canvas/src/modules/nodes/`.

**Docs to update alongside the code:**
- `docs/examples/leakage_proof.md:459` — scope the "leakage-free by design" conclusion (see the
  companion enforcement plan).
- `docs/index.md:99`, `docs/user_guide/validation_vs_sklearn.md:290` — same.
- `changelog/0.7.x.md` and `docs/` engine-parity notes — record the T2 behaviour changes,
  especially the HashEncoder artifact break.
- Any doc asserting Polars/pandas parity should not make that claim until T2 ships.
