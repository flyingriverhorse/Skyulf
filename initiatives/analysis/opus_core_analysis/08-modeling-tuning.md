# Skyulf Deep Audit (Opus) — Modeling: estimators, cross-validation & tuning

> Part of [`opus_core_analysis`](./README.md). Severity: 🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low. Finding IDs use the `OC-` prefix.

**Scope:** `skyulf-core/skyulf/modeling/` excluding `_evaluation/` and `_explainability/` (covered in [report 04](./04-evaluation-explainability.md)) — estimator wrappers, cross-validation, splitting, `_tuning/`, hyperparameter spaces, model persistence.

**Headline: the leakage audit is clean.** All seven fit/apply boundaries were traced and none leaks. Given how much of this library's value rests on that, it is the most important *negative* result in the audit.

---

## Findings

### OC-66
### 🟠 High — `CalibratedClassifierCV`'s user-selected base estimator is silently discarded during tuning

**Files:** `skyulf/modeling/classification.py:206-282` (resolution) vs
`skyulf/modeling/_tuning/engine.py:495-499` and `_tuning/params.py:53-77` (tuning path)

`CalibratedClassifierCalculator._resolve_base_estimator()` — the sole place that
turns the UI's `base_estimator` string (`"random_forest"`, `"svc"`,
`"gradient_boosting"`, …) into a real estimator under the `"estimator"` key — is
called **only from its own `fit()`**. Unlike `_BaseEnsembleCalculator`, the
calibrated calculator does **not** override `prepare_tuning_params`/`default_params`.

`TuningCalculator.tune()` builds the base estimator straight from
`model_calculator.default_params`, permanently frozen to the class default set in
`__init__`. So every tuning strategy — grid, random, halving, optuna — always
calibrates a plain `LogisticRegression`.

```text
default_params after prepare_tuning_params:
  {'estimator': LogisticRegression(max_iter=1000), 'method': 'sigmoid', 'cv': 5}
Tuned model type: <class 'sklearn.calibration.CalibratedClassifierCV'>
Calibrated base estimator actually used: LogisticRegression
Expected: RandomForestClassifier  (config base_estimator='random_forest')
```

**Impact:** A user who configures a Calibrated Classifier with a non-default base
estimator and then runs "Advanced" (tuned) training gets a model calibrating the
wrong base learner — worse accuracy and calibration, no error, no warning, and a
UI still showing the config they intended. This is the same silent-no-op pattern
as OC-13/OC-14/OC-15, but it originates **inside the backend**, not at the
frontend boundary.

**Fix:** Give `CalibratedClassifierCalculator` a `prepare_tuning_params` override
mirroring `_BaseEnsembleCalculator`, resolving `base_estimator` into
`default_params["estimator"]` before tuning starts; add `base_estimator` to a
`STRUCTURAL_TUNING_KEYS` tuple so a fixed single-candidate search space excludes it.

**Confidence:** 9/10

---

### OC-67
### 🟡 Medium — Tuning metrics `pr_auc`, `pr_auc_weighted`, `g_score` crash the entire search

**File:** `skyulf/modeling/_tuning/metrics.py:19-36`, `:127-146`

These three names are treated as valid classification tuning metrics throughout
the codebase — `INVALID_REGRESSION_METRICS` lists them as classification-only,
`refit.py`'s threshold fallback comment names them, and the frontend already
ships direction metadata and tooltips for all three in `metricMeta.ts` and
`format.ts`. But `METRIC_ALIAS_MAP` has **no entry** translating them to a real
scorer name (sklearn's PR-AUC scorer is `"average_precision"`; there is no
built-in `g_score` or weighted PR-AUC scorer at all). `resolve_scorer` passes the
raw name to `get_scorer`, which raises on the first candidate.

```text
metric=pr_auc:          ValueError: Hyperparameter tuning failed: All trials failed.
                        First trial error: 'pr_auc' is not a valid scoring value.
metric=g_score:         ValueError: ... 'g_score' is not a valid scoring value.
metric=pr_auc_weighted: ValueError: ... 'pr_auc_weighted' is not a valid scoring value.
```

**Impact:** Not reachable from the shipped dropdown today (which offers only
`accuracy/f1/roc_auc/mse/rmse/mae/r2`), but the frontend already carries the
metadata for these three, and **nothing validates `metric` against an allow-list
before it reaches the tuner**. The moment a widened dropdown, direct API/SDK call,
or advanced-JSON editor sets one, every tuning run fails with a confusing "all
trials failed" instead of the real cause.

**Fix:** Add `"pr_auc": "average_precision"`, a `make_scorer`-based weighted
variant, and a `make_scorer(geometric_mean_score)` entry for `g_score` — mirroring
how `_evaluation/metrics.py` already computes all three for evaluation reports.
Also validate `metric` against the resolvable set up front.

**Confidence:** 8/10

---

### OC-68
### 🟠 High — Legacy short model-name aliases are incomplete and task-unaware

**File:** `backend/ml_pipeline/_execution/engine/_node_runners.py:1157-1176`

> **Escalated and moved.** This was first filed here as ⚪ Low on the assumption an
> unmapped short name would merely crash. The backend audit found a case that does
> *not* crash and silently trains the wrong estimator family instead. Full detail
> now lives in [report 09 → OC-68](./09-backend.md#oc-68); the original
> modeling-side observation is kept below for context.

`_get_model_components`'s `alias_map` covers only 6 legacy spellings
(`logisticregression`, `randomforestclassifier`, `random_forest`,
`ridgeregression`, `ridge`, `randomforestregressor`). Short names like `xgboost`,
`lightgbm`, `knn`, `adaboost`, `decision_tree`, `extra_trees`,
`gradient_boosting`, `hist_gradient_boosting`, `elasticnet`, `lasso` are **not**
aliased.

Tracing the live data flow shows the *crash* case is **not currently reachable**:
the Classification/Regression nodes populate `model_type` exclusively from
`registryApi.getAllNodes()` (real registry ids), and `pipelineConverter.ts`
forwards `algorithm: node.data.model_type` unchanged. A short name would raise
`ValueError: Unknown algorithm` — loud, not silent.

> This **corrects** the assumption recorded earlier in this audit that the
> frontend emits short model names. It does not: it emits full registry ids. The
> `alias_map` is legacy residue. The genuinely dangerous case —
> `"random_forest"` resolving to the *classifier* even for a `RegressionNode`,
> with no `problem_type` cross-check — is documented in
> [report 09](./09-backend.md#oc-68).

**Fix:** Delete the unused entries, or complete them to match the short-name set
the ensemble resolver already supports (`_BASE_KEY_TO_REGISTRY_CLF/REG`), and add
a `problem_type` vs `step_type` assertion after resolution.

**Confidence:** 8/10

---

<a id="leakage-audit-table--all-clean"></a>
## Leakage audit table — all clean

| Step | file:line | Fitted on | Applied to | Leak? |
|---|---|---|---|---|
| K-Fold / Stratified / TimeSeries / Shuffle CV | `cross_validation.py:154-193` | each fold's train rows | held-out fold rows | ✅ No — refit per fold |
| Time-series sort / column drop | `cross_validation.py:_sort_by_time` | sorts by time col, drops from `X` | all rows pre-split | ✅ No — sorting isn't fitting; drop applied on both engines |
| Per-fold preprocessing (`FoldPreprocessor`) | `fold_preprocessing.py`, `_tuning/grid_random.py:132-135`, `_tuning/fold_pipeline.py:120-126` | `fit_transform` on fold-train only | `transform`-only on fold-val | ✅ No — re-fit every fold/candidate; halving/optuna get it via `FoldAwareModelStep` inside the searcher's own folds |
| Holdout tuning (`validation_data`) | `_tuning/splitters.py` | `PredefinedSplit` masks train −1 / val 0 | val rows scored, never trained on | ✅ No |
| Decision-threshold tuning | `_tuning/refit.py:139-217` | threshold grid on `validation_data` only | never touches `dataset.test` | ✅ No |
| Final best-model refit | `_tuning/engine.py:426-438` | full train split | — | ✅ No |
| Nested CV inner loop | `cross_validation.py:_run_inner_cv` | inner-fold train | inner-fold val | ✅ No — diagnostic only |

---

## Hyperparameter space validity

**389 (model, param, extreme-value) combinations** were tested across 30
calculators by constructing `model_class(**{param: value})` and, where feasible,
running a full `fit()` at each declared min/max/select-option.

**Result: every declared range is valid.** Only two entries need annotation:

| Estimator | Param | Note |
|---|---|---|
| `LogisticRegression` | `penalty=l1/elasticnet` with default `solver=lbfgs` | Invalid *combination*, but pre-validated by `_validate_solver_penalty` with a clear, actionable `ValueError` before reaching sklearn — working as designed |
| `CalibratedClassifierCV` | `base_estimator` | Not a real ctor param after the sklearn 1.4 `base_estimator`→`estimator` rename; correctly translated in the non-tuned `fit()` path only — see **OC-66** |

Notably, XGBoost/LightGBM params reach the estimator via `**kwargs` (their sklearn
wrappers expose only `objective, **kwargs` to `inspect.signature`, which makes a
naive constructor-signature check produce false positives).
`_filter_supported_params`/`instantiate_model` correctly special-case
`accepts_kwargs`. Ensemble base/final estimator keys match `ensemble.py`'s
`BASE_ESTIMATORS_CLF/REG` factories 1:1.

---

## Prior-finding re-verification

| Prior ID | Claim | Still fixed? | Evidence |
|---|---|---|---|
| F-02 | `random_state` silently ignored during tuning | ✅ Fixed | Two identical runs with `random_state=123` gave byte-identical `best_params`, `best_score` and predictions; seed 999 gave different `best_params` |
| F-04 | No class-imbalance handling | ✅ Fixed | `class_weight` present on RF/ET/LightGBM/XGBoost classifier sets; `_compute_sample_weight_for_fit` raises clearly rather than silently no-op'ing |
| F-06 | All-folds-failed still returns a model | ✅ Fixed | Now raises `ValueError: Hyperparameter tuning failed: All trials failed…` with the real underlying error — verified via the OC-67 repro |
| F-13 | Threshold tuning built but never wired | ✅ Fixed | `TuningCalculator.fit()` calls `tune_decision_thresholds(...)` using `validation_data` only |
| F-24 | Only first fold error kept | ✅ As documented | `fold_errors` reports the first plus "(N more fold failures suppressed)" |

---

## What I checked and found sound

- **CV stratification**: `StratifiedKFold` is used automatically for
  classification in both plain CV and tuning; regression correctly falls back to
  plain `KFold`.
- **Determinism**: `random_state` is threaded end-to-end.
  `DEFAULT_RANDOM_STATE` is auto-injected via `_inject_default_seed` whenever the
  wrapped estimator's constructor accepts it; an explicit caller value (including
  `None`) always wins; and the splitter-shuffling seed (`cv_random_state`) is
  independent of and correctly threaded alongside the model seed.
- **Time-series CV**: `_sort_by_time` handles both engines, drops the sort column
  from features, and is invoked identically from `perform_cross_validation` and
  `TuningCalculator.fit`.
- **Per-fold preprocessing**: correctly wired for grid/random (direct loop),
  halving/optuna (via `FoldAwareModelStep`, deep-copying both preprocessor and
  estimator per candidate/fold so parallel searchers never share fitted state),
  and holdout tuning.
- **Fail-fast validation**: `_validate_solver_penalty` and `ensemble.py`'s
  mirrored check both raise an actionable message rather than letting an
  incompatible combo reach sklearn's opaque error.
- Ensemble base-estimator keys verified 1:1 against `EnsembleSettings.tsx` and
  `pipelineConverter.ts`'s `ENSEMBLE_BASE_KEY_BY_MODEL_TYPE`.

---

## Improvement opportunities (not defects)

- `DecisionTreeRegressor(criterion="poisson")` and `MultinomialNB`/`BernoulliNB`
  correctly reject negative target/feature data, but with sklearn's raw message —
  unlike the Logistic Regression solver/penalty check, skyulf adds no pre-flight
  validation or UI hint.
- `SVM_PARAMS` has no `random_state` field, unlike every other seeded family. It
  is still seeded via `_inject_default_seed`, but the seed isn't user-visible.
- `_boosting_progress.py:_MAXIMIZE_METRICS` contains a harmless typo
  `"precession"` (dead duplicate entry; `"precision"` is listed correctly).
- `_tuning/engine.py`'s `tune()` (C901 = 16) is dense but correct; all branches
  were exercised by the repros. Still a good candidate for leaf-module extraction.
