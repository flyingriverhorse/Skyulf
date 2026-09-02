# 14 — Hyperparameters (`modeling/hyperparameters/`)

**Scope:** all 11 modules of `skyulf/modeling/hyperparameters/` (2,194 lines),
cross-checked against the real estimator constructors.

**Method — this was machine-checked, not eyeballed.** For all **38** models in
`MODEL_HYPERPARAMETERS` I resolved the node's `model_class` via
`NodeRegistry.get_calculator(key)().model_class`, then diffed every declared
`HyperparameterField.name` (and every search-space key) against
`model_class().get_params()`. Only 2 models showed a mismatch; both were then
verified by execution, and **one of the two turned out to be a false positive**
(see "Checked and found sound").

New ids continue from the master at **OC-100**.

| ID | Severity | Issue | Location |
| --- | --- | --- | --- |
| [OC-101](#oc-101) | 🟡 Medium | `calibrated_classifier` declares a `random_state` field its estimator never accepts | `hyperparameters/_calibration.py` |
| [OC-102](#oc-102) | ⚪ Low | Five tunable models return an empty search space from the live defaults endpoint | `hyperparameters/_registry.py` |
| [OC-100](#oc-100) | ❌ **Retracted** | ~~Search-space dicts are dead code~~ — **false positive, withdrawn**; the dicts are live via `get_default_search_space()` → HTTP → frontend | — |

Plus one **merge into an existing finding** (not re-filed) — see
[OC-66 root cause](#oc-66-root-cause-confirmed-merged-not-re-filed).

---

<a id="oc-100"></a>
<a id="oc-100"></a>
### OC-100
### ❌ RETRACTED — false positive (my own, caught in cross-check)

> **This finding was wrong and has been withdrawn.** It is kept here rather than
> deleted so the error is auditable.

**Original claim:** `DEFAULT_SEARCH_SPACES` / `GRID_SEARCH_SPACES`
(`hyperparameters/_registry.py:96-507`) are dead configuration consumed by
nothing.

**Why it was wrong.** My search looked for the *dict names*:

```bash
grep -rn "DEFAULT_SEARCH_SPACES\|GRID_SEARCH_SPACES" --include=*.py .
```

That query is structurally incapable of finding the real consumer, which reaches
the dicts through **one hop of indirection** and then leaves Python entirely.
The full chain, each hop verified by me:

| # | Hop | Evidence |
| --- | --- | --- |
| 1 | Wrapper reads both dicts | `_registry.py:503` `get_default_search_space(model_key, strategy)` |
| 2 | Exposed as HTTP route | `backend/ml_pipeline/_internal/_routers/meta.py:212-215` |
| 3 | Router actually mounted | `backend/ml_pipeline/api.py:45,62` (`include_router(_meta_router)`) |
| 4 | Frontend calls the route | `frontend/ml-canvas/src/core/api/jobs.ts:221` |
| 5 | UI consumes the result | `modules/nodes/modeling/TrainingSettings.tsx:298` |
| 6 | Covered by a frontend test | `core/api/jobs.test.ts:186-191` |

Executed proof that **both** dicts are load-bearing (the grid variant is genuinely
distinct, not a copy):

```
get_default_search_space('random_forest_classifier', 'random')
  n_estimators=[50,100,200,500]  max_depth=[None,5,10]  min_samples_split=[2,5,10,20]
get_default_search_space('random_forest_classifier', 'grid')
  n_estimators=[100,200,500]     max_depth=[5,10,20]    min_samples_split=[2,10]
random != grid -> True
```

The dicts are read, served over the network, tested, and rendered in the
advanced-tuning UI. They are simply not consumed *inside* `skyulf-core`, which is
where my search stopped.

**Process note.** My first re-test also "confirmed" the finding by returning `{}`
— because I passed `'random_forest'`, which is not a key (the dicts are keyed
`random_forest_classifier`). That is the same class of calling-convention trap
catalogued in [`00-validation-log.md`](./00-validation-log.md#calling-convention-traps):
a wrong-key probe produces a clean, plausible, *false* negative. See
[the log's correction entry](./00-validation-log.md#oc-100) for the full account.

---

<a id="oc-102"></a>
### OC-102
### ⚪ Low — Five tunable models return an empty search space from the live defaults endpoint

Surfacing this was the one durable part of the retracted OC-100 — and confirming
the endpoint is live *raises* its relevance, because the gap is now user-visible
rather than theoretical.

Five models declare `tunable=True` fields but appear in **neither** search-space
dict, so the mounted `/pipeline/hyperparameters/{model_type}/defaults` route
returns `{}` and the advanced-tuning UI has nothing to populate:

```
birch             random={}  grid={}
gaussian_mixture  random={}  grid={}
kmeans            random={}  grid={}
minibatch_kmeans  random={}  grid={}
voting_regressor  random={}  grid={}
random_forest_classifier -> 6 params (control: works)
```

| model | tunable fields declared |
| --- | --- |
| `birch` | `n_clusters`, `threshold`, `branching_factor` |
| `gaussian_mixture` | `n_components`, `covariance_type` |
| `kmeans` | `n_clusters`, `n_init` |
| `minibatch_kmeans` | `n_clusters`, `batch_size`, `n_init` |
| `voting_regressor` | `base_estimators`, `n_jobs` |

All five are clustering/ensemble models where CV-based tuning is genuinely
awkward, so the omission may well be deliberate — which is why this is ⚪ Low.
But nothing in the code or the API says so, and the UI cannot distinguish
"deliberately not tunable" from "defaults missing". **Fix:** add entries for the
five, or have the endpoint return an explicit `not_tunable` marker.

---

<a id="oc-101"></a>
### OC-101
### 🟡 Medium — `calibrated_classifier` declares a `random_state` field its estimator never accepts

**File:** `skyulf-core/skyulf/modeling/hyperparameters/_calibration.py`

`CALIBRATED_CLASSIFIER_PARAMS` declares four fields:

```
[('base_estimator', tunable=True), ('method', True), ('cv', True), ('random_state', False)]
```

but sklearn 1.8's `CalibratedClassifierCV` accepts only:

```
['cv', 'ensemble', 'estimator', 'method', 'n_jobs']
```

`random_state` is not among them and never has been — calibration inherits
randomness from its inner estimator and its CV splitter. Because
`CalibratedClassifierCV` does not take `**kwargs`,
`_filter_supported_params` (`sklearn_wrapper.py:193-212`) strips it and emits
only a `logger.warning`. The field is still rendered as a configurable
hyperparameter, so a user who sets it gets no error and no effect.

Note the inverse consequence too: `_ensure_random_state`
(`sklearn_wrapper.py:181-189`) deliberately *skips* seeding any estimator whose
constructor lacks `random_state`, so calibrated classifiers are **unseeded** —
the declared field creates a false impression that the run is reproducible.

**A second, independent reason the field can never work.** Even if
`_filter_supported_params` let it through, the seed still could not reach
anything, because the base-estimator factories are **zero-argument lambdas that
hardcode the seed** (`modeling/classification.py:217-226`):

```python
BASE_ESTIMATORS: ClassVar[dict[str, Callable[[], BaseEstimator]]] = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest":       lambda: RandomForestClassifier(n_estimators=100,
                                                          random_state=DEFAULT_RANDOM_STATE),
    "gradient_boosting":   lambda: GradientBoostingClassifier(random_state=DEFAULT_RANDOM_STATE),
    "decision_tree":       lambda: DecisionTreeClassifier(random_state=DEFAULT_RANDOM_STATE),
    "gaussian_nb":         GaussianNB,
    "svc":                 lambda: SVC(probability=True, random_state=DEFAULT_RANDOM_STATE),
}
```

Every factory takes no parameters, so the user's seed has no path in; four of the
six pin `DEFAULT_RANDOM_STATE` (42) unconditionally. Executed:

```
base_estimator='random_forest', user random_state=1     -> inner RandomForest.random_state = 42
base_estimator='random_forest', user random_state=99999 -> inner RandomForest.random_state = 42
predict_proba identical despite different requested seeds: True
```

And the CV path offers nothing to seed either — `CalibratedClassifierCV` with an
integer `cv` builds `StratifiedKFold(n_splits=5)` with `shuffle=False`, so there
is no shuffling randomness for a seed to control.

**Fix:** drop `random_state` from `CALIBRATED_CLASSIFIER_PARAMS`. If
reproducibility is genuinely wanted, the factories must accept a seed
(`lambda seed: RandomForestClassifier(..., random_state=seed)`) *and* a seeded CV
splitter must be passed to `cv` — the field alone cannot deliver it.

---

<a id="oc-66-root-cause-confirmed-merged-not-re-filed"></a>
### OC-66 root cause confirmed — *merged, not re-filed*

> **Merged, not re-filed:** this is the mechanism behind
> [OC-66](./08-modeling-tuning.md#oc-66) ("`CalibratedClassifierCV`'s
> user-selected base estimator is silently discarded during tuning"), which is
> already filed at 🟠 High. Recorded here because the hyperparameter
> cross-check independently pinpointed the exact cause.

The registry declares the field as **`base_estimator`**. sklearn renamed this
parameter to `estimator` in 1.2 and **removed `base_estimator` entirely in
1.4**; this repo runs sklearn 1.8. Verified:

```
CalibratedClassifierCV ctor params: ['cv', 'ensemble', 'estimator', 'method', 'n_jobs']
 -> 'base_estimator' present? False
 -> 'estimator' present?      True

_filter_supported_params({'base_estimator': 'random_forest', 'method': 'sigmoid'})
  WARNING: Dropped parameters not supported by CalibratedClassifierCV: {'base_estimator'}
  returns: {'method': 'sigmoid'}
```

So the user's chosen base estimator is dropped before construction and
`CalibratedClassifierCV` silently falls back to its own default. **The fix is a
one-word rename** in `_calibration.py`: `base_estimator` → `estimator` (plus the
matching key wherever the UI/pipeline sends it). That is a much smaller fix than
OC-66's write-up implies, so fix it there first and re-test.

---

## Checked and found sound

**A false positive I caught and did not file.** The cross-check flagged
`xgboost_classifier` as declaring `class_weight`, which `XGBClassifier` does not
accept as a named constructor parameter. This looked like a silent no-op on
imbalanced data — a serious bug. It is not. `XGBClassifier` takes `**kwargs`, so
`_filter_supported_params` skips filtering (`sklearn_wrapper.py:203-204`), and
the wrapper has an explicit compensating path:
`_constructor_accepts_class_weight()` returns `False` for XGBoost, so
`_compute_sample_weight_for_fit` translates `class_weight` into a per-sample
weight array. End-to-end test on 285/15 imbalanced data:

```
without class_weight -> positives predicted: 1
with    class_weight -> positives predicted: 25
```

The parameter demonstrably works. This is a good example of why the
declared-vs-accepted diff alone is not sufficient evidence.

**Everything else matched.** For the other 36 models, every declared
hyperparameter name and every search-space key is accepted by the corresponding
estimator constructor — no typos, no stale sklearn names, no drift. Given that
this registry hand-maintains ~2,200 lines mirroring a fast-moving sklearn API,
finding only one genuinely stale name (`base_estimator`) is a good result.

The four ensemble models (`stacking_classifier`, `stacking_regressor`,
`voting_classifier`, `voting_regressor`) could not be constructor-checked because
their `__init__` requires the `estimators` argument; their fields were reviewed
by hand against the sklearn signatures instead and are correct.

## Improvements

- **Make the cross-check a test.** The diff above is ~15 lines and runs in under
  a second. As a `pytest` case it would have caught `base_estimator` the moment
  the project moved to sklearn ≥1.4, and will catch the next rename for free.
  This is the single highest-value change in this domain.
- **Mark fields that are deliberately non-tunable.** `voting_regressor`'s
  `base_estimators` is `tunable=True` but is a structural choice, not a
  numeric knob; the flag conflates "user-editable" with "CV-searchable".
