# 00 — Validation log (independently re-verified findings)

Every row below was **re-tested by me directly against the repo**, not accepted
from a sub-agent's report. Each entry lists the command-level evidence actually
observed. Findings that turned out to be wrong or overstated are listed first —
**do not spend fix effort on those until you read the correction**.

Environment for all runs:

```
cd /Users/BH7043/Skyulf
PYTHONPATH=skyulf-core .venv/bin/python <script>
```

(`PYTHONPATH` is required because of OC-02 — the editable install is dangling.)

<a id="calling-convention-traps"></a>
Calling conventions that probes must respect (getting these wrong produces
false "not reproduced" results — it happened to me **four** times):

| Thing | Correct form |
| --- | --- |
| Calculator fit | `calc.fit(df, config)` — 2 args; `@fit_method` adapts to `(self, X, y, config)` |
| Supervised fit (`y`) | pass a **tuple**: `calc.fit((X, y), config)` |
| Model calculator fit | `calc.fit(X, y, config)` — **3** args, unlike the above |
| Applier | `appl.apply(df, params)` |
| `Casting` | calculator takes `column_types` / `target_type`+`columns`; the **applier** takes `type_map` |
| `FeatureMath` ops | dispatch key is `operation_type`, columns are `input_columns` / `secondary_columns` |
| `Split` | config key is `stratify` (bool), not `stratify_col` |
| Search-space keys | fully qualified — `random_forest_classifier`, **not** `random_forest` |
| Node metadata | `NodeRegistry.get_calculator(id).__node_meta__.params` |

> **The deeper lesson.** Every one of these traps fails *silently and
> plausibly* — a wrong key returns `{}`, a wrong arity raises nothing, and the
> probe looks like it exonerated the code. A clean negative result from a probe
> is only trustworthy once the probe has been shown to produce a positive on a
> known-good control. [OC-100](#oc-100) below is what happens when that control
> is skipped.

---

## Corrections — findings that are wrong or overstated as filed

<a id="oc-100"></a>
### OC-100 — ❌ RETRACTED, false positive (**my own finding, not an agent's**)

**Claim:** `DEFAULT_SEARCH_SPACES` / `GRID_SEARCH_SPACES` are dead code consumed
by nothing.

**Verdict: wrong. Withdrawn entirely.** The dicts are live, served over HTTP, and
rendered by the frontend.

**How I got it wrong — two compounding errors.**

*Error 1 — a search that could not succeed.* I grepped for the **dict names**:

```bash
grep -rn "DEFAULT_SEARCH_SPACES\|GRID_SEARCH_SPACES" --include=*.py .
```

The real consumer reaches them through a wrapper, `get_default_search_space()`
(`_registry.py:503`), and the ultimate consumer is a *TypeScript* file calling an
*HTTP route*. No `.py` grep for the dict names could ever have found it. I also
scoped the search to `skyulf-core` when the consumers live in `backend/` and
`frontend/`. Verified chain: `_registry.py:503` → `meta.py:212-215` →
`api.py:45,62` (mounted) → `jobs.ts:221` → `TrainingSettings.tsx:298`, with a
frontend test at `jobs.test.ts:186-191`.

*Error 2 — a "confirming" probe that was itself broken.* My first re-test called
`get_default_search_space('random_forest')` and got `{}`, which I read as
confirmation. But `'random_forest'` is **not a key** — the dicts are keyed
`random_forest_classifier`. With the correct key both dicts return real,
*different* content:

```
random -> n_estimators=[50,100,200,500] max_depth=[None,5,10]  min_samples_split=[2,5,10,20]
grid   -> n_estimators=[100,200,500]    max_depth=[5,10,20]    min_samples_split=[2,10]
random != grid -> True   (both dicts are load-bearing)
```

I had no control case, so a broken probe silently agreed with my hypothesis.

**What survives.** The observation that five tunable models are absent from both
dicts is real, and confirming the endpoint is live *raises* its impact — re-filed
as [OC-102](./14-hyperparameters.md#oc-102) (⚪ Low).

**Why this entry matters more than the finding.** I filed OC-100 during the phase
where I had stopped trusting agent output and was doing the work myself — and I
committed exactly the sin I had flagged agents for: asserting a negative from a
search whose scope could not have found the positive. It was caught by
cross-checking against an agent I had already written off. Neither source is
reliable alone; the cross-check is what worked.

<a id="oc-01"></a>
### OC-01 — ⚠️ PARTLY WRONG (claim "always reports a stale, wrong version")

Not "always". There are **two** `skyulf-core` distributions visible, and which
one wins is purely `sys.path` order:

```
name= skyulf-core ver= 0.8.8 at= skyulf-core/skyulf_core.egg-info
name= skyulf-core ver= 0.5.8 at= .venv/lib/python3.12/site-packages/skyulf_core-0.5.8.dist-info
resolved version() -> 0.8.8
```

`setup.py` declares `version="0.8.8"`. In the normal repo-dev configuration
(repo first on path) `__version__` resolves **correctly** to `0.8.8`. The real
defect is **duplicate-distribution shadowing**: any context where the stale
`0.5.8` dist-info is found first silently reports a version three minor
releases behind.

**Restate as:** *`skyulf.__version__` is ambiguous — a stale `0.5.8` dist-info
shadows the real `0.8.8`, so the reported version is `sys.path`-order
dependent.* Keep severity 🟠 High (a wrong version in artifact provenance is
still serious), but the fix is "remove the stale dist-info / reinstall
cleanly", not "change the version-reading code".

<a id="oc-46"></a>
### OC-46 — ⚠️ PARTLY WRONG (claim "break strict JSON")

The primary EDA ship path does **not** emit invalid JSON:

```
backend/eda/router.py:237,251 →  orjson.dumps(report.to_dict())

orjson  -> b'{"mean":null,"std":null}'      # non-finite coerced to null, VALID
stdlib  -> {"mean": NaN, "std": Infinity}   # INVALID
```

Pydantic does accept the non-finite values (`NumericStats(mean=nan)` is
constructed without error), and `json.dumps(model_dump())` would indeed be
invalid — but `orjson` is what actually serialises the EDA profile, and
`model_dump_json()` also yields `null`. So the "breaks strict JSON" outcome
only applies to stdlib-`json` paths such as the monitoring persistence at
`backend/monitoring/router.py:449,569` (no `default_response_class`/
`ORJSONResponse` is configured, so those use FastAPI's stdlib default).

**Restate as:** *non-finite stats reach public payloads and are silently
coerced to `null` by orjson (a real `NaN`-vs-missing ambiguity); only the
stdlib-json monitoring paths can emit literally invalid JSON.*
**Downgrade 🟠 High → 🟡 Medium.**

---

## Upgrades — findings that are worse than filed

<a id="oc-12"></a>
### OC-12 — 🔴 CONFIRMED, and worse

```
in rows=4 -> X_out=3  y_out=4
X_out.index: [0, 1, 2]   y_out.index: [0, 0, 1, 2]
y_out values: [10, 20, 30, 40]
```

`drop_rows.py:67` does `y.loc[X_clean.index]`. With a duplicate index this not
only **desyncs the lengths** (3 vs 4) — the surviving `y` still contains `20`,
which is the label of the row that was *dropped*. So labels are both misaligned
**and** wrong, silently. Confirms the Critical rating.

<a id="oc-18"></a>
### OC-18 — CONFIRMED, and worse

```
OneHotEncoder on DataFrame({"c": ["a","b"], "c_a": [9,9]})
output columns: ['c_a', 'c_a', 'c_b']
```

Not merely "can collide" — it emits a frame with **duplicate column labels**,
which makes every later `df["c_a"]` return a 2-column frame and breaks
downstream selection.

<a id="oc-42"></a>
### OC-42 — CONFIRMED, with a concrete behavioural repro

`analyzer.py:223-224` uses polars `.skew()` / `.kurtosis()`, which default to
the **biased** estimators, while `recommendations.py:7` hardcodes
`SKEWNESS_TRANSFORM_THRESHOLD = 1.5`:

```
data [1,2,3,4,10]
polars .skew()  (biased,   used)  = 1.1384   -> 1.1384 > 1.5 is False -> rule does NOT fire
polars .skew(bias=False) / pandas = 1.6971   -> 1.6971 > 1.5 is True  -> rule SHOULD fire

kurtosis: polars biased = -0.2120   vs   pandas unbiased = 3.1520   (sign flip)
```

So the transform recommendation **silently fails to fire on genuinely skewed
data**. Note also `backend/ml_pipeline/_internal/_advisor.py:195` applies a
threshold of `1.0` to **pandas (unbiased)** skew — two different rules for the
same concept across layers.

<a id="oc-40"></a>
### OC-40 — CONFIRMED, root cause identified

The docstring says "Mean-impute", but:

```
input          : [1.0, 2.0, nan, 5.0]
null_count     : 0            <- NaN is NOT null in polars
fill_null(mean): [1.0, 2.0, nan, 5.0]     <- no-op
_impute_matrix : [1.0, 2.0, 0.0, 5.0]     <- NaN became 0.0
true mean      : 2.667
```

Root cause is **polars' null/NaN distinction**, not the `fill_null(0)` fallback
that the finding blames. `fill_null(strategy="mean")` cannot see NaN at all, so
every NaN falls through to `np.nan_to_num(..., nan=0.0)`. PCA/clustering are
therefore biased toward 0. The fix must target NaN (`fill_nan`), not `fill_null`.

---

## Confirmed exactly as filed

| ID | Evidence observed |
| --- | --- |
| OC-02 | `import skyulf` outside the repo → `ModuleNotFoundError: No module named 'skyulf'` |
| OC-16 | `KNNImputer` and `IterativeImputer` both raise `ValueError: Columns must be same length as key` on an all-missing fitted column |
| OC-17 | all-null column, `strategy="mean"`: polars raises `ValueError: must specify either a fill value or strategy`; pandas returns `[nan, nan]` |
| OC-21 | `dist=(n+reg)/(total+reg)` — over 2 categories the "distribution" sums to **1.0345 ≠ 1.0**; smoothing is not normalised by category count |
| OC-23 | `1.0 / -1e-12` → polars `+999999999.99`, pandas `-999999999.99`. `_polars_ops.py:112` substitutes `+epsilon` regardless of denominator sign |
| OC-24 | null group key, mean: polars `[1.0, 3.0, 3.0]` (nulls grouped) vs pandas `[1.0, nan, nan]` (nulls dropped) |
| OC-26 | `norm="none"` → `InvalidParameterError: The 'norm' parameter of normalize must be a str among {'l1','l2','max'}` |
| OC-28 | Box-Cox on `[-1, 0, 5, 9]` returns the input unchanged, no warning, no error |
| OC-32 | all-constant input → `ValueError: No feature in X meets the variance threshold 0.00000` |
| OC-33 | single column, `degree=2`, `interaction_only=True` → output columns `['a']`; no self-product produced |
| OC-34 | stop-word-only corpus → `ValueError: empty vocabulary; perhaps the documents only contain stop words` |
| OC-41 | `analyzer.py:221-222` calls bare `.quantile(0.25)`; polars default is **nearest** → `2.0/3.0` vs pandas linear `1.75/3.25` |
| OC-44 | `drift.py:181-195` thresholds `norm_wd = wd/std_ref` but reports `value=float(wd)` next to the normalized threshold — `value` vs `threshold` disagrees with `has_drift` |
| OC-45 | `drift.py:76-98` — `drifted_count` only iterates `common_columns`; `missing_columns`/`new_columns` are returned but never counted |
| OC-58 | `coerce_on_error=False` (strict): polars silently returns `[False, True, True, True]`; pandas raises `TypeError: Need to pass bool-like values` |
| OC-62 | identical artifact, two processes → `37b734f7ac2fb2df…` vs `3f949c8333ead1b3…` |
| OC-63 | cyclic dict → `RecursionError`; the docstring promises `TypeError` |
| OC-74 | `list_models()` omits all 4 Ensemble models |
| OC-75 | installed polars `1.40.1` vs declared floor `>=1.43.2` |
| OC-78 | `py.typed` declared in packaging metadata, file absent |
| OC-79 | `joblib` imported at module scope, absent from `install_requires` |

---

## Status

**27 of 84 findings independently re-verified.** 25 stand (4 of them are worse
than originally filed), **2 require correction** (OC-01, OC-46).

Not yet independently re-verified: the remaining findings are still supported
only by their originating agent's pasted evidence. Treat those as *probable but
unconfirmed* until re-run. The highest-value unverified ones are OC-03
(22-node schema misprediction), OC-13/14/15/25/27 (UI params ignored),
OC-35/36 (metrics), OC-43, OC-59/60, OC-64/65, OC-66/67, OC-69/70.
