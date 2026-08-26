# skyulf-core Second-Pass Deep Review

**Date:** 2026-08-26 · **Status:** Investigation complete, prioritized fix
list ready, **no fixes applied**

## What this is

The [first-pass review](README.md) covered backend + core at breadth. This
is a second, deeper pass over `skyulf-core/` only, run by four parallel
specialists scoped to areas the first pass touched lightly:

1. **Modeling layer** — CV, tuning strategies, metric handling, thresholds
2. **Data transforms & profiling** — every preprocessing node, profiling
   subsystem, dual-engine parity at apply time
3. **Adversarial edge cases** — concurrency/state, index handling, numeric
   guards, memory, serialization (in the Celery worker reality)
4. **SDK surface & DX** — README claims vs reality, discovery APIs,
   packaging, exceptions, logging

Agents were briefed with the first-pass E1–E9 and dual-engine findings as
do-not-repeat, so everything below is **new**.

**Methodology caveat:** static review plus targeted probe execution
(several findings were run live, e.g. the README quickstart). Reproduce
before fixing anything marked CRITICAL/HIGH.

## Headline numbers

| Severity | Count | Examples |
|---|---|---|
| CRITICAL | 1 | README quickstart crashes as written — the first thing a new user runs |
| HIGH | 10 | Time-series X/y misalignment, `group_agg` target leakage, metric-direction inversion in tuning, cross-job contamination via global loggers |
| MED | ~22 | 8 engine divergences, index-explosion chains, SentenceEmbedder offline failure, eager optional imports |
| LOW | ~8 | Exception taxonomy, packaging metadata, reproducibility, test hygiene |

Two of the HIGHs are **silent-corruption** bugs (wrong model selected,
wrong labels paired) — worse than crashes because nothing errors.

---

## TS. Time-series & alignment (P0 — silent data corruption)

**TS1. LagFeatures / RollingAggregate sort X but never y (HIGH, confirmed by two reviewers).**
With `sort_by` set, X is sorted but `_y` is returned in original order, so
every label pairs with the wrong row. The polars variant is worse: the
null mask is computed on *sorted* X and applied *positionally* to unsorted
y (`time_series/lag.py:45-60`), dropping the wrong y rows while row counts
still match — even a length check passes. `pack_pipeline_output`
(`utils.py:159-163`) then positionally resets both, cementing the
misalignment. `rolling.py:63-67,111-124` has the same shape.
**Fix:** apply the identical sort/gather to y, or reject `sort_by` unless
y is co-sorted; add a red-green test with unsorted time-series input.

**TS2. LagFeatures pandas `drop_na` leaves y unfiltered (HIGH).**
X is `dropna()`-ed but `_y` is returned at full length → shape mismatch
downstream; it also drops rows null in ANY column, not just lag columns.
The polars path does filter y → cross-engine divergence
(`time_series/lag.py:88-90`).
**Fix:** filter y with the same mask; restrict the drop to lag columns.

**TS3. Row-drop nodes explode y on duplicated index labels (HIGH).**
`y.loc[X_clean.index]` (`drop_and_missing/deduplicate.py:47`,
`drop_rows.py:69`): with duplicated labels — common after `pd.concat`
without `ignore_index` or after oversampling — `.loc` returns every row
per label, so y gains rows X doesn't have; tuple mode has no length guard,
so the misshapen pair propagates silently into fit.
**Fix:** reset the index before filtering or select y positionally.

**TS4. Oversampling feeds TS3 (MED).**
`_finalize_resampled` never resets the index on imblearn output
(`resampling.py:74-82`); sampling with replacement yields duplicated
labels. So the realistic chain `Oversampling → Deduplicate/DropMissingRows`
in one pipeline detonates TS3.
**Fix:** `reset_index(drop=True)` on resampled X/y.

---

## L. Leakage & metric correctness (P0 — wrong numbers, no error)

**L1. `group_agg` leakage: aggregation computed over the whole frame at apply time (HIGH).**
The operation is stored with `learns_from_data=False` and no artifact
stats (`preprocessing/feature_generation/generation.py:35-46`); at apply
time both engines compute the aggregation over the *entire input frame*
via `over`/`transform` (`_polars_ops.py:242-256`, `_pandas_ops.py:192-203`).
Consequences: test-fold rows leak into train-fold features during CV, and
if the aggregated column is the target this is un-cross-fitted target
encoding. Note pipeline-level resampling *is* correctly train-gated
(`pipeline.py:32`) — this is the one aggregation path that isn't.
**Fix:** learn group stats on the train frame at fit and apply via
mapping; block or hard-warn when the target column is aggregated.

**L2. Metric direction inverted for unmapped error metrics — tuning silently picks the WORST model (HIGH).**
All tuning strategies *maximize* the objective
(`modeling/_tuning/engine.py:890,1143`); `_resolve_metric`
(`engine.py:689-714`) has no explicit direction table, so any error-type
metric not in the known-good map — including `mape`, a name the SDK
itself emits (`metrics.py:339`) — is maximized, i.e. the candidate with
the worst error wins. No exception is raised.
**Fix:** explicit direction table keyed by metric name; reject unknown
names with a clear error instead of defaulting to maximize.

**L3. `log_loss` dropped on folds missing a trained class (MED).**
No `labels=` is passed (`metrics.py:299-300`), so a fold where predict_proba
returns fewer columns than the full class set raises/skips — unlike the
adjacent `roc_auc` fix which does pass `labels=` (`metrics.py:266-277`).
**Fix:** mirror the roc_auc treatment.

**L4. `optimize_thresholds` rejects tuned models (MED).**
Reads `classes_` directly off the step's model, which for tuned models is
a `(model, tuning_result)` tuple (`pipeline.py:369-371`,
`modeling/base.py:565-571`) → AttributeError, so threshold optimization
is unusable exactly where tuning happened.
**Fix:** unwrap via the existing `_unwrap_tuned_model` helper.

---

## CV. Tuning & cross-validation robustness (P1)

**CV1. Grid/random + holdout + preprocessing fails instead of the documented fallback (MED).**
Halving/optuna fall back to a frameless split when preprocessing is in
play (`engine.py:1390-1401`); grid/random lack that branch
(`engine.py:1445`), producing a numpy `PredefinedSplit` payload the frame
adapters can't consume (`:798-800`) → every trial fails → "All trials
failed".
**Fix:** reuse the frameless fallback for grid/random.

**CV2. `shuffle_split` is never stratified for classification (MED).**
`cross_validation.py:426` + `engine.py:545-551` always use plain
`ShuffleSplit`; imbalanced targets get unbalanced folds.
**Fix:** `StratifiedShuffleSplit` when the task is classification.

**CV3. Grid threshold strategy accepted for 3+ classes (LOW).**
`thresholds.py:210-211` doesn't restrict the grid strategy to binary.
**Fix:** validate and raise with the supported-class count.

**CV4. No CV-size validation (LOW).**
`n_splits` larger than the smallest class passes silently into sklearn,
which errors opaquely mid-fit. **Fix:** pre-check fold size ≥ 2 per class
with a named error.

**CV5. Optuna reproducibility broken at `n_jobs>1` (LOW).**
A fixed seed with parallel search doesn't reproduce; documented neither
way. **Fix:** warn, or serialize when `seed` is set.

---

## P. Dual-engine parity — new apply-time divergences (P1)

Nine more live instances of the dual-engine bug class (first-pass E1–E6);
every one means "same config, different data per engine":

| # | Divergence | Evidence | Fix |
|---|---|---|---|
| P1 | **GeneralBinning ordinal codes off-by-one** — polars `cut().cast(UInt32)` is 1-based, pandas `pd.cut(labels=False)` 0-based; same row gets bin 1 vs 0 | `bucketing.py:100-102` | subtract 1 in polars or map categories explicitly |
| P2 | **Bucketing `missing_strategy`/`missing_label` pandas-only** — polars ignores both, emits null | `bucketing.py:106-134` vs `200-211` | implement via `fill_null` in `_build_polars_exprs` |
| P3 | **DateFeatures string parsing** — polars ISO-only `strict=False` vs pandas coerce; "12/31/2021" works in pandas, nulls in polars | `date_features.py:92-104` | share one parser/format-inference |
| P4 | **Resampling y shape** — `pl.from_pandas(y_res)` on a Series returns a DataFrame in polars | `resampling.py:121` | `.to_series()` / `pl.Series(...)` |
| P5 | **Stratified split null policy** — pandas `value_counts()` excludes NaN (then sklearn raises) while polars counts null as a class | `split.py:59-94` | identical null policy, error up front |
| P6 | **Deduplicate y realignment** — pandas `.loc` (KeyError risk, see TS3) vs polars positional | `deduplicate.py:47` | positional in both |
| P7 | **Profiling `duplicate_rows`** — polars counts ALL members of duplicate groups, pandas extras-only; metric differs by engine | `profiling/analyzer.py:172` | pick one definition (e.g. `n_unique`-based) |
| P8 | **Profiling eta²** — `ss_total`/global mean include null-target rows, `ss_between` excludes them → biased eta² | `target.py:50-77` | restrict all terms to non-null-target rows |
| P9 | **Profiling dates** — format inferred from `head(50)`, whole-column cast `strict=False` nulls non-matches silently; all-null date columns serialize `min_date`/`max_date` as the literal string `"None"` | `dates.py:31-46,164-171` | report parse-failure rate; emit JSON null |

---

## C. Concurrency & worker-process state (P1)

These matter specifically because core runs inside **prefork Celery workers
serving concurrent jobs** (first-pass J-series):

**C1. Concurrent tuning jobs cross-contaminate via the process-global "optuna" logger (HIGH).**
Every optuna run attaches a `_TrialFailureHandler` to
`logging.getLogger("optuna")` and appends to its own `captured` list
(`engine.py:1201-1243`). Two concurrent tunings capture each other's
trial-failure messages (wrong "First trial error" diagnostics), and one
job's `finally: removeHandler` can strip the handler mid-search of the
other, silencing its failure capture.
**Fix:** dedicated child logger or `LoggerAdapter` per search, or filter
records by thread id.

**C2. `tracemalloc.start()/stop()` on every fit is process-global (HIGH).**
Each step fit starts/stops tracemalloc (`preprocessing/base.py:165-185`);
concurrent fits race — thread B's `stop()` kills tracing mid-measurement
of thread A, corrupting both jobs' `peak_memory_bytes`.
**Fix:** process lock + reference count, or `resource.getrusage` deltas.

**C3. SentenceEmbedder: unlocked cache + inference-time model fetch (MED).**
Unsynchronized check-then-set on `_MODEL_CACHE`
(`vectorization/sentence_embedder.py:25,33-44`) → concurrent jobs
duplicate multi-GB loads. Worse, the artifact stores only `model_name`,
so `apply()` calls `_load_model()` again at transform time — serving hits
the HuggingFace hub and hangs/fails on fresh or offline workers.
**Fix:** lock the cache; snapshot weights into the artifact or a pinned
local dir at fit.

**C4. `get_fitted_split()` silently refits the live pipeline (MED).**
It calls `self.feature_engineer.fit_transform(data)` (`pipeline.py:291`),
which wipes `fitted_steps` (`preprocessing/pipeline.py:90`). After
`fit()`, calling it with any other data replaces the preprocessing
artifacts that `predict()` uses — silent corruption of a trained model.
**Fix:** run a throwaway `FeatureEngineer`.

**C5. Optuna lazy-load race (LOW).**
`_ensure_optuna_loaded()` mutates four globals without a lock
(`engine.py:51-107`); a concurrent reader can see
`_optuna_load_attempted=True` with `OptunaSearchCV` still `None`.
**Fix:** module-level `Lock`.

---

## T2. Transform robustness (misc, MED)

**T2a. `value_replacement` boolean key coercion.** `_coerce_key` maps any
string other than `"true"/"1"` to `False` (`value_replacement.py:21-34`)
— `mapping={"yes": 1}` on a Boolean column replaces genuine `False`
values, both engines. **Fix:** reject unparseable keys; never
silent-coerce.

**T2b. Interaction name collision.** `"_x_".join(sorted(cols))` can
collide with an existing column literally named `a_x_b`
(`interaction.py:30-44,88,118`): polars `with_columns` silently
overwrites (data loss), pandas creates duplicates. **Fix:** collision
check → rename or raise.

**T2c. PII phone false positives.** Anchored regex + ≥7-digit count flags
numeric ID columns (e.g. `123456789`) as phone PII with
`severity="error"` (`text.py:106-123`) → noisy profiling alerts.
**Fix:** require separators/leading `+` or column-context checks.

**T2d. Dense vectorization memory blowup.** Hashing output always
`.toarray()`; only a column-count warning, no row guard
(`vectorization/_common.py:187-198`). **Fix:** chunk rows or keep sparse.

---

## DX. SDK surface, docs & packaging (P1)

**DX1. README quickstart crashes as-written (CRITICAL).**
The snippet imputes only `income` while `age` contains nulls
(`README.md:80-98`); `fit()` dies in `sklearn_wrapper.fit` with
`ValueError: Input X contains NaN` (via `pipeline.py:242`). Verified by
execution; the escaped notebook hides it by synthesizing null-free data.
**Fix:** impute `["income", "age"]` in the snippet — verified to make the
whole quickstart (fit→save→load→predict) pass — **and** add a CI smoke
test that executes README snippets (see DX9; this is how it shipped).

**DX2. README leakage-check snippet contradicts default behavior (HIGH).**
`README.md:248` shows `warnings = skyulf.validate_leakage_safety(config)`,
but `on_leakage` defaults to `"raise"` (`leakage.py:112`) — verified it
raises instead of returning warnings. **Fix:** show `on_leakage="warn"`.

**DX3. Node discovery miscategorizes ensembles (HIGH).**
`list_models()` filters `category == "Modeling"` (`registry.py:101-108`);
ensembles register as `"Ensemble"`, so `voting_classifier` et al. are
absent from `list_models()` yet appear in `list_transformers()` — while
`README:202` points users to exactly these methods. **Fix:** treat
`Ensemble` as models in both list methods.

**DX4. `py.typed` declared but missing (HIGH).**
`setup.py:22` and `MANIFEST.in:3` reference `skyulf/py.typed`; the file
doesn't exist and is absent from the built wheel — PEP 561 typing is
silently disabled for every downstream user. **Fix:** add the empty
marker file.

**DX5. Optional deps imported eagerly at `import skyulf` (MED).**
xgboost/lightgbm in module-level try/except
(`classification.py:23-31`, `regression.py:16`, `ensemble.py:57-64`,
`_boosting_progress.py:37`), vaderSentiment in profiling utils — measured
12 xgboost modules + vader in `sys.modules` after a bare `import skyulf`.
optuna is already lazy (`engine.py:47-55`). **Fix:** apply the same lazy
pattern.

**DX6. LightGBM logging globally silenced at import (MED).**
`classification.py:45-46` registers a silent logger on the lightgbm
package at import time — mutes lightgbm for the entire host process,
including the user's own code. **Fix:** move into the LGBM fit or gate
behind a flag.

**DX7. statsmodels hard-required for one EDA feature (MED).**
Pinned in `setup.py:31`; sole use is a guarded `adfuller` import
(`profiling/_analyzer/_utils.py:110`). **Fix:** move to an `[eda]` extra.

**DX8. `compute_shap_explanation` returns `None` silently when shap is missing (MED).**
(`shap_explanation.py:296-300`) — the visualizer prints install hints,
the function should too. **Fix:** raise/log `pip install
skyulf-core[explainability]`.

**DX9. No exception hierarchy + message inconsistencies (LOW).**
~95 raise sites, ~85 bare `ValueError`; users can't `except SkyulfError`.
Also: `get_applier` omits the available-nodes list `get_calculator` gives
(`registry.py:62 vs 71`); `pipeline.py:185` dumps ~90 names with no typo
suggestion (`difflib` is already used in `config_validation.py`);
`pipeline.py:313` conflates "not fitted / no model" without the `fit()`
hint used at `:365`. **Fix:** `SkyulfError(ValueError)` base for public
surfaces; borrow the typo-suggestion pattern.

**DX10. Packaging & test hygiene (LOW).**
Classifiers list only Python 3.12 (`setup.py:88-90`) though tests run on
3.14; `lgbm_regressor` has hyperparameter spaces but no node
(`_registry.py:71,296,491`) while xgboost has both — dangling surface.
`numpy>=1.24.0` is unpinned while every artifact embeds numpy arrays
(`setup.py:25`; sklearn/pandas are capped) → cap `numpy<3` and stamp
versions into saved artifacts. `conftest.py:23,41` mutates global
`np.random.seed` (fixture leakage); `test_utils.py:442` gates 9 tests on
optional polars though polars is a hard dep; no pytest config anywhere.

**Verified clean (positives):** numeric guards (log/sqrt negative → NaN,
divide epsilon, exp clipping); `FoldAwareModelStep.fit` deep-copies
preprocessor/model so searcher `n_jobs>1` is leak-safe (recent fix
`ecc4c280` confirmed complete); artifacts hold picklable sklearn objects
only; clean `__all__`; no `basicConfig`; extras all valid; examples
resolve; suites green and fast (1584+1849 passed, ~80s).

---

## Execution proposal (core-only waves)

**Wave A — silent-corruption stopgaps, ~3–5 days:**
TS1–TS4 (time-series y alignment + index chain), L1 (group_agg leakage),
L2 (metric direction table). Each red-green; these produce wrong models
with no error.

**Wave B — parity batch, ~1 week:**
P1–P9 as one red-green parity-test batch (extends the first-pass Wave 2
and the dual-engine T5 contract test); CV1/CV2; L3/L4.

**Wave C — concurrency hardening, ~3–5 days:**
C1–C5. Directly complements backend J-series fixes — same worker
reality.

**Wave D — DX & packaging, ~2–3 days (cheap, high visibility):**
DX1–DX4 first (the top-5 wins: fix quickstart + snippet smoke test,
ensemble categorization, ship `py.typed`, fix leakage snippet), then
DX5–DX10. DX1 alone fixes the literal first-run experience of every new
SDK user — pairs with growth/ Stage 2 activation.

---

## Relation to other initiatives

- **[README.md](README.md) (first pass):** this is its core-side
  companion; waves A–D slot between/alongside first-pass Wave 2
  ("core parity & safety"). L1 is the same leakage *class* that
  `ecc4c280` fixed for CV — this is the remaining live instance.
- **dual-engine-correctness/:** P1–P9 are nine more live instances of its
  bug class — further evidence to ship the T5 parity-test contract rather
  than fixing divergences one at a time forever.
- **growth/:** DX1 (quickstart crash) is a first-run activation killer for
  the clone-driven audience (2,709 clones / 14 days per the traffic
  check in `2026-08-12-execution-tasks.md`).
- Effort figures are judgement estimates for sequencing only, per repo
  convention.
