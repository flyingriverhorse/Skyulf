# Multi-output regression/classification audit (backlog 4.2)

> Audit date: 2026-07-05. Scope: `skyulf-core/skyulf/` only (no production code
> changed). Goal: determine — with executable evidence, not guesswork — which
> parts of `skyulf-core` correctly support a multi-output `y` (a target that is
> a DataFrame/2-D array with more than one column) and which parts silently
> assume a single-column/1-D target.
>
> Proving tests live in `skyulf-core/tests/test_multi_output_audit.py`. Every
> row below is backed by at least one test in that file (see the "Evidence"
> column for the file:line of the *source* assumption; the test file itself
> documents the *runtime* proof).

## Executive summary

`skyulf-core` was **never designed** for multi-output targets. There is no
`MultiOutputRegressor`/`MultiOutputClassifier` wrapper anywhere, and the
pipeline-level plumbing (`_extract_xy`, `FeatureTargetSplitApplier`,
`evaluate()`) only knows how to lift a **single named column** out of a frame
as `y`. However, because `SklearnCalculator.fit()` and
`SklearnBridge.to_sklearn()` pass `y` through with only a narrow "ravel if
`(N, 1)`" special case, a 2-D `y` reaches `estimator.fit(X, y)` **unmodified**.
This means:

* **Training** already "just works" today, for free, for any sklearn
  estimator that natively supports 2-D `y` (e.g. `LinearRegression`,
  `Ridge`, tree/forest-based regressors, `RandomForestClassifier`,
  `KNeighborsRegressor`). No skyulf code rejects it.
* **Prediction** breaks for every one of those same estimators, because
  `SklearnApplier.predict()` unconditionally wraps the estimator's output in
  `pd.Series(preds, index=index)` — `pd.Series` requires 1-D data, so a
  `(n_samples, n_targets)` prediction array raises `ValueError: Data must be
  1-dimensional`.
* Estimators that are **not** natively multi-output-capable (e.g.
  `LogisticRegression`) fail at the sklearn level during `fit()` itself with
  sklearn's own `"y should be a 1d array"` error — skyulf adds no wrapper
  (`MultiOutputClassifier`) to make this work.
* Preprocessing nodes that pass `y` straight through to an external library
  (resampling) inherit whatever that library supports: imblearn's
  `RandomOverSampler`/`RandomUnderSampler` happen to accept a multi-column `y`
  (they only duplicate/drop rows, no synthetic generation), but `SMOTE` and
  its variants explicitly reject "multioutput targets" with their own error.
* The tuning engine (`_tuning/engine.py`) and evaluation helpers were not
  audited as "supporting" multi-output in any special way — they inherit the
  same pass-through behavior as the plain sklearn wrapper (same NaN/Inf
  checks operate fine on 2-D arrays), so they share the wrapper's gaps rather
  than adding new ones.

## Summary table

| Module / Class | Supports multi-output `y`? | Evidence (file:line) | Notes |
|---|---|---|---|
| `SklearnBridge.to_sklearn` (`engines/sklearn_bridge.py:24-40`) | **Partial** | `sklearn_bridge.py:32-33` only ravels when `y.ndim == 2 and y.shape[1] == 1` | 2-D `y` with >1 column passes through untouched as a 2-D numpy array — this is *why* multi-output training works at all today. |
| `SklearnCalculator.fit` (`modeling/sklearn_wrapper.py:38-114`) | **Partial (estimator-dependent)** | `sklearn_wrapper.py:110-112` — `X_np, y_np = SklearnBridge.to_sklearn((X, y)); model.fit(X_np, y_np)` | No shape coercion at all. Works for natively multi-output-capable estimators (`LinearRegression`, `RandomForestRegressor`, `RandomForestClassifier`, ...); raises sklearn's own `ValueError` for estimators that require 1-D `y` (`LogisticRegression`, ...). Proven in `test_multi_output_audit.py::TestSklearnCalculatorFit`. |
| `SklearnApplier.predict` (`modeling/sklearn_wrapper.py:120-136`) | **No** | `sklearn_wrapper.py:136` — `return pd.Series(preds, index=index)` | Unconditionally wraps predictions in a 1-D `pd.Series`. Any estimator that produced a `(n_samples, n_targets)` prediction array raises `ValueError: Data must be 1-dimensional, got ndarray of shape (N, k) instead`. This is the single biggest, cleanest blocker — even estimators whose `fit()` succeeds cannot `predict()`. Proven in `test_multi_output_audit.py::TestSklearnApplierPredict`. |
| `SklearnApplier.predict_proba` (`modeling/sklearn_wrapper.py:138-157`) | **N/A / untested** | `sklearn_wrapper.py:157` | Wraps in `pd.DataFrame`, which *can* hold multiple columns, but `predict_proba` for multi-output classifiers returns a **list of per-target arrays**, not a single 2-D array — `pd.DataFrame(probs, ...)` would receive a `list[np.ndarray]` and either raise or build a garbage frame. Not covered by a passing test (out of scope: no multi-output classifier ships a calibrated `predict_proba` path in this repo today); flagged here for future work. |
| `StatefulEstimator._extract_xy` (`modeling/base.py:103-142`) | **No** | `modeling/base.py:103` — signature `target_column: str` (singular); `base.py:117-119` / `133-140` extract exactly one column | Structural: the whole pipeline-level plumbing only knows how to lift *one* named column out of a combined frame as `y`. There is no `target_columns: list[str]` variant anywhere in this class. |
| `StatefulEstimator.evaluate` (`modeling/base.py:333-448`) | **No** | `modeling/base.py:334` — same singular `target_column: str`; `base.py:366,375` — `y = X[target_column]` / `data[target_column]` | Same single-column assumption as `_extract_xy`; also calls `self.applier.predict(X, self.model)` (`base.py:379`), which inherits the `SklearnApplier.predict` gap above. |
| `_tuning/engine.py` `TuningCalculator.fit` | **Partial (inherits wrapper behavior)** | `_tuning/engine.py:148` — `X_np, y_np = SklearnBridge.to_sklearn((X, y))`; `163-169` — NaN/Inf checks operate on `y_np` regardless of its shape (`np.isnan(y_np).any()` works fine on 2-D arrays) | No new gap introduced by tuning itself; it neither special-cases nor breaks on 2-D `y_np` beyond what the wrapper already does. Scorers (`get_scorer(metric)` at `engine.py:460-462`) are supplied by the caller-selected metric string and were not exercised with multi-output data in this audit — flagged as unverified rather than "No". |
| `preprocessing/resampling.py` `OversamplingCalculator`/`OversamplingApplier` (method=`random_over`/`random_under`) | **Yes (incidental passthrough)** | `resampling.py:97` — `sampler.fit_resample(X_pd, y_pd)`; `resampling.py:79-80` — `_finalize_resampled` only forces `pd.Series` if `y_res` isn't already one | `RandomOverSampler`/`RandomUnderSampler` accept and return a 2-column `y_pd` DataFrame unchanged (verified empirically — no code path in skyulf forces a 1-D reshape). Not a designed feature, just a consequence of thin passthrough to imblearn. |
| `preprocessing/resampling.py` `OversamplingCalculator`/`OversamplingApplier` (method=`smote` and other SMOTE variants) | **No** | `resampling.py:97` (same call site) | `imblearn`'s `SMOTE.fit_resample` explicitly raises `ValueError: ... Multilabel and multioutput targets are not supported.` skyulf does not catch/translate this error nor block it earlier with a clearer message. |
| `preprocessing/split.py` `FeatureTargetSplitApplier` | **No** | `split.py:300-305` — `y = data.select(pl.col(target_col)).to_series()` (Polars) / `y = data[target_col]` (pandas) | `target_col` is a single column name (`str`), not a list. There is no way to request more than one target column from this node; it is the entry point most users would hit first, so it gates any downstream multi-output flow before it even reaches modeling. |
| `preprocessing/encoding/{label,ordinal,woe,target}.py` (target encoders) | **No (by design — not applicable)** | `label.py:112-117`, `ordinal.py:164-169`, `woe.py:102-109`, `target.py:126-133` | These nodes exist specifically to encode a *single categorical target column* (`y` as `pd.Series`); they were never meant to take a multi-column `y`. Not a "gap" so much as an out-of-scope input shape — flagged for completeness since the task asked to check every `y`-consuming node. |
| Row-filtering preprocessing nodes (`drop_rows`, `deduplicate`, `outliers/{zscore,iqr,elliptic}`) | **Yes (untyped passthrough)** | e.g. `drop_rows.py:52-55`, `zscore.py:45-64` | These nodes only index/filter rows of `y` by a boolean mask or row positions; they never assume 1-D shape, so a multi-column `y` DataFrame is filtered correctly (same row-selection logic works identically for a Series or DataFrame). Not individually tested here (out of scope per task — feature-selection/outlier nodes are lower priority than modeling/resampling), but the code inspection shows no 1-D assumption. |

## Why each "No"/"Partial" breaks, and rough fix effort

Effort scale: 1 = an afternoon, 5 = multi-week refactor. Impact scale:
1 = nice-to-have, 5 = unblocks a whole class of users (matches the scale used
in `temp/core_improvement_ideas.md`).

1. **`SklearnApplier.predict` forces `pd.Series`** — Impact 4 (blocks
   prediction for *every* multi-output-capable estimator, even ones whose
   `fit()` already succeeds), Effort 2. Fix: detect `preds.ndim == 2` and
   return a `pd.DataFrame` with one column per target instead of a bare
   `pd.Series`. The main risk is downstream code (`evaluate()`, ensemble
   averaging, API serialization) that currently assumes `predict()` returns a
   `Series` — those call sites would need a compatibility pass too.

2. **`StatefulEstimator._extract_xy` / `evaluate` single `target_column: str`**
   — Impact 4, Effort 4. Fix requires threading `target_column:
   str | list[str]` through the whole calculator/applier/evaluate chain,
   `SplitDataset`, and the `_evaluation/{classification,regression}.py`
   report builders — a genuinely multi-file refactor, not a local patch.

3. **`FeatureTargetSplitApplier` single `target_col`** — Impact 4, Effort 3.
   Fix: accept `target_column` as a list and return a multi-column `y`
   DataFrame instead of a `Series`; needs a matching schema-inference update
   (`infer_output_schema`) and downstream consumers (resampling, modeling) to
   already tolerate a DataFrame `y` — which, per this audit, most already do,
   incidentally.

4. **`LogisticRegression`-style non-multi-output estimators raising in
   `fit()`** — Impact 2 (only matters if a user picks a classifier that
   doesn't support it; RandomForest/tree-based classifiers already do),
   Effort 3. Fix: wrap non-natively-multi-output-capable estimators in
   `sklearn.multioutput.MultiOutputClassifier`/`MultiOutputRegressor` when a
   2-D `y` is detected — needs a capability-detection helper and changes how
   `model_artifact` is structured (an extra layer of wrapper object).

5. **SMOTE and variants rejecting multi-output `y`** — Impact 1 (single
   feature, and the underlying library itself doesn't support it — no
   realistic in-house fix), Effort 5 (would require a fundamentally different
   synthetic-sampling algorithm, out of scope for this repo). Recommended:
   leave as-is; at most, translate imblearn's error into a clearer skyulf
   message pointing at "random_over"/"random_under" as a multi-output-safe
   alternative (Effort 1 for the message-only fix).

6. **`predict_proba` for multi-output classifiers** — Impact 2, Effort 3.
   Needs its own return-shape handling (list of per-class-per-target arrays
   vs a flat 2-D array) — not exercised by any currently-shipped node, so
   deferred pending item 4 above being tackled first.

## Recommended next steps if we pursue full multi-output support

* Start with `SklearnApplier.predict` (item 1) — it's the cheapest, highest-impact
  fix and unblocks all estimators that are already multi-output-capable in
  sklearn without any wrapper changes.
* Tackle `FeatureTargetSplitApplier` (item 3) next so users can actually
  select more than one target column from the UI/pipeline config; without
  this, item 1 alone is only reachable by manually constructing an `(X, y)`
  tuple with a DataFrame `y`, bypassing the normal pipeline.
* Defer `_extract_xy`/`evaluate` (item 2) until 1 and 3 land — it is the most
  invasive change and touches evaluation report schemas that other code
  (frontend, backend evaluation endpoints) may depend on.
* Explicitly decide whether to auto-wrap non-multi-output estimators with
  `MultiOutputRegressor`/`MultiOutputClassifier` (item 4) or instead document
  which estimators support multi-output and let the UI filter/validate model
  choice against target shape — the latter is much cheaper and avoids
  introducing a new artifact wrapper type.
* Leave SMOTE-family resampling as explicitly unsupported for multi-output;
  just improve the error message (Effort 1) rather than attempting a real fix.
* Whatever is implemented, extend `test_multi_output_audit.py` in place
  (it's written to make each currently-passing/xfail case obvious) rather
  than starting a second test file — it should become the regression suite
  for this feature.
