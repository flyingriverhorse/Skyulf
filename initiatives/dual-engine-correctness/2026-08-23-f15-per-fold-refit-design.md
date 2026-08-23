# F-15 — Per-Fold Preprocessing Refit: Design Note

**Date:** 2026-08-23
**Status:** **Shipped** — skyulf-core 0.7.0, branch `082` (see amendment below).
**Mandate:** Leakage-enforcement plan §Phase 4, item 10 — "Deliver a design note first — refit contract,
performance budget (`n_splits`× preprocessing cost), migration plan for users whose scores will drop,
and whether to land it opt-in (`refit_preprocessing_per_fold=True`) before flipping the default."
**Companion:** `2026-08-11-audit-findings.md` §4 (the architectural question) and F-15.

---

## Amendment — implementation outcome (2026-08-23)

Shipped the same day as **always-on**, collapsing the §6 migration phases:

- **No users exist**, so the opt-in → bake → default-flip sequence was skipped; there was no
  migration to manage. The app (`PipelineEngine`) always resolves and threads the adapter.
- **No frontend toggle** was built (the planned `refit_preprocessing_per_fold` param was dropped
  entirely — core keeps `preprocessing=None` only as the "caller already transformed" default).
- **Fallbacks instead of rejections:** merged-branch graphs and datasets with a validation split
  fall back to pre-transformed scoring with an explicit job-log warning ("scores may be
  optimistically biased") — never a failed run. Core still raises if the hook is passed into an
  unsupported path; the app simply doesn't pass it there.
- **Payload reconstruction** replaced the planned `cv_preprocessing_steps` job-config key: the
  engine walks the linear upstream chain at training time, re-running the splitter-only step
  prefix on the raw loader frame (or loading the splitter node's artifact) to obtain the
  pre-transform train payload.
- Halving/optuna support landed the same release: preprocessing + model are wrapped in an sklearn
  `Pipeline` (`FoldPreprocessingStep`) whose searcher-internal CV drives the per-fold refit.
- The `docs/examples/leakage_proof_pandas.md` CV caveat is removed (now "Covered").

Everything else in this note (clone contract, adapter shape, performance budget, semantics,
test plan) landed as designed.

---

## 1. Problem statement (verified against current code)

Every CV and tuning score Skyulf reports is systematically optimistic, because preprocessing is
fitted once on the full training split *before* CV begins, and only the **model** is re-fit per fold:

- `perform_cross_validation()` (`skyulf/modeling/cross_validation.py:69`) accepts only
  `calculator`/`applier` — the preprocessing layer is entirely outside its view. `_run_cv_fold`
  (line 208) calls `calculator.fit(X_train_fold, y_train_fold, config)` on data that was already
  transformed with full-training-set statistics.
- The tuning engine (`skyulf/modeling/_tuning/engine.py`) converts X/y to numpy upfront and searches
  over the same already-transformed data; `_refit_best_model` refits the model only.
- `StatefulEstimator.cross_validate` (`skyulf/modeling/base.py:191`) passes
  `dataset.train` post-preprocessing straight through.
- In the app, preprocessing runs in a *separate node*: `_feature_eng.py:279` runs
  `FeatureEngineer(node.params["steps"]).fit_transform(df)` and saves the transformed frame as the
  node artifact; the training node (`_node_runners.py:742`, `_run_tuned_cv`) receives the
  transformed data and never sees the steps config.

So every fold's validation rows already influenced imputation means, scaler statistics, outlier
bounds, and encoder mappings. This is architectural, not a patch.

---

## 2. Why a refactor is possible at all: the clone contract

The feared blocker — "a clean clone/reset contract on every calculator" — turns out to be **already
satisfied by construction**, and this is the key fact that makes the design cheap:

- `FeatureEngineer.__init__` (`preprocessing/pipeline.py:47`) takes only `steps_config` (a list of
  plain dicts, validated by `validate_preprocessing_steps`).
- `fit_transform` (line 85) starts with `self.fitted_steps = []` and builds **fresh**
  calculator/applier instances per step from the registry
  (`_get_transformer_components`, line 612: `NodeRegistry.get_calculator(type_name)()`).
- Calculators therefore have no cross-fit state to reset: **reconstructing
  `FeatureEngineer(steps_config)` *is* the clone operation.** No deep-copy of fitted state is ever
  needed. (F-48 deleted the dead `StatefulEstimator.refit()`; nothing needs to replace it — the
  config list is the single source of truth.)

**Refit contract (new, small, engine-agnostic), to live in `skyulf.modeling`:**

```python
class FoldPreprocessor(Protocol):
    def fit_transform(self, X, y) -> tuple[X, y]:
        """Fit on this fold's training rows only; return transformed train data."""
    def transform(self, X) -> X:
        """Apply fitted artifacts to held-out rows without refitting."""
```

`FeatureEngineer` already fulfils half of this contract: `transform()` (line 59) skips splitters,
resampling and row-dropping steps — exactly the train-only discipline inherited from F-18. What it
lacks is an `(X, y)`-oriented entry point (it consumes a DataFrame/`SplitDataset`, and target-aware
steps need `y`). Implementation adds a thin adapter:

- **`FeatureEngineerFoldAdapter(steps_config)`** — filters out splitter steps
  (`TrainTestSplitter`, `feature_target_split`; already executed upstream), reconstructs a fresh
  `FeatureEngineer` per `fit_transform` call, and exposes the protocol above.
- Per-fold errors (e.g. SMOTE failing on a rare class in a small fold) are caught per fold and
  recorded as NaN with a warning, mirroring `_run_inner_cv`'s existing failure handling
  (`cross_validation.py:486-494`) — a fold failure must never silently corrupt aggregated scores.

---

## 3. API surface and threading

**Core (skyulf-core 0.7.0):**

| Function | Change |
|---|---|
| `perform_cross_validation()` | New optional param `preprocessing: FoldPreprocessor \| None = None`. When set, `_run_cv_fold` runs `X_tr, y_tr = preprocessing.fit_transform(...)` before `calculator.fit`, and `X_val = preprocessing.transform(X_val)` before scoring. `None` keeps byte-for-byte current behavior. |
| `StatefulEstimator.cross_validate` | Pass-through of the same optional param. |
| Tuning engine | Same hook inside the inner CV evaluation so **tuning and CV cannot disagree** (explicit requirement from findings §4). The final `_refit_best_model` keeps using the full-split preprocessing artifact — that is what serving uses, and it is correct for the final model. |

**Backend wiring (`backend/ml_pipeline`):**

1. At pipeline compile time, collect the upstream Feature-Engineering node(s)' `steps_config` for
   each training node and attach it to the job config (new key, e.g. `cv_preprocessing_steps`).
2. The training node needs the **pre-transform** training frame when the flag is on. FE nodes save
   their *input* under the upstream node's artifact id, so the runner can fetch it from the artifact
   store instead of the transformed output.
3. **v1 scope restriction:** a single linear FE chain into the training node. Merged multi-branch
   preprocessing cannot be re-run fold-wise yet — when the graph is a merge, the flag is rejected
   with an explicit diagnostic ("per-fold refit not supported for merged preprocessing branches"),
   never silently ignored.

**Frontend (repo policy: backend params are mirrored):** a `refit_preprocessing_per_fold` toggle on
the training node's CV/tuning settings in `frontend/ml-canvas/src/modules/nodes/`, default off,
with helper text: "Re-fit preprocessing inside every fold. Leak-free scores; usually lower than
legacy runs."

---

## 4. Performance budget

Refitting multiplies preprocessing cost:

| Path | Extra preprocessing fits | Notes |
|---|---|---|
| k-fold CV | k fits on (k−1)/k rows ≈ k× one full fit | Usually dominated by the model fits. |
| Nested CV | outer k × inner k′ | Same structure, multiplicative. |
| Tuning | `n_trials × cv_folds` fits | The expensive case. |

Cheap steps (scalers, imputers with mean/median, encoders, binning) are vectorized Polars/pandas
aggregations — negligible next to model training. The genuinely expensive steps are **KNNImputer,
IterativeImputer, EllipticEnvelope, ModelBasedSelection** (each fit is itself a model). Budget
rules:

1. The flag is opt-in in v1, so the cost is always an explicit user choice.
2. Surface per-fold preprocessing `fit_time` in the CV result (FeatureEngineer already tracks it
   per step) so the cost is visible, not mysterious.
3. Tuning already has a parallel backend (`n_jobs`/`parallel_backend` from settings); fold-level
   preprocessing runs inside the same joblib tasks — no new parallelism plumbing needed.
4. No caching is possible (folds differ) and none should be added.
5. Documented guidance: with KNN/Iterative imputation in the chain, expect roughly
   `n_trials × cv_folds × imputer_fit` to dominate wall time; reduce trials/folds or impute once
   upstream if that is unacceptable (with the understanding that the scores are then the legacy
   optimistic ones).

---

## 5. Semantics users must understand (what changes, what doesn't)

- **Scores move down.** The old numbers were optimistic; the new ones are the honest estimate.
  Nothing is "broken" — the leaderboard gets more truthful.
- **Historical runs are untouched.** Stored metrics are immutable; only re-runs produce the new
  numbers. The experiments comparison view must therefore treat the flag like a metric-comparability
  dimension (F-36-style: runs with different `refit_preprocessing_per_fold` values are not
  directly comparable — surface a warning in the comparison UI).
- **Per-fold feature sets may differ** (feature selection, one-hot cardinality). That is correct:
  each fold's model is evaluated against its own training view. The *served* model keeps the
  full-training-split pipeline artifact — inference behavior is unchanged.
- **Final (post-CV) model fit is unchanged** — it trains on the full training split with the full
  pipeline, exactly as today.

---

## 6. Migration & communication plan

| Phase | Release | Contents |
|---|---|---|
| **1 — opt-in** | core **0.7.0**, app next minor | Land the protocol + adapter + `preprocessing` param (default `None`/off), backend threading, frontend toggle, red-green tests. Changelog entry flagged **scores-will-change-when-enabled**. Docs: update the CV caveat in `docs/examples/leakage_proof.md` ("available behind a flag") but keep the qualification. |
| **2 — bake** | ≥1 release cycle | Collect real-world deltas (how much do scores drop per dataset class); tune the UI copy with measured guidance instead of guesses. |
| **3 — default flip** | following minor | Default on; record as **breaking** in the changelog (reported scores change for everyone); remove the docs caveat; the leakage gate can then extend its guarantee to CV scores. |

Release-note language for Phase 3 must say, plainly: *"CV and tuning scores computed after this
release are lower than before because they no longer leak validation data into preprocessing.
Your model did not get worse; the old estimate was too high."*

---

## 7. Test plan (red-green, per repo policy)

1. **Pure-noise target, target-aware preprocessing** (extends the F-48 pattern):
   TargetEncoder chain + CV with refit on a noise target → ROC-AUC ≈ 0.5 (±0.05), both engines.
   The same test *without* the flag must show the inflated score — proving the test detects the bug.
2. **Fold isolation probe:** instrumented imputer asserting fit statistics come from training-fold
   rows only (validation rows absent from the fit input), both engines.
3. **Determinism:** adapter reconstruction from `steps_config` reproduces identical artifacts across
   folds given the same seed/order (registry construction is pure).
4. **Tuning/CV agreement:** with the flag on, tuned-selection CV and plain CV on the same folds use
   the same per-fold pipeline (no silent divergence).
5. **Merge rejection:** multi-branch graph + flag → explicit diagnostic, not a crash, not silence.
6. **Failure containment:** a step that raises inside one fold (monkeypatched) produces a NaN fold
   + warning, and aggregation skips it.
7. **Byte-for-byte legacy:** flag off ⇒ identical results to current code (characterization test).

---

## 8. Open questions (resolve at implementation kickoff, not here)

- Should the experiments page store the flag in the run record for comparability filtering, or
  only warn? (Recommendation: store it; cheap and enables honest comparisons later.)
- Time-series splits with per-fold refit: cost grows with fold count since folds are expanding —
  acceptable, but worth a doc note.
- Whether `cv_preprocessing_steps` should be validated against the live graph at job start
  (drift between compiled pipeline and replayed JSON) — recommend yes, same validation path as
  `validate_preprocessing_steps`.

---

*Definition-of-Done update:* this note satisfies the leakage-enforcement plan's Phase 4 requirement
"Phase 4 (CV refit) has a written design note". The docs CV caveat was removed when F-15 shipped
always-on (see amendment).
