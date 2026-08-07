# Threshold Tuning — Phase 1 (Library) Design

Status: Approved (Phase 1 scope only)
Author: Copilot CLI session, in collaboration with the user
Date: 2026-07-26

## Background

While using `skyulf-core` for a Kaggle classification competition, per-class
decision-threshold tuning was done ad hoc in the notebook via a Nelder-Mead
search over `scipy.optimize.minimize`, tuning each class's decision threshold
against a chosen metric on a held-out validation split. The library today has
no reusable API for this — only ROC/PR curve helpers
(`skyulf.modeling._evaluation`) and OOF-based `stacking_classifier`. This spec
covers adding a first-class, reusable threshold-tuning API to `skyulf-core`.

A related, larger idea — actually applying tuned thresholds to real
`/predict` inference in the deployed app (backend + frontend + DB) — was
discussed and intentionally **descoped to a separate, later spec** (see
"Non-goals" below). This spec is library-only.

## What threshold tuning is for

A classifier's `predict()` implicitly uses a fixed decision rule (e.g.
"predict the class with highest probability", or "predict positive if
probability ≥ 0.5"). That default rule is rarely optimal for the metric you
actually care about — e.g. under class imbalance, maximizing accuracy at
threshold 0.5 can badly hurt recall on the minority class, or F1/MCC/balanced
accuracy can be improved noticeably by moving the decision boundary. Threshold
tuning searches for per-class threshold values, evaluated against a
user-chosen metric on held-out validation data, that the caller can then use
to convert predicted probabilities into class predictions — instead of the
model's default argmax/0.5 rule.

## Non-goals (explicitly out of scope for this spec)

- Persisting tuned thresholds against a deployment, or having the app's
  `POST /predict` route apply them to real inference. This requires a DB
  migration (`Deployment` table has no such column today), a new backend
  endpoint, and a `predict_proba` code path that doesn't currently exist in
  `DeploymentService`. This is real, separate follow-on work and will get its
  own spec once this library piece ships.
- Any frontend changes. The existing client-side threshold slider in
  `PerClassConfusionMatrix.tsx` / `classificationCharts.ts` is untouched by
  this spec.
- Pseudo-labeling / self-training utility (separate, not-yet-started
  feature request from the same brainstorm).

## API

### Module

New file: `skyulf-core/skyulf/modeling/_evaluation/thresholds.py`, following
the existing `_evaluation` module conventions (see `metrics.py`,
`classification.py`). Re-exported from `skyulf.modeling._evaluation.__init__`
and from `skyulf.modeling` top-level, matching how
`calculate_classification_metrics` is exposed today.

### `optimize_thresholds`

```python
def optimize_thresholds(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    metric: Callable[[ArrayLike, ArrayLike], float],
    classes: ArrayLike | None = None,
    strategy: str | None = None,
    grid_points: int = 101,
) -> dict[Any, float]:
```

- `y_true`: 1D array-like of true labels.
- `y_proba`: 2D array-like of predicted probabilities, shape
  `(n_samples, n_classes)`, in the same column order as `classes`.
- `metric`: a callable `(y_true, y_pred) -> float` to **maximize** (e.g.
  `sklearn.metrics.f1_score` with `average="macro"` partially applied by the
  caller). Fully configurable — the library ships no default metric.
- `classes`: explicit class label order matching `y_proba`'s columns. If
  `None`, defaults to `sorted(np.unique(y_true))` (same convention used
  elsewhere in `_evaluation`, e.g. `metrics.py`).
- `strategy`: `"grid"` or `"nelder-mead"`. If `None` (default),
  auto-selects: **`"grid"` for binary** (2 classes), **`"nelder-mead"` for
  multiclass** (3+ classes). `scipy.optimize.minimize` is already a core
  dependency (confirmed in `setup.py`), so no new dependency is introduced.
- `grid_points`: only used by the `"grid"` strategy — number of threshold
  candidates evenly spaced over `(0, 1)` exclusive.
- Returns: `dict` mapping each class label to its tuned threshold (a single
  float for binary — conventionally keyed to the positive/second class; one
  float per class for multiclass).

Binary search: simple grid search over `grid_points` candidate thresholds in
`(0, 1)`, picking the value that maximizes `metric(y_true, y_pred)` where
`y_pred = (y_proba[:, 1] >= t)`.

Multiclass search: `scipy.optimize.minimize(method="Nelder-Mead")` minimizing
`-metric(y_true, apply_thresholds(y_proba, thresholds, classes))`, with
thresholds reparameterized (e.g. optimized in log-space or squashed through a
sigmoid) so the optimizer can't drive them non-positive. Initial guess is a
uniform threshold per class (e.g. `1 / n_classes`, matching an unweighted
argmax starting point).

### `apply_thresholds`

```python
def apply_thresholds(
    y_proba: ArrayLike,
    thresholds: dict[Any, float] | float,
    classes: ArrayLike | None = None,
) -> np.ndarray:
```

- Binary (`thresholds` is a single float, or a dict with one entry): returns
  `classes[1]` where `y_proba[:, 1] >= threshold`, else `classes[0]`.
- Multiclass (`thresholds` is a dict of `{class: threshold}` covering all
  classes): **scaled argmax** —
  `classes[argmax(y_proba / thresholds_array, axis=1)]`, dividing each
  class's probability column by its own threshold before taking argmax. This
  was chosen (over one-vs-rest-with-fallback or additive-bias approaches)
  because it stays well-defined for every input row (always produces exactly
  one prediction, no fallback-class special-casing) and reduces to plain
  argmax when all thresholds are equal.
- Raises `ValueError` if `thresholds` doesn't cover all classes implied by
  `y_proba`'s column count, or if `y_proba` isn't 2D.

### `SkyulfPipeline` wrapper

```python
def optimize_thresholds(
    self,
    X_val: pd.DataFrame | SkyulfDataFrame,
    y_val: pd.Series | ArrayLike,
    metric: Callable[[ArrayLike, ArrayLike], float],
    strategy: str | None = None,
    grid_points: int = 101,
) -> dict[Any, float]:
```

- Requires the caller to explicitly supply held-out validation data
  (`X_val`, `y_val`) — this method does **not** reuse the pipeline's internal
  train/test split. This mirrors the lesson learned from the pseudo-labeling
  leakage bug found during the Kaggle work: never silently reuse internal
  splits for something that needs a clean, independent holdout.
- Internally: transforms `X_val` via `self.feature_engineer.transform`, gets
  `y_proba` via `self.model_estimator.applier.predict_proba(...)` (the same
  code path `predict()` already uses for probability output elsewhere in the
  codebase), then calls the standalone `optimize_thresholds()` function.
- Stores the result on `self._tuned_thresholds` and also returns it.
- Raises `ValueError` if the pipeline has no fitted model, or if the
  underlying estimator doesn't support `predict_proba`.

### `SkyulfPipeline.predict` — opt-in flag

```python
def predict(
    self,
    data: pd.DataFrame | SkyulfDataFrame,
    use_tuned_thresholds: bool = False,
) -> Any:
```

- Default (`False`): unchanged behavior — existing default decision rule.
- `True`: after feature engineering, computes `y_proba` via `predict_proba`
  and calls the standalone `apply_thresholds()` using
  `self._tuned_thresholds`.
- Raises `ValueError` if `use_tuned_thresholds=True` but
  `optimize_thresholds()` was never called on this pipeline instance (no
  silent fallback to the default rule).

## Error handling

- `optimize_thresholds()`/`apply_thresholds()`: `ValueError` for shape
  mismatches (`y_proba` column count vs. `classes` length), unknown
  `strategy` values, and thresholds/classes coverage mismatches in
  `apply_thresholds`.
- Pipeline wrapper: `ValueError` for unfitted pipeline, missing
  `predict_proba` support, and calling `predict(use_tuned_thresholds=True)`
  before tuning.

## Testing plan

- `skyulf-core/tests/test_evaluation_thresholds.py` (new):
  - Binary grid search recovers a known-optimal threshold on a synthetic
    dataset where the optimum is analytically known.
  - Multiclass Nelder-Mead improves the chosen metric vs. plain argmax on a
    synthetic imbalanced multiclass dataset.
  - `apply_thresholds` binary and multiclass correctness on hand-computed
    small arrays (including the "equal thresholds reduces to argmax" case).
  - Error cases: shape mismatch, unknown strategy, incomplete threshold
    coverage.
- `skyulf-core/tests/test_pipeline_*.py` (extend existing pipeline test
  file): `optimize_thresholds()` wrapper end-to-end on a small fitted
  pipeline with a synthetic split; `predict(use_tuned_thresholds=True)`
  raises before tuning and works after; confirms `use_tuned_thresholds=False`
  is unaffected (regression safety).

## Documentation

- Add a `optimize_thresholds()` / `apply_thresholds()` usage section to
  `skyulf-core/README.md`, alongside the existing `get_fitted_split()`
  section, with a short example showing binary and multiclass usage.
- New `## Unreleased` (or next version) changelog entry once implemented.
