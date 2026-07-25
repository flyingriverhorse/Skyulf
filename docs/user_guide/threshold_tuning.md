# Threshold Tuning

`predict()`'s default decision rule — `argmax` for multiclass, `0.5` for
binary — is rarely optimal for imbalanced classes or for a metric you
actually care about (F1, MCC, balanced accuracy, a custom business metric, …).
Skyulf can search for **per-class decision thresholds** that maximize a metric
you supply, evaluated on held-out validation data.

The functionality is available at two levels:

- **Pipeline level:** `SkyulfPipeline.optimize_thresholds()` +
  `predict(use_tuned_thresholds=True)` — handles preprocessing and
  `predict_proba` for you.
- **Array level:** `skyulf.modeling.optimize_thresholds()` /
  `apply_thresholds()` — plain NumPy in/out, for use outside a pipeline (e.g.
  with a raw sklearn estimator).

## Pipeline usage

Tune against explicit, out-of-sample validation data — **never** the
pipeline's internal train/test split. Get a clean holdout via
[`get_fitted_split()`](validation_vs_sklearn.md) (or your own split) first:

```python
from sklearn.metrics import f1_score

# Clean, independent holdout for tuning (uses the pipeline's configured preprocessing).
X_train, y_train, X_val, y_val = pipeline.get_fitted_split(
    customers, target_column="purchased"
)
pipeline.fit(customers, target_column="purchased")

thresholds = pipeline.optimize_thresholds(
    X_val,
    y_val,
    metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
)
# thresholds: {False: 0.62, True: 0.38} (binary)
#             or one entry per class (multiclass)

# Later, predict using the stored thresholds instead of the default rule:
tuned_predictions = pipeline.predict(new_customers, use_tuned_thresholds=True)
```

`optimize_thresholds()` runs the pipeline's already-fitted preprocessing on
`X_val` internally, calls the model's `predict_proba`, searches thresholds,
and stores the result on the pipeline. `predict(use_tuned_thresholds=True)`
then applies those stored thresholds; it raises a `ValueError` if
`optimize_thresholds()` was never called, and requires a model that supports
`predict_proba`.

### `SkyulfPipeline.optimize_thresholds` signature

```python
pipeline.optimize_thresholds(
    X_val,                 # validation features, NOT yet transformed
    y_val,                 # validation true labels
    metric,                # Callable (y_true, y_pred) -> float, to maximize
    strategy=None,         # "grid" | "nelder-mead" | None (auto-select)
    grid_points=101,       # candidates for the "grid" strategy
) -> dict[Any, float]      # {class_label: threshold}
```

## Array-level usage

Use these directly when you are not going through a `SkyulfPipeline` — for
example, tuning a raw sklearn estimator's thresholds:

```python
from skyulf.modeling import optimize_thresholds, apply_thresholds
from sklearn.metrics import f1_score

y_proba = model.predict_proba(X_val)  # shape (n_samples, n_classes)

thresholds = optimize_thresholds(
    y_val,
    y_proba,
    metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
)

y_pred = apply_thresholds(model.predict_proba(X_test), thresholds)
```

### `optimize_thresholds` signature

```python
optimize_thresholds(
    y_true,                # 1D array-like of true labels
    y_proba,               # (n_samples, n_classes) predicted probabilities
    metric,                # Callable (y_true, y_pred) -> float, to maximize
    classes=None,          # explicit class order; defaults to sorted(unique(y_true))
    strategy=None,         # "grid" | "nelder-mead" | None (auto-select)
    grid_points=101,       # candidates for the "grid" strategy, over (0, 1) exclusive
) -> dict[Any, float]
```

### `apply_thresholds` signature

```python
apply_thresholds(
    y_proba,               # (n_samples, n_classes) predicted probabilities
    thresholds,            # a single float (binary) or {class_label: threshold} dict
    classes=None,          # explicit class order; required for 3+ classes with a dict
) -> np.ndarray            # 1D array of predicted class labels
```

## How the search works

| Classes | Default strategy | Decision rule |
|---|---|---|
| Binary (2) | `grid` | positive class when `y_proba[:, 1] >= threshold`, else negative |
| Multiclass (3+) | `nelder-mead` | scaled argmax: `classes[argmax(y_proba / threshold_per_class)]` |

- **Binary** uses a grid search over `(0, 1)` (`grid_points` candidates,
  exclusive of the endpoints).
- **Multiclass** uses Nelder–Mead optimization over a scaled-argmax rule.
  Equal thresholds across all classes reduce to plain `argmax`, so the tuned
  result can only match or beat the default rule on the tuning metric.

Pass `strategy="grid"` or `strategy="nelder-mead"` explicitly to override the
auto-selection.

## Notes

- The `metric` is fully caller-supplied — Skyulf ships **no** default metric.
  Any callable `(y_true, y_pred) -> float` that returns a value to *maximize*
  works (wrap "lower is better" metrics accordingly).
- Always tune on data that is independent of both training and the final test
  set. See [Validation vs scikit-learn](validation_vs_sklearn.md) and
  [SplitDataset & Leakage](splitdataset_and_leakage.md).
