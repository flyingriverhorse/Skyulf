# Threshold Tuning

`predict()`'s default decision rule — `argmax` for multiclass, `0.5` for
binary — is rarely optimal for imbalanced classes or for a metric you
actually care about (F1, MCC, balanced accuracy, a custom business metric, …).
Skyulf can search for **per-class decision thresholds** that maximize a metric
you supply, evaluated on held-out validation data.

The functionality is available at three levels:

- **Pipeline level:** `SkyulfPipeline.optimize_thresholds()` +
  `predict(use_tuned_thresholds=True)` — handles preprocessing and
  `predict_proba` for you.
- **Array level:** `skyulf.modeling.optimize_thresholds()` /
  `apply_thresholds()` — plain NumPy in/out, for use outside a pipeline (e.g.
  with a raw sklearn estimator).
- **Tuning-engine integration:** `TuningConfig(tune_threshold=True)` — after
  hyperparameter tuning, the engine selects the threshold on the validation
  split and applies it in every subsequent prediction automatically (binary
  classifiers only). See [Integrated with hyperparameter tuning](#integrated-with-hyperparameter-tuning).

## After training in the canvas

You can tune a completed classifier from **Experiments → Model Evaluation →
Threshold Tuning**, even if **Tune decision threshold** was off during training.
The evaluation must include class probabilities.

Available objectives are **Accuracy**, **F1**, **Precision**, **Recall**, and
**Balanced Accuracy**. ROC AUC measures probability ranking, which a decision
threshold does not change, so it is not a threshold-tuning objective. Choose
Balanced Accuracy to balance recall across classes. ROC AUC remains available
for model evaluation and hyperparameter search.

1. Choose a metric and click **Preview** to calculate thresholds. This does not
   save them or change predictions.
2. Click **Save** to persist the thresholds and enable them for predictions.
3. Use **Use tuned thresholds at prediction time** to turn the saved set off
   or on. The switch stays disabled until a saved set exists.
4. **Clear** deletes the saved set and restores the default prediction rule.

Preview uses validation data when available, otherwise the test split, as shown
in the panel. No retraining is required. If training already saved thresholds,
the panel loads that set and its enabled state automatically.

Previously saved thresholds marked `roc_auc` remain available to use, toggle,
or clear. Their cutoffs are preserved. To replace them, choose a supported
objective and run **Preview**, then **Save**. Save stays disabled for a loaded
set whose metric is unavailable for new tuning until a supported preview succeeds.
New preview/save API requests
using `roc_auc` return HTTP 400 with a supported alternative.

## Pipeline usage

Tune against explicit, out-of-sample validation data — **never** the
pipeline's internal train/test split. Carve the holdout out of the **raw**
data before fitting, and pass those raw rows in:

```python
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Raw holdout, carved before fit(): optimize_thresholds() preprocesses it itself.
train_raw, val_raw = train_test_split(customers, test_size=0.2, random_state=42)
X_val = val_raw.drop(columns=["purchased"])
y_val = val_raw["purchased"]

pipeline.fit(train_raw, target_column="purchased")

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
`(X_val, y_val)` internally — exactly once, the same single pass `predict()`
makes on the features — then calls the model's `predict_proba`, searches
thresholds, and stores the result on the pipeline. Labels follow any row sorting
or filtering, so each probability is scored against its matching target. Raw
features and labels must have equal lengths and correspond positionally.
That one pass is why `X_val` has to be raw: it is what
makes the probabilities the cutoffs are fitted against the same probabilities
`predict(use_tuned_thresholds=True)` later reproduces.
`predict(use_tuned_thresholds=True)` then applies those stored thresholds; it
raises a `ValueError` if `optimize_thresholds()` was never called, and requires
a model that supports `predict_proba`.

**Do not tune on [`get_fitted_split()`](../reference/api/pipeline.md) output.**
That helper returns frames the preprocessing chain has **already transformed**;
it exists so you can compare raw sklearn/XGBoost/CatBoost estimators against
the exact split the pipeline would use. Handing its `X_test` to
`optimize_thresholds()` preprocesses the holdout a *second* time, so the search
fits cutoffs against a distribution inference never reproduces. Nothing errors
— the returned dict looks plausible, and the metric can even score *better* on
the doubly-transformed fold, which is why the mismatch stays invisible.

If your config already contains a `TrainTestSplitter`, fitting on `train_raw`
splits it a second time internally. That is fine, and it is what keeps the
model off your tuning data: the model trains on the splitter's train part, and
`val_raw` is untouched by both.

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

## Integrated with hyperparameter tuning

When a classifier is trained through the tuning engine — a canvas Training node
in Advanced mode, the `"hyperparameter_tuner"` pipeline model type, or
`TuningCalculator` directly — set `tune_threshold: true` to have the engine
pick the decision threshold right after the final refit. It grid-searches the
positive-class cutoff that maximises your tuning metric on the validation
split, and every subsequent prediction uses that cutoff instead of the default
`0.5`:

```python
config = {
    "modeling": {
        "type": "hyperparameter_tuner",
        "base_model": {"type": "logistic_regression"},
        "strategy": "random",
        "search_space": {"C": [0.1, 1.0, 10.0]},
        "metric": "f1",
        "tune_threshold": True,
    },
}
```

In the canvas, this is the **Tune decision threshold** checkbox in the
Training node's Tuning Strategy section (classification models, Advanced
mode).

The selected threshold is stored on the tuning result
(`decision_thresholds`, plus `decision_threshold_metric` naming the metric the
sweep maximised) and surfaces in the job metrics; the model's predictions
apply it automatically — there is no separate "use tuned thresholds" flag at
this level.

When the job runs through the backend (canvas Training node), the selected
threshold is also seeded into the job's saved-threshold store — the same one
the Experiments / Inference **Threshold Tuning** panel reads — and enabled by
default. A threshold tuned at training time therefore shows up in the panel
("Optimized for …, computed from validation split", labelled *seeded at
training*) and is active for deployed predictions immediately, and the usual
panel controls apply: toggle it off, replace it with a manual preview, or
clear it. Predict-time precedence is unchanged: request-level override >
saved + enabled thresholds > default decision rule.

**Gates** — the engine logs a skip in the job log and predictions keep the
default decision rule when any of these don't hold:

- the model is a **classifier** exposing `predict_proba`;
- the target is **binary** (for multiclass, use the pipeline- or array-level
  APIs above);
- a **validation split** exists (configure a train/validation split upstream);
- the validation split contains **both classes** — with one class present every
  candidate cutoff scores identically, so there is nothing to tune against.

Probability-only metrics (`roc_auc`, `log_loss`, `pr_auc`, …) cannot be
computed from hard labels, so the cutoff sweep maximises `balanced_accuracy`
instead — the log says so, and `decision_threshold_metric` records it.
Threshold tuning is best-effort: an error during the sweep never aborts the
tuning run.

## How the search works

| Classes | Default strategy | Decision rule |
|---|---|---|
| Binary (2) | `grid` | positive class when `y_proba[:, 1] >= threshold`, else negative |
| Multiclass (3+) | `nelder-mead` | scaled argmax: `classes[argmax(y_proba / threshold_per_class)]` |

- **Binary** uses a grid search over `(0, 1)` (`grid_points` candidates,
  exclusive of the endpoints). Metrics like F1 are piecewise constant in the
  cutoff, so exactly-tied plateaus are common; ties break toward the default
  `0.5` cut. Applying the returned per-class thresholds reproduces the same
  `>=` rule the search scored with, an exact tie included. When *every*
  candidate ties — a validation split holding one class, probabilities so
  saturated that no cutoff straddles a boundary, or a metric that ignores its
  predictions — there is no signal to tune against: the search keeps the default
  `0.5` cut and logs a warning saying so, rather than silently returning cutoffs
  indistinguishable from a genuine optimum.
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
