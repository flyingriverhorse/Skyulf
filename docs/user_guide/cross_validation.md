# Cross-Validation

Skyulf supports five cross-validation strategies out of the box. CV can be used standalone via `StatefulEstimator.cross_validate()` or inside the hyperparameter tuning pipeline.

## Supported methods

| Method | Key | Splitter | Best for |
|---|---|---|---|
| K-Fold | `k_fold` | `sklearn.model_selection.KFold` | General purpose |
| Stratified K-Fold | `stratified_k_fold` | `sklearn.model_selection.StratifiedKFold` | Classification with imbalanced classes |
| Shuffle Split | `shuffle_split` | `sklearn.model_selection.ShuffleSplit` | Random repeated train/test splits |
| Time Series Split | `time_series_split` | `sklearn.model_selection.TimeSeriesSplit` | Temporal data (no future leakage) |
| Nested CV | `nested_cv` | Dual-loop (outer KFold/Stratified + inner KFold/Stratified) | Unbiased generalization estimate |

## Quick example (standalone)

```python
from skyulf.modeling.base import StatefulEstimator

estimator = StatefulEstimator(model_type="random_forest_classifier")
estimator.fit(X_train, y_train)

cv_results = estimator.cross_validate(
    X_train, y_train,
    n_folds=5,
    cv_type="stratified_k_fold",
)

print(cv_results["aggregated_metrics"])
# {'accuracy': {'mean': 0.92, 'std': 0.01, 'min': 0.90, 'max': 0.94}, ...}
```

## Quick example (pipeline config)

```python
config = {
    "preprocessing": [...],
    "modeling": {
        "type": "random_forest_classifier",
        "cv_enabled": True,
        "cv_type": "k_fold",
        "cv_folds": 5,
    },
}
```

## Method details

### K-Fold

Splits data into `n_folds` equal parts. Each fold is used once as validation while the remaining folds form the training set. Data is shuffled by default.

```python
cv_type="k_fold", n_folds=5, shuffle=True
```

### Stratified K-Fold

Same as K-Fold but preserves the class distribution in each fold. Automatically falls back to K-Fold for regression problems.

```python
cv_type="stratified_k_fold", n_folds=5
```

### Shuffle Split

Generates `n_folds` random train/test splits. Unlike K-Fold, the same sample can appear in the test set of multiple iterations. Uses a fixed 80/20 train/test ratio per split.

```python
cv_type="shuffle_split", n_folds=10
```

### Time Series Split

Expands the training window forward in time. Fold 1 trains on the first chunk and validates on the second; fold 2 trains on the first two chunks and validates on the third; and so on.

```python
cv_type="time_series_split", n_folds=5, time_column="order_date"
```

**Auto-sort behavior:**

1. If `time_column` is provided, data is sorted by that column and the column is dropped from features (prevents date leakage).
2. If omitted, the first `datetime64` column is auto-detected.
3. If no datetime column exists, a warning is logged and row order is assumed correct.

### Nested CV

Runs a dual-loop cross-validation:

- **Outer loop** (K-Fold or Stratified): evaluates generalization on held-out data.
- **Inner loop** (3-fold, capped at `n_folds - 1`): trains within each outer training set to check hyperparameter stability.

This prevents the optimistic bias that occurs when the same data is used for both tuning and evaluation.

```python
cv_type="nested_cv", n_folds=5
```

Each fold result includes an `inner_cv_mean` score for diagnostics.

## Return structure

All CV methods return the same dictionary shape:

```python
{
    "aggregated_metrics": {
        "accuracy": {"mean": 0.92, "std": 0.01, "min": 0.90, "max": 0.94},
        # ... other metrics
    },
    "folds": [
        {"fold": 1, "metrics": {...}},
        {"fold": 2, "metrics": {...}},
        # ...
    ],
    "cv_config": {
        "n_folds": 5,
        "cv_type": "k_fold",
        "shuffle": True,
        "random_state": 42,
    },
}
```

Nested CV adds `inner_cv_mean` to each fold entry and `inner_folds` to `cv_config`.

## Configuration reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cv_type` | `str` | `"k_fold"` | One of `k_fold`, `stratified_k_fold`, `time_series_split`, `shuffle_split`, `nested_cv` |
| `n_folds` / `cv_folds` | `int` | `5` | Number of CV folds |
| `shuffle` | `bool` | `True` | Shuffle data before splitting (K-Fold / Stratified only) |
| `random_state` | `int` | `42` | Random seed |
| `time_column` / `cv_time_column` | `str\|null` | `null` | Column for chronological sorting (Time Series Split) |

## ML Canvas UI

Both the **Basic Training** and **Advanced Tuning** nodes expose a CV type dropdown with all five methods. When Time Series Split is selected, an additional date column picker appears with a warning to verify the selected column.

## Integration with tuning

When using `hyperparameter_tuner`, the CV splitter is passed directly to the search strategy (GridSearchCV, RandomizedSearchCV, etc.). For nested CV in tuning mode, the inner folds drive the search while the outer folds provide an unbiased evaluation estimate.

See [Hyperparameter Tuning](hyperparameter_tuning.md) for tuning-specific configuration.
