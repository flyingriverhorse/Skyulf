# Cross-Validation

Skyulf supports random, chronological and group-aware cross-validation. CV can be used standalone via `StatefulEstimator.cross_validate()` or inside the hyperparameter tuning pipeline.

## Supported methods

| Method | Key | Splitter | Best for |
|---|---|---|---|
| K-Fold | `k_fold` | `sklearn.model_selection.KFold` | General purpose |
| Stratified K-Fold | `stratified_k_fold` | `sklearn.model_selection.StratifiedKFold` | Classification with imbalanced classes |
| Shuffle Split | `shuffle_split` | `sklearn.model_selection.ShuffleSplit` | Random repeated train/test splits |
| Time Series Split | `time_series_split` | `sklearn.model_selection.TimeSeriesSplit` | Temporal data (no future leakage) |
| Group K-Fold | `group_k_fold` | Whole groups stay together | Repeated customers, patients or devices |
| Stratified Group K-Fold | `stratified_group_k_fold` | Group isolation with class balancing | Grouped classification |
| Nested CV | `nested_cv` | Separate inner selection and outer evaluation | Evaluate the tuning procedure |

## Quick example (standalone)

```python
from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator
from skyulf.registry import NodeRegistry

# training_frame contains features and the target column "label".
estimator = StatefulEstimator(
    calculator=NodeRegistry.get_calculator("random_forest_classifier")(),
    applier=NodeRegistry.get_applier("random_forest_classifier")(),
    node_id="classifier",
)
cv_results = estimator.cross_validate(
    SplitDataset(train=training_frame, test=training_frame.iloc[:0]),
    target_column="label",
    config={"params": {"n_estimators": 50}},
    n_folds=5,
    cv_type="stratified_k_fold",
)
print(cv_results["aggregated_metrics"])
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
2. If omitted, the first supported date/datetime column in input order is used,
   including pandas object columns of native `datetime.date` values and Polars
   Date/Datetime columns. Native date columns may contain missing values.
3. If no datetime column exists, a warning is logged and row order is assumed correct.

Automatic detection does not parse date strings or select mixed date/text,
mixed date/datetime object columns, or entirely missing object columns. Normalize
such inputs explicitly before temporal CV. Sorting remains stable, places
missing dates last and keeps targets aligned. Rerun temporal CV on affected
native pandas date inputs; previous scores may have used nonchronological folds.

### Group policies

Choose `group_k_fold` or, for classification, `stratified_group_k_fold` and
provide `cv_group_column`. Each entity stays on one side of every split. The
column is excluded from learned features. Missing group identifiers,
insufficient groups and incomplete class coverage fail before fitting.
Reserved test/validation partitions must also have groups absent from training.

### Nested CV

With tuning enabled, `cv_type="nested_cv"` runs these stages:

1. Reserve one outer fold for evaluation.
2. Search candidate parameters using only inner folds of the outer training rows.
   Every learned preprocessing step is fitted again inside each training fold.
3. Refit that fold's winning recipe and score the untouched outer fold.
4. Repeat for every outer fold; report the mean and spread of outer scores.
5. Run a separate search on all training rows and fit the deployable model.
   The final test/validation holdout does not choose parameters or thresholds.

`cv_folds` sets the outer count. `cv_inner_folds` sets the inner count; when
omitted it is `min(3, cv_folds - 1)`, with two inner folds for two outer folds.
Trial limits and timeouts apply to each search, including the final search.

`cv_nested_type="auto"` uses stratified folds for classification and K-Fold for
regression. Explicit options are `k_fold`, `stratified_k_fold`,
`time_series_split`, `group_k_fold` and `stratified_group_k_fold`.

#### Temporal nested CV

Set `cv_nested_type="time_series_split"`, `cv_time_column` and
`cv_shuffle=False`. Both levels sort stably by the clock and train before their
validation rows. The final reserved holdout must follow all training timestamps.
Missing timestamps and equal timestamps crossing a fold boundary are rejected.
The clock is excluded from model features; prediction requests retain their row order.

- `cv_gap`: excluded rows between each training and validation partition (default 0).
- `cv_test_size`: validation rows per fold; `None` selects automatic sizing.
- `cv_max_train_size`: maximum training rows; `None` expands the window, a positive
  value creates a rolling window.

These settings use row counts, not durations. They apply to both nested levels
and the separate final search. The earliest outer training partition must be
large enough for the requested inner windows. Ordinary temporal CV with explicit
window settings uses the same strict metadata checks.

#### Threshold selection

For binary classification, `tune_threshold=True` collects fresh inner
out-of-fold probabilities for the selected recipe. The threshold is chosen from
those training-only predictions and applied to the outer fold's hard decisions.
ROC-AUC and other probability/ranking metrics retain the original probabilities.
A separate threshold is selected from final-search OOF predictions for the saved
model. Temporal warmup rows without OOF predictions are excluded and counted.

This also works with eligible voting/stacking classifiers. The estimator must
expose `predict_proba`; regression, multiclass and hard-voting thresholds are
rejected. Target encoding must preserve an identifiable raw-to-model class mapping.

## Results

Ordinary fixed-model CV returns `aggregated_metrics`, `folds` and `cv_config`.
Tuning stores its nested report under `TuningResult.nested_cv`, including:

- `mean_score` / `std_score`: outer evaluation scores, separate from final search score.
- `folds`: selected parameters, inner best score and outer score for each fold.
- `split_policy`, `split`, `inner_splits`, `final_splits`: effective settings and
  timestamp boundaries or group counts/membership hashes.
- `threshold_selection`: per-outer-fold and final OOF threshold evidence when enabled.

Search scores use sklearn's higher-is-better convention; losses remain negative.
A fixed-model evaluation does not invent a search space: explicit nested policies
use a single candidate, while the legacy standalone automatic nested method
retains its fixed-parameter inner diagnostic.

## Configuration across interfaces

| Concept | Core tuning / Canvas payload | Standalone `cross_validate` | Databricks workflow |
|---|---|---|---|
| Method | `cv_type` | `cv_type` | `cv_type` |
| Outer folds | `cv_folds` | `n_folds` | `cv_folds` |
| Inner folds | `cv_inner_folds` | `inner_folds` | `cv_inner_folds` |
| Nested policy | `cv_nested_type` | `cv_nested_type` | `cv_nested_type` |
| Group identifier | `cv_group_column` | `group_column` | `cv_group_column` |
| Clock | `cv_time_column` | `time_column` | `event_column` |
| Gap / test / window | `cv_gap`, `cv_test_size`, `cv_max_train_size` | `gap`, `test_size`, `max_train_size` | Same `cv_*` fields |
| Shuffle / seed | `cv_shuffle`, `cv_random_state` | `shuffle`, `random_state` | Same `cv_*` fields |

## ML Canvas and Databricks

Classification, regression and ensemble settings expose **Method**, with
**Outer folds**, **Inner folds** and **Nested split policy** for Nested CV.
Temporal policies show clock/gap/window controls; group policies require an
identifier column. Shuffle is disabled for chronological splits. For K-Fold,
stratified and group policies, **Fold Split Seed** controls reproducible shuffling
when shuffle is enabled. The seed is not specific to nested CV.

The backend carries raw split metadata through preprocessing and checks reserved
holdout isolation. Basic mode evaluates fixed parameters; Advanced mode searches
inside each outer fold. Results display the saved policy and threshold evidence.

Databricks uses the same Core search and split policies. Configure group-isolated
or temporal final holdouts in the workflow. See [Databricks Bundle](databricks_bundle.md)
for generated settings and examples, and [Hyperparameter Tuning](hyperparameter_tuning.md)
for strategy controls.
