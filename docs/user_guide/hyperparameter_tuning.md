# Hyperparameter Tuning

Skyulf wraps scikit-learn's search utilities and Optuna into a single `hyperparameter_tuner` model type. You configure tuning entirely through the pipeline config — no code changes required.

## Supported strategies

| Strategy | Key | What it does |
|---|---|---|
| Grid Search | `grid` | Exhaustive search over every combination in the search space |
| Random Search | `random` | Samples `n_trials` random combinations |
| Optuna (Bayesian) | `optuna` | Uses TPE (Tree-structured Parzen Estimators) to intelligently explore the space |
| Halving Grid | `halving_grid` | Successive halving — trains on small subsets first, promotes the best |
| Halving Random | `halving_random` | Random sampling + successive halving |

## Quick example

```python
config = {
    "preprocessing": [
        {"name": "split", "transformer": "TrainTestSplitter",
         "params": {"test_size": 0.25, "random_state": 42,
                    "stratify": True, "target_column": "target"}},
        {"name": "impute", "transformer": "SimpleImputer",
         "params": {"columns": ["age"], "strategy": "mean"}},
    ],
    "modeling": {
        "type": "hyperparameter_tuner",
        "base_model": {"type": "random_forest_classifier"},
        "strategy": "optuna",
        "search_space": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, "none"],
        },
        "n_trials": 25,
        "metric": "accuracy",
        "strategy_params": {
            "sampler": "tpe",
            "pruning": True,
        },
    },
}
```

> **Note:** `"none"` (string) in the search space is automatically converted to Python `None`.

## Configuration reference

These keys go inside the `"modeling"` block when `"type"` is `"hyperparameter_tuner"`:

| Key | Type | Default | Description |
|---|---|---|---|
| `base_model` | `dict` | *required* | The model to tune, e.g. `{"type": "logistic_regression"}` |
| `strategy` | `str` | `"random"` | One of `grid`, `random`, `optuna`, `halving_grid`, `halving_random` |
| `search_space` | `dict` | `{}` | Parameter name to list of candidate values |
| `n_trials` | `int` | `10` | Number of trials (ignored for `grid` which tests all combos) |
| `metric` | `str` | `"accuracy"` | Scoring metric (`accuracy`, `f1`, `roc_auc`, `mse`, `r2`, etc.) |
| `timeout` | `int\|null` | `null` | Max seconds for tuning (Optuna only) |
| `strategy_params` | `dict` | `{}` | Strategy-specific settings (see below) |
| `cv_enabled` | `bool` | `true` | Whether to use cross-validation |
| `cv_folds` | `int` | `5` | Number of CV folds |
| `cv_type` | `str` | `"k_fold"` | One of `k_fold`, `stratified_k_fold`, `time_series_split`, `shuffle_split`, `nested_cv` |
| `cv_time_column` | `str\|null` | `null` | Column name to sort by when using `time_series_split`. Auto-detects datetime column if omitted |

## Strategy-specific params

Pass these inside `"strategy_params"`:

### Optuna

| Key | Default | Description |
|---|---|---|
| `sampler` | `"tpe"` | Optuna sampler: `tpe`, `random`, or `cmaes` |
| `pruning` | `false` | Enable Optuna pruning (early stopping of bad trials) |

### Halving (grid / random)

| Key | Default | Description |
|---|---|---|
| `factor` | `3` | Successive halving factor (how aggressively to discard candidates) |
| `min_resources` | `"smallest"` | Minimum resources for the first iteration |

## Cross-validation types

| Key | When to use |
|---|---|
| `k_fold` | General purpose, default |
| `stratified_k_fold` | Classification with imbalanced classes |
| `time_series_split` | Time-ordered data (no future leakage) |
| `shuffle_split` | When you want random train/test splits per fold |
| `nested_cv` | Unbiased evaluation — outer loop for generalization, inner loop for hyperparameter stability |

> **Nested CV** runs a dual-loop: an outer K-Fold evaluates the model on held-out data, while an inner 3-fold CV (capped at `n_folds - 1`) trains within each outer training set. This prevents optimistic bias when tuning and evaluating on the same splits.
>
> **With Advanced Tuning:** When `nested_cv` is selected, the tuning search uses the inner CV folds to score candidates. After finding the best parameters, the post-tuning evaluation automatically uses `stratified_k_fold` (classification) or `k_fold` (regression) instead of re-running the full nested loop — because the inner loop already ran during the search.

### Time column for Time Series Split

When using `time_series_split`, Skyulf auto-sorts your data chronologically:

1. If `cv_time_column` is set, data is sorted by that column (and the column is dropped from features to prevent leakage).
2. If omitted, the first `datetime64` column is auto-detected and used.
3. If no datetime column exists, a warning is logged and row order is assumed correct.

In the ML Canvas UI, selecting Time Series Split reveals a date column picker.

## Install requirements

Optuna strategies require the tuning extra:

```bash
pip install skyulf-core[tuning]
```

Grid, random, and halving strategies work out of the box (scikit-learn only).

## What happens under the hood

1. `SkyulfPipeline` detects `"type": "hyperparameter_tuner"` and creates a `TuningCalculator` wrapping the base model.
2. During `fit()`, the `TuningCalculator.tune()` method runs the chosen search strategy.
3. The best parameters are used to refit the model on the full training set.
4. The fitted model is stored in the pipeline and used for `predict()` / `save()`.

## Notebook export — seeing tuning in action

When you export a canvas pipeline that includes an **Advanced Tuning** node, Skyulf generates a fully self-contained Jupyter notebook. The exported training cell mirrors exactly what the backend engine does:

- Wraps the base calculator/applier in `TuningCalculator` / `TuningApplier` so `fit_predict` actually runs the search (Optuna, grid, random, etc.) instead of a single default fit.
- Passes a `progress_callback` that prints `Trial N/M — score=0.9306` for each completed trial directly in the notebook output.
- Passes `log_callback=print` so the engine's own log lines (`Tuning Completed (optuna). Best Score: …`, `Best Params: {…}`, `Refitting best model…`) appear in the cell output too.

Example output you'll see while the cell runs:

```
Trial 1/10 — score=0.9221
Trial 2/10 — score=0.9306
...
Tuning Completed (optuna). Trials evaluated: 10. Best Score: 0.9497
Best Params: {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.05, ...}
Refitting best model with params: {...}
```

After `fit_predict` completes, the cell calls `estimator.evaluate()` and renders a coloured `train vs test` metrics DataFrame (accuracy, F1, ROC-AUC, PR-AUC, and more) using pandas Styler `background_gradient`.

### Using tuning directly (without the canvas)

If you want to run tuning in a plain Python script or existing notebook:

```python
from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling._tuning.engine import TuningApplier, TuningCalculator
from skyulf.registry import NodeRegistry

ALGORITHM = "xgboost_classifier"
base_calc = NodeRegistry.get_calculator(ALGORITHM)()
base_apply = NodeRegistry.get_applier(ALGORITHM)()

# Wrapping is required — without it, tuning_config is silently ignored.
tuner_calc = TuningCalculator(base_calc)
tuner_apply = TuningApplier(base_apply)
estimator = StatefulEstimator(node_id="my_tuner", calculator=tuner_calc, applier=tuner_apply)

dataset = SplitDataset(
    train=X_train.assign(**{"target": y_train}),
    test=X_test.assign(**{"target": y_test}),
    validation=None,
)

tuning_config = {
    "strategy": "optuna",
    "metric": "f1",
    "n_trials": 25,
    "search_space": {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
    },
    "cv_enabled": True,
    "cv_folds": 5,
    "cv_type": "stratified_k_fold",
    "random_state": 42,
}

def _on_trial(current, total, score=None, params=None):
    msg = f"Trial {current}/{total}"
    if score is not None:
        msg += f" — score={score:.4f}"
    print(msg)

_ = estimator.fit_predict(
    dataset=dataset,
    target_column="target",
    config=tuning_config,
    progress_callback=_on_trial,
    log_callback=print,
)

metrics = estimator.evaluate(dataset=dataset, target_column="target")
```

> **Important:** Always pass `TuningCalculator(base_calc)` — not the raw base calculator. The `TuningConfig` keyword filter inside `TuningCalculator.fit` reads `strategy`, `n_trials`, `metric`, `search_space`, `cv_*`, etc. directly from the config dict. If you pass the unwrapped base calculator, `tuning_config` is ignored and a single default-param fit runs instead.

## Tips

- Start with `"strategy": "random"` and `"n_trials": 20` for a quick baseline.
- Switch to `"optuna"` when you want smarter exploration (Bayesian optimization).
- Use `"halving_random"` for large search spaces where grid search is infeasible.
- Always run a `TrainTestSplitter` before tuning to avoid data leakage.
- When exporting to a notebook, use the **Export → Full** mode to get per-branch training cells with live trial output; the **Compact** mode runs all branches in a loop and shows a coloured `(branch × split)` comparison table at the end.
