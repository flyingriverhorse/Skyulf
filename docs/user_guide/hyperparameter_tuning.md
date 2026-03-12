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
| `cv_type` | `str` | `"k_fold"` | One of `k_fold`, `stratified_k_fold`, `time_series_split`, `shuffle_split` |

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

## Tips

- Start with `"strategy": "random"` and `"n_trials": 20` for a quick baseline.
- Switch to `"optuna"` when you want smarter exploration (Bayesian optimization).
- Use `"halving_random"` for large search spaces where grid search is infeasible.
- Always run a `TrainTestSplitter` before tuning to avoid data leakage.
