# Modeling Nodes

This page documents modeling configuration for `SkyulfPipeline`.

> **Note:** The canvas used to offer separate "Basic Training" and "Advanced Tuning" nodes. These have been superseded by four task-scoped nodes — **Classification**, **Regression**, **Text Classification**, and **Segmentation** — each with a `run_mode: "basic" | "advanced"` toggle (Segmentation has no toggle; clustering is always direct-fit). The old node types still open in previously-saved canvases for backward compatibility, but are no longer offered when building new pipelines.

## Common config shape

`SkyulfPipeline` expects a modeling block like:

```python
{
  "type": "logistic_regression",
  "node_id": "model_node",       # optional
  "execution_mode": "merge",     # optional; "merge" (default) or "parallel"
  "params": { ... }              # optional; estimator hyperparameters
}
```

### execution_mode *(v0.4.0+)*

When a training node has 2+ incoming connections:

| Value | Behavior |
|---|---|
| `"merge"` (default) | Combine all upstream DataFrames into one before training |
| `"parallel"` | Each incoming branch runs as a separate job |

This field is set via the Merge/Parallel toggle on training nodes in the canvas UI.

The sklearn wrapper supports both:

- Nested params (preferred): `{ "params": {"C": 1.0} }`
- Flat params (legacy): `{ "C": 1.0, "type": "..." }`

Example (RandomForestClassifier):

```python
{
  "type": "random_forest_classifier",
  "params": {"n_estimators": 50, "random_state": 42}
}
```

## Classification

### logistic_regression

Backed by `sklearn.linear_model.LogisticRegression`.

Defaults:

- `max_iter=1000`
- `solver=lbfgs`
- `random_state=42`

Learned params:

- fitted sklearn estimator (stored in-memory and pickled when saving the pipeline)

### random_forest_classifier

Backed by `sklearn.ensemble.RandomForestClassifier`.

Defaults include:

- `n_estimators=50`, `max_depth=10`
- `min_samples_split=5`, `min_samples_leaf=2`
- `n_jobs=-1`, `random_state=42`

Learned params:

- fitted sklearn estimator

### svc

Backed by `sklearn.svm.SVC`.

Defaults:
- `C=1.0`, `kernel=rbf`, `gamma=scale`
- `probability=True`, `random_state=42`

### k_neighbors_classifier

Backed by `sklearn.neighbors.KNeighborsClassifier`.

Defaults:
- `n_neighbors=5`, `weights=uniform`
- `algorithm=auto`, `n_jobs=-1`

### decision_tree_classifier

Backed by `sklearn.tree.DecisionTreeClassifier`.

Defaults:
- `max_depth=None`, `min_samples_split=2`
- `criterion=gini`, `random_state=42`

### gradient_boosting_classifier

Backed by `sklearn.ensemble.GradientBoostingClassifier`.

Defaults:
- `n_estimators=100`, `learning_rate=0.1`
- `max_depth=3`, `random_state=42`

### adaboost_classifier

Backed by `sklearn.ensemble.AdaBoostClassifier`.

Defaults:
- `n_estimators=50`, `learning_rate=1.0`
- `random_state=42`

### xgboost_classifier

Backed by `xgboost.XGBClassifier`.

Defaults:
- `n_estimators=100`, `max_depth=6`
- `learning_rate=0.3`, `n_jobs=-1`
- `random_state=42`

### gaussian_nb

Backed by `sklearn.naive_bayes.GaussianNB`.

Defaults:
- `var_smoothing=1e-9`

## Regression

### ridge_regression

Backed by `sklearn.linear_model.Ridge`.

Defaults:

- `alpha=1.0`, `solver=auto`, `random_state=42`

### lasso_regression

Backed by `sklearn.linear_model.Lasso`.

Defaults:
- `alpha=1.0`, `selection=cyclic`
- `random_state=42`

### elasticnet_regression

Backed by `sklearn.linear_model.ElasticNet`.

Defaults:
- `alpha=1.0`, `l1_ratio=0.5`
- `selection=cyclic`, `random_state=42`

### random_forest_regressor

Backed by `sklearn.ensemble.RandomForestRegressor`.

Defaults include:

- `n_estimators=50`, `max_depth=10`
- `min_samples_split=5`, `min_samples_leaf=2`
- `n_jobs=-1`, `random_state=42`

### svr

Backed by `sklearn.svm.SVR`.

Defaults:
- `C=1.0`, `kernel=rbf`, `gamma=scale`

### k_neighbors_regressor

Backed by `sklearn.neighbors.KNeighborsRegressor`.

Defaults:
- `n_neighbors=5`, `weights=uniform`
- `algorithm=auto`, `n_jobs=-1`

### decision_tree_regressor

Backed by `sklearn.tree.DecisionTreeRegressor`.

Defaults:
- `max_depth=None`, `min_samples_split=2`
- `criterion=squared_error`, `random_state=42`

### gradient_boosting_regressor

Backed by `sklearn.ensemble.GradientBoostingRegressor`.

Defaults:
- `n_estimators=100`, `learning_rate=0.1`
- `max_depth=3`, `random_state=42`

### adaboost_regressor

Backed by `sklearn.ensemble.AdaBoostRegressor`.

Defaults:
- `n_estimators=50`, `learning_rate=1.0`
- `random_state=42`

### xgboost_regressor

Backed by `xgboost.XGBRegressor`.

Defaults:
- `n_estimators=100`, `max_depth=6`
- `learning_rate=0.3`, `n_jobs=-1`
- `random_state=42`

## Ensemble Meta-Models (v0.6.0)

Ensemble meta-models combine multiple base estimators to construct stronger predictive models under a unified interface. You can use them either programmatically in `skyulf-core` or directly on the canvas through the **Ensemble Node**.

### Registered Ensemble Families

| Step Registry ID | scikit-learn Class | Task |
|---|---|---|
| `voting_classifier` | `sklearn.ensemble.VotingClassifier` | Classification |
| `stacking_classifier` | `sklearn.ensemble.StackingClassifier` | Classification |
| `voting_regressor` | `sklearn.ensemble.VotingRegressor` | Regression |
| `stacking_regressor` | `sklearn.ensemble.StackingRegressor` | Regression |

### Core Configuration Parameters

Configuration is structured within the nested `params` dictionary of the modeling config (or `tuning_config` when running an advanced search):

| Key | Type | Applies To | Description |
|---|---|---|---|
| `base_estimators` | `List[str]` | All | Identifiers of base learners to combine (see lists below). |
| `voting` | `str` | Voting Classifier | `"soft"` (mean of predicted probabilities — default) or `"hard"` (majority label vote). |
| `final_estimator` | `str` | Stacking | The meta-learner trained on out-of-fold base predictions. Defaults to `logistic_regression` (clf) / `ridge` (reg). |
| `cv` | `int` | Stacking | Internal CV folds used to generate the out-of-fold base predictions. Default `5`. |
| `base_estimator_params` | `Dict[str, Dict]` | All | Fixed per-base-model hyperparameters (basic mode). |
| `final_estimator_params` | `Dict` | Stacking | Fixed hyperparameters for the meta-learner. |

**Supported base models — Classification:**
`logistic_regression`, `random_forest`, `extra_trees`, `gradient_boosting`, `hist_gradient_boosting`, `adaboost`, `decision_tree`, `gaussian_nb`, `svc` (probability-enabled), `knn` — plus `xgboost` / `lightgbm` when those optional wheels are installed.

**Supported base models — Regression:**
`linear_regression`, `ridge`, `lasso`, `elasticnet`, `random_forest`, `extra_trees`, `gradient_boosting`, `hist_gradient_boosting`, `adaboost`, `decision_tree`, `svr`, `knn` — plus `xgboost` / `lightgbm` when installed.

> **Cross-validation semantics:** Voting does *no* internal CV (each base model is fit once, then predictions are averaged/voted). Stacking *requires* an internal `cv` so the meta-learner trains on out-of-fold predictions — otherwise it over-fits on in-sample predictions.

### Python Example — Programmatic Usage in `skyulf-core`

Ensemble nodes are registered like any other modeling step, so they slot into the `modeling` block of a `SkyulfPipeline` config:

```python
import pandas as pd
from skyulf import SkyulfPipeline

config = {
    "preprocessing": [
        {
            "name": "split",
            "transformer": "TrainTestSplitter",
            "params": {"test_size": 0.2, "random_state": 42, "target_column": "target"},
        },
    ],
    "modeling": {
        "type": "stacking_classifier",
        "params": {
            "base_estimators": ["random_forest", "logistic_regression", "gradient_boosting"],
            "final_estimator": "logistic_regression",
            "cv": 5,
            # Fixed per-base-model hyperparameters (basic mode)
            "base_estimator_params": {
                "random_forest": {"n_estimators": 100, "max_depth": 12},
                "logistic_regression": {"C": 0.5},
            },
        },
    },
}

pipeline = SkyulfPipeline(config)
metrics = pipeline.fit(df, target_column="target")   # learns on the train split
predictions = pipeline.predict(new_df)               # feature-only dataframe
```

A **VotingClassifier** is configured the same way — swap the `type` for `voting_classifier` and add `"voting": "soft"` (or `"hard"`):

```python
"modeling": {
    "type": "voting_classifier",
    "params": {
        "base_estimators": ["random_forest", "svc", "knn"],
        "voting": "soft",
    },
}
```

You can also drive the underlying calculator/applier directly for low-level usage via the `NodeRegistry`:

```python
from skyulf import NodeRegistry

calc = NodeRegistry.get_calculator("stacking_classifier")()
applier = NodeRegistry.get_applier("stacking_classifier")()

# `fit(X, y, config)` returns the fitted sklearn meta-estimator
model = calc.fit(
    X_train,
    y_train,
    {
        "base_estimators": ["random_forest", "decision_tree"],
        "final_estimator": "logistic_regression",
        "cv": 3,
    },
)

# `predict(X, model)` / `predict_proba(X, model)` generate predictions
preds = applier.predict(X_test, model)
```

### Advanced Hyperparameter Tuning (Nested Parameters)

When an ensemble runs in **Advanced/Tuning mode** (`run_mode: "advanced"`), it is routed through the same hyperparameter search engine as a normal model. Set `tune_base_models: true` to auto-expand the search space into per-base-model dimensions using sklearn's double-underscore syntax (e.g. `random_forest__n_estimators`, `logistic_regression__C`). The search then optimizes the meta-estimator's own params (voting type, stacking `cv`) *and* each base learner simultaneously.

- Recommended outer search strategies: `optuna` or `halving_random`.
- **Cost warning:** Stacking `cv` × outer search = nested cross-validation (outer folds × stacking `cv` × trials × base models). Keep stacking `cv` small (e.g. `3`) or reduce trials when also running an outer search.

### Merge Strategy & Canvas Wiring

The **Ensemble Node** behaves differently from ordinary fan-in on the canvas:

**1. Merge strategy — are same-branch models taken as ensemble members?**

Yes. Unlike normal nodes (where multiple inputs trigger a column merge or a parallel-experiment split), the Ensemble Node classifies its incoming edges by source type:

- **One dataset edge** (e.g. a `train_test_split` output) supplies the rows/columns the ensemble trains on.
- **N model-spec edges** — any **Classification / Regression / Text Classification** node (or a legacy `Basic Training`/`Advanced Training` node from an older saved canvas) wired in is treated as a *base-learner specification*, not data. Only its recipe (`model_type` + hyperparameters) is read; its fitted weights are discarded because sklearn's Voting/Stacking always refit base learners on the composite dataset anyway.

So **models from the same branch are automatically adopted as ensemble members**. If no direct dataset edge exists (the common `split → model → ensemble` flow), the ensemble *inherits* the dataset its wired models consume and refits everything on that single dataset.

If you wire in a model trained on a **different dataset lineage**, the canvas raises a cross-dataset warning before committing the edge — mixing unrelated branches is almost always a wiring mistake.

**2. Manual dropdowns to choose models / strategy**

The settings panel exposes manual pickers so you don't have to wire nodes physically:

- **Base Models** — a multi-select chip list to add/remove each base learner.
- **Final Estimator** — a dropdown (Stacking only) to pick the meta-learner.
- **Voting type** — soft/hard toggle (Voting classifier).
- **Search Strategy** — a dropdown (`random`, `grid`, `optuna`, `halving_random`, `halving_grid`) with a gear button opening the per-strategy settings modal.

Wired model nodes *override* the chip selection; if no models are wired, the manual chips are used.

**3. How do wired models' search spaces work — automatically?**

Automatic. When a wired node's `run_mode` is `"advanced"`, the converter (`pipelineConverter.ts`):

- reads its `model_type` and adds the resolved base key to `base_estimators`,
- extracts its active `search_space` / `hyperparameters` and nests them under `base_estimator_params` (namespaced as `<base_key>__<param>`),
- forwards that nested space to the backend, where the Optuna/halving engine expands and optimizes all wired estimators together — no manual search-space entry required.

## Hyperparameter tuning

### hyperparameter_tuner

This mode wraps a base model and performs search.

Config:

- `type`: `hyperparameter_tuner`
- `base_model`: dict with a supported base model type (e.g., logistic regression)
- tuning options such as:
  - `strategy`: `grid` | `random` | `halving_grid` | `halving_random` | `optuna` (availability depends on installed packages)
  - `search_space`: dict of parameter → list/range
  - `metric`: e.g., `accuracy`, `f1`, `roc_auc`, `rmse`, `r2`
  - `cv_enabled`, `cv_type`, `cv_folds`, `random_state`

Learned params:

- a tuple `(best_model, tuning_result)` where `best_model` is a fitted estimator.

## Cross-validation

`StatefulEstimator.cross_validate()` can perform CV on the train split and returns aggregated fold metrics.

Five CV methods are supported:

| Key | Strategy | Notes |
|---|---|---|
| `k_fold` | K-Fold | Default. Shuffled. |
| `stratified_k_fold` | Stratified K-Fold | Preserves class distribution (classification). Falls back to K-Fold for regression. |
| `shuffle_split` | Shuffle Split | Random 80/20 splits; samples may repeat across folds. |
| `time_series_split` | Time Series Split | Expanding window. Auto-sorts by datetime column if `cv_time_column` is set. |
| `nested_cv` | Nested CV | Outer loop evaluates generalization; inner 3-fold loop checks HP stability. With advanced tuning, post-tuning eval auto-downgrades to `stratified_k_fold`/`k_fold` since the inner loop already ran during the search. |

Config keys: `cv_enabled`, `cv_type`, `cv_folds`, `cv_time_column`.

See the [Cross-Validation Guide](../user_guide/cross_validation.md) for details.

Note: `SkyulfPipeline` performs modeling through the same building blocks (a calculator + applier); `StatefulEstimator`
is the lightweight wrapper exposed for low-level usage.
