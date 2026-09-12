# Modeling Nodes

This page documents modeling configuration for `SkyulfPipeline`.

To discover registered model IDs, call `NodeRegistry.list_models()`. It includes
both `Modeling` and `Ensemble` categories; pass `category="Ensemble"` for just
`voting_classifier`, `stacking_classifier`, `voting_regressor` and
`stacking_regressor`. Passing `category="Modeling"` returns the other models.
`NodeRegistry.list_transformers()` excludes both categories. These helpers
preserve registration order, and unknown category filters return an empty list.

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

### Seeding

Every `random_state=42` default listed below comes from one place:
`skyulf.types.DEFAULT_RANDOM_STATE`, injected at model construction. You can
override it per node with `"params": {"random_state": 7}`, or pass
`"random_state": null` to leave an estimator unseeded. In the canvas, the same
control appears as **Random State** under Hyperparameters → Customize (basic
mode), in the Tuning Strategy section (advanced mode), and as **Fold Split
Seed** in each node's Cross Validation section. Full precedence rules:
[Reproducibility and seeds](../user_guide/configuration.md#reproducibility-and-seeds).

## Classification

Binary evaluation omits the ROC curve and ROC AUC when the held-out partition
contains only one trained class, since ROC requires both classes. Finite
precision-recall output and the other available metrics remain in the report.

`class_weight` is applied consistently in direct training, every tuning
strategy and final refitting. Models such as GradientBoosting and XGBoost that
do not accept it natively receive equivalent per-row training weights.
`"balanced"` weights are computed from each fold's training labels after
preprocessing; validation labels do not influence them. Explicit weight
dictionaries use the labels passed to model fitting. Estimators that support
neither class weights nor sample weights reject an active weighting request.
Previously affected tuned models must be retrained to apply these weights;
existing saved models retain their predictions.

### LightGBM row sampling

The `lgbm_classifier` and `lgbm_regressor` nodes apply `subsample` during basic
training, hyperparameter search and the final refit. With GBDT or DART,
`subsample=0.4` uses 40% of the training rows per boosting iteration. The default
`subsample=1.0` uses all rows. GOSS keeps its gradient-based sampling instead of
ordinary row bagging, so this fraction does not control GOSS sampling.

Skyulf resolves its automatic `subsample_freq=None` to 1 for ordinary boosting
and 0 for GOSS after each candidate's parameters are applied. Explicit numeric
frequencies and LightGBM's `bagging_freq` alias are preserved; `subsample_freq=0`
disables row bagging. Explicit combinations that LightGBM rejects, such as GOSS
with ordinary bagging enabled, continue to raise an error.

Previously the default frequency was 0, so changing `subsample` had no effect.
Existing fitted models retain their predictions. Retrain or rerun tuning to
apply a previously ignored fraction. To reproduce the previous sampling policy
in a new fit, set `subsample_freq=0` explicitly.

### logistic_regression

Backed by `sklearn.linear_model.LogisticRegression`.

With `penalty="elasticnet"` and `solver="saga"`, an omitted or null
`l1_ratio` resolves to `0.5`. Explicit numeric ratios, including `0` and `1`,
are preserved. The same default applies during all five tuning strategies,
fold preprocessing and final refitting. Search results record the resolved
ratio. Search Elastic Net separately from other penalties when its ratio is
unspecified; such mixed searches now raise a clear error. Retrain affected
models or rerun tuning to replace an earlier unintended L2 fit.

Defaults:

- `max_iter=1000`
- `solver=lbfgs`
- `random_state=42`

Learned params:

- fitted sklearn estimator (stored in-memory and pickled when saving the pipeline)

### calibrated_classifier

Calibrates a selected base classifier using sigmoid or isotonic calibration.
Defaults are `base_estimator="logistic_regression"`, `method="sigmoid"` and
`cv=5`. Other base choices are `random_forest`, `gradient_boosting`,
`decision_tree`, `gaussian_nb` and `svc`.

In advanced mode, Base Estimator is a search choice. For example,
`"search_space": {"base_estimator": ["random_forest"], "method": ["isotonic"],
"cv": [3]}` tunes a calibrated random forest. Selecting multiple bases compares
those classifier families. All five search strategies preserve this choice
during trials and the final refit, including with preprocessing fitted inside
each fold. The winning parameters retain the selected base name in reports.

`random_state` controls randomness in base estimators that support it, including
the models fitted inside calibration folds. It defaults to 42, accepts 0, and
can be set to `None` (`null` in JSON) for unseeded fitting. It also applies during
tuning and the final model refit. Deterministic estimators such as Gaussian
Naive Bayes are unaffected. An integer `cv` retains unshuffled calibration
folds; this seed does not change those splits.

For example, `{"base_estimator": "random_forest", "random_state": 7}` now
seeds the calibrated forest with 7. Previously the forest always used 42.
Existing fitted models retain their predictions; retrain to apply the setting.

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

When base-model calibration is enabled, nested tuned keys address the completed
calibration wrapper. For example, `logistic_regression__estimator__C=0.01`
sets the underlying classifier's regularization, while
`logistic_regression__method="isotonic"` sets its calibration method. These
selected values remain in effect during calculator refits and post-tuning
cross-validation. Fixed `base_estimator_params` still configure the underlying
base model before calibration.

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

`pr_auc_weighted` matches probability columns to the fitted model's classes,
even when a validation partition omits a trained class. Multiclass `pr_auc`
also resolves to this scorer. Multiclass scores use support-weighted average
precision; binary `pr_auc_weighted` uses the model's second class as positive.
A binary holdout without positive examples therefore scores 0, rather than
changing which class is positive.

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
