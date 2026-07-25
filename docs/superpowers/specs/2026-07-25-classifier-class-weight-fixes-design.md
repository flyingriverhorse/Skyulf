# skyulf-core: class_weight fixes, tuning-engine warning, and hyperparameter UI gaps

## Context

While building a real classification project (Kaggle "Predicting Student Health
Risk", Playground Series S6E7) entirely on `skyulf-core`, `class_weight="balanced"`
turned out to be the single biggest lever for balanced-accuracy on the severely
imbalanced (86/8/6%) target — worth roughly +8pp over an unweighted baseline.
That hands-on use surfaced several real gaps in the library:

1. **XGBoost silently ignores `class_weight`.** `XGBClassifier`'s constructor
   accepts arbitrary `**kwargs`, so skyulf-core's generic
   `SklearnCalculator._filter_supported_params()` (which normally strips
   unsupported constructor params) skips filtering entirely for XGBoost and
   lets `class_weight="balanced"` through. XGBoost stores it, does nothing
   with it, and only emits a native C++ warning at `fit()` time
   (`Parameters: {"class_weight"} are not used.`). Confirmed by direct
   reproduction. Today a user configuring `xgboost_classifier` with
   `class_weight: "balanced"` gets a silent no-op, not an error and not the
   requested behavior.
2. **`class_weight` is not exposed anywhere in the hyperparameter UI schema**,
   for any classifier — not Random Forest or LightGBM (which *do* support it
   natively today) and not XGBoost. The frontend's "Basic mode" hyperparameter
   panel (`TrainingSettings.tsx`) is schema-driven, fetching field
   definitions from the backend via `jobsApi.getHyperparameters(modelType)`,
   which resolves to `skyulf.modeling.hyperparameters._registry` field lists
   (`RANDOM_FOREST_CLASSIFIER_PARAMS`, `LGBM_PARAMS`, `XGBOOST_PARAMS`). None
   of these lists include a `class_weight` field, so the single most
   impactful imbalance-handling lever is reachable only by hand-authoring raw
   JSON pipeline configs — invisible in the UI.
3. **`skyulf/modeling/_tuning/engine.py` eagerly imports `optuna_integration`
   at module import time**, regardless of whether the tuner or Optuna
   strategy is ever used. When `optuna` is installed without
   `optuna-integration` (as on Kaggle's default Docker image), this prints a
   confusing warning (`Optuna installed but OptunaSearchCV not found. Install
   'optuna-integration'.`) merely from importing `skyulf.modeling`, even for
   users who never touch the tuner.
4. **No first-class way to extract a pipeline's fitted train/test split** for
   building a custom evaluation harness outside `SkyulfPipeline.fit()`/
   `.predict()`. We needed this to build a leakage-safe pseudo-labeling
   evaluation and ended up reverse-engineering `FeatureEngineer.fit_transform()`
   internals ourselves. Lower priority than 1–3, included as a small
   convenience addition.

### Corrected-during-investigation (not real issues, no action needed)

- `hyperparameter_tuner` **is** available and functional in the published
  PyPI `skyulf-core==0.5.3` — verified by direct execution. It is
  special-cased in `SkyulfPipeline._init_model_estimator()` rather than
  registered via `NodeRegistry`, which is why `NodeRegistry.get_all_metadata()`
  omits it and earlier gave a false impression it was dev-only. No release
  action needed for this.
- The apparent mismatch between ensemble `base_estimators` keys (`"xgboost"`,
  `"lightgbm"`, `"random_forest"`) and `NodeRegistry` model ids
  (`"xgboost_classifier"`, `"lgbm_classifier"`, `"random_forest_classifier"`)
  is already bridged by an explicit alias map in
  `skyulf/modeling/hyperparameters/_registry.py`, mirrored by the frontend's
  own alias map in `EnsembleSettings.tsx`. Deliberate, working design — no
  change needed.
- CatBoost: explicitly out of scope for this pass (may be revisited later).

## Goals

- Make `class_weight="balanced"` (or a class-weight dict) actually work for
  any classifier that skyulf-core wraps, regardless of whether the
  underlying library has native support, via a single generic mechanism —
  not a one-off XGBoost patch.
- Surface `class_weight` in the hyperparameter UI for Random Forest, LightGBM,
  and XGBoost classifiers, using the existing schema-driven mechanism (no new
  frontend code required).
- Stop the spurious Optuna warning from firing on unrelated imports.
- Add a small, well-scoped convenience API for extracting a pipeline's
  train/test split.
- Ship all of the above: passing tests, lint/type-check clean, version
  bumped, published to PyPI.

## Non-goals

- Adding CatBoost or any other new model backend.
- Any other frontend code changes beyond confirming the existing schema-driven
  hyperparameter panel renders the new field correctly (it should need zero
  code changes, since it already renders `type: "select"` fields generically
  for other params like RF's `criterion`).
- Changing ensemble base-estimator naming (already correct).
- Touching `hyperparameter_tuner`'s registration mechanism (already correct).

## Design

### 1. Generic class-weight-via-sample-weight shim

In `SklearnCalculator.fit()` (`skyulf-core/skyulf/modeling/sklearn_wrapper.py`):

- Before instantiating the model, detect whether `class_weight` is present in
  the resolved `params` **and** whether the target `model_class`'s
  constructor natively accepts a `class_weight` parameter (via
  `inspect.signature`, mirroring how `_filter_supported_params` already
  introspects constructor signatures).
- If `class_weight` is present but **not** natively supported: pop it out of
  the params passed to the constructor, and after converting `X`/`y` via
  `SklearnBridge.to_sklearn`, compute
  `sklearn.utils.class_weight.compute_sample_weight(class_weight, y_np)` and
  pass the result as `sample_weight=` to `model.fit(...)`, but **only if the
  target model's `.fit()` method accepts a `sample_weight` parameter**
  (checked the same way, via `inspect.signature` on `model.fit`) — if it
  doesn't, raise a clear `ValueError` explaining the model can't be balanced
  this way, rather than silently dropping it as XGBoost does today.
- If `class_weight` **is** natively supported (RandomForest, LightGBM,
  LogisticRegression, etc.): behavior is unchanged — passed through to the
  constructor as today.
- This is a generic mechanism: it transparently fixes XGBoost today and will
  correctly handle any future model lacking native `class_weight` support
  without further special-casing.

### 2. Expose `class_weight` in the hyperparameter UI schema

In `skyulf-core/skyulf/modeling/hyperparameters/_tree.py`:

- Add a `class_weight` `HyperparameterField` (`type="select"`, options
  `None` / `"balanced"`, default `None`) to `RANDOM_FOREST_CLASSIFIER_PARAMS`,
  `LGBM_PARAMS` (used for `lgbm_classifier`; `LGBM_PARAMS` is shared with the
  regressor variant in `_registry.py`, so the field must be added in a way
  that only appears for the classifier — either by branching in
  `_registry.py`'s per-model-id lookup or defining a classifier-only params
  list, matching the existing `RANDOM_FOREST_PARAMS` /
  `RANDOM_FOREST_CLASSIFIER_PARAMS` split pattern), and `XGBOOST_PARAMS`
  (same classifier-only split needed, since `XGBOOST_PARAMS` is also shared
  with the regressor).
- No frontend changes required — `TrainingSettings.tsx` already renders
  `type: "select"` fields generically from whatever
  `jobsApi.getHyperparameters(modelType)` returns.

### 3. Lazy Optuna import

In `skyulf-core/skyulf/modeling/_tuning/engine.py`: move the
`OptunaSearchCV`/`optuna_integration` import out of module level and into the
specific function/branch that constructs an Optuna-strategy search (i.e. only
attempted when `strategy == "optuna"` is actually requested). Keep the same
fallback/logging behavior, just deferred so it never fires from unrelated
imports.

### 4. Convenience split-extraction API

Add a small method (e.g. `SkyulfPipeline.get_fitted_split(data, target_column)`
or similar — exact naming to be finalized during implementation, following
existing naming conventions in `pipeline.py`) that runs the configured
preprocessing chain via `FeatureEngineer.fit_transform()` and returns
`(X_train, y_train, X_test, y_test)` as plain pandas objects (converting from
Polars/other engine types via the existing `to_pandas()`-style pattern),
saving callers from reimplementing this themselves for custom evaluation
harnesses.

### Testing

- Unit tests for the `class_weight` shim: a fake/real classifier without
  native `class_weight` support fit with `class_weight="balanced"` produces
  a model whose behavior differs measurably from an unweighted fit (e.g.
  compare predictions/`sample_weight` argument was actually passed); a model
  with native support is unaffected (constructor still receives
  `class_weight` directly, no `sample_weight` injected).
- A specific regression test for `xgboost_classifier` with
  `class_weight="balanced"` on a synthetic imbalanced dataset, asserting
  `recall`/`balanced_accuracy` on the minority class improves versus
  unweighted.
- Test that `RANDOM_FOREST_CLASSIFIER_PARAMS` / classifier-only `LGBM`/
  `XGBOOST` param lists include `class_weight` while their regressor
  counterparts do not (regression has no such concept).
- Test that importing `skyulf.modeling._tuning.engine` (or `skyulf.modeling`)
  does not emit the Optuna warning, and that requesting an actual
  `strategy="optuna"` tuning run still works/warns appropriately when
  `optuna-integration` is absent.
- Test for the new split-extraction convenience method: returned splits match
  what `SkyulfPipeline.fit()` would internally use (same shapes, same target
  column removed, same row counts for a fixed `random_state`).
- Full gate per repo conventions: `ruff check .`, `ruff format --check`, `ty
  check`, and the relevant `pytest` suite (skyulf-core) all passing.

### Release

- The repo already has an automated release pipeline
  (`.github/workflows/release.yml`): on push to `master`, it reads the
  version from `skyulf-core/setup.py`, and if that version has no matching
  `core-v<version>` git tag yet, builds and publishes to PyPI via Trusted
  Publishing (OIDC) automatically.
- Implementation step: bump the `version=` string in `skyulf-core/setup.py`
  (patch bump, e.g. `0.5.3` → `0.5.4`, since this is bugfixes + small additive
  schema fields, not a breaking change), commit, and push/merge to `master`.
  No manual `twine upload` or separate publish step needed.

## Out of scope reminders

No CatBoost work. No changes to ensemble naming or `hyperparameter_tuner`
registration. No frontend code changes beyond visual/behavioral verification
that the new `class_weight` field renders correctly through the existing
generic schema-driven panel.
