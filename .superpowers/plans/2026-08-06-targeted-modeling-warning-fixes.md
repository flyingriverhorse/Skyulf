# Targeted Modeling Warning Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the known Optuna experimental notice and sklearn feature-name warnings from supported pipeline paths without hiding unrelated warnings.

**Architecture:** Suppress the exact Optuna warning only at the `OptunaSearchCV` construction boundary. At the SHAP auxiliary prediction boundary, select Pandas or numpy based on the fitted estimator's `feature_names_in_` metadata while retaining Pandas for explainability labels and payloads.

**Tech Stack:** Python 3.12, scikit-learn, Optuna, SHAP, Pandas, NumPy, pytest, Ruff, ty.

## Global Constraints

- Public APIs and return payloads remain unchanged.
- Optuna remains an explicitly supported but upstream-experimental dependency.
- SHAP feature names and per-sample display data remain sourced from Pandas.
- All unrelated warning categories remain visible.
- Keep the v0.7.4 changelog update concise.

---

### Task 1: Suppress Only OptunaSearchCV's Known Experimental Notice

**Files:**
- Modify: `skyulf-core/skyulf/modeling/_tuning/engine.py:937-1003`
- Test: `skyulf-core/tests/test_tuning_engine.py:838-863`

**Interfaces:**
- Consumes: lazily loaded module-level `optuna` and `OptunaSearchCV`.
- Produces: unchanged `_build_optuna_searcher(...) -> Any`.

- [ ] **Step 1: Add a failing warning regression test**

Add `import warnings` to `tests/test_tuning_engine.py`. Update
`test_fit_optuna_strategy_basic()` so it records warnings around `tuner.fit()`:

```python
def test_fit_optuna_strategy_basic():
    """Optuna tuning succeeds without leaking its known experimental notice."""
    optuna_module = pytest.importorskip("optuna")
    X, y = _clf_xy(n=150)
    tuner = _tuner_clf()
    cfg = TuningConfig(
        strategy="optuna",
        metric="accuracy",
        search_space={"C": [0.1, 1.0, 10.0]},
        n_trials=4,
        cv_folds=3,
        random_state=42,
    )
    progress_calls: list[tuple] = []
    logs: list[str] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, result = tuner.fit(
            X,
            y,
            config=cfg.__dict__,
            progress_callback=lambda *a: progress_calls.append(a),
            log_callback=logs.append,
        )

    assert not any(
        issubclass(item.category, optuna_module.exceptions.ExperimentalWarning)
        and "OptunaSearchCV is experimental" in str(item.message)
        for item in caught
    )
    assert hasattr(model, "predict")
    assert result.n_trials > 0
    assert len(progress_calls) > 0
    assert any("Tuning Completed (optuna)" in msg for msg in logs)
```

- [ ] **Step 2: Verify the test fails on the current warning leak**

Run:

```bash
cd /Users/BH7043/Skyulf/skyulf-core
source /Users/BH7043/Skyulf/.venv/bin/activate
python -m pytest tests/test_tuning_engine.py::test_fit_optuna_strategy_basic -q
```

Expected: FAIL because `caught` contains
`optuna.exceptions.ExperimentalWarning: OptunaSearchCV is experimental...`.

- [ ] **Step 3: Add the exact local warning filter**

In `_build_optuna_searcher()`, wrap only the constructor:

```python
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"OptunaSearchCV is experimental.*",
        category=optuna.exceptions.ExperimentalWarning,
    )
    return OptunaSearchCV(
        estimator=base_estimator,
        param_distributions=distributions,
        n_trials=config.n_trials,
        timeout=config.timeout,
        cv=cv,
        scoring=metric,
        n_jobs=config.n_jobs,
        refit=False,
        verbose=0,
        callbacks=callbacks,
        study=study,
    )
```

Do not modify `TuningCalculator.fit()`'s general warning collection/re-emission
logic.

- [ ] **Step 4: Run focused Optuna tests**

Run:

```bash
python -m pytest \
  tests/test_tuning_engine.py::test_fit_optuna_strategy_basic \
  tests/test_tuning_engine.py::test_fit_optuna_strategy_samplers -q
```

Expected: all selected tests pass with no leaked experimental notice.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  skyulf-core/skyulf/modeling/_tuning/engine.py \
  skyulf-core/tests/test_tuning_engine.py
git commit -m "fix(modeling): contain Optuna experimental warning

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Match SHAP Auxiliary Prediction Input to Fit Metadata

**Files:**
- Modify: `skyulf-core/skyulf/modeling/_explainability/shap_explanation.py:60-73`
- Test: `skyulf-core/tests/test_explainability.py`

**Interfaces:**
- Consumes: sklearn's optional `feature_names_in_` fitted-estimator attribute.
- Produces: unchanged `_predicted_class_index(model, sample, n_classes) -> np.ndarray`.

- [ ] **Step 1: Add a failing parameterized pipeline regression test**

Update imports in `tests/test_explainability.py`:

```python
import warnings

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
```

Add:

```python
@pytest.mark.parametrize(
    ("model_class", "model_kwargs"),
    [
        (DecisionTreeClassifier, {"random_state": 0}),
        (RandomForestClassifier, {"n_estimators": 5, "random_state": 0}),
        (ExtraTreesClassifier, {"n_estimators": 5, "random_state": 0}),
    ],
    ids=["decision-tree", "random-forest", "extra-trees"],
)
def test_numpy_fitted_multiclass_tree_explainability_has_no_feature_name_warning(
    multiclass_data,
    model_class,
    model_kwargs,
):
    """Pipeline-style numpy fitting should not warn during SHAP class lookup."""
    X, y = multiclass_data
    model = model_class(**model_kwargs).fit(X.to_numpy(), y.to_numpy())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_shap_explanation(model, X, max_display_samples=5)

    assert result is not None
    assert not any("feature names" in str(item.message) for item in caught)
    assert result["feature_names"] == list(X.columns)
```

- [ ] **Step 2: Verify all three cases fail before the fix**

Run:

```bash
python -m pytest \
  tests/test_explainability.py::test_numpy_fitted_multiclass_tree_explainability_has_no_feature_name_warning -q
```

Expected: three failures showing warnings for DecisionTreeClassifier,
RandomForestClassifier, and ExtraTreesClassifier.

- [ ] **Step 3: Choose prediction input from fit metadata**

Change `_predicted_class_index()`:

```python
def _predicted_class_index(model: Any, sample: pd.DataFrame, n_classes: int) -> np.ndarray:
    """Best-effort per-row predicted-class index for a multi-class (3+) model.

    Falls back to all-zeros (first class) if the model can't predict or its
    `classes_` don't line up with the SHAP output's class axis.
    """
    try:
        classes = list(getattr(model, "classes_", []))
        predict_input = sample if hasattr(model, "feature_names_in_") else sample.to_numpy()
        preds = model.predict(predict_input)
        if classes and len(classes) == n_classes:
            return np.array([classes.index(p) if p in classes else 0 for p in preds], dtype=int)
    except Exception:
        logger.debug("Falling back to class 0 for per-row SHAP selection", exc_info=True)
    return np.zeros(len(sample), dtype=int)
```

Do not convert the sample used by SHAP, feature naming, or payload
construction.

- [ ] **Step 4: Run focused explainability tests**

Run:

```bash
python -m pytest tests/test_explainability.py -q
```

Expected: all explainability tests pass, including existing DataFrame-fitted
estimators and the three numpy-fitted tree estimators.

- [ ] **Step 5: Commit Task 2**

```bash
git add \
  skyulf-core/skyulf/modeling/_explainability/shap_explanation.py \
  skyulf-core/tests/test_explainability.py
git commit -m "fix(modeling): align SHAP prediction input metadata

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Document and Verify Warning Containment

**Files:**
- Modify: `changelog/0.7.x.md`
- Modify: `.superpowers/sdd/progress.md`

**Interfaces:**
- Consumes: completed Tasks 1 and 2.
- Produces: concise release note and durable verification record.

- [ ] **Step 1: Add concise documentation**

Add one v0.7.4 paragraph:

```text
Targeted modeling warning containment. Optuna tuning no longer leaks its known
OptunaSearchCV experimental notice, while all other warning categories remain
visible. Multiclass SHAP class selection now matches prediction input to the
estimator's fit-time feature-name metadata, removing pipeline warnings for
numpy-fitted DecisionTree, RandomForest, and ExtraTrees models without losing
Pandas feature labels.
```

Record root causes, files, regression tests, and verification commands in
`.superpowers/sdd/progress.md`.

- [ ] **Step 2: Run focused lint and type checks**

Run:

```bash
cd /Users/BH7043/Skyulf/skyulf-core
source /Users/BH7043/Skyulf/.venv/bin/activate
ruff check \
  skyulf/modeling/_tuning/engine.py \
  skyulf/modeling/_explainability/shap_explanation.py \
  tests/test_tuning_engine.py \
  tests/test_explainability.py
ruff format --check \
  skyulf/modeling/_tuning/engine.py \
  skyulf/modeling/_explainability/shap_explanation.py \
  tests/test_tuning_engine.py \
  tests/test_explainability.py
ty check \
  skyulf/modeling/_tuning/engine.py \
  skyulf/modeling/_explainability/shap_explanation.py
```

Expected: all checks pass.

- [ ] **Step 3: Run focused warning tests with warnings promoted**

Run:

```bash
python -m pytest \
  tests/test_tuning_engine.py::test_fit_optuna_strategy_basic \
  tests/test_explainability.py::test_numpy_fitted_multiclass_tree_explainability_has_no_feature_name_warning \
  -q
```

Expected: four selected cases pass and neither target warning is captured.

- [ ] **Step 4: Run the full Core suite**

Run:

```bash
python -m pytest -q
```

Expected: existing baseline plus the three new parameterized explainability
cases, with no regressions.

- [ ] **Step 5: Review and commit documentation**

Run:

```bash
cd /Users/BH7043/Skyulf
git --no-pager diff --check
git status --short
```

Then commit:

```bash
git add changelog/0.7.x.md .superpowers/sdd/progress.md
git commit -m "docs: record targeted modeling warning fixes

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

- [ ] **Step 6: Request final code review**

Review the complete Task 1-3 commit range against the approved design. Fix all
Critical and Important findings, rerun their covering tests, and re-review
before completion.

---

### Task 4: Exercise the Supported Tuning Prediction Boundary

**Files:**
- Modify: `skyulf-core/tests/test_pipeline_integration_tuning.py:21-24,93-132`
- Modify: `.superpowers/sdd/progress.md`

**Interfaces:**
- Consumes: `TuningApplier(base_applier)` and its expected
  `(fitted_model, tuning_result)` artifact.
- Produces: unchanged production behavior; the integration test predicts
  through Skyulf's supported numpy-normalizing applier boundary.

- [ ] **Step 1: Make the existing warning fail the integration test**

Add `import warnings`, `TuningApplier`, and
`RandomForestClassifierApplier` to the test imports. Wrap the prediction
section:

```python
        tuned_applier = TuningApplier(RandomForestClassifierApplier())
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message=r"X has feature names, but RandomForestClassifier was fitted without feature names",
            )
            preds = tuned_applier.predict(X_test, (model, result))
```

Before switching to `tuned_applier.predict`, temporarily keep
`preds = model.predict(X_test)` inside the warning context.

- [ ] **Step 2: Verify the raw-estimator prediction fails**

Run:

```bash
cd /Users/BH7043/Skyulf/skyulf-core
source /Users/BH7043/Skyulf/.venv/bin/activate
python -m pytest \
  tests/test_pipeline_integration_tuning.py::TestRandomSearchClassification::test_random_search_returns_valid_best_params_and_finite_test_score \
  -q
```

Expected: FAIL because the sklearn feature-name warning is promoted to an
exception.

- [ ] **Step 3: Predict through `TuningApplier`**

Replace the raw prediction with:

```python
        tuned_applier = TuningApplier(RandomForestClassifierApplier())
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message=r"X has feature names, but RandomForestClassifier was fitted without feature names",
            )
            preds = tuned_applier.predict(X_test, (model, result))
```

Keep all best-parameter and finite-accuracy assertions unchanged.

- [ ] **Step 4: Run focused integration and warning tests**

Run:

```bash
python -m pytest \
  tests/test_pipeline_integration_tuning.py::TestRandomSearchClassification::test_random_search_returns_valid_best_params_and_finite_test_score \
  tests/test_tuning_engine.py::test_fit_optuna_strategy_basic \
  tests/test_explainability.py::test_numpy_fitted_multiclass_tree_explainability_has_no_feature_name_warning \
  -q
```

Expected: five selected cases pass with no target warning.

- [ ] **Step 5: Run lint and the full Core suite**

Run:

```bash
ruff check tests/test_pipeline_integration_tuning.py
ruff format --check tests/test_pipeline_integration_tuning.py
python -m pytest -q
```

Expected: Ruff passes; full suite remains `2926 passed, 69 skipped, 1
xfailed`, and the RandomForest integration-test warning is absent from the
warning summary.

- [ ] **Step 6: Record and commit the follow-up**

Append the integration-test root cause, supported-boundary decision, and
verification result to `.superpowers/sdd/progress.md`, then commit:

```bash
git add \
  skyulf-core/tests/test_pipeline_integration_tuning.py \
  .superpowers/sdd/progress.md
git commit -m "test(modeling): use tuning applier for integration prediction

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

- [ ] **Step 7: Request final review**

Review Task 4 for spec compliance and code quality, then review the complete
warning-fix implementation range. Fix every Critical or Important finding
before completion.
