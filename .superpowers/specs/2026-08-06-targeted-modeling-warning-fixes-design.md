# Targeted Modeling Warning Fixes Design

## Goal

Remove two confirmed pipeline warning leaks without broadly suppressing
diagnostics or changing model, tuning, or explainability results.

## Root Causes

1. `TuningCalculator.fit()` intentionally records warnings so it can aggregate
   convergence warnings and re-emit all other categories. Constructing the
   supported `OptunaSearchCV` strategy emits Optuna's known
   `ExperimentalWarning`, which is therefore re-emitted to pipeline users.

2. Skyulf tuning fits final sklearn estimators on numpy arrays. During
   multiclass SHAP processing, `_predicted_class_index()` calls
   `model.predict()` with the named Pandas display sample. Sklearn therefore
   warns for every affected estimator fitted without feature names, confirmed
   with `DecisionTreeClassifier`, `RandomForestClassifier`, and
   `ExtraTreesClassifier`.

## Design

### Optuna boundary

Wrap only the `OptunaSearchCV(...)` constructor in a local warning context that
ignores the exact Optuna experimental category and message. Do not suppress
other Optuna, sklearn, convergence, validation, or user warnings.

### SHAP prediction boundary

Keep the Pandas sample for feature names and SHAP payload construction. For the
auxiliary predicted-class lookup:

- pass the Pandas sample when the estimator exposes `feature_names_in_`;
- otherwise pass `sample.to_numpy()`.

This matches prediction input shape to the estimator's fit-time metadata and
avoids both directions of sklearn's feature-name warning.

## Compatibility

- Public APIs and return payloads remain unchanged.
- Optuna remains an explicitly supported but upstream-experimental dependency.
- SHAP feature names and per-sample display data remain sourced from Pandas.
- All unrelated warning categories remain visible.

## Testing

- Add an Optuna tuning regression test that records warnings and asserts the
  known `ExperimentalWarning` is absent while tuning succeeds.
- Add a parameterized multiclass SHAP regression test for
  `DecisionTreeClassifier`, `RandomForestClassifier`, and
  `ExtraTreesClassifier`, fitting each estimator on numpy while passing a named
  Pandas DataFrame to explainability; assert no feature-name warning and a
  valid explanation.
- Keep existing DataFrame-fitted explainability tests to cover the
  `feature_names_in_` branch.
- Update the random-search classification integration test to predict through
  `TuningApplier(RandomForestClassifierApplier())` with the `(model, result)`
  artifact instead of calling the numpy-fitted raw estimator with a Pandas
  frame. This tests the supported Skyulf prediction boundary and removes the
  pre-existing feature-name warning without changing final-model fit behavior.
- Run focused tuning/explainability tests, Ruff, type checking, and the full
  Core suite.
