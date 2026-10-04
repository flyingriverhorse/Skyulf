# SM-36e - Connect optional SHAP setup and readable results

Status: DONE, 2026-09-30. User requested 2026-09-27. Depends on SM-36.
See [delivery and live evidence](121-sm36e-shap-delivery.md).

Baseline before this task: `pipeline.explainability` enables bounded training-only
SHAP through fitted preprocessing, writes MLflow `explanations.json`, and shows
status/sample count in the training task. Core's optional SHAP dependency is not
automatically included by generated projects. No Bundle SHAP charts exist yet.

## Delivered scope

- Offer an explicit opt-in during project configuration with bounded samples,
  transformed-feature guard and per-row display count. Keep default training free
  of optional explanation work.
- Connect opt-in to a compatible training-runtime SHAP dependency for serverless
  and classic compute; do not silently depend on an incidental installed package.
- Show a readable global feature-importance view and bounded per-row explanations
  with links to the exact model/run evidence. Identify transformed feature names.
- Preserve meaningful unavailable reasons, distinguish disabled/unavailable/completed,
  and explain that the feature budget is a guard, not top-K feature selection.
- Keep holdout, target and temporal metadata out of explanation inputs; reuse saved
  preprocessing without refit. Test regression/classification and ensemble paths.

## Acceptance

Generated setup and runtime dependencies agree; enabled/disabled and unavailable
cases pass tests; one bounded Databricks run verifies the actual explanation
artifact and visible report. No successful JSON write alone counts as a chart or
complete SHAP setup integration.
