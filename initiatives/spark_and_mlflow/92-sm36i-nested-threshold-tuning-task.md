# SM-36i: threshold selection within nested tuning

Date: 2026-09-27. Completed: 2026-09-28. Status: DONE for documented scope.
Dependency: SM-36f true nested tuning. Independent of SM-36g and SM-36h initially;
their combinations require explicit integration tests before claiming support.

## Outcome

Include decision-threshold selection in the procedure evaluated by outer CV.
Every outer score must evaluate both selected model parameters and the threshold
selected without access to that outer test fold.

## Scope

- Define an out-of-fold or reserved-inner-validation threshold-selection policy
  restricted to the current outer training partition. Do not select a threshold
  using in-sample predictions or untouched outer labels.
- Preserve positive-class/string-label mapping, scoring direction and probability
  requirements. Define binary/multiclass support explicitly, including ensemble
  calibration and models that cannot expose suitable probabilities.
- Apply each fold's selected threshold when computing threshold-dependent outer
  metrics. Distinguish probability/ranking metrics and hyperparameter search scores.
- Select the final deployable threshold using training-only evidence from the
  separate final search; never reuse outer winners or tune against final holdout.
- Persist fold thresholds, metric, selection provenance, outer scores and final
  threshold in Core artifacts, backend results, Canvas and Databricks/MLflow output.
- Replace the current nested+tune_threshold rejection only after the supported
  policy is complete. Existing ordinary threshold tuning must remain compatible.

## Acceptance

- Compare threshold selection and outer predictions with an independent reference.
  Prove changing outer/holdout labels cannot change the corresponding selected
  threshold or hyperparameters; outer scores should still reflect those labels.
- Cover imbalanced and string-label classification, probability-unavailable
  failures, no-finite-candidate handling and eligible voting/stacking models.
- Exercise all five strategies with bounded real fits, payload mappings, artifact
  reload and prediction parity with threshold application enabled/disabled.
- Run bounded pandas/Polars Databricks cases and verify persisted thresholds and
  readable results before recording DONE.

## Completion evidence

Implemented across Core/backend/Canvas/Bundle, with real nested selection and
saved evidence. Final changed-feature suite: 291 passed; frontend: 134 passed;
three generated bundles passed strict CLI validation. Databricks: 16/16 cases,
384 replay predictions and independent persisted SHA verification.
See [combined acceptance and limitations](101-sm36ghi-nested-policy-acceptance.md).
Nested changes are included in the delivery commit; baseline graph commit is 42bc0d37.
