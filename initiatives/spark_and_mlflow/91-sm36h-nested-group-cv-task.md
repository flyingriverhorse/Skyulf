# SM-36h: nested group cross-validation

Date: 2026-09-27. Completed: 2026-09-28. Status: DONE for documented scope.
Dependency: SM-36f true nested tuning. Independent of SM-36g and SM-36i.

## Outcome

Evaluate tuning without sharing a customer, patient or other group between
training and validation in either the inner or outer loop.

## Scope

- Define group-column and group-aware split policy in Core. Connect backend,
  Canvas, Bundle setup/config and SDK preview to that same contract.
- Establish ordinary group CV plumbing where absent; explicitly distinguish
  GroupKFold from stratified group behavior for classification. Reject unsupported
  combinations rather than silently using ordinary KFold.
- Carry groups as aligned split metadata through filtering, row ordering and
  preprocessing. Group-only identifiers must not become model input features.
- Split inner folds using only the current outer training groups. Keep learned
  preprocessing and all parameter selection inside those boundaries.
- Define missing-group, insufficient-group and classification-coverage validation,
  deterministic splitting and final holdout group-isolation policies.
- Preserve a separate full-training group-aware final search and save the effective
  split policy, group counts and bounded membership evidence with outer results.

## Acceptance

- Assert disjoint group membership for every inner/outer pair and final holdout.
  Include repeated entities, uneven sizes, row filters, missing values and too few
  groups. Reject invalid cases before returning a successful aggregate.
- Compare against independent sklearn group-aware nested loops; test preprocessing
  membership, all five strategies with bounded fits and representative ensembles.
- Verify Canvas payload/backend execution and generated Bundle settings. Exercise
  pandas/Polars, saved artifact replay and persisted group evidence on Databricks.
- Document supported policies and record actual checks before marking DONE.

## Completion evidence

Implemented across Core/backend/Canvas/Bundle, with real nested selection and
saved evidence. Final changed-feature suite: 291 passed; frontend: 134 passed;
three generated bundles passed strict CLI validation. Databricks: 16/16 cases,
384 replay predictions and independent persisted SHA verification.
See [combined acceptance and limitations](101-sm36ghi-nested-policy-acceptance.md).
Nested changes are included in the delivery commit; baseline graph commit is 42bc0d37.
