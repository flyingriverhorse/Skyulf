# SM-36g: nested temporal cross-validation

Date: 2026-09-27. Completed: 2026-09-28. Status: DONE for documented scope.
Dependency: SM-36f true nested tuning. Implementation and acceptance are recorded below.

## Outcome

Evaluate a complete tuning procedure using past-to-future splits in both outer
and inner loops, preserving an untouched final holdout and a separate final search.
Keep ordinary Time Series CV and current stratified/KFold nested behavior intact.

## Scope

- Define explicit nested split-policy settings in Core; carry the same resolved
  settings through backend, Canvas controls, Bundle setup/config and SDK preview.
  Do not silently reinterpret existing `nested_cv` recipes as temporal.
- Keep event metadata aligned through sorting, filtering and fold preprocessing;
  exclude split-only time columns from model features unless separately requested.
- Define expanding/rolling training windows, test size and optional gap boundaries.
  Resolve equal/missing timestamps explicitly; never put a later training event
  ahead of an earlier validation event in either loop.
- Instantiate each inner splitter only on that outer training partition. Fit all
  learned preprocessing there; outer rows cannot guide parameter selection.
- Run a separate final temporal search on training rows only. Persist effective
  boundaries, gap/window settings, counts, selected params and outer scores.
- Support regression, classification and eligible ensembles; reject insufficient
  chronological class coverage and incompatible settings with actionable errors.

## Acceptance

- Compare outer scores and winners against independent chronological sklearn loops.
- Assert chronology and gap isolation for every inner/outer fold, including tied
  times, unsorted input, missing times, row filters and preprocessing membership.
- Cover all five strategies with bounded real fits, plus classification/regression
  and voting/stacking cases. Do not claim full Cartesian coverage from samples.
- Verify Canvas payload/backend execution, generated Bundle settings, artifact
  replay and persisted evidence. Run bounded pandas/Polars cases on Databricks.
- Update user documentation and record exact passed/failed/skipped evidence before
  changing this task to DONE.

## Completion evidence

Implemented across Core/backend/Canvas/Bundle, with real nested selection and
saved evidence. Final changed-feature suite: 291 passed; frontend: 134 passed;
three generated bundles passed strict CLI validation. Databricks: 16/16 cases,
384 replay predictions and independent persisted SHA verification.
See [combined acceptance and limitations](101-sm36ghi-nested-policy-acceptance.md).
Nested changes are included in the delivery commit; baseline graph commit is 42bc0d37.
