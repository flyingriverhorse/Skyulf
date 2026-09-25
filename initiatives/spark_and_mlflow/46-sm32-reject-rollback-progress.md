# SM-32: explicit rejection and repeatable rollback

Date: 2026-09-25. Status: local library implementation; SM-32 remains ACTIVE.
Baseline: `94b28320` on branch `090`. This change delivers the rejection/rollback slice.
Parent: [selection and approval design](38-model-selection-and-approval-design.md).

## Delivered behavior

- `run_action(..., "reject")` takes a concrete candidate version, comparison
  digest, expected champion and human-readable reason. Explicit manual policy
  is required. It loads the candidate's saved evidence and verifies controlled
  champion/challenger state before recording the decision.
- Rejection preserves aliases and evaluation results. Model-version tags
  `approval_status=rejected` and `approval_reason` describe the operator's
  decision separately from `validation_status` and quality metrics.
- Initialization, promotion, restaging and renomination cannot implicitly
  override rejection. A later training run may nominate a new version; this
  slice provides no reopen/force-approval action.
- `run_action(..., "rollback")` requires the configured model's completed
  promotion receipt and its expected champion. It works under manual or
  automatic promotion policy and preserves a separate controlled challenger.
- Unchanged rejection/rollback retries return the original committed receipt.
  Changed evidence, inconsistent control state and unresolved writes fail.
  Partial decision-tag writes retain a pending event for reconciliation.
- Neither operation fits, registers or uploads a model, reads training rows,
  changes a scoring pin, launches score or rewrites earlier predictions.
  Existing Core registry/admission/receipt services are reused.

## Verification

The combined promotion, manual approval, local workflow/retraining, Bundle
lifecycle/template/notebook and comparison suites passed **145 tests**, with
five expected legacy-selection deprecation warnings, in 120.38 seconds.
Evidence: `rehearsals/sm32-reject-verified.log`. Scoped Ruff lint/format,
full repository ty, strict MkDocs and `git diff --check` also passed.
MkDocs evidence: `rehearsals/sm32-reject-mkdocs-final.log` (exit 0).

The cases cover both pandas and Polars with actual local MLflow artifacts,
manual rejection of failed-quality candidates, valid first-model rejection,
no implicit reconsideration, controlled-marker disagreement, unchanged
registry version counts, automatic-policy rollback and safe retries.
Failure injection covers partially written decisions and conflicting rollback
control state. The reviewer identified the local rejection wrapper's missing
controlled-champion check; that check and its regression are now included.

## Boundaries and next work

The generated Bundle still has the legacy selector and job graph. Do not
migrate only its JSON or treat these library calls as deployed operator tasks.
All lifecycle mutations require the same externally serialized writer as
training; no new job or admission table is created by this slice. The low-level
registry API preserves its existing legacy-champion compatibility; the local
workflow wrapper requires controlled state.

Next: implement `previous_challenger` replacement history, then expose the
independent Bundle choices and operator inputs through its serialized lifecycle
job, with optional score handoff using the existing score job. Complete local
generation/validation checks before the combined Databricks rehearsal. No
Databricks resources were changed or live jobs run in this continuation.

Examples: [Bundle user guide](../../docs/user_guide/databricks_bundle.md) and
[registry user guide](../../docs/user_guide/mlflow_registry.md).
