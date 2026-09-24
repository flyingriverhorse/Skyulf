# SM-32: manual approval from saved candidate evidence

Date: 2026-09-24. Status: implemented locally; SM-32 remains ACTIVE.
Parent: [selection and approval design](38-model-selection-and-approval-design.md).

## Delivered library behavior

`run_action(..., "approve", candidate_version=..., comparison_sha256=...,
expected_champion_version=...)` is an explicit manual-policy operation.
The new `local_approval` service shares existing Core training snapshot,
comparison, initialization, promotion and receipt verification functions.
It neither fits nor registers a model and never changes the scoring pin.

Training now saves `candidate_training_spec.json` beside its comparison.
Approval reads both artifacts from the requested model version's run. It
verifies the requested comparison digest, model identity, source contract,
engine, current metric policy and expected champion. The original source
version and split window are restored; today's training dates are not used.
Current budgets can tighten the saved read limits. Quality is re-evaluated
on the pinned holdout before a checked alias mutation.

An unchanged repeated approval returns the same still-active committed
receipt. A newer champion, incompatible proof, pending mutation, failed
quality gate or changed policy is refused. The public approval service itself
also enforces manual policy, even when called without `run_action`.

## Local verification

- Initial approval tests failed because the action was not implemented.
- Real MLflow cases passed with both pandas and Polars: manual bootstrap,
  improved candidate, same-receipt retry, pinned score version preservation,
  exact version counts (no retraining/registration), rejected tied candidate,
  wrong digest, changed metric policy/data contract, pending alias event,
  changed holdout and stale expected champion.
- The public-service policy regression first failed and was then corrected.
- Full repository ty and scoped Ruff passed.
- Combined relocation, workflow, approval, retraining and promotion suite:
  **321 passed**, 24 dependency/deprecation warnings, in 139.15 seconds.
  Evidence: `rehearsals/sm32-moves-approval-final.log`.
- Tests now live in `skyulf-core/tests/integrations`; relocation details are
  in [the inventory](44-core-test-relocation.md).

## Remaining work and Databricks timing

Reject, rollback, previous_challenger and the matching Bundle operator inputs/
score handoff remain open. The existing generated Bundle still uses its
legacy interface; this library slice has not been uploaded or run remotely.

Run the Databricks rehearsal after those lifecycle operations are wired and
local/generated-project checks pass. Inspect and reuse the existing personal
test workspace resources where appropriate instead of adding another schema
for each subtask. The reviewable live plan must cover:

1. Train/register a passing manual contender and inspect its MLflow metrics.
2. Approve that exact existing version with no new training/model registration.
3. Score with champion selection and with a fixed version; inspect Delta output.
4. Repeat approval/score to verify receipt reuse and no duplicate predictions.
5. Replace a challenger, reject a contender, and perform controlled rollback.
6. Verify both local engines, stale evidence rejection and failed-score recovery.

No new live run or cleanup was performed in this slice. Models trained before
the saved specification exists need explicit evidence migration; approval must
never silently reconstruct their original evaluation window from current config.
