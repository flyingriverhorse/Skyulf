# SM-32: recent challenger history

Date: 2026-09-25. Baseline: `6a7f9e95` on branch `090`.
Status: implemented locally, uncommitted; SM-32 remains ACTIVE.
Design: [independent model selection and approval](38-model-selection-and-approval-design.md).

## Behavior

Replacing a controlled challenger sets `previous_challenger` to the displaced
version. Both nomination and direct staging use this behavior. A repeated
nomination leaves the history and its event marker unchanged. Later replacements
rotate the pointer; older versions and their records are retained.

Promotion keeps a separate historical contender. Rollback or nomination
clears history if its version becomes champion or challenger. The pointer
is not restored to a conflicting role or used as a score fallback.

Implementation is centralized in `integrations/mlflow/promotion.py` at the
existing checked mutation boundary. No new job, table or dependency is added.
The old promotion/rollback receipt format stays unchanged. Auxiliary bounded
history tags retain prepared/committed intent and a current-event marker,
including cleared pointers. Every mutation uses the same pending-event and
writer-admission rules as existing lifecycle changes. Invalid aliases or
history evidence stop retries; uncertain writes require reconciliation.

## Verification

The initial seven targeted cases failed because history aliases and protections
were absent. The first implementation passed all 42 promotion tests. An
additional test reproduced an approval-replay bypass; controlled champion
resolution now checks history before that replay can return success.

The final combined suite passed **155 tests**, with five expected legacy-mode
deprecation warnings, in 170.87 seconds. Evidence:
`rehearsals/sm32-history-regression.log`. Scoped Ruff lint/format, full repository
ty, strict MkDocs and `git diff --check` also passed. Documentation build
evidence: `rehearsals/sm32-history-mkdocs.log` (exit 0).

Coverage includes rotation, unchanged retries, direct staging, promotion,
rollback, re-nomination of a historical version, raw/deleted aliases, missing
markers, malformed/prepared history receipts, and injected alias/marker write
failures. Existing pandas/Polars training/approval and Bundle compatibility
tests are included in the combined suite.

A read-only review found no additional transition defect and identified useful
coverage for re-nomination and corrupted history receipts; those cases were added.

## Remaining work

Expose independent selector/promotion choices and operator inputs in the
existing serialized Bundle lifecycle job. Connect optional score handoff to
the existing score job, preserving pins and avoiding retraining. Then validate
generated projects and perform the combined Databricks rehearsal. No new
Databricks run or cloud resource change occurred in this slice.
