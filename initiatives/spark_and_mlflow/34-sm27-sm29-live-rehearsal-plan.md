# SM-27/SM-29 isolated live rehearsal

Status: awaiting approval for personal-workspace resources and serverless runs.
Catalog: `workspace`; proposed schema: `skyulf_sm27_sm29_20260924`.
Profile: `skyulf`; target: `dev`. Company targets are out of scope.

## Test resources

| Resource | Purpose |
| --- | --- |
| `workspace.skyulf_sm27_sm29_20260924.source` | One bounded, CDF-enabled Delta source containing labeled history and later inserts |
| `...predictions_append` | One physical incremental prediction table |
| `...predictions_full` | One stable full-rebuild view over `_v1` and `_v2` physical generations |
| `...model` | One registered UC model with v1, v2 and v3 candidates |
| `skyulf_sm29_verify_train`, `skyulf_sm29_verify_score` | The generated Bundle's only two persistent jobs |

The source will have 160 deterministic rows at initialization, with
`entity_id`, `feature_value`, `target`, `event_time`, and `label_at` columns.
The target is a simple customer-value estimate in dollars derived from the
feature, so the test can check model quality as well as row identity. Ten new
rows will be appended only after the initial runs. This is a bounded ML
contract test, not a benchmark or production dataset. A one-time ephemeral
serverless setup run creates the schema and source; it does not create a
persistent setup job. Each submitted task will have a 15-minute timeout.

## Rehearsal sequence

1. Generate a Polars, serverless, `auto_champion` Bundle with a heldout RMSE
   gate and `incremental_append`. Build the current wheel, validate the Bundle
   strictly, deploy only its `dev` target, and inspect that deployment itself
   created no UC table/model.
2. Train v1 with a deliberately underfit linear model. The first model must
   pass an absolute RMSE gate, receive a committed `@champion` receipt, and
   trigger the score job. Verify 160 v1 prediction rows.
3. Change only the scoring policy/output to `full_rebuild` and score v1.
   Verify the complete `_v1` table and managed view.
4. Train v2 with an intercept on the same pinned labeled snapshot. Verify a
   lower RMSE, the configured minimum improvement, `@challenger` staging,
   promotion to `@champion`, `@previous_champion=v1`, and automatic scoring
   into complete `_v2` before the view switches. `_v1` must remain unchanged.
5. Return to append mode, insert ten new source rows, and score. Verify that
   the append table retains its 160 v1 rows and adds exactly ten v2 rows.
   Score full mode again and verify its v2 generation/view reaches 170 rows.
6. Train v3 with the same configuration as v2. The tie should fail the
   improvement gate, leave champion on v2, and let scoring process no new
   rows. Verify the concrete model-version metadata, aliases, receipts,
   MLflow heldout metrics, target versions, and no-op behavior.

Record run URLs/IDs, counts, model metrics, aliases, table/view state, and
any failure before calling either task DONE. Read-only audits will inspect
outputs. Keep test resources for user inspection until cleanup is explicitly
requested. Production and company workspaces remain untouched.

This plan does not establish exclusive alias-write permissions by itself.
Before treating the Bundle as production-ready, give the train identity sole
model-alias write authority and verify a restricted identity is denied. If
the personal workspace cannot provide that isolation, record it as an open
deployment gate rather than claiming it passed.
