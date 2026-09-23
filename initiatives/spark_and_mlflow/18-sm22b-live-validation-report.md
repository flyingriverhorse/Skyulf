# SM-22b isolated validation — 2026-09-23

SM-22b adds explicit, version-checked MLflow champion promotion and rollback.
It re-evaluates the supplied comparison against concrete registered versions,
uses one shared non-expiring admission, records a prepared/committed receipt,
and treats an uncertain alias write as an operator-reconciliation case. It
does not train a candidate or promote automatically.

## Local gate

The focused MLflow suite passed **40 tests** in the isolated MLflow 3.16.1
environment (promotion, comparison, registry and local model packaging).
Promotion cases use a real SQLite MLflow registry and cover normal promotion
and rollback, forged reports, stale and missing champion aliases, admission
contention, access denial, uncertain write response and stale rollback receipts.
Ruff, formatting and ty passed on the new Python files.

## Unity Catalog gate

The selected `skyulf` profile points to the previously approved Databricks
workspace. Only disposable resources in
`workspace.skyulf_sm24a_20260923` were used:

- Model: `skyulf_sm22b_promotion_r1`, with existing local-pipeline package
  versions 1 (reference) and 2 (candidate).
- Alias: `@champion`, initialized explicitly to version 1 for this test.
- Delta admission table: `skyulf_sm22b_alias_admission_r1`, with one stable
  alias key and nullable owner.
- Notebook folder:
  `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_sm24a_20260923/sm22b_alias_r1`.
  The successful run loaded wheel SHA-256
  `E6F13F4DD3F0D090A1B986FA4B5C0D710257BFD0637BA9D42C423ACD46A20AED`.

[Final-code owner run 73594561805168](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/702249165139866/run/73594561805168)
completed SUCCESS. On the same labeled holdout, candidate MSE was
`2.5849394142282116e-27` versus reference MSE `100.00000000000088`.
The run rejected a second admission holder, promoted version 2 (event
`e9be7c187a114c41afb21e79284bd3e9`), and rolled back to version 1
(event `c74b6ca99c4b462a9401e9991c61d38e`). The final alias was version 1. The earlier r4 owner run also passed, but r5 used the exact final tag names and admission guard.

The existing restricted test service principal
`skyulf-sm16-restricted-20260922` received only `USE_SCHEMA` on the test
schema and `EXECUTE` on this test model. It has no model ownership or registry
write grant. [Restricted run 821722911076530](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/1033450117407968/run/821722911076530)
completed SUCCESS: it read champion version 1, its attempt to set champion to
version 2 returned `PERMISSION_DENIED`, and a second read still returned 1.
No credential was created or copied.

Three earlier isolated owner attempts failed before promotion: serverless
Spark rejected `DataFrameWriter.mode("errorifexists")` for control-table
provisioning; Unity Catalog rejected dots in model-version tag keys; then it
rejected a receipt tag value over 256 bytes. The final implementation uses
`CREATE TABLE ... USING DELTA AS SELECT`, underscore tag keys and a bounded
compact receipt value. Runs `245964031922258`, `1102712691086932` and
`111298932154094` remain failed in the job history and are not counted as
passing gates.

## Operational boundary

All promotion/rollback writers must share the same admission row, and direct
alias writers must be denied by Unity Catalog permissions. The registry alias
API does not offer compare-and-swap. A bypass writer or a crashed owner still
requires operator inspection; a prepared event alone does not prove completion.
The labeled holdout snapshot must be pinned by the caller. SM-28a adds the
label-aware candidate-training workflow; no Bundle was created here.
