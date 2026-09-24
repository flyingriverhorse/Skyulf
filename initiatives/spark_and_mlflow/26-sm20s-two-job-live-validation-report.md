# SM-20S two-job local Bundle validation

Date: 2026-09-24. Personal Databricks profile: `skyulf`; `dev` target only.
This run reused the existing CDF-enabled real-taxi source and registered
Polars model version 1 in `workspace.skyulf_bundle_first_20260924`. The new
Bundle is `skyulf_twojob`; the test did not deploy to a company workspace.

## Bundle and Delta results

The serverless project passed `databricks bundle validate --strict`, then
deployed exactly two persistent jobs: `skyulf_twojob_train`
(`290396749537917`) and `skyulf_twojob_score` (`1046235940974377`). The
score job has `max_concurrent_runs: 1`. The policy-cluster template also
rendered as exactly these two job keys with that score limit; a real policy
cluster was not provisioned or run. Deployment itself created no UC table.

| Step | Run | Result |
| --- | --- | --- |
| First score | [192862744456152](https://dbc-45604623-c18b.cloud.databricks.com/jobs/1046235940974377/runs/192862744456152?o=7474646244882000) | Created only `skyulf_twojob_predictions`; 650 inputs, 650 outputs, target commit 1 |
| Unchanged replay | [383783499229420](https://dbc-45604623-c18b.cloud.databricks.com/jobs/1046235940974377/runs/383783499229420?o=7474646244882000) | 0 inputs, `noop=true`, target commit stayed 1 |
| Append one source row | One-time run `1116828359967045` | Exactly one distinct `entity_id` appended to the existing source |
| Incremental score | [183922007779250](https://dbc-45604623-c18b.cloud.databricks.com/jobs/1046235940974377/runs/183922007779250?o=7474646244882000) | 1 input, 1 output, target commit 2 |
| Independent audit | One-time run `963251710081057` | 651 source rows, 651 prediction rows, 651 distinct prediction keys, one appended-row prediction, only model version 1 |
| Final code replay | [341978630819415](https://dbc-45604623-c18b.cloud.databricks.com/jobs/1046235940974377/runs/341978630819415?o=7474646244882000) | Redeployed final notebook; 0 inputs, `noop=true`, target commit stayed 2 |

After the final replay, the prior three-job test's `setup`, `train`, and
`score` jobs and its old prediction and score-control tables were removed.
The final CLI inventory returned exactly the two `skyulf_twojob` jobs and two
managed Delta tables in the schema: existing `skyulf_reset_dev_source` and
new `skyulf_twojob_predictions`. The registered UC model remains a model,
not a table. The one-time append and audit notebooks were removed from the
new Bundle's workspace files after their runs finished.

## Local checks and operational boundary

Focused workflow/admission tests: 18 passed. A real local Delta test passed
first score, one-row append, and unchanged replay using
`SingleWriterAdmission` without a control table. Ruff lint/format, Ty, and
strict MkDocs build passed. The serverless test does not establish policy
cluster behavior or writer isolation: the no-table admission mode assumes a
single serialized score job and no other principal writing its target. A
deployment with multiple publishers must use Core's shared admission path.
