# Company-compatible tags and clean SM-30 verification

Date: 2026-09-24. Status: 104 local tests and live verification passed.
This is the user-requested prerequisite before SM-31.

## Tag contract

The reference training notebooks attach data versions to runs and attach the
following metadata to registered model versions. Skyulf now uses these names
on its training run and model version:

| Tag | Meaning |
| --- | --- |
| train_data_destination | Full training source table name |
| test_data_destination | Full evaluation source table name |
| train_data_version | Pinned Delta source version used for fit |
| test_data_version | Pinned Delta source version used for evaluation |
| model_type | Selected Core modeling type |
| candidate_date_tag | Training workflow date, YYYY-MM-DD in UTC |
| risk_category | Optional project classification from workflow.json |

Skyulf splits one snapshot by event and label times, so its train/test table
names and Delta versions may match. The reference can use separate tables.
train_start, test_start, and data_end make our split readable; engine records
pandas or Polars. task=training replaces phase=candidate_training.
The exact dataset identity stays in training_data.json and the comparison
artifact instead of a long concatenated run tag. An unusually long UC value
points to training_data.json; the artifact and run tags retain the full value.

The reference also copies version metadata onto the registered model object.
Skyulf keeps it on each version so a rejected contender cannot make top-level
metadata appear to describe champion. The reference best_model_name belongs
to model search, planned in SM-36; the single-candidate workflow uses model_type.
PayingRegNo is a separate compute cost tag, configurable through the existing
policy-compute cost_tag_key/cost_tag_value fields. These tests use serverless.

validation_status remains pending/passed/rejected/error. validation_reason
uses plain English, such as "No improvement over champion". Comparison
artifacts retain stable machine reason codes. promotion_status records
promotion separately; aliases identify current champion/challenger roles.

Technical event tags remain for interrupted-write recovery and receipt
verification. Their JSON uses action/from/proof/parent/state/previous instead
of single-letter fields; older receipts remain readable. Event IDs and hashes
are integrity references, not user settings. The 256-byte value limit remains.

## Reset and rehearsal scope

The user explicitly requested deletion and recreation. CLI inventory verified
owners, exact table/model contents, empty volumes and no active runs. Function
entries corresponded to the registered models.

Deleted job IDs: 1106811466161550, 793298964293930, 1046235940974377,
290396749537917. Deleted schemas including their test tables/models:
workspace.skyulf_sm27_sm29_20260924 and workspace.skyulf_bundle_first_20260924.
The following inventory showed no jobs and only default/information_schema.
Historical local evidence and test source files were preserved.

New target: workspace.skyulf_lifecycle_test. One source table, one model and
one predictions table per engine. The generic Bundle deploys two jobs.
An ephemeral four-task run performs setup, independent pandas and Polars
lifecycles, then appended-row/no-op checks. No schedules are activated.
Each task and the overall run have a 900-second timeout and no automatic retry.

First submission 425389607887141 was cancelled after a local wheel ACL blocked
upload. Identical bytes were copied into a readable workspace file;
replacement run: 607409241163605. Wheel SHA-256:
e76934549cd47b768c818f5fded2163a38ec92a9dbeeaa7ba0bb4a7d54c51481c.

Local tests: 104 passed in 79.20 seconds, rehearsals/tags-final.log.
All four tasks in run 607409241163605 returned SUCCESS. Polars finished with
champion=2 and rejected challenger=3. Pandas additionally nominated v4,
retained v5 with evaluation error, and rolled champion from v2 back to v1
while preserving challenger v5. Each successful training version had matching
company tags on its run and version, all six regression metrics, readable
receipt fields, and the full provenance artifact.

The final audit appended ten rows to the original 160-row source. Both output
tables contained 170 unique keys. Replaying both scorers returned noop=true
without changing their Delta commit versions. The main Polars output kept
160 v1 rows and appended ten v2 rows; the pandas rollback output used v1.

A later local wording correction distinguishes positive improvements below
the configured minimum from no improvement; its targeted test passed. The
live tie scenario keeps the same tested wording. It will ship in the SM-31 wheel.
Exact CLI inventories and scripts are under rehearsals/sm30_live/.

Reference: supplied dbml-mlops-template training notebook and
src/training/services/registry_service.py.tmpl. Registration tag forwarding
uses the [MLflow client API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html).
