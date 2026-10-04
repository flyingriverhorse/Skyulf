# Training parameters: one live Databricks acceptance

Date: 2026-09-29. This validates the expanded MLflow Parameters display on one
real training run, including the pending fitted-model/ensemble/split/recipe changes.

## Execution and independent verification

- Profile: `skyulf`; existing workspace `dbc-45604623-c18b.cloud.databricks.com`.
- One serverless task, client environment 4, zero retries; result **SUCCESS**.
- [Job run 352466724829816](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/587320338665929/run/352466724829816).
- Task run: `631747212357335`.
- [Experiment run](https://dbc-45604623-c18b.cloud.databricks.com/ml/experiments/1364695824110660/runs/92a0583430a841878c8b3cc10d3fef04?o=7474646244882000).
- Experiment ID: `1364695824110660`; run ID: `92a0583430a841878c8b3cc10d3fef04`.
- Experiment contains **74 parameters**. An independent client reread their exact
  values, three JSON artifacts, metrics, and the registered version's run linkage.
- Wheel SHA256: `25804c4ed6b10b8dafe4ffc10d06b4f77130674151fdbafcde777ff6ad180e08`.
  All 289 Python module hashes matched current source before upload and installed
  modules before training.

## Scenario and observed values

| Item | Observed |
|---|---|
| Model | Voting Regressor: `linear_regression`, `ridge` |
| Member weights | `[2, 1]` |
| Search | Random; two requested and actual trials |
| Search space | `ridge__alpha: [0.1, 1.0]` |
| Selected alpha | `0.1`, equal in fitted model and tuning evidence |
| CV | K-Fold, three folds |
| Seeds | Search 17; CV 42; train/holdout split 23 |
| Source rows | 121 synthetic rows |
| Pre-split | `complete_features`: drop one row missing an input |
| Preprocessing | `scaled_numeric`: StandardScaler for `x`, `z` |
| Partitions | 96 training / 24 holdout rows |
| Heldout RMSE | `0.0367010779056233` |
| Model version | `workspace.skyulf_validation_20260929.parameters_0929_r1_voting`, v1 |

The source table is `workspace.skyulf_validation_20260929.parameters_0929_r1_source`.
The test used the public `train_local_candidate` route, including Spark Delta read,
pre-split, fold-local preprocessing/tuning, heldout scoring and UC registration.
It did not change champion aliases. No existing test tables or models were deleted.

## Artifacts and scope

`training_parameters.json`, `tuning.json`, and `pre_split_filters.json` were
downloaded independently and checked against the experiment values. Model version
1 points to this exact experiment run.

The frozen wheel, notebook and manifest remain under the personal workspace path
`/Workspace/Users/edwardwolfe99@gmail.com/skyulf_validation_20260929/parameters_r1`.
Local reproduction helpers and raw receipts are ignored under
`initiatives/spark_and_mlflow/rehearsals/parameters_20260929/`.

This is one pandas voting-regression acceptance, not a new full model/strategy
matrix or lifecycle test campaign. Existing project jobs were not redeployed;
they need the updated wheel to emit the new fields on future training runs.
