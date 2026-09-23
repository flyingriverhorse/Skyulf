# SM-22a Databricks metrics and comparison validation

Status: LIVE VERIFIED, 2026-09-23. This is an isolated, bounded serverless
validation of Skyulf Core held-out metrics, MLflow tracking, and a pinned
Unity Catalog model comparison. It does not promote an alias.

## Execution

- Workspace profile: `skyulf`; existing test schema:
  `workspace.skyulf_sm24a_20260923`.
- [Serverless job run](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/338095925832270/run/402693440245633):
  parent run `402693440245633`, task run `779149119285800`,
  `TERMINATED / SUCCESS`, one attempt, 15-minute task timeout.
- Notebook: `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_sm24a_20260923/sm22_metrics_r1/live_metrics`;
  uploaded from `skyulf-core/examples/databricks_local_sm22_metrics_live.py` at commit
  `68d90c0a`. The current example now calls `SkyulfPipeline.fit` directly;
  that refactor passed a local pandas/Polars artifact smoke test but was not
  rerun on Databricks.
- Wheel: `skyulf_core-0.9.0-py3-none-any.whl`, SHA-256
  `B8ACD810FAF0AC07E1A066B4E186F485CFD4B0A60DADDF7C24F45BDD2C39050C`.
- Runtime: skyulf-core 0.9.0, MLflow 3.16.1, pandas 2.2.3,
  Polars 1.44.2, scikit-learn 1.6.1.

The notebook fitted actual Skyulf local pipeline artifacts for each engine
and task, evaluated held-out data through `evaluate_local_holdout`, logged
all finite metrics, then read each MLflow run back and compared every value.
An independent `databricks experiments get-run` check confirmed all four
runs were `FINISHED` and retained the metric keys shown below.

| Case | MLflow run ID | Persisted metrics | Selected values |
| --- | --- | ---: | --- |
| pandas regression | `546df4fed4d443999990707df7255335` | 6 | MAE `4.54747e-14`, RMSE `5.08423e-14`, R2 `1.0` |
| Polars regression | `dc5cdc23ad424e61b143d2d0b9410697` | 6 | MAE `4.54747e-14`, RMSE `5.08423e-14`, R2 `1.0` |
| pandas classification | `856b7d03794143b78c6f533374e938e2` | 12 | accuracy `1.0`, F1 `1.0`, log loss `0.0969631`, ROC AUC `1.0` |
| Polars classification | `c703eb554cef48bf914d8ce082d0d9b7` | 12 | accuracy `1.0`, F1 `1.0`, log loss `0.0969631`, ROC AUC `1.0` |

The six regression keys were `heldout_mae`, `heldout_mse`, `heldout_rmse`,
`heldout_r2`, `heldout_mape`, and `heldout_explained_variance`. The twelve
classification keys were `heldout_accuracy`, `heldout_balanced_accuracy`,
`heldout_precision_weighted`, `heldout_recall_weighted`,
`heldout_f1_weighted`, `heldout_matthews_corrcoef`, `heldout_precision`,
`heldout_recall`, `heldout_f1`, `heldout_log_loss`, `heldout_roc_auc`, and
`heldout_pr_auc`.

## Pinned Unity Catalog comparison

The test registered two real local pipeline packages in
`workspace.skyulf_sm24a_20260923.skyulf_sm22_metrics_r1` and resolved their
concrete versions. Version 1 was the reference model, trained with a deliberate
target offset; version 2 was the candidate. On the same synthetic labeled
holdout, candidate `heldout_mse = 2.58494e-27` versus reference
`heldout_mse = 100.00000000000088`. With `min_improvement=1.0` and
`quality_threshold=1.0`, the report returned `eligible=true`. The reference
model's MLflow run was `3fcb9236b2434eb1bec2eb42af8b1d45`. CLI registry
checks independently found the two model versions; version 2 was `READY`.

"Reference" or "champion" here means a pinned comparison version. No
`@champion` alias was created, changed, or tested. SM-22b owns alias promotion
and rollback.

## Boundary

The data are deterministic synthetic frames, not a production dataset. This
proves finite metric calculation and MLflow persistence for these four cases
on one Databricks serverless runtime, plus one pinned UC comparison. It does
not prove performance on other data, alias concurrency, Delta admission,
prediction-table writes, or a Bundle. The experiment, five runs, two UC model
versions, uploaded notebook and wheel remain in the isolated test area.
