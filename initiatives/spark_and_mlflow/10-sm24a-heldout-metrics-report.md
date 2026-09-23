# SM-24a held-out metrics in the model MLflow run

Status: completed on 2026-09-23 for the isolated small-data Databricks workflow. The [notebook](../../skyulf-core/examples/databricks_local_sm24a_job.py) now evaluates the saved local pipeline on the 200 rows reserved from each 1,000-row training source and logs the resulting metrics in the same MLflow run as its model artifact. The reusable calculation is `evaluate_local_holdout` in the local Databricks integration.

## Split and metric contract

- Source: `workspace.skyulf_sm24a_20260923.skyulf_sm24a_train_source`, Delta version 0. The bounded reader returns 1,000 rows ordered by `entity_id`; the first 800 are used for fit and the remaining 200 for held-out evaluation. The test labels are never inputs to prediction. The split is deterministic by key, neither randomized nor a true time split.
- The saved artifact predicts the raw held-out features through its recorded pandas or Polars engine. The evaluator does not refit feature engineering or the model. Regression logs `heldout_mae`, `heldout_rmse`, `heldout_r2`. Binary classification logs `heldout_accuracy`, `heldout_f1` for the model's second class, and `heldout_f1_weighted`.
- Within one `mlflow.start_run`, the notebook logs the source/split parameters and held-out metrics, then passes `run.info.run_id` to `log_local_model`. Registration uses the resulting `runs:/<run_id>/model` URI and a concrete UC model version. No alias is moved.

## Live results

Training parent run `579498050611865`, task `209306932523734`: **SUCCESS**, 900-second timeout, zero retries. It used the existing version-0 source tables and registered new test-only `skyulf_sm24a_metrics_*` model names at version 1. The wheel SHA-256 is `3548c90bf34bae417716ad176c3ccfdb9d85d73b270999672c3a101f5971bdee`.

| Case | Engine / model | MLflow run ID | 200-row held-out metrics |
| --- | --- | --- | --- |
| R1 | pandas / linear regression | `fc3fe38d14854198956d143415bb856e` | MAE 0.673; RMSE 0.819; R2 0.993 |
| R2 | Polars / random forest regression | `66c68cd3618f45ff95beed7c16bbb503` | MAE 20.379; RMSE 22.531; R2 -4.299 |
| C1 | pandas / logistic regression | `418d55e524954227b4d24d43d7090165` | Accuracy 0.710; binary F1 0.773; weighted F1 0.711 |
| C2 | Polars / random forest classification | `934ee10da4ea4182b9ae76d151136e19` | Accuracy 0.575; binary F1 0.706; weighted F1 0.538 |
| R3 | pandas / gradient boosting regression | `910362725ecc472c870beeed4e040890` | MAE 16.855; RMSE 19.438; R2 -2.944 |

The Databricks tracking API confirmed, for **each** of the five finished runs, its held-out metric keys, `heldout_rows=200` parameter and `model` artifact directory. The UC registry API confirmed all five version-1 records were READY and each pointed to the matching `runs:/<run_id>/model` URI. The full unrounded values, model digests and versions are in `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_sm24a_20260923/metrics_r1/training_receipt.json` and local `.cache/sm24a-metrics-receipt.json`.

Independent scoring parent run `1112210778543224`, task `655818910990971`: **SUCCESS**. It loaded the five new concrete model versions and scored the pinned version-0 monthly source. Each model returned all 80 January and 80 February entity keys; all 10 model-month outputs matched the training job's saved-artifact reference. This proves the new metered packages still load and score in a separate job. The reference is a replay comparison, not an independent model-quality oracle.

The negative R2 values and the C2 classification scores show that some example models generalize poorly to this key-ordered held-out set. The logging contract works; these scores are **not** a model-quality approval for production. Model selection, representative splitting and promotion thresholds require separate decisions. Earlier SM-24a runs remain unchanged and do not gain metrics retroactively.

## Validation and scope

The focused local Databricks batch/five-model gate passed **14 tests, one optional skip**. A full Core test run reached **10,142 passed and 348 skipped**, with one unrelated failure: the optional SentenceEmbedder wrapped-Polars test attempted a Hugging Face download and the current sandbox denied network access (`WinError 10013`). The new evaluator's regression and classification tests passed on both pandas and Polars. Ruff, Ruff format and Ty passed for the changed Python files.

This audit did not create a prediction Delta table, deploy a Bundle, change an alias, or test an HTTP serving endpoint. SM-15L remains the next monthly Delta-publication task. Its infrastructure test may use these pinned example models, but their held-out scores should be reviewed before any production promotion.
