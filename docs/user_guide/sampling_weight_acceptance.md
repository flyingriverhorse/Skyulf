# Single-Model Sampling Weight Acceptance

On 2026-10-03, the same single-model workflow ran twice on Databricks: SMOTE
with sample weights, then SMOTE without sample weights. Both completed source
reading, training, MLflow logging, Unity Catalog registration, registered-model
reload, holdout prediction and Delta output verification.

The test used the production Bundle project loader and candidate-training APIs
with editable `single_model.py` and `features/preprocessing.py` files. It was
submitted as an isolated serverless notebook, not as the full scheduled Bundle
lifecycle. CV, tuning, promotion and undersampling were not exercised by this
particular cloud comparison.

## Run and results

- [Databricks acceptance run](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/282657934122811/run/184285607434457): `SUCCESS`.
- [Weighted MLflow run](https://dbc-45604623-c18b.cloud.databricks.com/ml/experiments/1521897804897496/runs/1d50b15930a4451fac7613c6d3cb061f).
- [Unweighted MLflow run](https://dbc-45604623-c18b.cloud.databricks.com/ml/experiments/1521897804897496/runs/0a8f777371c84da99f383af13c213d84).

These workspace links require access to the user's Databricks workspace.

| Check | With sample weights | Without sample weights |
| --- | --- | --- |
| Source rows | 240 | Same 240 |
| Original training rows | 192 | Same 192 |
| SMOTE synthetic rows | 92 | Same 92 |
| Rows actually passed to estimator fit | 284 | Same 284 |
| Holdout predictions | 48 | Same 48 inputs |
| `WEIGHT_COLUMN` | `"importance"` | `None` |
| `synthetic_weight` | `"class_mean"` | Omitted |
| Actual `fit(sample_weight=...)` | 284 aligned weights | `None` |
| `class_weight` | `None` | `None` |
| Synthetic class-1 weight | 88.84 | No weight vector |
| UC model version | 1 | 2 |
| Maximum probability error against independent reference, in Databricks | 0 | 0 |
| Maximum error after downloading CSVs and recomputing locally | 3.33e-16 | 2.78e-16 |

The weighted and unweighted models differed by up to **0.0755992581** in holdout
class-1 probability (about **7.56 percentage points**). This demonstrates that
weights affected fitting on this fixture; it does not claim improved accuracy.

## What proves the weights were used?

1. Both cases pin the same Delta source at version 0, split seed 42 and SMOTE
   seed 42. Training/holdout key digests match, as do the sampled feature and
   label digests. Neither `importance` nor record keys appear in model features:
   the feature schema is exactly `x, z`.
2. A test-only observer copies the real `LogisticRegression.fit` inputs and then
   calls the original sklearn fit method with unchanged arguments. It asserts
   that each direct training case performs exactly one fit.
3. Independent imbalanced-learn SMOTE reconstructs the expected sampled X/y.
   The observer's inputs must match those arrays. For the weighted case,
   original weights must remain unchanged and appended weights must equal the
   original training class mean. The unweighted case must receive `None`.
4. The test reloads the registered UC model and compares coefficients, intercept,
   predicted labels and probabilities with an independently fitted sklearn
   reference. Scoring receives only feature columns.
5. All 96 holdout outputs are persisted to Delta, read back and compared with
   the expected prediction rows. Model aliases remain unchanged.
6. The MLflow evidence CSVs were downloaded outside the notebook and used to
   reconstruct SMOTE and sklearn fitting again. Both comparisons passed; the
   tiny local errors in the table are floating-point/CSV round-trip differences.

The comparison keeps `class_weight=None` in both cases to isolate sample weights.
The source has variable weights within classes, so this is not merely a check
that a uniform vector was accepted.

## Inspect the saved evidence

Each MLflow run contains:

| Artifact | Content |
| --- | --- |
| `sampling_weight_proof/actual_fit_inputs.csv` | Actual X/y, original-versus-synthetic flag, actual and expected sample weights |
| `sampling_weight_proof/original_training_rows.csv` | Pre-SMOTE training rows; weighted case includes `importance` |
| `sampling_weight_proof/holdout_predictions.csv` | Registered-model predictions, reference probabilities and errors |
| `sampling_weight_proof/proof.json` | Per-case counts, digests, settings and assertions |
| `sampling_weight_proof/comparison.json` | Combined two-case result |
| `sampling_weight_proof/resolved_workflow.json` | Settings resolved by the production project loader |
| `sampling_weight_project/` | Exact editable feature/model source files used in the case |

Resources are retained for inspection in
`workspace.skyulf_sampling_ab_20261003_fd9d6ead`:
`source`, registered `single_model`, and `predictions` (96 rows).

## Repeat the test

The notebook source is
`skyulf-core/examples/databricks_sampling_weight_comparison.py`.
Import it into your Databricks workspace as a Python source notebook and submit
a serverless notebook task with these base parameters:

```json
{
  "acceptance_id": "YOUR_UNIQUE_ALPHANUMERIC_ID",
  "experiment": "/Users/YOUR_USER/unique-sampling-weight-experiment"
}
```

Use a new ID for each run. The script deliberately creates a fresh schema rather
than overwriting an earlier test. It requires permission to create test tables,
an MLflow experiment and registered models in the `workspace` catalog.

The verified environment used the current Core wheel, `mlflow==3.16.1`,
`scikit-learn==1.8.0`, `numpy==1.26.4`, `pandas==2.3.2`, and
`imbalanced-learn==0.14.1`, with serverless environment client `4`.

- Wheel SHA256: `fba802b78a69ad0a61994b497fd6db31195b95a8c3dd55a8bf580602f6a38992`.
- Notebook SHA256: `2e56a1e4a85dc872416b122cbe8d9314e03543b8a14e8edfc88807f0d826352e`.

For normal configuration and undersampling behavior, see
[Weighted Training & Support](weighted_training.md). Undersampling selects
existing rows, so it uses their existing weights and needs no `synthetic_weight`.

## Full unweighted Bundle pipeline acceptance

On 2026-10-03, two freshly generated Bundle projects ran the complete unweighted
lifecycle on Databricks, using pandas and Polars respectively. Both used
`WEIGHT_COLUMN = None`, no sampling, StandardScaler on `x, z`, linear regression
with a two-candidate parameter search, and three-fold CV. The 240-row Delta
training source split into 192 training and 48 holdout rows.

- [Pandas training and automatic scoring handoff](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/705870002186601/run/232003443436595)
- [Polars training and automatic scoring handoff](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/991940283615220/run/169747484500332)

Every active training task succeeded, including preparation, tuning, validation,
registration, model comparison, automatic champion selection, monitoring
registration, and the child scoring job. Each scoring job wrote 240 predictions,
then completed monitoring, drift reporting and the disabled-retraining branch.
Both initial monitoring observations were `healthy`.

A separate Databricks task reloaded the registered models and read the actual
Delta predictions. It independently reconstructed the split, fitted scaling
from training rows only, and solved ordinary least squares with NumPy. It also
recomputed both candidate CV scores with scaling fitted separately in each fold.
Maximum errors were below 2.67e-15 for predictions, 8.89e-16 for fitted coefficients,
and 2.23e-16 for CV scores; both engines selected the independently best candidate.
The verification checked all 330 installed Python source files against the
worktree manifest. Wheel SHA256:
`6de1aaf3a164e1729d10e85831970bb655bfe95d6948cf61cc45f27fcf15f5b2`.

The actual score jobs were then run twice more:

| Check | Pandas | Polars |
| --- | --- | --- |
| Initial scoring | 240 predictions, Delta version 1 | Same |
| Unchanged source replay | 0 predictions, `noop=true`, version 1 | Same |
| Three new source records | Exactly 3 predictions, version 2 | Same |
| Final table | 243 unique keys | Same |
| First 240 prediction values | Original digest preserved | Original digest preserved |
| Final CSV download versus local NumPy calculation | Maximum error 3.11e-15 | Maximum error 3.11e-15 |

Both repeat score jobs completed all active tasks. Monitoring reported `healthy`
for the original population and `drift` for the final three-row batch, with zero
monitoring failures. Automatic retraining was configured as disabled, and its
conditional branch was excluded as intended. This run does not prove automatic
drift-triggered training submission, wall-clock cron firing, rollback/reject,
concurrent writers, production permissions, or every model/recipe combination.

The [final numerical verification run](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/29311890525706/run/17702981942162)
saved source/prediction CSVs and per-engine evidence in MLflow run
`6941e36d0c0742bbbb46fc37e124356e`. Downloaded CSVs were recomputed locally without
importing Skyulf. Exact run URLs, task states, numerical errors, Delta receipts,
local verification results and scope limits were captured in
`full_pipeline_unweighted_result.json`. That historical JSON capture was removed
with the examples and is no longer distributed with these docs.

The first test deployment stopped at monitoring registration because its new
test namespace lacked the required central monitoring tables. After initializing
the documented monitoring store, both pipelines were rerun from the beginning
in a fresh namespace. A separate verifier initially used a different sklearn
version and was rejected by the artifact runtime check; its successful run used
the same environment dependencies as the generated Bundle jobs.

A separate four-case cross-task competition check also passed for both engines,
with and without sample weights: the independently better candidate won, saved
predictions matched weighted/unweighted least squares, weight-only changes
reported zero fresh training rows, and a changed training label reported one.
Its receipts are included in the same JSON file.
