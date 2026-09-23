# SM-24a live Databricks validation plan

Status: completed for the documented small-data scope on 2026-09-23. The
[live report](08-sm24a-live-validation-report.md) records concrete evidence.
This is a validation gate for the reusable local training and bounded batch
workflow, not the first Databricks Bundle or the monthly Delta sink.

## Test area and data

Use the previously selected `skyulf` CLI profile and its workspace. The user
selected the new isolated UC schema `workspace.skyulf_sm24a_20260923` on
2026-09-23. Use names prefixed `skyulf_sm24a_` and retain the final
resource list and job IDs in the validation report.

Create two small managed Delta source tables with a fixed seed:

| Table | Purpose | Initial size |
| --- | --- | ---: |
| `skyulf_sm24a_train_source` | Explicit training/validation rows with regression and classification labels | about 1,000 rows |
| `skyulf_sm24a_score_source` | New entities with the same raw feature columns but no labels; two monthly periods | about 80 rows per period |

Use distinct entity IDs across training and scoring. Include numeric nulls and
outliers, skewed categorical values, an unseen scoring category, timestamps,
and deterministic target rules. Record both Delta source versions. Training
must not read labels from the scoring table, and scoring must not accept label
columns. The second table is the user's requested independent inference input.
SM-24a returns predictions and a diagnostic report; SM-15L will later publish
them to a separate UC Delta prediction table.

## Model and preprocessing gate

First prove each selected configuration locally on both its fit engine and a
fresh loaded artifact. The live training job then fits five separate pipelines
on Databricks; a separate scoring job loads their concrete UC model versions.
These are planned cases, not claims of current Databricks parity:

| Case | Fit engine | Model family | Preprocessing under test |
| --- | --- | --- | --- |
| R1 | pandas | `linear_regression` | numeric imputation, standard scaling, feature math |
| R2 | Polars | `random_forest_regressor` | numeric bounds, custom binning, one-hot encoding |
| C1 | pandas | `logistic_regression` | imputation, ordinal encoding, robust scaling |
| C2 | Polars | `random_forest_classifier` | categorical encoding, numeric scaling, interaction |
| R3 | pandas | `gradient_boosting_regressor` | KNN imputation, selection, power or general transform |

The exact node parameters and order are fixed only after local fit, save, load
and prediction parity succeeds; incompatible combinations must be corrected or
reported, never quietly dropped. Regression outputs are compared by row key
within a stated numeric tolerance. Classification labels must match exactly;
probability column order must match saved classes and each row must sum to one.
Include null and unseen-category scoring rows in the comparison.

The five pipelines cannot certify all preprocessing registrations. The current
[inventory](NODE_SUPPORT.md) lists 62 preprocessing IDs, including aliases,
optional dependencies, inspection/split steps, resampling and other train-only
operations. Add a machine-readable per-ID report with one row for every ID:
`inference_eligible`, `train_only`, `inspection`, `optional_dependency`, or
`unsupported`, plus local result, live result, parameters and reason. Run
fit/save/load/predict probes on Databricks for each inference-eligible local
node that can use a bounded fixture; test train-only effects separately without
claiming they run at inference. A failed or inapplicable row remains visible.
Broader fixes remain in SM-17 after the first Bundle, but the audit itself is
part of this gate.

The [current 62-ID matrix](07-sm24a-node-matrix.csv) records all unrun rows
explicitly. A category is a triage judgment, not a claim of full compatibility.

## Cross-job sequence

1. Provision the two test tables once. Record their schema, row counts, distinct
   keys and Delta versions. Keep training and scoring rows disjoint.
2. Run the training job on Databricks serverless with the tested Skyulf wheel
   and pinned dependencies. Give it explicit train/validation inputs. Package
   each fitted local pipeline, log it to MLflow, and register five test UC
   model versions. Do not set or move a champion alias.
3. Run a separate scoring job. For one requested month and pinned source
   version, filter the Delta source before moving data to the Python driver.
   Project only row keys and required raw input columns. Apply `max_rows + 1`
   in Spark, then enforce decoded serialized-row and local-frame byte budgets.
   Never call unbounded `collect()` or `toPandas()` on the full table. The
   selected Spark iterator does not report exact wire bytes; a single wide row
   can arrive before the decoded-byte check rejects it. This is an explicit
   small-data boundary, not a hard network-byte guarantee. A workload that
   requires a hard transport limit needs a separate proven adapter before use.
4. Resolve each chosen model once to a concrete version, load the trusted
   package, run local pandas/Polars prediction and retain row keys separately.
   Compare with the training job's saved pre-package reference predictions and
   a small direct-estimator oracle. Record model version/digest, code version,
   source version, UTC period boundaries, timezone, counts and comparison.
5. Repeat for the second month. Verify the first month's input is not rescored.
   Re-run one identical request to check deterministic prediction and identity.
   Publication/replay of a Delta prediction table belongs to SM-15L.

Negative cases: reordered/missing input columns, wrong model artifact kind,
incompatible runtime, an over-row and over-byte source, unknown model version,
and a changed alias after pinning. Each must fail with a bounded, actionable
result before an unbounded read or any target write. `max_bytes` means decoded
payload/local-frame bytes, not Spark wire bytes. Keep serverless tasks
time-bounded, record actual compute use, and never modify production tables.

## Completion evidence

SM-24a is ready to close only after local and live reports identify the exact
wheel/dependency versions, UC source versions, five concrete registered-model
versions, the per-node matrix, passed/failed cases and retained test resources.
The report must distinguish the demonstrated five-model path from untested
model families. It must not claim monthly Delta publication or Bundle deployment.

Platform references: [Delta versioned reads](https://docs.databricks.com/aws/en/delta/tutorial),
[serverless limitations](https://docs.databricks.com/aws/en/compute/serverless/limitations),
and [serverless task dependencies](https://docs.databricks.com/aws/en/compute/serverless/dependencies).
