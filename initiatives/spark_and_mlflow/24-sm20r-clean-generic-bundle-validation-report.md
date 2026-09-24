# SM-20R clean generic Bundle validation

Date: 2026-09-24. Personal profile: `skyulf`; target: `dev` only. The clean
schema is `workspace.skyulf_bundle_first_20260924`. This is a real-data,
serverless Polars fit/predict test using `samples.nyctaxi.trips`; it does not
exercise a company workspace or classic compute.

## Reset and object inventory

Before this run, ten earlier Skyulf test jobs and three earlier Skyulf test
schemas were deleted. `jobs list` then returned no persistent jobs, and the
`workspace` catalog listed only `default` and `information_schema`.

Creating the fresh schema made no table. `bundle deploy` uploaded six files
and created exactly three persistent jobs: `skyulf_reset_dev_setup`,
`skyulf_reset_dev_train`, and `skyulf_reset_dev_score`. It created no table.
The source-preparation notebook created the one managed Delta input table
`skyulf_reset_dev_source` before training. The same source is pinned at Delta
version 1 for training and read incrementally for scoring. Source preparation
selected 600 unique real trips: 303 fit and 297 temporal holdout rows.

| Step | Job or run | UC effect | Evidence |
| --- | --- | --- | --- |
| Prepare source | One-time `1121337641193762` | One CDF-enabled source table, 600 rows, version 1 | [run](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/998376432225508/run/1121337641193762) |
| Deploy | Bundle `dev` | Three persistent jobs, zero tables | `bundle deploy` output: 3 created, 6 files uploaded |
| Train | `998255871992406` | UC model `skyulf_reset_dev_model` version 1 and MLflow run | [run](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/402117143173384/run/998255871992406) |
| Setup | `447359392118318` | One empty prediction table and one internal score-control table | [run](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/5775604932565/run/447359392118318) |
| First score | `630863655201312` | 600 predictions in the same table | [run](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/1089013339073761/run/630863655201312) |
| Append source | One-time `861959622894913` | 50 more source rows; source total 650 | [run](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/352133155519821/run/861959622894913) |
| Second score | `858963997908575` | Only 50 new predictions | [run](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/1089013339073761/run/858963997908575) |
| Unchanged replay | `661436133664918` | No new prediction or target commit | [run](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/1089013339073761/run/661436133664918) |
| Final verification | One-time `523752795761361` | Read-only check: 650 unique predictions using model version 1 | [run](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/842299042681083/run/523752795761361) |

The training run used Polars `SimpleImputer`, `StandardScaler` and a Skyulf
linear-regression model. The registered version was 1, with 297 held-out rows.
MLflow reported MAE `0.6844`, RMSE `2.5676`, R2 `0.9363`, MAPE `0.0621`
and explained variance `0.9363`. An independent `experiments get-run` query
confirmed all six held-out metrics persisted on MLflow run
`6c5c58c5d44045e89ae0054bbbec4a27`. The first score receipt reported
`input_count=600`, `output_count=600`, source end version 1, target commit
version 1, `noop=false`. The second score receipt reported source version 2,
`input_count=50`, `output_count=50`, target commit version 2 and `noop=false`.
The unchanged replay reported `input_count=0`, `output_count=0`, target commit
version still 2 and `noop=true`.

After setup, an independent `tables list` showed exactly these three managed
tables in the schema: `skyulf_reset_dev_source`,
`skyulf_reset_dev_predictions`, `skyulf_reset_dev_score_admission`. No
alias-control table was created because lifecycle jobs were disabled. The
registered model is a UC model, not another table.

## Scope and remaining checks

The generic template renders `dev`, `test`, `syst`, `prod` with distinct target
bindings. Serverless and policy-cluster variants passed strict `dev` Bundle
validation. Six other target *shapes* passed strict validation using temporary
placeholder substitutions, not real company configuration. Only personal
serverless `dev` was deployed. The policy-backed cluster variant, optional
champion/challenger jobs, company hosts/catalogs and Spark-native work remain
unverified by this clean run.

The final read-only notebook found 650 source and 650 prediction rows, 650
distinct prediction keys, and only registered model version 1 in the output.
The final CLI inventory at the time listed exactly the three Bundle jobs and
the three tables above. SM-20S later removed those three old jobs and the old
prediction/control tables after its two-job validation; see
[SM-20S evidence](26-sm20s-two-job-live-validation-report.md).

Local regression tests: `13 passed` with a workspace-local pytest
temp directory. Ruff and Ty passed on the changed Python files; strict MkDocs
build passed with the existing unlisted segmentation-page notice.
