# Local-engine Databricks Bundle

The custom Skyulf template generates one editable Bundle with `dev`, `test`,
`syst` and `prod` targets. It fits and predicts with pandas or Polars. Spark
reads bounded Unity Catalog Delta rows and publishes predictions; local
feature engineering and model prediction are not distributed Spark work.

Initialize a project from a Skyulf checkout:

```powershell
databricks bundle init templates/databricks --output-dir ./generated
```

The short path asks for project name, engine, serverless or policy-backed job
compute and the existing `dev` catalog/schema. Serverless is the default. Reviewable
noninteractive examples are in `templates/databricks/examples/`. The generated
project has its own `databricks.yml`; Skyulf's root has no Bundle config.

`dev` uses the selected CLI profile's workspace host. `test`, `syst` and
`prod` each have a different placeholder host and catalog in the generated
file. Edit each target's host, catalog and input/output/metadata schema
bindings, then validate with its designated profile. No Danske, Danica or
personal-workspace value is built into those three targets. Policy-backed
compute uses the policy name, runtime, node type and cost tag supplied at
initialization. Serverless compute needs none of those fields.

## What is created

`bundle deploy` creates only two jobs and uploads their code:

| Job | Purpose | UC objects created when run |
| --- | --- | --- |
| `train` | Fit one candidate and log held-out metrics | One registered model/version |
| `score` | Verify the source and pinned model, create output if absent, then score initial and later CDF inserts | One prediction table, only if absent; rows in that table |

No schedule, endpoint or Unity Catalog table is created by deployment alone.
The starting Bundle has no alias lifecycle job. Skyulf Core's registry
comparison and promotion services remain available for a separately designed
champion/challenger workflow.

The four relevant Unity Catalog names have different roles:

| Name | Meaning |
| --- | --- |
| `training_table` | Existing labeled input table, pinned to a Delta version for training |
| `score_source_table` | Existing CDF-enabled source of rows to predict |
| `prediction_table` | The one output table for this model |
| `model_name` | Registered Unity Catalog model, not a table |

The first two references point to **the same existing table by default**.
Neither reference creates a table. Separate them only when labeled training
data and new scoring data have different lifecycles. If they stay together,
the first `score` run predicts all existing rows, including historical labeled
rows; review whether that is intended for your use case.

## First run

Build and place the matching Skyulf wheel in the generated project's `dist/`,
then edit `config/workflow.json` for real source columns, preprocessing, model,
temporal split and size limits. The JSON values are an example, not a dataset.
Enable Change Data Feed on the scoring source before later inserts arrive.

```powershell
databricks bundle validate --strict -t dev --profile <profile>
databricks bundle deploy -t dev --profile <profile>
databricks bundle run train -t dev --profile <profile>
```

Inspect the registered model version, put that concrete value in
`model_version`, redeploy the changed JSON, then run `score`. The first score
rejects a missing source, disabled CDF, unsuitable row keys, a model output
mismatch or an existing target schema mismatch before it creates the single
prediction table. It checks initial row count against `max_rows`; each score
also checks decoded transfer bytes against `max_bytes`. Existing tables are
never overwritten. The first score processes the current source
snapshot; later runs process only new inserts since the committed Delta
receipt. A repeat without new rows is a no-op. No monthly date or source
version is entered for each run.

The first candidate does not become champion automatically. Scoring stays
pinned to its configured model version; changing aliases does not rewrite old
predictions. `max_concurrent_runs: 1` serializes the generated score job, but
does not coordinate other jobs. Grant prediction-table writes only to this
job's identity. For multiple publishers, use Skyulf Core's shared admission
provider. Full-history rescore, schedules, online endpoints and Spark-native
FE/model execution are separate work.

The older SM-20a personal serverless rehearsal passed, but its jobs and test
schemas were removed at the user's request. The subsequent clean generic
`dev` rehearsal trained a Polars model from 600 real taxi rows, wrote 600
initial and 50 later predictions to one table, and replayed without another
commit. The personal test resources remain available for inspection. The
That rehearsal used the earlier three-job template; the two-job design needs
its own live validation. The `test`, `syst` and `prod` placeholders have not
been deployed in a company workspace.
