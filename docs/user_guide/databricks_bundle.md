# Local-engine Databricks Bundle

Skyulf's custom template creates a self-contained Databricks Bundle project
with separate jobs for candidate training, read-only model comparison,
challenger staging, explicit promotion and incremental batch scoring. Choose
`pandas` or `polars` when generating the project. The selected engine fits
feature engineering and the model, then applies the saved pipeline at scoring
time. Spark reads bounded Unity Catalog Delta rows and writes predictions;
this first Bundle does not distribute local feature engineering across Spark
workers.

Generate a project from a Skyulf checkout:

```powershell
databricks bundle init templates/databricks --output-dir ./generated
```

The four questions are project name, local engine, catalog and schema. The
generated project contains its own `databricks.yml`, `config/workflow.json`,
Skyulf job entry point and five job resources. It does not put a
`databricks.yml` in the Skyulf repository root. Follow the generated README
to build and place the matching Skyulf wheel in `dist/`, select real tables,
features, preprocessing/model configuration and temporal split, then provision
the source, target and admission tables. The generated JSON is an editable
starting point; there is no claim that arbitrary feature engineering can be
inferred from a table name.

```powershell
databricks bundle validate --strict -t dev --profile <profile>
databricks bundle deploy -t dev --profile <profile>
databricks bundle run train -t dev --profile <profile>
```

Training pins a labeled Delta snapshot and temporal holdout. Skyulf logs the
fitted artifact and held-out metrics to MLflow and registers a concrete Unity
Catalog model version. Inspect that version and set `model_version` in the
config before deploying again and running `score`. The score job reads an
existing CDF-enabled Delta source, prepares the pinned local pipeline once,
and scores the initial bounded snapshot. Later runs read only new inserts
since the last committed Delta receipt. They require no date or source-version
input. An empty repeat is a no-op. Source updates and deletes are rejected
until a rescore policy is chosen. Keep the same admission control row for all
writers to the target.

```powershell
databricks bundle run score -t dev --profile <profile>
# Add new rows to the source Delta table.
databricks bundle run score -t dev --profile <profile>
```

`compare` reads two configured concrete versions against the same pinned
holdout. It never changes aliases. With a separately initialized `@champion`
alias and a provisioned alias admission row, `stage` can assign an eligible
candidate to `@challenger`; `promote` then moves it to `@champion` and stores
the prior version as `@previous_champion`. These are explicit jobs, and they
recheck the comparison and expected champion version. The first candidate
does not become champion automatically. The scoring job continues to use its
configured concrete version until the operator changes it. Alias promotion
does not rewrite old prediction rows.

The template has no schedule. Add a scoring schedule only after checking the
source's append policy and data permissions. Monthly retraining scheduling,
full-history rescores, Spark-native execution and endpoints are separate
extensions. See [the local SDK](databricks_local_sdk.md) for the underlying
configuration and [MLflow registry](mlflow_registry.md) for model resolution.
