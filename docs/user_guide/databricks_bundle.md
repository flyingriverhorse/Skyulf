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

Choose `personal` for a serverless `dev` target, or `company` for
policy-backed `test`, `syst` and `prod` targets. Personal mode asks for project
name, local engine, catalog and schema. Company mode asks for the workspace
host, target catalogs, input/metadata/result schemas, approved cluster policy
and `PayingRegNo`. Supply reviewed answers with `bundle init --config-file`
when the team has them; the company examples in the questionnaire are not
verified production settings. The
generated project contains its own `databricks.yml`, `config/workflow.json`,
Skyulf job entry point and five job resources. It does not put a
`databricks.yml` in the Skyulf repository root. Follow the generated README
to build and place the matching Skyulf wheel in `dist/`, select real tables,
features, preprocessing/model configuration and temporal split, then provision
the source, target and admission tables. The generated JSON is an editable
starting point; there is no claim that arbitrary feature engineering can be
inferred from a table name.

For company mode, copy and review
`templates/databricks/examples/company-init.example.json`, especially its
placeholder host and policy/tag values, then initialize with:

```powershell
databricks bundle init templates/databricks --config-file <reviewed-config.json> --output-dir ./generated
```

| Job | Purpose | Normal cadence |
| --- | --- | --- |
| `train` | Fit and register a candidate with held-out MLflow metrics | When labeled data warrants a new model |
| `compare` | Read-only candidate/champion check | During model review |
| `stage` | Assign an eligible `@challenger` | Explicit release step |
| `promote` | Move staged candidate to `@champion` | Explicit approval step |
| `score` | Append predictions for new source inserts | Every scoring run |

Only `score` is needed for recurring inference. A generated project has one
prediction table per model. Its existing labeled/source tables are inputs; the
score and alias admission tables are internal coordination state in the
metadata schema. They are not additional prediction outputs. Company `test`
adds a per-user suffix to output/model/control names, while `syst` and `prod`
use stable names in their respective catalogs. The notebook verifies that
these writes stay in the active target's catalog and schema.

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

The personal `dev` Bundle was live-tested on serverless compute. Company
targets are generated but have not been validated or run in the company
workspace. Before a company rollout, verify host/catalog/policy/tag values,
validate each target with its own CLI profile and perform an isolated `test`
run. A different company workspace per target requires editing each target's
`workspace.host` after initialization.
