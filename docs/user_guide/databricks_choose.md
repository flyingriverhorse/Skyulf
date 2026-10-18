# Choose a project shape

Three questions decide most of a new Databricks project. Answer them before
`databricks bundle init`; each answer links to the page that owns the details.
The closest [NYC taxi example](databricks_project_setup.md#start-from-the-closest-example)
is named where one exists.

## Single model, competition or model set?

| You have | Choose | Closest example | Details |
| --- | --- | --- | --- |
| One target and one model you already picked | `single_model` | `taxi_fare` | [Training configuration](databricks_training_config.md) |
| One target and 2-8 candidate algorithms to compare | `model_competition`: CV picks one winner | `taxi_high_fare` | [Training configuration](databricks_training_config.md) |
| 2-8 targets approved and scored together | `multi_target`: one versioned model set | `taxi_trips` | [Model sets](databricks_model_sets.md) |
| One candidate that combines several learners | A voting or stacking model in any layout | none | [Ensembles](databricks_ensembles.md) |

A competition winner that fails its holdout gate does not promote the runner-up.
The layout shapes the generated train job, so choose it at `bundle init`; see
[layouts and setup modes](databricks_project_setup.md#choose-a-layout-and-setup-mode).

## Local batch, Spark batch, REST or `ai_query`?

| Who reads the predictions | Choose | Closest example | Details |
| --- | --- | --- | --- |
| A scheduled job writing a Delta table within `max_rows`/`max_input_mb` | `score` job, `inference_mode: local` (default) | `taxi_high_fare` | [Scoring](databricks_scoring.md) |
| A scheduled job on a table too large for one Python process | `score` job, `inference_mode: spark` | `taxi_fare` | [Spark scoring](databricks_spark_scoring.md) |
| An application sending raw feature values | REST serving endpoint | `taxi_fare_robust` | [Serving](databricks_serving.md) |
| SQL users, SQL jobs and dashboards | SQL function calling the endpoint with `ai_query` | `taxi_fare_robust` | [SQL function](databricks_serving.md#3-create-the-sql-function) |
| Requests that send only an entity key | Endpoint with online feature lookup | `taxi_fare_online` | [Online features](databricks_online_features.md) |

The batch job never calls an endpoint, and an endpoint never writes the
prediction table. Models packaged with feature lookup score in batch only with
`inference_mode: spark`. Runtime and cost per path are compared in
[how predictions are produced](databricks_serving.md#choose-how-predictions-are-produced).

For rows excluded by preprocessing at scoring, start with `taxi_fare_robust`.
Its Spark and raw-input endpoint examples retain row identity and explain
`excluded` status, null predictions and the resulting monitoring coverage.

## Which steps run on Spark workers and endpoints?

Spark workers and serving endpoints both run the saved pandas pipeline on
independent batches, so the same admission applies to both.

| Part | On Spark workers and endpoints | Details |
| --- | --- | --- |
| Training, tuning and CV | Never distributed: one bounded pandas or Polars process | [Spark scoring](databricks_spark_scoring.md) |
| Built-in models | Reviewed linear/logistic/tree models and `XGBRegressor`; additional models and voting/stacking with matching training probe evidence | [Support table](databricks_spark_scoring.md#what-spark-scoring-supports) |
| Built-in preprocessing steps | Reviewed steps (including median imputers), plus eligible row-preserving recipes with matching training probe evidence | [Recipe boundaries](databricks_spark_scoring.md#built-in-recipes-and-model-evidence) |
| Your `column_step` and `fitted_step` functions, saved scoring rules | Admitted when training lints and probes them | [When the probe fails](databricks_spark_scoring.md#when-the-probe-fails) |
| Steps that need other rows (`group`, `window`, `global`), `custom_step` classes | Rejected; keep `inference_mode: local` | [Custom steps](databricks_custom_steps.md) |

A single-model or competition Spark project rejects an unsupported model or step
before registration, so the failure appears in the train job, not in the first
score run. A model set is checked when it is activated and again at scoring.
