# Integrations documentation index

Find every Databricks, MLflow and Spark guide here, including the preprocessing
catalog used by generated projects. Each link includes a short description so
you can choose a page without reading the whole documentation set.

For a first project, follow **setup → data → training → model approval →
scoring → monitoring**. Add serving and online features when you need online
requests. The guides below explain their own prerequisites, supported execution
paths and configuration fields.

## Pick a starting point

| What you want to do | Start here |
| --- | --- |
| Build a project from source data to saved predictions | [End-to-end project guide](databricks_bundle_walkthrough.md): follow one complete example. |
| Decide between one model, competition and multiple targets | [Choose a project shape](databricks_choose.md): compare layouts and prediction routes. |
| Configure preprocessing without writing Python | [Build preprocessing with YAML](preprocessing_yaml_walkthrough.md): use complete recipe files and inspect saved predictions. |
| Decide which rows to predict and what business columns to return | [Scoring and publication](databricks_scoring.md): configure `eligibility`, `outputs`, model selection and the output table. For cross-model `combined` rules, use [model sets](databricks_model_sets.md). |
| Score a large table on Spark workers | [Spark scoring](databricks_spark_scoring.md): check admission, worker dependencies and execution settings. |
| Query a model through REST or SQL | [Serving endpoints](databricks_serving.md): set up the endpoint and use REST or `ai_query`. |
| Find drift, performance loss or serving problems | [Monitoring overview](databricks_monitoring.md): follow the store, enrollment, observations and dashboard flow. |
| Use Python without generating a Bundle | [Python SDK](databricks_sdk.md): fit, inspect and score from Python; choose the cloud operations you need. |

## Project setup and data preparation

| Guide | What it covers |
| --- | --- |
| [End-to-end project guide](databricks_bundle_walkthrough.md) | A complete project from prerequisites and source rows through training, publication and monitoring, with optional feature and serving stages. |
| [Databricks Bundle topic index](databricks_bundle.md) | Direct links to individual Bundle configuration topics and the public guides referenced by generated project documentation. |
| [Choose a project shape](databricks_choose.md) | Single-model, competition and model-set layouts; batch, Spark, REST and SQL prediction choices. |
| [Project setup and preview](databricks_project_setup.md) | Example selection, setup questions, generated files, smoke checks and offline previews before deployment. |
| [Tables and objects in your workspace](databricks_workspace_tables.md) | Where source data, predictions, feature tables and monitoring records live, and which project settings name them. |
| [Check or create source tables](databricks_source_tables.md) | Validate existing input tables or prepare missing ones using the project's source definitions. |
| [Feature tables and merges](databricks_features.md) | Optional feature groups, Spark transforms, join keys and merged inputs for training and scoring. |
| [Deployment and access](databricks_deployment.md) | Bundle deployment, environments, package and compute settings, job resources and execution identities. |

## Training and model choices

| Guide | What it covers |
| --- | --- |
| [Training data and holdout](databricks_training_data.md) | Source windows, labels and their availability, sampling, evaluation splits, holdout selection and bounded input limits. |
| [Training YAML reference](databricks_training_config.md) | Accepted `training.yml` fields, defaults, inheritance and complete files for the supported project layouts. |
| [Model and search-space catalog](databricks_model_catalog.md) | Available classifiers and regressors, model parameters, search spaces and optional dependencies. |
| [Search strategies](databricks_tuning.md) | Grid, random, halving and Optuna searches, cross-validation and how the choices affect total training work. |
| [Voting and stacking ensembles](databricks_ensembles.md) | Ensemble members, nested parameter searches, calibration examples and execution boundaries. |
| [Classification decision thresholds](databricks_decision_thresholds.md) | Native, fixed and calibrated decision policies and how probabilities become predicted classes. |
| [Multi-target model sets](databricks_model_sets.md) | Independent target branches, saved component versions, set approval, output modes and cross-model `combined` rules. |

## Preprocessing recipes and custom steps

| Guide | What it covers |
| --- | --- |
| [Build preprocessing with YAML](preprocessing_yaml_walkthrough.md) | Complete pre-split and fitted recipes, recipe selection and checks that saved inference reuses the learned state. |
| [Preprocessing and pre-split catalog](preprocessing_recipes.md) | The registered-step index, phase selection and how to read each step's parameter and output contract. |
| [Missing values, cleaning and types](preprocessing_recipes_cleaning.md) | Imputation, missing indicators, dropping, deduplication, value replacement, text cleaning and casting. |
| [Scaling and numeric transformations](preprocessing_recipes_scaling.md) | Scalers, normalization, quantile and power transforms, and fixed numeric transformations. |
| [Bounds and outliers](preprocessing_recipes_outliers.md) | Fixed bounds, clipping and learned outlier filters, including the distinction between changed values and excluded rows. |
| [Categorical encoding](preprocessing_recipes_encoding.md) | One-hot, ordinal, frequency, target and other encoders, rare-category grouping and unseen-category behavior. |
| [Binning](preprocessing_recipes_binning.md) | Fixed or learned bin edges, ordinal codes, labels and shared output settings. |
| [Feature generation and selection](preprocessing_recipes_features.md) | Interactions, polynomial and formula features, plus variance, correlation, statistical and model-based selection. |
| [Time and geography](preprocessing_recipes_time_geo.md) | Calendar features, lag and rolling features, required history, geographic distance and H3 cells. |
| [Text and embeddings](preprocessing_recipes_text.md) | Tokenization, count and TF-IDF vectors, hashing and sentence embeddings with their dependencies. |
| [Split, sampling and diagnostic nodes](preprocessing_recipes_sampling.md) | Core split, resampling and inspection nodes, including which belong outside generated Bundle recipes. |
| [Custom factory contracts and ready functions](preprocessing_recipes_custom.md) | Function wrappers, shipped helper factories, captured assets and the contracts custom steps must follow. |
| [Write custom preprocessing steps](databricks_custom_steps.md) | Implement project Python factories, select them in YAML and preserve their source with the saved model. |
| [Write custom pre-split filters](databricks_custom_pre_split.md) | Define row-filter functions, configure their parameters and understand training and scoring replay. |
| [Preprocessing diagnostics and context](preprocessing_context.md) | Probe repeated, reordered and partitioned inputs; understand row, group and history requirements and history continuation. |

## Training results and model lifecycle

| Guide | What it covers |
| --- | --- |
| [Training reports and charts](databricks_training_reports.md) | Locate evaluation metrics, chosen parameters, reports, charts and task failures after a run. |
| [Experiment artifacts](databricks_artifacts.md) | Explain each training JSON file, the fitted model package and lifecycle records; distinguish configuration, data fingerprints and decisions. |
| [SHAP explanations in Databricks](databricks_explanations.md) | Enable explanations and locate and interpret saved feature and individual-prediction reports. |
| [Approve, reject and roll back](databricks_lifecycle.md) | Compare candidates, apply quality gates, control champion/challenger aliases and reverse a promotion. |
| [Interrupted alias recovery](databricks_lifecycle_recovery.md) | Inspect and recover an interrupted model-version alias operation. |

## Batch scoring and prediction publication

| Guide | What it covers |
| --- | --- |
| [Scoring and publication](databricks_scoring.md) | Input and output tables, model selection, incremental startup, saved eligibility/output rules and prediction receipts. |
| [Score on Spark workers](databricks_spark_scoring.md) | `batch_compute: spark`, supported saved models and recipes, worker environments, admission checks and model-set scoring. |
| [Databricks Python SDK](databricks_sdk.md) | Python entry points for fitting, inspecting and scoring, followed by links to individual cloud workflows. |
| [Monthly batch publication](databricks_batch.md) | Score a pinned source snapshot, replace one calendar period atomically and retry without duplicating a committed write. |

## Online serving and feature lookup

| Guide | What it covers |
| --- | --- |
| [Endpoint setup and queries](databricks_serving.md) | Deploy a saved model, configure serving resources and query raw inputs through REST or SQL `ai_query`. |
| [Online feature lookup](databricks_online_features.md) | Connect lookup keys, Delta features, online publication and the online store; handle missing or stale feature values. |
| [Gradual serving rollout](databricks_serving_rollout.md) | Increase challenger traffic under observation gates, stop or roll back a rollout and control final promotion. |
| [Serving observations](databricks_serving_monitoring.md) | Request logging, latency and error measurements, incomplete logs and performance evaluation when labels are available. |

## Monitoring and automatic retraining

| Guide | What it covers |
| --- | --- |
| [Monitoring overview](databricks_monitoring.md) | How monitoring runs and the order for setting up storage, enrollment, measurements and investigation. |
| [Monitoring store setup and upgrades](databricks_monitoring_setup.md) | Choose a shared destination, create the monitoring store and upgrade its schema. |
| [Enrollment and references](databricks_monitoring_enrollment.md) | Enroll model identities, prepare reference data and run monitoring for those identities. |
| [Delayed labels](databricks_monitoring_labels.md) | Match later outcomes to saved predictions using record keys and availability times. |
| [Drift and feature data quality](databricks_drift.md) | Compare observed inputs with a reference population and interpret drift and quality measurements. |
| [Model performance loss](databricks_performance.md) | Evaluate labeled predictions against the selected metric, baseline and allowed performance limits. |
| [Guarded automatic retraining](databricks_retraining.md) | Turn monitoring decisions into retraining requests while retaining data, policy and cooldown checks. |
| [Shared monitoring dashboard](databricks_monitoring_dashboard.md) | Configure the dashboard, filter model/version/context identities and investigate its tables and charts. |

## MLflow tracking, packages and registry

| Guide | What it covers |
| --- | --- |
| [Track a training run](mlflow_tracking.md) | Log measured metrics, parameters, configuration, dataset Inputs and preprocessing diagnostics to an experiment. |
| [Package and reload a model](mlflow_models.md) | Save fitted pipelines or portable bundles as MLflow models and verify reloaded predictions. |
| [Register and select versions](mlflow_registry.md) | Register models, compare versions on the same holdout, manage champion/challenger aliases and pin a version for inference. |

## Spark execution and portable inference

| Guide | What it covers |
| --- | --- |
| [How inference works](inference_flow.md) | Follow raw inputs through saved preprocessing and model prediction, and choose the appropriate inference route. |
| [Spark execution paths](spark.md) | Native Spark preprocessing, portable learned state and Python model inference on workers, with explicit support boundaries. |
| [Measure Spark inference capacity](spark_benchmark.md) | Run the inference benchmark, read its measurements and check a local PySpark runtime. |
| [Portable inference bundles](inference_bundles.md) | Build, persist and validate a portable package containing supported preprocessing state and a fitted estimator. |

## Operations and troubleshooting

| Guide | What it covers |
| --- | --- |
| [Upgrade a generated project](databricks_project_upgrade.md) | Align project configuration and integration code with a new Core version and validate the migration. |
| [Remove a deployed project](databricks_teardown.md) | Review project-owned resources and follow the cleanup procedure while protecting shared resources. |
| [Databricks troubleshooting](databricks_troubleshooting.md) | Find a validation or runtime message and navigate to the guide that explains its cause and correction. |
| [Databricks glossary](databricks_glossary.md) | Definitions of data, model, scoring, feature, runtime and operational terms, with the relevant configuration or action. |

## Related Core guides

These pages explain the underlying library when you need more detail than the
integration workflow provides.

| Guide | What it covers |
| --- | --- |
| [Pipeline quickstart](pipeline_quickstart.md) | Train, evaluate, predict and save/reload a small model without cloud services. |
| [Step-by-step without configuration](step_by_step_no_config.md) | Use Calculator/Applier pairs directly to see what training learns and inference reuses. |
| [Preprocessing placement and leakage](preprocessing_placement.md) | Decide where transformations belong relative to splitting and understand row alignment during prediction. |
| [SHAP explainability](shap_explainability.md) | Read the library's explanation outputs and create your own feature and prediction plots. |
| [Serialization](serialization.md) | Understand saved pipeline contents, loading, fingerprints and persistence constraints. |
| [Drift monitoring in Core](drift_monitoring.md) | Use the library's drift measurements directly, outside the Databricks monitoring workflow. |
