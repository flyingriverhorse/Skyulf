# Databricks runtime

Find implementation modules by responsibility:

| Package | Responsibility |
| --- | --- |
| `jobs/training/` | Training nodes and branch notebook orchestration |
| `jobs/monitoring/` | Monitoring notebook tasks and Spark job orchestration |
| `jobs/lifecycle/` | Lifecycle and retraining task adapters |
| `jobs/shared/` | Notebook runtime, task output and error diagnostics |
| `training/fitting/` | Candidate fitting, ensembles and pre-split preparation |
| `training/competition/` | Candidate competition, evaluation and winner selection |
| `training/tuning/` | Search, cross-validation and search result handling |
| `training/thresholds/` | Decision thresholds and their training/CV integration |
| `training/weights/` | Weight configuration and training integration |
| `training/shared/` | Training evidence and parameter reporting |
| `scoring/batch/` | Batch contracts, local prediction and Spark prediction |
| `scoring/incremental/` | History, incremental prediction and recovery |
| `scoring/shared/` | Prediction output contracts and pre-split scoring transforms |
| `observability/monitoring/` | Monitoring configuration, registration, sources and storage |
| `observability/monitoring/spark/` | Distributed drift, quality, metrics and windows |
| `observability/monitoring/performance/` | Performance policy, actions and retraining evidence |
| `observability/charts/` | Evaluation chart data, runs and rendering |
| `observability/reports/` | Training and explanation reports |
| `model_sets/` | Multi-model packaging, quality gates, release and scoring |
| `projects/` | Project recipes, configuration, source capture and checks |
| `data/delta_io/` | Delta access, admission and CDF recovery contracts |
| `data/training/` | Training dates and local/Spark retraining sources |
| `lifecycle/` | Training state, approval, workflow coordination and retraining requests |
| `shared/` | Cross-domain column/table contracts, bounded frames, manifests and finite JSON identities |

New code imports modules from these packages, for example
`skyulf.integrations.databricks.jobs.shared.job_runtime` or
`skyulf.integrations.databricks.training.fitting.candidate`.
The public objects exported by `skyulf.integrations.databricks` remain available.

Previous flat module paths and the `_compat` package have been removed. Update
deployed notebooks and dynamic import or monkeypatch paths to the owning package:

| Previous import | Current import |
| --- | --- |
| `skyulf.integrations.databricks.job_runtime` | `skyulf.integrations.databricks.jobs.shared.job_runtime` |
| `skyulf.integrations.databricks.batch` | `skyulf.integrations.databricks.scoring.batch.batch` |
| `skyulf.integrations.databricks.monitoring_config` | `skyulf.integrations.databricks.observability.monitoring.monitoring_config` |
| `skyulf.integrations.databricks.training_dates` | `skyulf.integrations.databricks.data.training.training_dates` |

Artifacts that serialize a removed module path can raise `ModuleNotFoundError`
when loaded. Load those artifacts in their original environment and create a
replacement using the current code before upgrading their jobs. A fitted
pipeline's exact runtime-version check remains in force; it does not protect all
outer MLflow model pickles from removed import paths. Current model artifacts
serialize the owning modules directly. Importing the root remains lazy.

Put a helper beside its owning domain. Use that domain's `shared/` package when
multiple components in the domain reuse it; use the top-level `shared/` only for
contracts reused across domains. `shared/json_contracts.py` owns the identical
finite JSON digest used by monitoring, performance, training evidence and CDF
recovery. Domain wrappers retain their existing validation and public signatures.
Other digest encodings remain distinct because they define persisted identities.

Small cohesive packages keep their modules directly under the package. For
example, the scoring SDK and publication adapter stay in `scoring/`, branch
coordination stays in `training/`, and the chart task stays in `jobs/`.

Generated project notebooks remain in their existing flat `src/jobs/` directory.
