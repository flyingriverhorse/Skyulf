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
| `observability/monitoring/local/` | Bounded local monitoring and metrics |
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
`skyulf.integrations.databricks.training.fitting.local_retraining`.
The public objects exported by `skyulf.integrations.databricks` remain available.

`_compat/` preserves previous flat module paths used by deployed notebooks and
serialized model artifacts. The root package adds that directory to `__path__`;
each compatibility module aliases the canonical module in `sys.modules`. Both
paths therefore share the same classes and module globals, including monkeypatches.
Aliases load on demand; importing the root does not load every runtime module.
Keep implementation changes in the responsibility packages, not in `_compat/`.

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
