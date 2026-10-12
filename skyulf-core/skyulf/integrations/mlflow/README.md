# MLflow integration

Find implementation modules by responsibility:

| Package | Responsibility |
| --- | --- |
| `models/` | Bundle, local pipeline and model-set pyfunc packaging |
| `spark/` | Distributed prediction adapter, worker source snapshots and output contracts |
| `lifecycle/` | Validation, promotion/rejection and single/model-set challengers |
| `registration/` | Model resolution, registry errors and registered artifact loading |
| `runs/` | Tracking configuration, run context, metrics and artifacts |
| `shared/` | Lazy clients, dtype/signature metadata and nullable frame transport |

Use canonical paths in new code, for example
`skyulf.integrations.mlflow.models.pipeline_model` and
`skyulf.integrations.mlflow.registration.registry`.
The root still exports `TrackingConfig`, `TrackingRun` and `track_run`.

Importing the root or shared client helpers does not import optional MLflow.
Explicit pyfunc adapter imports require the MLflow extra. Keep subpackage
initializers lightweight so optional dependencies remain optional.

Previous flat module paths and the `_compat` package have been removed. Update
imports and string-based patch targets to their owning modules:

| Previous import | Current import |
| --- | --- |
| `skyulf.integrations.mlflow.model` | `skyulf.integrations.mlflow.models.model` |
| `skyulf.integrations.mlflow.model_set` | `skyulf.integrations.mlflow.models.model_set` |
| `skyulf.integrations.mlflow.spark_model` | `skyulf.integrations.mlflow.spark.spark_model` |
| `skyulf.integrations.mlflow.registry` | `skyulf.integrations.mlflow.registration.registry` |
| `skyulf.integrations.mlflow.tracking` | `skyulf.integrations.mlflow.runs.tracking` |
| `skyulf.integrations.mlflow.promotion` | `skyulf.integrations.mlflow.lifecycle.promotion` |

This also breaks old pickles that name those modules. MLflow can deserialize its
outer PythonModel before `load_context` checks the inner artifact's runtime, so
`ModuleNotFoundError` can occur before any version-mismatch error. Keep such
models in their original environment until a replacement has been packaged and
validated with current code. Changing requirements alone is not a migration.
Current model classes serialize their canonical module paths directly.

Shared dtype normalization belongs in `shared/_model_metadata.py`. Both local
and Spark signatures use the same primitive aliases. Existing public/internal
entrypoints delegate to this owner to preserve their callers.

`spark/_spark_environment.py` owns package source-root discovery for both worker
wheels and source certificates. It identifies the entire `skyulf` package;
moving an adapter cannot silently narrow the packaged source to its subfolder.

Registry and tracking clients retain their distinct URI and error contracts.
Artifact-specific requirements and validation stay with their model adapters.
