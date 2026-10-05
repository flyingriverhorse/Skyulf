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
`skyulf.integrations.mlflow.models.local_model` and
`skyulf.integrations.mlflow.registration.registry`.
The root still exports `TrackingConfig`, `TrackingRun` and `track_run`.

Importing the root or shared client helpers does not import optional MLflow.
Explicit pyfunc adapter imports require the MLflow extra. Keep subpackage
initializers lightweight so optional dependencies remain optional.

`_compat/` preserves released flat paths, including model classes referenced in
saved artifacts. Aliases resolve lazily to the canonical module itself; class
identity and monkeypatch targets are shared rather than copied into wrappers.
Keep implementation changes in the responsibility packages.

Shared dtype normalization belongs in `shared/_model_metadata.py`. Both local
and Spark signatures use the same primitive aliases. Existing public/internal
entrypoints delegate to this owner to preserve their callers.

`spark/_spark_environment.py` owns package source-root discovery for both worker
wheels and source certificates. It identifies the entire `skyulf` package;
moving an adapter cannot silently narrow the packaged source to its subfolder.

Registry and tracking clients retain their distinct URI and error contracts.
Artifact-specific requirements and validation stay with their model adapters.
