# SM-22a local candidate/champion validation

Status: LOCAL VERIFIED, 2026-09-23. Unity Catalog comparison remains to be
validated before closing the platform gate. No Bundle or alias mutation was run.

## Delivered

- `compare_registered_local_models` consumes concrete `ResolvedModel`
  versions and their checked local pipeline digests. It scores both versions on
  the same bounded pandas/Polars labeled frame and reports dataset ID, row count,
  model and code identities, metrics, improvement, quality threshold and decision reason.
- RMSE/MAE minimize; R2, accuracy and F1 maximize. Ties cannot qualify. A
  missing champion yields a candidate report without implicit initialization.
  Task and class-label contracts must agree; non-finite metrics fail closed.
- Held-out metric calculation moved from the Databricks adapter to a shared
  inference module. The old Databricks import remains available for callers.

## Evidence

The isolated MLflow 3.16.1 environment passed 25 focused tests with one
optional Spark skip. The tests cover pandas regression, Polars classification,
minimum-improvement and absolute quality gates, malformed references, access denial and
a real local SQLite MLflow registry with two registered local pipeline
versions. The comparison left the champion alias on version 1. Ruff, ty and
strict MkDocs passed for the changed scope.

The caller supplies `dataset_id` after pinning the source snapshot and
evaluation split. The comparison records that identifier but does not verify
the external table snapshot or write the report to a run. It produces a
read-only report, not a promotion authorization.

## Next

SM-22b adds explicit alias promotion/rollback with shared admission, restricted
writer permissions, expected-version checks and durable receipts. The
[MLflow client](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)
and [Unity Catalog alias API](https://docs.databricks.com/api/uc-registered-models/v1/model-alias)
do not expose an expected-prior-version parameter for alias writes; this is
why a read-then-set check alone cannot guarantee race safety. SM-28a then
adds label-aware challenger training before SM-20a packages these services.
