# Serving and feature lookup delivery contract

Date: 2026-09-24. Status: explicitly requested backlog, not implemented.
Prerequisite: SM-43a local Bundle acceptance. Existing task IDs are retained.
These capabilities are delivered in sequence; users select only the features
their generated project needs. The batch-only project keeps its two jobs.

| Order / task | Deliverable | Completion evidence |
| --- | --- | --- |
| 1 / SM-19a | Optional HTTP endpoint for the existing trusted FE+model artifact; explicit version deployment, request/output schema, readiness and endpoint configuration | Real endpoint predictions match local predictions for supported pandas/Polars artifacts; authentication, invalid/missing inputs, warm/cold requests and bounded load are tested |
| 2 / SM-19b | Optional ai_query use of a compatible endpoint with named feature inputs, explicit output contract and caller privileges | SQL predictions match HTTP/local outputs; null/type/order handling, denied access and endpoint failures are tested; no implicit Spark FE conversion |
| 3 / SM-19d | Configurable A/B or canary serving with pinned model versions, explicit traffic allocation, rollout readiness and rollback | Two-version routing, traffic totals, failed rollout recovery and restoring the prior configuration are verified; changing an alias alone is not claimed to update a serving endpoint |
| 4 / SM-21a | Optional UC feature lookup with explicit entity/time keys, selected columns and point-in-time training joins | No future-feature leakage, duplicate/missing key behavior and prediction input assembly match the saved model contract; external feature preparation stays separate from fitted artifact FE |
| 5 / SM-21b | Optional online feature publication/lookup with explicit store resources, freshness policy and serving integration | Offline/online feature parity, stale/missing values, update propagation, access controls and endpoint prediction parity are verified |

Implementation locations: add optional serving/feature adapters under
`skyulf-core/skyulf/integrations/databricks/`, reuse existing local MLflow model
packaging and prediction contracts, and extend the Bundle schema/resources
only behind feature choices. Add corresponding integration tests and English
examples under the generated project and user guide.

Each task begins with platform/API verification and a concrete resource plan.
Validate generated Bundle configuration strictly before a scoped live test.
Endpoints and online stores have persistent resources and costs distinct from
the prior serverless batch rehearsal; retain exact resource/run inventories.
No endpoint/store is provisioned by this planning change.

Monitoring uses existing SM-23a/b contracts. Continuous streaming remains
parked under SM-19c; it is not required to deliver the five tasks above.
