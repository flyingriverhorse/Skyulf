# Optional feature production and simpler model configuration

> Execution: implement independent domains with focused agents, then review their
> integration. This plan records the user's approved four-part direction.

**Goal:** Support Spark feature groups and checked joins, one authoritative YAML
configuration, broader pandas-model Spark scoring, and point-in-time feature data.

**Architecture:** Spark prepares and joins Delta data; bounded pandas training
fits the existing pipeline; Spark workers replay fitted pandas pipelines. Ready
input tables remain the default. Feature groups describe data domains and can be
shared by any number of models. No distributed fitting is implied.

**Tech stack:** Existing Databricks Bundle, Python, PySpark, Delta, MLflow, YAML.

## Constraints and interfaces

- Keep workflow.json and Python modeling projects readable for old deployments.
  New YAML settings must reject conflicting duplicate ownership, never silently
  override an existing Python model declaration or JSON setting.
- Custom transformations remain Python. Saved model/recipe replay remains
  self-contained; training-fold preprocessing must not fit on merged full data.
- Feature groups are opt-in. Persist each group separately; independent tasks can
  run concurrently and be repaired individually. Merge depends on all groups.
- Group and merged keys must be non-null and unique. Use declared time columns,
  reject overlapping feature names and unexpected join row-count changes.
- Point-in-time lookup must select only a feature row at or before the observation
  timestamp. Reject tied entity/time keys and future/ambiguous records. Native UC
  Feature Engineering client integration and online-store deployment must have
  separate, explicit acceptance; do not equate a local as-of join with all SM-21a.
- Admit only reviewed fitted estimators and deterministic row-local composition;
  custom callbacks, batch statistics and arbitrary Python composition stay denied.
- Cloud execution remains user-deferred. Keep local evidence separate from native
  acceptance. No frontend/backend changes are required for this Bundle-only flow.

## 1. Feature data and orchestration (root)

Files: new `skyulf/integrations/databricks/features/` config, joins, runtime and
graph modules; new `jobs/features/` adapter; generated feature config, notebooks,
graph refresh tool and documentation in `templates/databricks/`.

- [x] Write failing config/join/graph tests, including duplicate/null keys,
  time mismatch, missing group matches, name collisions and stale graph settings.
- [x] Parse bounded feature YAML; execute project-contained Spark transform
  functions; write checked Delta tables and return row counts and versions.
- [x] Generate independent group tasks followed by a merge task, with explicit
  selection/reuse rules and no feature job for a ready-table project.
- [x] Add time-aware joins and focused past/future/tie/date tests before enabling
  them in the feature configuration.

## 2. Single-source YAML (configuration agent)

Files: projects config loader and project/model-set adapters; job config reader;
generated training/inference configs, modeling adapters and focused tests.

- [x] Reproduce absent YAML support with a generated-project fixture.
- [x] Resolve training.yml/inference.yml through one loader into existing runtime
  contracts. Shared defaults plus per-model overrides; emit selected layout only.
- [x] Preserve legacy projects and reject duplicate keys/dual definitions.
- [x] Exercise single, competition, model-set, project checks and saved replay.

## 3. Spark partition support (inference agent)

Files: inference partition certification and row-local model-set composition;
focused unit/integration tests and native Spark parity cases.

- [x] Add failing fitted tree-model and declarative composition parity tests.
- [x] Extend exact-type admission only for verified built-in tree estimators.
- [x] Add explicit deterministic composition operations with schema/state checks;
  keep arbitrary custom source untrusted for distributed execution.
- [x] Verify whole-frame vs reordered/split batches, nulls, empty batches and
  saved-model replay. Record every admitted model/preprocessing operation.

## 4. Acceptance and operator documentation (root)

- [x] Review integration, dependency manifests, schema generation and old project
  behavior. Run the deduplicated affected test set once at final state.
- [x] Run Ruff, CI-scope Ty, CCN <=10, template generation/validation checks.
- [x] Document exact edit locations, graph refresh and selective job repair,
  pandas training versus worker inference, limits and unverified cloud behavior.
- [x] Update queue with precise delivered/partial status. SM-21b stays optional
  and unimplemented until online serving lookup is requested.

## Delivered boundary

Local delivery and verification are recorded in [Delivery181](181-feature-lifecycle-delivery.md).
YAML is an explicit migration, preserving legacy initialization. SM-21a native
training/logging/scoring lifecycle integration and Databricks acceptance remain
open; callable SDK adapters and offline as-of joins are delivered separately.
SM-21b stays unimplemented because online lookup has not been requested.
