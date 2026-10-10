# Direct YAML projects and native feature lookup lifecycle

User-approved continuation after f4cd2d32. Existing projects need no migration;
fresh projects must directly emit authoritative YAML. Native Databricks execution
remains deferred, so SDK/JVM/offline checks must be distinguished from cloud acceptance.

## Architecture and ownership

1. YAML agent: generate training.yml/inference.yml directly for all three layouts,
   keep features.yml optional, remove migration/Python model scaffolding, centralize
   loading, refresh tools/docs and verify real CLI generation locally. Retain
   saved-artifact and public legacy-reader compatibility where still meaningful.
2. Root: opt-in feature_lookup workflow contract; checked native training-set
   preparation and durable lookup evidence; connect single/competition/branch
   training, registration and saved replay without fitting preprocessing twice.
3. UC agent: reuse exact local pyfunc packaging under native Feature Engineering
   logging; retain signatures, artifact digests, worker code, nullable transport
   and certificates. Inspect SDK implementation before adopting its API.
4. Spark agent: map/review scoring boundaries first; implement the agreed native
   route only after contract decisions. Existing keyed Delta receipt and model-set
   publication semantics must remain intact.

## Acceptance

- [x] New CLI-generated projects need no migration and contain no duplicate model
  definitions or generated workflow.json. Single/competition/model-set smoke,
  preview, graph generation and strict local CLI validation remain usable.
- [x] Feature lookup is explicit and absent by default. Validate entity/time keys,
  selected features and temporal types. Preserve bounded training materialization,
  metadata required for splitting and saved immutable lookup evidence.
- [x] Preserve the native TrainingSet feature specification through fit/logging;
  task boundaries must reconstruct it from verified inputs rather than pickle an
  SDK object or silently reread changing feature data.
- [x] Feature-packaged artifacts route through native score_batch only with pinned
  model identity, partition checks, explicit schema and key/cardinality validation.
  Unsupported combinations fail before writes rather than silently ignoring lookup.
- [x] Run test-first focused regressions, affected direct-consumer file unions,
  current-source real Spark where appropriate, full static gates and package check.
- [x] Document file locations, activation, limits and exact remaining native gates.
  SM-21b stays optional; no automatic online-store deployment.
- [ ] Native Databricks acceptance: user-deferred, so SM-21a is not marked DONE.

No user confirmation is needed to implement this already-approved local scope.
No remote resource deletion is implied by resetting generated project structure.
