# Spark and MLflow initiative

Updated: 2026-09-22. **SM-00–SM-16 are complete for their documented scopes;
SM-26 is next.** Target: [0.9.0 — Unreleased](../../changelog/0.9.x.md).
The first new deliverable is a pandas/Polars Databricks Bundle for bounded
training and monthly local prediction. Spark can handle UC table I/O, while
Spark FE/model execution becomes a later Bundle option. Read the current
[queue](OPEN_QUEUE.md) for the authoritative status and order.

## Reading order

1. [OPEN_QUEUE.md](OPEN_QUEUE.md): current task status and order.
2. [HANDOFF.md](HANDOFF.md): verified baseline and execution boundaries.
3. [04-databricks-integration-plan.md](04-databricks-integration-plan.md): local-first adapters.
4. [05-sm20-bundle-plan.md](05-sm20-bundle-plan.md): separate Bundle plan and two-month UC table rehearsal.
5. [01-core-spark-plan.md](01-core-spark-plan.md),
   [02-inference-plan.md](02-inference-plan.md) and
   [03-mlflow-batch-delivery-plan.md](03-mlflow-batch-delivery-plan.md): earlier task detail.
6. [ARCHITECTURE.md](ARCHITECTURE.md) and [VALIDATION.md](VALIDATION.md): contracts and gates.

Historical research:

- [Readiness review](reports/2026-09-21-spark-databricks-readiness.md)
- [Source, node and test inventory](reports/2026-09-21-spark-core-inventory.md)

These reports describe their inspected commits, not the current branch.
The original planning HEAD was `2397c11648c27645a4ee332412a503c394c1c2f2`;
branch `090` and local master later advanced past the 0.8.24 merge point
`97536eae20e5422220f9824bc591ceb05576ee50`. That restored the current
Polars/ContextVar engine behavior, so the older pandas-default observation is
historical. SM-00 recorded a 231-test baseline and two Spark smoke tests.
The user guide is [Spark](../../docs/user_guide/spark.md). This initiative
does not change the global default engine.

## Completed foundation and next deliverable

The initial SimpleImputer/StandardScaler Spark paths, compatible Python-worker
inference, MLflow packaging/registry and a selected serverless Delta batch path
have documented evidence through SM-16. This does not cover every node or model.
The next deliverable strengthens fitted pandas/Polars packaging and uses it in
a generated local-engine Bundle. Its monthly output is tested across two
consecutive months and source versions with one pinned model version. The
second run scores only the new month. See the separate SM-20 plan.

## Working rules

- The next task is **SM-26**. Complete each queue dependency before its consumer.
- For each task, show a failing test, implement the narrow change and record
  the passing gate. New tests have a docstring and a real assertion.
- Recheck source paths before implementation; planned APIs are not current APIs.
- A written file, passing local tests and a successful Databricks run are
  separate claims. A DONE task records commands, results, runtime and commit
  or working-tree state.
- Commit and push are separate actions. A requested commit needs fresh checks
  and DCO sign-off.
- Core/SDK and the local UC output precede SM-20a. Backend/Canvas remains
  parked SM-18; endpoints and streaming are not first-Bundle dependencies.

## Related initiatives

Earlier MLflow proposals are historical input. This initiative uses explicit
run scopes instead of
global fit callbacks or mandatory backend-job tracking. The first Bundle has
no Canvas work. Joblib/ONNX, Ray and deep-learning initiatives are not its
prerequisites. Check shared artifact and pipeline files for concurrent changes
before each implementation task.
