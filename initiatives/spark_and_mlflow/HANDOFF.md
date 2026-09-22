# Session handoff - 2026-09-21

Work is parked at the user's request after completing SM-10. Resume with SM-11 when requested.
Target release: 0.9.0. Branch: `090`.

## Starting point

- SM-00 through SM-10 are complete; SM-11 is the next READY task.
- Last implementation commit: `df63d231` (worker-local Python FE and model inference).
- Last guide commit: `add50eb7` (SM-10 availability docs and updated worker diagram).
- Read [OPEN_QUEUE.md](OPEN_QUEUE.md), [ARCHITECTURE.md](ARCHITECTURE.md) and
  the SM-11 section of [02-inference-plan.md](02-inference-plan.md) before coding.
- User-facing guide: [How inference works](../../docs/user_guide/inference_flow.md).

## Confirmed terminology and intended workflow

`local` describes single-machine execution, not the user's personal computer.
The runtime environment and the execution engine are independent choices.
Local pandas/Polars FE and sklearn training may run inside Databricks; a later
Spark inference job may also run inside Databricks.

The user's intended workflow is:

```text
Databricks training job: pandas/Polars FE + sklearn model
    -> Save fitted FE, model and input contract
    -> Databricks scoring job: Spark FE + Python model on workers
    -> Write predictions to a Delta table, for example monthly
```

This is the intended deployment scenario, not a claim that the complete
Databricks integration has already been validated. Scheduling, Delta delivery,
MLflow/Unity Catalog, endpoints and templates retain their later queue stages.
Keep new documentation and diagram labels in English.

## Current execution boundaries

- `predict_local` consumes the new standalone bundle; the frontend does not
  call it yet. Frontend inference uses `POST /deployment/predict` and
  `DeploymentService`, which loads the existing artifact, applies FE, aligns
  model columns and predicts. The backend artifact bridge belongs to SM-18.
- `predict_spark(mode="native_features")` supports raw regression bundles
  with supported native FE. Spark applies the saved rules; Python workers run
  the same fitted model on batches. There is no full-data driver collection.
- Portable FE currently supports SimpleImputer mean/constant, StandardScaler
  and an empty chain. Unsupported steps fail explicitly.
- Spark FE fit -> export -> restore -> Spark apply is available. This does
  not imply that end-to-end distributed model training is implemented.
- `mode="python_pipeline"` now runs compatible fitted Python FE and model
  execution together inside workers, without refitting. It rejects unsupported
  context-dependent, row-changing and non-portable FE operations.

## Verification and workspace

Before parking, the English guide's example was checked with both pandas and
Polars training: local and Spark predictions matched `[400.0, 200.0]`. SM-10
also passed its focused 14-test Spark lane and the native inference file passed
52 tests.
The strict documentation build, four rendered diagrams and commit hooks passed.
Broader implementation evidence is recorded under SM-09 in OPEN_QUEUE.md;
those suites were not rerun for this handoff-only update.

Three pre-existing untracked directories remain outside this change:
`.tmp-review-model/`, `.tmp-spark-review-full/`, `.tmp-spark-review-pytest/`.
Preserve them; do not stage or delete them as part of the next task.
