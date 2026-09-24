# SM-29 Automatic Champion Implementation Plan

> **For agentic workers:** Implement inline with test-driven development and
> check each step before moving to the next.

**Goal:** Let a small-data Databricks Bundle retrain, select champion by a
chosen heldout metric and thresholds, and score with that concrete champion.

**Architecture:** Reuse Skyulf Core's pinned comparison and guarded alias
receipts. Add first-champion validation, a documented exclusive-job admission
option, and Bundle orchestration that calls its existing score job. Resolve
champion once per score run so existing incremental/rebuild writers stay pinned.

**Tech Stack:** Python, pandas/Polars, MLflow UC, Delta, Databricks Bundle.

**Spec:** `initiatives/spark_and_mlflow/31-sm29-auto-champion-design.md`.

## Global constraints

- Preserve `pinned_version` behavior and the existing two job resources.
- No additional UC table; alias writes belong to one serialized job identity.
- First champion requires an absolute heldout quality threshold.
- A failed score never changes the active full-rebuild view.
- Keep SM-29 open until an isolated live test verifies the full lifecycle.

## Tasks

1. Add failing MLflow integration tests for first-champion threshold,
   concurrent or stale alias state, and verified receipt. Implement the
   guarded first-champion operation and exclusive-writer admission contract.
2. Add failing generated-workflow tests for auto train selection and score
   alias resolution. Implement the two paths using the pinned training
   snapshot, existing stage/promote APIs, and a concrete score-run version.
3. Add failing template tests for initialization fields, editable metric and
   thresholds, and dependent `run_job_task` calling the existing score job.
   Implement the Bundle template and document its first-run/failed-run flow.
4. Generate both modes through the CLI and run strict validation. Run focused
   tests, Ruff, ty and strict docs build; record the results and update the
   open queue. Live Databricks resources require a separate concrete test run.
