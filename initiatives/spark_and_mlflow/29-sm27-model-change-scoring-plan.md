# SM-27: Selectable model-change scoring and editable retraining schedule

## Goal

At Bundle initialization choose what happens **after an operator pins a new
registered model version**. Training, challenger staging and champion alias
changes alone must never change scoring. Keep the existing two jobs.

`incremental_append` is the default. It preserves old rows and their
`model_version`. A newly pinned version scores only later source inserts.

`full_rebuild` preserves the previous physical generation. A newly pinned
version scores one bounded full source snapshot into a new generation. After
that score succeeds, it switches the stable prediction view. Later inserts
append to the new generation using its own CDF receipt.

The full-rebuild logical `prediction_table` is a Unity Catalog view. Physical
Delta generations are named `<prediction_table>_v<model_version>` and keep
their own model name, version, digest and source watermark. An existing
generation with different model provenance is rejected. A changed model
version selects a new generation, so the current incremental scorer safely
bootstraps from all source rows instead of trying to overwrite prior keys.
The same version reuses the generation and processes only later source inserts.
The view is created only after a nonempty successful first score; an existing
Skyulf-managed view is altered only after the candidate generation is complete
and its schema matches. Never replace an unrelated table or view. Use `ALTER
VIEW ... AS` for an existing view to retain its grants. The old generation
remains queryable. Switching from an already deployed append-mode physical
`prediction_table` to rebuild mode requires a new logical view name or an
explicit migration; this workflow must fail rather than replace that table.

The default `incremental_append` path remains the existing one-table behavior.
Neither mode adds a third job, admission table, auto-promotion, or Spark-native
feature/model execution. `score` remains the sole serialized writer for each
target. Multiple writers still require shared admission before use.

## Schedule configuration

For the optional `monthly_paused` train schedule, initialization asks for a
Quartz cron expression and timezone (default `0 0 3 3 * ?`, `UTC`). Render
them as editable Bundle variables and reference those variables from the job
schedule. Keep the schedule paused by default. `manual` still emits no
schedule. The cron determines **when** training runs; the current monthly
training action still uses UTC calendar-month fit/holdout boundaries. A custom
nonmonthly cron is outside this action's contract and should be rejected or
documented as a monthly-frequency requirement.

## Acceptance and verification

1. Test both policies through the generated workflow: an appended v2 increment
   leaves v1 rows intact; rebuild v2 uses a new target, fills it before view
   activation and keeps v1 generation; same-version replay remains incremental.
2. Reject invalid modes/versions, missing source, empty generation, incompatible
   output schema, existing physical table at the logical view name and any
   non-Skyulf view. A failed score never changes the active view.
3. Generate default and rebuild projects through Databricks CLI and strictly
   validate their serverless Bundle shape. Test that cron and timezone values
   supplied at initialization appear as editable variables, not literals in
   the resource file.
4. Run focused Core/Bundle tests, Ruff, ty, `mkdocs build --strict`, and
   document any live test not run. Do not mutate a company workspace.
