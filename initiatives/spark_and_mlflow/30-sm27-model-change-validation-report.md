# SM-27: Model-change scoring and configurable schedule validation

Date: 2026-09-24. Branch: `090`.

## Delivered locally

The existing two-job Bundle asks how a newly pinned concrete model version
affects scoring. `incremental_append` remains the default and preserves old
predictions while scoring only later source inserts with the new model.
`full_rebuild` creates one physical Delta generation per model version,
bootstraps it from the current bounded source snapshot, and only then points
the stable prediction view at it. Subsequent source inserts are appended to
that generation using its own committed CDF receipt. Old generations remain.
The logical view rejects an existing physical table, a foreign view, an empty
new generation and an incompatible output schema. A known view-name conflict
fails before model loading or generation-table creation. Training a candidate
or changing an MLflow alias alone still does not select a scoring version.
Each generation table records the model name, version and artifact digest.
Reusing its numeric version with a different model or artifact is rejected.

The optional `monthly_paused` train job now reads its Quartz cron and IANA
timezone from editable Bundle variables initialized from the user's answers.
The schedule stays paused; the default is the previous monthly time. The
monthly training window remains a UTC calendar-month window regardless of the
schedule timezone.

## Verification

- The focused suite passed: **46 passed, 18 skipped**. The generated-workflow
  tests cover both mode selections, call order,
  failed scoring, view ownership/schema guards, generation provenance and
  editable schedule fields.
- Databricks CLI generated both a default append project and a custom
  full-rebuild project with `0 30 7 10 * ?` / `Europe/Vilnius`. Both generated
  serverless projects passed `bundle validate --strict -t dev --profile
  skyulf` after the final workflow source was copied into each generated
  project. This was read-only validation, not deployment or a job run.
- Ruff check/format, ty and `mkdocs build --strict` passed.
- A real local Delta v1/v2 generation test was added. The normal local venv
  skips optional Delta; the Spark venv cannot start local Delta on this Windows
  host because Hadoop's `winutils.exe` is absent. Its assertions have therefore
  not passed yet, and no live Databricks full-rebuild job has run.

## Live validation still needed

Use an isolated personal-workspace source, model and output prefix. Score an
initial bounded source with v1, insert a new row, pin v2 and run the selected
full-rebuild mode. Verify the v1 physical generation remains unchanged, the
v2 generation contains all current source rows, the stable view points to v2,
and a later insert appends only to v2. Repeat the test in append mode to show
mixed `model_version` values in one physical target. Check the view's grants
after `ALTER VIEW`; then remove the isolated test resources after recording
evidence. Company targets remain untested.
