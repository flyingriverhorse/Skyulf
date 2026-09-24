# SM-20a local-engine Bundle validation - 2026-09-24

Status: **PASS** for the first bounded pandas/Polars Databricks Bundle. The
generated project, not only a direct SDK script, was initialized, strictly
validated, deployed and run through its own jobs in the `skyulf` workspace
profile. Target release remains 0.9.0.

## Delivered contract

`skyulf-core/templates/databricks` asks for project name, local engine and existing UC
catalog/schema. It emits its own `databricks.yml`, editable workflow JSON,
one thin Skyulf entry point and five separate serverless jobs: `train`,
`compare`, `stage`, `promote` and `score`. Each job installs the matching
`skyulf-core` wheel and MLflow 3.16.1. The entry point delegates to the
existing Skyulf training, comparison, promotion and incremental scoring
services; it does not copy their feature engineering or model code.

Training pins a Delta snapshot, bounded row/byte budget and temporal holdout.
Scoring pins a concrete registered model version and derives new source rows
from committed Delta receipts without per-run dates or source-version input.
The generated project has no schedule or automatic alias promotion. The first
`@champion` alias must be initialized explicitly before guarded staging and
promotion can operate. Production users must create the declared UC tables,
enable source CDF before new inserts and provision separate prediction/alias
admission rows as described in the generated README.

## Local and template gates

- `databricks bundle init skyulf-core/templates/databricks --config-file ... --output-dir ...`
  rendered the `skyulf_sm20a_r1` project with `engine=polars`.
- `databricks bundle validate --strict -t dev --profile skyulf` passed after
  generation and after each test configuration change.
- Four focused tests passed. They pin engine propagation, no-date incremental
  scoring, read-only comparison and the separate eligible stage gate. Ruff
  lint and format checks passed for the new Python files.
- `mkdocs build --strict` passed for the new user guide page.

## Isolated serverless rehearsal

The test reused `workspace.skyulf_sm24a_20260923` and created only
`skyulf_sm20a_r1_*` tables/model under it, plus an owned dev Bundle workspace
folder. The one-time [data setup job](https://dbc-45604623-c18b.cloud.databricks.com/jobs/91107508891891/runs/764189335699478?o=7474646244882000)
created an initial two-row CDF source, an empty prediction table and shared
score admission. It used real NYC taxi rows distinct from the training set.

The first [Bundle training job](https://dbc-45604623-c18b.cloud.databricks.com/jobs/899218432862214/runs/315074221066656?o=7474646244882000)
fit a Polars local pipeline and registered UC model version 1. The unchanged
[score job](https://dbc-45604623-c18b.cloud.databricks.com/jobs/859114751278280/runs/623601582323720?o=7474646244882000)
predicted the two existing rows. After two new rows were appended, the same
[score job](https://dbc-45604623-c18b.cloud.databricks.com/jobs/859114751278280/runs/863264817851057?o=7474646244882000)
predicted only those two. The verifier found four unique target keys, two
distinct two-row incremental receipts and one pinned model identity. A later
[empty replay](https://dbc-45604623-c18b.cloud.databricks.com/jobs/859114751278280/runs/764460864735699?o=7474646244882000)
returned `noop=true`, `input_count=0`, `output_count=0`, and unchanged target
Delta version 2. The strengthened verifier confirmed no later target commit.

A second [Bundle training job](https://dbc-45604623-c18b.cloud.databricks.com/jobs/899218432862214/runs/314248115042058?o=7474646244882000)
registered Polars candidate version 2 from the same pinned table version 0:
1,724 fit rows, 1,770 holdout rows and six labels unavailable by cutoff. Its
MLflow run `3d374999c4de42e0ade85c68fe554cb5` logged the full metrics and
comparison. Held-out RMSE was 2.088749 versus version 1's 2.252941; the
independent [compare job](https://dbc-45604623-c18b.cloud.databricks.com/jobs/405452823167260/runs/202747732483469?o=7474646244882000)
returned `eligible=true` without moving an alias. A test-only bootstrap job
explicitly placed initial `@champion` on version 1 and provisioned the separate
alias admission row. The guarded [stage job](https://dbc-45604623-c18b.cloud.databricks.com/jobs/34741568106407/runs/1089047629551174?o=7474646244882000)
placed version 2 on `@challenger`; the separate [promote job](https://dbc-45604623-c18b.cloud.databricks.com/jobs/733893149679423/runs/542805401182400?o=7474646244882000)
changed `@champion` from 1 to 2. The final
[alias verifier](https://dbc-45604623-c18b.cloud.databricks.com/jobs/1053643319015412/runs/423058698933542?o=7474646244882000)
confirmed `@champion=2`, `@previous_champion=1`, no `@challenger`, and all four
previous predictions still labeled model version 1.

The first deploy failed locally because the CLI could not read the newly built
wheel under Windows ACLs; granting the CLI user read access resolved it. The
first alias verifier failed in test-only code because MLflow 3.16.1 returned
`RegisteredModel.aliases` as a mapping, rather than iterable alias objects.
That verifier was corrected and rerun successfully; no promotion rollback was
needed. Both failures remain recorded here rather than counted as passes.

## Boundaries and next work

The live run used Polars; pandas is covered by existing local/integration and
SM-15I live evidence, but this generated Bundle was not separately deployed
with pandas. The Bundle rehearsal did not force a remote Delta output failure;
the underlying SM-15I service has local failure/retry tests. The generated
project does not bootstrap the first champion, schedule retraining, perform
full-history rescores, distribute FE/model work with Spark, or create serving
endpoints. These remain explicit later tasks. Next is optional SM-28b monthly
retraining scheduling, followed by the other open-queue items in their listed
order.
