# SM-15I automatic incremental local scoring: validation record

Status: the two-stage Databricks rehearsal passed on 2026-09-23 for the
documented small-data, insert-only pandas model path. The first generated
Bundle remains SM-20a.

## Contract demonstrated

A scheduled scorer receives stable source, target, model, row-key and budget
configuration. It takes no period or source-version parameter. The first job
scores the bounded source snapshot. The next job reads Delta Change Data Feed
inserts after the source version recorded with the previous target append.
The source and target in this proof have **no date column**. Spark performs
Delta I/O, pandas runs the fitted FE and model, and a shared admission protects
the target. Predictions and the source watermark share one Delta commit.

## Local evidence

- The real-Delta local publication suite passed 17 tests. It covers pandas,
  Polars, a source with no date column, an older event timestamp on a new row,
  update rejection, duplicate-key rejection, missing CDF, target-reset
  rejection, failed-write retry, lost-ack replay, shared admission, and an
  injected expired-CDF read failure.
- The SDK/reader suites passed 33 tests with one optional skip.
- Ruff, ty, `git diff --check` and `mkdocs build --strict` passed.
- A fresh wheel was built from the modified source. SHA-256:
  `97918179a0708dde3ce710083829f75a4e83a910df117a0068b20fa52e3265d6`.
  Its packaged `local_incremental.py` bytes matched the workspace source.

## Isolated Databricks resources

- Authenticated CLI profile: `skyulf`.
- Existing test schema: `workspace.skyulf_sm24a_20260923`.
- New test-owned tables: `skyulf_sm15i_source_r1`,
  `skyulf_sm15i_predictions_r1`, `skyulf_sm15i_admission_r1`.
- Uploaded notebook and wheel folder:
  `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_sm24a_20260923/sm15i_r1`.
- Existing pinned UC model:
  `workspace.skyulf_sm24a_20260923.skyulf_sm24a_metrics_r1` version 1,
  digest `e4281f2feb9349b2dd745bcdd894128c47d434f0bd8779eb42d474adb8aed662`.
- Both one-time serverless jobs used the `STANDARD` performance target,
  a 900-second timeout and zero retries. Their environment installed the
  uploaded 0.9.0 wheel and `mlflow==3.16.1`.

| Stage | Parent/task run | Source high version | Target commit | Observation |
| --- | --- | ---: | ---: | --- |
| Initial snapshot | [483287295672086](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/778480091488220/run/483287295672086) / `709869826305307` | 1 | 1 | SUCCESS: 80 existing rows predicted; exact direct-local values |
| Source append | [742044640919376](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/900021231417258/run/742044640919376) / `16280116334355` | 2 | 2 | SUCCESS: 80 new rows predicted; 160 total; first 80 unchanged; no-op retry kept version 2 |

The source fixture came from the earlier 160-row SM-24a test table by stable
key order, with only `entity_id`, `x` and `z` carried into the new source.
Its two halves were staged solely to test arrival between jobs; neither
scoring invocation used a date or source-version argument. Both jobs compared
every persisted prediction with the pinned local model's direct output by key.
The second job also compared the first 80 complete target rows before and
after append, then called the scorer again and observed zero new rows with no
new target commit.

This evidence covers one pinned pandas model on serverless compute and
small bounded insert-only batches. It does not claim support for source
updates/deletes, expired CDF ranges, large driver-side data or a generated
Bundle. Those boundaries remain explicit in the code and queue.
