# SM-15I: automatic incremental local scoring before the first Bundle

Status: complete for the documented bounded insert-only scope on 2026-09-23.
See the [local and live validation report](13-sm15i-live-validation-report.md).
This task corrects the first Bundle contract:
operators cannot enter a period or source version for every scheduled run.
SM-15L remains the verified explicit-period/backfill writer; its two live jobs
used scripted January/February boundaries and source versions. It did not
automatically discover newly appended source rows.

## Required behavior

A scheduled job receives stable configuration (UC source/target, globally
unique row key, pinned model selection, local engine and row/byte budgets), not
a manually selected date interval. The target must start empty. A timestamp
column is optional and is never used to select which rows to score.

1. On first run, capture the current source Delta table ID and version. Score
   the existing bounded snapshot pinned to that version and publish its
   predictions. If no rows exist,
   leave the target unchanged and retry the bootstrap on the next run.
2. On later runs, derive the last *committed* source version from the target's
   durable publication receipt. Capture the current source version as the upper
   bound. Read only source inserts in the inclusive change-version range
   `last_committed + 1 .. current`; do not rescore earlier rows.
3. Apply the saved pandas or Polars pipeline to only those bounded new rows.
   If an event timestamp exists, retain it; neither it nor the schedule time
   determines row membership. Late arrivals and undated rows are source inserts
   and must be scored.
4. Publish only those final keyed predictions using an append-safe Delta path
   under the existing shared admission. The current `replaceWhere` period
   writer cannot receive only new rows in an overlapping period: it would
   remove earlier predictions in that period.
5. Record source table ID, committed source-version range, pinned model
   name/version/digest, code version, run ID and counts in the *same target
   Delta commit* as the predictions. Read the resulting target version from
   Delta history after commit. Derive the next watermark from the committed
   receipt, avoiding a separate non-atomic checkpoint.
6. If no new inserts exist, return a no-op without a target write. If a job
   fails before commit, the watermark stays put. If the commit succeeds but
   the caller loses its acknowledgement, the next run recognizes the receipt
   and does not duplicate predictions.

The first increment is append-only. Reject source updates and deletes
explicitly; their desired prediction policy is a separate decision. Reject
duplicate source keys within an increment and keys already present in the
target. Require an explicit globally unique prediction key (for example an
`event_id`, or a verified composite key), since an entity can appear in
multiple periods. Never silently skip a duplicate or reinterpret an update as
an insert. The initial snapshot and each change range obey the same local
row/byte ceilings; oversized ranges fail without advancing the watermark.

## Source-change requirement

Use Delta Change Data Feed (CDF) for bounded batch reads between source
versions. Provisioning must enable a compatible feed before the changes to be
consumed, or prove that the source already supports it. Do not assume future
automatic CDF rollout is available in the selected workspace/runtime.
Fail if the required change version has expired from retention; do not silently
fall back to a full scan or a date filter. A repair/bootstrap workflow must be
explicit and independently verified.

Databricks reference:
[Change Data Feed](https://docs.databricks.com/aws/en/tables/features/change-data-feed)
and [`table_changes`](https://docs.databricks.com/aws/en/sql/language-manual/functions/table_changes).
CDF change ranges are inclusive, so the next read begins at
`last_committed + 1`.

## Acceptance

- Local real-Delta tests: initial existing rows; a second source append whose
  `event_time` overlaps the first run; only new keys scored/published; old
  predictions unchanged; no-op; exact retry; lost acknowledgement; competing
  writers; duplicate key; update/delete; expired feed; and a failed write
  leaving the source watermark unchanged.
- Isolated Databricks two-job proof: configure source/model/target once, run
  without manual period/version parameters, append new source rows, run again,
  and compare exact predictions, target rows and committed version receipts.
  Include at least one late-arriving row in an earlier calendar month.
- The generated SM-20a Bundle calls this tested service on a schedule. An
  explicit period replacement remains available for controlled backfills;
  SM-27's `full_rebuild` remains a later, separate operation.
