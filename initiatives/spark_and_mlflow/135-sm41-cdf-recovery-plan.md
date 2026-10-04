# SM-41: optional CDF expiry recovery

User-approved scope: automatic full rescore only after a recognized CDF history
expiry, disabled by default. A separate recovery task is visible inside the
existing score job. Batch is the execution mechanism; model_change_mode remains
independent. No new lifecycle job, training, alias mutation, monitoring or cost tool.

## Implementation contract

- auto_rebuild_on_cdf_expiry is a boolean defaulting to false in workflow config.
- Recognized structured CDF expiry errors become CdfRecoveryRequired carrying a
  pinned recovery request. Permission, connectivity, disabled CDF, replaced
  tables, schema issues and readable source UPDATE/DELETE retain existing errors.
- Request carries source/target names and IDs, expected target Delta version,
  source watermark interval and concrete model/version/digest; serialized size is bounded.
- Recovery rechecks identities and expected output state under writer admission,
  recomputes the pinned full snapshot, resets temporal carry and atomically
  overwrites compatible output with a recovery receipt. Retries are no-ops only
  for that exact committed recovery request. Normal incremental operation resumes.
- Recovery uses a single Delta overwrite without dropping the output table or
  recreating existing views; live table-ID/grant preservation was subsequently verified in report136.
  Existing model-version generations remain; same-generation Delta rollback
  depends on retention.
- Score emits recovery_required and cdf_recovery_request task values only when
  opt-in is enabled. A condition routes to recover_predictions. scoring_report
  joins using NONE_FAILED and displays the actual scoring/recovery result.
- Both single/competition and model-set paths share the request/error contract.
- Existing defaults remain functional; malformed opt-in values fail validation.
- Default schedules, job identities, retry/timeout controls and output guards remain.

## Tasks and verification ledger

- [x] Shared error/request contract and single-model atomic recovery, meaningful tests.
- [x] Model-set recovery using the same pinned request and publication protocol.
- [x] Notebook request routing, recovery entrypoint, truthful run summaries.
- [x] Optional config/schema, separate tasks, generated notebook and CLI tests.
- [x] Review; affected pytest/collection, Ruff/format/full Ty/CCN/schema checks.
- [x] Queue evidence and precise remaining SM-41 retention/live-test boundaries.

Ruling: work in the established branch 091 checkout, preserving unrelated edits;
the user requested continuation in this shared workspace, not a new branch.
Ruling: this delivery implements the approved automatic-CDF-recovery slice.
Generation cleanup and source-replacement acceptance are not silently enabled.

## Behavior and review

Enabled jobs contain `score`, `recovery_needed`, `recover_predictions` and
`scoring_report`. The report joins normal/recovered success via `NONE_FAILED`;
it does not read excluded-task values. Disabled projects retain one score task.
Task-value summaries exclude large temporal continuation state; the originating
notebook still retains its full JSON result. Pending recovery never claims a write.

Model and source pins are captured before the task split. Recovery rejects a
changed table ID, target commit, model digest or destination. An exact committed
recovery retry is a no-op; later ordinary scoring resumes at the recovered source
watermark. Empty valid snapshots clear predictions, and temporal state restarts
from that full snapshot. Row/byte limits remain in force before publication.

Independent review found and fixed two edge cases: model-set overwrite now pins
`partitionOverwriteMode=static` and `overwriteSchema=false`; an already active
full-rebuild generation may become empty while remaining readable through its
existing view. A fresh empty generation still cannot replace the active view.
Tests cover the recovery, retry and following ordinary no-op workflow.

Only exact recognized Delta history conditions trigger recovery. Spark Connect
may wrap them in a read-file error, so normalization also accepts an anchored
Delta exception cause in its server stack trace; generic missing files, permission
failures and free-form message matches do not qualify. Reference:
[Databricks CDF file error](https://kb.databricks.com/en_US/delta/job-failing-with-delta_change_data_file_not_found-error),
[Delta error conditions](https://docs.databricks.com/gcp/en/error-messages/error-classes),
[Spark Connect exception conversion](https://github.com/apache/spark/blob/master/python/pyspark/errors/exceptions/connect.py).

Existing projects must regenerate/redeploy the matching graph when enabling the
option; editing only the boolean is rejected by the deployed contract marker.

## Initial delivery acceptance boundary

At this initial delivery checkpoint, no SM-41 Databricks deployment or live job
execution had run. Subsequent live evidence is recorded in
[report136](136-sm41-live-cdf-recovery.md). Local
Spark/Delta tests are present but require dependencies absent from the Windows
environment. CLI generation/schema validation is not cloud runtime acceptance.
The planned follow-up checks at that checkpoint covered expired-CDF classification,
full/empty replacement, unchanged IDs/grants, pinned-model replay, incremental
continuation and successful/failing branch joins. Their later results are in report136.
Source-table replacement and automatic generation cleanup remain unimplemented.
No packaging/compute expansion, monitor, cost report or acceptance runner added.

## Verification (2026-09-30)

- Final consolidated runtime regression: **286 passed, 15 skipped**, 30.82 seconds.
  Scope: shared/single/model-set CDF recovery, notebook routing, model-set batch,
  Delta/source corrections/project/output/handoff, job runtime/output and local
  workflow. The skipped cases require Spark/Delta; existing sklearn, MLflow and
  legacy-selection warnings remain.
- Template/config/schema/generation/deployment suite: **296 passed**, including
  real CLI init and strict loopback Bundle validation for three layouts and both
  compute modes. This creates local generated projects, not live jobs.
- Integration collection passed; 2,533 cases collected before the final added
  notebook regression. All new affected test files execute in the final runtime run.
- Repository Ruff, CI format scope (1,171 files), full CI Ty scope, Lizard CCN 10
  for backend/Core, generated schema freshness and `git diff --check` passed.
- Meaningful RED checks preceded the new runtime/model-set recovery code and the
  empty active-generation fix. Independent review findings were fixed and covered.
- No commit requested for this slice; changes remain in the working tree. Existing
  unrelated queue deletions and the user's temporary review directory are preserved.
