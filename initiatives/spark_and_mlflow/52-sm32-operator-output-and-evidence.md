# SM-32 follow-up: readable output and automatic comparison proof lookup

Date: 2026-09-25. This is an operator usability follow-up to SM-32; the broader
SM-33 configuration/migration work remains READY.

## User observation and verified initial state

The user approved Polars candidate v5 through the train job. Parent run
`24807365707427` and child score run `379146043586828` succeeded. Promotion
event `74ff48555f304272ae50c59f771295b0` changed champion from v1 to v5.
The score result was a no-op: input/output counts were zero and the prediction
Delta version remained 2. Its manifest described the previous v1 write,
not a new prediction with v1.

The raw `Notebook exited` result was difficult to read and the manual form
required copying a 64-character comparison digest.

## Implementation

- `job_output.py` renders a shared, HTML-escaped report for train, approve,
  reject, rollback and score. Reports separate operation/alias outcomes,
  comparison metrics, prediction counts and next-action parameter tables.
  Technical JSON remains available in a disclosure and in the Jobs API result.
- Notebook entrypoints render the report in the first cell and return the
  machine-readable JSON in a separate final cell. A live R3 experiment proved
  that `notebook.exit` replaces the same-cell result; R4 addresses that behavior.
- The generated form has five fields. `comparison_sha256` is removed from its
  job defaults and base parameters. Approval/rejection requires a concrete
  candidate and explicit expected champion (`none` for bootstrap).
- The adapter reads the full proof from that candidate's active committed
  lifecycle receipt. It delegates to the existing strict Core operation,
  which verifies artifact/receipt consistency, aliases, policy and quality.
  No hash truncation, latest-candidate fallback or extra table is introduced.
- Explicit full digests remain supported for older/API callers. Direct Core
  approval/rejection APIs retain their required digest contract. Rollback still
  uses the exact promotion receipt and expected champion.
- If rendering fails, indented JSON is printed and the already-completed
  lifecycle action is not turned into a failure inviting a mutation retry.

## Verification

- 80 focused behavior tests passed, including pandas/Polars with explicit and
  automatically resolved proof, bootstrap, approval/rejection retries, stale
  champion, changed comparison artifacts, wrong policy, rejection and rollback.
- 16 real CLI generation combinations passed; 46 promotion lifecycle tests passed.
- After the live same-cell discovery, 19 output/notebook tests passed, including
  two separate-cell regressions. Scoped Ruff, ty and strict docs passed.
- R3 parent `605352825761827` succeeded without a digest parameter, returning
  the user's exact existing promotion event; the new five-field form was read
  back through Jobs API. Exported notebook output exposed the same-cell issue.
- R4 parent `719498349185749` succeeded. Task `1009765289788596` returned the
  exact same promotion event without a hash input. Jobs export proved two
  notebook cells: a persisted HTML report (`listResults`) followed by the raw
  JSON exit result. Both the summary and Technical details disclosure exist.
- Child score `959432187722626` succeeded: zero input/output rows, `noop=true`,
  Delta version 2. Its exported HTML explicitly labels the manifest as the
  previous write. Final aliases: champion v5, previous_champion v1,
  previous_challenger v4; no challenger alias remains on the promoted version.

Live follow-up is complete. The existing schema/model/output and two persistent
jobs were reused; no company-target readiness claim is made.

R4 wheel: `skyulf_lifecycle_test/sm32/r4/skyulf_core-0.9.0-py3-none-any.whl`.
SHA-256: `2c0bea77d8d09418f2aa8b3d465cc0eb047da9418615eb8a74745716f9654d06`.
Both jobs retain 900-second timeouts. No new schema, source/prediction table,
registered model version or persistent job was created by this follow-up.

Local rehearsal evidence is under `rehearsals/sm32_live/output-*` (ignored).
The walkthrough is `docs/user_guide/databricks_bundle_walkthrough.md`.

## R5 presentation clarification

The user found the rollback block misleading. It now says **If rollback is
needed**, explicitly states that rollback is optional, and separates **Required
current champion: v5** from **Restore version: v1** for the retained example.
The full receipt (including its comparison proof) is inside a closed parameters
disclosure. API values and rollback validation are unchanged.

Seven report tests, scoped Ruff/ty and strict docs passed. R5 was deployed to
the same two jobs and their wheel references were read back successfully. No
new job run was started for this presentation-only change; historical outputs
remain unchanged. A preview using the actual R4 receipt is saved locally in
`rehearsals/sm32_live/operator-output-preview.html`.

R5 wheel directory: `skyulf_lifecycle_test/sm32/r5`.
SHA-256: `9e40dfbbd5af4791d06c1f04a61d23c3cf8f9646258fd34ee4f305b741bf5882`.

## Requested R5 live rerun and commit validation

The user subsequently requested a live rerun before commit. Parent lifecycle
run `769913416634457`, train task `1092954776398504`, and child score run
`690155923861600` all succeeded on R5. Approval supplied only candidate v5 and
expected prior champion v1, without a comparison digest. It replayed the exact
existing event `74ff48555f304272ae50c59f771295b0`.

The exported, executed notebook was decoded and its actual HTML checked:

- The report says **If rollback is needed** and states rollback is optional.
- Required current champion is v5; restore version is v1.
- Visible text outside disclosures contains neither `comparison_sha256` nor
  `promotion_receipt_json`; the complete proof remains in collapsed details.
- The separate final cell retains the raw JSON result for API clients.
- Child score also persisted its HTML report and returned zero input/output
  rows, `noop=true`, prediction Delta version 2.
- Final aliases remained champion v5, previous_champion v1 and
  previous_challenger v4. No rollback or new training was executed.

Fresh commit verification: 144 affected tests passed, including both local
engines, explicit/saved proof checks, lifecycle history, notebook output,
rollback/retry, and 16 real CLI generation combinations. Strict docs passed.
Rehearsal evidence filenames use the `output-replay-r5-*` prefix.
