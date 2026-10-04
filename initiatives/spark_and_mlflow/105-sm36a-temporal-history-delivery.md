# SM-36a: temporal history delivery

Date: 2026-09-28. Implementation included in this delivery commit.
Authoritative queue: OPEN_QUEUE_updated.md. This completes the temporal history
slice, not all remaining SM-36a feature/output rules.

## Implemented

- Explicit `history_mode=carry` for LagFeatures/RollingAggregate. Existing batch
  mode remains default. Bounded per-entity seed is saved in the model artifact;
  prediction never modifies that artifact. Each chained step owns its own tail.
- Training uses the train-transform hook and never prepends its own tail.
  Heldout/fold transformations preserve requested row order and return only
  requested rows. History cannot enlarge later scaler/imputer/model fits.
- Numeric or typed datetime clocks, nonmissing unique entity/time keys,
  forward-only continuation, explicit row/byte budgets, and target exclusions.
  Carry mode is learned state for pre-split leakage admission. Random/group CV
  is rejected for carry; ordinary/nested temporal CV retains required clocks
  until preprocessing finishes, then removes split metadata before modeling.
  CV-disabled tuning accepts an explicit later validation batch.
- `TemporalHistorySession` binds explicit continuation to immutable model
  identity, reuses the same input context on predict/probability passes and
  proposes state only when the wrapped operation succeeds.
- Backend `/deployment/predict`: `continue_history=true` starts continuation;
  `history_state` resumes it. Caller owns durable persistence/serialization.
  This is not an automatic server-side history database or mutable model cache.
- Databricks incremental initial snapshot reconstructs context without the
  artifact seed. Subsequent increments read bounded history from the target's
  latest receipt. Output/history/watermark share one Delta transaction. Receipt
  budget is 64 KiB. Period scoring accepts explicit earlier context and binds
  request identity to it. Changed models cannot silently reuse another tail.
- Canvas selector, inline template recipe, SDK documentation and changelog.

## Verified evidence

- Initial targeted checks: 136 temporal/alignment tests passed.
- Core history contract: 41 tests passed, including both engines, JSON/pickle
  replay, entity isolation, timestamp nanoseconds, failure/retry, chained steps,
  scalar-fit population, five strategies x ordinary/nested Time Series, final
  refit, CV-disabled holdout tuning and explicit gap-row isolation.
- Real backend HTTP tests reload stored models, continue batches, reject late
  rows, replay identical requests and confirm the artifact seed remains fixed.
- Broad final regression suite: **395 passed, 19 skipped**. Skips require the
  optional local Delta runtime; the new publication contract passed on actual
  Databricks Delta. Two additional real MLflow logging/loading tests passed
  (pandas and Polars), including explicit continuation after reload.
  Final focused history/artifact/MLflow suite: **46 passed** (overlaps the broad
  suite; these counts must not be added as distinct tests).
- Integration collection: 1,520 tests collected without import errors.
- Frontend focused tests: 8 passed. Lint, complexity, production build and
  size-check passed; generated assets refreshed. No browser E2E run in this slice.
- Ruff, full CI Ty scope, CCN <= 10 and strict MkDocs passed. Final format gate
  checks 1,070 Python files. Commit verification is recorded below.

## Live Databricks acceptance

Workspace: dbc-45604623-c18b.cloud.databricks.com, existing skyulf test profile.
Run `875016300611787`, task `821490642468114`: **SUCCESS**.
Wheel SHA256: `5688bedcb4c071e7a7a34b56e6c6e3856fe90d375d75dfaf90213367464e66ad`.

- pandas/Polars x linear regression, logistic regression, voting regression and
  soft voting classification: 8 cases passed; chunked predictions/probabilities
  matched uninterrupted prediction. Both voting families used two estimators.
- Both engines: 4 initial + 2 appended Delta predictions; history tail [4,5].
  Injected precommit failure left the previous receipt unchanged; retry passed;
  no-op made no commit; late insert was rejected without advancing the receipt.
- Target commit version 2 on both isolated targets:
  `workspace.skyulf_lifecycle_test.sm36a_history_20260928_d566bc8c_{pandas,polars}_predictions`.
- This uses loaded artifact fixtures to test actual Delta publication; it does
  not claim new MLflow registry/alias tests. Test source tables intentionally
  contain the late insert used to verify rejection.
- First run `176237093459458` failed due to a test-only temporary path reuse;
  it was canceled during automatic retry. Unique artifact paths fixed the
  harness before the successful run; production code was not responsible.

Final wheel additionally includes the CV-disabled routing fix:
`88b50732d70ce60f6c6a4a281cee303c2350562dd99aca3b7d6da03113ed0325`.
Run `946727131136976`, task `1060445614267200`: **42 passed, zero failures,
zero errors, zero skipped** on that final wheel. This includes all five tuning
strategies with ordinary/nested Time Series and CV-disabled tuning.
Earlier run `1033982723441910` could not collect tests because workspace FUSE
does not support pytest's `__pycache__` creation. Copying the unchanged tests
to `/tmp` fixed the harness; no Core change was needed.

Rehearsal payloads/results are local under `rehearsals/history_live_20260928/`.
User requested the local delivery commit; no push. SM-36a still has keyed scoring exclusions, external
dependency/asset delivery, and additional project-owned output rules to address.

## Delivery commit verification

- Fresh history/artifact/MLflow/backend regression selection: **57 passed**.
- Ruff whole-repository lint and CI format scope passed (1,070 files).
- All applicable staged pre-commit hooks passed: whitespace/EOF, Ruff lint and
  format, full CI Ty scope, backend/Core CCN 10, frontend lint and complexity.
- Updated queue retains SM-36a PARTIAL; keyed scoring exclusions are next.
  The active queue replaces the removed older OPEN_QUEUE.md. Rehearsal output,
  model artifacts and unrelated temporary files are excluded from the commit.
