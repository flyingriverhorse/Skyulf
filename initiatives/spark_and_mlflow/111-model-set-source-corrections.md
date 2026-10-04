# Model-set source corrections

Approved behavior: preserve insert appends, optionally reuse the existing atomic
full-rebuild path when source CDF contains UPDATE/DELETE even if the model set
has not changed. This changes scoring, not training.

## Scope and plan

- [x] Add a typed source-change error to the shared insert selector. Other errors
  (missing CDF, permissions, limits, unknown source identity) must still fail.
- [x] Add model-set `source_change_policy`: `reject` (legacy default) or
  `rebuild_on_change`. Recovery reads the same pinned upper source snapshot,
  resets prior watermark/history for computation and overwrites once only after
  complete model/rule success. Empty snapshots must clear output and history.
- [x] Record the selected policy, write mode and source-rebuild flag in receipts.
  Preserve model-change behavior, no-ops, target admission and all output modes.
- [x] Add a multi-target Bundle question and model_set.py setting. Keep the
  standalone single-model insert scorer's current contract unchanged. Document
  that corrections rescore all rows with the selected set and existing limits.
- [x] Verify source-selection failures, updates/deletes, mixed changes, repeats,
  temporal reset/continuation and failed-publication preservation. Add real Delta
  coverage and report live execution separately from local tests.

## Acceptance cases

1. Inserts only -> append; same source -> no-op, without duplicate output.
2. UPDATE or DELETE -> reject by default; rebuild when explicitly enabled.
3. Same model + corrected source -> all snapshot rows recomputed; deleted rows gone.
4. Last source row deleted -> empty output/history committed, then no-op.
5. Rebuild computation fails or exceeds bounds -> prior target/receipt unchanged.
6. CDF access failures do not trigger a permissive fallback.
7. Temporal result/history after correction matches an uninterrupted fresh snapshot;
   a later append continues from the corrected history.
8. New set + full_rebuild still rebuilds without source changes. Source correction
  can rebuild with either model-change mode, using the selected set for all rows.

## Verification, 2026-09-29

- Local source-policy/output/project/template subset: **64 passed, 13 skipped**.
  The skips are the real Delta scenarios; the local environment has no Spark/Delta.
- Four actual CLI multi-target initializer cases passed. Strict validation of a
  generated separate-views Bundle with the current wheel passed.
- Broader artifact/registry/custom scoring/history/branch regression suite:
  **285 passed, 5 skipped** (optional local capabilities). Test payload packaging
  was checked, and all **281 wheel module hashes** match the current source.
- Full repository Ruff, formatting, Ty CI scope and production CCN <= 10 passed.
- New real Delta scenarios cover mixed update/delete/insert across all three
  output modes, all-row deletion, failure preservation/retry and both engines'
  persisted temporal history after a correction and subsequent append.
- Runtime recovery happens **inside the next scoring invocation**. Selecting
  `rebuild_on_change` does not create a scheduler or table-update trigger.

Live acceptance payload: `rehearsals/clean_validation_20260929/`.
Wheel SHA256: `e8700764c7879cb1bfb3b1c53ca57875abdefdb0851f685993a262d4585d980a`.
Fourteen prepared serverless tasks cover contract/Delta tests, real UC lifecycle,
four-branch project training and saved recipes, 34 models x 5 search strategies,
strategy settings/CV/ensembles and nested policy acceptance over both engines.
Historical rehearsal assertions were updated to require actual nested search
evidence instead of the former post-selection diagnostic.

New catalogs using this account's Default Storage require the Databricks UI;
CLI creation was rejected by Databricks. A fresh schema was created instead:
`workspace.skyulf_validation_20260929`. The old lifecycle schema was preserved.
The user explicitly approved this destination/payload after automatic review's
initial rejection. Live run **161612155854062** started with 14 serverless tasks.
The contract suite completed: **516 passed, 4 CLI-only skips**. All **20 real
Delta tests** passed, including the source corrections and temporal continuation
cases. The four skipped CLI generation cases passed separately on the local CLI.
Real UC lifecycle and the four-branch project also passed. The broader matrix
found a nested-halving defect; corrected-wheel evidence and remaining test
environment checks are tracked in [report112](112-clean-databricks-acceptance.md).
The archive is uploaded as RAW: AUTO tries to expand a ZIP as workspace content.
