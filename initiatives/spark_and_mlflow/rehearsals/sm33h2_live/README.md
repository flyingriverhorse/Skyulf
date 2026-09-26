# SM-33H2 personal serverless rehearsal

Status: SUCCESS. User explicitly approved this rehearsal on 2026-09-26.
All three tasks passed in 701.797 seconds; see report 62 for the retained evidence.
The sections below retain the approved scope and local preparation history.

The H1 baseline was committed as `55b31ca9`. The current H2 wheel must pass local
tests and review before upload. This harness uses the same public Core Bundle
action runtime; it is a test harness, not the production template.

## Exact resource scope

- Previously selected CLI profile: `skyulf`. CLI 1.17.0 and the user's identity
  were checked read-only on 2026-09-26.
- Existing schema: `workspace.skyulf_lifecycle_test`, confirmed read-only.
  No new schema and no modification of existing Bundle jobs or their schedules.
- New source: `workspace.skyulf_lifecycle_test.sm33h2_20260926_r1_source`.
  Exactly 240 synthetic records initially, then three appended records, CDF on.
- New models: `workspace.skyulf_lifecycle_test.sm33h2_20260926_r1_pandas_model`
  and `workspace.skyulf_lifecycle_test.sm33h2_20260926_r1_polars_model`.
  Three versions per model: bootstrap champion, promoted replacement, rejected
  challenger; finally roll back to v1. Existing models/aliases are not touched.
- New output tables: `workspace.skyulf_lifecycle_test.sm33h2_20260926_r1_pandas_predictions`
  and `workspace.skyulf_lifecycle_test.sm33h2_20260926_r1_polars_predictions`.
  Each receives 240 predictions, then three new predictions, then a no-op.
- New uploaded wheel, notebook and MLflow experiment under
  `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_lifecycle_test/sm33h2/r1`.
- One ephemeral serverless run with three sequential tasks: `pandas_train`,
  `polars_train`, `score`. Whole-run and task timeouts: 900 seconds. Task retries
  disabled and serverless auto-optimization disabled. No persistent job created.
  The score task starts separately and loads the recorded models/code afresh.

## Assertions

1. Source target-null rows and invalid ages are removed before splitting through
   the real H1 Core filters. Numeric imputation/scaling and three-fold CV run
   through the existing training implementation.
2. A candidate with a different age eligibility range is compared with the
   existing champion on that candidate's evaluation population.
3. Training emits versioned filter evidence and MLflow heldout/CV metrics.
   Editing today's Python file to raise must not prevent saved-evidence approval.
4. v1 bootstrap, v2 promotion, v3 explicit rejection and v2-to-v1 rollback use
   `run_bundle_action`, including automatic resolution of saved comparison pins.
5. Scoring through `run_bundle_action` keeps all requested rows, including rows
   excluded from training. Separate local and MLflow prediction inputs contain
   features only, with no target or filter-only age column.
6. The three-row append is incremental; another score is a no-op with unchanged
   Delta commit version. Sorted output keys must be exactly `0..239` initially
   and `0..242` after append, with model version 1 and finite predictions. The
   original 240 key/prediction/version triples must remain unchanged after both
   append and no-op. Registry model versions and output tables remain for inspection.

Local tests, not this live run, exercise intentional artifact corruption. The
rehearsal refuses existing source/target/model names instead of overwriting them.
A failed run is inspected before any retry; retries may need a new resource suffix.

## Before requesting execution approval

- Finish H2 local tests, independent review and scoped lint/type/doc gates.
- Build the current wheel; record its SHA-256 and verify generated Bundle config.
- Verify the harness imports and uses the final evidence/runtime interfaces.
- Review `submit.json` and its exact resource paths. Upload and `jobs submit`
  are separate, approval-dependent actions; no command runs automatically.

Expected persistent resources and compute use above require explicit approval
for this rehearsal. No permission to delete previous test resources is inferred.

## Local preparation evidence (2026-09-26)

- `verify_local.py` completed with exit 0 using real local SQLite MLflow for
  pandas and Polars: three registered versions each, bootstrap, promotion,
  explicit rejection and rollback to v1. Only Delta snapshot reads are replaced
  by the fixture. Local artifacts remain in `.cache/h2-harness-z2mi3e_4` for
  inspection; retaining them avoids MLflow's open SQLite handles on Windows.
- All 15 real WSL Spark/Delta training-date tests passed, with Delta required.
- All 56 installed-CLI Bundle generation tests passed. Strict dev validation of
  the generated project with the current wheel passed. Strict MkDocs passed.
- Harness Ruff and formatting checks passed. Runtime test evidence and final
  independent review are recorded in report 62 and the SDD task report.
- Wheel: `.cache/sm33h2-cli-final/test_cli_emits_independent_pol0/output/`
  `sm33_generated/dist/skyulf_core-0.9.0-py3-none-any.whl`.
- Wheel SHA-256: `d5982ff99ebfc8e9e5e572be05ae1b60a3fd88d974084943cf8336945cba186d`.

The local lifecycle harness was run before the final removal of a test-double
fallback in automatic evidence loading; its real-result path is unchanged.
The final automatic-path regression tests cover that removal. Review also found
a declared JSON-null receipt bypass; real pandas/Polars red/green tests cover the
fix, followed by 102 passing related lifecycle tests and full ty/Ruff checks.
The final wheel was rebuilt and strict dev Bundle validation passed again.
Live execution and exact Delta prediction output checks subsequently passed.

## Approved execution

Run ID: `926706369150614`. Task IDs: pandas `967082093196374`, Polars
`1090464827962261`, scoring `947036177005319`. The reviewed wheel and notebook
were uploaded to the approved r1 folder; one ephemeral run was submitted.
All tasks finished SUCCESS. Each engine produced 240 initial and 3 appended
predictions, followed by a no-op at Delta version 2. Exact row and prediction
identity assertions passed. See `verified-summary.json` and report 62.
