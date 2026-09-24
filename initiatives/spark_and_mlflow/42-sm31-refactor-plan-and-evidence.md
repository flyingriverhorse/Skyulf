# SM-31 thin Bundle orchestration implementation plan

**Goal:** Preserve the verified lifecycle and publication behavior in reusable
Core services, leaving a small generated notebook boundary.

**Architecture:** local_workflow.py owns target binding, train/score selection
and orchestration. prediction_output.py owns output schema checks, generation
provenance and grant-preserving view activation. The notebook reads widgets,
loads the editable business config, invokes run_action and serializes results.

**Spec:** SM-31 in 37-local-bundle-improvement-program.md.
**Baseline:** User-requested SM-30/tag live run 607409241163605 passed first.
Work continues in the existing 090 checkout, preserving uncommitted changes.

- [x] Pin the notebook-to-library delegation contract with a failing test.
- [x] Move existing function bodies into two Core modules without changing
  selection, validation, output, admission or failure semantics.
- [x] Move service behavior tests into Core; keep generated config/job and
  notebook boundary tests, including pandas/Polars real MLflow lifecycle tests.
- [x] Run focused/full relevant suites, Ruff and scoped ty; review the extraction.
- [x] Build a wheel, verify imports outside the source checkout, regenerate and
  validate the Bundle, and verify its thin score entry point in the test workspace.
- [x] Update the English operator guide, changelog, queue and handoff with
  exact local/live evidence and the next task.

No new job type, control table, alias policy or automatic schedule is added.

## Local evidence

The notebook is 44 lines. AST review against the archived pre-extraction
workflow confirmed unchanged function/class bodies and constants. Every
existing test was retained; service tests now live in Core, seven generated
project tests remain in the root suite, and six notebook-boundary cases cover
both engines and all three actions. Real local MLflow lifecycle tests still
enter through the generated notebook's imported run_action.

110 tests passed in 90.08 seconds (rehearsals/sm31-final.log). Ruff and scoped
ty passed on all changed library modules and Core tests. A no-dependency
installation into an isolated target directory passed import and target
selection checks with Python -I, verifying module paths came from the wheel.

SM-31 wheel SHA-256:
b0a5113db1dbc05eddd2c2164c1abc16510acda2e2207c1a2cfdd4d9fb455f01.
The same two test jobs were updated after strict Bundle validation. The
rehearsal uses a distinct SM-31 wheel path to avoid cached old code.
Score verification run 383572337148835 and task 922087987590767 returned
SUCCESS. The installed-library notebook returned noop=true, zero new inputs
and outputs, and unchanged Delta commit version 2. This is a deployed score
check after extraction; fresh training through the extracted modules was
validated locally with both engines, while the preceding SM-30 live run used
the equivalent pre-extraction orchestrator. No new live model was trained by SM-31.

Strict MkDocs completed with exit code 0. A freshly initialized final-template
project also passed strict Bundle validation. git diff --check passed.
No commit was made. Next: SM-32 independent approval and score model selection.
