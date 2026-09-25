# SM-33: validated configuration and per-run scoring versions

Date: 2026-09-25. Status: complete for the local/personal-serverless scope.
Implementation is uncommitted, based on `5ba388ec`.

## Delivered behavior

- `workflow_config.py` validates version-1 resolved projects without remote
  calls. It reuses Core pipeline validation, registered calculators, training
  snapshot contracts, model selection validation and supported metric sets.
- Validation covers task/model/metric compatibility, finite threshold domains,
  limits, source/output resource names, distinct column roles, composite keys,
  reserved metadata collisions and action-specific training inputs. Unknown
  project settings fail explicitly. Real source and output schema checks remain
  in the existing read/preflight/publication services.
- Explicit migration returns a copy and maps the old coupled mode to separate
  selection/promotion policies. It rejects mixed policies, unknown versions and
  contradictory task/handoff choices. It neither invents snapshots nor deploys.
- Generated notebooks require the new config version and matching deployed
  contract/handoff markers. Both entrypoints and job definitions must migrate
  together. These are consistency checks, not an authorization system or a
  complete audit of arbitrary manual Jobs edits.
- Initialization supports regression/classification defaults, existing training
  and score source names, features, labels, event columns and composite keys.
  Manual training dates/version are unset until configured. Init-file budgets
  are positive integer strings and render as JSON numbers; array fields reject
  injection and non-JSON whitespace through actual CLI validation.
- `score_model_version` is an optional score-job parameter. Empty follows the
  saved selector; a positive concrete version overrides that run only. It does
  not mutate config or aliases. Lifecycle calls reject it; automatic handoff
  explicitly clears it. Both append and full-rebuild policies remain in force.
- Score reports distinguish the selected model from an earlier write's model,
  including no-op runs. English configuration docs, operator guide/diagram,
  generated README and changelog were updated.

## Local verification

| Check | Result |
| --- | --- |
| Config, runtime, output, notebook, workflow, lifecycle and approval suites | 178 passed |
| Template checks and real CLI generation/rejection cases | 40 passed (20 generated projects, 9 rejected inputs, 11 offline checks) |
| Real Delta append/replay test in the existing WSL environment | 1 passed; validates selected-model identity on first write, new inserts and no-op |
| Ruff lint/format on changed Python files | Passed |
| Full repository `ty` command | Passed |
| `mkdocs build --strict` | Passed |
| Existing dev Bundle `validate --strict` | Passed |
| Independent final correctness review | No important unresolved findings |

The affected-suite command included `test_databricks_workflow_config.py`,
`test_databricks_job_runtime.py`, `test_databricks_job_output.py`,
`test_databricks_bundle_notebook.py`, `test_databricks_local_workflow.py`,
`test_databricks_bundle_lifecycle.py` and `test_databricks_local_approval.py`.
CLI tests used `SKYULF_BUNDLE_CLI_TEST_PROFILE=skyulf`. The real Delta case was
`test_incremental_local_batch_discovers_appends_without_period_inputs`.

## Live personal-workspace acceptance

Profile: `skyulf`. Existing schema: `workspace.skyulf_lifecycle_test`.
Existing model: `workspace.skyulf_lifecycle_test.sm32_model_polars`.
Existing jobs: train `155738051514173`, score `684955889505992`.
No new persistent jobs, schemas, source tables or model versions were created.
The same two jobs retain 900-second limits and inactive schedules.

One deployment installed the new wheel and migrated the existing config/job
definitions. The subsequent runs changed only job parameters:

| Run | Input | Observed result |
| --- | --- | --- |
| `518924307260524` / task `484473496348514` | `score_model_version=1` | SUCCESS; selected v1, no-op, 0 output rows, Delta version 2 |
| `440928047261182` / task `952177177787198` | Empty override | SUCCESS; selected champion v5, no-op, 0 output rows, Delta version 2 |
| `805365754123156` | `score_model_version=0` | Expected negative result: version validation raised before Core dispatch/model loading/table writes; job displays FAILED/INTERNAL_ERROR |

No redeployment occurred between these runs. Exported notebook HTML was checked
for the selected v1/v5 labels, previous-write wording and no-op result.
Registry aliases remained champion v5, previous_champion v1 and
previous_challenger v4. Existing prediction rows were not rewritten.

Wheel SHA-256:
`9c653cf3ed7ae549a1bc3169dc664657db0b1eed7f4db222fa92ffd9e309cd6e`.
Remote wheel directory:
`/Workspace/Users/edwardwolfe99@gmail.com/skyulf_lifecycle_test/sm33/r1`.
Local evidence and reproducible driver:
`rehearsals/sm33_live/` (ignored scratch artifacts; the report is the durable summary).

## Boundaries and next task

Live SM-33 tests used existing Polars artifacts and unchanged input data. They
prove per-run model loading/selection and no-op behavior, not a new live
classification training matrix. Local tests cover both notebook engines and
task initialization; the focused real Delta test covers new prediction writes.
Company identity/permission readiness and the full combined acceptance remain
SM-37/SM-43 work. Schedule and window improvements are next in SM-34.
