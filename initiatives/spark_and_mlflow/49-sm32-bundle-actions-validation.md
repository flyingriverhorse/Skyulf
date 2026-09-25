# SM-32: Bundle operator actions and independent policies

Date: 2026-09-25. Branch: `090`; committed baseline: `6a7f9e95`.
Status: implemented and verified locally, uncommitted. SM-32 remains ACTIVE
until the live operator rehearsal. No deployment or job run occurred here.

## Delivered behavior

Initialization exposes independent `score_model_selection`, `promotion_policy`
and `score_handoff`. All four selection/promotion pairs are supported. A
promotion or rollback never rewrites a configured score version pin.

The existing `train` job owns training, approval, rejection and rollback.
Operators select `lifecycle_action` and copy explicit version/evidence inputs
from `next_actions` in the earlier output. Approval reuses the saved candidate;
rejection/rollback do not train or register another model. Empty bootstrap
versions are rejected; the explicit `none` value represents no prior champion.

The lifecycle job has three tasks: execute, check the `score_requested` task
value, and conditionally call the existing score job. Handoff is disabled by
default. When enabled it follows a successful champion initialization,
promotion or rollback. Rejection, tied/nonpromoted training and unresolved
alias writes do not trigger scoring. Score still follows its independent
selector. Failed score can be retried without training.

There are still two jobs, each queued with one active run, and no new control
table. Separate tiny notebooks fix their lifecycle/score roles in code. The
shared `job_runtime` adapter parses parameters and delegates to existing Core
services. Run parameters cannot change the score notebook into an alias writer.
Exclusive writer permissions remain an external deployment requirement.

New projects use the new configuration contract. Older generated projects
must migrate configuration, both entrypoints and job graph together. Core
direct callers retain the explicit legacy compatibility path.

## Verification

| Check | Result |
| --- | --- |
| Combined affected regression suite | **188 passed**, five expected legacy-policy deprecation warnings, 168.91 seconds |
| Real CLI template generation | **16 passed**: two selectors x two promotion policies x two handoff modes x two compute types |
| Strict Bundle validation | **8 serverless dev configurations passed**, using the selected `skyulf` profile |
| Wheel | Built `skyulf_core-0.9.0-py3-none-any.whl`; imported adapter directly from wheel and matched packaged source to working source |
| Static checks | Scoped Ruff lint/format, full repository ty, `git diff --check` passed |
| Documentation | Strict MkDocs build exited 0 |
| Review | No correctness blocker; added explicit injected-role/action and malformed rollback receipt coverage |

The regression suite covers real local MLflow artifacts for pandas and Polars,
manual bootstrap, saved-evidence approval, rejection, rollback/retry, historical
challengers, automatic promotion, tied candidates, independent score pins,
notebook delegation, parameter guards and failed/uncertain write boundaries.

Generation tests invoke the installed Databricks CLI rather than substituting
Go-template text. They inspect actual emitted YAML/JSON, dynamic references,
condition dependencies, Run Job routing, schedules and compute branches. Both
engines are represented. The optional CLI tests require the explicit
`SKYULF_BUNDLE_CLI_TEST_PROFILE` environment variable. The generation run had
one pytest-cache permission warning; all 16 cases passed.

Strict validation used CLI v1.17.0 and the built wheel. Policy-cluster projects
were generated and inspected, but company policy/compute resolution was not
validated against the personal workspace. Validation did not deploy resources,
run compute or prove runtime task-value/Run Job behavior.

Evidence under `rehearsals/`:

- `sm32-bundle-regression.log`: final combined suite.
- `sm32-bundle-generation.log`: real CLI generation matrix.
- `sm32_bundle_generation_tmp/*/output/sm32_generated/validation.stdout.json`
  and `validation.stderr.txt`: strict validation output for serverless cases.
- `sm32-bundle-final-ty.log`, `sm32-bundle-mkdocs-clean.log`.
- `sm32-bundle-wheel/skyulf_core-0.9.0-py3-none-any.whl`.

## Remaining acceptance

Use the existing personal test deployment for a controlled pandas/Polars
rehearsal. Verify manual candidate output, first approval, rejected candidate,
later approval and rollback through job parameters. Check actual task-value
conditions, queued score handoff, selector preservation and recovery without
new training. Preserve two jobs and no admission tables. Include the prior
`previous_challenger` slice in the live verification.

SM-33 continues broader configuration/migration validation; SM-37 owns enforced
production identities and writer permissions. Company targets and policy
compute remain separate acceptance gates. This evidence does not establish
production readiness or close SM-32.
