# SM-54 — Single-target model competition

Status: DONE for single-target scope; local and live acceptance passed.
Baseline: `4bceb39f`. Changes are not committed yet.

## Delivered scope

- `training_layout=model_competition` competes models for one target. Existing
  `single_model` and `multi_target` behavior stays separate. Competitions inside
  model sets are not part of this delivery; SM-36d remains PARKED.
- `src/modeling/candidates.py:build_candidates(task)` owns named candidates,
  estimator parameters, tuning strategy/search space and preprocessing recipe.
  Task-compatible linear, forest and voting starters are provided for both tasks.
- Shared source snapshot, pre-split population, target, holdout, CV policy and
  selection metric are pinned before fitting. Learned preprocessing is fold-local.
- Candidates execute sequentially with bounded input, candidate count and search
  configurations. The configuration budget includes nested searches and halving
  survivors; it is not a total estimator-fit or memory budget.
- Ordinary fixed/tuned candidates use comparable shared-fold scores; tuned scores
  are labeled post-selection diagnostics. Nested candidates use genuine outer
  scores. The leaderboard is never ranked on the reserved final holdout.
- Complete successful evidence is required. Exact ties use ascending candidate
  name. Only the selected winner is evaluated on the reserved holdout and
  registered under the one configured model name. Losers retain child MLflow runs
  and fitted artifacts without registry versions.
- Existing quality gates, automatic/manual promotion, challenger nomination,
  rejection, rollback and score jobs apply to the selected winner. A failed
  quality gate does not try runners-up against the holdout.
- Saved winning custom code and fitted preprocessing travel with the artifact.
  Editable candidates/features are unnecessary for later scoring/operator actions.
- Common custom pre-split classes have one captured identity across different
  candidate preprocessing recipes. Fresh-process registration and scoring replay
  retain the common filter, its assets and exact dependency pins. Tampered common
  filters or a substituted builder are rejected.
- Existing phased jobs are reused. The legacy single-call `run_action(train)`
  rejects this layout explicitly rather than silently fitting one candidate.
- `select_best_model` and the final report show candidate, estimator, strategy,
  mean/spread, evaluation mode, child run and selected winner.

## Tests and analysis

- 353 integration tests passed in the combined evaluator/project/selection and
  existing lifecycle/notebook/runtime/output regression run (323.46 seconds).
- 12 additional complete local MLflow competition cases passed (46.79 seconds):
  both tasks/engines; fixed, tuned and voting candidates; all five strategies;
  ordinary/nested CV; tuned winner registration; failed quality without runner-up
  evaluation; failed candidate; modified selection evidence; retry rejection;
  custom saved-code replay after editable project code changes.
- Evaluator-specific coverage includes voting and stacking, group/time nested
  policies, threshold-aware nested scores, negative-scorer units, failed folds,
  incompatible split evidence and nonfinite/overflow rejection.
- Template agent: 204 template regressions passed, plus 11 competition cases
  covering real CLI generation for classification/regression × serverless/cluster,
  loading and generated preview; 11 ensemble template regressions passed.
- Final documentation/template regression: 70 passed, four opt-in CLI cases
  skipped in that invocation; the CLI cases passed in the explicit agent run.
- Full CI Ty scope, Ruff, backend/Core Lizard CCN10, generated schema freshness,
  changed-file pre-commit and diff whitespace passed. Ruff's full-tree run
  excluded only local untracked `.tmp` and `.tmp-review-model` test artifacts;
  no repository lint exclusion or threshold changed.
- No frontend or workflow source changed; frontend/actionlint/mkdocs builds were
  not required for this slice.
- Final shared-source regression: 179 passed and one existing Windows symlink
  privilege skip, plus 12 lifecycle cases passed. This includes fresh-process
  registration after project edits and saved-filter scoring with pinned NumPy
  and an asset. Full Ty/Ruff/format (1,129 files)/CCN10/schema and pre-commit passed
  again after the fix.

## Live acceptance

Host: `dbc-45604623-c18b.cloud.databricks.com`.
Workspace folder: `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_validation_20260929/sm54_r1`.
UC prefix: `workspace.skyulf_validation_20260929.sm54_20260929_r2_*`.

Wheel SHA256: `6b6653ae9b8560d9f848aca3d3b9eab636c29ab09e6fd1f21da044d4a95e4272`.
All 288 wheel Python modules matched local production source before upload.
Cloud notebooks independently verify the installed modules against this manifest.

Run `1042257718088140`: contract task `115817427417609` passed 372 tests with
four opt-in CLI skips. The real-Delta tasks hit a test-harness-only Spark Connect
save-mode spelling error before training; the run was cancelled to stop automatic
optimization retries. The notebook now uses `mode("error")`, and later submissions
disable automatic optimization explicitly.

Run `837376719969902`: pandas task `89297945335675`, Polars task
`881652152092624`. Both tasks and the overall run are SUCCESS. Both regression
and classification passed on each engine: fixed/tuned/voting candidates, one
winner per version, automatic initial/replacement promotion, manual first
approval, tied-repeat non-promotion, reject, rollback, scoring and no-op.
Pandas used ordinary CV; Polars used nested 2x2 CV.

Each engine/task registered three successive winners under one model name;
the final rollback restored champion v1. Each prediction table retained 120
unique keys. Regression selected the linear candidate; classification selected
the tuned forest in round two. Losing candidates remained child runs/artifacts,
and no child run recorded reserved-holdout metrics.

The acceptance calls
the real generated-job notebook adapters and durable graph-3 phases against
actual Delta inputs and UC model versions. No old test resources were deleted.

Final shared-source wheel: `765d1d2bec006d7059ca788c50e6c42a21074c470b6e8bb000384224d49afa87`.
Its 288 Python modules match the final source exactly. Run `845512547977070`
uses isolated `sm54_r3` workspace/UC names and tests the new common-filter capture,
different named recipes, a tuned voting winner, saved asset/pin replay and keyed
scoring exclusions. All three tasks and the overall run are SUCCESS:

| Task | Task run ID | Verified outcome |
| --- | --- | --- |
| Final contracts | `9312502618133` | 134 passed, zero failures/skips; exact installed module hashes |
| pandas | `739702773790075` | Tuned voting winner v1/champion; shared filter + distinct recipes; saved asset/pin; 124 keys = 121 predicted + 3 excluded |
| Polars | `342338837150916` | Same assertions; new valid row prediction approximately 11.5; invalid row retained with exclusion reason |

Editable feature files were removed after training and before winner registration.
Reloading the registered artifact retained the exact NumPy pin and `filter.json`
bytes. Scoring reused the captured custom completeness filter and gave the new
invalid record `pre_split:minimum_completeness` instead of dropping its key.

The full lifecycle matrix used the earlier wheel; the final wheel's focused
contracts and real Delta/UC replay cover the subsequent shared-source fix.
Local rehearsal receipts are in `rehearsals/sm54_20260929/` and
`rehearsals/sm54_20260929_r3/`; their `run.json`, `contract.json` and result JSONs
retain task states, hashes, leaderboard and lifecycle outputs.

## Explicit limits

- One target, sequential candidates. No AutoML dependency or automatic feature
  discovery. Model-set competitions remain outside this scope.
- Built-in row-changing candidate preprocessing is rejected; common eligibility
  belongs in pre-split. Custom Python remains trusted code and must preserve row
  membership/order. Preflight cannot prove arbitrary custom Python behavior.
- A nested leaderboard still selects among candidates. Its winning score is not
  an unbiased estimate of that overall selection; use the independent holdout.
- Existing lifecycle repair/retry requests are rejected before repeated effects.
  Inspect failure evidence and start a fresh invocation; ambiguous registry
  outcomes still require reconciliation.
- Production exclusive alias-writer ownership remains the existing requirement.

## Guided candidate setup follow-up

The user approved generating candidates from Bundle questions. Competition setup
now asks for 2-8 candidate slots with task-specific model menus, one common named
preprocessing recipe, shared strategy/default-or-custom controls and an aggregate
configuration budget (initializer default 1000). The runtime's existing admission
bounds remain unchanged. The lifecycle metric remains the common search/selection
metric; separate per-candidate metric questions are hidden.

Selected voting/stacking models open task-appropriate base-model questions and
voting weights/type or stacking meta-learner/internal folds/passthrough controls.
Classification ensembles also expose optional calibration. Inactive model slots
do not activate these questions. Ensemble base search spaces stay automatic.

`src/modeling/candidates.py.tmpl` renders a normal editable `candidates.py`;
users need no Python edits to activate their choices. Its `_candidate(...)`
calls allow later independent params/recipe/strategy/budget/space overrides.
Existing tuning/ensemble hooks retain their established override behavior.
The generated preview now prints all candidates with their individual details.

Verification: 207 related tests passed, including real CLI generation for both
tasks/compute modes, all five search strategies and four actual generated-recipe
training/selection cases (task x pandas/Polars). After the preview follow-up,
14 targeted CLI/schema tests passed. Strict validation of a generated serverless
Bundle passed with the previously verified wheel placed in its local dist folder.
No new wheel or cloud job was needed: this follow-up changes template questions,
generated Python/configuration and tests, not the installed Core runtime. The
earlier live acceptance above remains the evidence for that unchanged runtime.

### Independent ensemble setup

Each selected voting/stacking candidate now has independent base-model count and
task-specific dropdowns. No comma-separated model IDs or JSON weights need to be
typed. Voting asks for each selected base model's weight and classification voting
type; stacking asks for its own meta-model, internal folds and passthrough.
Classification candidates also have independent calibration controls. Search/CV
remain shared competition policies; Core derives automatic spaces from each
candidate's actual selected base models. The generated Python remains editable
for per-candidate preprocessing recipes and further overrides.

To avoid eight copies of these definitions, `competition_ensemble.json` defines a
single SLOT question group that build_schema.py expands into ordinary CLI fields.
The existing generated schema freshness gate covers this expansion; tests pin
unique prompt orders and reject incorrectly shared property names.

Verification on 2026-09-29:

- 218 related tests passed with the real authenticated CLI; no skips. This covers
  both tasks, five strategy configurations, distinct settings for two voting and
  two stacking candidates together, and all 13/14 base positions in candidate 8.
- Four generated-project training/selection tests (classification/regression x
  pandas/Polars) each fit standalone, voting and stacking candidates without edits.
- Seven final schema-builder tests passed after adding its invalid-name guard.
- Full Ruff/format, full CI Ty scope, Lizard CCN 10, schema freshness and changed
  file pre-commit hooks passed. A generated four-ensemble Bundle passed
  `bundle validate --strict --target dev --profile skyulf` using the previously
  verified wheel in its local dist folder.

The expired profile session was renewed by the user before the final CLI checks.
An intermediate offline check used a localhost test identity; the final 218-test
run used the actual skyulf profile. No new wheel, deployment or paid cloud job was
run for this template-only follow-up. Changes remain uncommitted.
