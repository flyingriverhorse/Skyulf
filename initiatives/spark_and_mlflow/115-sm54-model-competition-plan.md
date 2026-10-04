# SM-54 Model Competition Implementation Plan

> **For agentic workers:** Use the executing-plans skill to implement this plan
> task by task. Checkboxes below reflect verified delivery; committing remains pending.

**Status:** DONE for the approved single-target scope; local and live acceptance passed. Uncommitted.
**Evidence:** [Delivery116](116-sm54-model-competition-delivery.md).
**Goal:** Train task-compatible candidates for one target in one lifecycle job,
select one winner using shared training-side evaluation, then reuse the existing
single-model registration and champion lifecycle.
**Architecture:** Reuse Core estimators, preprocessing, search spaces, tuning and
CV. Add bounded competition orchestration and immutable comparison evidence; only
the winner reaches registration and the alias writer.
**Tech stack:** Python, existing pandas/Polars adapters, Core/scikit-learn search,
MLflow, Databricks Bundle jobs and Delta source snapshots.
**Spec:** OPEN_QUEUE_updated.md, SM-54 acceptance section.

## Global constraints

- SM-36d remains PARKED at the user's request.
- Keep existing single_model and multi_target layouts and the two existing jobs.
- Introduce a proposed `model_competition` layout and project-owned
  `src/modeling/candidates.py`. This is distinct from independent-target branches.
- One target/task, source snapshot, eligible row population, holdout, selection
  metric/direction and evaluation fold plan per competition.
- Shared pre-split rules determine eligibility once. Candidate preprocessing may
  differ, but must preserve evaluation membership and fit only on training folds.
- Candidate definitions own estimator/fixed parameters, preprocessing recipe,
  tuning strategy/budget and optional search-space overrides. Reuse Core defaults.
- CV policy is common. Ordinary CV uses common folds; nested CV uses common outer
  and inner policies and ranks by outer evaluation, not the final search score.
  Do not mix ordinary and nested results in one leaderboard.
- Never rank candidates using holdout metrics. Evaluate only the selected winner
  on the reserved holdout against quality gates and the pinned champion.
- Failed winner gates keep the existing champion; do not try runners-up against
  the holdout until one passes. A tie with the champion cannot promote.
- All requested candidates must finish successfully; otherwise fail competition,
  preserve child evidence and perform no winner registration or alias mutation.
- Proposed first delivery uses sequential candidates (concurrency 1), explicit
  per-candidate and total budgets. Do not advertise simultaneous execution.
- Core/backend Lizard and frontend ESLint CCN remain <=10. No new AutoML dependency.
- Changes require focused local tests, real CLI generation and live Databricks
  acceptance. Never infer a cloud pass from bundle validation.

## Proposed configuration ownership

| Location | Responsibility |
| --- | --- |
| `config/workflow.json` | Source, target/task, holdout, CV, selection metric, output and promotion |
| `src/modeling/candidates.py` | Named standalone/ensemble candidates and their overrides |
| `src/features/pre_split.py` | Common eligible population |
| `src/features/preprocessing.py` | Named candidate-selectable preprocessing recipes |
| `src/modeling/tuning.py` | Existing optional custom search-space function |

The initializer asks the layout and common controls. Candidate-only settings live
in candidates.py instead of repeating model/tuning prompts for an unknown number
of candidates. Exact names/signatures below are proposed new internal interfaces.

## Task 1: Validate competition and freeze shared data

**Files:** create `skyulf-core/skyulf/integrations/databricks/local_competition.py`
for competition contracts/preflight; reuse `local_retraining.py` snapshot/split
services and `local_cv.py`. Test in `test_databricks_competition.py` under the
existing Core integrations test directory.

- [x] Write failing tests for mixed targets/tasks, duplicate names, candidate
  attempts to override shared rows/folds, incompatible metric and invalid budgets.
- [x] Validate and capture candidate/workflow dictionaries in the existing saved
  request contract; pin source/record membership and shared evaluation policy.
  Reuse existing JSON evidence rather than adding parallel public spec classes.
- [x] Pin champion identity at preparation and save recipe/config hashes.
- [x] Prove every candidate receives identical evaluation keys; time/group rules
  preserve temporal ordering and group/holdout isolation.
- [x] Run affected tests and analysis gates before moving to candidate execution.

## Task 2: Execute candidates and build comparable evidence

**Files:** create `local_competition_training.py` beside the contracts. Reuse
`local_search.py`, `local_ensemble.py`, `local_search_results.py`, Core tuning/CV
and local pipeline artifact builders. If necessary, extract the smallest fitting
boundary from `local_retraining.py` so fitting does not register or evaluate the
reserved holdout. Do not invoke independent branch registration for each candidate.

- [x] Write failing tests for fixed versus tuned candidates and mixed standalone/
  voting/stacking families on both engines.
- [x] Implement candidate fitting with a parent MLflow run and separate child runs,
  saving fitted artifacts, parameters, recipe, fold scores and failure evidence.
- [x] Reuse automatic/custom search spaces and all five strategies. Validate
  metric direction and finite results, including searchers with negative scorers.
- [x] Produce CandidateEvidence with explicit metric/policy, per-fold scores and
  membership digest. Do not blindly compare arbitrary best_score values from
  incompatible search stages/resources; evaluate finalists on the shared plan.
- [x] Nested mode uses existing fold-local inner tuning and outer held-out scores;
  the final full-training search remains separate from candidate selection scores.
- [x] Verify built-in preprocessing/ensemble calibration/stacking and threshold
  fitting keep evaluation rows out of fitting; document the row-preserving
  contract required of trusted custom Python.
- [x] Enforce budgets and preserve failed progress. Default to concurrency 1;
  budget exhaustion or candidate failure cannot silently produce a winner.

## Task 3: Select, register and activate one winner

**Files:** create `local_competition_selection.py`; integrate narrowly with
`lifecycle_tasks.py`, `local_workflow.py` and existing MLflow services. Tests:
`test_databricks_competition_lifecycle.py`.

- [x] Test deterministic selection from complete, compatible CandidateEvidence;
  persist the tie rule (proposed stable candidate-name ordering for exact ties).
- [x] Emit a readable leaderboard and immutable winner decision with provenance.
- [x] Register only the winner under one common registered model. Keep losers as
  inspectable child runs/artifacts. First registration is v1 regardless of family.
- [x] Evaluate winner against reserved holdout, absolute gates and pinned champion.
  Reuse manual/automatic promotion, challenger history, reject and rollback.
- [x] Save exact winning preprocessing/custom code for prediction replay.
- [x] Test stale baselines and retries: no duplicate registered version, no extra
  alias mutation, explicit reconciliation after uncertain registry outcomes.
- [x] Test failed gates do not fall back to the next candidate and do not overwrite
  champion predictions. Verify scoring after algorithm changes and rollback.

## Task 4: Connect Bundle configuration and readable job output

**Files:** template `schema/project.json` and applicable topic files; new
`template/{{.project_name}}/src/modeling/candidates.py`; existing train resource,
job notebook adapters and `job_output.py`; template README and SDK guide.
Regenerate `databricks_template_schema.json` using build_schema.py.

- [x] Add tests for initializer visibility and config validation before edits.
- [x] Wire model_competition into the existing train job and shared score job.
  Do not create per-candidate production jobs or a third lifecycle job.
- [x] Show candidate names, family, strategy, CV score/spread, winner and promotion
  outcome. Keep algorithm selection and production approval visibly distinct.
- [x] Document setup using one regression competition and one classification
  competition, each including an ensemble and named preprocessing recipes.
- [x] Generate real CLI projects for both compute modes and verify unchanged
  single_model/multi_target behavior.

## Task 5: Acceptance and delivery

- [x] Local tests cover pandas/Polars, regression/classification, voting/stacking,
  fixed/tuned candidates, five search strategies, ordinary/nested evaluation and
  applicable group/time/threshold policies. Unsupported combinations fail early.
- [x] Exercise exact ties, a failed candidate, budget exhaustion, stale champion,
  retry, manual reject/approve, automatic initial/replacement and rollback.
- [x] Live Databricks runs use bounded datasets and real generated entrypoints:
  regression+ensemble and classification+ensemble, winning-version registration,
  model reloading, score output/no-op and lifecycle failure scenarios.
- [x] Inspect child runs, registered versions/aliases and persisted Delta output;
  record task IDs and tested artifact hashes in an initiative receipt.
- [x] Run full CI Ruff/format/Ty scopes, CCN10, schema freshness and changed-file
  pre-commit hooks. No workflow or frontend source changed, so actionlint and
  frontend gates were not needed for this delivery.
- [x] Update the queue and delivery report with verified evidence.
- [ ] Commit with DCO when requested; no commit or push in this delivery turn.

## Implementation mapping

The planned concerns are implemented in `competition_project.py`,
`competition_evaluation.py`, `competition_training.py` and
`local_competition.py`, with narrow existing lifecycle/project adapters.
Tests are the five `test_competition_*.py` files; existing project/scoring
regressions cover the shared canonical source fix. The initial internal
filenames and class names above were planning sketches. Report116 records
the actual files, test boundaries, two wheel hashes and cloud task IDs.

## Boundaries

This is configurable model competition using existing Core functionality, not
automatic discovery of arbitrary features or preprocessing recipes. Candidate
CV scores select a winner; final holdout evidence assesses that chosen result.
Nested outer scores are not an unbiased performance estimate of the entire
multi-candidate selection procedure; preserve the independent final holdout.
Combining per-target competitions inside a multi_target model set is a separate
extension, not silently included in this first delivery.
