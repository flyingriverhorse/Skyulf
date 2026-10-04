# Model-set challenger lifecycle implementation and acceptance

Goal: complete the set lifecycle with challenger history and explicit rejection,
then verify all current uncommitted set changes on Databricks before signed commit.

Architecture: reuse ChallengerLifecycle and the existing controlled alias receipt,
history and rejection replay mechanisms. Keep set quality and whole-set validation
mandatory. Do not change component model aliases or introduce another job.

## Plan

- [x] Add real-registry tests for nomination, displaced history, rejection replay,
  rejected approval, stale candidate and successful activation/rollback.
- [x] Add model_set_challenger.py for set identity/baseline validation, nomination
  and explicit rejection; reuse shared alias writer and receipts.
- [x] Update set approval to validate the controlled challenger and remove it on
  activation. Retain legacy functional-only SDK compatibility without bypassing
  the quality requirements of new packages.
- [x] Wire nomination into branch training for manual and automatic policies;
  expose reject through the existing train job and document tags/actions.
- [x] Run relevant local tests and CI analysis scopes. Build and verify a wheel.
- [x] Prepare isolated serverless acceptance: real UC/Delta pipelines for pandas
  and Polars with regression/classification/ensemble, quality failures, manual
  and automatic activation, rejection, challenger history, rollback and scoring.
- [x] Upload/run on the previously selected skyulf workspace; inspect every task,
  persisted result and remote aliases. Record concrete run IDs and limitations.
- [x] Review staged changes and run commit hooks; create DCO signed commit only
  after passing acceptance. Do not push or include unrelated temporary files.

## Local evidence

- New real SQLite lifecycle tests initially failed because the requested API was
  absent; implemented using the existing shared alias machinery.
- First expanded run: 55 passed, one obsolete mock failed. The mock was corrected
  to include explicit nomination. Final quality/project/auto suite: 38 passed.
- Actual CLI Bundle generation and layout/prompt checks: 31 passed.
- Full Ruff, format (1117 files), Ty and backend/Core Lizard CCN10 passed.
- Staged pre-commit hooks passed, including complexity and type checks. Frontend
  hooks correctly skipped because no frontend files changed.

## Live acceptance passed (2026-09-29)

Workspace/profile: `skyulf`, `dbc-45604623-c18b.cloud.databricks.com`.
Wheel SHA256: `92d7dc18ea8a5026c7d86421520cd3b9bef165622c83b4816560e0e3fd1324bd`.
Workspace files: `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_validation_20260929/set_lifecycle_r1`.
Serverless parent run: `119140789004450`.
Tasks: pandas `551257986947700`, Polars `440447540740168`, contracts `233552710097983`.
Owned UC prefix: `workspace.skyulf_validation_20260929.setlife_20260929_r1_<engine>_*`.
No older resources were deleted. Each pipeline verifies installed source hashes,
trains regression/classification/voting ensemble through the actual Bundle Python
entrypoint and exercises real UC/Delta APIs. All three tasks terminated SUCCESS; both persisted pipeline results are passed.
Final registry aliases and rejection tags were also read back independently.


### Observed results

- Cloud contract tests: **226 passed, 6 skipped**, zero failures/errors. All six
  skips need a Databricks CLI executable (Bundle generation); local CLI/prompt
  suite passed 31 tests. No test failures are treated as skips.
- Both engines trained three real branches: ridge regression (revenue), logistic
  classification (risk), voting regression ensemble (cost).
- Set v1: manual nomination creates challenger, approval clears it and creates
  champion. Model-set inventory tags identify all three exact model versions.
- Improved v2: automatic quality checks pass and replace v1; previous_champion=v1.
- Equal v3: all three comparisons fail strict improvement; champion remains v2.
  Manual approval cannot bypass gates. Explicit reject records the reason, leaves
  the contender available for inspection and replays the same receipt safely;
  subsequent approval is blocked.
- Equal v4: challenger=v4, previous_challenger=v3. Rejecting displaced v3 or using
  an outdated champion expectation fails without changing roles.
- Rollback/replay restores v1 and preserves challenger v4/history v3. Component
  model aliases remain empty throughout. Final champion v1 is intentional.
- Scoring returns 80 rows on initialization, replacement and rollback. Saved
  package scoring survives removal of editable branches.py; combined profit
  equals revenue prediction minus cost prediction. Repeated scoring is a no-op.
- all, combined_only and separate_views publications pass. Each backing table
  contains 80 rows; combined_only omits individual prediction columns and the
  combined named view returns 80 rows.
- Every installed Core module hash matches the reviewed wheel; local Core source
  also matches that wheel after hooks.

[Compact persisted receipt](114-model-set-live-receipt.json) records task IDs,
artifact hashes, lifecycle events, output manifests and independently read aliases.

### Scope and limits

This run used actual Bundle Python entrypoints, real Unity Catalog registration
and Delta publication through an isolated serverless acceptance job. It did not
redeploy the two persistent Bundle jobs or retest schedules, cross-user permissions
or concurrent writers. Full repository pytest and MkDocs were not repeated;
affected local/cloud suites and full Python analysis scopes passed. No frontend
source changed. No push; temporary scripts, model artifacts and wheels are excluded
from the commit.
