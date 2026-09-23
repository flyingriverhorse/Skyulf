# Pre-Bundle Model Lifecycle Implementation Plan

> For agentic workers: execute the tasks in order, using a failing test before each behavior change.

**Goal:** Complete candidate/champion validation, explicit promotion, and a reproducible local-engine retraining workflow before generating the first Databricks Bundle.

**Architecture:** Skyulf Core owns engine-neutral held-out metrics; the optional MLflow adapter owns registered-version loading and the comparison policy. Alias changes remain a separate operation. Databricks jobs later call these services; the Bundle packages and schedules them without copying model lifecycle logic.

**Tech stack:** Python, pandas/Polars, Skyulf fitted local pipeline, MLflow registry, pytest. No new frontend/backend dependency.

**Spec:** [Databricks integration plan](04-databricks-integration-plan.md), SM-22 and SM-28 sections.

## Global constraints

- Compare concrete candidate and champion versions on the same bounded, labeled evaluation frame and record its caller-pinned source/split identity.
- Preserve the model's saved feature engineering, class order and tuned thresholds. Reject incompatible task or class contracts.
- Report evaluation only; never move an alias, deploy an endpoint, or write predictions as a side effect.
- Promotion and rollback are explicit, version-checked operations with a durable prior/new-version receipt. A missing champion is not auto-initialized.
- Monthly retraining must pin source version, label cutoff and temporal holdout before fit. Bundle scheduling is a later integration step.

### Task 1: SM-22a comparison report

**Files:** `skyulf-core/skyulf/integrations/mlflow/validation.py`; focused integration tests; MLflow user guide.

**Interface:** Compare two `ResolvedModel` references with a bounded pandas/Polars holdout, a target column, source/split identity, chosen metric and minimum improvement. Return model identities, both metric maps, comparison decision and reason.

- [x] Write failing tests for same-frame candidate/champion parity, regression and classification metric direction, pandas/Polars, missing champion, incompatible models and invalid inputs.
- [x] Load concrete registered local artifacts through the existing registry adapter and reuse `evaluate_local_holdout`.
- [x] Keep report creation read-only and deterministic; test that no MLflow alias changes.
- [x] Run focused tests, Ruff, ty and the existing local MLflow/holdout suites.

Local implementation and tests are complete. The isolated Unity Catalog gate
passed with four pandas/Polars metric runs and one pinned model comparison;
see the [live report](17-sm22a-live-metrics-report.md). No alias was mutated.

## Alias concurrency decision

The [MLflow alias client](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)
accepts model name, alias and new version, while the
[Unity Catalog alias PUT API](https://docs.databricks.com/api/uc-registered-models/v1/model-alias)
takes a new version number without an expected prior version. **Inference from
those API contracts:** read-then-set alone is not compare-and-swap and cannot
prevent a concurrent alias overwrite. SM-22b therefore requires one shared
non-expiring admission authority for every participating alias writer, plus
registry write permissions restricted to that controlled path. Re-read the
alias under admission, verify the expected prior version, mutate explicitly,
then verify and record the result. Writers bypassing that authority remain an
operational limitation and must not be described as race-safe.

### Task 2: SM-22b controlled promotion and rollback

**Files:** MLflow registry adapter; focused registry tests; registry guide.

**Interface:** Accept an approved comparison report, expected champion version and candidate version. Re-read the alias immediately before mutation, reject conflicts, set the alias explicitly and return a prior/new-version receipt. Rollback uses that receipt and an expected current version.

The promotion path must revalidate the pinned model identities and decision; an in-memory comparison report is not an authorization token and its metric dictionaries can be changed by a caller.

- [x] Write failing tests for normal promotion, stale alias, missing champion, access denial, no implicit first champion and rollback conflicts.
- [x] Implement explicit version-checked operations without changing the tracking run or inference code path.
- [x] Verify with a disposable local MLflow registry and an isolated UC permission/conflict run; see [live evidence](18-sm22b-live-validation-report.md).

### Task 3: SM-28a label-aware retraining workflow

**Files:** local Databricks workflow adapter; focused temporal-split tests; MLflow guide.

**Interface:** Consume a caller-pinned bounded source snapshot, label availability cutoff and temporal holdout; fit with Skyulf, log metrics/artifact, register a candidate, then call SM-22a. Do not promote automatically.

- [ ] Write failing tests for late labels, leakage across cutoff, reproducible split, failed candidate and unchanged champion.
- [ ] Reuse current fitted artifact packaging and registry publication; store source/code/model/evaluation identities.
- [ ] Run a small isolated Databricks validation after local tests pass, without touching production aliases.

### Task 4: SM-20a Bundle integration

**Files:** generated project and job resources in the SM-20 plan.

- [ ] Package the verified train/compare/promote/score services into separate jobs with explicit dependencies and permissions.
- [ ] Keep automatic incremental scoring as the default and pin the selected champion once per scoring run.
- [ ] Validate, deploy and run the two-insert rehearsal; add an opt-in retraining schedule only after the standalone SM-28a workflow passes.

SM-27 full-history rescore, Spark expansion, live endpoints and `ai_query` remain later independent work.
