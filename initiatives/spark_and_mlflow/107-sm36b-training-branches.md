# SM-36b: independent training branches — implementation and evidence

Date: 2026-09-29. Status: DONE (implemented and verified).
Spec: [report58, SM-36b](58-custom-fe-and-multi-model-bundle-plan.md).

Follow-up: [report108](108-sm36b-named-recipes.md) adds independently selected
preprocessing/pre-split recipes, inline asset help and shared-only initializer
prompts. Its successful cloud run exercises the actual notebook adapter and
registered scoring for eight models, extending the service-level evidence here.

## Goal and decisions

Train several independently useful target models from one pinned Delta snapshot
in one job. Each branch owns its features, preprocessing, estimator, tuning/CV,
quality metric and registered model identity. Every required branch must succeed
before the parent reports complete. Existing single-model projects stay the default.

- Sequential bounded execution reuses `train_local_candidate`; no fitted models
  are accumulated in memory by the coordinator.
- Validate all branches before opening Spark or MLflow. Distinct targets/model
  names distinguish this feature from same-target algorithm competition.
- Resolve latest source version once, or use the consistent explicit version;
  each branch reads that exact version. One run instant resolves rolling windows.
- Drop missing labels separately in each branch, including before bounded sampling.
  Other targets are excluded from branch features to prevent target leakage.
- Save a replayable plan with exact snapshots, configs, seeds and pinned champions.
  Fresh replay may register new versions; retries are not exactly-once operations.
- Link child runs to a parent before fitting. Record each completed candidate and
  a failure boundary. A failed branch leaves an incomplete parent, with no complete
  collection receipt. Preserve immutable partial candidates for inspection.
- This step compares candidates to each target's pinned champion but does not move
  aliases. Coherent model-set activation and composed scoring remain SM-36c.
- Bundle opt-in `training_layout=multi_target` uses the existing train job with
  one `train_models` task. The score job keeps its existing single-model contract.
  Multi-target training requires manual promotion policy and disabled score handoff.

## Implementation plan

- [x] Add explicit missing-label policy to LocalTrainingSpec with strict default
  compatibility; include sample eligibility and population-count regressions.
- [x] Add `local_branches.py`: validated branch contract, one-time snapshot pin,
  plan serialization/restoration and sequential parent/child orchestration.
- [x] Add optional Bundle layout, editable `src/modeling/branches.py`, fixed
  notebook entrypoint and clear configuration/runtime output.
- [x] Verify three real locally trained branches with independent labels,
  regression/classification/ensemble recipes, CV/tuning, replay and failure.
- [x] Run affected regression tests, full Ruff/Ty and CCN 10, plus actual CLI
  template generation for both layouts. Record untested cloud boundaries.
- [x] Review changes and update the authoritative queue/handoff with evidence.

## File ownership / review map

- `local_retraining.py`, `test_training_label_policy.py`: label admission and
  child-run tags. Existing default false preserves legacy training behavior.
- `local_branches.py`, `test_local_branches.py`: branch coordination and evidence.
- `branch_notebook.py`, template schema/resources/modeling/jobs,
  `test_databricks_branch_template.py`: optional Bundle entry and user configuration.
- This report, queue/handoff and central user guide: integration decisions,
  acceptance boundaries and delivery evidence.

## Scope boundaries

The earlier SM-36a scoring examples were changed to inactive comments at the
user's request. Those staged changes remain separate from this new work; their
54 local tests passed. Cloud SM-36a acceptance (156 tests) predates that template
default adjustment.

## Local verification

- Combined affected regression suite: **398 passed, 2 skipped**, 59 warnings
  in 351.80 seconds. Both skips require local Spark/Delta; cloud acceptance below
  exercises real Delta source reads and bounded missing-label sampling.
- Branch service: 37 passed. Actual pandas/Polars fits, SQLite MLflow registration,
  saved prediction, exact voting members, CV, replay membership, failed-parent
  evidence, saved endpoint binding and invalid-plan preflight are covered.
- Template regression: 162 passed, including actual CLI generation, serverless
  and policy-cluster layouts, both task types and notebook guards/delegation.
- Custom-step regressions: 21 passed. Existing training-only fixtures now
  explicitly disable scoring; the generated pre-split reuse fixture declares its
  required quality inputs. Production scoring validation was not relaxed.
- Full CI Ruff, format check (1,089 files), Ty scope, backend/Core CCN <= 10,
  and strict documentation build passed. No frontend files changed.
- Independent read-only review found and resolved differing prepare/run registry
  endpoints. Final review found no remaining concrete correctness issue.

## Databricks acceptance

- Workspace: `dbc-45604623-c18b.cloud.databricks.com`, profile `skyulf`.
- Isolated path: `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_lifecycle_test/sm36b_20260929_r1`.
- Wheel SHA256: `099bf58a01d07793351a0e20809470b95dd58d33fe2c43f31664c8e257e2be55`.
- Initial run `546997832198410`, task `958820953030624`, failed after the first
  registration because the rehearsal omitted the Bundle's `mlflow==3.16.1` pin.
  Runtime MLflow rejected `download_artifacts(registry_uri=...)`. No success is
  claimed for that run. The rehearsal dependency was corrected; source unchanged.
- Corrected run `612123557010767`, task `483504591561147`: TERMINATED/SUCCESS,
  notebook `status=passed`, no truncated result; execution 206 seconds.
  [Run](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/953161445688916/run/612123557010767).
- The smoke trains regression with CV, grid-tuned classification and a two-member
  voting ensemble on both engines. It checks independent missing-label pools,
  bounded sampling, source version preservation after an append, child linkage,
  registry identities, saved CV/tuning evidence and a failed second branch.


### Cloud results

| Engine | Target/model | Train / holdout | Missing labels excluded | UC version |
|---|---|---|---|---|
| pandas | amount / linear regression + 2-fold CV | 56 / 14 | 10 | 1 |
| pandas | category / logistic grid search + stratified CV | 56 / 14 | 10 | 1 |
| pandas | ensemble / linear + ridge voting, bounded sample | 48 / 12 | 10 | 1 |
| polars | amount / linear regression + 2-fold CV | 56 / 14 | 10 | 1 |
| polars | category / logistic grid search + stratified CV | 56 / 14 | 10 | 1 |
| polars | ensemble / linear + ridge voting, bounded sample | 48 / 12 | 10 | 1 |

- Model names share `workspace.skyulf_lifecycle_test.sm36b_20260929_37224abf_`,
  followed by engine and target. All alias maps stayed empty.
- Parent runs: pandas `e0679c10b2f2489c8cb6a55172d18b7c`,
  Polars `81b12228b99b47a9a3f298c6b79f6809`.
- Controlled second-branch failure: parent `9d6145c3b2e846ef92e8c832731179f7`
  was FAILED, progress retained only amount, failed_branch was category, and
  `branch_training_result.json` did not exist. The source fixture was dropped;
  test model registrations and MLflow evidence remain available.
- Saved plan JSON roundtrip, source version 0 after a source append, distinct
  per-target holdout membership, child parent/branch/plan tags, saved training
  specs and CV/tuning artifacts were asserted by the notebook.
- This live check calls the Core branch service directly. The notebook adapter
  and generated resource layouts are covered locally; persistent Bundle jobs
  were not deployed or changed. No model-set activation/scoring is claimed.

### Bundle packaging and final boundaries

Strict `databricks bundle validate --strict -t dev --profile skyulf` passed on
an actual generated multi-target/serverless project containing the tested wheel.
This exposed stale 0.9.0 wheel paths: all train/score references and the README
now match Core 0.9.1. A regression checks references against the actual Core build
version; 16 focused template tests passed after this correction.

Replay is a fresh run and registration, not an automatic repair of partially
registered components. The Bundle rejects repair/retry entrypoints. Candidates
are registered and compared without staging aliases; single-model approve/reject
shortcuts are not model-set activation. SM-36c is the next READY task.

Delivery includes the requested inactive-scoring-example changes and the named
recipe follow-up in report108. Rehearsal payloads and raw output stay under ignored
`rehearsals/sm36b_20260929`; no generated wheel or temporary project belongs in
the commit.
