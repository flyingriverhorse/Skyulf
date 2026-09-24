# SM-32: independent policies, first implementation slice

Date: 2026-09-24. Status: ACTIVE, partially implemented locally.
Design: [independent selection and approval](38-model-selection-and-approval-design.md).
Baseline commit: `fcfcee31` (SM-30, SM-31 and company/readable tags).
The baseline passed 116 tests and the applicable commit hooks, including ty.

## Implemented at the library boundary

`skyulf.integrations.databricks.local_workflow.run_action` now validates
`score_model_selection` and `promotion_policy` independently. Automatic
promotion never overwrites a scoring pin. Champion selection resolves once
per score run, and missing champion fails before output preparation.
Both new fields are required; combining either with the old mode is rejected.
Legacy modes keep their established behavior and emit a migration warning.

New configurations use controlled champion resolution for evaluation in both
promotion modes. An explicit stale champion expectation fails before fit.
Automatic training requires an absolute quality gate regardless of scoring
selection. Manual mode still registers, nominates and evaluates a contender.
Scoring does not train or register a model.

The existing generated template keeps its legacy field and graph until the
complete lifecycle interface is ready. This avoids publishing a partially
migrated graph as a finished Bundle feature.

## Verification

- 17 new regression cases failed against the old coupled implementation.
- 78 focused tests passed, covering all four combinations on both engines,
  malformed/partial/mixed migration, missing champion, stale expected champion,
  automatic quality-gate requirements and legacy migration warnings.
- Real local MLflow stores exercised pandas and Polars with both scoring
  selectors: bootstrap, improved candidate, tied contender, manual contender
  and evaluation error. Existing notebook boundary and template tests passed.
- Log: `rehearsals/sm32-matrix.log` (local ignored test evidence).
- Scoped Ruff and ty passed. Independent read-only review found no blocking
  correctness or regression issues in the library/test changes.
- This slice has not been deployed or tested live in Databricks.

## Remaining SM-32 work, in order

1. Existing-version approve is now implemented at the library boundary; see
   [manual approval progress](45-sm32-manual-approval-progress.md).
   Finish reject using the same stored evidence and serialized writer.
2. Add controlled rollback to the same lifecycle entry point and keep all
   actions behind the same serialized writer. Preserve unknown-outcome rules.
3. Add previous_challenger on actual contender replacement, with controlled
   history receipts, retry idempotency and promotion/rollback reconciliation.
4. Wire independent prompts, lifecycle action parameters and optional score
   handoff to the two existing Bundle jobs. Match configuration and job graph;
   extend migration validation with SM-33 instead of silently rewriting pins.
5. Verify manual approval of a passing existing version without training,
   rejection/stale evidence, receipt retries and history transitions. Exercise
   the generated Bundle and obtain live evidence for the completed interface.

No Spark expansion or SM-33 work should start while this task is incomplete.
