# SM-32 Bundle operator actions implementation plan

Date: 2026-09-25. Implements the already approved selection/approval design.
Prior history implementation remains uncommitted. Keep two persistent jobs.

## Scope and file map

1. Add a narrow `integrations/databricks/job_runtime.py` adapter. Parse job
   parameters, enforce lifecycle/score roles, call existing `run_action`,
   expose copyable operator evidence and decide optional score handoff.
   Training, evaluation, registry mutation and prediction stay in existing Core services.
2. Keep two thin template entrypoints with fixed lifecycle/score roles. The
   shared adapter owns widgets, target resolution, temporary artifacts,
   task-value publication and JSON output.
3. Change init schema/config/examples to independent `score_model_selection`
   and `promotion_policy`, plus `score_handoff=disabled|after_alias_change`.
   Quality gates are configurable for manual approval as well as automatic promotion.
4. Keep the stable `train` and `score` job resource keys. `train` owns all
   lifecycle actions via `lifecycle_action` and explicit evidence parameters.
   Its condition task calls the existing queued score job only when the
   successful action returns a champion transition and handoff is enabled.
   All new projects contain this graph, so changing a policy in config does
   not require conditional YAML regeneration. No new table or third job.
5. Update generated/root docs, changelog and queue. Legacy Core callers remain
   supported, but migrate old generated projects' config, notebook and job
   graph together; the new adapter refuses legacy/partial policy configuration.

## Operator inputs

- `lifecycle_action`: train, train_monthly, approve, reject or rollback.
- approve/reject: candidate_version, comparison_sha256 and
  expected_champion_version. Explicit `none` means bootstrap; blank is an error.
- reject additionally requires rejection_reason.
- rollback: promotion_receipt_json copied from a completed promotion result,
  plus expected_champion_version. Never infer latest or roll back initialization.
- Output includes candidate parameters and, when applicable, ready-to-copy
  rollback parameters. Scoring pins never change implicitly.

## Verification sequence

1. Failing parameter/role/handoff tests before implementing the adapter.
2. Exercise generated notebook actions and real pandas/Polars MLflow candidate
   lifecycles, including manual bootstrap, approval, rejection and rollback.
3. Generate actual projects with Databricks CLI for all four policy pairs,
   handoff modes, schedules and both compute branches; parse emitted YAML/JSON
   and validate resource/task dependencies and parameter bindings.
4. Strict dev validation on the selected skyulf profile, with a locally built
   matching wheel. No deploy/run in this local implementation slice.
5. Relevant regression suites, Ruff, full ty, strict docs build and review.
   SM-32 remains active until the live rehearsal verifies operator actions.

## Platform references

Use [job parameters](https://docs.databricks.com/aws/en/jobs/parameters) for
operator overrides and [task-value conditions](https://docs.databricks.com/aws/en/jobs/tasks/if-else)
for conditional handoff. The notebook uses the documented
[widgets.getAll](https://docs.databricks.com/aws/en/dev-tools/databricks-utils#widgets-getall)
mapping; selected serverless/policy runtimes meet its 13.3 LTS minimum.
