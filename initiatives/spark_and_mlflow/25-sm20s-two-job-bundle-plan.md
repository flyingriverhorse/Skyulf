# SM-20S: two-job, table-minimal local Bundle

Updated: 2026-09-24. The user approved moving explicit `setup` into the first
`score` run and removing admission tables from the starting Bundle. The
single-writer boundary is intentional: one Bundle score job owns writes to one
prediction table, with `max_concurrent_runs: 1`. Other publishers must not
write that table. Existing admission-backed Core callers retain their stronger
cross-job coordination contract.

## Contract

- `train` is manual/on-demand and registers a candidate model with MLflow
  metrics. It creates no UC table.
- `score` checks the existing CDF-enabled source and pinned model; if its
  prediction table is missing, it creates exactly that table, then scores the
  first source snapshot or only new inserts. A repeated run is a no-op.
- The default generated Bundle has two jobs and no `setup`, comparison, stage,
  promotion or alias jobs. It creates no score/alias admission table. The
  target Delta transaction receipt and pinned model version stay in place.
- Default inference uses a concrete model version. A new model requires an
  explicit `model_version` configuration change and redeployment before scoring
  switches versions. This keeps candidate training separate from activation.
- `max_concurrent_runs: 1` serializes this one Databricks job only. Access to
  the prediction table must restrict other writers. The no-table option does
  not promise cross-job coordination. The former shared admission path remains
  available in Skyulf Core for multi-writer deployments.

## Steps and evidence

- [x] Add a focused failing Core test: a real local Delta initial/append/replay
  cycle using an explicit single-writer provider, without a control table.
- [x] Add the minimal provider and document its external single-writer
  precondition; keep existing admission validation for other callers.
- [x] Add failing generated-workflow tests for two job keys, no admission
  table/alias configuration, and score creating only its missing prediction
  table before invoking the Core incremental writer.
- [x] Refactor template config, workflow, prompts and job resources; keep both
  pandas and Polars, serverless and policy-cluster render paths, and four
  environment targets.
- [x] Update English Bundle guide, generated README, queue and changelog.
- [x] Verify focused tests, Ruff, Ty, generated strict Bundle validation and
  strict docs build. Perform a bounded personal serverless first/append/replay
  run, then inventory jobs and tables before claiming live parity.

Evidence: [two-job personal live report](26-sm20s-two-job-live-validation-report.md).
Only the serverless `dev` target was strictly validated and run; the policy
variant was rendered and parsed but needs an approved policy for live validation.
