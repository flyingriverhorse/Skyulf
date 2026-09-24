# SM-30 challenger lifecycle validation

Date: 2026-09-24. Scope: local implementation and verification; not deployed.

## Delivered behavior

Bundle training explicitly nominates a registered contender before comparison.
Generic `train_local_candidate` does not mutate aliases unless its caller
provides `on_registered`; the Bundle supplies `ChallengerLifecycle.registered`.
Training and all alias operations require the same serialized writer.

Model-version tags show `validation_status` (pending, passed, rejected, error),
`validation_reason`, and `promotion_status`. A rejected contender stays visible
as challenger; the next registered contender replaces that pointer. Promotion
still requires fresh evidence and all existing gates. A first model uses its
absolute quality threshold before initialization.

Pinned scoring keeps its concrete model version. Training resolves the actual
champion for comparison; a stale explicit champion_version fails before fit.
An evaluation or final evidence-read failure records error for a pending
contender. Completed validation is not overwritten by later promotion failure.
Uncertain registry writes retain pending receipts for reconciliation.

Rollback preserves a separate contender with a verified lifecycle receipt.
It still refuses an uncontrolled alias or one pointing at either rollback
participant. Repeated nomination is idempotent; an older contender cannot
replace a newer one through the nomination API.

## Verification scope

The real local MLflow tests run the generated workflow for both pandas and
Polars: initialize v1, promote improved v2, retain tied v3, nominate manual v4
without changing the score pin, and retain failed-evaluation v5. Only Spark's
source-read boundary is substituted with a bounded temporal dataset.

Registry tests additionally cover retained-challenger rollback, fresh report
checks, first-model cleanup, repeated/stale nominations and partial status
writes. A partial registry transition must block controlled champion reads.
Bundle tests cover failure during final evidence checking and rejection of a
stale configured comparison version before training.

The generated README, Bundle guide, registry guide and 0.9.0 changelog describe
the new semantics. A second code review found no remaining actionable issue
after fixing stale pinned-training expectations and final-comparison errors.

Final verification:

- 104 tests passed in 74.57 seconds across MLflow promotion, validation,
  registry, Databricks local retraining, Bundle template and both-engine
  lifecycle suites. Log: `rehearsals/sm30-final.log`.
- Ruff passed on all eight changed Python files; scoped ty passed on the
  three library modules and two Core test modules.
- `mkdocs build --strict` returned exit code 0. Log:
  `rehearsals/sm30-docs-verified.log`. The existing unlisted segmentation page
  produced an informational message, not a strict-build failure.
- `git diff --check` passed. No commit or live deployment was performed.

## Boundaries and next work

No cloud job, alias, table, endpoint or schedule was changed. The retained
live SM-29 v3 record still reflects the previous implementation. SM-43a
provides the combined live acceptance gate for this improvement program.

SM-31 next extracts reusable orchestration from the generated workflow.
SM-32 separates manual/automatic promotion from pinned/champion scoring and
adds approval of an existing candidate without retraining. Production writer
ownership remains an explicit SM-37 and SM-43b gate.

Serving, ai_query, A/B and feature lookup remain approved work in
[the extension delivery contract](39-serving-and-feature-lookup-delivery-plan.md),
after SM-43a. They are not implemented by SM-30.
