# SM-29: Automatic local champion selection before Spark expansion

The generated Bundle keeps two jobs: `train` and `score`. Initialization can
select `pinned_version` (current behavior) or `auto_champion`. The latter asks
for a task-compatible heldout metric, a nonnegative minimum improvement, and
an absolute quality threshold. All remain editable in `config/workflow.json`.

`train` fits a candidate on a pinned, label-aware Delta snapshot. The existing
comparison evaluates candidate and champion on the **same** temporal holdout.
An existing champion changes only when the candidate is strictly better,
meets minimum improvement, and passes the absolute quality threshold. The
first champion has no comparison baseline, so it may be initialized only if
the mandatory absolute threshold passes on that same holdout.

Alias mutation uses the existing prepared/committed registry receipts and
expected-version checks. The Bundle adds no control table: only the serialized
`train` job identity may write model aliases. This is an operational contract;
another alias writer voids the guarantee. Unknown alias outcomes fail closed
and require reconciliation. A pending model tag is verified before alias
mutation and cleared only after the committed receipt is verified. Automatic
train and score refuse pending tags or aliases lacking a controlled committed
receipt. The `train` job calls the existing serialized
`score` job after a successful training action. A candidate that fails the
gate leaves champion unchanged; scoring can still process new source rows
with the old champion.

In automatic mode, each score run resolves `@champion` once to a concrete
version and uses the existing `incremental_append` or `full_rebuild` policy.
Prediction rows retain their concrete model version. Alias promotion and Delta
publication are separate transactions: if scoring fails, the prior prediction
view/table remains, and the score job must be retried. Do not silently roll
back the model alias. Full rebuild must retain its score-before-view-switch
ordering. Manual mode continues to require a concrete `model_version`.

No Spark-native FE/model support, new endpoint, extra UC table, or third job is
part of SM-29. Live Databricks validation must cover first champion, passing
v2, rejected v3, append and rebuild behavior, and retry after score failure.
