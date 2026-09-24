# SM-28b: Optional monthly retraining validation

Date: 2026-09-24. Branch: `090`.

The generated local-engine Bundle still has exactly two jobs. The default
`manual` mode has no schedule; `monthly_paused` adds a paused UTC monthly
schedule to the same `train` job, with `max_concurrent_runs: 1`. The `score`
job and its pinned model version do not change. No admission table or second
writer is introduced.

The monthly action calculates a calendar-month fit/holdout window, captures
the source's current concrete Delta version, and delegates to the existing
bounded label-aware Skyulf candidate-training service. It resolves a current
`@champion` to one concrete version for comparison, permits an absent alias
for first-model training, and propagates other registry errors. It does not
stage, promote, or score the new candidate. The source snapshot is pinned at
run start; the label cutoff is the start of the current UTC month. The
`label_at` column must faithfully record availability by that cutoff.

Verification:

- `pytest tests/integration/test_sm20a_bundle_template.py -q`: 21 passed.
  Cases cover year rollover, explicit source version, invalid lookback,
  missing version, champion resolution and scorer isolation.
- The template suite plus the underlying label-aware retraining suite passed
  together: 31 tests.
- `ruff check` on the changed workflow and integration tests: clean.
- Databricks CLI `bundle init` generated both manual and monthly projects.
  Inspection found only `train` and `score`, with `train_monthly` and
  `pause_status: PAUSED` only in the monthly project.
- Both generated serverless projects passed `databricks bundle validate
  --strict -t dev --profile skyulf` after a matching 0.9.0 wheel was built.
  Validation was read-only; no Bundle was deployed or job run.

A current-month live training run was not attempted: the existing personal
test source is historical and cannot provide the required recent labeled
fit and holdout rows. The optional schedule remains paused until a user
configures a real source and deliberately unpauses it. Re-running a month
can register another candidate; alias activation remains explicit.
