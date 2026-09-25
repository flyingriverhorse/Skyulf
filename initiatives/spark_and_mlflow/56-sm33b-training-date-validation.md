# SM-33B: explicit training date interpretation

Date: 2026-09-25. Branch: `090`. SM-33A was committed as `fa7a1171` with
DCO sign-off and hooks; no push. This report covers the subsequent SM-33B work.

## Contract and implementation

`TrainingDateSpec` declares `format`, `timezone` and `date_only` independently
for `event_time_parsing` and `result_time_parsing`. Native Spark timestamp
instants use default rules. Strings require explicit full numeric calendar
formats; local times require an IANA timezone. Dates require explicit local
midnight semantics. No automatic day/month, missing-year or locale inference.

The shared scalar parser validates calendar dates and rejects DST gaps and
ambiguous local times. Native timestamp columns use Spark integer microseconds;
strings, native dates and timezone-free timestamps use the shared parser on
Spark workers. Invalid dates are checked before the event window filter,
including invalid rows that would otherwise disappear outside that window.
A missing result is unavailable; a malformed result fails.

Distributed filtering and transport retain integer microseconds. The driver
restores aware UTC values before split/availability checks, keeping the local
row and decoded-byte limits. Validation can scan the pinned source snapshot;
these limits do not cap distributed validation work.

Parsing rules are serialized into `candidate_training_spec.json`, included in
the dataset identity and reconstructed for approval replay. The generated
workflow includes editable default parsing objects. Conditional initializer
prompts remain SM-33E, after the random/date-free contract is settled.

Implementation locations: `training_dates.py`, `local_retraining.py`,
`local_workflow.py`, `workflow_config.py`, `local_approval.py` and the existing
Databricks public API exports. English Bundle/SDK guides, generated README and
changelog explain the new contract and source-clock vs schedule-clock distinction.

## Verification

- Before adding generated defaults, the source-date template regression failed
  with missing `event_time_parsing`; it passed after the change.
- Actual CLI generation plus Bundle/template/lifecycle/approval/workflow/runtime
  regression suite: **161 passed** (5 existing legacy-policy warnings).
- Final focused parser/retraining/config suite: **72 passed**. Covers native,
  string and date policies, null values, strict formats, DST rejection,
  microsecond precision, both local engines and UTC boundary ordering.
- Updated actual MLflow approval suite: **7 passed**, including pandas/Polars,
  saved and explicit evidence, and nondefault string formats/source zones.
  Changing today's parsing config does not change replay's saved specification
  or dataset identity.
- Real Delta date suite in the existing WSL environment: **9 passed**. Exercises
  Spark Asia/Tokyo plus Python America/New_York, exact start/holdout/cutoff and
  microsecond boundaries, row/byte limits, mixed offsets, unknown results,
  invalid rows outside the window, DST gap/overlap and DATE/TIMESTAMP_NTZ rules.
- Full repository `ty check`, scoped Ruff lint/format and diff checks passed.
- `mkdocs build --strict`: passed.
- Independent read-only review and final scoped re-review found no actionable
  defects. A same-ZoneInfo DST-fold boundary-order regression was reproduced
  before changing comparisons to UTC instants; its positive/negative cases pass.

Representative commands:

```powershell
.venv/Scripts/python.exe -m pytest skyulf-core/tests/integrations/test_databricks_training_dates.py skyulf-core/tests/integrations/test_databricks_local_retraining.py skyulf-core/tests/integrations/test_databricks_workflow_config.py -q --tb=short -p no:cacheprovider
wsl -d Ubuntu -- bash .cache/sm15-linux-run.sh -m pytest skyulf-core/tests/integrations/test_databricks_training_dates_delta.py -q -p no:cacheprovider -o addopts= --tb=short
```

Spark/Delta fixtures run in separate processes; native Windows Delta lacks
Hadoop filesystem support here. Base tests do not require importing PySpark.
Timezone data comes from the execution environment; keep it aligned across
driver/workers/replay. Serialized rules store zone identifiers, not a tzdb copy.

## Scope boundaries

This is local verification. No new Databricks deployment, remote test runs or
resource deletion were performed; combined live acceptance is SM-33E.
Existing remote jobs still run their previous matching wheel/config.

Training still requires temporal inputs. SM-33C adds date-free/random training
and optional result availability; SM-33D connects Core CV. SM-34 remains WAIT
until the pre-SM-34 sequence is complete. New SM-33B work is not yet committed.
