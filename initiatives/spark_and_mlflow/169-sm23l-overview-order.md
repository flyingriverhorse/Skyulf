# SM-23l - Overview summary order and registration wording

Date: 2026-10-05. Base commit: `a0b4fa4a`.

## User request and change

Explain Enrolled contexts and recent delivery; move Drift and data quality and
Latest saved performance higher in Overview.

The counter label is now Monitoring registrations, with an environment/project/model
explanation. Each identity is one enrollment; a model can appear in multiple
projects/environments, so five distinct models can produce six registrations.
Version changes update the existing enrollment rather than adding a new identity.
Paused registrations remain in this inventory total.

Final order: page title / selectors / inventory counts / Drift and data quality /
Latest saved performance / daily charts / model-table mapping. Both requested
sections are above the charts and mapping; widget IDs, data bindings and filters
are preserved. Daily chart labels use the same registration vocabulary.

Only dashboard presentation and its documentation change. All 31 dataset objects
and the other three pages compare equal to the previous source. No runtime,
job configuration, table data or retraining policy change is part of this turn.

## Verification

Final focused command:

```powershell
.venv/Scripts/python.exe -m pytest skyulf-core/tests/integration/platforms/test_monitoring_dashboard.py skyulf-core/tests/integration/platforms/test_monitoring_compute_dashboard.py skyulf-core/tests/integration/platforms/test_monitoring_overview_performance.py skyulf-core/tests/integration/platforms/test_monitoring_overview_summary.py --no-cov -q --basetemp=tmp_repro_artifacts/sm23l/pytest-dashboard-final
```

58 passed, including grid non-overlap and dataset/filter binding checks. No new
implementation-mirroring test was added for this reversible layout move.
An initial edit script stopped before writing on an incorrect axis-key assumption;
the consequent test setup lacked its temporary parent directory. The script was
corrected to use the actual widget displayName, the directory created, and the
final source above passed. No failed candidate was deployed.

All 31 datasets ran through the actual CLI with defaults and NULL parameters:
62 queries succeeded before publication. After Chrome exposed clipping in the
small counter description, only that text was shortened; all dataset objects
were verified identical to those 62 validated statements before republishing.

Published same dashboard ID01f1c090238e1b6da5d633032ad9960b at
2026-10-05T12:18:29.037Z; source SHA256 12e52053f1e2f0dcd32a5d6de4b3ef6646b7229f9f703863a952cae073ef3f21.
Warehouse d047a4d9aa276958 and embed_credentials=false preserved. API normalized
readback matched the source with existing deployment defaults.

Authenticated isolated Chrome verified the new label and6 registrations, with
summary headings before both the daily charts and model/table mapping. Actual
vertical positions were registrations414 / drift594 / performance1014 /
charts1494 / mapping1854 pixels, independent of page scroll. Screenshots inspected;
the counter description was shortened to fit. Counts remained5 models/6
registrations/3 scoring input tables/5 prediction tables.

Ignored artifacts: tmp_repro_artifacts/sm23l/.
No Python/runtime/frontend code changed; full Ty and frontend build were not
rerun for this presentation-only edit. Relevant dashboard tests and commit hooks
provide this change's local verification. No job reruns or model retraining were
needed. Pre-commit results are recorded by the signed commit.
