# SM-23g — Performance presentation and template cleanup

Delivered 2026-10-05, following SM-23f (`ef358161`).

## Scope and behavior

The user requested a clearer Model performance page, removal of unused template
notebooks, and an explanation of native dashboards, compute, storage and retraining.
They explicitly approved deployment to the existing single, competition and
model-set Bundles and one competition monitoring run. No new training was requested.

The native AI/BI performance page now presents:

- Selected model/version/context and latest measurement time.
- Predictions, matched outcomes and label coverage from the latest full observation.
- A selected measured metric and its history, with native circle markers so one
  measurement is visible without implying a trend.
- A separate latest policy decision, baseline/current/limit chart and compact
  eight-column history. Full reasons and IDs remain in reports and storage.
- Confusion counts beside classification availability, followed by all metrics.

The policy chart uses its own metric: selecting MAE does not hide an RMSE policy.
Latest policy selection ranks window end before measurement time, so late backfill
does not replace newer evidence. Missing evidence is not zero or an older success;
disabled policies have an explicit status. Policy-only records, including JSON-null
row-count markers, cannot replace full observation cards or confusion counts.

Six unreferenced generated notebook wrappers were removed: `drift_report.py`,
`retrain_on_drift.py`, `check_retraining.py`, `retraining_skipped.py`,
`monitor_model.py` and `train_models.py`. Current job and tool references resolve;
the public library entrypoints remain compatible. The `monitor_model` job task
still exists and runs the project monitoring notebook.

## Live delivery

Profile `skyulf`; existing namespace `workspace.skyulf_sm23d_20261005_dc93e08c`.
All three Bundles validated strictly and deployed in place. Nine job IDs remained
unchanged; no job resources were created or deleted. Eighteen obsolete notebook
copies were removed from generated projects and the corresponding workspace paths.
The wheel remained unchanged:
`202f703e1fbf2ea15dcfb561d38d1f91a47d693151a618ce0cec07930e433491`.

[Competition monitoring run 844667177339309](https://dbc-45604623-c18b.cloud.databricks.com/jobs/382789798978890/runs/844667177339309)
passed `monitor_model -> monitoring_report -> evaluate_retraining`. Drift was not
detected, performance was measured, the latest policy window lacked evidence,
and retraining was disabled. No training/scoring run was added; six source, training,
prediction and label Delta table versions were unchanged. Schedules remain paused.
The old run `572186726575447` retains its historical task names by design.

[Published dashboard](https://dbc-45604623-c18b.cloud.databricks.com/dashboardsv3/01f1c090238e1b6da5d633032ad9960b/published)
keeps ID `01f1c090238e1b6da5d633032ad9960b`, warehouse `d047a4d9aa276958` and
`embed_credentials=false`. Published revision: `2026-10-05T10:16:55.186Z`.
Source SHA256:
`9ba08ce3022cc9693327b365f490bb7d113496e646e3c7e87907b4f44d3be5d5`.

## Validation

- Template inventory regression failed first with exactly six unused wrappers.
  After cleanup, 15 CLI/template tests passed (12 generated projects, 192 resolved
  notebook references). Dashboard snapshot/policy regressions also failed before
  implementation. Final dashboard file: 15 passed. Total: 30 distinct focused tests.
- All 26 dashboard datasets executed successfully with defaults and NULL parameters:
  52 successful SQL statements before publication. Final native-marker change was
  widget-only; exact dataset definitions were compared and remained identical.
- Independent review probed empty policy anchoring, disabled policies, late backfill,
  deterministic ties, foreign contexts, JSON-null markers, zero/empty full reports
  and filter scope. Disabled labeling and missing context column were corrected;
  final review found no remaining blocker.
- Authenticated Chrome on the published page: single-model cards show 40 predictions,
  40 matched outcomes and 100% coverage. Observed MAE and independent RMSE policy
  both appear. Historical RMSE loss displays baseline 1.16, actual 1.53, limit 1.26;
  the newer unavailable window remains the latest decision. Regression explicitly
  has no confusion matrix. Classification shows weighted F1, its separate accuracy
  policy, and the measured confusion matrix `[[20, 0], [20, 0]]`. An initial browser
  probe read the native refresh/loading placeholder too early; waiting for the
  selected context and actual matrix cells resolved the probe without source edits.
- Full CI Ruff, format, Ty and CCN-10 static scopes passed. Applicable pre-commit
  checks passed, including full Ty. No full pytest suite, frontend
  build or MkDocs build was needed for this native dashboard/template change.

Ignored evidence: `tmp_repro_artifacts/sm23g/`, including SQL results, API definition
readbacks, published screenshots, independent review and `template-cleanup/` live
deployment/run inventories. These are verification outputs, not commit artifacts.

## Operator boundaries

The central README now documents the following explicitly:

- Cluster selection does not activate monitoring. Successful scoring calls a
  separate Spark monitoring job when configured/enabled in production mode;
  development targets bypass shared monitoring. This job has its own serverless
  environment. Monitoring schedules start paused for delayed-label revisits.
- `monitoring_catalog` and `monitoring_schema` select shared storage independently
  of model/data locations. Three Delta tables and three views hold inventory,
  observations, decisions and presentation. Four immutable Spark reference tables
  per evidence-bound identity preserve training/source/seen/receipt information;
  ordinary score runs reuse these references and append shared observations.
- We use native Databricks AI/BI, not the separate native `quality_monitors`
  profiling service. Dashboard Refresh reads stored results; it does not calculate
  metrics. No dashboard schedule or native Dashboard job task is configured.
  Table writes do not invalidate native caches; blank/latest selectors across
  concurrent updates can require Refresh. Explicit context selection and Refresh
  were used for final acceptance. No continuously pushed browser display is claimed.
- Execution/cost views use scoped `system.lakeflow` and `system.billing` queries.
  The reference template was inspected, not copied wholesale; cluster CPU/RAM
  panels are not implemented and attributed billing may have no rows yet.
- Retraining is OR across independently enabled drift and performance policies,
  followed by freshness, cooldown, active-request and duplicate guards. Performance
  additionally requires usable baseline, mature labels, coverage and consecutive
  qualifying windows. Both signals produce at most one whole-project request.
  Report-only, unavailable and historical-backfill evidence cannot independently
  request training. Existing model promotion and approval policy still applies.

SM-23f remains the preceding interaction/runtime repair. This follow-up does not
close the inherited production identity/approval gates of SM-37/SM-43b.
