# SM-23h — Native refresh, cluster utilization and clear performance results

Date: 2026-10-05. Base commit: `c499e614` (SM-23g).

## Requested behavior and design

The user explicitly requested automatic native dashboard refresh, CPU/RAM cards,
removal of the Selected model table, and understandable performance decisions.
They also asked why native profiling was not used and how to enable performance
retraining. This question did not authorize enabling retraining on the test models.

The implementation preserves Spark measurement and the existing durable store.
After monitoring, a native Dashboard task refreshes the published dashboard.
No email subscription or shared-credential change is made. Classic node telemetry
supplies job-scoped CPU/RAM; unavailable serverless values are never displayed as zero.

## Changes

- Producer helper `src/tools/configure_monitoring_dashboard.py` accepts dashboard
  ID or copied published URL (including a page URL), optional warehouse, and one
  or more targets. It writes/removes only its owned target override. Validation
  errors leave existing files intact; unowned files are never overwritten.
- Configured producer graph adds `dashboard_refresh_enabled` after
  `evaluate_retraining`, then `refresh_monitoring_dashboard`. Refresh requires
  production mode and enabled monitoring. Unconfigured projects have no invalid
  placeholder task and no new initialization prompt.
- The central example supplies optional `monitoring_refresh.job.yml`, included
  alongside its dashboard resource. Its `dev` target extends `observe_models`
  with native refresh. This central development target intentionally provisions
  the shared service; it is distinct from producer development bypass.
- Performance removes the Selected model table and its unused dataset/bindings;
  model, version and context selectors remain. Performance checks replaces Policy
  history. Performance result and Training action replace ambiguous Decision/Action
  labels. Eligibility is explicitly different from an actual training request.
- Execution adds average/peak CPU and memory cards, availability, and two trends.
  SQL merges overlapping task intervals before matching node samples by account,
  workspace, cluster and time. Averages weight node observation seconds; peaks
  represent individual node samples. Missing CPU/RAM have separate denominators.
- README explains native Data Profiling versus AI/BI, deprecated `quality_monitors`
  versus current `data_quality`, refresh permission/cache boundaries, configuration
  commands, and per-model `mode: retrain` with existing evidence and request guards.

## Validation and live evidence

- Final 20 dashboard/compute tests passed. Three compute tests execute the actual
  shipped interval/weight SQL under SQLite with bounded fixtures: duplicate and
  nested tasks, partial minutes, disjoint portions of one node minute, duplicate
  node rows, null metrics and foreign identities. Missing datasets failed against
  the isolated prior committed dashboard before the new implementation passed.
- Refresh helper plus existing CLI generation consumers: 35 distinct tests passed.
  These include all three layouts and serverless/policy-cluster generation, invalid
  inputs, copied page URLs, quoted numeric target names and owned-file preservation.
- All 27 datasets passed 54 live default/NULL SQL executions. Native same-ID
  publication and source/API readback match succeeded. Published revision:
  `2026-10-05T10:46:40.158Z`; SHA256:
  `9e7bd3b8cb8441f707ba740e5e00d218237e5baa2c1257110567f3665e5cb739`.
  Final availability-height/counter-description polish changed widgets only;
  exact dataset definitions still match the 54 validated SQL cases.
- Read-only Databricks VALUES fixtures independently exercised the complete compute
  SQL, including task selection and explode. Expected CPU average 36.153846%, memory
  average 41.538462%, peaks 90%/80% matched. The shipped display uses ratios.
- Three producer Bundles passed strict validation and deployed with all nine job
  IDs unchanged. Existing training/scoring/monitoring tasks, schedules, policies
  and frozen wheel remained unchanged. Only native refresh tasks were added.
- Central optional resource merge passed strict CLI validation with both
  `observe_models` and `refresh_monitoring_dashboard` present. Initial duplicate
  top-level resource merging was rejected by the current CLI; target overrides
  resolved it. Validation output stayed in the ignored evidence directory.
- Independent read-only review found no blockers: filter bindings, non-overlapping
  layouts, policy action labels, dev bypass and compute weights were inspected.
  A complementary-null probe independently verified CPU 20%, memory 80%.
- Full CI Ruff, format, Ty and CCN-10 scopes passed. Ruff reported access warnings
  while traversing unrelated local directories. A new test formatting difference
  was corrected and the full format scope passed (1,333 files).
  Applicable pre-commit checks also passed, including YAML/JSON and full Ty.
  No frontend source changed; frontend build and MkDocs were not run.
- Authenticated Chrome confirmed the removed table, 40 predictions / 100% label
  coverage, readable Performance result / Training action values and historical
  loss, plus all CPU/RAM cards, trends and explicit serverless availability.
  Final availability height was checked at a narrow viewport. One automation probe
  clicked Refresh while the button was acting as Cancel; its aria-label confirmed
  cancellation. A completed refresh and cell-based waits resolved the probe.
  No dashboard query repair was needed for that browser automation issue.

Competition run
[825802718768390](https://dbc-45604623-c18b.cloud.databricks.com/jobs/382789798978890/runs/825802718768390)
completed in 370.291 seconds. All five tasks, including the native dashboard
refresh, returned SUCCESS. `evaluate_retraining` returned `{"status":"disabled"}`.
Fresh API snapshots confirmed all three training run-ID lists unchanged.
No training was launched. The live dashboard preserves its ID, warehouse and
`embed_credentials=false` publication setting.

## Limits and operational meaning

The visible last 30 days contain no classic node samples and 1,911 serverless
compute entries. The SQL is accessible and returns honest unavailable CPU/RAM.
A real nonempty classic-cluster telemetry join is not claimed as tested; numerical
join behavior is verified with live SQL fixtures and local regressions. No new
classic cluster was created just to populate a dashboard.

Node averages are not CPU-core-weighted fleet utilization. Shared-cluster samples
can include other workloads during this job. System tables can lag and very
short-lived classic nodes may be missing. Cost remains attributed list-price
estimation, not a complete invoice.

Native refresh uses the job Run as identity. `embed_credentials=false` remains;
other viewers retain independent permissions/caches. A Dashboard task refreshes
results, not every already-open browser tab. It does not calculate monitoring or
replace delayed-label revisit scheduling.

Drift and performance triggers form OR after independent opt-in. Performance
requires baseline, mature labels, coverage, tolerance and the configured consecutive
windows. Both triggers share fresh-training-data, cooldown and active/duplicate
request guards. A request trains the project; promotion/approval is still separate.
Test policies remain drift disabled and performance report-only.

Evidence: ignored `tmp_repro_artifacts/sm23h/` with SQL/publish readbacks,
`compute/FINDINGS.md`, numeric fixtures, `central-validate/`, browser probes and
`refresh/` deployment/job/run inventories. No credentials are included in this record.
