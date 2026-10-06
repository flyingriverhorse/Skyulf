# SM-23f - Monitoring dashboard runtime and decision visibility

2026-10-05 follow-up to SM-23e, base d9628d7b. User reports empty native
dashboard selectors and charts despite prior successful direct SQL checks.
Previous SQL validation does not establish working dashboard interaction.

## Work groups

1. Reproduce actual dashboard query/default/filter binding failures. Compare the
   published definition, supported Lakeview schema and the supplied local example.
   Fix the cause with regression coverage; validate all dataset SQL before updating
   and publishing the same dashboard ID. State any remaining browser limits.
2. Expose independent drift and performance evidence in the monitoring report.
   Preserve the existing OR policy and a single guarded retraining request. Rename
   generated task nodes to monitoring_report and evaluate_retraining, retaining
   compatibility for existing public entrypoints. Test neither/drift/performance/
   both signals, unavailable outcomes and report-only policy behavior.
3. Inspect real saved baseline/current/window evidence. Populate missing historical
   evaluations only from actual saved predictions and eligible labels; never invent
   performance loss or imply a historical window authorizes a current retrain.
4. Explain the native AI/BI definition and shared deployment bundle location, and
   compare execution/compute with the supplied reference without copying unrelated
   serving-token metrics or treating deleted clusters as failures.

## Ownership and verification

- dashboard_audit: dashboard JSON, binding tests and live query/publication evidence.
- integration_review: reporting/retraining runtime, generated graph and focused tests.
- layout_live: real performance evidence and final generated-job acceptance.
- root: reference audit, central documentation, integration review and static gates.

Profile skyulf is already selected. Preserve unrelated AGENTS.md/.claude/
.tmp-review-model changes. Do not push. No whole local test suite or MkDocs run.
Run full CI Ruff/Ty/CCN and relevant pre-commit hooks. Keep fixture data labeled
as acceptance data and preserve source/prediction snapshots.

## Investigation evidence

- Existing retraining coordinator independently evaluates drift and performance,
  combines ready triggers and submits one idempotent request. Label unavailability
  cannot satisfy the performance degradation predicate. Current generated task
  naming and output summaries obscure this behavior.
- Supplied reference uses native Lakeview JSON resources. Its model-performance
  notebook copies champion training-run MLflow metrics into Delta; it does not
  measure live prediction/outcome degradation. Infrastructure queries use system
  compute, serving and billing tables. Serving-token metrics do not describe our
  batch model jobs; deleted clusters do not by themselves establish failure.
- Current monitoring.lvdash.json is the source of the published AI/BI dashboard,
  deployed through examples/databricks_monitoring/resources/monitoring.dashboard.yml.
  Producers link to that shared dashboard instead of creating a copy per model.
- A direct SQL gate using only empty strings missed NULL default parameters:
  predicates such as `:model = '' OR model_name = :model` reject all rows for
  NULL. Independent peer reproduction confirmed the difference across datasets.
- Authenticated Chrome also reproduced real model names hidden under
  `Filtered out`: model/version/context options shared a field-filtered dataset.
  Adding an associativity helper to one draft filter did not fix the symptom.
  The repair under test separates option sources and makes optional SQL
  parameters NULL-safe. A blank metric field filter also hides policy evidence.
- Browser hard reload can reuse native query result caches. The dashboard's
  own Refresh action is required when checking newly saved measurements.

## Verification log

Runtime, template and dashboard verification completed. The final dashboard
acceptance includes authenticated Chrome interaction, not only direct SQL.

- 129 distinct focused runtime/template cases passed: the original 112-case
  report/retraining union, 14 real CLI template generation cases, and three
  additional disabled/report-only explanation regressions. The final retraining
  file passed all 40 cases after the independent review correction.
- Full CI Ruff, formatting (1,336 files), Ty and backend/Core CCN <= 10 passed
  after that correction. No full pytest, frontend build or MkDocs run was needed.
- Final independent runtime integration review found no further defects in
  packaged entrypoints, generated task dependencies, saved-observation status
  reporting, legacy JSON compatibility or operator documentation. Tests were
  not repeated without a relevant source change.
- Final runtime wheel SHA-256:
  `202f703e1fbf2ea15dcfb561d38d1f91a47d693151a618ce0cec07930e433491`.
  All 353 packaged Python modules matched the source. All three generated
  bundles passed strict validation and deployed in place, retaining all nine
  train/score/monitor job IDs and paused schedules.
- Model-set monitor run [466385124633286](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/52053718085306/run/466385124633286)
  completed all three tasks successfully. Exported HTML shows feature drift,
  observed metrics and performance policy evidence for both components, with
  40/40 matched labels each. Retraining output explains report-only performance
  and disabled drift automation; no training request was made.
- The latest one-minute policy window correctly remains unavailable when no
  predictions fall in that window; this differs from measured full-batch metrics.
  Historical evaluation [111514651543587](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/690300845899767/run/111514651543587)
  used the original score windows, saved holdout baselines and actual labels.
  Single and model-set regression RMSE increased from 1.16349432208 to
  1.52983161853, exceeding the unchanged absolute tolerance of 0.1. Competition
  and classification remained within their tolerances. All four comparisons
  used 40 labels, coverage 1.0, report mode and no training action.
- Complete before/after Delta histories were unchanged: training/source v0,
  prediction and label tables v1. No data or policy was fabricated to show loss.
- Initial authenticated Chrome displayed classifier confusion counts
  `[[20, 0], [20, 0]]` while selectors remained broken; this evidence alone
  was not treated as dashboard acceptance.
- The independent-option draft subsequently displayed all five real model
  choices without the `Filtered out` group. Its 24 datasets passed all 48
  empty-string/NULL SQL cases before same-ID publication.
- Independent review executed all 15 exact context-selection SQL fragments with
  multiple contexts for the same model/version: defaults select one latest
  identity, explicit contexts select exactly that identity, and conflicting
  context/version filters return no rows. Additional latest-observation probes
  confirm a later failed/empty full observation suppresses older values while
  performance-window-only observations cannot replace the full observation.
  Unique names and page-specific parameter isolation also passed.

## Final dashboard acceptance

- Same dashboard ID `01f1c090238e1b6da5d633032ad9960b`, published revision
  `2026-10-05T09:39:20.535Z`. Source SHA-256:
  `a27f1db3aae51239e28c270dec2e5d832155757eedff29c2c7abf66f3dd848c0`.
  API readback matches the source with its approved namespace and execution
  defaults; embedded credentials remain disabled.
- Three regressions reproduced before correction. The affected dashboard test
  file passed 13 cases; after strengthening nullable-context coverage against
  vacuous regex matching, that one changed case passed again. Together with
  runtime/template verification, 142 distinct focused cases passed.
- Browser model/version/context transitions and incompatible-context empty
  state passed. Native Reset restores the latest matching identity. Independent
  option sources show all five model names; no associativity helper is shipped.
- Published single-model RMSE policy table shows degraded historical evidence:
  baseline 1.16, current 1.53, threshold 1.26, 40 labels and no training action.
  Published classifier confusion counts are `[[20, 0], [20, 0]]`.
- Published execution table shows run `552686286815455` as SUCCEEDED. Attributed
  billing has no rows; this is not represented as zero cost.
- Root independently inspected the published model selection and rendered drift
  charts in Chrome. The latest classifier PSI is below its own limit and the
  drift trend shows zero flagged features. Immediately refreshing during page
  transition briefly left a native refresh prompt; refreshing the loaded page
  returned its completed results. Native query cache freshness is documented.
- Final full CI Ruff, formatting (1,336 files), full Ty and CCN <= 10 passed.
  Applicable pre-commit text, JSON, Ruff, formatting, Lizard and Ty hooks passed.
  No frontend source changed; no frontend build or full pytest/MkDocs run.

Browser evidence: `published-browser-acceptance.json`,
`dashboard-runtime-final-summary.json`, `published-drift-root.png` and the
per-page screenshots under `tmp_repro_artifacts/sm23f/`. Chrome remains on the
published performance page with the single model and RMSE selected.

Ignored detailed evidence: `tmp_repro_artifacts/sm23f/` (deployment, live run,
exported HTML, browser snapshots) and
`tmp_repro_artifacts/sm23e/layouts/performance-window/` (historical comparisons).
