# Frontend CCN refactor batch 10 implementation plan

> For agentic workers: use subagent-driven-development with disjoint ownership,
> original-behavior characterization and independent task/integration reviews.

**Goal:** remove 34 selected CCN violations across 24 original production files
while preserving frontend behavior.

**Architecture:** separate coherent presentation and calculation policies beside
their current owners. Preserve shared interfaces and state/effect ownership;
three task groups can be reviewed and tested independently.

**Tech stack:** React, TypeScript, React Flow, Vitest, ESLint and Playwright.

**Spec:** the user requested continuation of the same parallel CCN workflow,
followed by controlled review, personal frontend checks and a later commit.
Prior conversation sets strict10, report8 and release notes under v0.8.19.

## Global constraints

- Behavior-preserving CCN refactor: strict <=10 per function, informational
  report at 8. Keep readable functions at 9/10; no waivers, higher limits,
  generic frameworks or pointless wrappers to meet a numeric quota.
- Base c56d6cca on the existing 0819 checkout; previous batch is committed.
  Baseline strict56 functions/44 files/max26; report147/106 with91 optional9/10.
  Queue62 open/4 parked; preserve OC-223/225/226/227/228 and parked
  OC-71/72/73/185. DRIFT-01 stays deferred. Confirm new defects against original
  source, check the tracker and record them separately without silent fixes.
- Three fresh Astra 6 implementers own disjoint tasks1-3. Primary owns task4,
  integration, browser scheduling, docs and assets. User explicitly requested
  this parallel workflow; no repeated approval or task-selection questions.
- Public exports/types, default values, DOM/text/CSS/accessibility, state and
  effect lifetimes, callbacks, ordering, optional properties, numeric fallback
  behavior and HTTP payloads must remain unchanged. Runtime state stays with
  its existing owner. Use type-only imports where needed to avoid entry cycles.
- Before production edits, characterize the original public behavior. Reuse
  meaningful existing tests; add missing boundary/branch coverage rather than
  implementation-shaped tests. Assert real values/actions/series; mock only
  external boundaries. New tests describe why the behavior matters.
- No backend/Core/dependency/API client/shared-store/registry changes. Helpers
  stay beside their owner or as cohesive module-local functions. Dedicated
  helper folder names are the entry stem with its first letter lowercased;
  preserve an existing same-name folder. Adjacent tests belong to that owner.
- Agents do not commit, spawn children, run full suites/builds/browsers, edit
  shared docs or alter another task's files. Report source stability, exact
  changed paths, original/final measurements and verification. Final focused
  noEmit may observe peer edits; report them without editing sibling files.
- Windows: use npm.cmd/npx.cmd and explicit UTF-8 for authored files. Avoid
  Python -c quoting and PowerShell brace globs; write temporary Python scripts
  under tmp_repro_artifacts. Capture native exit codes before reading logs.
  Use project tsc --noEmit, not tsc -b. Large uniform JSON follows TOON rules.
- Native briefs/ledger replace unavailable Bash helpers. Keep the established
  non-main checkout and ignored review artifacts. No cleanup of other plans.
  Existing user workflow overrides per-task commits and final workspace deletion.
- After sources stabilize, run focused consumer checks, full Vitest, ESLint,
  noEmit, both CCN reports, production build,11 bundle budgets and full Chromium.
  Use mocked HTTP with real stores/controls/charts for browsers. Wait for
  transitions/ResizeObserver and actual SVG geometry before screenshots.
- Independently review each task and the integrated result; fix confirmed
  regressions and review again. Update concise v0.8.19 notes, queue, tracker,
  inventory and this plan. Leave the new batch uncommitted for user checks.

## Task 1: Canvas connections and node inspection

**Owned production files and original CCNs:**

- `frontend/ml-canvas/src/components/canvas/ConnectionPort.tsx` — 26.
- `frontend/ml-canvas/src/components/canvas/FlowCanvas.tsx` — 16/12.
- `frontend/ml-canvas/src/components/layout/PropertiesPanel.tsx` — 11/23.
- `frontend/ml-canvas/src/components/shared/NodeInspectorModal.tsx` — 26.
- `frontend/ml-canvas/src/modules/nodes/inspection/DataPreviewComponents.tsx` — 14/24.
- `frontend/ml-canvas/src/modules/nodes/shared/ColumnMultiSelect.tsx` — 21.
- `frontend/ml-canvas/src/components/layout/resultsPanel/MergeWarningsBanner.tsx` — 26.
- `frontend/ml-canvas/src/core/hooks/useKeyboardShortcuts.ts` — 16.

Own adjacent matching tests and dedicated helper folders as defined above.

**Interfaces:**

Preserve all exported components, props and hooks. ConnectionPort and
FlowCanvas consume existing graph/registry/guidance stores; keep handles, port
IDs, pointer/keyboard behavior, rejection messages and history semantics.
PropertiesPanel/NodeInspectorModal retain selection, tabs, focus, resize bounds,
Escape/dismissal and request lifetimes. DataPreviewComponents retains table
shape/order, empty/null values, expansion and export behavior. ColumnMultiSelect
retains every variant, accessible name, filtering/ordering, disabled/loading
states, invisible selections and exact callback values. MergeWarningsBanner and
useKeyboardShortcuts retain notification keys/order, copy/navigation and input
guards, event subscriptions/cleanup and modifier semantics. Shared hooks,
stores, registry, ModalShell and node definitions are read-only.

**Characterization and consumer scope:**

Reuse the adjacent PropertiesPanel, NodeInspectorModal, DataPreviewComponents
and ColumnMultiSelect public tests, plus existing FlowCanvas/keyboard/connection
consumers. Characterize absent selections/results, stale requests, tab and
exclude/menu interactions, keyboard handling inside inputs, connected/disabled
port differences, source/target handles and unchanged resize/focus behavior.
Pin warning ordering/duplicates and exact public actions. All ten preprocessing
settings characterized in the preceding batch consume the real shared column
control, so include InvalidValueReplacementNode.test.tsx as a final consumer.

- [x] Read complete entries and relevant callers/tests, then write a small
  behavior map in the report. Save original source copies under the task's
  tmp_repro_artifacts folder. Extend only meaningful missing public tests.
- [x] Run the focused public characterization against original production and
  record exact commands/results before any source extraction.
- [x] Extract cohesive policies/presentation while retaining state and side
  effect ordering. Do not alter public APIs or correct incidental quirks.
- [x] Run the same public tests plus existing named consumers and scoped ESLint
  at complexity10 including all new production helpers. Run project noEmit
  after source stabilizes; report any concurrent diagnostics accurately.
- [x] Measure every owned source/helper, self-review the complete diff against
  c56d6cca, report original/final evidence and signal source stable. Pass an
  independent spec and quality review; address confirmed introduced findings.

## Task 2: Dataset lifecycle and model operation screens

**Owned production files and original CCNs:**

- `frontend/ml-canvas/src/components/data/DatasetPreviewModal.tsx` — 26.
- `frontend/ml-canvas/src/pages/DataSources.tsx` — 24/15/20.
- `frontend/ml-canvas/src/components/data/AddSourceModal.tsx` — 16.
- `frontend/ml-canvas/src/components/data/IngestionJobsModal.tsx` — 11/16.
- `frontend/ml-canvas/src/components/data/PipelineVersionsModal.tsx` — 18/11/12.
- `frontend/ml-canvas/src/pages/ModelRegistry.tsx` — 25.
- `frontend/ml-canvas/src/components/pages/DeploymentsPage.tsx` — 14.

Own adjacent matching tests and dedicated helper folders as defined above.

**Interfaces:**

Keep route exports and all modal/component props. Dataset selection, table
formatting/pagination, source validation and upload/ingestion payloads retain
their original semantics. Preserve pipeline version ordering, graph summaries,
restore/preview/download actions and confirmation behavior. Model registry and
deployment filters, loading/error/empty states, forms/actions, record links and
refresh/cancellation timing remain unchanged. API clients, stores, shared modal
infrastructure and format exports are read-only. Tests mock network boundaries;
do not mutate a live backend or invoke real deployment/data actions.

**Characterization and consumer scope:**

Reuse DataSources.test.tsx and any adjacent model/deployment/modal tests.
Extend public tests for table/missing values, pagination/sort/filter persistence,
source-specific fields and exact payloads, upload validation and response shapes,
ingestion status/poll cleanup, pipeline version summaries/restore confirmation,
registry/deployment action availability, error retry and selected record routes.
Test async request and modal lifetimes before extraction. Pin relevant branch
differences, including empty/zero values, instead of testing new helper names.

- [x] Read complete entries and relevant callers/tests, then write a small
  behavior map in the report. Save original source copies under the task's
  tmp_repro_artifacts folder. Extend only meaningful missing public tests.
- [x] Run the focused public characterization against original production and
  record exact commands/results before any source extraction.
- [x] Extract cohesive policies/presentation while retaining state and side
  effect ordering. Do not alter public APIs or correct incidental quirks.
- [x] Run the same public tests plus existing named consumers and scoped ESLint
  at complexity10 including all new production helpers. Run project noEmit
  after source stabilizes; report any concurrent diagnostics accurately.
- [x] Measure every owned source/helper, self-review the complete diff against
  c56d6cca, report original/final evidence and signal source stable. Pass an
  independent spec and quality review; address confirmed introduced findings.

## Task 3: Drift and statistical analysis views

**Owned production files and original CCNs:**

- `frontend/ml-canvas/src/pages/DataDriftPage.tsx` — 22.
- `frontend/ml-canvas/src/pages/drift/DriftTable.tsx` — 12/25.
- `frontend/ml-canvas/src/pages/drift/JobSelector.tsx` — 13.
- `frontend/ml-canvas/src/pages/drift/_hooks/useDriftReport.ts` — 13.
- `frontend/ml-canvas/src/pages/drift/SummaryCards.tsx` — 11.
- `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ShapExplainabilityView.tsx` — 22.
- `frontend/ml-canvas/src/components/eda/tabs/BivariateTab.tsx` — 18.
- `frontend/ml-canvas/src/components/eda/tabs/PCATab.tsx` — 12.

Own adjacent matching tests and dedicated helper folders as defined above.

**Interfaces:**

Preserve all exported props, route components and useDriftReport return
shape. Keep dataset/job selection, URL/history state, drift request/cancellation,
filters/sort/thresholds, warning classifications, PSI/KS displays and reference/
current row counts exactly as before. No drift backend/Core changes or DRIFT-01
enhancement. SHAP artifact coverage, run/class/feature selection, chart series
and data fallback semantics stay unchanged. Bivariate/PCA profile selection,
matrix/scatter orientation, labels, missing/zero data and chart geometry retain
their public contract. API clients, chart wrappers, profiles/providers and shared
format helpers are consumed read-only. OC-223 disposition modal is out of scope.

**Characterization and consumer scope:**

Reuse DataDriftPage and existing drift/table/report/SHAP/EDA public tests.
Pin job/schema metadata, missing and zero metrics, NaN/Infinity behavior, sorted
and filtered rows, selection/history transitions and async request lifetimes.
Pin displayed drift counts independently of metric severity without changing
algorithms. For SHAP and statistical charts, mock heavy chart renderers only at
their boundary and assert actual series/matrix/profile inputs, labels and
selected run/class/feature callbacks. Preserve current empty/unsupported states.

- [x] Read complete entries and relevant callers/tests, then write a small
  behavior map in the report. Save original source copies under the task's
  tmp_repro_artifacts folder. Extend only meaningful missing public tests.
- [x] Run the focused public characterization against original production and
  record exact commands/results before any source extraction.
- [x] Extract cohesive policies/presentation while retaining state and side
  effect ordering. Do not alter public APIs or correct incidental quirks.
- [x] Run the same public tests plus existing named consumers and scoped ESLint
  at complexity10 including all new production helpers. Run project noEmit
  after source stabilizes; report any concurrent diagnostics accurately.
- [x] Measure every owned source/helper, self-review the complete diff against
  c56d6cca, report original/final evidence and signal source stable. Pass an
  independent spec and quality review; address confirmed introduced findings.

## Task 4: Primary-owned metric and ensemble formatting

**Owned production files and original CCNs:**

- `frontend/ml-canvas/src/core/utils/format.ts` — 12/17.

Own adjacent matching tests and dedicated helper folders as defined above.

**Interfaces:**

Preserve every existing export and signature in format.ts; other tasks
consume it without import changes. getMetricDescription retains exact wording,
prefix precedence, CV std/mean treatment and unknown/prototype key behavior.
extractEnsembleSummary retains array filtering/aliasing, optional own properties,
voting/stacking exclusions, zero/nonfinite number handling and calibration rules.
Keep other formatting, descriptions and model-label utilities unchanged.

**Characterization and consumer scope:**

Reuse format.test.ts and ensemble/job-summary consumers. Add missing public
characterization for CV/unknown metric contexts, supported ensemble prefixes,
malformed/empty buckets, mixed arrays, inherited keys, reference identity and
zero/NaN values where meaningful. Run original tests before extraction, then
same tests plus existing consumer suites. Extract cohesive description parsing
and ensemble option policies; avoid generic value-conversion frameworks.

- [x] Read complete entries and relevant callers/tests, then write a small
  behavior map in the report. Save original source copies under the task's
  tmp_repro_artifacts folder. Extend only meaningful missing public tests.
- [x] Run the focused public characterization against original production and
  record exact commands/results before any source extraction.
- [x] Extract cohesive policies/presentation while retaining state and side
  effect ordering. Do not alter public APIs or correct incidental quirks.
- [x] Run the same public tests plus existing named consumers and scoped ESLint
  at complexity10 including all new production helpers. Run project noEmit
  after source stabilizes; report any concurrent diagnostics accurately.
- [x] Measure every owned source/helper, self-review the complete diff against
  c56d6cca, report original/final evidence and signal source stable. Pass an
  independent spec and quality review; address confirmed introduced findings.

## Task 5: Primary integration and controlled delivery

- [x] Read all task reports and complete diffs, including new files; dispatch
  fresh independent reviewers with task briefs and evidence packages. Resolve
  introduced defects without broadening behavior changes.
- [x] Reuse existing browser suites for canvas focus/resize, connection guidance,
  column selection, data pages and drift. Add only missing representative public
  browser coverage to e2e/ccn10-inspection-data-analysis.spec.ts; preserve HTTP
  contracts with deterministic mocks and assert visible behavior/payloads.
- [x] Run shared gates after stable source: npm.cmd test, npm.cmd run lint,
  npx.cmd tsc --noEmit, npm.cmd run complexity:check/report, npm.cmd run build,
  npm.cmd run size-check and complete Chromium. Inspect relevant screenshots.
  Global strict failure is expected until the remaining backlog is cleared.
- [x] Regenerate the existing CCN inventory from measured output. Update
  changelog/0.8.x.md only under v0.8.19, opus_core_analysis-open_queue.md and
  opus_core_analysis-tracker.md. Record original-source findings immediately,
  preserve parked decisions, and verify generated asset references/ownership.
- [x] Final independent integration review, scope/whitespace/asset checks and
  manual frontend checklist; leave verified batch uncommitted for user checks.

## Measured result and verification

All 34 selected violations across 24 original production files are removed;
the new `connectionPort/portPresentation.ts` helper also passes strict10.
All four task groups passed fresh independent spec and quality review. Their
source/test diffs match the independently reviewed packages byte-for-byte.
Final independent integration review also passed with no introduced findings.
Scope, UTF-8, whitespace, older-release preservation and generated references
passed the final delivery audit. The user reported the frontend checks working
and authorized the batch commit on 2026-09-10.

| Group | Original violations | Final violations | Original/final public tests |
|---|---:|---:|---|
| Canvas and inspection | 11 | 0 | 145 tests / 13 files |
| Dataset and model screens | 12 | 0 | 35 tests / 8 files |
| Drift and statistical analysis | 9 | 0 | 39 tests / 10 files |
| Metric and ensemble formatting | 2 | 0 | 100 tests / 3 files |

- Strict10: **56 -> 22 functions**, **44 -> 20 files**, **max26 -> max24**.
  Its exit1 is the expected remaining backlog; selected sources/helpers pass.
- Report8: **147/106 -> 139/101** functions/files; **117** functions at9/10
  are optional. Policy, dependencies, API clients, backend and Core are unchanged.
- Full final Vitest: **2,330 tests / 176 files pass**. Normal ESLint, explicit
  new-E2E lint with `--no-ignore`, `tsc --noEmit`, build and11 size budgets pass.
- Generated main entry `index-BjHkPcSK.js`; **251** relative built imports resolve.
  Main gzip size323.7KiB is within the unchanged325KiB budget.
- Complete final Chromium: **131 passed**, no failures or retries. Four new
  cases at1440/1100px use mocked HTTP with real controls, stores and charts.
  Preview checks cover false/zero/null cells, sample100->600, actual schema/profile
  conversion, statistics and close/reopen reset. Drift checks cover selected job,
  uploaded file and thresholds, actual histogram bars after stable geometry,
  sort/details/filter, client threshold changes, downloaded CSV and reset.
- The first full browser run was130 passed/1failed: an unchanged theme case
  encountered a blank page while waiting for Upload File. Three unchanged
  diagnostic repeats and the final complete run passed. Cause unconfirmed;
  no production change or weaker assertion was made for that timeout.
- New preview tests initially matched both Close buttons; using the visible
  footer text fixes only the test selector. Screenshots wait for transitions;
  preview and drift images were visually inspected.
- Original inspector race **OC-229** is preserved and separately filed. Queue:
  **63 open / 4 parked**. OC-223/225/226/227/228 remain open; OC-71/72/73/185
  stay parked and DRIFT-01 deferred. Release notes are under v0.8.19 only.

Evidence is under `tmp_repro_artifacts/ccn10-*`: `vitest-final.log`, `lint.log`,
`browser-lint.log`, `types.log`, `strict10.log`, `report8.log`, `build.log`,
`size.log`, `browser-focused-final.log`, `theme-diagnostic.log`,
`browser-all-final.log`, `delivery-audit.log` and `screenshots/`.
Task-specific original/final source copies, commands and measurements are in
`ccn10-task1/`, `ccn10-task2/`, `ccn10-task3/` and `ccn10-format-*.log`.
Ignored reports and reviews are in the matching `.superpowers/sdd/` workspace.

## Manual frontend checks before committing

1. Canvas: connect nodes, open settings/inspection, change Input/Output and
   branch/split, resize the panel, select columns with search/All/None, try
   keyboard shortcuts and inspect merge warnings.
2. Data Sources: upload a file or open S3 source fields, preview sample and
   statistics, Load More, close/reopen, filter datasets and inspect pipeline
   versions/restore. Check ingestion status if a job is available.
3. Model Registry and Deployments: open version history/artifacts, inspect
   metrics and record links, and use your normal deployment controls if needed.
4. Drift: select a model, upload current data, run analysis, open Details,
   sort/filter, change thresholds and export CSV.
5. Experiments/EDA: switch SHAP runs/views, bivariate axes and PCA dimensions;
   check labels, charts and metric/ensemble descriptions.

The known OC-229 race can still occur when inspector responses complete out of
order. This batch does not claim to fix that separately queued behavior.
The user completed these checks and authorized committing the verified batch.
