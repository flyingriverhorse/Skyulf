# Frontend CCN refactor batch 11 implementation plan

> For agentic workers: use subagent-driven-development with disjoint ownership,
> original-behavior characterization and fresh task/integration reviews.

**Goal:** finish the remaining 22 CCN violations across 20 original source files
and pass the whole-source strict10 gate while preserving frontend behavior.

**Architecture:** separate cohesive response, selection and presentation policies
beside their current owners; preserve all shared interfaces and effect lifetimes.

**Tech stack:** React, TypeScript, React Flow, Vitest, ESLint and Playwright.

**Spec:** user requested "onlarida bitirelim" after committing batch10 and
confirming22 violations remain. Same parallel Astra6 workflow, strict10/report8,
v0.8.19 notes, shared review/testing and user checks before a later commit apply.

## Global constraints

- Behavior-preserving refactor of the full remaining CCN backlog: strict10,
  informational8 unchanged. No exemptions, increased limits or narrow source
  list. Every existing/new production function must pass10 at the end.
- Base73c0b7e7 on existing0819 checkout; tracked tree clean, temporary artifacts
  untracked. Baseline22 violations/20files/max24; report139/101 with117 optional9/10.
  Queue63 open/4 parked; preserve OC223/225/226/227/228/229 and parked71/72/73/185;
  DRIFT01 stays deferred. New confirmed bugs require original-source reproduction
  and tracker check, then separate filing by primary. Do not silently fix them.
- Three fresh Astra6 implementers own disjoint tasks1-3; primary owns task4 and
  integration/docs/assets. User authorized the same parallel workflow and asks
  to finish all remaining violations. Preserve readable9/10 functions and real
  responsibilities; avoid metric-only wrappers and generic frameworks.
- Preserve every export/prop/type/default, optional-property semantics, DOM,
  CSS/text/accessibility, callback payload/order, number fallbacks and evaluation
  timing. State, effects, subscriptions, refs and timers stay with their owners.
  Prefer cohesive module-local helpers; if a dedicated folder is needed it is
  the owner stem with a lowercased first letter, beside the entry. Adjacent
  matching tests belong to the owner; preserve any existing helper folders.
- No backend/Core/API client/shared-store/registry/dependency/lint-policy changes.
  Read shared interfaces without editing another task's files. Mock HTTP rather
  than performing real uploads/deployments or other live backend mutations.
- Characterize original exported behavior before production edits. Reuse useful
  tests and add boundary checks that assert actual output, actions and data;
  do not merely mirror implementation. New tests explain why behavior matters.
- Agents may restore only their owned original source for extra characterization,
  always in try/finally. Report SOURCE FROZEN only after all original replays,
  final code/comment edits, self-review, scoped tests/lint and measurement.
  After that signal, no source writes unless primary explicitly requests a fix.
- Agents do not commit, spawn children, run global suites/builds/browsers, or edit
  shared docs. Root owns browser scheduling after every source group is frozen.
- On Windows use npm.cmd/npx.cmd, tsc --noEmit and UTF-8. Keep temporary scripts
  and logs under tmp_repro_artifacts/ccn11-taskN. Avoid nested Python -c quotes
  and destructive cleanup. esbuild may need approved normal config access;
  capture native exit codes before reading logs. Large uniform JSON uses TOON.
- Fresh independent spec+quality reviews for each group, then integrated review.
  Final gates: full Vitest, lint, noEmit, strict10 and report8, production rebuild,
  all11 size budgets and complete Chromium. Browser assertions use real controls,
  stores and chart libraries, wait for transitions and actual stable SVG geometry.
- Update v0.8.19, queue, tracker, inventory and this plan with executed evidence.
  Keep the established branch and ignored workspace; no per-task commits or
  cleanup. Leave this new batch uncommitted for user frontend checks.

## Task 1: Execution, training and evaluation hooks

**Owned production files and original CCNs:**

- `frontend/ml-canvas/src/core/hooks/useJobPolling.ts` — 20.
- `frontend/ml-canvas/src/core/hooks/useTrainingNodeContext.ts` — 14/20.
- `frontend/ml-canvas/src/core/hooks/useExecutionWarnings.ts` — 14.
- `frontend/ml-canvas/src/core/hooks/useTuningTrials.ts` — 13.
- `frontend/ml-canvas/src/components/pages/ExperimentsPage/hooks/useEvaluationFetch.ts` — 15.

Own adjacent matching tests and dedicated helper folders only.

**Interfaces and behavior:**

Preserve every export, argument, return shape and public type. Keep effects,
subscriptions, dependency arrays, request arbitration, timers, cleanup, refs,
state and callback ownership with their existing hooks. Extract coherent pure
selection, payload or response policies without changing invocation timing.
Job polling must retain HTTP/realtime merging, progress/completion behavior,
retry/backoff and selected-job transitions. Training keeps graph traversal order,
dataset precedence, task-specific submission, leakage blocking and exact full
graph/target/job payloads, labels, pending-state guards and store action order.
Execution warnings retain notification and persistence ordering, nullish values,
identity deduplication and swallowed persistence failures. Tuning/evaluation
retains trial merging, metric choice, filters, terminal states and cache behavior.
Shared stores, clients, converter and other hooks remain read-only.

**Characterization and consumer scope:**

Use existing useJobPolling.test.ts, useTrainingNodeContext.test.tsx,
useTuningTrials.test.ts and useEvaluationFetch.test.ts. Characterize public hook
outputs/actions against original source before extraction, including HTTP and
realtime ordering, changed job/run IDs, loading/error/terminal states, cleanup,
duplicate warnings and exact submitted payloads. Add adjacent useExecutionWarnings
coverage for notification store receipt, automatic bell event and persisted
ordered errors/warnings using mocked HTTP. Include training settings and
experiment consumer tests where relevant; assert meaningful external values.

- [x] Read complete owned entries, callers and tests. Save original source and
  write a brief behavior map; extend meaningful public characterization.
- [x] Run those tests against original source and record exact evidence.
- [x] Extract cohesive policies/presentation with unchanged public contracts,
  state/effects and invocation order. Self-review complete tracked/new diffs.
- [x] Finish all original replays and source edits. Run the same public tests,
  relevant consumers, scoped ESLint strict10 (including helpers/tests) and
  project noEmit. Record concurrent peer diagnostics without editing peers.
- [x] Measure all owned functions/helpers, report exact paths and original/final
  evidence, and declare SOURCE FROZEN. Pass fresh independent spec/quality review.

## Task 2: Jobs, experiments and monitoring pages

**Owned production files and original CCNs:**

- `frontend/ml-canvas/src/pages/Jobs.tsx` — 24.
- `frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx` — 19.
- `frontend/ml-canvas/src/pages/SlowNodesPage.tsx` — 17.
- `frontend/ml-canvas/src/pages/Dashboard.tsx` — 16.
- `frontend/ml-canvas/src/components/eda/JobsHistoryModal.tsx` — 15/15.

Own adjacent matching tests and dedicated helper folders only.

**Interfaces and behavior:**

Preserve route/public component exports and props. Keep query/mutation,
selection, observer, timer, modal and request state/effects in current owners.
Jobs retains task/non-task pagination pools and caches, deep-link resolution,
fallback loading, filtering, promotion/actions and record routes. Experiments
retains run order, dataset/model/metric filters, split toggles, evaluated run,
chart/table/diff switches, downloads, promotion and metric grouping/fallbacks.
Dashboard/SlowNodes retain API response handling, statistics/units, navigation,
filters and sort/refresh behavior. EDA history retains selected job, report
fetch guard, retry/cancel/confirmation and exact onSelect/onFetchReport arguments.
Task1 owns the evaluation hook; Task4 owns ModalShell/Navbar. Consume their
existing interfaces read-only. Do not alter shared clients/stores/chart wrappers.

**Characterization and consumer scope:**

Reuse Jobs.test.tsx, Dashboard.test.tsx, SlowNodesPage.test.tsx and existing
experiment/EDA history consumers. Add adjacent public characterization only for
missing branches: task tabs, pagination/deep links and filters; experiment run
selection, metrics/params/pipeline toggles and actual boundary chart inputs;
empty/zero/missing statistics and monitoring sort; EDA selected report fetch,
late responses and cancellation confirmation. Mock HTTP boundaries, retain real
router/query/modal controls when practical. Existing browser suites cover many
experiment and operational flows; root owns any browser additions and runs.

- [x] Read complete owned entries, callers and tests. Save original source and
  write a brief behavior map; extend meaningful public characterization.
- [x] Run those tests against original source and record exact evidence.
- [x] Extract cohesive policies/presentation with unchanged public contracts,
  state/effects and invocation order. Self-review complete tracked/new diffs.
- [x] Finish all original replays and source edits. Run the same public tests,
  relevant consumers, scoped ESLint strict10 (including helpers/tests) and
  project noEmit. Record concurrent peer diagnostics without editing peers.
- [x] Measure all owned functions/helpers, report exact paths and original/final
  evidence, and declare SOURCE FROZEN. Pass fresh independent spec/quality review.

## Task 3: Remaining processing and modeling settings

**Owned production files and original CCNs:**

- `frontend/ml-canvas/src/modules/nodes/processing/VectorizerNodes.tsx` — 21.
- `frontend/ml-canvas/src/modules/nodes/modeling/components/StrategySettingsModal.tsx` — 18.
- `frontend/ml-canvas/src/modules/nodes/modeling/TrainTestSplitNode.tsx` — 14.
- `frontend/ml-canvas/src/modules/nodes/processing/TimeSeriesNode.tsx` — 12.
- `frontend/ml-canvas/src/modules/nodes/modeling/components/HyperparameterInput.tsx` — 11.
- `frontend/ml-canvas/src/modules/nodes/modeling/components/SearchSpaceInput.tsx` — 11.
- `frontend/ml-canvas/src/modules/nodes/processing/TransformationNode.tsx` — 11.

Own adjacent matching tests and dedicated helper folders only.

**Interfaces and behavior:**

Preserve node definitions and registration exports, props, defaults, config
patch shapes, validation messages/order and callback timing. Extract coherent
control groups and validation policies beside their owner. Vectorizers retain
text-column choice, method-specific fields, coercion and preview payloads.
Strategy modal retains active strategy, local draft/reset lifetime, input names,
apply/cancel and conditional controls. Hyperparameter/search-space inputs retain
null/empty/zero/false values, numeric parsing, categorical/multi-value handling,
bounds and exact onChange values. TrainTestSplit retains ratios, shuffle/seed,
stratification and warning behavior. TimeSeries validation retains first-error
precedence and every field/method rule; Transformation retains operation choices,
numeric parameters and callback values. ModalShell and shared hooks/stores/
registry/ColumnMultiSelect are read-only, with their public contracts unchanged.

**Characterization and consumer scope:**

Read existing settings/registry/converter and consumer tests. Add adjacent
public settings tests where missing; drive actual exported components and node
validation rather than testing names of new helpers. Pin defaults, field
visibility, numeric zero/NaN/empty cases, local draft cancellation/reopening,
method switches and exact emitted config/Preview payloads. Characterize ordered
TimeSeries validation errors on original definitions. Use real shared controls;
mock schema/HTTP/execution boundaries. Preserve existing defects and report a
confirmed original-source reproduction separately instead of silently fixing.

- [x] Read complete owned entries, callers and tests. Save original source and
  write a brief behavior map; extend meaningful public characterization.
- [x] Run those tests against original source and record exact evidence.
- [x] Extract cohesive policies/presentation with unchanged public contracts,
  state/effects and invocation order. Self-review complete tracked/new diffs.
- [x] Finish all original replays and source edits. Run the same public tests,
  relevant consumers, scoped ESLint strict10 (including helpers/tests) and
  project noEmit. Record concurrent peer diagnostics without editing peers.
- [x] Measure all owned functions/helpers, report exact paths and original/final
  evidence, and declare SOURCE FROZEN. Pass fresh independent spec/quality review.

## Task 4: Primary-owned navbar, notification center and modal shell

**Owned production files and original CCNs:**

- `frontend/ml-canvas/src/components/layout/Navbar.tsx` — 13.
- `frontend/ml-canvas/src/components/layout/NotificationCenter.tsx` — 11.
- `frontend/ml-canvas/src/components/shared/ModalShell.tsx` — 11.

Own adjacent matching tests and dedicated helper folders only.

**Interfaces and behavior:**

Preserve all public props and defaults, exported modal types and sizes.
Keep ModalShell focus/portal/Escape listener and backdrop ownership untouched;
extract only cohesive header/label presentation. Preserve the generated title
ID algorithm, optional own props, truthy rendering, footer/body layout, z-index,
keyboard focus and exact close callbacks. Navbar retains view switches, readonly
chip and tablet behavior, help tab routing, notification location and focus.
NotificationCenter retains store reads, mark-read/clear actions, preview/jobs
and error-log navigation, counts and outside-click/Escape cleanup. It intentionally
opens only on click and has no open-bell event listener. Detail metadata retains
all truthy/nullish fields, colors and formatting.
Tasks1-3 consume these public interfaces; do not change their files or callers.

**Characterization and consumer scope:**

Reuse Navbar.test.tsx, NotificationCenter.test.tsx and ModalShell.test.tsx
against original source. Characterize title/header/close visibility, generated
and explicit IDs, truthy zero/empty values, dismissed versus persistent backdrops,
Escape and focus restoration. Pin notification metadata/actions/preview and error-log
navigation and read-only/tablet/help behavior. Run real modal consumers including
DatasetPreviewModal and the strategy/history modal tests after peers freeze.

- [x] Read complete owned entries, callers and tests. Save original source and
  write a brief behavior map; extend meaningful public characterization.
- [x] Run those tests against original source and record exact evidence.
- [x] Extract cohesive policies/presentation with unchanged public contracts,
  state/effects and invocation order. Self-review complete tracked/new diffs.
- [x] Finish all original replays and source edits. Run the same public tests,
  relevant consumers, scoped ESLint strict10 (including helpers/tests) and
  project noEmit. Record concurrent peer diagnostics without editing peers.
- [x] Measure all owned functions/helpers, report exact paths and original/final
  evidence, and declare SOURCE FROZEN. Pass fresh independent spec/quality review.

## Task 5: Primary integration and delivery

- [x] Read reports and diffs; dispatch fresh independent task reviewers and
  correct introduced findings within scope. Preserve original defects separately.
- [x] Add only missing representative browser checks and run them after all sources freeze
  in `frontend/ml-canvas/e2e/ccn11-final-controls.spec.ts`. Reuse existing suites
  for training/Preview, experiment selection, jobs routes, modals, notifications,
  settings and accessibility. Mock HTTP only; observe actual UI and payloads.
- [x] Run full Vitest, normal ESLint, explicit browser-test lint --no-ignore,
  project tsc --noEmit, complexity:check and complexity:report. Strict must pass0.
- [x] Rebuild with npm.cmd run build, run size-check (all11 budgets), then full
  Chromium with failure traces. Review actual screenshots and generated imports.
- [x] Regenerate inventory for zero required violations; retain informational8
  counts. Update v0.8.19, queue/tracker and plan with final executed evidence.
- [x] Final independent integration review, scope/whitespace/policy/reference
  audit and manual frontend checklist. Record user authorization before committing.

## Executed outcome

All 20 owned production files use module-local extractions; no new production
files, shared interfaces or dependency/policy changes were needed. Final source
freeze followed all original replays, test additions and self-review. Fresh
independent spec and quality reviews passed for Tasks 1/4, Task 2 and Task 3;
the four source/test diffs exactly match their review packages.

| Group | Original / final public tests | Selected violations removed |
|---|---|---:|
| Hooks and evaluation | 195 tests / 10 files, both passed | 6 |
| Jobs, Experiments and monitoring pages | 245 / 29, both passed | 6 |
| Processing and modeling settings | 330 / 15, both passed | 7 |
| Navbar, notifications and shared modal | 34 / 4, both passed | 3 |

Selections include overlapping consumers and are not additive. Full Vitest
in the frozen final tree passed **2,402 tests / 186 files**. Independent review
additionally reran the eight hook/shell suites: **98 tests passed**. The initial
root JSX extraction briefly caused import parse errors before source freeze;
the closing-delimiter extraction was repaired, then all original/final and full
tests passed. Existing expected error-path/jsdom stderr is also present in
original consumer tests. No original functional bug was newly confirmed.

| Final command (from frontend/ml-canvas) | Result |
|---|---|
| `npm.cmd test` | PASS: 2,402 tests / 186 files |
| `npm.cmd run lint` | PASS, zero warnings |
| `npx.cmd eslint e2e/ccn11-final-controls.spec.ts --no-ignore --max-warnings 0` | PASS |
| `npx.cmd tsc --noEmit` | PASS |
| `npm.cmd run complexity:check` | PASS: 0 functions above 10 |
| `npm.cmd run complexity:report` | 133 informational warnings / 96 files, all CCN 9/10 |
| `npm.cmd run build` | PASS, generated assets rebuilt |
| `npm.cmd run size-check` | PASS, all 11 budgets unchanged |
| `npx.cmd playwright test --project=chromium --workers=2 --trace=retain-on-failure` | PASS: 133 tests, no failures or retries |

The two new browser cases also passed their focused run. They exercise the real
Count, TF-IDF and Hashing definitions, shared schema controls, per-node values,
empty/null max features, conditional controls, expanded/narrow panels and Preview
payloads at 1440/1100px. Both expanded screenshots were inspected. Existing browser
coverage includes training/tuning, threshold save/toggle/clear, Jobs/Experiments,
modals, notifications, keyboard focus, read-only mode and responsive layout.
HTTP endpoints are mocked; live training and backend integration are not exercised
by these browser tests. No backend or skyulf-core source changed.

Required backlog: **22 functions / 20 files -> 0 / 0**. Highest source CCN:
**24 -> 10**. Informational CCN 8: **139/101 -> 133/96**, all optional 9/10.
The inventory retains each optional function's exact file, line, column and CCN.
Functional queue rows are byte-for-byte unchanged: **63 open / 4 parked**;
OC-223/225/226/227/228/229 and deferred DRIFT-01 retain their existing status.
Release notes are under v0.8.19; older releases are unchanged.

Main asset `index-B8ImxiHA.js`: **324.5 KiB gzip / 325 KiB budget**. All **251**
relative generated JS/CSS imports resolve. Ownership, UTF-8, EOF, source whitespace,
unchanged ESLint waivers, policy, queue and older-release audits pass. Generated
bundles retain load-bearing template whitespace and are excluded from source
whitespace repair. Evidence is under `tmp_repro_artifacts/ccn11-*`; task and review
reports are in `.superpowers/sdd/frontend_ccn_refactor_batch11_2026-09-10/`.

## Manual frontend checklist

1. **Training and tuning:** run a normal and a parallel training job; watch job
   status, progress and tuning charts. Open Optuna/halving strategy settings,
   edit/reset/apply values, close without applying and reopen. Check numeric
   hyperparameters and search-space bounds after leaving the field.
2. **Jobs and Experiments:** change tabs, status/search filters and Load More;
   open a job and return. Select runs, switch comparison/chart views and metric
   splits, then download results. Check saved threshold state after revisiting.
3. **Node settings:** try all three text vectorizers, split ratios/stratification,
   Transformation methods and Time Series settings. Expand/collapse the panel,
   switch nodes and confirm values survive; use Preview data.
4. **Monitoring and history:** refresh Dashboard and Slow Nodes, change sorting
   and open a detail. Open EDA job history, select a report and check its details;
   cancel the confirmation for cancelling a job.
5. **Navigation and modals:** switch Canvas/Experiments/Inference; open help and
   notification details. Check modal Close, Escape and returned keyboard focus;
   check the read-only control at a tablet width.

## Final integration review

Fresh independent integration review: **spec PASS, quality PASS**, with no
actionable introduced defect. The reviewer independently reran both delivery/
reviewed-source audit scripts and the source whitespace check, matched all 133
optional function paths/locations/CCNs/labels to the report across 96 files,
counted all 133 successful browser cases and inspected both expanded screenshots.
The review checked hook/page/settings/modal integration, documentation and built
imports; it distinguishes inspected gate logs from freshly executed audits.
Its report is `task-5-independent-review.md` in the ignored batch workspace.
The user authorized committing this verified batch on 2026-09-10. The source
and test diffs still match the independently reviewed packages; the checklist
above is retained for reference.
