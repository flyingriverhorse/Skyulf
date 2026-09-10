# Frontend CCN refactor batch 9 implementation plan

> For agentic workers: use subagent-driven-development with disjoint ownership,
> original-behavior characterization and independent reviews.

**Goal:** handle a larger coherent batch without changing frontend behavior:
25 existing files containing 31 functions above CCN 10.

**Architecture:** preserve public exports, node definitions, subscriptions and
wire contracts. Extract cohesive policies, calculation stages and presentation
beside each owner; shared infrastructure remains read-only.

**Tech stack:** React, TypeScript, React Flow, Vitest, ESLint and Playwright.

**Spec:** user requested more parallel subagent work, final controlled review,
and then personal frontend checks. Existing conversation sets strict CCN 10,
informational CCN 8 and concise release notes under v0.8.19. Base `45c3f142` on `0819`.

## Global constraints

- Preserve behavior: exports/types, exact text/order, numeric zero/empty/NaN
  handling, defaults, callback/effect lifetimes, DOM/CSS/accessibility, filters,
  node identities and submitted params. Characterize relevant edge cases on
  original production before extraction. Reuse sound existing tests; add only
  meaningful missing behavior checks, not tests mirroring helpers.
- Target each selected entry and resulting helper at CCN <=10. Do not raise
  thresholds, add waivers, generic frameworks or pointless metric-only wrappers.
- Three Astra 6 agents work on disjoint groups; primary owns the EDA card,
  browser integration, docs, measurement and build. No agent commits, subagents,
  shared-file edits, global suites or production builds. Report source stability.
- No backend/Core/API/dependency/registry/store/shared-hook edits. Tasks preserve
  their consumed interfaces. Record independently evidenced existing defects;
  do not silently fix them during behavior-preserving extraction.
- Queue baseline61 open/4 parked. Preserve OC223/225/226/227, OC71/72/73/185
  parking and deferred DRIFT01. CCN baseline87 functions/69files/max32;
  informational159/113, with72 optional9/10 functions.
- Use npm.cmd/npx.cmd and UTF-8 explicitly on Windows. Never rewrite Unicode
  through default PowerShell/Python encoding. Use project tsc --noEmit, not tsc-b.
  Logs and original snapshots go under tmp_repro_artifacts; no temporary commits.
- Keep existing non-main checkout and user review workflow. Native Python
  creates equivalent briefs/ledger because Bash helpers were unavailable.
  Retain this plan's ignored review artifacts; do not delete earlier workspaces.
- Final shared gates run once after stable sources, with focused reruns only
  when changes or failures warrant them. Leave batch9 verified and uncommitted
  for the user's frontend checks.

## Task 1: Preprocessing settings group

**Owned source:**
- `frontend/ml-canvas/src/modules/nodes/processing/InvalidValueReplacementNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/DropColumnsNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/BinningNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/CastTypeNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/AliasReplacementNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/MissingIndicatorNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/PolynomialFeaturesNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/DeduplicationNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/DropRowsNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/FeatureInteractionNode.tsx`

Own adjacent matching *.test.tsx files and dedicated per-node helper folders
invalidValueReplacement/, dropColumns/, binning/, castType/, aliasReplacement/,
missingIndicator/, polynomialFeatures/, deduplication/, dropRows/,
featureInteraction/. Do not edit shared hooks or ColumnMultiSelect.

**Interfaces:** retain every exported NodeDefinition, settings props/config,
defaults, handles, validate and bodyPreview. Existing registry/converter/store,
upstream/schema/drop-column hooks and shared controls are read-only consumers.
**Baseline max CCNs in listed order:**21,20,17,16,14,13,13,12,12,11 (10 violations).

- [x] Map each node's existing controls, schema selection and callbacks. Add
  efficient public NodeDefinition/settings characterization and run against
  original production. Group similar fixture work in one owned test file if
  clearer; do not force ten duplicate harnesses. Keep tests able to distinguish
  upstream dataset IDs, filtering/order, dropped columns and missing schema.
- [x] Pin method/mode controls, hidden values, numeric zero/cleared inputs,
  checkboxes, column selection, custom values/lists, add/remove operations,
  validation/body summaries, existing result feedback and recommendations.
  Capture exact original configured payloads through public callbacks; ensure
  rerenders and narrow/wide settings do not lose controlled values or local state.
- [x] Extract cohesive control sections, schema/data derivation or feedback only
  where needed. Keep readable9/10 functions. Preserve DOM labels/classes and
  effect ordering; report any tempting behavior fix separately with evidence.
- [x] Run all owned public tests plus existing converter/node-validation/
  recommendation consumers; scoped ESLint including helpers at complexity10 and
  project noEmit. Do not run global suites/build while peers are writing.
- [x] Measure all owned production CCNs, self-review complete diffs against
  base45c3f142, record original/final commands/logs and any concerns in report.
  Signal source stable; pass independent spec and quality review.

## Task 2: Graph rules, comparison, layout and export

**Owned source:**
- `frontend/ml-canvas/src/core/utils/graphDiff.ts`
- `frontend/ml-canvas/src/core/utils/connectionValidation.ts`
- `frontend/ml-canvas/src/core/utils/pipelineLeakageValidation.ts`
- `frontend/ml-canvas/src/core/utils/pipelineCycleValidation.ts`
- `frontend/ml-canvas/src/core/utils/canvasLeakageIssues.ts`
- `frontend/ml-canvas/src/core/utils/canvasExport.ts`
- `frontend/ml-canvas/src/components/pages/experiments/pipelineDiffLayout.ts`

Own adjacent matching *.test.ts/tsx files and dedicated helpers beside these
owners: graphDiff/, connectionValidation/, pipelineLeakageValidation/,
pipelineCycleValidation/, canvasLeakageIssues/, canvasExport/ under core/utils;
pipelineDiffLayout/ beside its entry. Shared store/registry/converter/API are
read-only. Avoid helpers that import runtime state back through their entry.

**Interfaces:** preserve every named export and exact signature, model type
lists/messages, leakage sets/revisions/subscription functions, graph diff maps,
layout constants and PNG/SVG public functions. Other tasks consume these through
existing registry/store/format interfaces; no sibling imports need changing.
**Baseline:**diffGraphs27; connectionIssue21; leakage functions22/18;
findCycleIssues13; canvas leakage traversal11; canvas export helper12;
applyDiffStylingToSide callback13 (8 violations in7 files).

- [x] Extend meaningful existing public tests, run original production first.
  Connection rules: invalid/missing endpoints/handles, self and indirect cycles,
  model endpoint exceptions and exact precedence/messages. Preserve mutations
  and rejection timing; graph-store consumers must retain prior behavior.
- [x] Leakage rules: registry overrides/revision/subscriber order/reset, explicit
  stateless exceptions, target-only encoders, malformed params, split ancestry,
  branching/cycles/shared ancestors, deterministic issue/message ordering. Never
  weaken preprocessing-before-split restrictions or change CV exemptions.
- [x] Graph diff/layout: ID matching/fallbacks, duplicate/renamed/config/order
  cases, input immutability, map aliasing, edge handle identity, styling and
  layout coordinates/ordering. Export: DOM filtering, dimensions, bounds,
  background/theme, image async failure/cleanup and filename/options, with
  meaningful public tests using boundary mocks instead of testing new helpers.
- [x] Separate coherent policies/traversal/calculation stages while retaining
  runtime state identity and exact outputs. Do not redesign algorithms or
  change Map/Set ordering. Keep all resulting functions <=10 without waivers.
- [x] Run owned plus graph-store/converter/guided-connection/layout consumers,
  scoped ESLint/CCN10 and project noEmit. Measure original/final owned functions;
  write report with paths/commands/evidence, self-review and signal stable for
  independent spec/quality review. Do not run global suites/build or browsers.

## Task 3: Experiment summaries and evaluation charts

**Owned source:**
- `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ClassificationChartsForSplit.tsx`
- `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx`
- `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/RegressionChartsForSplit.tsx`
- `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/SegmentationView.tsx`
- `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/JobListSidebar.tsx`
- `frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/classificationCharts.ts`
- `frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/jobMeta.ts`

Own adjacent matching *.test.ts/tsx and dedicated helper folders next to entries:
classificationChartsForSplit/, perClassConfusionMatrix/,
regressionChartsForSplit/, segmentationView/, jobListSidebar/ in components;
classificationCharts/ and jobMeta/ in utils. Shared chart wrappers, stores, APIs,
hooks, EvaluationView, format and task-inference utilities are read-only.

**Interfaces:** retain exported component props, all utility/type exports,
metric aliases/fallbacks, split/class ordering, plot props and callbacks,
threshold/calibration navigation, selected-job/filter behavior and descriptions.
**Baseline:**ClassificationChartsForSplit20; classificationCharts13/16/18;
PerClassConfusionMatrix13/13; RegressionChartsForSplit14; SegmentationView25;
JobListSidebar13/21; jobMeta11/12 (12 violations in7files).

- [x] Map public inputs/call sites. Reuse existing tests and add missing public
  characterization before extraction. Pin supported nested/legacy API shapes,
  missing/zero/NaN values, positive-class identification, split defaults and
  ordering, class labels and matrix dimensions, empty and partial metrics.
- [x] Cover classification PR/ROC/confusion/calibration/per-class/chart toggles,
  regression fit/residual/QQ and uncertainty data, segmentation metrics/noise/
  reference labels and optional charts, job badges/subtitles/selection/filter
  behavior. Keep precision, labels, accessible names and actual plot series.
  Mock expensive chart renderers only at their boundary and assert their props.
- [x] Extract meaningful metric derivation and presentation sections while
  preserving memo/effect state and controlled callbacks. Preserve intentional
  quirks and stale-request handling; report existing defects, do not fix by guess.
- [x] Run owned plus EvaluationView/experiments/job-detail/classification consumers,
  scoped ESLint/CCN10 and project noEmit. No shared edits, builds, browsers or
  global test suite. Keep each entry/helper <=10 and avoid generic chart engines.
- [x] Self-review against45c3f142 and write report with original/final tests,
  commands/logs/CCNs and any concerns. Signal source stable for independent review.

## Task 4: Primary-owned EDA variable card

**Owned source:** frontend/ml-canvas/src/components/eda/VariableCard.tsx;
new adjacent VariableCard.test.tsx and helpers in variableCard/.
**Interfaces:** preserve profile/onClick/onToggleExclude/isExcluded props,
clickableProps behavior and ColumnProfile/dtype helpers. EDA tabs/providers,
VariableRow, chart wrappers and callbacks are read-only. Baseline card32.

- [x] Characterize original public card: every dtype mini chart, histogram
  formatting, categorical top5, missing/zero flags, normality, Healthy,
  excluded state, mouse/keyboard activation and exclusion propagation.
- [x] Extract chart data and cohesive status/exclusion presentation; preserve
  original DOM/CSS/accessibility, precision, props and callback behavior.
- [x] Run public tests on original and final source, relevant EDA tests,
  scoped ESLint/CCN10/noEmit; measure all helpers and pass independent review.

## Task 5: Primary integration and controlled delivery

- [x] Review scoped reports and full diffs including new files; dispatch fresh
  independent reviewers per group, fix confirmed regressions and review again.
- [x] Add real browser coverage for representative preprocessing modes/column
  choices/Preview payloads, graph rejection/diff/export, evaluation charts and
  accessible experiment navigation. Reuse existing suites rather than duplicate
  them. VariableCard has no current production consumers; verify its public
  component contract in Vitest instead of inventing a browser route.
  Wait for panel transitions/ResizeObserver before layout screenshots/assertions.
- [x] After stable source, run full Vitest, normal ESLint, project noEmit,
  production build,11bundle budgets and complete Chromium. Run both CCN commands;
  regenerate inventory from actual output. Globalstrict failure is expected
  while backlog remains; selected scopes must pass. Record failures honestly.
- [x] Update concise v0.8.19 entries, open queue, tracker and this plan with exact
  measured results. Preserve older releases/parked work. Verify generated assets,
  source ownership, whitespace and untracked helper inclusion in review packages.
- [x] Final independent integrated review and primary inspection; give user
  short manual frontend checklist. Leave verified work uncommitted for their check.

## Completion evidence

All four scoped spec/quality reviews and the final independent integration
review passed. Every selected entry/helper is <=10; no thresholds or waivers
changed. The strict backlog is **56 functions / 44 files / maximum 26**, down
from 87/69/32. Informational CCN 8 lists **147 functions / 106 files**, including
91 optional functions at 9/10. Source locations and counts are refreshed in the
[inventory](frontend_ccn_remaining_2026-09-10.md).

Full verification: **2,268 Vitest tests / 168 files**, **127 Chromium tests**,
normal ESLint plus both new E2E files, project noEmit, production build and all
11 size budgets pass. All 251 relative built imports resolve; main asset is
`index-BYpfjj1i.js`. The global strict command still exits 1 for the remaining
backlog; selected sources pass. Browser APIs are mocked, with real stores,
controls and charts. Screenshots of expanded settings at 1440/1100 px and fully
rendered regression/segmentation charts were visually inspected.

Test-authoring corrections involved the responsive export menu, the fixture's
default ROC class, visible sidebar selectors and waiting for actual SVG shapes
after Recharts animation. They required no production behavior changes. Initial
restricted-sandbox esbuild config reads failed before tests loaded; approved
environment runs supplied the reported results. Existing jsdom AggregateError
stderr was also present in original experiment tests; no suite tests failed.

CCN work and concise v0.8.19 notes are complete. The user reported frontend
checks working and authorized the batch commit after the reviewed handoff.
OC-228 records a separately reproduced existing casting issue; queue 62/4,
earlier parking and DRIFT-01 deferral are preserved. Task 5's experiment-browser
file was narrowly delegated to the graph agent after its source task; the
primary coordinated all browser runs and reviewed the resulting screenshots.

## Manual frontend checks

1. On a connected dataset, change settings in Invalid Value Replacement,
   Drop Columns, Binning, Casting, Alias Replacement, Missing Indicator,
   Polynomial Features, Deduplication, Drop Rows and Feature Interaction.
   Switch nodes and expand/collapse settings; confirm selected columns and
   values persist. Run Preview data and check the output/feedback.
2. In Experiments, select classification and regression runs. Switch the
   active run and train/test splits; move the classification threshold and
   check the confusion matrix/metrics. Check regression and per-class charts,
   and collapse/expand the run sidebar.
3. Open a completed Segmentation run, switch available splits and compare
   cluster sizes, centroids/profiles and the chart's data table.
4. On Canvas, check valid/invalid connections and leakage warnings, compare
   two saved runs in Pipeline Diff, then export PNG and SVG.

Known existing issue OC-228: when every available column already has a casting
rule, Add Casting Rule can replace the first rule with `float`. This behavior
was reproduced before the refactor and is queued separately. VariableCard is
currently unused by production routes; no manual navigation check is required.
