# Frontend CCN refactor batch 8 implementation plan

> For agentic workers: use subagent-driven-development with disjoint ownership,
> original-behavior characterization and independent task reviews.

**Goal:** simplify graph validation/connection rules, operational record links,
and scaling/outlier settings while preserving their public behavior.

**Architecture:** retain store entry points, context exports and node definitions.
Move coherent graph policies, record codecs and node presentation beside their
owners. Shared contracts and dependency direction remain unchanged.

**Tech stack:** React, TypeScript, Zustand/Zundo, Vitest, ESLint and Playwright.

**Spec:** the user requested committing batch 7 and continuing the CCN backlog.
Base is signed commit `6af9a600` on `0819`. Strict limit10; informational limit8.
Prioritize meaningful simplification, with behavior changes separately evidenced.

## Global constraints

- Add public-behavior characterization and run it on original production before
  extracting code. Preserve exports/types, props, exact messages, query strings,
  ordering, numeric/empty fallbacks, callback/effect lifetime and DOM/CSS.
- Every selected entry/helper must pass CCN <=10. Retain readable9/10 functions;
  no artificial wrappers, raised limits, new complexity waivers or dependencies.
- Agents edit only owned files. No commits, shared docs/assets or subagents.
  Primary owns integration/browser tests, inventory, queue, tracker and changelog.
- Three owners may work concurrently in the existing non-main workspace per the
  user's explicit Astra 6 authorization. GraphStore is Task1-owned; other owners
  consume its existing interface without changing it or shared selectors/hooks.
- No backend/Core/shared API/registry/converter/dependency edits. Existing bugs
  must be reported with evidence, not silently fixed during the extraction.
- Preserve OC-71/72/73/185 parking, OC-223/225/226 status and deferred DRIFT-01.
  Queue baseline60 open/4 parked. CCN baseline98 functions/73 files above10;
  informational166/111 above8; max32. Release notes belong under v0.8.19.
- Use npm.cmd/npx.cmd on Windows; checks may require approved expanded access
  for esbuild. Log under tmp_repro_artifacts. Type-check using
  `npx.cmd tsc --project tsconfig.json --noEmit`, without emitted build configs.
- Initial delivery stays uncommitted for frontend checks. The user subsequently
  confirmed those checks and authorized committing this batch.

## Task 1: Graph validation, connection policies and undo equality

**Owned files:** `frontend/ml-canvas/src/core/store/useGraphStore.ts`, existing
`useGraphStore.test.ts`, and new cohesive helpers in adjacent `graphStore/`.
**Interfaces:** retain GraphState, collectGraphValidationIssues, useGraphStore,
useTemporalStore and every store action. Registry, connectionIssue, converter,
leakage/cycle validation, React Flow, toast and Zundo are read-only dependencies.

- [x] Characterize public collectGraphValidationIssues and store actions on
  original production. Pin preview-node exclusion, issue order/labels/fields,
  unknown definitions, missing input/output, leakage and cycles.
- [x] Pin synchronous onConnect/add-next-node acceptance and cancellation:
  invalid connection stops first; disjoint ensemble dataset warning; missing
  TrainTestSplitter warning; distinct-source counting and duplicate handles;
  training/processing fan-in prompts; ensemble/preview exceptions. Preserve
  prompt text/order and ensure canceled operations do not mutate graph/history.
- [x] Cover undo/redo equality for edge identity, reordered/replaced nodes,
  data/type/id/length, selection-only updates, dragging positions and drag-end,
  execution-result exclusion and history limit. Do not improve its semantics
  without a separately established defect and primary ruling.
- [x] Extract graph validation, connection policy and history equality by
  responsibility. Leave unrelated reducers and temporal/store construction
  stable. Avoid runtime circular imports; type-only contract imports are okay.
- [x] Run owned plus relevant graph/guided-connections/leakage/converter tests,
  scoped ESLint/CCN10 and noEmit. Baseline collect17, confirmConnection32,
  equality12; measure entry and every helper afterward.
- [x] Write task report; pass independent spec and code-quality review.

## Task 2: Operational context codecs and record descriptions

**Owned files:** `frontend/ml-canvas/src/core/utils/operationalContext.ts`, existing
`operationalContext.test.ts`, helpers in adjacent `operationalContext/`.
**Interfaces:** preserve all exported constants/unions/types and functions:
serializeOperationalContext, parseOperationalContext, buildRecordHref and
describeOperationalRef. RecordLink, useOperationalContext and all pages are
read-only consumers. Do not change accepted URL syntax or validation rules.

- [x] Extend public tests before extraction: every record kind, required and
  optional IDs, numeric vs opaque IDs, exact serialized key/insertion order,
  URL-hostile text, absent vs blank fields, prefix handling and forward-compatible
  unrelated params. Use exact expectations as well as round trips.
- [x] Pin whitespace trimming, Number/Number.isInteger semantics, duplicate
  query keys, optional invalid values, unknown kind/time range, filter values
  and ordering, no input mutation, owning routes and accessible descriptions.
- [x] Replace sprawling switches with coherent typed per-record parsing,
  serialization and identity/description responsibilities. Avoid loosely typed
  descriptor machinery, unchecked broad casts or a generic schema framework.
- [x] Run owned plus RecordLink/OperationalContext consumer tests, scoped
  ESLint/CCN10 and noEmit. Baseline serialize19, parseRef28, describe11; keep
  public signatures and output strings exact and measure all helper CCNs.
- [x] Write task report; pass independent spec and code-quality review.

## Task 3: Scaling and Outlier settings

**Owned files:** `frontend/ml-canvas/src/modules/nodes/processing/ScalingNode.tsx`,
`OutlierNode.tsx`, new adjacent `ScalingNode.test.tsx` and `OutlierNode.test.tsx`,
helpers in adjacent `scaling/` and `outlier/`. Shared hooks, ColumnMultiSelect,
RecommendationsPanel, metrics helpers, store, converter and registry are read-only.
**Interfaces:** preserve both exported NodeDefinitions, settingsComponent,
config/defaults/validation/bodyPreview/handles and controlled update payloads.

- [x] Add public NodeDefinition/settings tests and run on original production.
  Cover numeric schema filtering, upstream dropped columns, missing/loading data,
  narrow/wide settings, selected columns, method changes and hidden-option state.
- [x] Scaling: standard/minmax/maxabs/robust controls, checkbox values, numeric
  zero/empty fallbacks, bounds/quantiles, validation, and per-column execution
  feedback including missing/zero/partial metrics and existing precision/text.
- [x] Outlier: iqr/zscore/winsorize/elliptic_envelope controls and defaults,
  upstream resolution, validation, feedback method branches/formatting, runtime
  recommendations (>20%, exactly20%, zero loss, winsorize, elliptic warning),
  backend recommendation ordering/application and metrics fallbacks.
- [x] Extract settings sections, data selection, feedback/recommendation
  derivation and validation as cohesive units. Preserve local state/mounting,
  callbacks, IDs/labels, styling and exact serialized configuration; no shared
  graph/API/store edits or new generalized preprocessing framework.
- [x] Run both public suites and relevant validation/converter/metrics tests,
  scoped ESLint/CCN10 and noEmit. Baseline Scaling max27 (feedback callback19);
  Outlier max26 (feedback22, recommendations15). Measure all resulting helpers.
- [x] Write task report; pass independent spec and code-quality review.

## Task 4: Primary integration and delivery

- [x] Review complete scoped diffs including new helpers and original/final
  evidence; independently review each task and then combined integration.
- [x] Exercise browser graph wiring/confirmation cancellation, undo/redo,
  operational record links with preserved origin/filter scope, and compact/
  expanded Scaling/Outlier settings. Use real UI/stores with explicit HTTP mocks.
- [x] Run full Vitest, normal ESLint, noEmit/Vite build, bundle budgets, complete
  Chromium and both CCN commands. Remaining global strict violations are expected;
  selected scopes must pass <=10. Record any intermittent diagnostics honestly.
- [x] Refresh the measured inventory, open queue, tracker and concise v0.8.19
  notes. Rebuild generated assets; preserve older release notes and parked work.
- [x] Finish final review and scope/whitespace checks; report prior signed commit
  and bounded frontend checks. User review completed; commit authorized.

## Verification and delivery evidence

- Original characterization: graph63, context81, settings50 tests. Independent
  reviews passed; history/data identity and dataset traversal tests were improved
  after review, then passed their59/52-test follow-up suites.
- Full Vitest:2181 tests/160files passed. Normal ESLint, project noEmit,
  production build and11 bundle budgets passed. Main asset:index-51eF5Jsl.js.
- Complete Chromium confirmation:122/122 passed. Earlier121/1 included a blank
  startup timeout before Add Dataset in unchanged validation-navigation. That
  suite passed15/15 repeated cases; no cause/fix is claimed for the initial event.
- Final five new browser cases passed again after adding transition completion
  and single/two-column layout assertions. They cover graph confirmation,
  cancellation, undo/redo, URL filter restoration and submitted Preview fields.
- Strict10:87 functions/69files/max32 (expected exit1), down from98/73.
  Informational8:159/113, including72optional9/10 functions. Policy unchanged.
- Inventory, open queue and v0.8.19 notes refreshed; older release text preserved.
  OC227 records original drag-end history behavior. Queue61open/4parked;
  existing parked items, OC223/225/226 and DRIFT01 keep their prior status.
- Scope audit:30ownedtextfiles; UTF8/EOF/whitespace and251 built import paths
  plus index references checked. Temporary diagnostics remain outside delivery.

For manual frontend checks: connect a second branch and cancel/accept its prompt,
then undo/redo; change Scaling/Outlier methods and columns in docked/expanded
settings and run Preview; open a filtered Jobs record, reload, then return to the
list. A drag-only movement has the separate existing OC227 undo limitation.
Batch7 is committed as6af9a600; the user confirmed frontend checks and requested
committing this batch8.
