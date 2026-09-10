# Frontend CCN refactor batch 6 implementation plan

> For agentic workers: use subagent-driven-development with independent file
> ownership and task-level review. Continue in the user's active workspace.

**Goal:** simplify audit history, drift alert investigation and feature-generation
settings while preserving their user-visible behavior and public interfaces.

**Architecture:** retain entry components and node definitions. Move coherent
state, data preparation and presentation beside each owner; keep shared APIs,
hooks, stores and registration unchanged. Primary owns browser tests and delivery.

**Tech stack:** React, TypeScript, Vitest, ESLint, Playwright and Vite.

**Spec:** continuation explicitly requested by the user after manual validation
of batch 5. Current policy: strict CCN <=10 per function, informational report at
8. Simplify responsibilities without metric-only fragmentation. Concise release
notes belong to v0.8.19. Prior policy/inventory commit: `534d1e94` on `0819`.

## Global constraints

- Pure refactors: characterize original behavior before production extraction.
  Report independently suspected bugs to primary with reproduction evidence;
  do not silently change semantics within an extraction.
- Preserve labels, CSS, public exports/props, API arguments, ordering, defaults,
  numeric/empty fallbacks, memo/effect dependencies, request and local-state lifetime.
- Target <=10 in each selected entry and every helper. Do not force readable
  functions from 9/10 to 8; no raised limits, waivers or artificial wrappers.
- No backend, dependency, shared API/hook/store or registration edits. Agents
  do not commit, build assets, edit shared docs, or spawn subagents.
- Primary owns this plan, browser tests, tracker, queue, inventory, changelog
  and generated assets. Preserve temporary/review artifacts and parked findings.
- Queue baseline: 57 open / 4 parked (OC-71/72/73/185). CCN baseline: 112 functions
  above 10 in 80 files; informational report 172 above 8 in 110 files; max32.
- Existing non-main checkout preserves the user's manual frontend test workflow.
  Parallel owners have disjoint production/test scopes. The user has now checked
  this batch and requested a signed commit, including the separate OC-224 fix.
- Use npm.cmd/npx.cmd on Windows. Log scoped checks under tmp_repro_artifacts.
  Use tsc --project tsconfig.json --noEmit, never tsc -b or emitted config files.

## Task 1: Audit Log

**Owned files:** `frontend/ml-canvas/src/pages/AuditLogPage.tsx`, its existing
`AuditLogPage.test.tsx`, and new cohesive helpers in `src/pages/auditLog/`.
**Interface:** retain named no-props `AuditLogPage`; preserve `useUsableDatasets`,
`pipelineVersionsApi.audit` and operational feedback contracts as read-only inputs.

- [x] Extend public-page tests for initial/numeric dataset IDs, limit and
  server-side actor/kind/date filters, persistent facets, summaries, genesis rows,
  expansion, loading/error/empty states, retries and stale responses.
- [x] Run all owned tests on original source and record passing evidence.
- [x] Extract request/state management, history summaries/filters and row
  presentation where responsibilities differ. Keep request-generation behavior,
  API sentinel values, date strings, time formatting and row keys unchanged.
- [x] Rerun tests, scoped ESLint/CCN10 and noEmit type check; measure maximum
  before/after. Baseline maxima: page31, AuditRow12, load11.
- [x] Pass independent spec and quality review before integration closeout.

## Task 2: Drift alert detail

**Owned files:** `frontend/ml-canvas/src/pages/drift/DriftAlertModal.tsx`, new
`DriftAlertModal.test.tsx`, and new helpers in `src/pages/drift/alertDetail/`.
**Interface:** retain every existing modal prop and named export. Read but do not
edit monitoring API types, `useDriftAlertDetail`, badges, `ModalShell`, `RecordLink`,
`ChartDataTable` or parent `DataDriftPage`.

- [x] Characterize public modal closed/loading/error/detail precedence, metadata
  fallbacks, evidence rows/metric rounding and order, threshold/version/context
  links, disposition actions by status, actor/note trimming and validation,
  pending/falsey/truthy results, history rows, close/retry and local state lifetime.
- [x] Run tests against the original modal and record baseline evidence.
- [x] Extract evidence preparation, detail/identity/history sections and
  disposition controls as coherent units. Preserve state ownership and mounting,
  promise/result semantics, field labels, modal accessibility and styling.
- [x] Rerun tests, scoped ESLint/CCN10 and noEmit check. Baseline maximum31;
  entry/helpers must all be <=10, with no needless 9/10 decomposition.
- [x] Pass independent spec and quality review before integration closeout.

## Task 3: Feature Generation settings

**Owned files:** `frontend/ml-canvas/src/modules/nodes/processing/FeatureGenerationNode.tsx`,
its existing `FeatureGenerationNode.test.tsx`, and helpers under adjacent
`featureGeneration/`. Existing data/recommendation/schema/validation hooks,
`ColumnMultiSelect`, graph store, converters and registry are read-only.
**Interface:** preserve exported `FeatureGenerationNode: NodeDefinition`,
`settingsComponent`, config/operation fields, defaults, validate/bodyPreview,
execution feedback, handle definitions and payloads.

- [x] Characterize each operation type, defaults, method/column/constant/date
  selections, output naming, add/update/delete, expand/reveal/index changes,
  upstream dropped columns, informational recommendations, execution feedback,
  validation messages, body preview and wide/narrow layouts through public seams.
- [x] Run tests on the original node before production extraction.
- [x] Extract operation editors, data/reveal handling and execution/validation
  presentation where responsibilities differ. Preserve controlled updates,
  isExpanded/revealedOperations semantics, validation field names and callbacks.
- [x] Rerun owned and relevant validation/converter tests, scoped ESLint/CCN10
  and noEmit types. Baseline settings21, operation rendering30, validate14.
- [x] Pass independent spec and quality review before integration closeout.

## Task 4: Primary integration and delivery

- [x] Review all task diffs and original/refactored test evidence independently.
- [x] Verify actual-page browser paths for audit filters/expansion, drift
  investigation/disposition and feature generation settings, including focus
  and compact/wide layout where relevant.
- [x] Run full Vitest, normal ESLint, TypeScript/Vite build, bundle budgets,
  relevant/full Chromium, and both global CCN commands. Selected scopes must
  pass <=10; the global gate may still fail on the documented remaining backlog.
- [x] Refresh the remaining-work inventory, tracker/open queue and concise
  v0.8.19 notes. Keep closed historical counts/thresholds intact.
- [x] Run final independent review and whitespace/scope checks. Give the user
  bounded frontend checks; the user subsequently authorized the batch commit.

## Task 5: OC-222 wording correction after Task 1 review

**Owned files:** AuditLogPage.tsx and its existing public-page test.

- [x] After the pure extraction passes review, update the existing conflicting
  copy assertion and run it red. Correct the stale hint to say actor, action
  kind and time filters apply across full history before the page limit.
- [x] Keep paragraph/CSS, API behavior and remaining tests unchanged. Verify
  the suite green and scoped lint; independent review confirms the wording-only
  diff before primary closes OC-222 and runs final integration gates.

OC-223 is separately filed open for a modal/parent request-lifetime repair.
It predates this extraction; batch 6 does not change drift request semantics.

## Verification results

- Independent specification and quality reviews pass for all three extractions
  and the separate OC-222 copy fix (red: 1 failed / 22 passed; green: 23 passed).
- Vitest: 157 files / 1,998 tests pass. Chromium: 111 tests pass, including six
  new actual-page cases with mocked API routes. No live-backend integration claim.
- Normal ESLint, TypeScript/Vite production build and all 11 size budgets pass.
  Main bundle: 320.1 KiB gzip / 325 KiB. Generated assets are rebuilt.
- Selected entries/helpers pass CCN 10: Audit max31->10, Drift max31->8,
  Feature Generation max30->7. Cohesive functions at 9/10 need no forced split.
- Global strict check: 105 functions / 77 files above 10 (expected exit 1),
  down from 112 / 80. Informational report at 8: 167 / 109 (exit 0), down from
  172 / 110. Remaining maximum32. Inventory refreshed from both measurements.
- Queue: 58 open / 4 parked after closing OC-222 and filing OC-223. Prior
  policy commit534d1e94 is signed; the user approved this batch for commit after
  frontend checks and the live OC-224 drift verification.
- Final independent integration review passes with no actionable findings.
  Exact inventory locations, queue counts, asset references, change scope and
  whitespace are checked; v0.8.18 and older notes still match the branch base.

## Manual frontend checks

1. Audit Log: select a dataset with saved versions; change Actor, Action kind,
   From/To and page limit, then Refresh. Expand a version and inspect its changes.
2. Drift: open an alert's details; inspect evidence and contextual links, then
   enter your name/note and try Acknowledge, Resolve and Reopen. Close/reopen it.
3. Canvas Feature Generation: add/edit/remove arithmetic and date operations,
   select input columns, set an output name and date features. Collapse/reopen
   operations and expand the settings panel; verify values persist. Run Preview
   data to check the generated columns with your dataset.

For OC-223, switching alerts while a disposition is pending can still clear a
new note when the older response arrives; that existing issue remains queued.
