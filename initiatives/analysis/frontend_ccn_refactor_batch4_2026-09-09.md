# Frontend CCN refactor batch 4 implementation plan

> For agentic workers: use subagent-driven-development with independent ownership
> and review after each task. Continue in the user's active workspace.

**Goal:** simplify the next four frontend hotspots while preserving behavior.

**Architecture:** keep public node/page interfaces and API calls unchanged;
extract coherent form sections, presentation, and state resolution into adjacent
modules. Target function CCN <=8 where the resulting code remains readable.

**Tech stack:** React, TypeScript, Vitest, ESLint, Playwright, Vite.

**Spec:** user request to continue the accepted CCN work after committing batch 3;
the governing quality rule is readability and testable responsibilities, not
splitting a short coherent function merely to reach a metric. Prior workflow:
[`frontend_ccn_refactor_batch3_2026-09-09.md`](frontend_ccn_refactor_batch3_2026-09-09.md).

## Global constraints

- Baseline: `63745ca3`, branch `0819`; batch 3 and OC-218/219 committed.
- Preserve labels, settings defaults, numeric parsing, validation, request
  payloads, focus, selection reset, loading/error states and mutation lifetimes.
- No dependencies, backend changes, metric waivers or relaxed budgets.
- Write characterization tests first and run against the unchanged entry.
  A discovered defect requires separate evidence and tracker filing before repair.
- No agent edits shared package/workflow/docs/assets; primary owns integration.
- Rebuild frontend, run full tests/lint/strict CCN and browser tests; check sizes.
- Record concise completed changes under v0.8.19 and update tracker/queue as needed.
- Preserve parked OC-71/72/73/185 and unrelated temporary artifacts.

## Tasks and ownership

### Task 1: Resampling settings (Astra agent)

Files: `frontend/ml-canvas/src/modules/nodes/processing/ResamplingNode.tsx`,
new `ResamplingNode.test.tsx`, adjacent `resampling/` helpers.
Consumes/produces the same NodeDefinition/config and settings callback contract.

- [x] Map methods, dataset discovery, conditional parameters and last-run results.
- [x] Pin behavior in tests and run against baseline.
- [x] Extract coherent sections; verify tests, typing, lint and complexity.
- [x] Independent review and scoped fixes.

### Task 2: EDA page (Astra agent)

Files: `frontend/ml-canvas/src/pages/EDAPage.tsx`, `EDAPage.test.tsx`,
adjacent `edaPage/` helpers. Preserve public page and API/store contracts.

- [x] Map analysis lifecycle, job/dataset changes, navigation and error states.
- [x] Extend behavior coverage and run against baseline.
- [x] Extract coherent sections/state without changing request sequencing.
- [x] Independent review and scoped fixes.

### Task 3: Imputation settings (Astra agent)

Files: `frontend/ml-canvas/src/modules/nodes/processing/ImputationNode.tsx`,
new `ImputationNode.test.tsx`, adjacent `imputation/` helpers.
Preserve NodeDefinition/config and settings callback contracts.

- [x] Map method-specific options, dataset discovery, defaults and preview text.
- [x] Pin behavior in tests and run against baseline.
- [x] Extract cohesive controls/data resolution; verify focused checks.
- [x] Independent review and scoped fixes.

### Task 4: Node inspection (primary)

Files: `frontend/ml-canvas/src/components/layout/NodeInspectionPanel.tsx`,
`NodeInspectionPanel.test.tsx`, adjacent `nodeInspection/` helpers.
Preserve `({nodeId, side})` contract and existing `useNodeInspection` hook.

- [x] Pin branch/port/split selection, run resets, table matching and schema bounds.
- [x] Extract table presentation, measured comparison and selection resolution.
- [x] Verify focused regressions and independent review.

### Task 5: Integration and delivery (primary)

Files: `frontend/ml-canvas/package.json`, `.github/workflows/frontend-tests.yml`,
`changelog/0.8.x.md`, audit tracker/open queue, this record, built frontend assets.

- [x] Review each diff and gate clean scopes without metric-only helpers.
- [x] Run complete unit, lint, complexity, build, size and browser checks.
- [x] Record final evidence and a short frontend manual-check list.

## Execution record

- Preflight: tracker/queue inspected; 57 open and 4 parked after OC-219 closure.
- Ruling: use the existing non-main branch and user-visible workspace, as in
  batches 2/3. Reuse the three Astra 6 agent slots because the thread limit
  prevents creating fresh agents. Each new assignment has explicit file ownership.
- Ruling: independent tasks may run in parallel; shared configuration, assets,
  tracker and release notes remain primary-owned to avoid overlapping writes.
- Ruling: preserve a readable function slightly above 8 if splitting has no
  semantic benefit; document it and leave its entry report-only if necessary.
- Baseline inventory: 189 functions above 8, maximum 40 (batch 3 report).
- Node inspection: entry 38 -> 7, six cohesive modules <=7. Six new cases bring
  original/extracted related tests to 41; all five existing browser scenarios pass.
  Independent spec and quality review passes with no introduced issue.
- Imputation: entry 35 -> 7, four modules <=8. Seventeen new cases pass against
  original and extracted code; all 198 related tests pass. Independent primary
  review confirms byte-equivalent NodeDefinition, defaults, validation and preview.
- Resampling: entry 40 -> 7, two modules <=8. All 34 new cases pass original and
  extracted versions; all 117 related/accessibility tests and full tsc pass.
- EDA: page 30 -> 1, content renderer 40 -> 8; six modules <=8. All 19 cases pass
  original and extracted source. AST comparison finds all 25 API/store/query-key
  expressions unchanged. Independent spec and quality review passes.
- Full integration: 1,867 unit tests in 154 files pass (72 new cases), full lint
  and expanded strict CCN 8 gate pass. Global inventory is 183 warnings in 115
  files, maximum 34, down from 189 / maximum 40. Four entry files and 18 helper
  modules are gated.
- Production build and all 11 size budgets pass. Main: 318.2 KiB gzip / 325 KiB;
  EDA: 84.0 KiB / 140 KiB. Main asset `index-1un3ohfj.js` generated.
- All 99 browser tests pass, including the committed OC-219 lifecycle regression.
  All four task-level independent reviews pass; no runtime fixes were folded in.
- Final whole-batch Astra review passes with no actionable finding: public
  interfaces, component boundaries, all 18 helper imports and strict-gate scope
  agree with the recorded results. Per-task reports are summarized here; the
  ignored temporary review workspace is retained because automatic approval
  review declined its deletion without explicit disposal authorization.
- No audit finding was closed by this refactor: open queue remains 57 open / 4
  parked. Changelog v0.8.19 and tracker log updated. Batch 4 remains uncommitted
  for the user's manual frontend check; temporary artifacts are excluded.

## Manual frontend checks

- Resampling: switch Random/SMOTE/undersampling methods, edit target and neighbors,
  apply a recommendation, and confirm the saved settings survive reopening.
- Imputation: switch Simple/KNN/Iterative, choose columns, set Constant fill,
  and check execution feedback after Preview data.
- EDA: switch datasets and tabs; apply/reset filters and exclusions; load a saved
  report from History and verify the target and report content agree.
- Node inspection: run Preview data, switch Input/Output and path/port/split,
  then edit a node and rerun Preview to check stale labels and selection reset.

## Manual-feedback follow-up: OC-220

- The user reported native Resampling target suggestions opening away from the
  field. This markup predates batch 4; the native popup offset is user-reported,
  while the replacement's DOM geometry is measured in Chromium.
- Filed separately before repair. `TargetColumnField` now uses the existing
  Radix Popover to anchor a filtered editable listbox; target values, upstream
  discovery, dropped-column filtering and settings validation remain unchanged.
- Two new unit tests and two docked/expanded browser tests fail against the
  native list and pass with the replacement. The 119 related settings cases
  cover keyboard selection, Escape, free typing and distinct instance IDs;
  browser checks also cover resizing, Tab and outside-click focus.
- Independent review found no behavioral defect. Its test-fixture accessibility
  lint finding was corrected. Full 1,869 unit tests in 154 files, 101 Playwright
  tests, lint, strict CCN 8, TypeScript/Vite build and all 11 size budgets pass.
  Main bundle is now
  318.9 KiB gzip / 325 KiB; generated main asset is `index-DbgON88r.js`.
- OC-220 closed in the tracker; queue returns to 57 open / 4 parked. Concise
  v0.8.19 note added. Batch 4 and this feedback repair remain uncommitted.
- Manual check: refresh with Ctrl+Shift+R, open Resampling Target Column in
  docked and expanded settings, then select a suggestion or type a column name.

## Release placement and delivery

The user's follow-up requests a signed commit and places all branch `0819`
changes under v0.8.19. Notes added since the `0818` base (`784649d9`) now live
there, including earlier frontend CCN batches and OC-218/219/220. The v0.8.18
and older release contents are unchanged from that base. Related plan/tracker
references are corrected; verification and review artifacts remain outside
the commit.
