# Frontend complexity refactor batch 2 implementation plan

> For agentic workers: use subagent-driven-development with independent review.

**Goal:** Reduce the next five frontend hotspots and their new helpers to CCN 8
while preserving current configuration payloads, rendering and interaction.

**Architecture:** Extract cohesive settings sections and derived state helpers
beside their callers. Separate branch graph analysis from edge presentation;
preserve ordering and existing public component/hook interfaces.

**Tech stack:** React, TypeScript, React Flow, ESLint, Vitest and Playwright.

**Spec:** The user's request to continue frontend complexity cleanup, with
multiple Astra 6 subagents for detailed independent work. Follow batch 1's
behavior-preservation, verification and documentation requirements.

## Constraints and baseline

- Base commit: `4bd55065` on `0819`; tracked tree clean at start. Preserve
  existing `tmp_repro_artifacts/`. Commit after the user's acceptance.
- Current report: 212 functions above CCN 8 across 128 of 521 TS/TSX files.
- Preserve hook lifetimes, callbacks, public exports, labels, defaults,
  accessibility, validation focus, read-only behavior and serialized values.
- No lint waivers, raised thresholds, new dependencies or generic frameworks.
- Up to three Astra 6 agents work concurrently in disjoint scopes; primary owns
  branch colors and shared CI, docs, build output and integration.
- Existing tests run before extraction. Add characterization tests only for
  material coverage gaps, and verify them against the original behavior.
- Existing helper modules outside a task's ownership stay unchanged. Newly
  extracted production helper folders are included in the strict CCN 8 gate.
- Confirmed behavior defects require reproduction and a separate tracker entry;
  complexity cleanup alone closes no audit finding.

## Tasks and file map

- [x] **Training settings (Astra 6):** modify
  `frontend/ml-canvas/src/modules/nodes/modeling/TrainingSettings.tsx` (max 72)
  and `TrainingSettings.test.tsx`; create focused helpers in adjacent
  `trainingSettings/`. Preserve basic/tuning switching, model defaults,
  search strategies, CV/seed/threshold configuration, run and download actions,
  validation reveal, responsive controls and request guards. Existing exports
  `TrainingConfig`, `TrainingRunMode` and the settings component stay compatible.
- [x] **Results panel (Astra 6):** modify
  `frontend/ml-canvas/src/components/layout/ResultsPanel.tsx` (max 58) and its
  tests; create helpers in `resultsPanel/presentation/`. Preserve branch/split
  tab order, selected-pane lifetime, error and validation navigation, dismissal
  and reopening, panel resizing, confirmation and read-only restrictions.
  Keep the component's `maxHeight` prop and existing subcomponent contracts.
- [x] **Encoding settings (Astra 6):** modify
  `frontend/ml-canvas/src/modules/nodes/processing/EncodingNode.tsx` (max 58)
  and its adjacent tests; create helpers in `encoding/`. Preserve node
  definition/registration, all seven method settings, number parsing and zero
  values, schema filtering, result metrics, recommendations and accessibility.
- [x] **Branch colors (primary):** modify
  `frontend/ml-canvas/src/core/hooks/useBranchColors.ts` (max 70) and its tests;
  create graph helpers in `branchColors/`. Preserve exported color generation
  and `BranchEdgeInfo`, BFS terminal ordering, cycles, merge/parallel grouping,
  feature-target y passthroughs, ensemble providers, fallback/duplicate labels,
  representative terminal labels and first-branch coloring of shared edges.
- [x] **Canvas edges (Astra 6, after encoding handoff):** modify
  `frontend/ml-canvas/src/components/canvas/CustomEdge.tsx` (max 53) and its
  tests; create focused helpers in `edge/`. Preserve curve routing, handles,
  labels, selection/deletion, read-only state, leakage appearance, execution
  progress, theme behavior and React Flow memoization boundaries.
- [x] **Independent reviews:** review each implementation against the base
  commit for spec compliance and behavior regressions. Fix material findings,
  then obtain a final review covering interactions across the five scopes.
- [x] **CI and documentation:** extend `complexity:check` in
  `frontend/ml-canvas/package.json` to all five entries and new helper folders;
  update `.github/workflows/frontend-tests.yml` scope comments. Record each
  verified task in `opus_core_analysis-tracker.md` and concise v0.8.19 notes in
  `changelog/0.8.x.md`. Update the open queue only for actual defect closures.
- [x] **Integration verification:** full frontend tests, lint, complexity
  check/report, build, bundle-size check, and Playwright. Rebuild
  `static/ml_canvas/`; inspect scope and whitespace, preserve generated code.

## Verification commands

From `frontend/ml-canvas/`, use `npm.cmd` on Windows. Subprocess checks may need
sandbox escalation, as confirmed during batch 1. Each implementer runs only
their focused tests and scoped ESLint; primary runs shared build/browser gates.

```powershell
npm.cmd test -- TrainingSettings ResultsPanel EncodingNode useBranchColors CustomEdge --maxWorkers=4 --reporter=dot --silent
npm.cmd run lint
npm.cmd run complexity:check
npm.cmd run complexity:report
npm.cmd test -- --maxWorkers=4 --reporter=dot --silent
npm.cmd run build
npm.cmd run size-check
npm.cmd run test:e2e -- --workers=2
```

## Execution record

- Planning: user authorized parallel Astra 6 work; proceed in this shared
  workspace with disjoint file ownership and primary-controlled integration.
- Branch colors: original 10 tests plus four new characterization cases passed
  before and after extraction (14 total); strict CCN 8 passed. A temporary
  differential probe compared ordered maps for 2,000 varied graphs against
  `4bd55065` with no differences. Both temporary reference/probe files removed.
- Encoding: original test plus 17 new characterizations passed before and after
  extraction (18 total); entry CCN 58 -> 6, extracted helpers at most 6.
- Encoding review: plain-object method dispatch introduced inherited-key
  rendering failures. Three cases failed before own-property guards; the final
  18-case suite passes against both original and fixed code. Existing WOE input
  parsing semantics remain unchanged.
- Training settings: entry CCN 72 -> 2, all extracted helpers at most 8; six
  added characterization cases pass before/after extraction (19 tests total).
- Results panel: entry CCN 58 -> 8, all new presentation helpers at most 8;
  focused 21-test suite passes against original and extracted implementations.
- Independent training/results review: no actionable regressions; 40 focused
  tests and strict CCN 8 passed. UI literals and extracted control attributes
  were compared against the original; effect order/dependencies remain intact.
- Independent graph/encoding review: no further regressions; 32 focused tests
  and strict CCN 8 passed. Original ordering, branch membership, method fallback,
  shared controls and public exports were checked directly.
- Canvas edges: entry CCN 53 -> 2; six new helper files have maxima between 4
  and 6. Original nine tests plus nine characterization cases pass before/after
  extraction (18 total), covering routing thresholds, split geometry, branch
  styling, hover cleanup and measured warning positions/fallbacks.
- Final integration review: no actionable regressions; 90 focused tests across
  eight files and the expanded strict CCN 8 gate passed independently. Reviewed
  metadata handoff, public contracts, geometry, state lifetimes and CI scope.
  The review's type-only observation was addressed by allowing nullable branch
  labels, matching the existing FlowCanvas values and unchanged render guards.
- Full verification: 1,712 unit tests across 149 files and 98 browser tests
  passed. Full ESLint and the expanded CCN 8 gate passed. TypeScript/build and
  all existing size budgets passed; main bundle 316.3 KiB gzip versus 325 KiB.
- Final inventory: 204 functions above CCN 8 in 123 of 555 files; highest CCN
  now 52. All five entry files and 32 new production helper modules are within
  CCN 8. Remaining leaders: EnsembleSettings (52), VariableRow (51), EDASidebar
  (47), FeatureSelectionNode (46), and experiment EvaluationView (43).
- Housekeeping: removed four temporary outputs from an agent's `tsc -b` run
  (`tsconfig*.tsbuildinfo`, `vite.config.js`, `vite.config.d.ts`), and normalized
  final newlines in five helpers. Production build repeated after cleanup.
- Final state: all tasks complete; the user confirmed the frontend works after
  manual testing and requested a commit. The open queue remains 57 open and
  4 parked; release notes are under v0.8.19 as requested.
- Scheduling: encoding completed first, freeing an Astra 6 slot for canvas
  edges while primary performs branch verification and independent review.
