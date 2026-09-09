# Frontend complexity refactor implementation plan

> For agentic workers: use subagent-driven-development and independent review.

**Goal:** Reduce the five reported frontend hotspots and their extracted helpers
to ESLint CCN 8 while preserving existing behavior and API payloads.

**Architecture:** Separate graph traversal from node-specific parameter mapping.
Split React orchestration, derived presentation state and existing screen sections
into named functions, hooks and components with explicit responsibilities.

**Tech stack:** React, TypeScript, ESLint, Vitest and Playwright; existing dependencies.

**Spec:** The user's approved conversation proposal: preserve behavior with tests,
start with pipeline conversion and the four reported React hotspots, then enforce
the cleaned scope in the frontend CI complexity gate.

## Constraints

- Preserve exported interfaces, payload defaults, edge ordering, React state,
  side effects, accessible controls, styling and user-visible text.
- No complexity waivers, threshold increases, unsafe casts or generic frameworks
  introduced to bypass the measurement.
- Keep extracted helpers within each task's owned files/folder; coordinate shared
  package/workflow/changelog/tracker edits through the primary agent.
- Baseline: clean tracked worktree; preserve existing `tmp_repro_artifacts/`.
- Run existing characterization tests before and after refactoring. Add focused
  public-behavior coverage where extraction exposes a material test gap.
- Update v0.8.18 and the tracker after each verified task. Do not commit or push.

## Tasks and file map

- [x] **Pipeline conversion (primary):** `src/core/utils/pipelineConverter.ts`,
  adjacent converter tests/snapshots, and a `pipelineConversion/` helper folder.
  Preserve `convertGraphToPipelineConfig(nodes, edges)`, dataset traversal,
  merge/input handles, ensemble wiring, fixed/tuned dispatch and every parameter
  mapping. Verify exact converted payloads using existing tests and snapshots.
- [x] **Job details (agent):** `src/components/panels/jobs/JobDetailsView.tsx`,
  its tests and focused `jobDetails/` helpers. Preserve loading/error/empty states,
  metric and tuning rendering, tab contents, learning curves and pipeline display.
- [x] **Toolbar (agent):** `src/components/layout/Toolbar.tsx`, its tests and
  `toolbar/` helpers. Preserve keyboard/menu behavior, read-only restrictions,
  preview/train/cancel controls, save/load/export and disabled states.
- [x] **Inference (agent):** `src/components/pages/InferencePage.tsx`, its tests
  and `inference/` helpers. Preserve model/feature loading, dataset/manual
  prediction submission, errors, run history and result export. The existing
  page has no pagination; no new behavior is introduced.
- [x] **Canvas node wrapper (primary characterization, agent implementation):**
  `src/components/canvas/CustomNodeWrapper.tsx`, its tests and `nodeWrapper/`
  helpers. Preserve port geometry, deletion/read-only controls, validation pulse,
  leakage/schema indicators, performance telemetry and execution summaries.
- [x] **Review and gate (primary/reviewers):** inspect each task's diff against
  HEAD, address behavior or design regressions, then expand `complexity:check`
  in `frontend/ml-canvas/package.json` to all five files and helper folders.
  Update `.github/workflows/frontend-tests.yml` scope comments and add concise
  results to `changelog/0.8.x.md` and the audit tracker.
- [x] **Integration adjustment:** the first build exceeded the main gzip budget
  by 1.1 KiB. `MainLayout.tsx` now loads inference on first visit using the
  existing Suspense/ErrorBoundary pattern and retains its mounted view.
  `e2e/inference-loading.spec.ts` reproduced eager loading before the change
  and verifies deferred loading plus unsaved form state afterward. Add a
  separate 20 KiB inference budget in `scripts/check-bundle-size.mjs`.
- [x] **Final verification:** full Vitest, ESLint, complexity gate/report,
  production build, bundle-size budgets and relevant Playwright scenarios.
  Record exact results and remaining report-only frontend complexity.

## Verification commands

Run from `frontend/ml-canvas/`; Windows uses `npm.cmd`/`npx.cmd`.

```powershell
npm.cmd test -- pipelineConverter JobDetailsView Toolbar InferencePage CustomNodeWrapper --maxWorkers=4 --reporter=dot --silent
npm.cmd run complexity:check
npm.cmd test -- --maxWorkers=4 --reporter=dot --silent
npm.cmd run lint
npm.cmd run complexity:check
npm.cmd run complexity:report
npm.cmd run build
npm.cmd run size-check
npm.cmd run test:e2e -- --workers=2
```

Individual task verification precedes integration; full checks run after all
implementers finish so shared source changes do not invalidate the result.

Final results: **1,669 unit tests**, **98 browser tests**, lint, strict CCN 8,
TypeScript/build and all size budgets passed. The global complexity report has
**212 remaining warnings**, down from 230. All five entry files and their
extracted production helpers are within CCN 8 and covered by CI. Independent
reviews found no material regressions. Release notes and the tracker are updated;
no commits or pushes were made.
