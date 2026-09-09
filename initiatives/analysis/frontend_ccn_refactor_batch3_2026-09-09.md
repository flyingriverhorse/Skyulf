# Frontend complexity refactor batch 3 implementation plan

> For agentic workers: use subagent-driven-development with independent review.

**Goal:** Simplify the next five frontend hotspots through cohesive boundaries
while preserving configuration, display, requests and interaction behavior.

**Architecture:** Keep related helpers beside their existing callers and retain
public exports. Separate settings, data derivation and presentation where those
responsibilities already exist. Use CCN 8 as a target; do not add artificial
indirection solely to satisfy a number.

**Tech stack:** React, TypeScript, React Flow, ESLint, Vitest and Playwright.

**Spec:** The user's continuation request after accepting and committing batch 2,
including the discussion that readability and behavioral coverage matter more
than a mechanical CCN threshold. Multiple Astra 6 agents are authorized.

## Constraints and baseline

- Base commit: `abe5ea7d` on `0819`; tracked tree clean at start. Preserve
  existing `tmp_repro_artifacts/`. This is a new implementation batch; commit
  after user acceptance and keep temporary artifacts out of delivery.
- Baseline report: 204 functions above CCN 8 across 123 of 555 TS/TSX files.
- Preserve effects, hook lifetimes, DOM identity where it affects state/focus,
  callback payloads, defaults, numeric zero/empty handling, accessibility,
  errors, validation reveal and read-only restrictions.
- No new dependencies, raised thresholds or blanket lint waivers. Extend the
  strict gate only to scopes that are clean without artificial fragmentation.
- Primary owns Feature Selection, shared CI/docs, build and integration.
  Three Astra 6 agents own separate Ensemble, EDA and Evaluation scopes.
- Run existing tests before refactoring. Add characterization tests for
  meaningful coverage gaps and run them against original and extracted code.
- Confirmed unrelated defects must be reproduced and checked against the
  tracker before recording them. Maintenance alone closes no audit finding.

## Tasks and file map

- [x] **Ensemble settings:** modify
  `frontend/ml-canvas/src/modules/nodes/modeling/EnsembleSettings.tsx` and its
  existing test; add cohesive modules in adjacent `ensembleSettings/` as
  needed. Preserve task/strategy changes, connected base models, model registry
  filtering, weights, calibration, CV, advanced tuning, defaults and actions.
- [x] **EDA variables and sidebar:** modify
  `frontend/ml-canvas/src/components/eda/VariableRow.tsx`, `EDASidebar.tsx` and
  adjacent tests; use `variableRow/` and `edaSidebar/` for cohesive helpers if
  needed. Preserve expansion, charts, dtype/stat display, export, section and
  filter state, value parsing, signatures, apply gating and responsive layout.
- [x] **Feature selection:** modify
  `frontend/ml-canvas/src/modules/nodes/processing/FeatureSelectionNode.tsx`;
  add `FeatureSelectionNode.test.tsx` and adjacent `featureSelection/` helpers.
  Preserve all methods, inherited targets, dropped-column filtering, optional
  settings, validation reveal, result metrics, node exports/defaults and preview.
- [x] **Evaluation view:** modify
  `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/EvaluationView.tsx`
  and its test; use adjacent `evaluationView/` modules where useful. Preserve
  split/chart selection, threshold slider/tuning controls, mutations and error
  states, busy guards, stale request handling and saved/enabled semantics.
- [x] **Independent reviews:** compare each task with `abe5ea7d` for behavioral
  equivalence and usefulness of boundaries. Fix material findings, then review
  integration and public interfaces across the complete batch.
- [x] **OC-218:** separately repair the reproduced connected-model CV seed-zero
  fallback, with a failing regression before repair and real fixed/tuned
  conversion coverage. Record the intentional behavior change and close the
  finding after verification.
- [x] **CI and documentation:** extend `complexity:check` and workflow scope
  comments for clean entries/helpers. Update this record, tracker and concise
  v0.8.18 release notes. Change the open queue only for actual finding closures.
- [x] **Integration:** run full Vitest, ESLint, complexity check/report, build,
  size check and Playwright; inspect source and generated asset scope.

## Verification

Use `npm.cmd`/`npx.cmd` in `frontend/ml-canvas` on Windows. Node child processes
may require sandbox escalation. Implementers run only their focused tests and
scoped lint; primary owns shared build/browser runs. Do not run `tsc -b` because
it leaves compiler outputs in the source workspace; the final build runs `tsc`.

```powershell
npm.cmd test -- EnsembleSettings EDASidebar VariableRow FeatureSelectionNode EvaluationView --maxWorkers=2 --reporter=dot --silent
npm.cmd run lint
npm.cmd run complexity:check
npm.cmd run complexity:report
npm.cmd test -- --maxWorkers=4 --reporter=dot --silent
npm.cmd run build
npm.cmd run size-check
npm.cmd run test:e2e -- --workers=2
```

## Execution record

- Planning: tracker and open queue inspected first; parked decisions remain
  parked. Current maintenance scopes are independent of audit bug closures.
- Ruling: continue in the user's existing `0819` workspace with disjoint file
  ownership, following the accepted batch 2 workflow. New worktree isolation
  would separate the frontend the user is running from the reviewed files.
- Ruling: characterize existing behavior before extraction; tests are expected
  to stay green for a refactor. A real behavior fix requires a failing regression.
- Ruling: keep the cohesive Feature Selection body preview at CCN 9 after
  merging equivalent K-count cases. Gate its clean helper folder; keep this
  entry report-only instead of adding a metric-driven helper or lint waiver.
- Ruling: use the completed Astra 6 reviewer slot for Evaluation implementation
  because the thread limit prevents a fresh agent; other owners review its work.
- Refactor integration: 1,780 unit tests and 98 browser tests pass. Full ESLint,
  expanded CCN gate and report pass (189 warnings). Build found exact-optional
  and test typing issues in Ensemble and Feature Selection; subsequently repaired
  and confirmed by the final build.
- Review found OC-218 in pre-existing synchronization, independently traced to
  fixed and tuned payloads before filing. A separate regression-led repair is
  included because it silently changes an explicitly configured seed.

- Ensemble: five cohesive helpers; entry-file maximum 52 -> 6, helpers <=8.
  31 original/extracted characterization tests passed; independent 2,000-case
  differential matched the original (implementer also matched 4,000 cases).
- EDA: VariableRow 51 -> 7, EDASidebar 47 -> 4; seven helpers <=7. The same
  34 tests pass against original and refactored entries; 84 independent DOM
  comparisons matched with shared chart/UI stubs.
- Feature Selection: settings 46 -> 7, five helpers <=8, short bodyPreview9
  retained. All 48 original/extracted tests pass; independent related suite131
  passes. No target, numeric parsing or feedback semantics changed.
- Evaluation: entry43 -> 1, nine helpers <=8; original/extracted21 tests pass.
  Independent review confirmed state/retry/chart lifetimes and controls.
- OC-218: red3 failed/36 passed; green39 passed, including eight component-to-
  converter cases. Nullish seed fallback preserves zero. Independent review
  passed; closed tracker row/evidence and v0.8.18 note added, queue restored to
  57 open/4 parked. 32 comparisons confirmed equivalent optional-time patches.
- Final review: no actionable introduced issues; five public interfaces
  unchanged and all99 helper imports resolve. Different owners reviewed scopes.
- Final verification: 1,788 unit tests (152 files), 98 browser tests, full lint,
  strict gate and production build pass. All11 size budgets pass; main317.3KiB
  gzip/325KiB. Existing diagnostics/vendor/proxy warnings remain.
- Inventory: 189 functions above8 in119/584 files (baseline204 in123/555),
  maximum40. Strict CI covers26 new helper modules and four new entry files;
  FeatureSelection entry remains report-only because of its readable summary.
- Delivery scope: source, generated frontend, changelog, tracker and queue;
  preserve temporary artifacts outside the delivery scope.

## Manual-test follow-up: OC-219

The user's successful Preview followed by toggle HTTP 400 exposed a pre-existing
missing saved-state distinction. Preview does not persist; Save persists and
enables. Added per-job saved-state tracking, disabled the checkbox until a set
exists, and corrected the guidance. No backend semantics changed. The public
evaluation props now include this state explicitly; the refactor-only interface
comparison above predates this follow-up.

- Red: six failing regressions, 22 passing; green: 33 focused UI/hook/label tests.
- Full frontend suite: 1,795 tests in 152 files; lint and strict CCN gate pass.
- Existing backend service/router contract tests: 37 passed.
- Threshold browser spec: three passed, including real-page Preview/Save,
  disable/re-enable, job switching, reload and Clear with stateful API mocks.
- Independent review found no actionable introduced issue. OC-219 filed before
  repair and closed with evidence; v0.8.18 and user-guide instructions updated.
- Final delivery checks after OC-219: all 99 browser tests pass, production build
  passes, and all 11 bundle budgets pass (main 317.3 KiB gzip / 325 KiB).
