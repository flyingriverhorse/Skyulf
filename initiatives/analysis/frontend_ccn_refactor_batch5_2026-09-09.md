# Frontend CCN refactor batch 5 implementation plan

> For agentic workers: use subagent-driven-development with independent file
> ownership and review after each task. Continue in the user's active workspace.

**Goal:** simplify comparison, pipeline diff and error-log responsibilities while
preserving their current user-visible behavior.

**Architecture:** retain public entry components and API contracts; move cohesive
data preparation, state management and presentation beside their owning view.
The primary owns shared integration, browser tests, documentation and assets.

**Tech stack:** React, TypeScript, Vitest, ESLint, Playwright, Vite.

**Spec:** the user requests continuation of the accepted frontend CCN work after
removing the CCN 10 exception commit. Use 8 as a readability target, with no
metric-only splitting or renewed exception gate. Release notes belong to 0.8.19.
Previous record: [batch 4](frontend_ccn_refactor_batch4_2026-09-09.md).

## Global constraints

- Baseline: `15aa043a`, branch `0819`, no tracked changes. Preserve temporary files.
- Preserve public props, labels, callbacks, API arguments, request lifetimes,
  formatting, defaults, fallbacks, ordering, styling and keyboard interactions.
- Pure refactor: characterize original behavior before production edits. File
  any independently confirmed defect with evidence before a separate repair.
- Target function CCN <=8; do not increase limits, add exemptions or wrap a
  complex expression in many trivial helpers merely to change the number.
- No dependencies, backend edits or commits. The user checks the frontend before
  a later commit request. Agents edit only their assigned production/test scopes.
- Primary owns package/workflow, browser tests, release notes, tracker, queue and
  generated assets. Keep the 57 open and 4 parked findings unless a fix warrants it.
- Preserve parked OC-71/72/73/185. Record completed work under v0.8.19.
- Verify original/refactored targeted tests, independent review, full Vitest,
  ESLint, strict CCN, TypeScript/Vite build, size budgets and relevant browsers.

## Tasks

### Task 1: Comparison table (Astra implementer)

**Files:** `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/ComparisonTableView.tsx`,
new adjacent `ComparisonTableView.test.tsx`, cohesive modules under adjacent
`comparisonTable/`. Read `utils/jobMeta`, `core/utils/metricMeta` and `format` as
dependencies; do not edit them or another owner's files.

**Interface:** preserve all ten current Props members (`selectedJobs`,
`metricKeys`, four expansion values and four setters) and named component export.
New modules consume existing JobInfo and the view's private graph/config types.

- [x] Characterize full rows, expansion callbacks, missing metrics, best-value
  direction/ties, scoring-metric groups, nested training vs tuning configs,
  ensemble summaries and graph-chain alignment, including shared/branch/cyclic
  ancestors and operation summaries. Render the public component in these tests.
- [x] Run the new characterization tests against the original component and
  record results before extracting production code.
- [x] Separate pipeline graph preparation, metric rows, parameter/config rows
  and ensemble rows where responsibilities differ. Preserve merge order, key
  union order, finite/zero/fallback handling, tooltip text and CSS exactly.
- [x] Rerun focused tests and lint including strict CCN 8 for entry and helpers;
  report maximum before/after and all files added. Independent review required.

### Task 2: Pipeline diff (Astra implementer)

**Files:** `frontend/ml-canvas/src/components/pages/experiments/PipelineDiffView.tsx`,
new adjacent `PipelineDiffView.test.tsx`, helpers under adjacent `pipelineDiff/`.
Read but retain `pipelineDiffLayout.ts`, `DiffNode.tsx`, `StatusDot.tsx`, graphDiff
and `ExperimentsPage/components/RunIdentifierLabels.test.tsx` unchanged.

**Interface:** retain `PipelineDiffView({ jobs: JobLite[] })` and the existing
job-detail requests, diff/layout calls and read-only React Flow props.

- [x] Characterize non-two-job selection, loading, both/single missing graphs,
  per-side failures, swap direction and reset, metadata, stale request completion
  after job replacement/unmount, and diff summary/changed-parameter rendering.
- [x] Run these tests and existing run-label tests against original source.
- [x] Extract snapshot loading/state and coherent graph/summary presentation;
  preserve effect dependencies, cancellation, ready/missing/error distinction,
  graph ordering, swap reset, empty/partial views and fit/read-only behavior.
- [x] Verify focused cases plus existing graph layout/label tests; run strict
  CCN 8 on owned entry/helpers and report before/after for independent review.

### Task 3: Error log page (Astra implementer)

**Files:** `frontend/ml-canvas/src/pages/ErrorLogPage.tsx`, its existing test file,
and adjacent modules under `errorLog/`. Read shared operational navigation,
monitoring API and confirmation dependencies without editing them.

**Interface:** retain the named no-props ErrorLogPage component, monitoring API
arguments/returns, operational link context and notification/confirmation flows.

- [x] Extend existing public-page tests for event/issue tabs, HTTP/pipeline data,
  time/search/severity/job/node facets, timeline, refresh/loading/error/empty
  states, row details, resolution, clear confirmation, export and copy behavior.
  Pin relevant request arguments and visible results, not internal helper calls.
- [x] Run characterization tests against original source before refactoring.
- [x] Extract coherent request/state management and row/detail/filter/timeline
  presentation. Preserve timer/effect behavior, selected filters, paging/order,
  diagnostic navigation, CSV escaping and success/error notification paths.
- [x] Run focused tests and strict CCN 8 for owned entry/helpers. Record unchanged
  contracts and before/after values; independent review must examine async state.

### Task 4: Integration and delivery (primary)

**Files:** package/workflow, browser specs, this plan, changelog/0.8.x.md,
opus_core_analysis-tracker.md, open queue if needed, static/ml_canvas assets.

- [x] Review each task against baseline and original/refactored test evidence.
- [x] Verify actual-page browser paths for comparison/diff and error filtering,
  including keyboard and compact viewport; full suite also checks existing UI.
- [x] Add clean entries/helper folders to `complexity:check` and clarify workflow
  scope without changing thresholds or exemptions.
- [x] Run full unit tests, ESLint/CCN, TypeScript/Vite build, size and browser
  gates. Record exact results and the next report inventory.
- [x] Record concise completed v0.8.19 notes and tracker evidence; give the user
  a bounded manual check list. Keep changes uncommitted pending that workflow.

### Task 5: OC-221 request-lifetime repair (after Task 3 review)

**Files:** the extracted Error Log request hook and its page tests, plus the
primary-owned tracker, queue and v0.8.19 changelog note.

- [x] Review Task 3 as a pure refactor before changing its request semantics.
- [x] Convert the reproduced stale-result case into the desired latest-request
  assertion and run it red. Add reverse completion order, stale rejection,
  stale loading settlement, pipeline-log responses and unmount coverage.
- [x] Apply a request-generation guard to the existing loading path. A newer
  filter load or refresh supersedes older results; cleanup prevents stale
  completion after unmount. Keep HTTP success usable before pipeline logs arrive.
- [x] Verify the fixed behavior and independent scoped review, close OC-221 with
  evidence, and include the repair before Task 4 final integration gates.

## Preflight and execution record

- Baseline report: 181 functions above 8 in 113 files, maximum 34; current strict
  gate passes. Selected maxima: ComparisonTableView 34, PipelineDiffView 33,
  ErrorLogPage 32. The unreferenced VariableCard is not selected for metric work.
- Current branch also contains `15aa043a` (RestoreSessionBanner/RunFeedback).
  The removed `98112a47` CCN 10 exception commit is not restored.
- At preflight the tracker/queue was checked and no new defect was yet confirmed.
- Use the user's active non-main branch, preserving the established in-place
  manual frontend check workflow. Independent ownership permits parallel work.
- During Task 3 characterization, OC-221 was reproduced on unchanged source and
  filed separately: an older search response replaces newer results. Task 5
  handles this bounded defect after the extraction review, with its own red/green
  regression cycle. Other behavior-preservation constraints remain in force.

## Completed verification - 2026-09-10

| Scope | Maximum CCN before/after | Characterization before/after |
|---|---|---|
| Comparison table | 34 -> 8; entry 14 -> 5 | 11 / 11 public-component tests |
| Pipeline diff | 33 -> 8; entry 33 -> 5 | 35 / 35 related tests |
| Error Log extraction | 32 -> 8; page 32 -> 2 | 28 / 28 public-page tests |
| OC-221 separate repair | Hook maximum remains 7 | 11 failed + 27 passed -> 38 passed |

- Independent extraction reviews and the OC-221 review: spec/quality PASS,
  no actionable findings. There are 26 cohesive helper modules across the scopes.
- Four actual-page Chromium tests pass against original and extracted source
  at 1440px/900px; final complete suite: **105 passed**. API routes are mocked.
- Full Vitest: **156 files, 1,926 passed**. ESLint, strict CCN 8 and TypeScript/Vite
  build pass. All 11 size budgets pass; main bundle **319.7 / 325 KiB gzip**.
- Full report: **181 -> 172** functions above 8, **113 -> 110** affected files,
  maximum **34 -> 32**. No raised limit, waiver, backend or dependency change.
- Gate includes all three entries and helper folders; production assets rebuilt.
  Release notes are under v0.8.19, with v0.8.18 and older unchanged from branch base.
- OC-221 is closed with red/green evidence in the tracker; queue **57 open / 4 parked**.
  Logs: `tmp_repro_artifacts/ccn5-*-final.log`; independent reports remain under
  `.superpowers/sdd/frontend_ccn_refactor_batch5_2026-09-09/`.

## User verification and source-wide gate - 2026-09-10

The user confirmed the frontend checks below and requested a signed commit.
They also chose to enforce CCN 8 on all `src` TypeScript now, accepting CI failure
during the remaining cleanup. The long scope list becomes
`eslint src --ext ts,tsx --rule "complexity: [error, 8]" --max-warnings 0`;
new source files are covered automatically. The new command exits **1 with
172 errors** from the existing backlog; normal lint and a fresh production build
pass, with an unchanged main asset hash. This supersedes the
earlier scoped gate's passing status, without changing runtime behavior or the
completed refactor's test evidence. Workflow comments and v0.8.19 notes agree.

## User frontend checks

1. Select two completed runs in Experiments. Check Detailed Metrics & Params,
   best-score highlights, parameters and opening/closing each section.
2. Open Pipeline Diff, use Swap, switch the selected runs and repeat in a narrow
   window. Confirm graph differences and focus/scrolling remain usable.
3. In Error Log, change search, severity and time filters quickly, then Refresh.
   Confirm results match the latest filter; try Events/Issues, View sample,
   Copy and CSV export. Resolve/reopen a disposable event if available.
