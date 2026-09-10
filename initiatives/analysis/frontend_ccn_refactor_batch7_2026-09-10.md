# Frontend CCN refactor batch 7 implementation plan

> For agentic workers: use subagent-driven-development with disjoint ownership,
> original-behavior characterization and independent task reviews.

**Goal:** simplify app navigation, job history and segmentation settings while
preserving their behavior, accessibility and public interfaces.

**Architecture:** keep the existing entry components and store/API contracts.
Move cohesive state management and presentation beside their current owners.
Primary owns browser coverage, integration, shared documentation and assets.

**Tech stack:** React, TypeScript, Zustand, Vitest, ESLint, Playwright and Vite.

**Spec:** the user requested committing completed work and continuing the CCN
backlog. Base is signed commit `4a6db08e` on branch `0819`. Strict CCN <=10;
informational report stays at 8. Behavioral changes require separate evidence.

## Global constraints

- Characterize original behavior through public component tests before editing
  production code. Preserve labels, CSS, exports/props, defaults, ordering,
  payloads, effect dependencies, state/request lifetime and keyboard/focus behavior.
- Each selected entry and helper must pass CCN <=10. Keep readable 9/10 functions;
  no metric-only wrappers, raised limits or complexity waivers.
- Shared APIs, stores, hooks, converters, registry, backend and Core are read-only.
  Report independently suspected bugs with evidence; do not silently fix them.
- Agents own only their assigned source/tests/helpers. No commits, assets, shared
  docs or subagents. Primary reviews and rebuilds the frontend after integration.
- Continue in the existing non-main checkout. Three owners may work concurrently
  because file ownership is disjoint and the user explicitly approved Astra 6
  parallel work. The user has now authorized the signed commit for this batch.
- DRIFT-01 remains deferred. Preserve OC-71/72/73/185 parking and OC-223 status.
  Queue baseline is 58 open / 4 parked. Release notes go under v0.8.19.
- CCN baseline: 105 functions above 10 in 77 files; report 167 above 8 in 109
  files; max32. Global strict failure on the remaining backlog is expected.
- On Windows use npm.cmd/npx.cmd. Record original/final logs in
  tmp_repro_artifacts. Type-check with `npx.cmd tsc --project tsconfig.json
  --noEmit`; do not use tsc -b or emit config JS files.

## Task 1: App layout and navigation

**Owned files:** `frontend/ml-canvas/src/components/Layout.tsx`, existing
`Layout.test.tsx`, and cohesive helpers in `src/components/appLayout/`.
**Interfaces:** keep the public Layout export and Outlet behavior. Read-only
inputs include useViewport, monitoringApi, applyTheme and NotificationCenter.

- [x] Extend existing public Layout tests and run them on original production
  source. Cover route links/active state, canvas/EDA collapsed desktop sidebar,
  mobile menu/backdrop/Escape, first-link focus and opener restoration, route and
  viewport changes, theme toggles, notification placement and unchanged Outlet.
- [x] Pin monitoring badge behavior: drift status fallback, initial error fetch,
  5-minute polling, hidden document/errors-route suppression, failed requests and
  cleanup. Preserve current request lifecycle rather than adding new guards.
- [x] Separate responsive navigation/header presentation and monitoring/menu
  lifecycle where responsibilities differ. Keep classes, DOM semantics and effect
  dependencies intact; avoid remounting the Outlet or changing local-state scope.
- [x] Run public tests, scoped normal ESLint/CCN10 and noEmit types. Baseline
  entry max29. Record maximum across the entry and all new helpers.
- [x] Write task report with exact checks and evidence; pass independent spec
  and code-quality review.

## Task 2: Jobs drawer and job cards

**Owned files:** `frontend/ml-canvas/src/components/panels/JobsDrawer.tsx`,
existing `JobsDrawer.test.tsx`, helpers in adjacent `jobsDrawer/`,
`src/components/panels/jobs/JobCard.tsx`, new `JobCard.test.tsx`, and helpers in
adjacent `jobCard/`. Do not edit JobDetailsView or its helpers.
**Interfaces:** preserve named JobsDrawer and JobCard exports/props. Store, jobs
API, registry API, job metadata/format functions, VirtualList and useEscapeKey
are read-only. Keep the current VirtualList estimate, order and key behavior.

- [x] Characterize original public components before production extraction.
  Cover tabs, ensemble subfilters, search/status/model filters and first-seen
  facet order, clear/reset/persistence, open/close/reopen, refresh, row selection,
  details/back, focus restoration and Escape.
- [x] Pin auto-pagination limits/reset and loading guards; inspected-run ordering,
  runJobs precedence over history, missing jobs, filter bypass and return to all
  jobs. Pin progress overlap, four terminal statuses, missing jobs and percentage.
- [x] Cover JobCard error/completed/pending states, score split precedence,
  CV/non-CV precision, tuned Params found fallback, ensembles/engine badges,
  missing dataset/model/time and click/Enter/Space behavior.
- [x] Separate history selection/filter/progress derivation, drawer presentation
  and card score/identity presentation. Keep all hooks before closed returns and
  preserve parent-owned state while subviews mount/unmount. Do not change store
  subscription granularity or asynchronous request behavior during extraction.
- [x] Run owned and relevant job-detail/store tests, scoped ESLint/CCN10 and
  noEmit types. Baseline JobsDrawer max28 (also13), JobCard max24.
- [x] Write task report and pass independent spec and code-quality review.

## Task 3: Segmentation settings

**Owned files:** `frontend/ml-canvas/src/modules/nodes/modeling/SegmentationSettings.tsx`,
existing `SegmentationSettings.test.tsx`, and adjacent `segmentationSettings/`
helpers. Existing TrainingSettings/trainingSettings and SegmentationNode are
read-only, as are shared form, validation, feedback and data/schema hooks.
**Interfaces:** keep SegmentationSettings export/props and SegmentationConfig
shape. Preserve graph conversion, leakage handling and JobStore interaction.

- [x] Extend and run public settings tests on original production code. Cover
  upstream dataset resolution, model registry filtering/fallback, model change
  and parameter default seeding, numeric/select/boolean edits, existing values,
  asynchronous definitions behavior and execution/reference-column changes.
- [x] Cover narrow/wide presentation, information dismissal/session persistence,
  validation reveal and action availability. Keep state/mounting semantics when
  switching sections or resizing and preserve all IDs/labels/help.
- [x] Pin train payload/target/job type, pending duplicate prevention, no dataset,
  client/server leakage feedback, failed submission/retry, single and parallel
  responses, polling, tab/drawer/inspected-run actions and RunFeedback lifetime.
- [x] Extract model/parameter loading, submission coordination and settings
  sections with coherent responsibilities. Do not alter effect dependencies,
  captured config callbacks, pipeline conversion or error/promise semantics.
- [x] Run public and relevant modeling/leakage tests, scoped ESLint/CCN10 and
  noEmit types. Measured baseline: settings28, upstream dataset search15,
  handleTrain12; the parameter renderer is already5 and needs no metric-only split.
- [x] Write task report and pass independent spec and code-quality review.

## Task 4: Primary integration and delivery

- [x] Review task evidence and diffs independently, including new helper files.
- [x] Exercise real-page browser routes for desktop/mobile navigation, theme,
  job filters/details/keyboard and segmentation configuration/submission; API
  fixtures are acceptable and must be described as mocked.
- [x] Run full Vitest, normal ESLint, TypeScript/Vite build, size budgets,
  Chromium suite and both global CCN commands. Validate selected scopes <=10.
- [x] Update inventory, tracker, open queue and concise v0.8.19 release notes.
  Record any separately confirmed pre-existing defects without hiding them.
- [x] Complete final independent review, whitespace/scope checks and deliver
  the prior commit hash plus bounded frontend checks for this uncommitted batch.


## Verification evidence

Original public characterization passed before production extraction: Layout 22,
JobsDrawer/JobCard 49, Segmentation 29. All three independent task reviews report
Spec PASS and Quality PASS. Entry maxima: Layout 29 -> 5, JobsDrawer 28 -> 5,
JobCard 24 -> 1, Segmentation 28 -> 10; maximum including all helpers is 10.

Final integrated checks (logs in tmp_repro_artifacts):

| Check | Result | Log |
|---|---|---|
| Full Vitest | 2,045 tests  / 158 files pass | ccn7-vitest-final.log |
| Normal ESLint | pass | ccn7-lint-final.log |
| TypeScript + Vite production build | pass; assets rebuilt | ccn7-build-final.log |
| Bundle budgets | all 11 pass | ccn7-size-final.log |
| Strict CCN10 | expected exit 1; 98 functions / 73 files / max32 | ccn7-gate-final.log |
| Informational CCN8 | exit 0; 166 functions / 111 files | ccn7-report-final.log |
| Full Chromium confirmation | 117/117 pass, 2 workers | ccn7-browser-confirmation.log |
| Focused threshold repetition | 9/9 pass across 3 runs | ccn7-threshold-recheck.log |

The first four-worker Chromium run passed 116/117; its existing threshold
checkbox assertion saw an unchecked value after clicking. No source change
was made for that failure. Three focused repetitions and the complete two-worker
confirmation passed. A separate delayed-defaults diagnostic did not reproduce
the proposed reset mechanism (the checkbox stayed checked), so no root cause or
fix is claimed. Preserve the first failure and diagnostic logs for follow-up.

The six new browser cases exercise real pages/stores with mocked HTTP: mobile
and desktop navigation/theme/notification placement, job filters/cards/details,
and compact/expanded segmentation edits plus serialized training submission.
Jobs at 1100px and expanded Segmentation at 1440px screenshots were inspected and kept
under tmp_repro_artifacts. Browser checks do not claim live model training.

OC-225 (job-detail accessible name/Back label) and OC-226 (stale Segmentation
definitions) were independently confirmed as pre-existing and filed with
reproduction evidence; neither is repaired by the pure extraction. Queue now
60 open / 4 parked; DRIFT-01 remains deferred outside the audit counts.

Prior completed work is signed commit `4a6db08e`. This batch was presented for
the user's frontend check: navigation/theme/mobile Escape; Jobs filters,
tabs, detail/back; Segmentation algorithm/reference/parameters, panel expansion
and a real training submission. Final independent integrated review reports
PASS with no introduced regression or scope violation. The final delivery audit
also passes: 31 scoped text files, UTF-8/whitespace/EOF, older changelog sections,
actual queue counts, generated entry references and unchanged CCN policy.
The user subsequently authorized committing the completed batch and continuing.
