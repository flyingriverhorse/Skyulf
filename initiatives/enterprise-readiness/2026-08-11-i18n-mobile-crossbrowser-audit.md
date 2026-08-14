# Enterprise Readiness — i18n, Responsive UI & Cross-Browser Audit

**Date:** 2026-08-11  
**Scope:** `frontend/ml-canvas` (React/TypeScript). This audit intentionally
does not repeat the detailed keyboard/ARIA findings in
[2026-08-11-technical-debt-deep-dive.md](2026-08-11-technical-debt-deep-dive.md).
It focuses on customer-segment readiness: language/RTL, tablet/mobile use, and
browser support. Severity means impact on a real prospective customer segment,
not a generic code-quality score.

## Executive summary

The operations shell has credible narrow-screen adaptations, while the pipeline
authoring canvas deliberately degrades to inspect-only below 1024 px. That is a
reasonable B2B product decision if stated as a supported-device policy, but it
means tablet users cannot author pipelines by default. The UI is currently
English-only and is not structurally prepared for translation or RTL. Numeric
presentation is fragmented: some dates/counts use the browser locale, many ML
scores use fixed decimal formatting, and only selected statistical views use
scientific notation. Browser support is effectively “current Chromium,” with
no declared support matrix or Firefox/WebKit CI coverage.

## 1. Internationalization and RTL

| Finding | Severity | Effort |
|---|---|---|
| **No i18n framework, locale provider, message catalog, or translation boundary exists.** `package.json:23-65` lists the runtime dependencies and contains no `react-i18next`, FormatJS, Lingui, or equivalent. The root document fixes `lang="en"` (`index.html:2`). A scan found **1,017 direct JSX English-text candidate lines in 136 of 241 TSX files** (heuristic deliberately counts visible text nodes, not every prop/template string). Representative page strings include Dashboard headings/table headers (`src/pages/Dashboard.tsx:118-119,163,196,233,250-253`), Jobs (`src/pages/Jobs.tsx:360-368,378-379`), Data Sources (`src/pages/DataSources.tsx:228-255,305-326,355-360`), canvas empty state (`src/components/canvas/FlowCanvas.tsx:371-384`), and node-palette labels (`src/components/layout/Sidebar.tsx:68-76,100-113`). Translating later would require widespread extraction and translator-context cleanup. | **High** — blocks a non-English enterprise rollout | **Large** |
| **Dates are only partly locale-sensitive, and one dashboard chart explicitly forces U.S. English.** Many screens call `toLocaleString()`/`toLocaleDateString()` (for example Dashboard recent-job time, `src/pages/Dashboard.tsx:270-272`; Jobs uses the same approach). But the weekly-activity chart hardcodes `toLocaleDateString('en-US', ...)` (`src/pages/Dashboard.tsx:92-93`), and no application locale is passed/configured. This produces mixed date conventions once a locale strategy is introduced. | **Medium** | **Small–Medium** |
| **RTL is not implemented and physical directional styling is pervasive.** No document/component `dir` handling or CSS `direction` strategy was found. A source scan found **401 physical directional Tailwind tokens in 111 TSX files** and **zero CSS logical-property declarations**. Examples include the always-left mobile drawer (`src/components/Layout.tsx:125-128`), absolute left search icons and asymmetric `pl`/`pr` input padding (`src/pages/DataSources.tsx:268-275`; `src/components/layout/Sidebar.tsx:87-93`), right-aligned action columns (`src/pages/DataSources.tsx:350-360`), and a left-anchored collapsed palette control (`src/components/layout/Sidebar.tsx:48-59`). These will not mirror under `dir="rtl"`; flow diagrams and chart labels need separate visual validation. | **High** for Middle-East/Hebrew expansion; **Low** otherwise | **Large** |
| **Translation-sensitive strings are embedded in logic, tooltips, templates, and configuration labels, not only page chrome.** Examples include the canvas confirmation interpolation (`src/components/canvas/FlowCanvas.tsx:104-110`), raw chart strings/labels (`src/components/pages/ExperimentsPage/components/RegressionChartsForSplit.tsx`, e.g. “Actual”, “Predicted”, tick labels), node categories (`src/components/layout/Sidebar.tsx:68`), and status capitalization via string manipulation (`src/pages/DataSources.tsx:333-345`). A literal-extraction migration must preserve pluralization, variables, capitalization, and translator context rather than simply wrapping text. | **Medium** | **Medium–Large** |

### Recommended i18n baseline

Adopt one message system (for example `react-i18next` or FormatJS), put the
active locale and `dir` on `<html>`, and provide a locale-aware formatting
module (`formatDate`, `formatNumber`, `formatPercent`, `formatMetric`). Extract
the shell and the four highest-traffic pages first, then node configuration and
experiments. Replace physical layout utilities with logical CSS (or RTL-aware
Tailwind conventions) as each component is touched. Add English and one RTL
locale smoke test before claiming RTL support.

## 2. Mobile, tablet, and responsive usability

### What works

* The root page has a correct viewport meta tag (`index.html:6`).
* The operational application shell adapts below 768 px: `useViewport` defines
  mobile `<768`, tablet `<1024`, and desktop `>=1024`
  (`src/core/hooks/useViewport.ts:3-24`); `Layout` switches the persistent
  navigation to an off-canvas drawer (`src/components/Layout.tsx:15-19,
  110-128,201-214`).
* Dashboard uses sensible stacked grids (`src/pages/Dashboard.tsx:114-116,
  133,160,229`) and horizontally scrollable tables (`:246-277`). Jobs also
  stacks its header/filter layout and makes long tabs horizontally scrollable
  (`src/pages/Jobs.tsx:357-374`). Data Sources wraps header actions and gives
  its table horizontal overflow (`src/pages/DataSources.tsx:226-256,350-352`).
  These pages are usable on phone/tablet for monitoring and management,
  subject to manual viewport testing.

### Findings

| Finding | Severity | Effort |
|---|---|---|
| **The canvas is intentionally not a tablet authoring experience.** Below 1024 px `useReadOnlyMode` returns read-only unless the user explicitly overrides it (`src/core/hooks/useReadOnlyMode.ts:5-18,26-32`). `MainLayout` hides both authoring sidebars and describes the canvas as “pan/zoom + inspect-only” because touch drag-and-drop and panels do not work well (`src/components/layout/MainLayout.tsx:38-42,91-102`). `FlowCanvas` disables node dragging, connecting, reconnecting, and deletion in that state (`src/components/canvas/FlowCanvas.tsx:316-324`). This is explicit product behavior, not a silent rendering failure. | **Medium** for a desktop-first B2B ML tool; **High** if tablet field/exec authoring is a requirement | **Medium–Large** |
| **Canvas placement is HTML5 drag-and-drop/mouse oriented; no touch/pointer placement path was found.** Palette items rely on `draggable` and `onDragStart` (`src/components/layout/Sidebar.tsx:19-22,117-124`); the canvas consumes React `DragEvent` `onDragOver`/`onDrop` and `dataTransfer` (`src/components/canvas/FlowCanvas.tsx:200-203,279-315`). There are no `touchstart`/`touchmove`/`touchend` or Pointer Event handlers in these paths. Click-to-add is a useful fallback (`Sidebar.tsx:36-44,123-124`), but moving/connecting nodes remains disabled in the default tablet policy. | **Medium** | **Large** if tablet authoring is supported; **None** if policy remains desktop-only |
| **The tablet limitation is discoverable but easy to miss and reversible in a way that may expose a cramped unsupported layout.** Navbar shows a read-only chip with a hover/title explanation and permits “enable editing” (`src/components/layout/Navbar.tsx:15-25,84-101`), rather than a persistent responsive-policy message. The tooltip itself says the canvas is a “tablet view,” but a touchscreen user may not see hover text. | **Low–Medium** | **Small** |
| **Responsive implementation is component-by-component rather than protected by a viewport test matrix.** Existing tests include a few 390 px component checks (such as `ArtifactCoverageList.test.tsx`), but Playwright has only a Desktop Chrome project (`playwright.config.ts:25-30`). No configured tablet/mobile browser project tests Dashboard, Jobs, Data Sources, or canvas read-only behavior end to end. | **Medium** | **Medium** |

### Product decision required

Document a supported-device policy. If it is **desktop authoring; tablet
inspection; mobile operations**, make that language visible before users enter
the canvas and test the read-only route at 768/1024 px. If tablet authoring is
commercially required, build pointer/touch node placement and connection
interactions, then test Safari on iPad—not just Chromium emulation.

## 3. Cross-browser compatibility

| Finding | Severity | Effort |
|---|---|---|
| **No browser support contract or multi-engine CI exists.** There is no Browserslist configuration and `vite.config.ts:6-59` has no browser target. Playwright declares only Desktop Chrome (`playwright.config.ts:25-30`); its install script likewise installs only Chromium (`package.json:16-19`), and CI invokes that script before the one E2E run (`.github/workflows/frontend-tests.yml:119-128`). Vitest is jsdom-based (`package.json:81-95`), so it does not substitute for Firefox/WebKit rendering coverage. | **High** for enterprise customers standardized on Firefox/Safari | **Medium** |
| **Several APIs lack an application fallback or a documented baseline.** Canvas copy duplicates nodes using `structuredClone` with no fallback (`src/core/hooks/useClipboard.ts:35-38`), and element/layout logic unconditionally constructs `ResizeObserver` (`src/core/hooks/useElementSize.ts:17-22`). Infinite scrolling uses `IntersectionObserver` (`src/pages/ModelRegistry.tsx:105`), and copy actions use `navigator.clipboard` (for example `src/components/shared/RecordLink.tsx:63-72`; this one does fail safely, leaving the link usable). Modern evergreen browsers support these APIs, but an unsupported/managed older browser can lose an interaction or crash a mounted component. | **Medium** | **Small–Medium** |
| **Browser-specific CSS is present without a compatibility strategy.** Global scrollbar styling is WebKit-only (`src/index.css:136-151`), while UI overlays/nav use `backdrop-filter` (`src/styles/layout.css:20-23`; also canvas/modal utility classes). These are progressive visual issues in current Firefox, but without Firefox/WebKit visual smoke tests regressions will ship unnoticed. `-webkit-background-clip` is correctly paired with the standard property (`src/styles/layout.css:38-41`), which is a positive example. | **Low–Medium** | **Small** |
| **The canvas and visualization stack raise the testing priority.** The app ships React Flow, Plotly GL, Leaflet, Chart.js, and Recharts (`package.json:41-57`). These libraries rely heavily on SVG/canvas/WebGL/layout behavior, where Chromium-only E2E has weak predictive value for Safari and Firefox. This is a risk statement, not evidence of a confirmed rendering defect. | **Medium** | **Medium** |

### Recommended browser policy

Publish a minimum supported set (for example, latest two versions of Chrome,
Edge, Firefox, and Safari; explicitly state iPadOS support). Add Playwright
Firefox and WebKit projects to smoke routes plus a tablet/iPad viewport. Use
feature detection/fallbacks where loss blocks a workflow (`structuredClone`,
`ResizeObserver`), and treat visual enhancements such as blur/scrollbars as
progressive.

## 4. Numbers, metrics, and scientific notation

| Finding | Severity | Effort |
|---|---|---|
| **No shared numerical presentation policy/helper exists; fixed decimals dominate.** The metric comparison table renders most values with `toFixed(4)` and standard deviations with `toFixed(6)` (`src/components/pages/ExperimentsPage/components/ComparisonTableView.tsx:392`). Job details uses the same `_std`/four-decimal split (`src/components/panels/jobs/JobDetailsView.tsx:204-206`) and regression charts use repeated `toFixed(2–4)` in axes and tooltips (`src/components/pages/ExperimentsPage/components/RegressionChartsForSplit.tsx`). This loses meaningful small values (for example `0.000004` becomes `0.0000`) and is not locale-aware. | **Medium** — affects interpretation of ML results | **Medium** |
| **Scientific notation is used appropriately in some statistical surfaces, but inconsistently.** Feature-selection p-values use `toExponential(2)` (`src/modules/nodes/processing/FeatureSelectionNode.tsx:192`); EDA normality and ANOVA p-values also use scientific notation (`src/components/eda/DistributionChart.tsx:102`; `src/components/eda/tabs/TargetAnalysisTab.tsx:109,245,264`). In contrast, Drift Alert rounds KS p-values to four decimals before display (`src/pages/drift/DriftAlertModal.tsx:70-83`), and time-series/variable cards use `toFixed(3)`. A small significant p-value can therefore display as `0.0000` in one product area and `1.23e-8` in another. | **Medium** | **Small–Medium** |
| **Large counts have partial locale-aware formatting; score formatting does not.** Slow Nodes calls `toLocaleString()` for run/job counts (`src/pages/SlowNodesPage.tsx:256,261,491,520,653`), but uses `toFixed` for durations; Dashboard stat values are passed as raw numbers (`src/pages/Dashboard.tsx:133-155`) and therefore do not consistently gain grouping. `Intl.NumberFormat` was not found in source. | **Low–Medium** | **Small** |

### Recommended metric format policy

Create one formatter with locale injected from the eventual i18n provider:

* counts/rows/bytes: `Intl.NumberFormat(locale)` (or compact notation only
  where the full value remains available);
* percentages: `Intl.NumberFormat(locale, { style: 'percent', ... })`;
* ordinary bounded ML scores: a documented 3–4 significant-digit rule;
* p-values/losses: use scientific notation below a defined threshold (for
  example `< 1e-4`) and show a precision-preserving tooltip/raw value;
* avoid converting a formatted metric back to `Number`, as the drift alert
  currently does (`src/pages/drift/DriftAlertModal.tsx:76-82`).

Add formatter unit tests for `0`, `0.876543212`, `0.000004`, `1e-12`,
`1_234_567`, negative values, `NaN`, and `Infinity`, plus locale snapshots for
at least English and the first supported non-English locale.

## Prioritized top five

1. **Adopt i18n architecture before international sales work** — message
   catalog/provider, locale persistence, date/number formatter, and extraction
   plan. **High / Large.**
2. **Make an explicit device-support decision for the canvas** — either label
   desktop-only authoring and test tablet inspection, or fund pointer/touch
   authoring. **Medium–High / Small policy + Large implementation if needed.**
3. **Add RTL as a deliberate workstream** — `dir` support, logical layout
   conversion, and real RTL visual testing. **High for Middle-East expansion /
   Large.**
4. **Declare and enforce a browser matrix** — Firefox/WebKit E2E smoke tests
   and tablet viewport coverage; test React Flow/Plotly routes. **High /
   Medium.**
5. **Centralize numeric/metric rendering** — consistent significant-digit,
   p-value/scientific-notation, grouping, and locale behavior. **Medium /
   Medium.**
