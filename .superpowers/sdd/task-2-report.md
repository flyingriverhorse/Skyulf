# Task 2 Execution Report — Frontend UX Roadmap Rerun: Shared Foundations

## Status
DONE

## Commit hash(es)
- `3548b7e592000db1792965961f3daa54f2a14249` — "docs: refresh shared frontend UX evidence" (branch `075`)

## Files changed
- `docs/ux/frontend-ux-roadmap.md` (only file in the commit; +262 insertions, -3 deletions)

`git show --stat HEAD` confirms only this file is in the commit. The
pre-existing, unrelated modification to `.superpowers/sdd/progress.md` was
left unstaged and is not part of this commit, per the design doc's
instruction to keep unrelated modifications outside audit commits.

## Inputs read
- `.superpowers/sdd/task-2-brief.md`
- `docs/superpowers/specs/2026-08-07-frontend-ux-roadmap-rerun-design.md`
- `docs/ux/frontend-ux-roadmap.md` (existing `FND-001`–`FND-006` findings and
  Task 1's rerun scaffold)
- `.superpowers/sdd/task-1-report.md` (prior task's baseline, for continuity)
- `frontend/ml-canvas/src/components/Layout.tsx`
- `frontend/ml-canvas/src/components/shared/` (`LoadingState.tsx`,
  `EmptyState.tsx`, `ErrorState.tsx`, `ModalShell.tsx`, `Skeleton.tsx`,
  `index.ts`)
- `frontend/ml-canvas/src/components/ui/`
- `frontend/ml-canvas/src/components/layout/CommandPalette.tsx`
- `frontend/ml-canvas/src/components/layout/NotificationCenter.tsx`
- `frontend/ml-canvas/src/components/layout/ShortcutsOverlay.tsx`
- `frontend/ml-canvas/src/components/layout/Navbar.tsx`
- `frontend/ml-canvas/src/core/toast.ts`
- `frontend/ml-canvas/src/core/utils/a11y.ts`
- `frontend/ml-canvas/src/core/store/useViewStore.ts`
- `frontend/ml-canvas/src/core/store/useNotificationsStore.ts`
- `frontend/ml-canvas/src/modules/nodes/processing/EncodingNode.tsx`
- `frontend/ml-canvas/src/components/data/AddSourceModal.tsx`
- `frontend/ml-canvas/src/components/pages/InferencePage.tsx`
- `frontend/ml-canvas/src/components/eda/EDASidebar.tsx`
- `frontend/ml-canvas/src/pages/Dashboard.tsx`, `EDAPage.tsx`,
  `DeploymentsPage.tsx`, `ModelRegistry.tsx`,
  `ExperimentsPage/components/EvaluationView.tsx`,
  `ExperimentsPage/components/SegmentationView.tsx` (for `FND-004`
  retry-consistency re-check)
- `frontend/ml-canvas/e2e/a11y.spec.ts`
- `frontend/ml-canvas/playwright.config.ts`

## Step 1: Reinspect shared-state usage

Ran from `frontend/ml-canvas/`:

```bash
grep -RInE "LoadingState|EmptyState|ErrorState|PageSkeleton|toast\.|disabled=" src \
  --include="*.ts" --include="*.tsx"
```

Result: `256` matches. Spot-checked:
- `LoadingState.tsx`, `EmptyState.tsx`, `ErrorState.tsx` still contain no
  `role=` or `aria-live` attribute (`grep -n "role=\|aria-live"` on all three
  returned nothing).
- `PageSkeleton` is still used exactly once, as `App.tsx`'s route-level
  `Suspense` fallback (`RouteFallback`).
- `toast.` (`success`/`error`/`info`/`warning`) call sites now number `80`
  (`grep -RIn "toast\.\(success\|error\|info\|warning\)"`), consistent with
  `toast.ts`'s own "~40 call sites" comment as an approximation, not a
  material change to `FND-003`/`FND-004`.
- `disabled=` appears `85` times, same kind of usage as the original
  baseline (in-flight/unavailable actions).
- Re-read `Dashboard.tsx`, `EDAPage.tsx`, `DeploymentsPage.tsx` (still pass
  `onRetry` to `ErrorState`) vs. `ModelRegistry.tsx` (line 158, no
  `onRetry`) and `EvaluationView.tsx`/`SegmentationView.tsx` (no `onRetry`):
  same asymmetry as the original `FND-004` evidence, unchanged.

## Step 2: Repeat live shared-foundation walkthroughs

**Method:** Started the project's own Vite dev server exactly as
`playwright.config.ts`'s `webServer` does
(`npm run dev -- --host 127.0.0.1 --port 5173 --strictPort`), confirmed it
responded (`curl` → `200`), then drove a real, interactive Chromium browser
against it using the Playwright MCP browser tool
(`playwright-browser_navigate`, `_resize`, `_snapshot`, `_evaluate`,
`_click`, `_press_key`, `_take_screenshot`). This is a live browser session
against the actual running frontend, not a static reading of source.
**Widths exercised:** `1440×900`, `1024×900`, `768×1024`, `390×844`.

Key measurements (all **Observed** unless noted):

- **`/canvas` at 1440/1024/768/390:** the shared view switcher
  (`[data-testid="navbar-views"]`) measured `352.55px` wide at every width
  and stayed inside the main content pane at all four widths in this rerun,
  including 390 px (main pane `326px`, switcher ending at `x=403.28`,
  Notifications bell at `x=342–378`, `document.documentElement.scrollWidth
  === window.innerWidth`, i.e. no horizontal overflow).
- **`/` (Dashboard) at 390 px:** `<aside>` measured `256px` wide; the main
  pane began at `x=256` and measured `134px` wide; no document horizontal
  overflow. This reproduces the original `FND-001` evidence exactly.
- **`FND-002` overlay focus, 1440 px, `/canvas`:**
  - Shortcuts: opened it; the next `Tab` moved focus to the covered "Open
    command palette" navbar button, outside the dialog. Matches original
    evidence exactly.
  - Command Palette: opened it (Ctrl/Cmd+K), programmatically focused its
    last in-dialog focusable element (`38` focusables found), then pressed
    `Tab` once; focus moved to the "Open Tanstack query devtools" button
    behind the still-open dialog. This is new, concrete Observed evidence
    (the original audit only Inferred this).
  - Notification detail modal: seeded one notification via
    `localStorage.setItem('skyulf-notifications', ...)` (matching the
    store's own persisted-state shape in `useNotificationsStore.ts`),
    opened the panel, clicked the row to open the detail modal, and
    confirmed `document.activeElement` was still `<body>` (focus was never
    moved into the modal on open); pressing `Tab` once moved focus to the
    sidebar's "Collapse sidebar" button, escaping the modal. New, concrete
    Observed evidence (originally Inferred).
- **`FND-006` shell-view history, 1440 px, `/canvas`:** clicked
  "Experiments"; URL remained `/canvas` (no navigation); none of the three
  switcher buttons exposed `aria-current`/`aria-selected`/`aria-pressed`
  (checked via `getAttribute` on all three); `browser_navigate_back`
  returned to `/` (the route visited before Canvas), not to a prior shell
  view. Matches original evidence exactly, now with an explicit `aria-*`
  check.
- **New defect found live:** with the seeded notification, opening the
  panel produced a live React `validateDOMNesting(...)` console warning
  ("`<button>` cannot appear as a descendant of `<button>`") at
  `NotificationCenter.tsx:274`. The accessibility snapshot's computed
  accessible name for the row button concatenated the nested Dismiss
  button's own text into the row's name. Focusing the row and pressing
  `Tab` moved focus onto the nested `aria-label="Dismiss"` button,
  confirming the invalid nesting is live and keyboard-reachable. Recorded
  as new finding `FND-007`.
- Spot-checked FND-005 source evidence for material change: `EncodingNode.tsx`,
  `AddSourceModal.tsx`, `EDASidebar.tsx`, `InferencePage.tsx` all still match
  the original citations verbatim (visible `span` labels with no
  `htmlFor`/`id`/ARIA association; placeholder-only EDA filter controls; the
  Inference JSON `textarea` with no `label`/`aria-label`/`aria-labelledby`).
- Not independently re-driven this rerun (documented as Inferred, matching
  the original audit's own scope limits): Inference's own shell-view
  focus/history/selected-state behavior; EDA/Data/Operations route-level
  compact-navigation at 390 px beyond the Dashboard/Canvas cases actually
  driven; full keyboard-only completion of Canvas node forms, Add Source
  modal, EDA setup/filter controls, and the Inference editor.

Screenshots and MCP browser trace artifacts (`.playwright-mcp/`,
`canvas-390.png`) were generated during the walkthrough and deleted after
capturing the needed measurements; they were never staged or committed.

## Step 3: Repeat accessibility automation

Ran from `frontend/ml-canvas/`:

```bash
npm run test:e2e -- e2e/a11y.spec.ts --project=chromium
```

**Exit code: `1`.** All `4` tests (`dashboard (/)`, `canvas (/canvas)`,
`data (/data)`, `eda (/eda)`) failed identically before any route loaded or
axe ran:

```
Error: browserType.launch: Executable doesn't exist at
/Users/BH7043/Library/Caches/ms-playwright/chromium_headless_shell-1217/chrome-headless-shell-mac-arm64/chrome-headless-shell
```

This is the same missing-Chromium root cause Task 1 recorded in its full
E2E run. Because no browser launched, **axe-core never ran**, so this
rerun produced **zero** critical or non-blocking (`serious`) findings —
there is no current automated evidence, positive or negative, for the
dashboard `color-contrast` or canvas `scrollable-region-focusable` findings
the original audit recorded from a working system-Chrome run. Per audit
scope, this failure was recorded as-is; `npx playwright install` was not
run and no product or test code was changed to make it pass.

Investigated the mismatch without acting on it: `node_modules/playwright-core/browsers.json`
in this checkout pins Chromium headless-shell revision `1217`, while
`~/Library/Caches/ms-playwright/` on this machine only has revision `1228`
installed — the two do not match. A system Google Chrome
(`/Applications/Google Chrome.app`, version `151.0.7922.76`) exists on this
machine, but `playwright.config.ts` targets the default Chromium channel
(not `channel: 'chrome'`), and no `.superpowers/sdd/playwright.chrome.config.ts`
fallback (used by the original audit) exists in this checkout — consistent
with the Task 1 rerun's documented environment limitation. No config file
was created or modified to work around this, since only
`docs/ux/frontend-ux-roadmap.md` was in scope for this task's commit.

## Step 4: Reconcile every `FND-*` finding

Added one `2026-08-07 status` / `Current evidence` / `Delta` block to each
of `FND-001` through `FND-006` in place (immediately after each finding's
existing `Impact/Frequency/Effort/Risk/Dependencies/Milestone` line), and
added one new finding, `FND-007`, in the "Accessibility and Keyboard UX"
section:

| ID | Rerun status | Summary |
|----|--------------|---------|
| `FND-001` | Confirmed | Shell not usable at 390 px on Dashboard (256 px sidebar / 134 px main pane) reproduced live; Canvas switcher did not clip in this specific rerun, but the shell-level defect is not resolved. No material change. |
| `FND-002` | **Changed** | All three overlay surfaces (Shortcuts, Command Palette, Notification detail) now have direct live Observed evidence of missing focus containment, upgrading Command Palette and the notification modal from Inferred to Observed. Problem/surfaces/proposed behavior/priority unchanged. |
| `FND-003` | Confirmed | Source re-read confirms `LoadingState`/`EmptyState`/`ErrorState` still have no `role`/`aria-live`. No material change. |
| `FND-004` | Confirmed | Re-grepped/re-read all five call sites; same onRetry asymmetry (Dashboard/EDA/Deployments have it, Registry/Evaluation/Segmentation don't). No material change. |
| `FND-005` | Confirmed | Re-read all four cited files (EncodingNode, AddSourceModal, EDASidebar, InferencePage); all four citations still hold verbatim. No material change. |
| `FND-006` | Confirmed | Re-drove Canvas→Experiments live; URL/history/aria-state behavior unchanged. No material change. |
| `FND-007` | **New** | `NotificationCenter` list rows nest an interactive Dismiss `<button>` inside another interactive row `<button>`, confirmed live (console warning, accessible-name concatenation, Tab landing on the nested control). Not present in the previous roadmap because the panel was previously exercised only empty. |

Also added a `FND-007` row to the `## Prioritized Findings Inventory` table
(placed after `FND-006`) so the new finding is represented consistently
across the document, and updated the `### Finding Status Summary` counts to
`New: 1`, `Changed: 1`, `Confirmed: 5`, `Resolved: 0` (total `7`, matching
`FND-001`–`FND-007`). Added a `### Task 2 — Shared Foundations Rerun`
section (grep evidence, live-walkthrough method/measurements, and the a11y
automation result) directly after the Finding Status Summary table, per the
brief's Step 1–3 evidence requirements.

Out of scope for this task (per `task-2-brief.md`'s explicit steps and the
design doc's execution model, which reserves ranking/milestone/validation
recalculation for the final cross-journey synthesis after all journey
tasks land): recalculating the normalized ranking, rebuilding Now/Next/Later
milestones, refreshing component-boundary recommendations, and refreshing
the validation matrix. `CAN-*`/`DAT-*`/`EXP-*`/`OPS-*` journey findings are
reconciled by their own dedicated tasks, not this one.

## Step 5: Commit shared-foundation evidence

```bash
$ git add docs/ux/frontend-ux-roadmap.md
$ git commit -m "docs: refresh shared frontend UX evidence" \
    -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
[075 3548b7e5] docs: refresh shared frontend UX evidence
 1 file changed, 262 insertions(+), 3 deletions(-)
```

`git show --stat HEAD` and `git status --porcelain` confirm only
`docs/ux/frontend-ux-roadmap.md` is in the commit; the pre-existing
`.superpowers/sdd/progress.md` modification remains unstaged.

## Self-review

- Re-read the full diff (`git diff --stat` then full content) end-to-end:
  the new `### Task 2 — Shared Foundations Rerun` section, all seven
  per-finding rerun blocks, the updated Finding Status Summary counts, and
  the new `FND-007` inventory row are internally consistent (status counts
  match the seven per-finding blocks exactly; `FND-007`'s
  Impact/Frequency/Effort/Risk/Milestone match between its full finding
  entry and its inventory-table row).
- Verified `git diff --check` reports no whitespace errors and fixed one
  stray typo character (`×and`) introduced while drafting, before
  committing.
- Confirmed evidence labels (`Observed`/`Measured`/`Inferred`) are used
  precisely: upgraded Command Palette and the notification modal from
  Inferred to Observed only where a live Tab-focus check was actually
  performed; left Inference's own shell-view behavior and the remaining
  keyboard-only form completions as Inferred since they were not
  independently re-driven this rerun.
- Confirmed no product code, dependencies, or test files were modified —
  `git status --porcelain` after cleanup shows only the two pre-existing
  tracked modifications (`progress.md`, unstaged) and the committed roadmap
  file.
- Confirmed the `npm run test:e2e -- e2e/a11y.spec.ts --project=chromium`
  failure was recorded verbatim (exact error text, exit code, and affected
  test names) and was not worked around by installing browsers, adding a
  fallback Playwright config, or modifying `playwright.config.ts`.
- Cleaned up all scratch artifacts generated while gathering evidence
  (`.playwright-mcp/`, `frontend/ml-canvas/canvas-390.png`,
  `frontend/ml-canvas/test-results/`, `.superpowers/sdd/a11y-rerun.log`,
  `.superpowers/sdd/vite-dev.log`) and stopped the Vite dev server process
  this session started (PID 21823/21807); a second, pre-existing Vite dev
  server (PID 18709/18692, started before this session) was left running
  since it was not started by this task.

## Concerns

- The required `npm run test:e2e -- e2e/a11y.spec.ts --project=chromium`
  command still cannot produce real axe-core results in this environment
  (missing Chromium `chrome-headless-shell` revision, mismatched against
  the cached revision on this machine). This means `FND-003`'s reconciliation
  and the a11y non-blocking findings the original audit reported
  (`color-contrast`, `scrollable-region-focusable`) have no current
  automated re-verification — only the source-level and interactive
  Observed evidence gathered manually via the Playwright MCP browser tool.
  A future environment fix (matching cached browser revision to the
  project's pinned revision, or restoring a system-Chrome fallback config)
  would let a later rerun close this gap.
- `FND-002`'s status is `Changed` (evidence upgraded from Inferred to
  Observed for two of three overlay surfaces) rather than `Confirmed`; this
  is a stronger evidentiary claim than the original audit could make and
  should be visible to whoever consolidates rankings in the final synthesis
  task, since it may affect confidence/priority framing even though the
  underlying proposed fix is unchanged.
- `FND-007` is a genuinely new, concretely reproduced defect
  (button-in-button nesting in `NotificationCenter`) that was not caught by
  the original audit because its notification panel was previously only
  exercised in the empty state. It's independent of the other `FND-*`
  findings and should be picked up by whichever milestone-assignment pass
  runs next; I proposed `Milestone: Next` given its Medium impact/Low risk,
  but did not touch the shared Now/Next/Later roadmap section since that
  rebuild is reserved for the final cross-journey synthesis task per the
  design doc.

## Task 2 review fix

Two review findings against commit `3548b7e5` were fixed in
`docs/ux/frontend-ux-roadmap.md` (product code untouched).

### Finding 1 — FND-001's 390 px Canvas evidence was false

The prior text claimed the Canvas view switcher
(`[data-testid="navbar-views"]`) "stayed fully inside the main content
pane with no overlap" at 390 px, including at 390 px itself, and that "no
clipping/overlap [was] detected" — contradicting `FND-001`'s own headline
claim that global navigation and Canvas view controls clip/overlap at
390 px.

**Validation method:** A live Vite dev server for `frontend/ml-canvas` was
already running on this machine (`http://localhost:5173`, confirmed via
`lsof -i :5173` and `curl -s -o /dev/null -w "%{http_code}\n"
http://localhost:5173/canvas` → `200`); no new server was started. Drove a
real, interactive Chromium browser against it with the Playwright MCP
browser tool: `playwright-browser_navigate` to `http://localhost:5173/canvas`,
`playwright-browser_resize` to `390×844`, then `playwright-browser_evaluate`
to measure `getBoundingClientRect()` for `[data-testid="navbar-views"]`,
the `<main>` pane, the Notifications bell, the switcher's three buttons, and
(in a second call) the "Read-only" toggle button, plus
`document.documentElement.scrollWidth`/`window.innerWidth`.

**Command/expression run (via `playwright-browser_evaluate`):**

```js
() => {
  const switcher = document.querySelector('[data-testid="navbar-views"]');
  const rect = switcher ? switcher.getBoundingClientRect() : null;
  const mainCandidates = Array.from(document.querySelectorAll('main'));
  const mainRects = mainCandidates.map(m => m.getBoundingClientRect());
  const bell = document.querySelector('[aria-label*="otification" i], [data-testid*="notification" i]');
  const bellRect = bell ? bell.getBoundingClientRect() : null;
  const buttons = switcher ? Array.from(switcher.querySelectorAll('button')).map(b => ({text: b.textContent, rect: b.getBoundingClientRect()})) : [];
  return { switcherRect: rect, mainRects, bellRect, buttons,
    scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth };
}
```

**Output (390×844 viewport, `/canvas`):**

```json
{
  "switcherRect": { "x": 50.71875, "right": 403.28125, "width": 352.5625, "top": 7.5, "bottom": 47.5 },
  "mainRects": [ { "x": 64, "right": 390, "width": 326 }, { "x": 64, "right": 390, "width": 326 } ],
  "bellRect": { "x": 342, "right": 378, "top": 9.5, "bottom": 45.5 },
  "buttons": [
    { "text": "Canvas", "rect": { "left": 54.71875, "right": 150.9375 } },
    { "text": "Experiments", "rect": { "left": 154.9375, "right": 285.140625 } },
    { "text": "Inference", "rect": { "left": 289.140625, "right": 399.28125, "top": 11.5, "bottom": 43.5 } }
  ],
  "scrollWidth": 390,
  "innerWidth": 390
}
```

A second `playwright-browser_evaluate` call querying `button, [role="switch"],
[aria-label]` filtered to elements whose text/aria-label matches
`/read.?only/i` returned the Read-only toggle at
`{ "rect": { "left": 298, "right": 334, "top": 15.5, "bottom": 39.5 } }`.

**Interpretation:** The switcher (`x=50.72–403.28`) extends `13.28px` beyond
both the `326px` main pane (`x=64–390`) and the `390px` viewport itself. The
Inference button (`x=289.14–399.28`, `y=11.5–43.5`) spatially overlaps the
Read-only toggle (`x=298–334`, `y=15.5–39.5`) and the Notifications bell
(`x=342–378`, `y=9.5–45.5`) — both toggle and bell fall entirely inside the
Inference button's x/y span. This is the reviewer-reproduced current
evidence used to correct the roadmap text; it strengthens (rather than
weakens) `FND-001`'s existing `Confirmed`/`No material change` conclusion,
since the switcher-overflow behavior the original finding already claimed is
now directly reproduced rather than contradicted.

Corrected two passages: the "Shared live walkthroughs" bullet under `Task 2
— Shared Foundations Rerun`, and `FND-001`'s own `Current evidence` bullet
in its detailed finding entry, both of which previously asserted no
clipping/overlap at 390 px.

### Finding 2 — FND-002 header/inventory not synced to its upgraded evidence

`FND-002`'s `2026-08-07 status` was `Changed` with `Current evidence`
documenting new, direct Observed evidence for two of its three overlay
surfaces (Command Palette and the notification detail modal, upgraded from
Inferred), but the finding's own bolded header still read "**FND-002 —
Inferred:** ..." and the Prioritized Findings Inventory table's `Evidence`
column for `FND-002` still read `Inferred`, contradicting the upgrade
recorded in the same finding's `Current evidence`/`Delta` text.

**Validation method:** `grep -n "FND-002" docs/ux/frontend-ux-roadmap.md`
before and after editing, to confirm the header line, the `Evidence` bullet,
the `Current evidence`/`Delta` bullets, and the inventory table row all use
a consistent `Observed` label while the remaining Experiments/Inference-view
scope is still explicitly called out as source-inferred (not independently
re-driven this rerun — only `/canvas` was exercised).

**Command run:**

```bash
grep -n "FND-002" docs/ux/frontend-ux-roadmap.md
```

**Output (after fix, relevant lines):**

```
663:- **FND-002 — Observed: shell overlays lack a shared focus-containment and
665:  - **Evidence:** **Observed** at 1440 px on `/canvas` for all three shell
...
2037:| FND-002 | Observed | Shell overlays lack a shared focus-containment and focus-return contract. | Canvas, Experiments, Inference overlays; shared Navbar | High | Occasional | S | Low | ModalShell focus helpers | Now |
```

Updated the header evidence label from `Inferred` to `Observed` and the
`Evidence` bullet to state Observed evidence for all three overlays at
1440 px on `/canvas`, while adding an explicit `Inferred` clause noting the
generalization to the Experiments/Inference shell views remains
source-inferred (not independently re-driven). Updated the inventory row's
`Evidence` cell from `Inferred` to `Observed`. The `Current evidence`/`Delta`
bullets (already correct) were left unchanged.

### Whitespace/format check

```bash
$ git diff --check docs/ux/frontend-ux-roadmap.md
$ echo "exit: $?"
exit: 0
```

No trailing-whitespace or conflict-marker errors.

### Commit

```
$ git add docs/ux/frontend-ux-roadmap.md
$ git commit -m "docs: fix task 2 review findings in frontend UX roadmap" \
    -m "Correct FND-001's 390px Canvas evidence to match reviewer-reproduced
measurements (navbar-views switcher extends beyond the 390px viewport/main
pane; Inference button overlaps the Read-only toggle and Notifications
bell), and sync FND-002's detailed finding header and Prioritized Findings
Inventory evidence cell from Inferred to Observed to match its upgraded live
evidence for Command Palette and the notification detail modal, while
retaining the remaining Experiments/Inference-view source-inferred scope." \
    -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
[075 830bd403] docs: fix task 2 review findings in frontend UX roadmap
 1 file changed, 34 insertions(+), 30 deletions(-)
```

`git show --stat 830bd403` confirms only `docs/ux/frontend-ux-roadmap.md`
is in the commit (34 insertions, 30 deletions). No product code, tests, or
dependencies were touched. The screenshot (`canvas-390-review.png`) and MCP
trace artifacts (`.playwright-mcp/`) generated while gathering evidence were
deleted after use and were never staged or committed.

### Changed lines (summary)

- `docs/ux/frontend-ux-roadmap.md` lines ~222–234 (Task 2 rerun narrative,
  "Shared live walkthroughs" `/canvas` 390 px bullet): replaced the false
  "stayed fully inside ... no overlap" / "did not reproduce ... overlap"
  claim with the measured overrun (`403.28px` right edge vs. `390px`
  pane/viewport) and the Inference/Read-only/Notifications-bell overlap.
- `docs/ux/frontend-ux-roadmap.md` lines ~796–808 (`FND-001` detailed
  finding, `Current evidence` bullet): same correction applied to the
  per-finding rerun evidence block.
- `docs/ux/frontend-ux-roadmap.md` lines ~663–674 (`FND-002` detailed
  finding header and `Evidence` bullet): header evidence label changed
  `Inferred` → `Observed`; `Evidence` bullet rewritten to state Observed
  evidence for all three overlays at 1440 px on `/canvas`, with an explicit
  Inferred clause for the untested Experiments/Inference views.
- `docs/ux/frontend-ux-roadmap.md` line ~2037 (Prioritized Findings
  Inventory table): `FND-002` row's `Evidence` cell changed `Inferred` →
  `Observed`.

### Self-review

- Re-drove the exact 390 px Canvas measurement live (not reused from the
  original Task 2 walkthrough) via the Playwright MCP browser tool against
  the already-running dev server, and used those numbers verbatim in both
  corrected passages.
- Confirmed the two corrected `FND-001` passages and the `FND-002`
  header/inventory now agree with each other and with the already-correct
  `Current evidence`/`Delta` text that was not touched.
- Confirmed `git diff --check` is clean and `git show --stat` on the fix
  commit shows only `docs/ux/frontend-ux-roadmap.md`.
- Deleted all scratch artifacts (`canvas-390-review.png`, `.playwright-mcp/`)
  generated while gathering evidence; none were staged or committed.
- Left the pre-existing unstaged `.superpowers/sdd/progress.md` modification
  untouched, consistent with the original Task 2 report's handling.

### Concerns

- The corrected evidence was gathered against the pre-existing Vite dev
  server process already running on this machine (PID 18709, started
  before this session) rather than a freshly started one; this matches the
  same dev server / URL / config the original Task 2 rerun used
  (`http://127.0.0.1:5173` via `playwright.config.ts`'s `webServer`), just
  reached via `localhost` because `curl` to `127.0.0.1:5173` returned no
  response in this shell while `localhost:5173` (which resolves via IPv6
  `::1`) succeeded — the server itself was unaffected.
- `FND-002`'s Experiments/Inference-view overlay behavior is still
  source-inferred, not independently re-driven; this scope gap is now
  explicitly stated in the finding's `Evidence` bullet rather than being
  ambiguous, but a future rerun should still independently exercise
  Shortcuts/Command Palette/Notification detail from those two views to
  fully resolve it.
