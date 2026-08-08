# Frontend UX Roadmap Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the complete Skyulf frontend UX audit and reconcile the existing roadmap with current measured, observed, and inferred evidence.

**Architecture:** Keep `docs/ux/frontend-ux-roadmap.md` as the canonical roadmap. Add a dated rerun delta, refresh each audit area in sequence, assign every existing finding a rerun status, then rebuild prioritization and validation from the refreshed evidence without changing product code.

**Tech Stack:** React 18, TypeScript, Vite, Tailwind CSS, Zustand, TanStack Query, React Router, Vitest, Testing Library, Playwright, axe-core.

## Global Constraints

- Modify only `docs/ux/frontend-ux-roadmap.md` during audit tasks.
- Do not modify frontend product code, backend code, visual styling, or dependencies.
- Cover Shared Foundations, Canvas, Data/EDA, Experiments/Inference, and Operations with fresh evidence.
- Repeat live checks at widths `1440`, `1024`, `768`, and `390`.
- Keep `Observed`, `Measured`, and `Inferred` evidence labels separate from rerun status.
- Assign each existing finding exactly one rerun status: `New`, `Changed`, `Confirmed`, or `Resolved`.
- Preserve resolved findings in a historical section; do not silently delete them.
- Record baseline failures as evidence instead of fixing them.
- Keep `.github/workflows/dependency-review.yml` and all unrelated changes outside audit commits.
- Do not add dependencies.
- Every commit must include `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

---

### Task 1: Establish the Rerun Baseline

**Files:**
- Modify: `docs/ux/frontend-ux-roadmap.md`
- Read: `docs/superpowers/specs/2026-08-07-frontend-ux-roadmap-rerun-design.md`
- Read: `frontend/ml-canvas/package.json`
- Read: `frontend/ml-canvas/src/App.tsx`
- Read: `frontend/ml-canvas/src/components/Layout.tsx`
- Read: `frontend/ml-canvas/playwright.config.ts`
- Read: `frontend/ml-canvas/e2e/routes.spec.ts`

**Interfaces:**
- Consumes: Existing roadmap structure and historical baseline.
- Produces: A dated rerun section, current engineering baseline, route inventory, and status vocabulary used by Tasks 2-8.

- [ ] **Step 1: Add the rerun scaffold**

Add this section immediately after `## Executive Summary`:

```markdown
## 2026-08-07 Audit Rerun

### Delta Summary

### Current Engineering Baseline

### Current Route and Navigation Baseline

### Finding Status Summary

| Status | Count | Meaning |
|--------|-------|---------|
| New | 0 | Not present in the previous roadmap. |
| Changed | 0 | Evidence, scope, priority, or proposed behavior materially changed. |
| Confirmed | 0 | Current evidence still supports the finding without material change. |
| Resolved | 0 | Current evidence demonstrates that the prior user problem no longer occurs. |
```

- [ ] **Step 2: Run the complete current frontend baseline**

Run from `frontend/ml-canvas/`:

```bash
npm run lint
npx tsc --project tsconfig.json --noEmit
npm run build
npm run test -- --reporter=dot
npm run test:e2e -- --project=chromium
npm run size-check
```

Record exit status, test counts, build chunk sizes, bundle-size results, warnings,
and failures under `### Current Engineering Baseline`. Do not alter source code
when a command fails.

- [ ] **Step 3: Refresh route and navigation metadata**

Record the current route, lazy-loading, sidebar, alert-badge, and E2E coverage
state for:

```text
/
/jobs
/data
/eda
/drift
/canvas
/registry
/deployments
/errors
/slow-nodes
/audit
```

- [ ] **Step 4: Verify the rerun scaffold**

Run:

```bash
grep -n "^## 2026-08-07 Audit Rerun$" docs/ux/frontend-ux-roadmap.md
grep -n "^### Current Engineering Baseline$" docs/ux/frontend-ux-roadmap.md
grep -n "^### Finding Status Summary$" docs/ux/frontend-ux-roadmap.md
git diff --check
```

Expected: each heading appears once and `git diff --check` exits `0`.

- [ ] **Step 5: Commit the baseline**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: refresh frontend UX audit baseline" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Refresh Shared Foundations

**Files:**
- Modify: `docs/ux/frontend-ux-roadmap.md`
- Read: `frontend/ml-canvas/src/components/Layout.tsx`
- Read: `frontend/ml-canvas/src/components/shared/`
- Read: `frontend/ml-canvas/src/components/ui/`
- Read: `frontend/ml-canvas/src/components/layout/CommandPalette.tsx`
- Read: `frontend/ml-canvas/src/components/layout/NotificationCenter.tsx`
- Read: `frontend/ml-canvas/src/components/layout/ShortcutsOverlay.tsx`
- Read: `frontend/ml-canvas/src/core/toast.ts`
- Read: `frontend/ml-canvas/src/core/utils/a11y.ts`

**Interfaces:**
- Consumes: Current baseline and existing `FND-*` findings.
- Produces: Current shared-foundation evidence and rerun statuses for every `FND-*` finding.

- [ ] **Step 1: Reinspect shared-state usage**

Run from `frontend/ml-canvas/`:

```bash
grep -RInE "LoadingState|EmptyState|ErrorState|PageSkeleton|toast\\.|disabled=" src \
  --include="*.ts" --include="*.tsx"
```

Compare first-load, empty, failure, retry, success, warning, and unavailable
action behavior across all journeys.

- [ ] **Step 2: Repeat live shared-foundation walkthroughs**

At `1440`, `1024`, `768`, and `390`, verify current page orientation, collapsed
navigation naming, target size, browser history, overlay focus, keyboard
operation, form labels, validation timing, responsive overflow, terminology,
and perceived delays.

- [ ] **Step 3: Repeat accessibility automation**

Run:

```bash
cd frontend/ml-canvas
npm run test:e2e -- e2e/a11y.spec.ts --project=chromium
```

Record current failures and non-blocking findings exactly.

- [ ] **Step 4: Reconcile every `FND-*` finding**

For each existing finding, add a dated rerun note with:

```markdown
- **2026-08-07 status:** Confirmed
- **Current evidence:** Observed, Measured, or Inferred evidence.
- **Delta:** Exact change from the previous audit, or `No material change`.
```

Use `Changed` or `Resolved` only when current evidence demonstrates it. Add new
findings using the next available `FND-*` ID.

- [ ] **Step 5: Commit shared-foundation evidence**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: refresh shared frontend UX evidence" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Refresh the Canvas Journey

**Files:**
- Modify: `docs/ux/frontend-ux-roadmap.md`
- Read: `frontend/ml-canvas/src/pages/CanvasPage.tsx`
- Read: `frontend/ml-canvas/src/components/layout/`
- Read: `frontend/ml-canvas/src/components/canvas/`
- Read: `frontend/ml-canvas/src/core/store/useGraphStore.ts`
- Read: `frontend/ml-canvas/src/core/hooks/useKeyboardShortcuts.ts`
- Read: `frontend/ml-canvas/src/core/hooks/useCanvasAutoSave.ts`
- Read: `frontend/ml-canvas/src/core/utils/pipelineLeakageValidation.ts`
- Read: `frontend/ml-canvas/src/modules/nodes/`

**Interfaces:**
- Consumes: Refreshed shared-foundation evidence and existing `CAN-*` findings.
- Produces: Current Canvas evidence and rerun statuses for every `CAN-*` finding.

- [ ] **Step 1: Repeat pipeline creation**

Walk through:

```text
Open Canvas → add dataset → add preprocessing node → configure node →
connect nodes → add split → add training node → validate → save → run
```

Record discoverability, selection, connection, validation, action availability,
and recovery behavior at all four target widths.

- [ ] **Step 2: Repeat diagnosis and recovery checks**

Exercise invalid connections, missing configuration, leakage validation, run
failure, restore/version loading, undo/redo, autosave recovery, results
navigation, node warnings, keyboard shortcuts, and narrow pane behavior.

- [ ] **Step 3: Recompare node configuration forms**

Inspect current Dataset, Encoding, Feature Generation, Feature Selection,
Training, Ensemble, Segmentation, and Data Preview forms. Record only
user-visible consistency or regression evidence.

- [ ] **Step 4: Reconcile every `CAN-*` finding**

Add the dated status/evidence/delta block from Task 2 to every `CAN-*` finding.
Add new Canvas findings using the next available ID and reference shared `FND-*`
dependencies where applicable.

- [ ] **Step 5: Commit Canvas evidence**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: refresh canvas UX evidence" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Refresh Data and EDA Journeys

**Files:**
- Modify: `docs/ux/frontend-ux-roadmap.md`
- Read: `frontend/ml-canvas/src/pages/DataSources.tsx`
- Read: `frontend/ml-canvas/src/components/data/`
- Read: `frontend/ml-canvas/src/pages/EDAPage.tsx`
- Read: `frontend/ml-canvas/src/components/eda/`
- Read: `frontend/ml-canvas/src/core/store/useEDAStore.ts`
- Read: `frontend/ml-canvas/src/core/hooks/useEdaJobs.ts`
- Read: `frontend/ml-canvas/src/core/api/datasets.ts`
- Read: `frontend/ml-canvas/src/core/api/eda.ts`

**Interfaces:**
- Consumes: Refreshed shared evidence and existing `DAT-*` findings.
- Produces: Current Data/EDA evidence and rerun statuses for every `DAT-*` finding.

- [ ] **Step 1: Repeat data-source onboarding**

Exercise source creation, validation failure, connection failure, ingestion
progress, preview, empty datasets, and navigation to Canvas at all target widths.

- [ ] **Step 2: Repeat EDA workflows**

Exercise dataset selection, job creation, progress, failure, empty results, tabs,
filters, charts, tables, downloads, and return navigation. Verify source,
analysis, variable, and target context remain distinguishable.

- [ ] **Step 3: Recheck visualization usability**

Inspect legends, axes, tooltips, color meaning, dark mode, overflow, chart
alternatives, no-data states, and explanatory copy on desktop and narrow widths.

- [ ] **Step 4: Reconcile every `DAT-*` finding**

Add dated status/evidence/delta blocks, add new IDs sequentially, and preserve
resolved findings in the historical section.

- [ ] **Step 5: Commit Data/EDA evidence**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: refresh data and EDA UX evidence" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Refresh Experiments and Inference

**Files:**
- Modify: `docs/ux/frontend-ux-roadmap.md`
- Read: `frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx`
- Read: `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/`
- Read: `frontend/ml-canvas/src/components/pages/experiments/`
- Read: `frontend/ml-canvas/src/components/pages/InferencePage.tsx`
- Read: `frontend/ml-canvas/src/core/hooks/useJobPolling.ts`
- Read: `frontend/ml-canvas/src/core/hooks/useNodeJobSummaries.ts`
- Read: `frontend/ml-canvas/src/core/api/jobs.ts`
- Read: `frontend/ml-canvas/src/core/api/thresholdTuning.ts`

**Interfaces:**
- Consumes: Refreshed shared evidence and existing `EXP-*` findings.
- Produces: Current Experiments/Inference evidence and statuses for every `EXP-*` finding.

- [ ] **Step 1: Repeat experiment comparison**

Exercise job selection, filters, task type, split, metric comparison, pipeline
diff, feature importance, confusion matrices, regression charts, segmentation,
SHAP, and threshold tuning.

- [ ] **Step 2: Repeat inference**

Exercise model/job selection, schema display, input methods, invalid input,
execution, results, export, reset, retry, and long-running/failure states.

- [ ] **Step 3: Reassess component-boundary evidence**

Review `InferencePage.tsx`, `ExperimentsPage.tsx`,
`ClassificationChartsForSplit.tsx`, and `EvaluationView.tsx`. Retain a boundary
recommendation only when current user-facing risk supports it.

- [ ] **Step 4: Reconcile every `EXP-*` finding**

Add dated status/evidence/delta blocks and sequential new IDs.

- [ ] **Step 5: Commit Experiments/Inference evidence**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: refresh experiments and inference UX evidence" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: Refresh Operations Journeys

**Files:**
- Modify: `docs/ux/frontend-ux-roadmap.md`
- Read: `frontend/ml-canvas/src/pages/Jobs.tsx`
- Read: `frontend/ml-canvas/src/components/panels/`
- Read: `frontend/ml-canvas/src/pages/ModelRegistry.tsx`
- Read: `frontend/ml-canvas/src/components/pages/DeploymentsPage.tsx`
- Read: `frontend/ml-canvas/src/pages/DataDriftPage.tsx`
- Read: `frontend/ml-canvas/src/pages/ErrorLogPage.tsx`
- Read: `frontend/ml-canvas/src/pages/SlowNodesPage.tsx`
- Read: `frontend/ml-canvas/src/pages/AuditLogPage.tsx`
- Read: `frontend/ml-canvas/src/core/api/monitoring.ts`
- Read: `frontend/ml-canvas/src/core/api/deployment.ts`
- Read: `frontend/ml-canvas/src/core/api/registry.ts`

**Interfaces:**
- Consumes: Refreshed shared evidence and existing `OPS-*` findings.
- Produces: Current Operations evidence and statuses for every `OPS-*` finding.

- [ ] **Step 1: Repeat jobs and job-detail workflows**

Exercise filters, statuses, pagination/history, selection, details, cancellation
or retry where available, failures, progress, and navigation to origin context.

- [ ] **Step 2: Repeat registry and deployment workflows**

Exercise list, search, details, deployment actions, confirmation, feedback,
active state, and history. Verify training-job, model, and deployment lineage.

- [ ] **Step 3: Repeat monitoring investigations**

Exercise drift, errors, slow nodes, and audit logs. Verify severity, time range,
affected resource, next action, filters, details, and supported deep links.

- [ ] **Step 4: Reconcile every `OPS-*` finding**

Add dated status/evidence/delta blocks and sequential new IDs. Do not infer
cross-page links that current APIs cannot support.

- [ ] **Step 5: Commit Operations evidence**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: refresh operations UX evidence" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: Reconcile and Reprioritize the Roadmap

**Files:**
- Modify: `docs/ux/frontend-ux-roadmap.md`

**Interfaces:**
- Consumes: All refreshed `FND-*`, `CAN-*`, `DAT-*`, `EXP-*`, and `OPS-*` evidence.
- Produces: Current status summary, inventory, ranking, milestones, component recommendations, and validation matrix.

- [ ] **Step 1: Count and summarize rerun statuses**

Run:

```bash
for status in New Changed Confirmed Resolved; do
  printf "%s " "$status"
  grep -c "\\*\\*2026-08-07 status:\\*\\* $status" docs/ux/frontend-ux-roadmap.md
done
```

Update `### Finding Status Summary` with the exact counts.

- [ ] **Step 2: Rebuild normalized ranking**

Rank current findings by impact, frequency, journeys improved, accessibility or
data-loss risk, effort, then regression risk/dependencies. Use only:

```text
Impact: High | Medium | Low
Frequency: Frequent | Occasional | Rare
Effort: S | M | L
Risk: Low | Medium | High
Milestone: Now | Next | Later
```

- [ ] **Step 3: Rebuild milestones and dependencies**

Keep the smallest dependency-complete high-impact set in `Now`, broader
journey-level work in `Next`, and lower-frequency enhancements in `Later`.
Never place a dependency after its consumer.

- [ ] **Step 4: Refresh recommendations and validation**

Update the executive summary, component-boundary recommendations, inventory,
and validation matrix. Every current Now and Next item must have measurable
acceptance criteria plus automated/manual, responsive, keyboard, and
screen-reader validation where relevant.

- [ ] **Step 5: Commit reconciled priorities**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: reprioritize refreshed frontend UX roadmap" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 8: Validate the Rerun

**Files:**
- Modify: `docs/ux/frontend-ux-roadmap.md`
- Read: `docs/superpowers/specs/2026-08-07-frontend-ux-roadmap-rerun-design.md`

**Interfaces:**
- Consumes: Completed refreshed roadmap.
- Produces: A self-consistent, implementation-ready audit delta.

- [ ] **Step 1: Scan for placeholders and vague language**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

text = Path("docs/ux/frontend-ux-roadmap.md").read_text()
patterns = [
    "TB" + "D",
    "TO" + "DO",
    "FIX" + "ME",
    "implement " + "later",
    "appropriate error " + "handling",
    "similar " + "to",
    "may" + "be",
    "poss" + "ibly",
]
matches = [pattern for pattern in patterns if pattern.lower() in text.lower()]
if matches:
    raise SystemExit(f"Incomplete language found: {matches}")
PY
```

- [ ] **Step 2: Verify finding coverage**

Run:

```bash
grep -oE "FND-[0-9]{3}|CAN-[0-9]{3}|DAT-[0-9]{3}|EXP-[0-9]{3}|OPS-[0-9]{3}" \
  docs/ux/frontend-ux-roadmap.md | sort | uniq -c
```

Investigate every current finding lacking detail, inventory, roadmap/exclusion,
or rerun-status coverage.

- [ ] **Step 3: Re-run the complete frontend baseline**

Run from `frontend/ml-canvas/`:

```bash
npm run lint
npx tsc --project tsconfig.json --noEmit
npm run build
npm run test -- --reporter=dot
npm run test:e2e -- --project=chromium
npm run size-check
```

Record final results and explain any difference from Task 1 without modifying
product code.

- [ ] **Step 4: Verify repository scope**

Run:

```bash
git diff --check
git status --short
git --no-pager diff "$(git merge-base myfork/075 HEAD)"..HEAD --stat
```

Expected: audit commits modify only `docs/ux/frontend-ux-roadmap.md`; the design
and plan commits modify only their corresponding documents.

- [ ] **Step 5: Commit final corrections**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: finalize frontend UX audit rerun" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```
