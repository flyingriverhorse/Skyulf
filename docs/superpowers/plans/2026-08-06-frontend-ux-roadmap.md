# Frontend UX Roadmap Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an evidence-based, prioritized UX roadmap for every major Skyulf frontend journey.

**Architecture:** Build one version-controlled roadmap document incrementally. Audit shared UX foundations first, then the Canvas, Data/EDA, Experiments/Inference, and Operations journeys; finish by deduplicating findings into a dependency-aware Now/Next/Later plan. Record direct UI observations separately from code-inferred regression risks.

**Tech Stack:** React 18, TypeScript, Vite, Tailwind CSS, Zustand, TanStack Query, React Router, Vitest, Testing Library, Playwright, axe-core.

## Global Constraints

- Cover Canvas, Data/EDA, Experiments/Inference, and Operations equally.
- Prioritize shared improvements that benefit multiple pages.
- Do not implement visual restyling, backend features, or product code changes during the audit.
- Do not recommend broad refactors without a defined user-facing benefit.
- Include backend/frontend contract issues only when they create confusing, unavailable, invalid, or silently ineffective UI behavior.
- Include performance issues only when users perceive loading delays, interaction latency, slow charts, or canvas responsiveness.
- Every finding must identify its evidence as `Observed` or `Inferred`.
- Every roadmap item must include the user problem, affected journeys, proposed behavior, acceptance criteria, validation method, effort, risk, dependencies, and milestone.
- Do not add dependencies.

---

## File Structure

**Create:**

- `docs/ux/frontend-ux-roadmap.md` — the complete audit, findings inventory, prioritization, phased roadmap, component-boundary recommendations, and validation matrix.

**Read-only audit sources:**

- `frontend/ml-canvas/src/App.tsx` — top-level route structure and lazy loading.
- `frontend/ml-canvas/src/components/Layout.tsx` — global navigation and application shell.
- `frontend/ml-canvas/src/components/shared/` — shared loading, empty, error, modal, confirmation, status, skeleton, and virtual-list patterns.
- `frontend/ml-canvas/src/components/ui/` — base controls and tooltips.
- `frontend/ml-canvas/src/components/layout/` — canvas shell, toolbar, panels, command palette, notifications, shortcuts, and responsive layout behavior.
- `frontend/ml-canvas/src/components/canvas/` — canvas interaction and node/edge feedback.
- `frontend/ml-canvas/src/modules/nodes/` — node configuration forms and validation UX.
- `frontend/ml-canvas/src/pages/DataSources.tsx` and `frontend/ml-canvas/src/pages/EDAPage.tsx` — data and EDA workflows.
- `frontend/ml-canvas/src/components/eda/` — EDA navigation, visualizations, and result states.
- `frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx` and `frontend/ml-canvas/src/components/pages/InferencePage.tsx` — experiments and inference workflows.
- `frontend/ml-canvas/src/pages/Jobs.tsx`, `frontend/ml-canvas/src/pages/ModelRegistry.tsx`, `frontend/ml-canvas/src/components/pages/DeploymentsPage.tsx`, `frontend/ml-canvas/src/pages/DataDriftPage.tsx`, `frontend/ml-canvas/src/pages/ErrorLogPage.tsx`, `frontend/ml-canvas/src/pages/SlowNodesPage.tsx`, and `frontend/ml-canvas/src/pages/AuditLogPage.tsx` — operational workflows.
- `frontend/ml-canvas/e2e/` and frontend `*.test.ts(x)` files — current UX regression coverage.

**No product source file is modified by this plan.**

---

### Task 1: Establish the Audit Baseline and Report Scaffold

**Files:**
- Create: `docs/ux/frontend-ux-roadmap.md`
- Read: `frontend/ml-canvas/package.json`
- Read: `frontend/ml-canvas/src/App.tsx`
- Read: `frontend/ml-canvas/src/components/Layout.tsx`
- Read: `frontend/ml-canvas/playwright.config.ts`
- Read: `frontend/ml-canvas/e2e/routes.spec.ts`

**Interfaces:**
- Consumes: approved design at `docs/superpowers/specs/2026-08-06-frontend-ux-roadmap-design.md`.
- Produces: a stable report structure and baseline evidence used by every later task.

- [ ] **Step 1: Create the report with the final section structure**

Create `docs/ux/frontend-ux-roadmap.md` with this exact skeleton:

```markdown
# Frontend UX Roadmap

## Executive Summary

## Method and Evidence

### Evidence Labels
- **Observed:** Reproduced in the running interface.
- **Inferred:** Identified from code or test structure as a UX regression risk.

### Baseline

## Shared Foundations

### Navigation and Orientation
### Async and Feedback States
### Forms and Validation
### Accessibility and Keyboard UX
### Responsive Behavior
### Terminology and Visual Hierarchy
### Perceived Performance

## Journey Findings

### Canvas
### Data and EDA
### Experiments and Inference
### Operations

## Prioritized Findings Inventory

| ID | Evidence | User problem | Surfaces | Impact | Frequency | Effort | Risk | Dependencies | Milestone |
|----|----------|--------------|----------|--------|-----------|--------|------|--------------|-----------|

## Component-Boundary Recommendations

## Now / Next / Later Roadmap

### Now
### Next
### Later

## Validation Matrix

| Roadmap item | Acceptance criteria | Automated validation | Manual validation | Responsive coverage | Accessibility coverage |
|--------------|---------------------|----------------------|-------------------|---------------------|------------------------|
```

- [ ] **Step 2: Capture the engineering baseline**

Run from `frontend/ml-canvas/`:

```bash
npm run lint
npx tsc --project tsconfig.json --noEmit
npm run build
npm run test -- --reporter=dot
npm run test:e2e -- --project=chromium
npm run size-check
```

Expected: each command exits `0`. Record command status, build chunk sizes, bundle-size result, unit-test count, and E2E-test count under `### Baseline`. If a command fails, record the exact failure as pre-existing evidence; do not fix it during this audit.

- [ ] **Step 3: Record the route and navigation baseline**

Document these top-level routes from `src/App.tsx` and `src/components/Layout.tsx`:

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

Record which routes are lazy-loaded, which routes collapse the sidebar, which routes expose alert badges, and which routes are currently covered by `e2e/routes.spec.ts`.

- [ ] **Step 4: Verify the scaffold is complete**

Run:

```bash
grep -n "^## " ../../docs/ux/frontend-ux-roadmap.md
grep -n "^### " ../../docs/ux/frontend-ux-roadmap.md
```

Expected: every heading in Step 1 appears exactly once.

- [ ] **Step 5: Commit the baseline**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: establish frontend UX audit baseline"
```

---

### Task 2: Audit Shared UX Foundations

**Files:**
- Modify: `docs/ux/frontend-ux-roadmap.md`
- Read: `frontend/ml-canvas/src/components/Layout.tsx`
- Read: `frontend/ml-canvas/src/components/shared/LoadingState.tsx`
- Read: `frontend/ml-canvas/src/components/shared/EmptyState.tsx`
- Read: `frontend/ml-canvas/src/components/shared/ErrorState.tsx`
- Read: `frontend/ml-canvas/src/components/shared/ModalShell.tsx`
- Read: `frontend/ml-canvas/src/components/shared/ConfirmDialog.tsx`
- Read: `frontend/ml-canvas/src/components/shared/StatusBadge.tsx`
- Read: `frontend/ml-canvas/src/components/layout/CommandPalette.tsx`
- Read: `frontend/ml-canvas/src/components/layout/NotificationCenter.tsx`
- Read: `frontend/ml-canvas/src/components/layout/ShortcutsOverlay.tsx`
- Read: `frontend/ml-canvas/src/core/toast.ts`
- Read: `frontend/ml-canvas/src/core/utils/a11y.ts`

**Interfaces:**
- Consumes: report scaffold and baseline from Task 1.
- Produces: shared-foundation findings with IDs `FND-001`, `FND-002`, and upward.

- [ ] **Step 1: Audit navigation and orientation**

Inspect the global shell and run the application at widths `1440`, `1024`, `768`, and `390` pixels. For every top-level route, record:

```text
- Can the user identify the current page?
- Can the user predict where each navigation item leads?
- Does collapsed navigation retain an accessible name and usable target size?
- Is the page hierarchy clear without relying only on color?
- Does browser Back/Forward preserve useful context?
```

Add only evidenced findings under `### Navigation and Orientation`.

- [ ] **Step 2: Audit loading, empty, error, warning, success, and disabled states**

Search usage of shared states:

```bash
rg -n "LoadingState|EmptyState|ErrorState|PageSkeleton|toast\\.|disabled=" src --glob "*.{ts,tsx}"
```

For each top-level journey, compare the first load, empty data, API failure, retry, success, and unavailable-action behavior. Record inconsistent or missing states under `### Async and Feedback States`.

- [ ] **Step 3: Audit forms and validation**

Inspect shared controls and representative node forms:

```text
src/components/ui/input.tsx
src/components/ui/button.tsx
src/components/ui/tooltip.tsx
src/modules/nodes/modeling/TrainingSettings.tsx
src/modules/nodes/modeling/EnsembleSettings.tsx
src/modules/nodes/processing/EncodingNode.tsx
src/modules/nodes/processing/FeatureGenerationNode.tsx
src/modules/nodes/processing/FeatureSelectionNode.tsx
```

Record label consistency, required-state communication, defaults, validation timing, error placement, help text, destructive-action confirmation, and keyboard submission behavior under `### Forms and Validation`.

- [ ] **Step 4: Audit accessibility and keyboard UX**

Run:

```bash
npm run test:e2e -- e2e/a11y.spec.ts --project=chromium
```

Then manually verify keyboard operation for global navigation, command palette, dialogs, menus, tabs, forms, canvas controls, and focus return after overlays close. Record findings under `### Accessibility and Keyboard UX`.

- [ ] **Step 5: Audit responsive behavior, terminology, hierarchy, and perceived performance**

At each target width, inspect overflow, clipping, content order, minimum target size, chart readability, table usability, and modal fit. Compare repeated labels for the same concepts, including jobs, runs, training, tuning, pipelines, nodes, data sources, and datasets. Record route and interaction delays that are visible to users; do not list raw code complexity as a performance finding.

- [ ] **Step 6: Add normalized findings to the inventory**

For every shared-foundation finding, add one inventory row using:

```markdown
| FND-001 | Observed | Concise user problem | Affected routes/components | High/Medium/Low | Frequent/Occasional/Rare | S/M/L | Low/Medium/High | IDs or None | Now/Next/Later |
```

Each finding's detailed text must state the proposed behavior, acceptance criteria, and validation method.

- [ ] **Step 7: Commit shared-foundation findings**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: audit shared frontend UX foundations"
```

---

### Task 3: Audit the Canvas Journey

**Files:**
- Modify: `docs/ux/frontend-ux-roadmap.md`
- Read: `frontend/ml-canvas/src/pages/CanvasPage.tsx`
- Read: `frontend/ml-canvas/src/components/layout/MainLayout.tsx`
- Read: `frontend/ml-canvas/src/components/layout/Toolbar.tsx`
- Read: `frontend/ml-canvas/src/components/layout/Sidebar.tsx`
- Read: `frontend/ml-canvas/src/components/layout/PropertiesPanel.tsx`
- Read: `frontend/ml-canvas/src/components/layout/ResultsPanel.tsx`
- Read: `frontend/ml-canvas/src/components/canvas/FlowCanvas.tsx`
- Read: `frontend/ml-canvas/src/components/canvas/CustomNodeWrapper.tsx`
- Read: `frontend/ml-canvas/src/components/canvas/CustomEdge.tsx`
- Read: `frontend/ml-canvas/src/core/store/useGraphStore.ts`
- Read: `frontend/ml-canvas/src/core/hooks/useKeyboardShortcuts.ts`
- Read: `frontend/ml-canvas/src/core/hooks/useCanvasAutoSave.ts`
- Read: `frontend/ml-canvas/src/core/utils/pipelineLeakageValidation.ts`
- Read: `frontend/ml-canvas/src/modules/nodes/`

**Interfaces:**
- Consumes: shared-foundation criteria and `FND-*` findings.
- Produces: journey findings `CAN-001` and upward; references shared findings instead of duplicating them.

- [ ] **Step 1: Walk through pipeline creation**

Using mocked or local data, complete this workflow:

```text
Open Canvas → add dataset → add preprocessing node → configure node →
connect nodes → add split → add training node → validate → save → run
```

Record discoverability, connection affordances, configuration feedback, validation timing, action availability, and recovery from mistakes.

- [ ] **Step 2: Walk through pipeline diagnosis and recovery**

Exercise invalid connections, missing required configuration, leakage validation, run failure, restore/version loading, undo/redo, autosave recovery, results-panel navigation, and node warnings. Record whether each failure explains the cause, identifies the affected node, and offers a next action.

- [ ] **Step 3: Audit canvas accessibility and responsive behavior**

Verify keyboard shortcuts are discoverable, focus is visible, non-pointer users can reach controls, collapsed panels remain understandable, and narrow layouts do not hide required actions. Classify React Flow limitations separately from Skyulf-controlled UX.

- [ ] **Step 4: Audit node-configuration consistency**

Compare at least one node from each category:

```text
Data: DatasetNode
Processing: EncodingNode, FeatureGenerationNode, FeatureSelectionNode
Modeling: TrainingSettings, EnsembleSettings, SegmentationSettings
Inspection: DataPreviewNode
```

Record inconsistencies in section layout, labels, defaults, help, validation, column selection, advanced settings, and apply/reset behavior.

- [ ] **Step 5: Add Canvas findings and acceptance criteria**

Add detailed findings under `### Canvas` and inventory rows using `CAN-*`. Every row must either reference a `FND-*` dependency or explain why the issue is Canvas-specific.

- [ ] **Step 6: Commit the Canvas audit**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: audit canvas user journey"
```

---

### Task 4: Audit Data and EDA Journeys

**Files:**
- Modify: `docs/ux/frontend-ux-roadmap.md`
- Read: `frontend/ml-canvas/src/pages/DataSources.tsx`
- Read: `frontend/ml-canvas/src/components/data/AddSourceModal.tsx`
- Read: `frontend/ml-canvas/src/components/data/DatasetPreviewModal.tsx`
- Read: `frontend/ml-canvas/src/components/data/IngestionJobsModal.tsx`
- Read: `frontend/ml-canvas/src/components/data/PipelineVersionsModal.tsx`
- Read: `frontend/ml-canvas/src/pages/EDAPage.tsx`
- Read: `frontend/ml-canvas/src/components/eda/`
- Read: `frontend/ml-canvas/src/core/store/useEDAStore.ts`
- Read: `frontend/ml-canvas/src/core/hooks/useEdaJobs.ts`
- Read: `frontend/ml-canvas/src/core/api/datasets.ts`
- Read: `frontend/ml-canvas/src/core/api/eda.ts`

**Interfaces:**
- Consumes: shared-foundation criteria and findings.
- Produces: findings `DAT-001` and upward.

- [ ] **Step 1: Walk through data-source onboarding**

Exercise add source, validation failure, connection/API failure, ingestion progress, dataset preview, empty datasets, and navigation from a dataset to Canvas. Record whether terminology and progress communication remain consistent.

- [ ] **Step 2: Walk through EDA**

Exercise dataset selection, EDA job creation, loading/progress, empty results, failed analysis, tab navigation, filters, charts, tables, downloads, and return navigation. Verify that users can distinguish source data, analysis state, selected variable, and selected target.

- [ ] **Step 3: Audit visualization usability**

At desktop and narrow widths, inspect legends, axes, tooltips, color meaning, dark mode, overflow, chart alternatives, no-data states, and explanatory copy. Flag visualizations that require domain knowledge without nearby interpretation.

- [ ] **Step 4: Add Data/EDA findings**

Add detailed findings under `### Data and EDA` and inventory rows using `DAT-*`. Reference shared findings where a common component or pattern is the root cause.

- [ ] **Step 5: Commit the Data/EDA audit**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: audit data and EDA journeys"
```

---

### Task 5: Audit Experiments and Inference Journeys

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
- Consumes: shared-foundation criteria and findings.
- Produces: findings `EXP-001` and upward.

- [ ] **Step 1: Walk through experiment selection and comparison**

Exercise job selection, filters, task-type switching, split selection, metric comparison, pipeline diff, feature importance, confusion matrices, regression charts, segmentation results, and SHAP views. Record whether users retain context while switching jobs, tabs, classes, splits, and comparison modes.

- [ ] **Step 2: Audit interpretation and decision support**

Check whether metric names, units, directionality, missing values, warnings, threshold tuning, and explainability views communicate what users should conclude or do next. Distinguish missing product guidance from purely visual issues.

- [ ] **Step 3: Walk through inference**

Exercise model/job selection, schema display, input method, validation failure, execution, results, export, reset, and retry. Record whether required input shape, unsupported values, long-running work, and failures are actionable.

- [ ] **Step 4: Identify oversized-component UX risks**

Review `InferencePage.tsx`, `ExperimentsPage.tsx`, `ClassificationChartsForSplit.tsx`, and `EvaluationView.tsx`. Add a component-boundary recommendation only when the file combines independently changing UX responsibilities and the split would make states or behavior safer to test.

- [ ] **Step 5: Add Experiments/Inference findings**

Add detailed findings under `### Experiments and Inference` and inventory rows using `EXP-*`.

- [ ] **Step 6: Commit the Experiments/Inference audit**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: audit experiments and inference journeys"
```

---

### Task 6: Audit Operations Journeys

**Files:**
- Modify: `docs/ux/frontend-ux-roadmap.md`
- Read: `frontend/ml-canvas/src/pages/Jobs.tsx`
- Read: `frontend/ml-canvas/src/components/panels/JobsDrawer.tsx`
- Read: `frontend/ml-canvas/src/components/panels/jobs/`
- Read: `frontend/ml-canvas/src/pages/ModelRegistry.tsx`
- Read: `frontend/ml-canvas/src/components/pages/DeploymentsPage.tsx`
- Read: `frontend/ml-canvas/src/pages/DataDriftPage.tsx`
- Read: `frontend/ml-canvas/src/pages/drift/`
- Read: `frontend/ml-canvas/src/pages/ErrorLogPage.tsx`
- Read: `frontend/ml-canvas/src/pages/SlowNodesPage.tsx`
- Read: `frontend/ml-canvas/src/pages/AuditLogPage.tsx`
- Read: `frontend/ml-canvas/src/core/store/useJobStore.ts`
- Read: `frontend/ml-canvas/src/core/api/monitoring.ts`
- Read: `frontend/ml-canvas/src/core/api/deployment.ts`
- Read: `frontend/ml-canvas/src/core/api/registry.ts`

**Interfaces:**
- Consumes: shared-foundation criteria and findings.
- Produces: findings `OPS-001` and upward.

- [ ] **Step 1: Audit jobs and job details**

Exercise filters, status updates, pagination/history loading, selection, details, cancellation or retry actions where available, and navigation back to the originating workflow. Verify that statuses, timestamps, task types, failures, and progress are understandable and consistent.

- [ ] **Step 2: Audit registry and deployment management**

Exercise list states, search/filter behavior, model details, deployment actions, confirmation, success/error feedback, active deployment state, and empty history. Record whether the relationship between training job, registered model, and deployment is clear.

- [ ] **Step 3: Audit monitoring and investigation pages**

Exercise drift selection and thresholds, error-log filtering and resolution, slow-node diagnosis, and audit-log filtering/details. Verify that alerts link to actionable context and that severity, time range, affected resource, and next action are clear.

- [ ] **Step 4: Audit cross-page operational continuity**

Check whether users can move from an alert or failed job to the related pipeline, dataset, model, deployment, or error without manually copying identifiers. Record missing deep links, lost filters, and inconsistent naming.

- [ ] **Step 5: Add Operations findings**

Add detailed findings under `### Operations` and inventory rows using `OPS-*`.

- [ ] **Step 6: Commit the Operations audit**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: audit frontend operations journeys"
```

---

### Task 7: Synthesize and Prioritize the Roadmap

**Files:**
- Modify: `docs/ux/frontend-ux-roadmap.md`

**Interfaces:**
- Consumes: all `FND-*`, `CAN-*`, `DAT-*`, `EXP-*`, and `OPS-*` findings.
- Produces: deduplicated executive summary, component-boundary recommendations, milestones, and validation matrix.

- [ ] **Step 1: Deduplicate findings**

Merge journey findings into a shared `FND-*` item when the same root cause appears in two or more journeys. Preserve journey-specific evidence and acceptance criteria under the shared item.

- [ ] **Step 2: Rank every finding**

Apply this order:

```text
1. User impact and severity
2. Frequency
3. Number of journeys improved
4. Accessibility or data-loss risk
5. Implementation effort
6. Regression risk and dependencies
```

Use only these normalized values:

```text
Impact: High | Medium | Low
Frequency: Frequent | Occasional | Rare
Effort: S | M | L
Risk: Low | Medium | High
Milestone: Now | Next | Later
```

- [ ] **Step 3: Build the Now milestone**

Include the smallest dependency-complete set of high-impact cross-cutting items. Prefer consistent async states, navigation/context, form validation, accessible overlays, and responsive patterns when evidence supports them.

- [ ] **Step 4: Build Next and Later milestones**

Place journey-level redesigns and larger component-boundary changes in `Next`. Place lower-frequency enhancements and optional polish in `Later`. Do not place a dependency after the item that requires it.

- [ ] **Step 5: Write the executive summary**

Summarize 5-10 opportunities. Each summary item must link to one or more finding IDs and state the user outcome rather than an implementation detail.

- [ ] **Step 6: Complete component-boundary recommendations**

For each recommended split, document:

```markdown
### Component name
- **User-facing risk:** Specific inconsistency or regression risk.
- **Current responsibilities:** Independently changing concerns in the current file.
- **Proposed boundaries:** Named components/hooks and their responsibilities.
- **Required behavior preservation:** Existing interactions that must not change.
- **Validation:** Exact unit, interaction, or E2E coverage needed.
```

Do not recommend splitting a file solely because it is large.

- [ ] **Step 7: Complete the validation matrix**

Every `Now` and `Next` item must have measurable acceptance criteria and at least one automated or manual validation method. Every user-interface item must specify relevant desktop, tablet, mobile, keyboard, and screen-reader coverage.

- [ ] **Step 8: Commit the synthesized roadmap**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: prioritize frontend UX roadmap"
```

---

### Task 8: Validate the Final Roadmap

**Files:**
- Modify: `docs/ux/frontend-ux-roadmap.md`
- Read: `docs/superpowers/specs/2026-08-06-frontend-ux-roadmap-design.md`

**Interfaces:**
- Consumes: completed roadmap from Tasks 1-7.
- Produces: a self-consistent, implementation-ready final document.

- [ ] **Step 1: Verify spec coverage**

Confirm the roadmap includes:

```text
Executive summary
Shared-foundations backlog
Canvas backlog
Data/EDA backlog
Experiments/Inference backlog
Operations backlog
Now/Next/Later roadmap
Component-boundary recommendations
Validation matrix
Observed/Inferred evidence labels
```

- [ ] **Step 2: Scan for incomplete language**

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

Expected: exits `0` with no output. Replace vague language with an explicit decision, acceptance criterion, or documented exclusion.

- [ ] **Step 3: Validate finding IDs and milestone coverage**

Run:

```bash
rg -o "FND-[0-9]{3}|CAN-[0-9]{3}|DAT-[0-9]{3}|EXP-[0-9]{3}|OPS-[0-9]{3}" docs/ux/frontend-ux-roadmap.md | sort | uniq -c
```

Expected: each detailed finding ID appears in its detail section, inventory row, and roadmap or explicit exclusion. Investigate IDs with fewer than three occurrences.

- [ ] **Step 4: Verify every prioritized item has validation**

Read each `Now` and `Next` entry against the validation matrix. Add missing acceptance criteria, automated validation, manual validation, responsive coverage, or accessibility coverage.

- [ ] **Step 5: Re-run the frontend quality baseline**

From `frontend/ml-canvas/`:

```bash
npm run lint
npx tsc --project tsconfig.json --noEmit
npm run build
npm run test -- --reporter=dot
npm run test:e2e -- --project=chromium
npm run size-check
```

Expected: results match or improve on Task 1. Because this plan changes documentation only, any new frontend failure indicates an unrelated concurrent change and must be recorded rather than fixed here.

- [ ] **Step 6: Commit final corrections**

```bash
git add docs/ux/frontend-ux-roadmap.md
git commit -m "docs: finalize frontend UX improvement roadmap"
```
