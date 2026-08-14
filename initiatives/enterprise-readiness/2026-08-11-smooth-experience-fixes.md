# Smooth Experience — Consolidated Fix List

**Date:** 2026-08-11 (Round 3 investigation)
**Scope:** Concrete, verified UX friction points that affect whether the
day-to-day and first-run experience feels smooth and professional — as
distinct from [technical-debt-deep-dive.md](2026-08-11-technical-debt-deep-dive.md)
(architecture/correctness/a11y, round 2) and
[differentiation-strategy.md](2026-08-11-differentiation-strategy.md)
(competitive positioning). This doc is the "fix these so the product
doesn't feel janky" list.

## How this was produced

Two agents ran in parallel: one simulated a brand-new user's first 10
minutes end-to-end; one rubber-duck agent specifically hunted for
janky/untrustworthy-feeling issues, explicitly excluding everything
already documented in round 2 (god-store, a11y gaps, dataset-table
virtualization, silent monitoring failures, autosave divergence, token
duplication, job-cancellation race). Every finding below cites the exact
file:line the agent opened to verify it.

---

## Top 3 First-Run Fixes (highest leverage, do these first)

Per the first-run-UX agent's own prioritization:

1. **Sample-data guided baseline.** Removes the upload + ML-knowledge +
   blank-canvas barriers all at once. Concretely: ship one of the example
   CSVs that **already exist in the repo but are never surfaced in the UI**
   (`skyulf-core/examples/data/{online_retail,credit_card_fraud,santander}/*_sample.csv`
   — confirmed by the rubber-duck agent independently), add a "Load sample
   dataset" option to `AddSourceModal`, and bind at least one starter
   template to it so a new user can go template → run without bringing
   their own data. **This is unusually cheap to ship** — the data and the
   templates both already exist; only the binding and UI entry point are
   missing.
2. **Post-upload pipeline recommendation.** Turn the profiler's existing
   heuristic recommendations (missingness, imbalance, high-cardinality,
   skew — `skyulf-core/skyulf/profiling/recommendations.py`) into an
   actual assembled, pre-configured pipeline rather than fragmented
   per-node suggestions the user must discover one at a time. Directly
   overlaps with Differentiation Bet #2 in the companion strategy doc —
   build once, serves both onboarding and positioning.
3. **Progress + plain-English recovery.** Replace "Pipeline execution
   failed — Check console for details" (`useRunControls.ts:99-105`) and
   similar generic messages with `what happened / affected node or column
   / suggested fix / retry`, keeping raw logs behind a "Technical details"
   toggle rather than as the primary message.

---

## Verified Findings by Area

### A. First-run & onboarding (audit-first-run-ux)

| Finding | Evidence | Impact | Fix |
|---|---|---|---|
| No onboarding tour/tutorial/sample-data flow exists anywhere in the frontend | Grep for "onboarding"/"tutorial"/"walkthrough"/"tour" returns nothing; `Dashboard.tsx:122-129` "New Experiment" goes straight to a blank Canvas | High | Add a first-run wizard (see Top 3 above) |
| Templates require manual dataset binding + target setup even after selection | `TemplatesGalleryModal.tsx:14-18`, `pipelineTemplates.ts:104-127` | High | Bind at least one template to the sample dataset |
| Dashboard/Experiments/Jobs empty states give no next-action guidance (e.g. "No recent jobs found", "No jobs found matching your criteria" conflates zero-history with filtered-empty) | `Dashboard.tsx:239-245`, `JobListSidebar.tsx:55-59`, `Jobs.tsx:452-458` | High | Page-specific empty-state CTAs distinguishing "no data yet" from "no results for this filter," linking to the next prerequisite action |
| Several error messages are generic/technical rather than actionable ("Pipeline execution failed — Check console", "Failed to delete dataset") | `useRunControls.ts:99-105,154-156`, `DataSources.tsx:97-101,128-132` | Medium | Normalize to what/where/fix/retry; keep raw error behind a details toggle |
| Smart per-node suggestions exist (imputation/outlier/skew recommendations) but are fragmented and only surface after a synchronous preview, never assembled into a full pipeline | `_advisor.py:68-225`, `ImputationNode.tsx:63-75,146-153` | Medium | See Top 3 #2 |
| Preview is synchronous with no staged progress feedback; training only shows "queued" before jumping to the Jobs view | debounce/polling values in `useJobPolling.ts:78,187-217` | Medium | Stream execution-stage progress into Canvas/Results instead of a binary spinner |
| Node help is scattered across hover tooltips with no persistent "what should I do next" guidance | `Sidebar.tsx:116-135`, `TrainingSettings.tsx:444-450` | Medium | Add a persistent contextual assistant panel per selected node |
| Runtime failures point users to the browser console with no in-app plain-English explanation or repair suggestion | (same as above, `useRunControls.ts`) | Medium | "Explain and fix" affordance on failed preview/job (see Top 3 #3) |

**Positive, confirmed, worth preserving:** file-size/type validation
messages are already genuinely good (`FileUpload.tsx:52-55`,
`backend/data_ingestion/service.py:420-426` — specific limits and allowed
types named), graph validation errors name the exact node and required
fix (`useGraphStore.ts:193-225`), and undo/redo + retry-eligibility on
failed jobs are solid (`JobDetailsView.tsx`).

### B. Perceived performance (rubber-duck)

| Finding | Evidence | Impact | Fix |
|---|---|---|---|
| Inference input textarea re-parses the entire JSON payload 3× on every keystroke with no debounce (`analyseInput`, `schemaCheck`, `parsedInputRows` — three separate `useMemo`s, each doing a full `JSON.parse` plus, for one, a per-row/per-key scan) | `InferencePage.tsx:224,250,731,766-774,1673` | Non-blocking but real: typing/pasting a realistic batch (hundreds-thousands of rows) stutters | Debounce `inputData` (~250-300ms) into a derived value; parse once and share across all three memos |

**Positive, confirmed, worth preserving:** the realtime job-status
architecture (`useJobPolling.ts`, `useNodeJobSummaries.ts`,
`jobEventsSocket.ts`) is genuinely well built — WS-primary with a
stretched 30s safety-net poll, stale-response guards, per-job give-up
counters, reconnect jitter. Schema preview debounce (400ms + request-id
guard) and autosave debounce (1s trailing + unmount flush) are both
correctly implemented. No jank found in either.

### C. Trust signals during active use (rubber-duck)

| Finding | Evidence | Impact | Fix |
|---|---|---|---|
| WebSocket connection state (`onStatus`) is tracked internally to switch polling cadence but never rendered anywhere for the user | `jobEventsSocket.ts:44-90,118-131` used only internally in `useJobPolling.ts:209`, `useJobStore.ts:274` — zero rendered consumers | Non-blocking, but meaningful for a platform whose core value is watching long jobs | Add a small "Live / Reconnecting" indicator (e.g. Layout header or JobsDrawer) — the plumbing already exists, only the UI is missing |
| Deleting a dataset (explicitly flagged "cannot be undone" in its own confirm dialog) gives no success toast — only a failure toast exists, unlike deploy/redeploy/save which all confirm success | `DataSources.tsx:88-102` vs `DeploymentsPage.tsx:133,161`, `ModelRegistry.tsx:138`, `usePipelineActions.ts:165` | Non-blocking | Add `toast.success('Dataset deleted')` |

### D. Interaction-pattern consistency (rubber-duck)

| Finding | Evidence | Impact | Fix |
|---|---|---|---|
| `BestParamsModal` is hand-rolled instead of using the shared `ModalShell` — no `role="dialog"`, no Escape-to-close, no focus trap, unlike the other 12 modals in the app | `BestParamsModal.tsx:66-68` vs `ModalShell.tsx:66-92` | Non-blocking, but a real keyboard-nav dead spot | Port onto `ModalShell` |
| No right-click context menu anywhere in the canvas (grep for `onContextMenu`/`ContextMenu` returns zero hits) | app-wide grep | Suggestion only | Optional: add a canvas node context menu reusing existing store actions |

### E. Undo/redo granularity (rubber-duck)

| Finding | Evidence | Impact | Fix |
|---|---|---|---|
| Free-text/number node-config fields commit to the store on every keystroke, and each keystroke counts as a distinct entry in the 100-item undo history — typing a 15-character value creates 15 undo steps and can silently evict earlier structural operations (add-node, connect-edge) from the stack | `PolynomialFeaturesNode.tsx:122`, `OutlierNode.tsx:331-390`, `useGraphStore.ts:493-502,611-629` (`limit: 100`) | Non-blocking but surprising: a user typing a long value then hitting Ctrl+Z to undo an accidental deletion may find it's already been evicted | Debounce/coalesce text-field commits (commit on blur or after idle), or batch consecutive same-node edits into one history entry |

**Positive, confirmed:** node config changes ARE undoable (a common gap in
other tools per the market research — KNIME/RapidMiner reviews don't claim
this works well) — this finding is about granularity/eviction, not
missing coverage. The core undo model is sound.

### F. Onboarding data (rubber-duck) — same root cause as A above

Confirmed independently: example CSVs ship in the repo
(`skyulf-core/examples/data/*/`) but are never surfaced in the frontend;
`SampleDataTab.tsx` only previews the user's own uploaded data, not demo
content. Starter templates explicitly ship with a placeholder dataset node
per their own header comment in `pipelineTemplates.ts`. Two independent
agents converged on the identical finding from different angles — treat
this as high-confidence and see Top 3 #1.

### G. Notification consistency (rubber-duck)

Sampled 8 mutation sites: the app is **error-heavy, success-light** (59
`toast.error` vs 27 `toast.success` vs 1 `toast.info` app-wide). Save,
deploy/redeploy/deactivate, and run-experiments all correctly confirm both
success and failure. **Delete dataset** confirms only failure (see C
above). **Create data source** uses a different feedback channel
entirely — inline `setError` plus a silent `onSuccess` callback with no
toast at all (`AddSourceModal.tsx:63-70`).

**Fix:** adopt one rule — "explicit user-initiated mutations get a success
toast; destructive actions always do" — and apply it to delete-dataset and
create-source specifically. No general toast-fatigue problem was found
(no noisy per-poll/per-event toasts exist) — this is purely an
under-notification gap, not an over-notification one.

### H. Visual polish consistency (rubber-duck)

A `Skeleton` component exists (`components/shared/Skeleton.tsx`, with
Storybook stories) but has **zero real usages** anywhere in the app — 41
files use `animate-spin`/`Loader2` spinners and 13 use ad-hoc
`animate-pulse` text instead. Loading UX is inconsistent page-to-page with
no content-shaped skeletons despite the primitive already existing.

**Fix:** pick one convention (skeletons for list/table/card loads,
spinners only for button-level inline actions) and either adopt `Skeleton`
in the main loaders or remove the unused component so it stops implying an
unfulfilled standard. This overlaps directly with the "unify loading
states" recommendation already in
[redesign-existing-pages.md](2026-08-11-redesign-existing-pages.md)'s
cross-page section — do them together.

---

## Summary Judgment

**No blocking issues were found in this round** — the realtime job
architecture, undo model, and validation-error quality (where present) are
genuinely solid foundations. Everything found is real but non-blocking:
consistency gaps (some actions get success toasts, some don't; some
modals use the shared shell, one doesn't), a couple of missing
trust-signal pixels where the underlying plumbing already exists (WS live
indicator), one perf nit (inference textarea parsing), one undo-history
granularity issue, and the same "no reachable sample data" root cause
found independently by two different investigative approaches — which is
also the single highest-leverage fix, since Top 3 items #1 and the
Differentiation Bet #2 in the companion strategy doc reuse almost entirely
existing code and data.

## Cross-References

- [2026-08-11-differentiation-strategy.md](2026-08-11-differentiation-strategy.md) — competitive positioning; Bet #2 (guided baseline) and this doc's Top 3 #1/#2 are the same underlying feature
- [2026-08-11-technical-debt-deep-dive.md](2026-08-11-technical-debt-deep-dive.md) — round 2 architecture/a11y findings, deliberately not repeated here
- [2026-08-11-redesign-existing-pages.md](2026-08-11-redesign-existing-pages.md) — loading-state unification overlaps with finding H above
- [2026-08-11-master-fix-list.md](2026-08-11-master-fix-list.md) — where these fixes sit in the overall phased plan
