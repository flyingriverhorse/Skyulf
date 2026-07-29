# Evaluation View — Split Threshold Slider / Threshold Tuning into Two Tabs

**Status:** Approved for planning
**Depends on:** `docs/superpowers/specs/2026-07-26-threshold-tuning-phase2-design.md` (Phase 2, shipped)
**Fixes relied upon:** `skyulf-core/skyulf/modeling/_evaluation/thresholds.py` multi-restart
Nelder-Mead fix (commit `cd018acf`) — Threshold Tuning results are only trustworthy
because of this fix; no further change needed here.

## Context

The Evaluation view (`EvaluationView.tsx`) currently overloads a single page
with two independent, unrelated threshold controls sharing one `cmView`
toggle:

1. **Manual slider** (`selectedMetric` / `threshold` / `selectedRocClass`) —
   purely client-side, scans the already-loaded `y_proba` locally, drives
   `ClassificationChartsForSplit`'s "Overall" charts (ROC, PR, single
   confusion matrix per split) plus the `bestMetricInfos` badges.
2. **Threshold Tuning panel** (`selectedTuningMetric` / `tuningPreview` /
   `useTunedThresholds`) — calls the real backend optimizer
   (`/thresholds/preview`, `/thresholds/save`, `/thresholds/toggle`, `DELETE
   /thresholds`), and its result is the only thing `PerClassConfusionMatrix`
   ever visualizes.

This coupling has two real problems:

- The `Overall` / `Per Class` toggle that gates whether
  `PerClassConfusionMatrix` (and therefore any visual reflection of tuned
  thresholds) renders at all is **hidden entirely for binary jobs**
  (`{!isBinary && (...)}`), and `PerClassConfusionMatrix` itself hard-returns
  `null` when `classes.length <= 2`. So for binary classification jobs,
  tuned thresholds are saved and genuinely applied at real `/predict` time,
  but a user can **never see their effect** anywhere on the Evaluation page
  today.
- Both controls currently live in the same view with no clear separation,
  which is what prompted the user's original bug report ("metrics show the
  same score") — investigation showed no bug in that specific job, but
  confirmed the UI doesn't make clear which control is driving which chart.

## Goals

- Split the existing functionality into two tabs with a clean, single
  responsibility each — **no change to any existing backend/API contract**;
  this is a frontend-only reorganization of already-working code.
- Make Threshold Tuning's real effect visible for **both binary and
  multiclass** jobs (today: multiclass only).
- Keep the manual slider tab exactly as it behaves today (including its own
  existing Per-Class view for multiclass) — zero behavior change there.
- Add hints/tooltips so a user unfamiliar with Preview/Save/Toggle/Clear
  understands what each button does and what "enabled" means for real
  predictions.

## Non-goals

- No change to the optimizer itself (`optimize_thresholds()` /
  `apply_thresholds()` in `skyulf-core`) — already fixed and tested.
- No new classification-node parameters — confirmed with the user that
  exposing Nelder-Mead's internal simplex-step parameters isn't useful;
  the multi-restart fix already handles robustness automatically.
- No "positive class" selector for binary Tuning tab — binary confusion
  matrices are symmetric (TP/FP/FN/TN for one class fully determine the
  other), so no extra control is needed.

## Design

### Tabs

Two tabs inside the classification branch of `EvaluationView`, replacing
today's single `Overall` / `Per Class` toggle and the always-visible Tuning
panel:

- **Tab 1 — "Threshold Slider"**: today's manual-slider control bar (Class
  selector, Metric selector, Threshold slider, best-metric badges, split
  labels), the existing `Overall` / `Per Class` sub-toggle scoped to this
  tab (still hidden for binary, since a binary job's single class pair has
  no separate "per class" view to add), and the charts it drives
  (`ClassificationChartsForSplit` for Overall, existing
  `PerClassConfusionMatrix` call with `useTunedThresholds=false` for
  multiclass Per Class). **Verbatim today's behavior — moved, not changed.**
- **Tab 2 — "Threshold Tuning"**: the Threshold Tuning panel (Metric
  selector, Preview/Save/Clear buttons, "Use tuned thresholds at prediction
  time" toggle, split-used note, tooltips) plus a confusion-matrix view
  driven only by `tuningPreview.thresholds`, rendered for **both** binary
  and multiclass jobs.

Both tabs share the page-level Train/Test/Validation split checkboxes
(`showTrainMetrics`/`showTestMetrics`/`showValMetrics`) — one control,
same state, gating which splits' panels render in whichever tab is active.
No per-tab duplication of this state.

Tab switching is local UI state in `EvaluationView` (a new `activeTab:
'slider' | 'tuning'` state var, analogous to the existing `cmView`), reset
to `'slider'` on job switch — consistent with how other transient view
state (`selectedRegressionSplit`, `cmView`) already behaves.

### Tab 2 confusion matrix rendering (binary + multiclass)

`PerClassConfusionMatrix` gets a **new binary code path**, gated on
`classes.length === 2` instead of hard-returning `null`:

- **Multiclass** (`classes.length > 2`): unchanged — reuse the existing
  "N vs Rest" OvR panel grid exactly as today, called with
  `useTunedThresholds=true` and `tunedThresholds=tuningPreview.thresholds`.
- **Binary** (`classes.length === 2`, new): render **one** plain N×N (2×2)
  confusion matrix per enabled split — not two redundant "vs Rest" mirror
  panels (which are identical up to axis order for 2 classes). Computed via
  the existing `applyMulticlassThresholds(splitData, tunedThresholds)` —
  already generic over class count, verified to produce a correct full
  confusion matrix for 2 classes via per-class argmax-with-thresholds
  (same function already used for the multiclass path). Precision/Recall/F1
  chips shown per class, same styling as the existing per-class cells,
  just laid out as a single unified matrix (rows/cols = the two actual
  classes) rather than a "vs Rest" framing that doesn't add information for
  2 classes.
- Both paths keep today's existing Train/Test/Validation layout (train+test
  side-by-side, validation below), driven by the same shared checkboxes.
- If `tuningPreview` is null (user hasn't clicked Preview yet), Tab 2 shows
  a placeholder: *"Click Preview above to see tuned thresholds applied to
  your confusion matrix."* — no matrix rendered, no error.

Tab 1's `PerClassConfusionMatrix` usage (multiclass-only, manual-threshold
OvR path, `useTunedThresholds=false`) is untouched — the component now
serves both call sites with the same set of props it already exposed
(`tunedThresholds`, `useTunedThresholds` were already present, unused for
binary until now).

### Hints/tooltips (Tab 2)

Reusing the existing `InfoTooltip` component already used elsewhere on this
page:

- Tab 2 header: one-line description — *"Let the optimizer find the best
  per-class threshold(s) for a metric you choose, preview its effect, then
  save it to actually change how this model predicts."*
- Tab 1 header: *"Manually explore how a single threshold changes
  predictions — nothing here is saved or used for real predictions."*
- Metric selector: *"Which metric the optimizer maximizes when you click
  Preview. Uses your validation split if available, otherwise test."*
- Preview button: *"Runs the optimizer now and shows the result below —
  does not save or affect real predictions yet."*
- Save button: *"Persists the previewed thresholds to this model version.
  Still inactive until you also enable 'Use tuned thresholds.'"*
- "Use tuned thresholds..." toggle: *"When ON, every real `/predict` call
  for this model uses these saved thresholds instead of the default
  0.5/argmax rule."*
- Clear button: *"Deletes saved thresholds entirely and reverts predictions
  to the default rule."*

### Data flow / state changes

- New: `activeTab: 'slider' | 'tuning'` state (parent component holding
  `EvaluationView`, alongside existing `cmView` etc.) plus its setter,
  threaded into `EvaluationView`'s props the same way `cmView` already is.
- No changes to any existing prop's type or the API client
  (`core/api/thresholds.ts` / whatever currently backs
  `onPreviewThresholds`/`onSaveThresholds`/`onToggleThresholds`/`onClearThresholds`).
- `PerClassConfusionMatrix` props: add nothing new — `tunedThresholds` and
  `useTunedThresholds` already exist; only the internal `classes.length <=
  2` early-return guard changes to a binary-vs-multiclass branch.

### Error handling

- Unchanged from today: `tuningError` renders as today (red text) inside
  Tab 2's control bar.
- Binary matrix rendering has no new error states — same
  `applyMulticlassThresholds` call already used and tested for multiclass.

### Testing

- Frontend: `PerClassConfusionMatrix.test.tsx` already exists — extend it
  with cases covering: (a) binary job with `tunedThresholds` set renders one
  2×2 matrix, not two "vs Rest" panels; (b) existing multiclass tests
  continue to pass unchanged; (c) `classes.length <= 2` no longer early-
  returns `null`.
- Frontend: `EvaluationView` — verify tab switching doesn't lose slider
  state and that shared split checkboxes affect both tabs' rendering (via
  existing test patterns for this component, if present, or new ones
  covering the new `activeTab` branch).
- Run repo gate for touched files per project rules: `eslint`,
  `tsc --noEmit`, `npm run build`, targeted `vitest run`.

## Open items resolved during brainstorming

- Shared vs. per-tab split checkboxes → **shared** (confirmed).
- Binary "positive class" selector in Tab 2 → **not needed** (confirmed).
- Classification-node params for optimizer internals → **not adding**
  (confirmed; existing multi-restart fix already covers robustness).
- Tab names → **"Threshold Slider"** / **"Threshold Tuning"**.
