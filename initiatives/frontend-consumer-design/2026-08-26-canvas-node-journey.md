# Canvas Node Journey — Deep Dive

**Date:** 2026-08-26 · **Status:** Investigation complete, prioritized change list ready, **no fixes applied**

## What this is

A consumer-perspective deep dive into the canvas **node system**: the node
catalog, how nodes connect, and how they're configured — specifically the
dataset → model-training journey. Method: three parallel full-file reviews
(connection mechanics, data/processing node config, modeling node config).
Companion to [README.md](README.md) (the full-app review); this document
goes deeper on the canvas slice.

All paths relative to `frontend/ml-canvas/src/` unless noted.

## Verdict

> **Easy to train; hard to configure well.**
> The minimum dataset→trained-model path is genuinely 2 decisions (confirm
> the auto-filled target column, press "Start Training") — model, params,
> and CV all default sensibly. But every preprocessing node starts *empty
> and invalid* until columns are hand-picked, connecting wrong is only
> caught at run time, and anything beyond basic mode (custom
> hyperparameters, tuning) assumes ML expertise.

---

## 1. Connecting nodes

### 1.1 No guidance while dragging (MED → fix)

`isValidConnection` (`components/canvas/FlowCanvas.tsx:320-330`) only
checks cycles and model-endpoint rules. The declared port `type`
(`'dataset' | 'model' | 'report' | 'any'`, `core/types/nodes.ts:13`) is
**never consulted**. There is no `.valid`/`.connectingto` styling
(`styles/canvas.css:48-119`), so during a drag nothing highlights or dims —
every left-side handle looks connectable. Semantically odd edges (e.g.
Ensemble→Ensemble, allowed because `targetType === 'EnsembleNode'` passes,
`core/store/useGraphStore.ts:206-208`) are only discovered at run time.

**Change:** while dragging, highlight valid targets and dim invalid ones;
enforce the declared port types in `isValidConnection`.

### 1.2 Rejection messages are good — delivery is late (LOW)

Rejections arrive as a toast after release (`onConnectEnd`,
`FlowCanvas.tsx:334-343`). The copy is genuinely consumer-grade:

- Cycle → "This connection would create a loop. Pipelines must flow in one
  direction — remove the backwards wire instead." (`useGraphStore.ts:211`)
- Model output → "Training nodes are the end of a pipeline: their output is
  a trained model, not data. Only an Ensemble can consume a model
  output…" (`useGraphStore.ts:213`)

Dataset→model direct is allowed (correct — models accept raw data). Dataset
merges trigger a **native `window.confirm`** explaining merge/last-wins
(`useGraphStore.ts:507-518`) — jarring next to the app's own ConfirmDialog
(used for node delete, `FlowCanvas.tsx:106-121`).

**Change:** use ConfirmDialog for merges; show a hint at the hovered handle
before an invalid drop, not just a toast after.

### 1.3 Handles are labeled but subtle (LOW)

Most nodes: 1 in / 1 out, labeled — "Data"/"Training Data" in,
"Cleaned Data"/"Transformed"/"Trained Model" out (`core/factories/nodeFactory.ts:44-45`).
Splitters are clear: Train/Validation/Test
(`modules/nodes/modeling/TrainTestSplitNode.tsx:193-197`), Features (X)/
Target (y) (`FeatureTargetSplitNode.tsx:94-97`). Labels are 10px muted text
inside the card (`components/canvas/CustomNodeWrapper.tsx:519-537`);
left(slate)/right(teal) handle colors are the only in/out cue
(`styles/canvas.css:69-87`).

**Change:** bump label size/weight slightly; keep colors (they work).

### 1.4 Schema preview is a strength (keep)

`useSchemaPreview` (400ms debounce POST, `core/hooks/useSchemaPreview.ts:20-75`)
feeds `↳ N cols` badges with tooltips listing up to 12 predicted columns
(`CustomNodeWrapper.tsx:336-348`); data-dependent steps show `↳ ?` with a
friendly tooltip; broken column refs paint an amber border + chip:
"⚠ Column name not found in upstream output" (`CustomNodeWrapper.tsx:53-62`).
This is the best consumer feedback on the canvas — extend the pattern.

### 1.5 Silent traps (HIGH)

- The Run Preview button **disappears entirely** when validation fails
  (`components/canvas/Toolbar.tsx:595`, gated in `useRunControls.ts:36-43`).
  The issue list only exists in the Results panel "Issues" tab
  (`ResultsPanel.tsx:202-242`), which isn't auto-opened unless Run was
  clicked — so a stuck novice can never see *why* they're stuck.
  (Same item as parent doc §B.5 — this is the evidence.)
- Dropping a wire on empty space gives no feedback (React Flow default).

### 1.6 Jargon in the wire layer (MED)

Edge hover tooltip + aria-label expose raw internal IDs:
`dataset_node-8f3a… → imputation_simple-2b1c…`
(`components/canvas/CustomEdge.tsx:125,186`). Unknown nodes render
"Type: {definitionType}" (`CustomNodeWrapper.tsx:220`).

**Change:** edge tooltips should show node *names* ("Customers → Impute
missing values").

---

## 2. Configuring data & processing nodes

### 2.1 Every preprocessing node starts empty and invalid (HIGH — the biggest single friction)

Defaults ship as `columns: []` and fail validation with "Select at least
one column" (e.g. `modules/nodes/processing/ImputationNode.tsx:329,335-345`).
There is no "impute all numeric columns with missing values" one-click, so
**a pipeline cannot run without touching each node**. Mitigations that
exist: the RecommendationsPanel nudges (`RecommendationsPanel.tsx:147-154`)
and TrainTestSplit's sensible defaults (test 0.2, random_state 42, shuffle
on, `TrainTestSplitNode.tsx:220-226`).

**Change:** smart defaults per node type — e.g. Imputation defaults to
*all columns with missing values*, Scaling to *all numeric columns*,
Encoding to *all categorical columns* — with an "apply to all applicable
columns" toggle. Keep manual selection as the override.

### 2.2 Column picker gaps (MED)

Shared `ColumnMultiSelect` (`modules/nodes/shared/ColumnMultiSelect.tsx`):
checkbox list with search ("Search columns...", `:148`) but
All/None acts only on the *visible filtered* set (`:72-77,103-104`);
no dtype badges — nodes pre-filter instead (Scaling shows only numerics,
`ScalingNode.tsx:49-54`), so users never see *why* a column is missing
from a picker. EncodingNode's "Columns to Encode" actually lists **all**
non-dropped columns (`EncodingNode.tsx:71-73`) — the opposite problem.

**Change:** show all columns with dtype badges and disabled-state reasons
("numeric — scaling not needed"); All/None should apply to the full set
with a count shown.

### 2.3 No pre-run preview of a transform's effect (MED)

Feedback like "Imputed Values:" appears only after running
(`ImputationNode.tsx:85`); the only pre-run signal is the `↳ N cols` badge.
A consumer choosing binning edges or an encoding strategy is guessing.

**Change:** small "preview on sample" output in the settings panel once
upstream schema is known (or reuse Data Preview with the transform applied).

### 2.4 Settings surface & jargon (MED)

Right-side Properties Panel, 320px (`w-80`, expandable,
`PropertiesPanel.tsx:42-47`); deep nodes scroll heavily. Many selects have
one-line help ("Median (Middle Value)… (Robust to outliers)",
`ImputationNode.tsx:180-186`) — good. But jargon persists:

| Label | Where |
|---|---|
| "Iterative Imputer (MICE)", estimator "Bayesian Ridge" | `ImputationNode.tsx:165,258` |
| "Contamination" | `OutlierNode.tsx:382` |
| "False Discovery Rate (FDR)" | `FeatureSelectionNode.tsx:285` |
| "dummy variable trap (multicollinearity)" | `EncodingNode.tsx:234` |
| "Target Columns" (collides with target=y meaning) | `ImputationNode.tsx:289`, `BinningNode.tsx:205` |

**Change:** plain-language rewrites + HelpTooltips; rename "Target
Columns" → "Columns to act on".

### 2.5 Validation feedback is too quiet + one real bug (HIGH)

- Invalid config shows only as a red corner dot + tooltip plus a one-shot
  pulse (`CustomNodeWrapper.tsx:192-196,287-295`); no inline text in the
  settings panel. Good exceptions: stale-column amber chip, and
  TrainTestSplit's inline "Error: Total split size exceeds 100%"
  (`TrainTestSplitNode.tsx:121`).
- **Bug:** `TimeSeriesNode.tsx:270` and `EncodingNode.tsx:489` return
  `error:` instead of `message:` from validation, but the wrapper reads
  only `.message` (`CustomNodeWrapper.tsx:181`) — users see generic
  "Configuration incomplete." instead of the real reason.
- **Missing guard:** DropColumns lists every non-dropped column with no
  protection against dropping the target (`DropColumnsNode.tsx:30-33`).
- Deduplication/DropRows/MissingIndicator always validate true
  (`DeduplicationNode.tsx:136`) — fine, but inconsistent with the rest.

### 2.6 Split nodes: the target concept is under-explained (HIGH)

FeatureTargetSplit's entire explanation: "This column will be separated as
the target (y), and all other columns will be features (X)."
(`FeatureTargetSplitNode.tsx:79-81`) — one line, no guidance on *which*
column to pick, plain "-- Select Target --" dropdown (`:74`). TrainTestSplit
hides the target picker behind "Stratify by Target" (`:163-170`) and its
validation help is expert-speak: "grades hyperparameter candidates during
tuning (instead of CV folds)" (`:114-115`).

**Change:** a sentence on what a target is ("the thing you want to
predict"), plus heuristic hints from column metadata (e.g. "price looks
like a regression target").

### 2.7 Node body at-a-glance state is good (keep)

`bodyPreview` lines ("mean · 3 cols" `ImputationNode.tsx:322-327`,
"0.8 / 0.2" `TrainTestSplitNode.tsx:199-206`, "Set target"
`FeatureTargetSplitNode.tsx:102`) + green/red run chips + validation dot
make state legible at a glance. One oddity: Binning shows `q=5` for bin
count (`BinningNode.tsx:238`).

---

## 3. Modeling nodes (dataset → trained model)

### 3.1 Model selection: flat and unguided (MED)

A flat `<select>` of registry names only — no grouping by family, no
descriptions, even though the registry *has* a `description` field that's
unused (`modules/nodes/modeling/TrainingSettings.tsx:368-370`). No "which
should I pick?" guidance.

Positives to keep: sensible preselection (`ClassificationNode.tsx:21`
defaults to `random_forest_classifier`), task-scoped filtering, and the
"Scale Your Data — Consider adding a 'Feature Scaling' node" hint
(`TrainingSettings.tsx:383-389`).

**Change:** group by family (linear / tree-based / boosting / other) with
one-line descriptions from the registry.

### 3.2 Hyperparameters: great basic mode, broken custom mode (MED)

Basic mode is excellent: "Using default hyperparameters." behind an opt-in
"Customize" checkbox that seeds defaults (`TrainingSettings.tsx:648-662,688`).
Good tooltip copy exists (max_depth: "Deeper trees capture more patterns but
risk overfitting", `format.ts:80`).

**Broken:** when customized, values are raw **text inputs** —
`HyperparameterInput.tsx:53-54` renders `type="text"`, so the min/max/step
passed at `:60` are inert; invalid entries **silently revert** (`:41`) with
no error message. No sliders.

**Change:** sliders/range inputs with inline validation; never silently
revert.

### 3.3 Tuning: expert-gated but safe by default (HIGH for novices)

Exposed via "Basic | Advanced (Tuning)" toggle (`TrainingSettings.tsx:47`).

- Search space is **comma-separated raw text**, monospace, placeholder
  "e.g. 10, 50, 100" (`SearchSpaceInput.tsx:91`); select params get chips
  (good); no distributions (log-uniform) or range sliders.
- Search methods: "Random Search / Grid Search / Successive Halving
  (Grid) / … / Optuna Search" (`TrainingSettings.tsx:478-482`) with decent
  tooltips ("Random Search … Fast and often surprisingly effective", `:449`).
- Strategy modal: "Sampler: TPE (Bayesian Optimization) / CMA-ES",
  "Pruner: Median/Hyperband", "Factor", "Min Resources… 'exhaust'"
  (`StrategySettingsModal.tsx:165-172,211-213,238-240`) — unanswerable
  jargon for novices.

Mitigations: safe defaults ("Using defaults (sampler: tpe · pruner:
median)", `TrainingSettings.tsx:95`) and auto-loaded default search spaces
(`:288`), so it runs untouched. Inline errors in SearchSpaceInput are good
("'part' is not a valid number", `SearchSpaceInput.tsx:40,140-145`);
halving_grid gets a cost warning (`TrainingSettings.tsx:110-112`).

**Change:** keep "Advanced" but default it to a one-choice "Auto (Optuna)"
with sane budget; move raw space editing to an expert tab.

### 3.4 Results don't live on the node (MED)

The trained model's score never appears on the node — `bodyPreview` shows
only `model → target` (`core/factories/nodeFactory.ts:56-63`); scores live
in the job drawer/experiments page. `BestParamsModal.tsx:151-154` shows
"Accuracy: 0.9312" with **no direction indicator**, even though
"Higher/Lower is better" badges exist (`MetricDirectionBadge.tsx:11-12`)
and rich metric descriptions exist (`format.ts:2-29`) — neither surfaces
on the Metric dropdown (`TrainingSettings.tsx:501-508`), which lists bare
"Accuracy, F1 Score, ROC AUC, MSE…".

**Change:** show the best score + metric on the node body after training;
reuse MetricDirectionBadge and format.ts descriptions everywhere.

### 3.5 Metric/task mismatch bug (MED)

The classification node's metric dropdown includes MSE/RMSE/MAE, and the
regression node includes Accuracy/F1/ROC AUC — one shared list, no task
filtering (`TrainingSettings.tsx:501-508`). Ensemble *does* filter
correctly (`EnsembleSettings.tsx:158-171`).

**Change:** filter the metric list by task type.

### 3.6 Ensemble & segmentation: well-guarded (keep)

Ensemble has a plain-English banner ("Combine several models into one.
**Voting** averages their predictions; **Stacking** trains a
meta-learner…", `EnsembleSettings.tsx:1048`), task auto-detect from target
dtype (`:142-156`), safe defaults (3 base models, `:121-125`), and blocks
runs under 2 models ("Pick at least two base models", `:1121-1126`).
Segmentation: "Group rows into clusters by similarity — no target column
needed." (`SegmentationSettings.tsx:331`). Remaining jargon: "Calibrate
base models", "Passthrough features".

---

## 4. Difficulty assessment (dataset → trained model)

| Stage | Difficulty | Why |
|---|---|---|
| Add dataset node + bind data | Easy | Clear picker, schema table appears |
| Understand the data | Medium | Schema stats yes; row sample/distributions require a separate Data Preview node + run |
| Pick a target | Medium-Hard | One-line explanation, no heuristic help |
| Add preprocessing | **Hard** | Every node invalid until columns hand-picked; no smart defaults |
| Connect nodes | Medium | Easy to drag; no drag-time guidance; mistakes surface at run time |
| Choose model + train | **Easy** | 2 decisions; everything defaults |
| Read the score | Medium | Must open job drawer/experiments; no score on node |
| Tune / custom params | Hard-Expert | Raw text inputs, raw comma-separated spaces, sampler/pruner jargon |

**Minimum decisions for a trained model: 2.** (Confirm auto-filled target,
press Start Training.) The wall is preprocessing configuration, not
modeling.

---

## 5. Prioritized change list

### P0 — fix now (small, high relief)

| # | Change | Evidence |
|---|---|---|
| N1 | Smart defaults: processing nodes default to "all applicable columns" (impute→missing, scale→numeric, encode→categorical) | §2.1 |
| N2 | Fix `error:` vs `message:` validation bug (TimeSeries, Encoding) so real reasons show | §2.5 |
| N3 | Guard against dropping the target column in DropColumns | §2.5 |
| N4 | Filter metric dropdown by task type | §3.5 |
| N5 | Show best score + metric on the trained-model node body | §3.4 |
| N6 | Always-visible (disabled) Run button with inline "what's missing" checklist, auto-open Issues | §1.5 |

### P1 — guided building (1–2 weeks)

| # | Change | Evidence |
|---|---|---|
| N7 | Drag-time connection guidance: highlight valid targets, dim invalid; enforce declared port types | §1.1 |
| N8 | Target explanation + heuristic hints in split nodes | §2.6 |
| N9 | Sliders + inline validation for custom hyperparameters (never silent revert) | §3.2 |
| N10 | Model picker grouped by family with registry descriptions | §3.1 |
| N11 | dtype badges + disabled-reasons in ColumnMultiSelect; All/None on full set | §2.2 |
| N12 | Metric direction badges + descriptions everywhere scores/metrics appear | §3.4 |
| N13 | Edge tooltips with node names, not internal IDs | §1.6 |

### P2 — experience polish

| # | Change | Evidence |
|---|---|---|
| N14 | Pre-run transform preview ("apply on sample") | §2.3 |
| N15 | Jargon rewrite pass (MICE, Contamination, FDR, dummy variable trap, "Target Columns") | §2.4 |
| N16 | Tuning default = one "Auto (Optuna)" choice; raw space editing in expert tab | §3.3 |
| N17 | ConfirmDialog for dataset merges; hint before invalid drop | §1.2 |
| N18 | Keep & extend: `↳ N cols` schema badges, stale-column amber chips, bodyPreview lines, leakage warnings, ensemble/segmentation banners | §1.4, §2.7, §3.6 |

---

## 6. Settings panel placement (addendum, 2026-08-26)

**Question:** other tools open node settings inline/anchored to the node
("under" it) instead of covering the whole right edge of the page like
Skyulf's PropertiesPanel. Is that better?

### 6.1 How ours works today

`components/layout/PropertiesPanel.tsx`:

- Opens **automatically on node selection**; closes when selection clears
  (`PropertiesPanel.tsx:21-26,47`).
- Default width `w-80` (320px); the expand button grows it to
  `calc(100vw - 328px)` — **near-full viewport, hiding the canvas
  entirely** (`PropertiesPanel.tsx:42,47`).
- Header shows label + raw `ID: {selectedNode.id}` (`:97-98`).
- Deep nodes (Encoding: method + params + columns + feedback; Training:
  model + hyperparameters + tuning + CV) scroll heavily inside it (see
  §2.4).

### 6.2 Verdict

**Inline/anchored wins on feel; the side panel wins for depth. The fix is
to make ours less hijacking — not to copy the inline pattern wholesale.**

Why the inline pattern feels nicer:

- Proximity: you edit where the node is, no eye-jump to the right edge.
- It doesn't steal screen space; the canvas stays visible.
- Lightweight for 1–2 settings (ComfyUI-style node bodies, Figma-style
  popovers).

Why it breaks down for Skyulf specifically:

- Our settings are genuinely deep. Inline node bodies would balloon the
  canvas into a wall of open forms; anchored popovers get clipped and
  scrolling inside a floating card is miserable. Every tool with deep
  node config (n8n, KNIME, Azure ML Designer) converges on a side panel
  for exactly this reason.

### 6.3 Changes

| # | Change | Why |
|---|---|---|
| N19 | **Stop auto-opening on selection.** Single-click selects; panel opens on double-click or a gear button. | Kills the "covers the right page every time I touch a node" feeling — the main complaint. |
| N20 | **Cap expanded width** (~50% or a draggable overlay). | Near-full-viewport expansion hides the canvas, which is the user's context. |
| N21 | **Quick settings inline, rest in panel.** Extend the existing inline pattern (TrainTestSplit sliders in the node body, `TrainTestSplitNode.tsx:193-226`): every node exposes its top 1–2 controls on the card. | Best of both worlds; panel stops being the only surface. |
| N22 | **Shallow nodes get a popover instead of the panel** (nodes with ≤2 settings: CastType, Deduplication, MissingIndicator…). | The panel is overkill for one-select nodes. |
| N23 | **Tabs/accordions inside the panel** for the deepest nodes (Encoding, Training). | Kills the endless scroll (§2.4). |

(Also applies: drop the raw ID from the panel header — already covered by
§2.5/LOW findings.)

## Relation to other docs

- Parent review: [README.md](README.md) — items B.5, B.8, B.10 of that doc
  are expanded with node-level evidence here.
- `growth/` plan Stage 2 activation work overlaps N1/N8 conceptually
  (first-run activation) — coordinate when scheduling.
