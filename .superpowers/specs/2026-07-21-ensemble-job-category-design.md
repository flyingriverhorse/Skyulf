# Ensemble as a Distinct Job Category (Experiments, Job History, Job Page)

## Problem

Voting/Stacking ensemble jobs (`voting_classifier`, `stacking_classifier`,
`voting_regressor`, `stacking_regressor`) are currently indistinguishable
from plain Classification/Regression jobs in:

- Job History drawer (`JobsDrawer.tsx`)
- Full Job History page (`pages/Jobs.tsx`)
- Experiments page (`ExperimentsPage.tsx` + `HeaderAndTabs.tsx`)
- Job cards / Job Details modal (`JobCard.tsx`, `JobDetailsView.tsx`)

All of these resolve a job's task purely via
`getTaskForModelType(job.model_type, registryItems)` in `jobMeta.ts`, which
reads `RegistryItem.tags` (`classification`/`regression`/`text`/`nlp`/
`clustering`). Ensemble models carry those same task tags (for base-learner
selection UIs), so they silently merge into the Classification/Regression
buckets — there is no way to see "just my ensemble runs" anywhere.

## Key existing signal

The backend model registry already tags ensemble calculators with a
distinct `category="Ensemble"` in their `@node_meta` decorator
(`skyulf-core/skyulf/modeling/ensemble.py`), separate from `tags`. This
`category` field is already returned by `/pipeline/registry` and available
on the frontend's `RegistryItem` type — it's simply not consulted yet by
`getTaskForModelType()`. Because `model_type` is fixed on a job at
submission time, keying off `category` satisfies "determined when training
starts" with no backend/schema changes required.

## Design

### 1. Type: `core/types/taskType.ts`

Add `'ensemble'` to the `TaskType` union:

```ts
export type TaskType = 'classification' | 'regression' | 'text_classification' | 'segmentation' | 'ensemble';
```

### 2. `jobMeta.ts`

- `getTaskForModelType(modelType, registryItems)`: check
  `registryItems.find(r => r.id === modelType)?.category === 'Ensemble'`
  **before** the existing tag-based checks, returning `'ensemble'`. Update
  the `registryItems` parameter type to include `category?: string` so the
  category is visible to the function.
- New helper `getEnsembleSubTask(modelType, registryItems)` →
  `'classification' | 'regression' | undefined`, reading the same `tags`
  array already used today, for:
  - the Ensemble tab's sub-filter pill,
  - the ensemble badge's sub-task label,
  - resolving the correct metric-priority list in `getDisplayScore`.
- `SCORE_METRIC_PRIORITY` gains no new `'ensemble'` key. Instead, callers of
  `getDisplayScore` resolve the *effective* metric-priority task as
  `task === 'ensemble' ? (getEnsembleSubTask(modelType, registryItems) ?? 'classification') : task`
  before calling it, since the real scoring metrics are always the
  underlying classification/regression ones.
- New helper `getEnsembleStrategy(modelType)` → `'Voting' | 'Stacking' | undefined`,
  derived from the `model_type` prefix (`voting_*` / `stacking_*`), for the
  badge.

### 3. Job History drawer (`JobsDrawer.tsx`) & full Job History page (`pages/Jobs.tsx`)

- Add a 5th entry to `TASK_TABS`: `{ task: 'ensemble', label: 'Ensemble' }`
  (and matching icon in `pages/Jobs.tsx`, e.g. `Boxes`/`Layers`) and to
  `TASK_LABELS`.
- When `activeTab === 'ensemble'`, render a small secondary sub-filter pill
  row above the job list: **All / Classification / Regression**, filtering
  the already-tab-filtered jobs further via `getEnsembleSubTask`. Default:
  "All".

### 4. Experiments page (`ExperimentsPage.tsx` + `HeaderAndTabs.tsx`)

- Extend the `filterType` union (`ExperimentsPage.tsx` state) and the
  `HeaderProps`/`onChange` types in `HeaderAndTabs.tsx` to include
  `'ensemble'`.
- Add one `<option value="ensemble">Ensemble</option>` to the existing
  filter `<select>`. No sub-filter pill needed here — it's a flat dropdown,
  and the badge (see below) already surfaces the sub-task per row in the
  comparison table.

### 5. Badges: `JobCard.tsx` & `JobDetailsView.tsx`

- When a job's resolved task is `'ensemble'`, render a badge:
  `"Ensemble · {Voting|Stacking} · {Classification|Regression}"`
  (e.g. `"Ensemble · Stacking · Classification"`), using
  `getEnsembleStrategy(modelType)` + `getEnsembleSubTask(...)`.
- Non-ensemble jobs are unaffected — no badge shown.

## Data flow summary

```
job.model_type  ──lookup──>  RegistryItem { category, tags }
                                   │
                                   ├─ category === 'Ensemble'? ──yes──> task = 'ensemble'
                                   │                                      │
                                   │                                      ├─ tags.includes('classification'|'regression')
                                   │                                      │     -> sub-task (for pill/badge/metric priority)
                                   │                                      └─ model_type prefix 'voting_'/'stacking_'
                                   │                                            -> strategy (for badge)
                                   └─ else ── existing tag-based resolution (unchanged)
```

## Testing

- Unit tests for `getTaskForModelType` (ensemble model_types →
  `'ensemble'`, all existing cases unchanged) and the two new helpers
  `getEnsembleSubTask` / `getEnsembleStrategy`.
- Update/extend existing task-tab component tests (`JobsDrawer`,
  `pages/Jobs.tsx`, `ExperimentsPage`) to assert the new Ensemble
  tab/option renders, filters jobs correctly, and the sub-filter pill
  narrows further.
- Snapshot/assertion update for `JobCard`/`JobDetailsView` badge rendering
  on an ensemble job fixture.

## Out of scope

- No backend/schema changes — `category` is already exposed via the
  existing registry API.
- No changes to how ensemble jobs are trained/scored (that bug was already
  fixed separately in commit `a9674896`).
- No new detail-view content beyond the badge (e.g. no new "base
  estimators list" panel) — purely categorization/filtering/labeling.
