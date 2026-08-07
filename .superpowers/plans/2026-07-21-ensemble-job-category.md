# Ensemble Job Category Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give ensemble jobs (voting/stacking classifiers & regressors) their own "Ensemble" category across the Job History drawer, the full Job History page, and the Experiments page, instead of being merged into Classification/Regression.

**Architecture:** Reuse the codebase's existing `ENSEMBLE_MODEL_TYPES`/`isEnsembleModelType()` helpers (`core/utils/format.ts`) — a pure, synchronous, registry-independent check already used by `JobDetailsView.tsx`'s ensemble summary table — as the single source of truth for "is this an ensemble job", rather than depending on the async node registry's `category` field (a simplification over the original spec draft, found while inspecting `format.ts` during planning: it avoids a registry-load race and keeps detection consistent with the code that already renders ensemble details). Two new pure helpers, `getEnsembleSubTask` and `getEnsembleStrategy`, sit next to `isEnsembleModelType` and derive classification-vs-regression and voting-vs-stacking straight from the `model_type` string (no registry lookup needed, since there are only 4 known ensemble model types). `jobMeta.ts`'s `getTaskForModelType()` checks `isEnsembleModelType()` before its existing tag-based checks. Each of the three UI surfaces (drawer, full page, Experiments page) adds "Ensemble" as a peer of the existing task tabs/option, and the two tab-based surfaces (drawer, full page) add a secondary "All/Classification/Regression" sub-filter pill shown only while the Ensemble tab is active. `JobCard.tsx` and `JobDetailsView.tsx` render a badge (`"Ensemble · Voting/Stacking · Classification/Regression"`) for ensemble jobs.

**Tech Stack:** React + TypeScript (frontend/ml-canvas), Vitest for unit tests, Tailwind CSS for styling.

## Global Constraints

- No backend/schema changes — `model_type` is already available on every job record and is fixed at submission time.
- No changes to ensemble training/scoring behavior (that regression was already fixed in commit `a9674896`).
- Follow existing patterns in each file exactly (styling classes, naming, comment conventions) — do not introduce new UI patterns not already used nearby.
- All new pure functions must have unit tests colocated per existing convention (`*.test.ts` next to the source file).

---

### Task 1: Ensemble sub-task & strategy helpers in `format.ts`

**Files:**
- Modify: `frontend/ml-canvas/src/core/utils/format.ts` (add after the existing `isEnsembleModelType` function, ~line 205)
- Modify: `frontend/ml-canvas/src/core/utils/format.test.ts` (add a new `describe` block)

**Interfaces:**
- Consumes: `ENSEMBLE_MODEL_TYPES` (existing `Set<string>`), `isEnsembleModelType` (existing, `(modelType?: string | null) => boolean`) — both already defined in this file, no changes to either.
- Produces:
  - `getEnsembleSubTask(modelType?: string | null): 'classification' | 'regression' | undefined`
  - `getEnsembleStrategy(modelType?: string | null): 'Voting' | 'Stacking' | undefined`

  Both are consumed by Task 2 (`jobMeta.ts`), Task 3/4 (drawer/page sub-filter pills), and Task 6/7 (badges).

- [ ] **Step 1: Write the failing tests**

Add to `frontend/ml-canvas/src/core/utils/format.test.ts` (append at the end of the file, after the existing `describe('formatBytes', ...)` block):

```ts
describe('getEnsembleSubTask', () => {
  it('returns "classification" for voting_classifier', () => {
    expect(getEnsembleSubTask('voting_classifier')).toBe('classification');
  });

  it('returns "classification" for stacking_classifier', () => {
    expect(getEnsembleSubTask('stacking_classifier')).toBe('classification');
  });

  it('returns "regression" for voting_regressor', () => {
    expect(getEnsembleSubTask('voting_regressor')).toBe('regression');
  });

  it('returns "regression" for stacking_regressor', () => {
    expect(getEnsembleSubTask('stacking_regressor')).toBe('regression');
  });

  it('returns undefined for a non-ensemble model type', () => {
    expect(getEnsembleSubTask('random_forest')).toBeUndefined();
  });

  it('returns undefined for undefined input', () => {
    expect(getEnsembleSubTask(undefined)).toBeUndefined();
  });
});

describe('getEnsembleStrategy', () => {
  it('returns "Voting" for voting_classifier', () => {
    expect(getEnsembleStrategy('voting_classifier')).toBe('Voting');
  });

  it('returns "Voting" for voting_regressor', () => {
    expect(getEnsembleStrategy('voting_regressor')).toBe('Voting');
  });

  it('returns "Stacking" for stacking_classifier', () => {
    expect(getEnsembleStrategy('stacking_classifier')).toBe('Stacking');
  });

  it('returns "Stacking" for stacking_regressor', () => {
    expect(getEnsembleStrategy('stacking_regressor')).toBe('Stacking');
  });

  it('returns undefined for a non-ensemble model type', () => {
    expect(getEnsembleStrategy('logistic_regression')).toBeUndefined();
  });
});
```

Also update the top import line of `format.test.ts` from:

```ts
import { formatMetricName, formatBytes } from './format';
```

to:

```ts
import { formatMetricName, formatBytes, getEnsembleSubTask, getEnsembleStrategy } from './format';
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend/ml-canvas && npx vitest run src/core/utils/format.test.ts`
Expected: FAIL — `getEnsembleSubTask is not defined` / `getEnsembleStrategy is not defined` (or a TS compile error naming the missing exports).

- [ ] **Step 3: Implement the helpers**

In `frontend/ml-canvas/src/core/utils/format.ts`, immediately after the existing:

```ts
export const isEnsembleModelType = (modelType?: string | null): boolean =>
  !!modelType && ENSEMBLE_MODEL_TYPES.has(modelType);
```

add:

```ts
/**
 * Which underlying task ('classification'/'regression') an ensemble model
 * type covers, derived from its name (there are only 4 known ensemble
 * model types — voting/stacking classifier/regressor — so no registry
 * lookup is needed). Returns undefined for non-ensemble model types.
 */
export const getEnsembleSubTask = (modelType?: string | null): 'classification' | 'regression' | undefined => {
  if (!isEnsembleModelType(modelType)) return undefined;
  return modelType!.endsWith('_regressor') ? 'regression' : 'classification';
};

/**
 * Ensemble strategy label ('Voting'/'Stacking') derived from the model
 * type's prefix. Returns undefined for non-ensemble model types.
 */
export const getEnsembleStrategy = (modelType?: string | null): 'Voting' | 'Stacking' | undefined => {
  if (!isEnsembleModelType(modelType)) return undefined;
  return modelType!.startsWith('stacking') ? 'Stacking' : 'Voting';
};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend/ml-canvas && npx vitest run src/core/utils/format.test.ts`
Expected: PASS (all `getEnsembleSubTask`/`getEnsembleStrategy` cases plus all pre-existing `formatMetricName`/`formatBytes` cases).

- [ ] **Step 5: Commit**

```bash
cd frontend/ml-canvas
git add src/core/utils/format.ts src/core/utils/format.test.ts
git commit -m "feat(format): add getEnsembleSubTask/getEnsembleStrategy helpers"
```

---

### Task 2: Add `'ensemble'` to `TaskType` and teach `getTaskForModelType` about it

**Files:**
- Modify: `frontend/ml-canvas/src/core/types/taskType.ts`
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/jobMeta.ts`
- Test: create `frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/jobMeta.test.ts`

**Interfaces:**
- Consumes: `isEnsembleModelType`, `getEnsembleSubTask` from `core/utils/format.ts` (Task 1).
- Produces:
  - `TaskType` now includes `'ensemble'` (used by every task-tab consumer: `useJobStore.ts`, `JobsDrawer.tsx`, `pages/Jobs.tsx`).
  - `getTaskForModelType(modelType, registryItems): ExperimentsTask` — unchanged signature, now also returns `'ensemble'`.
  - `getDisplayScore(job, task)` — unchanged signature; callers (Task 6) must resolve the *effective* metric-priority task themselves before calling it (see Task 6).

- [ ] **Step 1: Write the failing test**

Create `frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/jobMeta.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { getTaskForModelType } from './jobMeta';

const registryItems = [
  { id: 'voting_classifier', tags: ['requires_scaling', 'classification'] },
  { id: 'stacking_classifier', tags: ['requires_scaling', 'classification'] },
  { id: 'voting_regressor', tags: ['requires_scaling', 'regression'] },
  { id: 'stacking_regressor', tags: ['requires_scaling', 'regression'] },
  { id: 'random_forest', tags: ['classification', 'regression'] },
  { id: 'logistic_regression', tags: ['classification', 'text', 'nlp'] },
  { id: 'kmeans', tags: ['clustering'] },
];

describe('getTaskForModelType — ensemble', () => {
  it('resolves voting_classifier to "ensemble" (not "classification")', () => {
    expect(getTaskForModelType('voting_classifier', registryItems)).toBe('ensemble');
  });

  it('resolves stacking_classifier to "ensemble"', () => {
    expect(getTaskForModelType('stacking_classifier', registryItems)).toBe('ensemble');
  });

  it('resolves voting_regressor to "ensemble" (not "regression")', () => {
    expect(getTaskForModelType('voting_regressor', registryItems)).toBe('ensemble');
  });

  it('resolves stacking_regressor to "ensemble"', () => {
    expect(getTaskForModelType('stacking_regressor', registryItems)).toBe('ensemble');
  });
});

describe('getTaskForModelType — pre-existing behavior unchanged', () => {
  it('resolves a plain classifier to "classification"', () => {
    expect(getTaskForModelType('random_forest', registryItems)).toBe('classification');
  });

  it('resolves kmeans to "segmentation"', () => {
    expect(getTaskForModelType('kmeans', registryItems)).toBe('segmentation');
  });

  it('resolves logistic_regression to "classification" (dual-tag default)', () => {
    expect(getTaskForModelType('logistic_regression', registryItems)).toBe('classification');
  });

  it('resolves undefined model type to "other"', () => {
    expect(getTaskForModelType(undefined, registryItems)).toBe('other');
  });

  it('resolves an unknown model type to "other"', () => {
    expect(getTaskForModelType('some_unregistered_model', registryItems)).toBe('other');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend/ml-canvas && npx vitest run src/components/pages/ExperimentsPage/utils/jobMeta.test.ts`
Expected: FAIL — the four "ensemble" cases return `'classification'`/`'regression'` instead of `'ensemble'` (the "pre-existing behavior" cases already pass, since that logic isn't changing).

- [ ] **Step 3: Implement the type and logic changes**

In `frontend/ml-canvas/src/core/types/taskType.ts`, change:

```ts
export type TaskType = 'classification' | 'regression' | 'text_classification' | 'segmentation';
```

to:

```ts
export type TaskType = 'classification' | 'regression' | 'text_classification' | 'segmentation' | 'ensemble';
```

In `frontend/ml-canvas/src/components/pages/ExperimentsPage/utils/jobMeta.ts`, add the import at the top (alongside the existing `TaskType` import):

```ts
import { isEnsembleModelType } from '../../../../core/utils/format';
```

Then change `getTaskForModelType` from:

```ts
export function getTaskForModelType(
  modelType: string | undefined,
  registryItems: { id: string; tags?: string[] }[],
): ExperimentsTask {
  if (!modelType) return 'other';
  const tags = registryItems.find(r => r.id === modelType)?.tags ?? [];
  if (tags.includes('clustering')) return 'segmentation';
  if (modelType === 'logistic_regression') return 'classification';
  if (tags.includes('text') || tags.includes('nlp')) return 'text_classification';
  if (tags.includes('classification')) return 'classification';
  if (tags.includes('regression')) return 'regression';
  return 'other';
}
```

to:

```ts
export function getTaskForModelType(
  modelType: string | undefined,
  registryItems: { id: string; tags?: string[] }[],
): ExperimentsTask {
  if (!modelType) return 'other';
  if (isEnsembleModelType(modelType)) return 'ensemble';
  const tags = registryItems.find(r => r.id === modelType)?.tags ?? [];
  if (tags.includes('clustering')) return 'segmentation';
  if (modelType === 'logistic_regression') return 'classification';
  if (tags.includes('text') || tags.includes('nlp')) return 'text_classification';
  if (tags.includes('classification')) return 'classification';
  if (tags.includes('regression')) return 'regression';
  return 'other';
}
```

Also update the function's doc comment (the block directly above it) to mention the new ensemble check runs first — append this sentence to the existing comment block:

```ts
 * Ensemble model types (voting/stacking classifier/regressor) are checked
 * first via `isEnsembleModelType` and always resolve to `'ensemble'`,
 * regardless of their underlying classification/regression tags.
```

And update the `SCORE_METRIC_PRIORITY` doc comment line (`/** Per-task priority list ... */`) — no code change needed there, since `'ensemble'` jobs never look themselves up in that record (callers resolve the effective sub-task before calling `getDisplayScore`, done in Task 6). Leave `SCORE_METRIC_PRIORITY`'s type and `getDisplayScore`'s `task === 'other' ? [] : SCORE_METRIC_PRIORITY[task]` line untouched in this task — Task 6 handles the caller-side resolution.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend/ml-canvas && npx vitest run src/components/pages/ExperimentsPage/utils/jobMeta.test.ts`
Expected: PASS — all 9 cases green.

- [ ] **Step 5: Run the full frontend test suite to check for regressions**

Run: `cd frontend/ml-canvas && npx vitest run`
Expected: PASS (no new failures; `TaskType`/`getTaskForModelType` are consumed by several files but none should break from an additive union member and an additive early-return branch).

- [ ] **Step 6: Commit**

```bash
cd frontend/ml-canvas
git add src/core/types/taskType.ts src/components/pages/ExperimentsPage/utils/jobMeta.ts src/components/pages/ExperimentsPage/utils/jobMeta.test.ts
git commit -m "feat(jobMeta): resolve ensemble model types to a distinct 'ensemble' task"
```

---

### Task 3: Ensemble tab + sub-filter pill in the Job History drawer (`JobsDrawer.tsx`)

**Files:**
- Modify: `frontend/ml-canvas/src/components/panels/JobsDrawer.tsx`

**Interfaces:**
- Consumes: `TaskType` (now includes `'ensemble'`, Task 2), `getTaskForModelType` (Task 2), `getEnsembleSubTask` (Task 1, imported from `core/utils/format.ts`).
- Produces: no new exports — purely an internal UI change to this component.

- [ ] **Step 1: Add the "Ensemble" tab and its label**

In `frontend/ml-canvas/src/components/panels/JobsDrawer.tsx`, change:

```ts
const TASK_TABS: { task: TaskType; label: string }[] = [
  { task: 'classification', label: 'Classification' },
  { task: 'regression', label: 'Regression' },
  { task: 'text_classification', label: 'Text Classification' },
  { task: 'segmentation', label: 'Segmentation' },
];

const TASK_LABELS: Record<TaskType, string> = {
  classification: 'classification',
  regression: 'regression',
  text_classification: 'text classification',
  segmentation: 'segmentation',
};
```

to:

```ts
const TASK_TABS: { task: TaskType; label: string }[] = [
  { task: 'classification', label: 'Classification' },
  { task: 'regression', label: 'Regression' },
  { task: 'text_classification', label: 'Text Classification' },
  { task: 'segmentation', label: 'Segmentation' },
  { task: 'ensemble', label: 'Ensemble' },
];

const TASK_LABELS: Record<TaskType, string> = {
  classification: 'classification',
  regression: 'regression',
  text_classification: 'text classification',
  segmentation: 'segmentation',
  ensemble: 'ensemble',
};

/** Sub-filter options shown only while the Ensemble tab is active. */
const ENSEMBLE_SUB_FILTERS: { value: 'all' | 'classification' | 'regression'; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'classification', label: 'Classification' },
  { value: 'regression', label: 'Regression' },
];
```

- [ ] **Step 2: Add the import and the sub-filter state**

Add to the import block (alongside the existing `getTaskForModelType` import):

```ts
import { getEnsembleSubTask } from '../../core/utils/format';
```

Add a new state hook near the existing `modelFilter`/`showFilters` state:

```ts
  const [ensembleSubFilter, setEnsembleSubFilter] = useState<'all' | 'classification' | 'regression'>('all');
```

Reset it whenever the active tab changes away from `'ensemble'`, alongside the existing auto-load-attempts reset effect:

```ts
  useEffect(() => {
    autoLoadAttemptsRef.current = 0;
  }, [isDrawerOpen, activeTab]);
```

becomes:

```ts
  useEffect(() => {
    autoLoadAttemptsRef.current = 0;
  }, [isDrawerOpen, activeTab]);

  useEffect(() => {
    if (activeTab !== 'ensemble') setEnsembleSubFilter('all');
  }, [activeTab]);
```

- [ ] **Step 3: Apply the sub-filter to the job list**

Change:

```ts
  const tabJobs = jobs.filter(job => getTaskForModelType(job.model_type, registryItems) === activeTab);
```

to:

```ts
  const tabJobs = jobs
    .filter(job => getTaskForModelType(job.model_type, registryItems) === activeTab)
    .filter(job => activeTab !== 'ensemble' || ensembleSubFilter === 'all' || getEnsembleSubTask(job.model_type) === ensembleSubFilter);
```

- [ ] **Step 4: Render the sub-filter pill row**

In the JSX, immediately after the closing `</div>` of the "Tabs" block and before the "Filter Bar" comment/div, insert:

```tsx
                {/* Ensemble sub-filter pill (only shown on the Ensemble tab) */}
                {activeTab === 'ensemble' && (
                  <div className="flex items-center gap-1.5 px-4 py-2 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
                    {ENSEMBLE_SUB_FILTERS.map(({ value, label }) => (
                      <button
                        key={value}
                        onClick={() => setEnsembleSubFilter(value)}
                        className={`px-2.5 py-1 text-xs rounded-full border transition-colors ${
                          ensembleSubFilter === value
                            ? 'bg-blue-50 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700 text-blue-600 dark:text-blue-400'
                            : 'bg-gray-50 dark:bg-gray-900 border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                )}
```

Place it right before the line `{/* Filter Bar */}`.

- [ ] **Step 5: Manually verify in the dev server**

Run: `cd frontend/ml-canvas && npm run dev` (or the existing dev command), open the app, submit or find an existing ensemble job (voting/stacking classifier or regressor), open Job History, click the new "Ensemble" tab, confirm:
- Only ensemble jobs show up.
- The sub-filter pill row appears and narrows to classification-only/regression-only ensemble jobs correctly.
- All four other tabs are unaffected (no ensemble jobs leak into Classification/Regression anymore).

Stop the dev server once verified.

- [ ] **Step 6: Commit**

```bash
cd frontend/ml-canvas
git add src/components/panels/JobsDrawer.tsx
git commit -m "feat(JobsDrawer): add Ensemble tab with classification/regression sub-filter"
```

---

### Task 4: Ensemble tab + sub-filter pill in the full Job History page (`pages/Jobs.tsx`)

**Files:**
- Modify: `frontend/ml-canvas/src/pages/Jobs.tsx`

**Interfaces:**
- Consumes: same as Task 3 (`TaskType`, `getTaskForModelType`, `getEnsembleSubTask`).
- Produces: no new exports.

- [ ] **Step 1: Add the "Ensemble" tab**

Change:

```ts
const TASK_TABS: { task: TaskType; label: string; icon: React.ReactNode }[] = [
  { task: 'classification', label: 'Classification', icon: <Tags size={16} /> },
  { task: 'regression', label: 'Regression', icon: <TrendingUp size={16} /> },
  { task: 'text_classification', label: 'Text Classification', icon: <FileText size={16} /> },
  { task: 'segmentation', label: 'Segmentation', icon: <Boxes size={16} /> },
];
```

to:

```ts
const TASK_TABS: { task: TaskType; label: string; icon: React.ReactNode }[] = [
  { task: 'classification', label: 'Classification', icon: <Tags size={16} /> },
  { task: 'regression', label: 'Regression', icon: <TrendingUp size={16} /> },
  { task: 'text_classification', label: 'Text Classification', icon: <FileText size={16} /> },
  { task: 'segmentation', label: 'Segmentation', icon: <Boxes size={16} /> },
  { task: 'ensemble', label: 'Ensemble', icon: <Layers size={16} /> },
];
```

Add `Layers` to the `lucide-react` import at the top:

```ts
import {
  Activity, CheckCircle, XCircle, Clock, Search,
  RefreshCw, Database, BarChart2, Filter, Tags, TrendingUp, FileText, Boxes, Layers
} from 'lucide-react';
```

- [ ] **Step 2: Add the import and sub-filter state**

Add alongside the existing `getTaskForModelType` import:

```ts
import { getEnsembleSubTask } from '../core/utils/format';
```

Add a new state hook near `statusFilter`/`showFilters`:

```ts
  const [ensembleSubFilter, setEnsembleSubFilter] = useState<'all' | 'classification' | 'regression'>('all');
```

Reset it alongside the existing `autoLoadAttempts` reset effect:

```ts
  useEffect(() => {
    setAutoLoadAttempts(0);
  }, [activeTab]);
```

becomes:

```ts
  useEffect(() => {
    setAutoLoadAttempts(0);
  }, [activeTab]);

  useEffect(() => {
    if (activeTab !== 'ensemble') setEnsembleSubFilter('all');
  }, [activeTab]);
```

- [ ] **Step 3: Apply the sub-filter to `visibleJobs`**

Change:

```ts
  const visibleJobs = isTaskTab(activeTab)
    ? pool.filter(job => getTaskForModelType(job.model_type, registryItems) === activeTab)
    : jobs;
```

to:

```ts
  const visibleJobs = isTaskTab(activeTab)
    ? pool
        .filter(job => getTaskForModelType(job.model_type, registryItems) === activeTab)
        .filter(job => activeTab !== 'ensemble' || ensembleSubFilter === 'all' || getEnsembleSubTask(job.model_type) === ensembleSubFilter)
    : jobs;
```

- [ ] **Step 4: Render the sub-filter pill row**

In the JSX, the "Tabs & Filters" block currently renders `TASK_TABS.map(...)` followed by the EDA/Ingestion `TabButton`s inside one flex row, then a separate `<div className="flex items-center gap-2">` for search/filters. Add the pill row as a new line directly below the closing `</div>` of the "Tabs & Filters" container (i.e., as a sibling after that whole block, before the `{showFilters && (...)}` block):

```tsx
      {activeTab === 'ensemble' && (
        <div className="flex items-center gap-1.5 px-4 py-2 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
          {(['all', 'classification', 'regression'] as const).map((value) => (
            <button
              key={value}
              onClick={() => setEnsembleSubFilter(value)}
              className={`px-2.5 py-1 text-xs rounded-full border transition-colors capitalize ${
                ensembleSubFilter === value
                  ? 'bg-indigo-50 dark:bg-indigo-900/20 border-indigo-200 dark:border-indigo-800 text-indigo-700 dark:text-indigo-300'
                  : 'bg-slate-50 dark:bg-slate-900 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800'
              }`}
            >
              {value}
            </button>
          ))}
        </div>
      )}
```

- [ ] **Step 5: Manually verify**

Run: `cd frontend/ml-canvas && npm run dev`, navigate to the full Jobs page, click the new "Ensemble" tab, confirm the same behavior as Task 3's manual check (only ensemble jobs shown, sub-filter pill narrows correctly, other tabs unaffected).

- [ ] **Step 6: Commit**

```bash
cd frontend/ml-canvas
git add src/pages/Jobs.tsx
git commit -m "feat(JobsPage): add Ensemble tab with classification/regression sub-filter"
```

---

### Task 5: "Ensemble" option in the Experiments page filter dropdown

**Files:**
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/HeaderAndTabs.tsx`
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx`

**Interfaces:**
- Consumes: `getTaskForModelType` (Task 2) — already imported in `ExperimentsPage.tsx`.
- Produces: no new exports.

- [ ] **Step 1: Widen the `filterType` union in `HeaderAndTabs.tsx`**

Change:

```ts
interface HeaderProps {
  datasets: { id: string; name: string }[];
  selectedDatasetId: string;
  setSelectedDatasetId: (v: string) => void;
  filterType: 'all' | 'classification' | 'regression' | 'text_classification' | 'segmentation';
  setFilterType: (v: 'all' | 'classification' | 'regression' | 'text_classification' | 'segmentation') => void;
}
```

to:

```ts
interface HeaderProps {
  datasets: { id: string; name: string }[];
  selectedDatasetId: string;
  setSelectedDatasetId: (v: string) => void;
  filterType: 'all' | 'classification' | 'regression' | 'text_classification' | 'segmentation' | 'ensemble';
  setFilterType: (v: 'all' | 'classification' | 'regression' | 'text_classification' | 'segmentation' | 'ensemble') => void;
}
```

- [ ] **Step 2: Add the option and widen the `onChange` cast**

Change:

```tsx
      <select
        className="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2.5"
        value={filterType}
        onChange={(e) => { setFilterType(e.target.value as 'all' | 'classification' | 'regression' | 'text_classification' | 'segmentation'); }}
      >
        <option value="all">All Experiments</option>
        <option value="classification">Classification</option>
        <option value="regression">Regression</option>
        <option value="text_classification">Text Classification</option>
        <option value="segmentation">Segmentation</option>
      </select>
```

to:

```tsx
      <select
        className="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2.5"
        value={filterType}
        onChange={(e) => { setFilterType(e.target.value as 'all' | 'classification' | 'regression' | 'text_classification' | 'segmentation' | 'ensemble'); }}
      >
        <option value="all">All Experiments</option>
        <option value="classification">Classification</option>
        <option value="regression">Regression</option>
        <option value="text_classification">Text Classification</option>
        <option value="segmentation">Segmentation</option>
        <option value="ensemble">Ensemble</option>
      </select>
```

- [ ] **Step 3: Widen the `filterType` state type in `ExperimentsPage.tsx`**

Change:

```ts
  const [filterType, setFilterType] = useState<'all' | 'classification' | 'regression' | 'text_classification' | 'segmentation'>('all');
```

to:

```ts
  const [filterType, setFilterType] = useState<'all' | 'classification' | 'regression' | 'text_classification' | 'segmentation' | 'ensemble'>('all');
```

No other change needed in `ExperimentsPage.tsx` — the existing `filteredJobs` `useMemo` already does `getTaskForModelType(job.model_type, registryItems) === filterType`, which now correctly matches `'ensemble'` jobs when `filterType === 'ensemble'` (Task 2 already made `getTaskForModelType` return `'ensemble'`).

- [ ] **Step 4: Manually verify**

Run: `cd frontend/ml-canvas && npm run dev`, open Experiments page, select "Ensemble" from the filter dropdown, confirm only ensemble jobs appear in the sidebar/comparison views and other filter options are unaffected.

- [ ] **Step 5: Commit**

```bash
cd frontend/ml-canvas
git add src/components/pages/ExperimentsPage/components/HeaderAndTabs.tsx src/components/pages/ExperimentsPage.tsx
git commit -m "feat(ExperimentsPage): add Ensemble option to the task filter dropdown"
```

---

### Task 6: Ensemble badge on `JobCard.tsx` (Job History list rows)

**Files:**
- Modify: `frontend/ml-canvas/src/components/panels/jobs/JobCard.tsx`

**Interfaces:**
- Consumes: `isEnsembleModelType`, `getEnsembleSubTask`, `getEnsembleStrategy` (Task 1), `getTaskForModelType`, `getDisplayScore` (Task 2, unchanged signatures).
- Produces: no new exports.

- [ ] **Step 1: Import the new helpers**

Change the existing import line:

```ts
import { formatMetricName, formatDuration } from '../../../core/utils/format';
```

to:

```ts
import { formatMetricName, formatDuration, isEnsembleModelType, getEnsembleSubTask, getEnsembleStrategy } from '../../../core/utils/format';
```

- [ ] **Step 2: Resolve the effective metric-priority task and compute badge info**

Change:

```tsx
export const JobCard: React.FC<JobCardProps> = ({ job, onClick, registryItems }) => {
  const task: ExperimentsTask = getTaskForModelType(job.model_type, registryItems);
  const score = job.status === 'completed' && !job.error ? getDisplayScore(job, task) : null;
```

to:

```tsx
export const JobCard: React.FC<JobCardProps> = ({ job, onClick, registryItems }) => {
  const task: ExperimentsTask = getTaskForModelType(job.model_type, registryItems);
  // Ensemble jobs are scored on their underlying classification/regression
  // metrics — resolve the effective task for metric-priority lookup so
  // getDisplayScore picks the right list (there is no 'ensemble' entry in
  // SCORE_METRIC_PRIORITY).
  const metricTask: ExperimentsTask = task === 'ensemble' ? (getEnsembleSubTask(job.model_type) ?? 'classification') : task;
  const score = job.status === 'completed' && !job.error ? getDisplayScore(job, metricTask) : null;
  const isEnsemble = isEnsembleModelType(job.model_type);
  const ensembleStrategy = getEnsembleStrategy(job.model_type);
  const ensembleSubTask = getEnsembleSubTask(job.model_type);
```

- [ ] **Step 3: Render the badge next to the model type**

Change:

```tsx
      <div className="flex items-center gap-1 mt-0.5 text-[10px] text-gray-500">
        <span className="font-medium truncate">{job.model_type || 'Unknown Model'}</span>
        {job.job_type === 'advanced_tuning' && job.search_strategy && (
          <span className="text-gray-400 truncate">({job.search_strategy})</span>
        )}
      </div>
```

to:

```tsx
      <div className="flex items-center gap-1 mt-0.5 text-[10px] text-gray-500 flex-wrap">
        <span className="font-medium truncate">{job.model_type || 'Unknown Model'}</span>
        {job.job_type === 'advanced_tuning' && job.search_strategy && (
          <span className="text-gray-400 truncate">({job.search_strategy})</span>
        )}
        {isEnsemble && (
          <span className="px-1.5 py-0.5 rounded border bg-violet-50 dark:bg-violet-900/20 text-violet-700 dark:text-violet-300 border-violet-200 dark:border-violet-800 whitespace-nowrap">
            Ensemble · {ensembleStrategy} · {ensembleSubTask === 'regression' ? 'Regression' : 'Classification'}
          </span>
        )}
      </div>
```

- [ ] **Step 4: Manually verify**

Run: `cd frontend/ml-canvas && npm run dev`, open Job History drawer, switch to the Ensemble tab (Task 3), confirm each ensemble job card shows the `"Ensemble · Voting/Stacking · Classification/Regression"` badge and still shows a correctly-computed Score column value (not blank) — this checks the `metricTask` fix.

- [ ] **Step 5: Commit**

```bash
cd frontend/ml-canvas
git add src/components/panels/jobs/JobCard.tsx
git commit -m "feat(JobCard): show Ensemble badge and fix metric priority for ensemble jobs"
```

---

### Task 7: Ensemble badge on `JobDetailsView.tsx` (Job Details header)

**Files:**
- Modify: `frontend/ml-canvas/src/components/panels/jobs/JobDetailsView.tsx`

**Interfaces:**
- Consumes: `isEnsembleModelType`, `getEnsembleSubTask`, `getEnsembleStrategy` (Task 1) — this file already imports several helpers from `core/utils/format` (`formatMetricName, getMetricDescription, extractEnsembleSummary, formatBaseEstimator`); extend that same import line.
- Produces: no new exports.

- [ ] **Step 1: Extend the existing format.ts import**

Change:

```ts
import { formatMetricName, getMetricDescription, extractEnsembleSummary, formatBaseEstimator } from '../../../core/utils/format';
```

to:

```ts
import { formatMetricName, getMetricDescription, extractEnsembleSummary, formatBaseEstimator, isEnsembleModelType, getEnsembleSubTask, getEnsembleStrategy } from '../../../core/utils/format';
```

- [ ] **Step 2: Render the badge in the header, next to the job ID chip**

Change:

```tsx
                    <div>
                        <h2 className="font-semibold text-gray-800 dark:text-gray-100 flex items-center gap-2">
                            Job Details
                            <span className="text-xs font-normal text-gray-500 font-mono bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded">
                                {job.job_id.slice(0, 8)}
                            </span>
                        </h2>
                    </div>
```

to:

```tsx
                    <div>
                        <h2 className="font-semibold text-gray-800 dark:text-gray-100 flex items-center gap-2">
                            Job Details
                            <span className="text-xs font-normal text-gray-500 font-mono bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded">
                                {job.job_id.slice(0, 8)}
                            </span>
                            {isEnsembleModelType(job.model_type) && (
                                <span className="text-xs font-normal px-1.5 py-0.5 rounded border bg-violet-50 dark:bg-violet-900/20 text-violet-700 dark:text-violet-300 border-violet-200 dark:border-violet-800">
                                    Ensemble · {getEnsembleStrategy(job.model_type)} · {getEnsembleSubTask(job.model_type) === 'regression' ? 'Regression' : 'Classification'}
                                </span>
                            )}
                        </h2>
                    </div>
```

- [ ] **Step 3: Manually verify**

Run: `cd frontend/ml-canvas && npm run dev`, open an ensemble job's details (from any of the three surfaces), confirm the badge renders correctly in the header next to the job ID chip, and the existing ensemble summary table further down (the one using `extractEnsembleSummary`) still renders unchanged.

- [ ] **Step 4: Commit**

```bash
cd frontend/ml-canvas
git add src/components/panels/jobs/JobDetailsView.tsx
git commit -m "feat(JobDetailsView): show Ensemble badge in the job details header"
```

---

### Task 8: Full verification pass

**Files:** none (verification only)

- [ ] **Step 1: Run the full frontend test suite**

Run: `cd frontend/ml-canvas && npx vitest run`
Expected: PASS — all pre-existing tests plus the new `format.test.ts` and `jobMeta.test.ts` cases from Tasks 1–2.

- [ ] **Step 2: Run the frontend type checker (if configured)**

Run: `cd frontend/ml-canvas && npx tsc --noEmit` (the project's `"build"` script is `tsc && vite build`; running just `tsc --noEmit` gives the same type-checking signal without a full Vite build)
Expected: no new type errors introduced by the widened `filterType`/`TaskType` unions or the new imports.

- [ ] **Step 3: Manual end-to-end smoke check**

With the dev server running, exercise all three surfaces end-to-end once more in sequence:
1. Job History drawer → Ensemble tab → sub-filter pill → open a job → confirm badge in `JobDetailsView`.
2. Full Jobs page → Ensemble tab → sub-filter pill.
3. Experiments page → filter dropdown → "Ensemble" → confirm the sidebar/comparison view only shows ensemble jobs and their badges are visible in `JobCard` rows.

- [ ] **Step 4: Update the changelog**

Check the repository's changelog file (likely `CHANGELOG.md` at the repo root, per this session's earlier changelog-maintenance work) and add an entry under an "Unreleased"/latest section, e.g.:

```markdown
### Added
- Ensemble jobs (Voting/Stacking Classifier/Regressor) now appear under their own "Ensemble" category in the Job History drawer, the full Jobs page, and the Experiments page filter, instead of being merged into Classification/Regression. Each surface also shows a "Ensemble · Voting/Stacking · Classification/Regression" badge on ensemble job cards and the Job Details header.
```

Commit this alongside no other changes:

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): note the new Ensemble job category"
```
