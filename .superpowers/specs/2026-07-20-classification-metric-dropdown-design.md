# Classification Metric Dropdown for Best-Threshold Optimization

## Problem

In the Experiments page's Model Evaluation tab (classification jobs only), a
"★ best {metric}: {threshold}" badge shows the threshold that maximizes some
metric for the selected class, per visible split. Today this badge is
mislabeled: it always internally optimizes plain **F1** regardless of what
label it shows (whatever `getJobScoringMetric()` returns for the job, e.g.
"accuracy" or "roc_auc"). There's no way for the user to choose which metric
drives the threshold search.

Regression and Segmentation are explicitly **out of scope**: neither has a
threshold concept (regression predictions are direct numeric outputs;
segmentation has no train/predict split-based decision boundary), so a metric
dropdown there wouldn't change any graph — it was considered and rejected
during design.

## Goals

- Add a "Metric:" dropdown in the classification evaluation controls bar,
  next to the existing Class/Threshold controls.
- Selecting a metric re-runs the best-threshold scan for that metric and
  updates: the ★ badge(s) (value + label) per visible split, and the
  threshold applied to the confusion matrix when a badge is clicked.
- Fix the underlying bug: the optimization actually matches the metric shown.
- Default to the job's own scoring metric (mapped to the closest dropdown
  option), preserving today's intent but now correctly.

## Metric List

`Accuracy`, `F1`, `F1 Weighted`, `Precision`, `Recall`. `ROC AUC` is excluded
— it's threshold-independent (it summarizes performance across all
thresholds at once), so it can't be "optimized" by picking a threshold.

Mirrors backend naming (`skyulf-core/skyulf/modeling/_evaluation/metrics.py`):
- `accuracy` — global, same for binary/multiclass.
- `f1` / `precision` / `recall` (bare) — binary only, positive-class metric
  (matches backend `_add_binary_unweighted_metrics`). For multiclass jobs,
  these fall back to the weighted-average formula (per user direction —
  "weighted ones for multiclass"), since the unweighted binary form doesn't
  exist for >2 classes.
- `f1_weighted` — weighted average across all classes (works for
  binary and multiclass); for multiclass, numerically identical to
  selecting bare `f1` (expected/acceptable, matches backend semantics).

## Approach

Generalize `findBestF1Threshold` (in
`utils/classificationCharts.ts`) into:

```ts
type ThresholdMetric = 'accuracy' | 'f1' | 'f1_weighted' | 'precision' | 'recall';

findBestThreshold(
  y_true: (string|number)[],
  y_proba: YProba,
  targetClass: string|number,
  metric: ThresholdMetric,
): { threshold: number; value: number } | null
```

- **Binary (2 classes):** keep the existing fast O(n) per-candidate scan
  (tp/fp/fn/tn counters), extended to compute all 5 metrics from the same
  counts, then select the one requested. No perf regression vs. today.
- **Multiclass (>2 classes):** reuse the existing `applyThreshold()` helper
  (already builds the full OvR-reassigned confusion matrix per threshold —
  used elsewhere for rendering) per candidate threshold, then derive
  accuracy (trace / total) or weighted precision/recall/f1 (support-weighted
  average over the confusion matrix) from it.
  - Safety cap: if there are more than ~300 unique candidate scores, sample
    down to ~300 evenly-spaced candidates before scanning, to bound the
    O(n² · k) multiclass cost on large validation sets. Binary path is
    unaffected (no such cap needed; O(n) per candidate).

`BestF1Info` → `BestMetricInfo` (rename `f1` field to `value`; keep
`metricName`/`splitLabel`/`threshold`). Call sites (`ExperimentsPage.tsx`,
`EvaluationView.tsx`) updated accordingly — badge text becomes e.g.
"★ test Precision: 0.62".

## UI Changes (`EvaluationView.tsx`)

- New `selectedMetric` dropdown (native `<select>`, matching existing
  Class selector styling) placed between "Class:" and "Threshold:" controls,
  shown only when `evaluationData.problem_type === 'classification'`.
- Options: Accuracy / F1 / F1 Weighted / Precision / Recall.
- Changing it triggers recomputation of `bestMetricInfos` (already a
  `useMemo` in `ExperimentsPage.tsx`, just add `selectedMetric` to its
  dependency array and pass it into `findBestThreshold`).
- Tooltip text updates to name the currently selected metric instead of a
  hardcoded "F1".

## Default Metric Resolution

New helper `mapJobMetricToDropdown(scoringMetric: string | undefined):
ThresholdMetric` in `jobMeta.ts`:
- `f1_weighted` → `f1_weighted`
- `f1` / `f1_macro` → `f1`
- `accuracy` → `accuracy`
- `precision*` → `precision`, `recall*` → `recall`
- anything else unmapped (e.g. `roc_auc`, `balanced_accuracy`) → fallback
  `f1_weighted` (safe default, works for binary and multiclass alike).

`selectedMetric` state initializes from this on job load, but remains a
normal piece of state afterward — user can freely switch it, it doesn't
snap back when switching splits/jobs mid-session unless a new job is
explicitly selected.

## Non-Goals / Explicitly Skipped

- Regression: no threshold concept; a metric dropdown wouldn't change any
  chart. Skipped per design discussion.
- Segmentation: no threshold concept either (silhouette/calinski-harabasz/
  davies-bouldin are direct quality scores, not threshold-dependent).
  Skipped.
- ROC AUC in the dropdown: threshold-independent by definition, excluded
  from the optimizable metric list.

## Testing

- Add/extend unit tests for `findBestThreshold` in
  `classificationCharts.ts` (currently untested) covering: binary metric
  parity with the old `findBestF1Threshold` behavior when `metric='f1'`,
  multiclass weighted-average correctness against a hand-computed small
  confusion matrix, and the >300-candidate sampling safety cap.
- Existing `tsc --noEmit` / `eslint --max-warnings 0` / `vitest run`
  (242/242 baseline) must stay clean after the change.
