/** Whether a larger or a smaller value of a metric represents a better model. */
export type MetricDirection = 'higher' | 'lower' | 'unknown';

const HIGHER_IS_BETTER = new Set([
  'accuracy',
  'balanced_accuracy',
  'f1',
  'f1_weighted',
  'f1_macro',
  'f1_micro',
  'roc_auc',
  'roc_auc_weighted',
  'roc_auc_ovr',
  'roc_auc_ovo',
  'roc_auc_ovr_weighted',
  'roc_auc_ovo_weighted',
  'pr_auc',
  'pr_auc_weighted',
  'precision',
  'precision_weighted',
  'precision_macro',
  'precision_micro',
  'recall',
  'recall_weighted',
  'recall_macro',
  'recall_micro',
  'r2',
  'explained_variance',
  'g_score',
  'matthews_corrcoef',
  'best_score',
  'silhouette',
  'silhouette_score',
  'calinski_harabasz',
  'calinski_harabasz_score',
  'iou',
  'dice',
]);

const LOWER_IS_BETTER = new Set([
  'rmse',
  'mae',
  'mse',
  'msle',
  'rmsle',
  'log_loss',
  'mape',
  'smape',
  'median_absolute_error',
  'max_error',
  'davies_bouldin',
  'davies_bouldin_score',
  'inertia',
  'fit_time',
  'score_time',
  'predict_time',
  'training_time',
  'peak_memory_bytes',
]);

const SPLIT_LABELS: Record<string, string> = {
  train: 'Train',
  test: 'Test (held-out)',
  val: 'Validation',
};

/** Strips a split prefix and the cv mean/std suffix, returning the bare metric name. */
function stripSplit(key: string): string {
  if (/^cv_.+_(mean|std)$/.test(key)) return key.replace(/^cv_/, '').replace(/_(mean|std)$/, '');
  if (key.startsWith('cv_')) return key.replace(/^cv_/, '');
  const match = /^(train|test|val)_(.+)$/.exec(key);
  return match?.[2] ?? key;
}

/**
 * Resolves whether a higher or lower value of a metric key is better.
 *
 * Returns 'unknown' for unrecognised metrics so callers can decline to rank
 * rather than silently assuming a direction.
 */
export function getMetricDirection(key: string): MetricDirection {
  if (!key) return 'unknown';

  // Fold-to-fold spread is a consistency measure regardless of the underlying metric.
  if (/^cv_.+_std$/.test(key) || key.endsWith('_std')) return 'lower';

  const base = stripSplit(key).toLowerCase();
  if (HIGHER_IS_BETTER.has(base)) return 'higher';
  if (LOWER_IS_BETTER.has(base)) return 'lower';
  return 'unknown';
}

/** Names the data split a metric was measured on, or null when it carries no split context. */
export function getMetricSplitLabel(key: string): string | null {
  if (key === 'best_score') return 'CV mean';
  if (/^cv_.+_std$/.test(key)) return 'CV std';
  if (key.startsWith('cv_')) return 'CV mean';

  const prefix = /^(train|test|val)_/.exec(key)?.[1];
  return prefix ? (SPLIT_LABELS[prefix] ?? null) : null;
}

/**
 * Returns the index of the best value in a row of per-run values.
 *
 * Yields null when the direction is unknown, when fewer than two runs are
 * comparable, or when the best value is tied — highlighting a "winner" in any
 * of those cases would assert a conclusion the data does not support.
 */
export function pickBestIndex(
  values: readonly (number | undefined | null)[],
  direction: MetricDirection,
): number | null {
  if (direction === 'unknown') return null;

  const comparable = values
    .map((value, index) => ({ value, index }))
    .filter((entry): entry is { value: number; index: number } => typeof entry.value === 'number' && Number.isFinite(entry.value));

  if (comparable.length < 2) return null;

  const best = comparable.reduce((acc, entry) =>
    direction === 'higher'
      ? (entry.value > acc.value ? entry : acc)
      : (entry.value < acc.value ? entry : acc),
  );

  const tied = comparable.filter(entry => entry.value === best.value).length > 1;
  return tied ? null : best.index;
}

/** Which data split a metric key was measured on, from the UI-visibility
 *  perspective. `best_score` is the cross-validated tuning score (its split
 *  label is already "CV mean"), so it belongs to the CV split even though it
 *  carries no `cv_` prefix. Unprefixed keys like `accuracy` or
 *  `silhouette_score` carry no split context. */
export function splitOfMetric(key: string): 'train' | 'test' | 'val' | 'cv' | 'other' {
  if (key === 'best_score' || key.startsWith('cv_')) return 'cv';
  if (key.startsWith('train_')) return 'train';
  if (key.startsWith('test_')) return 'test';
  if (key.startsWith('val_')) return 'val';
  return 'other';
}

/** Hide metric keys whose split the user has toggled off. Keys without a
 *  split context are always kept — there is no checkbox that could hide
 *  them. F-41: `best_score` must be gated by the CV flag together with the
 *  `cv_`-prefixed keys, otherwise "Show CV metrics" leaves it behind. */
export function filterMetricKeysBySplitVisibility(
  keys: string[],
  visibility: { train: boolean; test: boolean; val: boolean; cv: boolean },
): string[] {
  return keys.filter(key => {
    const split = splitOfMetric(key);
    return split === 'other' ? true : visibility[split];
  });
}
