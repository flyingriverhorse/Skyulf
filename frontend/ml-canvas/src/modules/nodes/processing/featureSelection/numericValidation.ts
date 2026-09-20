import { modelThresholdValid, numericIssue } from '../../../../core/utils/numericValidation';
import type { FeatureSelectionConfig } from './types';

/** Check the active selector's numeric domain without interpreting unused settings. */
export function featureSelectionNumericIssue(config: FeatureSelectionConfig) {
  const { method } = config;
  if (method === 'variance_threshold' || method === 'correlation_threshold') {
    return numericIssue('threshold', config.threshold, 0, false, method === 'correlation_threshold' ? 1 : Infinity);
  }
  if (method === 'select_from_model') {
    if (!modelThresholdValid(config.threshold)) return { isValid: false, field: 'threshold', message: 'Threshold must be finite, mean, median or a finite multiple of mean/median.' };
    return config.max_features == null ? undefined : numericIssue('max_features', config.max_features, 1, true);
  }
  if (method === 'select_k_best' || method === 'rfe') {
    const k = config.k as unknown;
    const count = (method === 'select_k_best' && k === 'all') || (method === 'rfe' && k === null)
      ? undefined : numericIssue('k', k, 1, true);
    return count ?? (method === 'rfe' ? numericIssue('step', config.step, Number.MIN_VALUE, (config.step ?? 1) >= 1) : undefined);
  }
  if (method === 'select_percentile') return numericIssue('percentile', config.percentile, 0, false, 100);
  if (['select_fpr', 'select_fdr', 'select_fwe'].includes(method)) return numericIssue('alpha', config.alpha, 0, false, 1);
  if (method === 'generic_univariate_select') {
    const mode = config.mode ?? 'k_best';
    const field = config.param !== undefined ? 'param' : mode === 'k_best' ? 'k' : mode === 'percentile' ? 'percentile' : 'alpha';
    return numericIssue(field, config[field], mode === 'k_best' ? 1 : 0, mode === 'k_best', mode === 'k_best' ? Infinity : mode === 'percentile' ? 100 : 1);
  }
}
