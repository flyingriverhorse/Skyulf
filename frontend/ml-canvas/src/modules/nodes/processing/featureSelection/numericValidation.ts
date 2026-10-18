import { modelThresholdValid, numericIssue } from '../../../../core/utils/numericValidation';
import type { FeatureSelectionConfig } from './types';

/** Check the active selector's numeric domain without interpreting unused settings. */
export function featureSelectionNumericIssue(config: FeatureSelectionConfig) {
  const { method } = config;
  if (['rfecv', 'sequential_feature_selection'].includes(method)) return crossValidatedIssue(config);
  if (method === 'variance_threshold' || method === 'correlation_threshold') {
    return numericIssue('threshold', config.threshold, 0, false, method === 'correlation_threshold' ? 1 : Infinity);
  }
  if (method === 'select_from_model') {
    return modelSelectionNumericIssue(config);
  }
  if (['select_k_best', 'rfe'].includes(method)) {
    return featureCountNumericIssue(config);
  }
  if (method === 'select_percentile') return numericIssue('percentile', config.percentile, 0, false, 100);
  if (['select_fpr', 'select_fdr', 'select_fwe'].includes(method)) return numericIssue('alpha', config.alpha, 0, false, 1);
  if (method === 'generic_univariate_select') {
    return genericSelectionNumericIssue(config);
  }
}

/** Validate common folds before the active cross-validated selector's own controls. */
function crossValidatedIssue(config: FeatureSelectionConfig) {
  const cv = numericIssue('cv', config.cv, 2, true);
  if (cv) return cv;
  if (config.method === 'rfecv') return numericIssue('min_features_to_select', config.min_features_to_select, 1, true)
    ?? numericIssue('step', config.step, Number.MIN_VALUE, (config.step ?? 1) >= 1);
  return sequentialCountIssue(config.n_features_to_select);
}

/** SFS accepts an automatic half selection, a count, or a positive fractional share. */
function sequentialCountIssue(count: FeatureSelectionConfig['n_features_to_select']) {
  if (count === undefined || count === 'auto') return undefined;
  if (!Number.isFinite(count) || count <= 0 || (count >= 1 && !Number.isInteger(count)))
    return { isValid: false, field: 'n_features_to_select', message: 'Select auto, a positive integer count, or a fraction between zero and one.' };
}

/** Validate the model threshold before its optional feature limit. */
function modelSelectionNumericIssue(config: FeatureSelectionConfig) {
  if (!modelThresholdValid(config.threshold)) return { isValid: false, field: 'threshold', message: 'Threshold must be finite, mean, median or a finite multiple of mean/median.' };
  return config.max_features == null ? undefined : numericIssue('max_features', config.max_features, 1, true);
}

/** Preserve selector-specific count sentinels and RFE's fractional step. */
function featureCountNumericIssue(config: FeatureSelectionConfig) {
  const { method } = config;
  const k = config.k as unknown;
  const count = (method === 'select_k_best' && k === 'all') || (method === 'rfe' && k === null)
    ? undefined : numericIssue('k', k, 1, true);
  return count ?? (method === 'rfe' ? numericIssue('step', config.step, Number.MIN_VALUE, (config.step ?? 1) >= 1) : undefined);
}

/** An explicit generic parameter takes precedence over the mode's named field. */
function genericSelectionNumericIssue(config: FeatureSelectionConfig) {
  const mode = config.mode ?? 'k_best';
  const field = config.param !== undefined ? 'param' : mode === 'k_best' ? 'k' : mode === 'percentile' ? 'percentile' : 'alpha';
  return numericIssue(field, config[field], mode === 'k_best' ? 1 : 0, mode === 'k_best', mode === 'k_best' ? Infinity : mode === 'percentile' ? 100 : 1);
}
