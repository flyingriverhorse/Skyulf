import type { ScalingConfig } from './types';

export const validateScaling = (config: ScalingConfig) => {
  if (config.columns.length === 0) {
    return { field: 'columns', isValid: false, message: 'Select at least one column' };
  }
  if (config.method === 'minmax') {
    return validateRange('feature_range', config.feature_range_min === undefined ? 0 : config.feature_range_min,
      config.feature_range_max === undefined ? 1 : config.feature_range_max);
  }
  if (config.method === 'robust') {
    return validateRange('quantile_range', config.quantile_range_min === undefined ? 25 : config.quantile_range_min,
      config.quantile_range_max === undefined ? 75 : config.quantile_range_max);
  }
  if (config.method === 'quantile') return validateQuantile(config);
  if (config.method === 'normalize') return validateNormalizer(config);
  return { field: 'columns', isValid: true, message: undefined };
};

function validateNormalizer(config: ScalingConfig) {
  if (!['l1', 'l2', 'max'].includes(config.norm ?? 'l2')) {
    return { field: 'norm', isValid: false, message: 'Choose L1, L2 or maximum norm' };
  }
  return { field: 'columns', isValid: true, message: undefined };
}

function validateQuantile(config: ScalingConfig) {
  const count = config.n_quantiles === undefined ? 1000 : config.n_quantiles;
  if (typeof count !== 'number' || !Number.isInteger(count) || count < 1) {
    return { field: 'n_quantiles', isValid: false, message: 'Quantile count must be a positive integer' };
  }
  if (!['uniform', 'normal'].includes(config.output_distribution ?? 'uniform')) {
    return { field: 'output_distribution', isValid: false, message: 'Choose a uniform or normal distribution' };
  }
  return { field: 'columns', isValid: true, message: undefined };
}

function validateRange(field: string, min: number | null, max: number | null) {
  const label = field === 'feature_range' ? 'Feature range' : 'Quantile range';
  if (!isFiniteBound(min) || !isFiniteBound(max)) {
    return { field, isValid: false, message: `${label} requires two finite numbers` };
  }
  if (field === 'feature_range' && min >= max) {
    return { field, isValid: false, message: 'Feature range minimum must be less than maximum' };
  }
  if (field === 'quantile_range' && !(0 <= min && min <= max && max <= 100)) {
    return { field, isValid: false, message: 'Quantile range must satisfy 0 ≤ minimum ≤ maximum ≤ 100' };
  }
  return { field, isValid: true, message: undefined };
}

function isFiniteBound(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}
