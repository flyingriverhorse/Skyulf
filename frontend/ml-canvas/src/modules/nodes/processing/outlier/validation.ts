import type { ManualColumnBounds, OutlierConfig } from './types';
import { numericIssue } from '../../../../core/utils/numericValidation';

function percentileIssue(lower: number, upper: number, prefix: string) {
  if (!Number.isFinite(lower) || lower < 0 || lower > 100)
    return { isValid: false, field: `${prefix}lower`, message: 'Lower percentile must be between 0 and 100.' };
  if (!Number.isFinite(upper) || upper > 100 || upper < lower)
    return { isValid: false, field: `${prefix}upper`, message: 'Upper percentile must be between the lower percentile and 100.' };
}

function winsorizeIssue(config: OutlierConfig) {
  const issue = percentileIssue(config.lower_percentile === undefined ? 5 : config.lower_percentile, config.upper_percentile === undefined ? 95 : config.upper_percentile, '');
  if (issue) return { ...issue, field: `${issue.field}_percentile` };
  const map = config.percentiles_by_column;
  if (map !== undefined && (map === null || typeof map !== 'object' || Array.isArray(map)))
    return { isValid: false, field: 'percentiles_by_column', message: 'Percentile overrides must be a column mapping.' };
  for (const column of config.columns) {
    const override = columnPercentileIssue(map?.[column], column);
    if (override) return override;
  }
}

function columnPercentileIssue(pair: { lower: number; upper: number } | undefined, column: string) {
  if (pair === undefined) return undefined;
  if (pair === null || typeof pair !== 'object' || Object.keys(pair).sort().join(',') !== 'lower,upper')
    return { isValid: false, field: 'percentiles_by_column', message: `Enter lower and upper percentiles for ${column}.` };
  return percentileIssue(pair.lower, pair.upper, `percentiles_by_column.${column}.`);
}

function jointDetectorIssue(config: OutlierConfig) {
  const contamination = numericIssue('contamination', config.contamination, Number.MIN_VALUE, false, 0.5);
  if (contamination) return contamination;
  if (config.method === 'local_outlier_factor') return numericIssue('n_neighbors', config.n_neighbors, 1, true);
  return numericIssue('n_estimators', config.n_estimators, 1, true)
    ?? numericIssue('random_state', config.random_state, 0, true, 4294967295);
}

function methodIssue(config: OutlierConfig) {
  if (config.method === 'winsorize') return winsorizeIssue(config);
  if (config.method === 'isolation_forest' || config.method === 'local_outlier_factor') return jointDetectorIssue(config);
}

/** Validate one selected interval without treating zero as an omitted endpoint. */
function validateManualBound(column: string, bound: ManualColumnBounds = {}) {
  const { lower, upper } = bound;
  if (lower == null && upper == null) {
    return { isValid: false, field: `bounds.${column}.lower`, message: `Set a lower or upper bound for ${column}.` };
  }
  if (lower != null && !Number.isFinite(lower)) {
    return { isValid: false, field: `bounds.${column}.lower`, message: `Lower bound for ${column} must be finite.` };
  }
  if (upper != null && !Number.isFinite(upper)) {
    return { isValid: false, field: `bounds.${column}.upper`, message: `Upper bound for ${column} must be finite.` };
  }
  if (lower != null && upper != null && lower > upper) {
    return { isValid: false, field: `bounds.${column}.upper`, message: `Upper bound for ${column} must be greater than or equal to the lower bound.` };
  }
  return { isValid: true };
}

export const validateOutlier = (config: OutlierConfig) => {
  const numeric = config.method === 'iqr'
    ? numericIssue('multiplier', config.multiplier, Number.MIN_VALUE)
    : config.method === 'zscore' ? numericIssue('threshold', config.threshold, Number.MIN_VALUE) : undefined;
  if (numeric) return numeric;
  const options = methodIssue(config);
  if (options) return options;
  if (config.columns.length === 0) {
    return { isValid: false, field: 'columns', message: 'Select at least one column.' };
  }
  if (config.method === 'manual_bounds' || config.method === 'clip') {
    for (const column of config.columns) {
      const result = validateManualBound(column, config.bounds?.[column]);
      if (!result.isValid) return result;
    }
  }
  return { isValid: true };
};
