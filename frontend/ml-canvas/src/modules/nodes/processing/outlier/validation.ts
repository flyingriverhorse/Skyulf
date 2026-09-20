import type { ManualColumnBounds, OutlierConfig } from './types';
import { numericIssue } from '../../../../core/utils/numericValidation';

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
  if (config.columns.length === 0) {
    return { isValid: false, field: 'columns', message: 'Select at least one column.' };
  }
  if (config.method === 'manual_bounds') {
    for (const column of config.columns) {
      const result = validateManualBound(column, config.bounds?.[column]);
      if (!result.isValid) return result;
    }
  }
  return { isValid: true };
};
