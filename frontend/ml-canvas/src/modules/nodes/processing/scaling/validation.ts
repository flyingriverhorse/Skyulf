import type { ScalingConfig } from './types';

export const validateScaling = (config: ScalingConfig) => ({
  field: 'columns',
  isValid: config.columns.length > 0,
  message: config.columns.length === 0 ? 'Select at least one column' : undefined
});
