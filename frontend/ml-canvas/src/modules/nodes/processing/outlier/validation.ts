import type { OutlierConfig } from './types';

export const validateOutlier = (config: OutlierConfig) => {
  if (config.columns.length === 0) {
    return { isValid: false, field: 'columns', message: 'Select at least one column.' };
  }
  return { isValid: true };
};
