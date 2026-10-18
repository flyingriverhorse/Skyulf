import { describe, expect, it } from 'vitest';
import { preprocessingConverters } from './preprocessing';

describe('additional scaling recipes', () => {
  it.each([
    ['quantile', 'QuantileTransformer', { output_distribution: 'normal', n_quantiles: 20 }],
    ['normalize', 'Normalizer', { norm: 'max' }],
    ['minmax', 'MinMaxScaler', { clip: true }],
    ['robust', 'RobustScaler', { unit_variance: true }],
  ] as const)('preserves %s settings in the exported recipe', (method, owner, options) => {
    // Canvas must name the actual core owner and keep each explicitly configured option.
    const result = preprocessingConverters.get('scale_numeric_features')!({
      id: 'scale', type: 'scale_numeric_features', position: { x: 0, y: 0 },
      data: { definitionType: 'scale_numeric_features', columns: ['x', 'z'], method, ...options },
    });
    expect(result.stepType).toBe(owner);
    expect(result.params).toMatchObject({ columns: ['x', 'z'], ...options });
  });
});
