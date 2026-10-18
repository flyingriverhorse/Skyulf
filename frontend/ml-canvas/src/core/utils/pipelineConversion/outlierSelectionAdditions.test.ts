import { describe, expect, it } from 'vitest';
import { preprocessingConverters } from './preprocessing';

describe('joint outlier and selection wire settings', () => {
  it.each([['isolation_forest', 'IsolationForest'], ['local_outlier_factor', 'LocalOutlierFactor']] as const)('routes %s to its actual owner', (method, owner) => {
    // Joint methods must never fall through to the legacy IQR fallback.
    const converted = preprocessingConverters.get('outlier')!({ id: 'joint', position: { x: 0, y: 0 }, data: { method, columns: ['a', 'b'], contamination: 0.1, n_neighbors: 8 } });
    expect(converted.stepType).toBe(owner);
    expect(converted.params).toMatchObject({ columns: ['a', 'b'], contamination: 0.1 });
  });
  it('omits inactive percentile overrides after the column picker changes', () => {
    // Hidden deselected-column settings cannot invalidate an otherwise valid recipe.
    const converted = preprocessingConverters.get('outlier')!({ id: 'tails', position: { x: 0, y: 0 }, data: { method: 'winsorize', columns: ['a'], percentiles_by_column: { a: { lower: 1, upper: 99 }, removed: { lower: 5, upper: 95 } } } });
    expect(converted.params.percentiles_by_column).toEqual({ a: { lower: 1, upper: 99 } });
  });
  it.each(['rfecv', 'sequential_feature_selection'])('preserves %s controls in the existing wrapper', (method) => {
    // Selection uses the existing wrapper and forwards only learned-at-fit configuration.
    const converted = preprocessingConverters.get('feature_selection')!({ id: 'select', position: { x: 0, y: 0 }, data: { method, cv: 3, direction: 'backward', n_features_to_select: 2 } });
    expect(converted.stepType).toBe('feature_selection');
    expect(converted.params).toMatchObject({ method, cv: 3, direction: 'backward', n_features_to_select: 2 });
  });
});
