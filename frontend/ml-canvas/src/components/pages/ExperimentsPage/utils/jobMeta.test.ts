import { describe, it, expect } from 'vitest';
import { getTaskForModelType } from './jobMeta';

const registryItems = [
  { id: 'voting_classifier', tags: ['requires_scaling', 'classification'] },
  { id: 'stacking_classifier', tags: ['requires_scaling', 'classification'] },
  { id: 'voting_regressor', tags: ['requires_scaling', 'regression'] },
  { id: 'stacking_regressor', tags: ['requires_scaling', 'regression'] },
  { id: 'random_forest', tags: ['classification', 'regression'] },
  { id: 'logistic_regression', tags: ['classification', 'text', 'nlp'] },
  { id: 'kmeans', tags: ['clustering'] },
];

describe('getTaskForModelType — ensemble', () => {
  it('resolves voting_classifier to "ensemble" (not "classification")', () => {
    expect(getTaskForModelType('voting_classifier', registryItems)).toBe('ensemble');
  });

  it('resolves stacking_classifier to "ensemble"', () => {
    expect(getTaskForModelType('stacking_classifier', registryItems)).toBe('ensemble');
  });

  it('resolves voting_regressor to "ensemble" (not "regression")', () => {
    expect(getTaskForModelType('voting_regressor', registryItems)).toBe('ensemble');
  });

  it('resolves stacking_regressor to "ensemble"', () => {
    expect(getTaskForModelType('stacking_regressor', registryItems)).toBe('ensemble');
  });
});

describe('getTaskForModelType — pre-existing behavior unchanged', () => {
  it('resolves a plain classifier to "classification"', () => {
    expect(getTaskForModelType('random_forest', registryItems)).toBe('classification');
  });

  it('resolves kmeans to "segmentation"', () => {
    expect(getTaskForModelType('kmeans', registryItems)).toBe('segmentation');
  });

  it('resolves logistic_regression to "classification" (dual-tag default)', () => {
    expect(getTaskForModelType('logistic_regression', registryItems)).toBe('classification');
  });

  it('resolves undefined model type to "other"', () => {
    expect(getTaskForModelType(undefined, registryItems)).toBe('other');
  });

  it('resolves an unknown model type to "other"', () => {
    expect(getTaskForModelType('some_unregistered_model', registryItems)).toBe('other');
  });
});
