import { describe, it, expect } from 'vitest';
import { getTaskForModelType, getJobTypeLabel } from './jobMeta';

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

describe('getJobTypeLabel', () => {
  it('renders "<Type> (basic)" for a basic training job', () => {
    const job = { job_type: 'training', model_type: 'random_forest' };
    expect(getJobTypeLabel(job, registryItems)).toBe('Classification (basic)');
  });

  it('renders "<Type> (advanced)" for a tuned/advanced job', () => {
    const job = { job_type: 'tuning', model_type: 'random_forest', search_strategy: 'random' };
    expect(getJobTypeLabel(job, registryItems)).toBe('Classification (advanced)');
  });

  it('resolves the real family for a regression job rather than the raw "training" mode', () => {
    const job = { job_type: 'training', model_type: 'random_forest' };
    // random_forest is dual-tagged classification/regression in the fixture
    // registry; getTaskForModelType defaults dual-tagged ids to classification.
    expect(getJobTypeLabel(job, registryItems)).not.toBe('training');
  });

  it('falls back to the job_type string for non-model jobs (eda/ingestion)', () => {
    expect(getJobTypeLabel({ job_type: 'eda' }, registryItems)).toBe('eda');
    expect(getJobTypeLabel({ job_type: 'ingestion' }, registryItems)).toBe('ingestion');
  });

  it('labels a legacy job with no resolvable model type as "Training (<mode>)"', () => {
    const job = { job_type: 'training' };
    expect(getJobTypeLabel(job, registryItems)).toBe('Training (basic)');
  });

  it('prefers the server-resolved model_family over re-deriving it from model_type', () => {
    // model_type is intentionally absent from the fixture registry (e.g. `rf_unified`,
    // not yet tagged) — without `model_family` this would degrade to "Training (advanced)".
    const job = { job_type: 'tuning', model_type: 'rf_unified', model_family: 'classification', search_strategy: 'random' };
    expect(getJobTypeLabel(job, registryItems)).toBe('Classification (advanced)');
  });

  it('resolves a basic job with an untagged model_type via model_family', () => {
    const job = { job_type: 'training', model_type: 'rf_unified', model_family: 'classification' };
    expect(getJobTypeLabel(job, registryItems)).toBe('Classification (basic)');
  });

  it('falls back to "Training (<mode>)" when model_family is null/absent and model_type is unresolvable', () => {
    const job = { job_type: 'training', model_type: 'unknown' };
    expect(getJobTypeLabel(job, registryItems)).toBe('Training (basic)');
  });
});
