import { describe, it, expect } from 'vitest';
import { getTaskForModelType, getJobTypeLabel, mapJobMetricToDropdown, groupJobsByScoringMetric } from './jobMeta';

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

describe('mapJobMetricToDropdown', () => {
  it('maps faithful 1:1 metrics to their dropdown equivalents', () => {
    expect(mapJobMetricToDropdown('accuracy')).toBe('accuracy');
    expect(mapJobMetricToDropdown('f1_weighted')).toBe('f1_weighted');
    expect(mapJobMetricToDropdown('f1')).toBe('f1');
  });

  // F-38: a job tuned on f1_macro was defaulted onto the binary
  // positive-class 'f1' scan — a number up to 0.29 away from the metric
  // the run actually optimised. Macro variants have no faithful
  // threshold-scan equivalent, so they must fall back to the safe
  // default instead of borrowing another metric's identity.
  it('does not conflate f1_macro with the binary positive-class f1', () => {
    expect(mapJobMetricToDropdown('f1_macro')).not.toBe('f1');
    expect(mapJobMetricToDropdown('f1_macro')).toBe('f1_weighted');
  });

  it('does not conflate other macro variants with their binary/weighted forms', () => {
    expect(mapJobMetricToDropdown('precision_macro')).toBe('f1_weighted');
    expect(mapJobMetricToDropdown('recall_macro')).toBe('f1_weighted');
  });

  it('keeps weighted precision/recall mapped to their dropdown entries', () => {
    expect(mapJobMetricToDropdown('precision_weighted')).toBe('precision');
    expect(mapJobMetricToDropdown('recall_weighted')).toBe('recall');
  });

  it('keeps the documented fallbacks for threshold-independent metrics', () => {
    expect(mapJobMetricToDropdown('roc_auc')).toBe('f1_weighted');
    expect(mapJobMetricToDropdown('balanced_accuracy')).toBe('f1_weighted');
    expect(mapJobMetricToDropdown(undefined)).toBe('f1_weighted');
  });
});

describe('groupJobsByScoringMetric', () => {
  // Jobs shaped like the Experiments page reads them: the scoring metric
  // lives on the result (getJobScoringMetric).
  const job = (metric?: string) => ({ result: metric ? { scoring_metric: metric } : {} });

  it('returns a single row when every job shares one metric', () => {
    const rows = groupJobsByScoringMetric([job('accuracy'), job('accuracy')]);
    expect(rows).toEqual([{ metric: 'accuracy', indices: [0, 1] }]);
  });

  // F-36: mixing a basic run (accuracy) with a tuned run (f1_weighted)
  // used to render one "Best Score" row starring across the two numbers.
  // They must be split into one row per metric, first-appearance order.
  it('splits jobs into one row per distinct metric', () => {
    const rows = groupJobsByScoringMetric([
      job('accuracy'),
      job('f1_weighted'),
      job('accuracy'),
    ]);
    expect(rows).toEqual([
      { metric: 'accuracy', indices: [0, 2] },
      { metric: 'f1_weighted', indices: [1] },
    ]);
  });

  it('groups jobs with no resolvable metric under undefined', () => {
    const rows = groupJobsByScoringMetric([job(), job('f1_weighted'), job()]);
    expect(rows).toEqual([
      { metric: undefined, indices: [0, 2] },
      { metric: 'f1_weighted', indices: [1] },
    ]);
  });

  it('returns no rows for an empty selection', () => {
    expect(groupJobsByScoringMetric([])).toEqual([]);
  });
});
