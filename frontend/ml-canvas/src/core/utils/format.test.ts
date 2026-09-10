import { describe, it, expect } from 'vitest';
import { formatMetricName, formatBytes, getEnsembleSubTask, getEnsembleStrategy, getMetricDescription, extractEnsembleSummary } from './format';

const accuracyDescription = 'Fraction of correctly classified samples. Can be misleading when classes are imbalanced.';

describe('getMetricDescription', () => {
  it.each([
    ['accuracy', accuracyDescription],
    ['train_accuracy', `Train set — ${accuracyDescription}`],
    ['test_accuracy', `Test set (held-out) — ${accuracyDescription}`],
    ['val_accuracy', `Validation set — ${accuracyDescription}`],
    ['cv_accuracy_mean', `Cross-validation mean — ${accuracyDescription}`],
    ['cv_accuracy_std', `Std deviation of accuracy across CV folds — lower means more consistent results. ${accuracyDescription}`],
    ['cv_custom_std', 'Std deviation of custom across cross-validation folds. Lower = more consistent.'],
    ['test_custom', 'Test set (held-out) —'],
    ['cv_custom_mean', 'Cross-validation mean —'],
    ['cv__std', 'Cross-validation mean —'],
    ['validation_accuracy', undefined],
    ['TRAIN_accuracy', undefined],
    ['unknown', undefined],
    ['', undefined],
  ])('preserves the exact description for %s', (key, description) => {
    // Metric tooltips must keep their split context and existing fallback wording.
    expect(getMetricDescription(key!)).toBe(description);
  });

  it('keeps best-score wording and existing inherited-key lookup behavior', () => {
    // Extraction must not accidentally change the shared description lookup contract.
    const description = 'Best cross-validation score found across all tuning trials. This value drove hyperparameter selection.';
    expect(getMetricDescription('best_score')).toBe(description);
    expect(getMetricDescription('cv_best_score')).toBe(`Cross-validation mean — ${description}`);
    expect(getMetricDescription('constructor')).toBe(Object);
    expect(getMetricDescription('test_constructor')).toBe(`Test set (held-out) — ${String(Object)}`);
  });
});

describe('extractEnsembleSummary', () => {
  it.each([undefined, '', 'random_forest', 'voting', 'stacking_unknown'])('omits summaries for unsupported model %s', modelType => {
    // Ordinary models must not acquire stale ensemble configuration rows.
    expect(extractEnsembleSummary(modelType, { base_estimators: ['ridge'] })).toBeNull();
  });

  it('omits missing buckets and preserves empty summary properties', () => {
    // Consumers distinguish an absent configuration from an empty ensemble selection.
    expect(extractEnsembleSummary('voting_classifier', undefined)).toBeNull();
    expect(extractEnsembleSummary('voting_classifier', {})).toStrictEqual({
      baseEstimators: [], finalEstimator: undefined, voting: undefined,
      passthrough: undefined, weights: undefined, nJobs: undefined,
      calibrateBaseModels: undefined, calibrationMethod: undefined, isStacking: false,
    });
  });

  it.each(['voting_classifier', 'voting_regressor'])('preserves numeric values, arrays and voting-only fields for %s', modelType => {
    // Filtering learner IDs must not clone valid weights or surface stale stacking controls.
    const baseEstimators = ['ridge', null, 4, 'random_forest', undefined];
    const weights = [0, NaN, Infinity];
    const summary = extractEnsembleSummary(modelType, {
      base_estimators: baseEstimators, final_estimator: '', voting: 'soft',
      passthrough: true, weights, n_jobs: 0, calibrate_base_models: true, calibration_method: '',
    });
    expect(summary).toStrictEqual({
      baseEstimators: ['ridge', 'random_forest'], finalEstimator: '', voting: 'soft',
      passthrough: undefined, weights, nJobs: 0, calibrateBaseModels: true,
      calibrationMethod: '', isStacking: false,
    });
    expect(summary!.weights).toBe(weights);
    expect(summary!.baseEstimators).not.toBe(baseEstimators);
    expect(baseEstimators).toEqual(['ridge', null, 4, 'random_forest', undefined]);
  });

  it.each(['stacking_classifier', 'stacking_regressor'])('keeps false passthrough and hides voting fields for %s', modelType => {
    // A strategy switch must not expose stale voting rules or weights.
    expect(extractEnsembleSummary(modelType, {
      base_estimators: 'ridge', final_estimator: 'ridge', voting: 'hard',
      weights: [1, 2], passthrough: false, n_jobs: -1,
      calibrate_base_models: false, calibration_method: 'isotonic',
    })).toStrictEqual({
      baseEstimators: [], finalEstimator: 'ridge', voting: undefined,
      passthrough: false, weights: undefined, nJobs: -1, calibrateBaseModels: undefined,
      calibrationMethod: undefined, isStacking: true,
    });
  });

  it('ignores incorrectly typed options without coercing them', () => {
    // Displaying a malformed saved configuration must retain existing type guards.
    expect(extractEnsembleSummary('voting_classifier', {
      base_estimators: {}, final_estimator: 1, voting: false, passthrough: 1,
      weights: [1, '2'], n_jobs: '0', calibrate_base_models: 'true', calibration_method: 1,
    })).toStrictEqual({
      baseEstimators: [], finalEstimator: undefined, voting: undefined,
      passthrough: undefined, weights: undefined, nJobs: undefined,
      calibrateBaseModels: undefined, calibrationMethod: undefined, isStacking: false,
    });
  });

  it('retains inherited options and empty weight identity', () => {
    // Public record inputs preserve existing property access and array identity semantics.
    const weights: number[] = [];
    const bucket = Object.create({ base_estimators: ['ridge'], weights, n_jobs: NaN }) as Record<string, unknown>;
    const summary = extractEnsembleSummary('voting_regressor', bucket);
    expect(summary!.baseEstimators).toEqual(['ridge']);
    expect(summary!.nJobs).toBeNaN();
    expect(summary!.weights).toBe(weights);
  });
});

describe('formatMetricName', () => {
  it('returns empty string for null/undefined/empty input', () => {
    expect(formatMetricName(undefined)).toBe('');
    expect(formatMetricName(null)).toBe('');
    expect(formatMetricName('')).toBe('');
  });

  it('returns the canonical label for known metrics', () => {
    expect(formatMetricName('accuracy')).toBe('Accuracy');
    expect(formatMetricName('roc_auc')).toBe('ROC AUC');
    expect(formatMetricName('r2')).toBe('R²');
    expect(formatMetricName('rmse')).toBe('RMSE');
    expect(formatMetricName('f1_weighted')).toBe('F1 Weighted');
  });

  it('collapses sklearn neg_* aliases to their positive name', () => {
    expect(formatMetricName('neg_mean_squared_error')).toBe('MSE');
    expect(formatMetricName('neg_mean_absolute_error')).toBe('MAE');
    expect(formatMetricName('neg_root_mean_squared_error')).toBe('RMSE');
    expect(formatMetricName('neg_log_loss')).toBe('Log Loss');
  });

  it('falls back to title-cased snake_case for unknown metrics', () => {
    expect(formatMetricName('balanced_accuracy_score')).toBe('Balanced Accuracy Score');
    expect(formatMetricName('custom_metric')).toBe('Custom Metric');
  });
});

describe('formatBytes', () => {
  it('returns "0 Bytes" for zero / falsy / NaN input', () => {
    expect(formatBytes(0)).toBe('0 Bytes');
    expect(formatBytes(NaN)).toBe('0 Bytes');
  });

  it('formats common sizes with the right unit', () => {
    expect(formatBytes(500)).toBe('500 Bytes');
    expect(formatBytes(1024)).toBe('1 KB');
    expect(formatBytes(1024 * 1024)).toBe('1 MB');
    expect(formatBytes(1024 * 1024 * 1024)).toBe('1 GB');
  });

  it('respects the decimals argument', () => {
    expect(formatBytes(1536, 0)).toBe('2 KB'); // 1.5 → rounded
    expect(formatBytes(1536, 2)).toBe('1.5 KB');
    expect(formatBytes(1536, 4)).toBe('1.5 KB');
  });

  it('clamps negative decimals to 0', () => {
    expect(formatBytes(1536, -1)).toBe('2 KB');
  });
});

describe('getEnsembleSubTask', () => {
  it('returns "classification" for voting_classifier', () => {
    expect(getEnsembleSubTask('voting_classifier')).toBe('classification');
  });

  it('returns "classification" for stacking_classifier', () => {
    expect(getEnsembleSubTask('stacking_classifier')).toBe('classification');
  });

  it('returns "regression" for voting_regressor', () => {
    expect(getEnsembleSubTask('voting_regressor')).toBe('regression');
  });

  it('returns "regression" for stacking_regressor', () => {
    expect(getEnsembleSubTask('stacking_regressor')).toBe('regression');
  });

  it('returns undefined for a non-ensemble model type', () => {
    expect(getEnsembleSubTask('random_forest')).toBeUndefined();
  });

  it('returns undefined for undefined input', () => {
    expect(getEnsembleSubTask(undefined)).toBeUndefined();
  });
});

describe('getEnsembleStrategy', () => {
  it('returns "Voting" for voting_classifier', () => {
    expect(getEnsembleStrategy('voting_classifier')).toBe('Voting');
  });

  it('returns "Voting" for voting_regressor', () => {
    expect(getEnsembleStrategy('voting_regressor')).toBe('Voting');
  });

  it('returns "Stacking" for stacking_classifier', () => {
    expect(getEnsembleStrategy('stacking_classifier')).toBe('Stacking');
  });

  it('returns "Stacking" for stacking_regressor', () => {
    expect(getEnsembleStrategy('stacking_regressor')).toBe('Stacking');
  });

  it('returns undefined for a non-ensemble model type', () => {
    expect(getEnsembleStrategy('logistic_regression')).toBeUndefined();
  });
});
