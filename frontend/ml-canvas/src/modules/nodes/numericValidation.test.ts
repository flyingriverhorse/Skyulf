import { describe, expect, it } from 'vitest';
import { TrainTestSplitNode } from './modeling/TrainTestSplitNode';
import { ClassificationNode } from './modeling/ClassificationNode';
import { EnsembleNode } from './modeling/EnsembleNode';
import { OutlierNode } from './processing/OutlierNode';
import { FeatureSelectionNode } from './processing/FeatureSelectionNode';

const invalidNumbers = [Number.NaN, Infinity, -Infinity, '', null, '1e400'];

describe('numeric node configuration boundaries', () => {
  /** Invalid drafts must block Run rather than becoming null or silently using defaults. */
  it.each(invalidNumbers)('rejects nonfinite or missing numeric values: %s', value => {
    for (const field of ['test_size', 'validation_size']) {
      expect(TrainTestSplitNode.validate({ ...TrainTestSplitNode.getDefaultConfig(), [field]: value }))
        .toMatchObject({ isValid: false, field });
    }
    expect(OutlierNode.validate({ ...OutlierNode.getDefaultConfig(), method: 'iqr', columns: ['x'], multiplier: value as number }))
      .toMatchObject({ isValid: false, field: 'multiplier' });
    expect(FeatureSelectionNode.validate({ method: 'variance_threshold', threshold: value as number }))
      .toMatchObject({ isValid: false, field: 'threshold' });
  });

  /** Integer budgets and fold counts are enforced even for pasted or saved graphs. */
  it.each([...invalidNumbers, 0, -1, 1.5])('rejects invalid trial budgets: %s', n_trials => {
    const config = { ...ClassificationNode.getDefaultConfig(), target_column: 'y', run_mode: 'advanced' as const,
      search_space: { max_depth: [null, 2] }, n_trials: n_trials as number };
    expect(ClassificationNode.validate(config)).toMatchObject({ isValid: false, field: 'n_trials' });
  });

  /** Model-specific and outer CV controls must respect their active domains. */
  it.each([...invalidNumbers, 0, 1, 2.5])('rejects invalid folds: %s', folds => {
    const base = { ...EnsembleNode.getDefaultConfig(), target_column: 'y' };
    expect(EnsembleNode.validate({ ...base, cv_folds: folds as number })).toMatchObject({ isValid: false, field: 'cv_folds' });
    expect(EnsembleNode.validate({ ...base, strategy: 'stacking', cv: folds as number })).toMatchObject({ isValid: false, field: 'cv' });
    expect(EnsembleNode.validate({ ...base, calibrate_base_models: true, calibration_cv: folds as number }))
      .toMatchObject({ isValid: false, field: 'calibration_cv' });
  });

  /** Valid zero seeds, all-core fitting and sklearn threshold sentinels stay available. */
  it('preserves supported special values and inactive fields', () => {
    expect(TrainTestSplitNode.validate({ ...TrainTestSplitNode.getDefaultConfig(), random_state: 0, validation_size: 0 })).toEqual({ isValid: true });
    expect(EnsembleNode.validate({ ...EnsembleNode.getDefaultConfig(), target_column: 'y', n_jobs: -1, cv_enabled: false, cv_folds: 0 })).toEqual({ isValid: true });
    for (const threshold of ['median', 'mean', '1.25*mean', 0]) {
      expect(FeatureSelectionNode.validate({ method: 'select_from_model', target_column: 'y', threshold })).toEqual({ isValid: true });
    }
    expect(ClassificationNode.validate({ ...ClassificationNode.getDefaultConfig(), target_column: 'y', run_mode: 'advanced',
      n_trials: 1, random_state: 0, search_space: { max_depth: [null, 2] } })).toEqual({ isValid: true });
  });

  /** Clearing the last candidate must not leave a named but unrunnable search parameter. */
  it('rejects empty and nonfinite candidate arrays', () => {
    for (const candidates of [[], [Number.NaN], [Infinity]]) {
      expect(ClassificationNode.validate({ ...ClassificationNode.getDefaultConfig(), target_column: 'y', run_mode: 'advanced', search_space: { max_depth: candidates } }))
        .toMatchObject({ isValid: false, field: 'search_space' });
    }
  });

  /** Saving a failed numeric edit must not reinterpret overflow as the valid None candidate. */
  it('preserves blocked search drafts through a JSON graph roundtrip', () => {
    const config = { ...ClassificationNode.getDefaultConfig(), target_column: 'y', run_mode: 'advanced' as const,
      search_space: { max_depth: [null] }, invalid_search_space: { max_depth: '1e400' } };
    const saved = JSON.parse(JSON.stringify(config)) as typeof config;
    expect(ClassificationNode.validate(saved)).toMatchObject({ isValid: false, field: 'search_space' });
    expect(ClassificationNode.validate({ ...saved, invalid_search_space: {} })).toEqual({ isValid: true });
  });

  /** Zero parallel workers is unsupported while negative joblib worker counts are valid. */
  it.each([...invalidNumbers.filter(value => value !== null), 0, 1.5])('rejects invalid ensemble workers: %s', n_jobs => {
    expect(EnsembleNode.validate({ ...EnsembleNode.getDefaultConfig(), target_column: 'y', n_jobs: n_jobs as number }))
      .toMatchObject({ isValid: false, field: 'n_jobs' });
  });

  /** Server-controlled workers and explicitly disabled CV must not block an otherwise valid run. */
  it('ignores overridden workers and disabled advanced CV while allowing joblib None', () => {
    const config = { ...EnsembleNode.getDefaultConfig(), target_column: 'y' };
    expect(EnsembleNode.validate({ ...config, n_jobs: null as unknown as number })).toEqual({ isValid: true });
    expect(EnsembleNode.validate({ ...config, run_mode: 'advanced', n_jobs: 0, cv_enabled: false, cv_folds: 0 }))
      .toEqual({ isValid: true });
  });
});
