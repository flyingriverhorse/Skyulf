import { describe, expect, it } from 'vitest';
import { TrainTestSplitNode } from './TrainTestSplitNode';
import { ClassificationNode } from './ClassificationNode';
import { SegmentationNode } from './SegmentationNode';
import { EnsembleNode } from './EnsembleNode';

describe('modeling validation field destinations', () => {
  /** Switching modes must not silently replace Basic 400/7 with one default candidate. */
  it('requires a search space after switching Basic parameters to Advanced', () => {
    const basic = { ...ClassificationNode.getDefaultConfig(), target_column: 'target',
      hyperparameters: { n_estimators: 400, max_depth: 7 }, n_trials: 50, random_state: 0 };
    expect(ClassificationNode.validate(basic)).toEqual({ isValid: true });
    expect(ClassificationNode.validate({ ...basic, run_mode: 'advanced', search_space: {} }))
      .toMatchObject({ isValid: false, field: 'search_space' });
    expect(ClassificationNode.validate({ ...basic, run_mode: 'advanced', search_space: { max_depth: [7, 10] } }))
      .toEqual({ isValid: true });
    expect(EnsembleNode.validate({ ...EnsembleNode.getDefaultConfig(), target_column: 'target', run_mode: 'advanced' }))
      .toEqual({ isValid: true });
  });
  /** Split errors must lead to the ratio that can correct the reported failure. */
  it.each([
    { test_size: 0, validation_size: 0, field: 'test_size' },
    { test_size: 0.2, validation_size: -0.1, field: 'validation_size' },
    { test_size: 0.6, validation_size: 0.4, field: 'validation_size' },
  ])('targets $field for test=$test_size and validation=$validation_size', ({ field, ...ratios }) => {
    expect(TrainTestSplitNode.validate({ ...TrainTestSplitNode.getDefaultConfig(), ...ratios }))
      .toMatchObject({ isValid: false, field });
  });

  /** Training and clustering use different required fields in the shared factory. */
  it('preserves required-field destinations across model types', () => {
    expect(ClassificationNode.validate(ClassificationNode.getDefaultConfig()))
      .toMatchObject({ isValid: false, field: 'target_column' });
    expect(SegmentationNode.validate({ ...SegmentationNode.getDefaultConfig(), model_type: '' }))
      .toMatchObject({ isValid: false, field: 'model_type' });
    expect(EnsembleNode.validate({ ...EnsembleNode.getDefaultConfig(), target_column: 'target', base_estimators: [] }))
      .toMatchObject({ isValid: false, field: 'base_estimators' });
  });
});
