// Unit tests for the per-node `bodyPreview` previews introduced in M1
// (`temp/node_body_content_plan.md`). These run without React Flow / DOM
// because each preview is a pure (config) => string|null function.

import { describe, it, expect } from 'vitest';
import { DropColumnsNode } from './processing/DropColumnsNode';
import { DropRowsNode } from './processing/DropRowsNode';
import { DeduplicationNode } from './processing/DeduplicationNode';
import { ImputationNode } from './processing/ImputationNode';
import { OutlierNode } from './processing/OutlierNode';
import { EncodingNode } from './processing/EncodingNode';
import { ScalingNode } from './processing/ScalingNode';
import { BinningNode } from './processing/BinningNode';
import { CastTypeNode } from './processing/CastTypeNode';
import { TransformationNode } from './processing/TransformationNode';
import { FeatureGenerationNode } from './processing/FeatureGenerationNode';
import { FeatureSelectionNode } from './processing/FeatureSelectionNode';
import { ResamplingNode } from './processing/ResamplingNode';
import { TextCleaningNode } from './processing/TextCleaningNode';
import { ValueReplacementNode } from './processing/ValueReplacementNode';
import { AliasReplacementNode } from './processing/AliasReplacementNode';
import { InvalidValueReplacementNode } from './processing/InvalidValueReplacementNode';
import { DataCleaningNode } from './processing/DataCleaningNode';
import { MissingIndicatorNode } from './processing/MissingIndicatorNode';
import { PolynomialFeaturesNode } from './processing/PolynomialFeaturesNode';
import { TrainTestSplitNode } from './modeling/TrainTestSplitNode';
import { FeatureTargetSplitNode } from './modeling/FeatureTargetSplitNode';
import { BasicTrainingNode } from './modeling/BasicTrainingNode';

describe('bodyPreview functions', () => {
  it('DropColumnsNode shows count and threshold', () => {
    expect(DropColumnsNode.bodyPreview!({ columns: [] })).toBeNull();
    expect(DropColumnsNode.bodyPreview!({ columns: ['a', 'b'] })).toBe('Drop 2 cols');
    expect(DropColumnsNode.bodyPreview!({ columns: [], missing_threshold: 50 })).toBe('Drop missing > 50%');
  });

  it('DropRowsNode handles flag + threshold', () => {
    expect(DropRowsNode.bodyPreview!({ drop_if_any_missing: true })).toBe('Drop rows with any missing');
    expect(DropRowsNode.bodyPreview!({ drop_if_any_missing: false, missing_threshold: 30 })).toBe(
      'Drop rows missing > 30%'
    );
    expect(DropRowsNode.bodyPreview!({ drop_if_any_missing: false })).toBeNull();
  });

  it('DeduplicationNode reports subset and keep policy', () => {
    expect(DeduplicationNode.bodyPreview!({ subset: [], keep: 'first' })).toBe('Subset: all · keep first');
    expect(DeduplicationNode.bodyPreview!({ subset: ['x'], keep: 'last' })).toBe('Subset: 1 col · keep last');
  });

  it('ImputationNode shows strategy and column count', () => {
    expect(ImputationNode.bodyPreview!({ columns: [], method: 'simple', strategy: 'mean' })).toBeNull();
    expect(ImputationNode.bodyPreview!({ columns: ['a', 'b'], method: 'simple', strategy: 'median' })).toBe(
      'median · 2 cols'
    );
  });

  it('OutlierNode uppercases method', () => {
    expect(OutlierNode.bodyPreview!({ method: 'iqr', columns: [] })).toBe('IQR');
    expect(OutlierNode.bodyPreview!({ method: 'zscore', columns: ['a', 'b', 'c'] })).toBe('ZSCORE · 3 cols');
  });

  it('EncodingNode and ScalingNode share method+cols pattern', () => {
    expect(EncodingNode.bodyPreview!({ method: 'onehot', columns: [] })).toBe('onehot');
    expect(ScalingNode.bodyPreview!({ method: 'standard', columns: ['x'] })).toBe('standard · 1 col');
  });

  it('BinningNode shows strategy + bin count', () => {
    expect(BinningNode.bodyPreview!({ strategy: 'equal_width', n_bins: 5, columns: [] })).toBe(
      'equal_width · q=5'
    );
    expect(BinningNode.bodyPreview!({ strategy: 'kmeans', n_bins: 3, columns: ['a', 'b'] })).toBe(
      'kmeans · q=3 · 2 cols'
    );
  });

  it('CastTypeNode counts cast columns', () => {
    expect(CastTypeNode.bodyPreview!({ column_types: {} })).toBeNull();
    expect(CastTypeNode.bodyPreview!({ column_types: { a: 'int', b: 'float' } })).toBe('Cast 2 cols');
  });

  it('TransformationNode collapses to single op detail when only one', () => {
    expect(TransformationNode.bodyPreview!({ transformations: [] })).toBeNull();
    expect(
      TransformationNode.bodyPreview!({
        transformations: [{ method: 'log', columns: ['a', 'b'] }],
      })
    ).toBe('log · 2 cols');
    expect(
      TransformationNode.bodyPreview!({
        transformations: [
          { method: 'log', columns: ['a'] },
          { method: 'square', columns: ['b'] },
        ],
      })
    ).toBe('2 ops');
  });

  it('FeatureGenerationNode counts ops', () => {
    expect(FeatureGenerationNode.bodyPreview!({ operations: [] })).toBeNull();
    expect(FeatureGenerationNode.bodyPreview!({ operations: [{}, {}] as never })).toBe('+2 features');
  });

  it('FeatureSelectionNode adapts to method', () => {
    expect(FeatureSelectionNode.bodyPreview!({ method: 'select_k_best', k: 7 })).toBe('select_k_best · k=7');
    expect(FeatureSelectionNode.bodyPreview!({ method: 'select_percentile', percentile: 25 })).toBe(
      'select_percentile · 25%'
    );
    expect(FeatureSelectionNode.bodyPreview!({ method: 'variance_threshold', threshold: 0.01 })).toBe(
      'variance_threshold · σ>0.01'
    );
    expect(FeatureSelectionNode.bodyPreview!({ method: 'rfe' })).toBe('rfe');
  });

  it('ResamplingNode shows method and target', () => {
    expect(ResamplingNode.bodyPreview!({ method: 'smote', target_column: '' } as never)).toBe('SMOTE');
    expect(ResamplingNode.bodyPreview!({ method: 'smote', target_column: 'y' } as never)).toBe('SMOTE → y');
  });

  it('TextCleaningNode counts ops and cols', () => {
    expect(TextCleaningNode.bodyPreview!({ columns: [], operations: [] })).toBeNull();
    expect(TextCleaningNode.bodyPreview!({ columns: ['a'], operations: [{ op: 'trim' } as never] })).toBe(
      '1 op · 1 col'
    );
  });

  it('Replacement variants summarise rule and column counts', () => {
    expect(ValueReplacementNode.bodyPreview!({ columns: [], replacements: [] })).toBeNull();
    expect(
      ValueReplacementNode.bodyPreview!({ columns: ['a'], replacements: [{}, {}, {}] as never })
    ).toBe('3 rules · 1 col');
    expect(AliasReplacementNode.bodyPreview!({ columns: [], mode: 'custom', custom_pairs: {} })).toBe('custom');
    expect(
      InvalidValueReplacementNode.bodyPreview!({ columns: ['x', 'y'], mode: 'negative_to_nan' })
    ).toBe('negative_to_nan · 2 cols');
  });

  it('DataCleaningNode and MissingIndicatorNode show counts', () => {
    expect(DataCleaningNode.bodyPreview!({ dropColumns: [], fillStrategy: 'mean' })).toBe('Fill missing: mean');
    expect(DataCleaningNode.bodyPreview!({ dropColumns: ['a', 'b'], fillStrategy: 'median' })).toBe(
      'Drop 2 cols · fill median'
    );
    expect(MissingIndicatorNode.bodyPreview!({ columns: [], flag_suffix: '_was_missing' })).toBeNull();
    expect(MissingIndicatorNode.bodyPreview!({ columns: ['a'], flag_suffix: '_was_missing' })).toBe(
      '1 col → flags'
    );
  });

  it('PolynomialFeaturesNode shows degree', () => {
    expect(PolynomialFeaturesNode.bodyPreview!({ columns: [], degree: 2 })).toBe('degree=2');
    expect(PolynomialFeaturesNode.bodyPreview!({ columns: ['a', 'b'], degree: 3 })).toBe('degree=3 · 2 cols');
  });

  it('TrainTestSplitNode formats ratios', () => {
    expect(TrainTestSplitNode.bodyPreview!({ test_size: 0.2, validation_size: 0, random_state: 42, stratify: false, shuffle: true })).toBe(
      '0.8 / 0.2'
    );
    expect(TrainTestSplitNode.bodyPreview!({ test_size: 0.15, validation_size: 0.15, random_state: 42, stratify: false, shuffle: true })).toBe(
      '0.7 / 0.15 / 0.15'
    );
  });

  it('FeatureTargetSplitNode shows target or prompt', () => {
    expect(FeatureTargetSplitNode.bodyPreview!({ target_column: '' })).toBe('Set target');
    expect(FeatureTargetSplitNode.bodyPreview!({ target_column: 'y' })).toBe('target: y');
  });

  it('createModelingNode default preview shows model and target', () => {
    expect(BasicTrainingNode.bodyPreview!({ model_type: 'random_forest_classifier', target_column: 'y' })).toBe(
      'random_forest_classifier → y'
    );
    expect(BasicTrainingNode.bodyPreview!({ model_type: 'random_forest_classifier' })).toBe(
      'random_forest_classifier'
    );
    expect(BasicTrainingNode.bodyPreview!({})).toBeNull();
  });
});
