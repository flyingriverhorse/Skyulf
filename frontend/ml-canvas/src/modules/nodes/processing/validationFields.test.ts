import { describe, expect, it } from 'vitest';
import { EncodingNode } from './EncodingNode';
import { FeatureGenerationNode } from './FeatureGenerationNode';
import { TimeSeriesNode } from './TimeSeriesNode';
import { TransformationNode } from './TransformationNode';

describe('processing validation field targets', () => {
  it('targets the specific invalid transformation row', () => {
    // A later invalid row must open its own column selector.
    const result = TransformationNode.validate?.({
      transformations: [
        { method: 'log', columns: ['age'] },
        { method: 'square', columns: [] },
      ],
    });

    expect(result).toEqual({
      isValid: false,
      field: 'transformations.1.columns',
      message: 'Each rule must have at least one column selected.',
    });
  });

  it.each(['arithmetic', 'group_agg'])('targets the missing operand for %s', (operationType) => {
    // Fixing the first operand must move the issue to the second control.
    const firstOperation = { operation_type: 'datetime_extract', method: 'year', input_columns: ['date'] };
    const invalidOperation = { operation_type: operationType, method: 'add', input_columns: [], secondary_columns: [] };
    const before = FeatureGenerationNode.validate?.({ operations: [firstOperation, invalidOperation] });
    const after = FeatureGenerationNode.validate?.({
      operations: [firstOperation, { ...invalidOperation, input_columns: ['age'] }],
    });

    expect(before).toMatchObject({ isValid: false, field: 'operations.1.input_columns' });
    expect(after).toMatchObject({ isValid: false, field: 'operations.1.secondary_columns' });
  });

  it('preserves the supported constant-operand validation behavior', () => {
    // Adding field metadata must not reject valid imported arithmetic configurations.
    expect(FeatureGenerationNode.validate?.({
      operations: [{ operation_type: 'arithmetic', method: 'add', input_columns: ['age'], constants: [2] }],
    })).toEqual({ isValid: true });
  });

  it('targets collection management when there are no operations or rules', () => {
    // Empty collections have no indexed control to focus.
    expect(FeatureGenerationNode.validate?.({ operations: [] })).toMatchObject({ field: 'operations' });
    expect(TransformationNode.validate?.({ transformations: [] })).toMatchObject({ field: 'transformations' });
  });

  it('exposes encoding errors through the shared message contract', () => {
    // The issue panel reads message, so the former error property lost this explanation.
    expect(EncodingNode.validate?.(EncodingNode.getDefaultConfig())).toEqual({
      isValid: false, field: 'columns', message: 'Select at least one column',
    });
    expect(EncodingNode.validate?.({ ...EncodingNode.getDefaultConfig(), method: 'woe', columns: ['city'] })).toEqual({
      isValid: false, field: 'target_column', message: 'WOE encoding requires a binary target column',
    });
  });

  it.each(['label', 'ordinal'] as const)('preserves empty-column target encoding for %s', (method) => {
    // Label and ordinal encoding intentionally allow operating on the target alone.
    expect(EncodingNode.validate?.({ ...EncodingNode.getDefaultConfig(), method })).toEqual({ isValid: true });
  });

  it.each([
    ['lag', 'lags', 'Provide at least one lag value'],
    ['rolling', 'aggregations', 'Select at least one aggregation'],
    ['date', 'features', 'Select at least one calendar feature'],
  ] as const)('exposes the %s-specific field and explanation', (method, field, message) => {
    // Each method must point to its mounted control, with visible explanatory text.
    expect(TimeSeriesNode.validate?.({
      ...TimeSeriesNode.getDefaultConfig(), columns: ['value'], method, [field]: [],
    })).toEqual({ isValid: false, field, message });
  });
});
