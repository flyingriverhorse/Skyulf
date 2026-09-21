import type { Node } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import { convertUnknownNode, preprocessingConverters } from './preprocessing';

/** Exercise conversion without registry labels or generated pipeline identifiers. */
function convert(definitionType: string, config: Record<string, unknown>) {
  const node: Node = { id: 'step', position: { x: 0, y: 0 }, data: { definitionType, ...config } };
  return preprocessingConverters.get(definitionType)!(node);
}

describe('preprocessing semantic parameters', () => {
  it('retains the compatibility payload for unknown saved node types', () => {
    /** Known-node cleanup must not discard an extension's opaque configuration. */
    const data = { definitionType: 'custom_node', isExpanded: 'custom', value: 0 };
    expect(convertUnknownNode({ id: 'custom', position: { x: 0, y: 0 }, data }).params).toEqual(data);
  });
  it.each([
    'simple_imputer', 'scale_numeric_features', 'encoding', 'label_encoding',
    'feature_target_split', 'feature_selection', 'outlier', 'ResamplingNode',
    'TimeSeriesNode', 'TextCleaning', 'count_vectorizer', 'tfidf_vectorizer',
    'hashing_vectorizer', 'tokenizer', 'sentence_embedder', 'ValueReplacement',
    'AliasReplacement', 'InvalidValueReplacement',
  ])('omits editor state from %s while retaining method parameters', (definitionType) => {
    // Node dispatch metadata must not become a backend method parameter.
    const params = { columns: ['x'], method: 'standard', custom_setting: 0, options: { isExpanded: 'semantic' } };
    expect(convert(definitionType, params).params).toEqual(params);
  });

  it('omits operation expansion state without mutating or defaulting saved operations', () => {
    // Optional semantic fields and older operations must survive the UI cleanup unchanged.
    const operations = [
      { operation_type: 'arithmetic', method: 'add', input_columns: ['x'], constants: [0], isExpanded: true },
      { operation_type: 'ratio', method: 'ratio', input_columns: ['x', 'y'], output_column: 'ratio' },
    ];
    const before = structuredClone(operations);
    const expected = [
      { operation_type: 'arithmetic', method: 'add', input_columns: ['x'], constants: [0] },
      operations[1],
    ];
    expect(convert('FeatureGenerationNode', { operations }).params).toEqual({ operations: expected });
    expect(convert('FeatureGenerationNode', { operations: expected }).params).toEqual({ operations: expected });
    expect(convert('FeatureGenerationNode', {}).params).toEqual({ operations: undefined });
    expect(operations).toEqual(before);
  });
});
