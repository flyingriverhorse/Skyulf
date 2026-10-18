import type { Node } from '@xyflow/react';
import { expect, it } from 'vitest';
import { preprocessingConverters } from './preprocessing';

it.each([
  { group_by: 'region' },
  { group_by: ['region', 'industry'], fallback: 'hierarchical' },
])('preserves group key representation and explicit fallback $group_by', options => {
  /** Saved legacy graphs stay scalar while new hierarchical graphs retain their key order. */
  const node: Node = { id: 'fill', position: { x: 0, y: 0 }, data: { definitionType: 'ImputationNode', method: 'group', columns: ['value'], strategy: 'median', ...options } };
  expect(preprocessingConverters.get('imputation_node')!(node)).toEqual({ stepType: 'GroupImputer', params: { columns: ['value'], strategy: 'median', ...options } });
});

it('forwards deviation as an existing FeatureMath operation', () => {
  /** The selected mean-deviation method must reach the learned core group operation unchanged. */
  const operations = [{ operation_type: 'group_agg', method: 'deviation', input_columns: ['region'], secondary_columns: ['value'], output_column: 'centered' }];
  const node: Node = { id: 'deviation', position: { x: 0, y: 0 }, data: { definitionType: 'FeatureGenerationNode', operations } };
  expect(preprocessingConverters.get('FeatureGenerationNode')!(node)).toMatchObject({ params: { operations } });
});
