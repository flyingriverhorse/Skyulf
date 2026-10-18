import { expect, it } from 'vitest';
import { CountVectorizerNode, TfidfVectorizerNode, HashingVectorizerNode } from '../../../modules/nodes/processing/VectorizerNodes';
import { preprocessingConverters } from './preprocessing';

it.each([CountVectorizerNode, TfidfVectorizerNode])('forwards $type column mode using the actual Canvas defaults', (definition) => {
  // The saved opt-in must reach Core without inventing a new node type.
  const config = { ...definition.getDefaultConfig(), columns: ['title', 'body'], column_mode: 'per_column' };
  const converted = preprocessingConverters.get(definition.type)!({ id: 'text', position: { x: 0, y: 0 }, data: { ...config, definitionType: definition.type } });
  expect(converted).toEqual({ stepType: definition.type, params: config });
});

it('keeps Hashing vocabulary-free and unchanged', () => {
  // Hashing remains joined and must not acquire a misleading vocabulary option.
  expect(HashingVectorizerNode.getDefaultConfig()).not.toHaveProperty('column_mode');
});
