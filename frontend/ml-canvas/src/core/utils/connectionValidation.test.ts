import { beforeAll, expect, it } from 'vitest';
import type { Node } from '@xyflow/react';
import { initializeRegistry } from '../registry/init';
import { connectionIssue } from './connectionValidation';

beforeAll(() => { initializeRegistry(); });

/** Supply real registered types without requiring mounted canvas nodes. */
function node(id: string, type: string): Node {
  return { id, position: { x: 0, y: 0 }, data: { definitionType: type } };
}

it('uses the same cycle, endpoint, missing-port, and duplicate rules for every entry point', () => {
  // Guidance must not promise a connection that the graph action will reject.
  const nodes = [node('a', 'imputation_node'), node('b', 'encoding'), node('model', 'classification'), node('ensemble', 'EnsembleNode')];
  const connection = { source: 'a', sourceHandle: 'out', target: 'b', targetHandle: 'in' };
  expect(connectionIssue(nodes, [], connection)).toBeNull();
  expect(connectionIssue(nodes, [{ id: 'back', source: 'b', target: 'a' }], connection)).toContain('loop');
  expect(connectionIssue(nodes, [], { ...connection, source: 'model', sourceHandle: 'model' })).toContain('trained model');
  expect(connectionIssue(nodes, [], { source: 'model', sourceHandle: 'model', target: 'ensemble', targetHandle: 'in' })).toBeNull();
  expect(connectionIssue(nodes, [], { ...connection, targetHandle: 'removed' })).toContain('Choose an output');
  expect(connectionIssue(nodes, [{ id: 'existing', ...connection }], connection)).toContain('already connected');
});

it('treats split sets as one connection and permits legacy single-port connections without handle IDs', () => {
  // Another member of an already-connected group must not create a duplicate bundle.
  const nodes = [node('split', 'TrainTestSplitter'), node('model', 'classification')];
  const train = { source: 'split', sourceHandle: 'train', target: 'model', targetHandle: 'in' };
  const edges = [{ id: 'train', ...train }];
  expect(connectionIssue(nodes, edges, { ...train, sourceHandle: 'test' })).toContain('already connected');
  expect(connectionIssue(nodes, edges, { ...train, sourceHandle: 'validation' })).toContain('Validation is disabled');
  expect(connectionIssue([node('a', 'imputation_node'), node('b', 'encoding')], [], { source: 'a', target: 'b', sourceHandle: null, targetHandle: null })).toBeNull();
});
