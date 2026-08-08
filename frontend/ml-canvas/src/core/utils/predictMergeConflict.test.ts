import { describe, expect, it } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import { predictMergeConflict } from './predictMergeConflict';

const node = (id: string, definitionType: string, data: Record<string, unknown> = {}): Node => ({
  id,
  type: 'custom',
  position: { x: 0, y: 0 },
  data: { definitionType, label: id, ...data },
});

const edge = (source: string, target: string): Edge => ({ id: `${source}->${target}`, source, target });

const transform = (id: string, columns: string[], method: string) =>
  node(id, 'TransformationNode', { transformations: [{ columns, method, params: {} }] });

describe('predictMergeConflict', () => {
  it('returns null for a node with a single input', () => {
    const nodes = [node('ds', 'dataset_node'), transform('t1', ['a'], 'log'), node('mi', 'MissingIndicator')];
    const edges = [edge('ds', 't1'), edge('t1', 'mi')];

    expect(predictMergeConflict('mi', nodes, edges)).toBeNull();
  });

  it('returns null when parallel branches write different columns', () => {
    const nodes = [
      node('ds', 'dataset_node'),
      transform('t1', ['SepalLengthCm'], 'log'),
      node('drop', 'drop_missing_columns', { columns: ['Id'] }),
      node('mi', 'MissingIndicator'),
    ];
    const edges = [edge('ds', 't1'), edge('ds', 'drop'), edge('t1', 'mi'), edge('drop', 'mi')];

    expect(predictMergeConflict('mi', nodes, edges)).toBeNull();
  });

  it('reports the shared columns when two branches write the same ones', () => {
    const nodes = [
      node('ds', 'dataset_node'),
      transform('t1', ['SepalLengthCm', 'SepalWidthCm'], 'log'),
      transform('t2', ['SepalLengthCm', 'SepalWidthCm'], 'cube'),
      node('mi', 'MissingIndicator'),
    ];
    const edges = [edge('ds', 't1'), edge('ds', 't2'), edge('t1', 'mi'), edge('t2', 'mi')];

    const conflict = predictMergeConflict('mi', nodes, edges);
    expect(conflict).not.toBeNull();
    expect(conflict!.columns).toEqual(['SepalLengthCm', 'SepalWidthCm']);
    expect(conflict!.branchIds).toEqual(['t1', 't2']);
  });

  it('ignores work done before the branches split, since both inherit it', () => {
    const nodes = [
      node('ds', 'dataset_node'),
      transform('shared', ['SepalLengthCm'], 'log'),
      node('scale', 'scale_numeric_features', { columns: ['PetalLengthCm'] }),
      node('encode', 'encoding', { columns: ['Species'] }),
      node('mi', 'MissingIndicator'),
    ];
    const edges = [
      edge('ds', 'shared'),
      edge('shared', 'scale'),
      edge('shared', 'encode'),
      edge('scale', 'mi'),
      edge('encode', 'mi'),
    ];

    expect(predictMergeConflict('mi', nodes, edges)).toBeNull();
  });

  it('detects a conflict introduced deeper inside each branch', () => {
    const nodes = [
      node('ds', 'dataset_node'),
      node('a1', 'imputation_node', { columns: ['x'] }),
      transform('a2', ['SepalLengthCm'], 'log'),
      node('b1', 'scale_numeric_features', { columns: ['y'] }),
      node('b2', 'encoding', { columns: ['SepalLengthCm'] }),
      node('mi', 'MissingIndicator'),
    ];
    const edges = [
      edge('ds', 'a1'),
      edge('a1', 'a2'),
      edge('ds', 'b1'),
      edge('b1', 'b2'),
      edge('a2', 'mi'),
      edge('b2', 'mi'),
    ];

    const conflict = predictMergeConflict('mi', nodes, edges);
    expect(conflict!.columns).toEqual(['SepalLengthCm']);
  });
});
