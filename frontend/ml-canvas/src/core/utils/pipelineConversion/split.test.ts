import type { Edge, Node } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import { convertGraphToPipelineConfig } from '../pipelineConverter';

describe('split configuration payload', () => {
  it('sends split settings without Canvas editor fields while retaining routing metadata', () => {
    // Editor fields must not produce unknown-setting warnings in the Core splitter.
    const config = {
      test_size: 0.25,
      validation_size: 0,
      random_state: 0,
      shuffle: false,
      stratify: false,
      target_column: 'target',
    };
    const nodes: Node[] = [
      { id: 'data', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'sample' } },
      { id: 'split', position: { x: 10, y: 0 }, data: {
        definitionType: 'TrainTestSplitter', label: 'Holdout', title: 'Old title',
        ...config,
      } },
    ];
    const edges: Edge[] = [{ id: 'edge', source: 'data', target: 'split' }];
    const before = structuredClone(nodes);
    const result = convertGraphToPipelineConfig(nodes, edges);
    expect(result.nodes.find(node => node.node_id === 'split')?.params).toEqual({
      ...config, _display_name: 'Holdout',
    });
    expect(nodes).toEqual(before);
  });

  it('keeps omitted optional settings absent so Core defaults remain effective', () => {
    // A partial saved graph must not introduce undefined values or replacement defaults.
    const nodes: Node[] = [
      { id: 'data', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'sample' } },
      { id: 'split', position: { x: 10, y: 0 }, data: { definitionType: 'TrainTestSplitter', test_size: 0.3 } },
    ];
    const result = convertGraphToPipelineConfig(nodes, [{ id: 'edge', source: 'data', target: 'split' }]);
    expect(result.nodes.find(node => node.node_id === 'split')?.params).toEqual({ test_size: 0.3 });
  });
});
