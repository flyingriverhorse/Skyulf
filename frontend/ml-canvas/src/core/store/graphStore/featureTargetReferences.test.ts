import { beforeAll, describe, expect, it } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import { initializeRegistry } from '../../registry/init';
import { collectGraphValidationIssues } from './validation';

beforeAll(initializeRegistry);

const node = (id: string, definitionType: string, config: Record<string, unknown>): Node => ({
  id, type: 'custom', position: { x: 0, y: 0 }, data: { definitionType, ...config },
});

describe('feature-only controls after target separation', () => {
  it.each([
    ['outlier', { method: 'manual_bounds', columns: ['target'], bounds: { target: { upper: 10 } } }, 'bounds'],
    ['GeoDistance', { lat1_col: 'target', lon1_col: 'lon', lat2_col: 'lat', lon2_col: 'lon', method: 'haversine', unit: 'km', output_column: '' }, 'lat1_col'],
  ])('blocks stale %s target selections even before schema preview returns', (type, config, field) => {
    // Saved graphs must not submit a no-op or missing coordinate after moving a split upstream.
    const nodes = [node('load', 'dataset_node', { datasetId: 'data' }),
      node('split', 'feature_target_split', { target_column: 'target' }), node('operation', type, config)];
    const edges: Edge[] = [{ id: 'a', source: 'load', target: 'split' }, { id: 'b', source: 'split', target: 'operation' }];
    const issues = collectGraphValidationIssues(nodes, edges);
    expect(issues).toEqual([expect.objectContaining({ nodeId: 'operation', category: 'configuration', field })]);
    expect(issues[0]?.message).toContain('target');
    expect(issues[0]?.message).toContain('separated');
  });

  it('allows bounds on raw target values before a downstream split', () => {
    // A column is still an available feature until the graph actually separates it into y.
    const nodes = [node('load', 'dataset_node', { datasetId: 'data' }),
      node('operation', 'outlier', { method: 'manual_bounds', columns: ['target'], bounds: { target: { upper: 10 } } }),
      node('split', 'feature_target_split', { target_column: 'target' })];
    const edges: Edge[] = [{ id: 'a', source: 'load', target: 'operation' }, { id: 'b', source: 'operation', target: 'split' }];
    expect(collectGraphValidationIssues(nodes, edges)).toEqual([]);
  });
});
