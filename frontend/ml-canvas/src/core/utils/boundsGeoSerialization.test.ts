import { afterEach, describe, expect, it } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import { convertGraphToPipelineConfig } from './pipelineConverter';
import { findPreprocessingBeforeSplitIssues, resetLeakageFlags } from './pipelineLeakageValidation';

/** Exercise the same saved-graph and JSON boundaries as preview/training submissions. */
function serializeOperation(definitionType: string, data: Record<string, unknown>) {
  const nodes: Node[] = [
    { id: 'load', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'coordinates' } },
    { id: 'operation', position: { x: 200, y: 0 }, data: { ...data, definitionType } },
    { id: 'split', position: { x: 400, y: 0 }, data: { definitionType: 'TrainTestSplitter', target_column: 'target' } },
  ];
  const edges: Edge[] = [
    { id: 'a', source: 'load', target: 'operation' },
    { id: 'b', source: 'operation', target: 'split' },
  ];
  const pipeline = convertGraphToPipelineConfig(JSON.parse(JSON.stringify(nodes)) as Node[], edges);
  return { pipeline, operation: pipeline.nodes.find(node => node.node_id === 'operation')! };
}

afterEach(resetLeakageFlags);

describe('Canvas bounds and distance payloads', () => {
  it('sends only selected bounds while retaining zero and one-sided limits', () => {
    // Deselected/stale rules must never filter extra rows on the backend.
    const { operation, pipeline } = serializeOperation('outlier', {
      method: 'manual_bounds', columns: ['age', 'balance'], multiplier: 1.5,
      bounds: {
        age: { lower: 18, upper: 65 }, balance: { lower: 0, upper: null },
        obsolete: { lower: 100, upper: 200 },
      },
    });
    expect(operation.step_type).toBe('ManualBounds');
    expect(operation.params).toEqual({ bounds: { age: { lower: 18, upper: 65 }, balance: { lower: 0 } } });
    expect(findPreprocessingBeforeSplitIssues(pipeline.nodes)).toEqual([]);
  });

  it('keeps column names such as __proto__ as literal configured rules', () => {
    // Arbitrary dataset column names must survive the mapping without prototype writes.
    const bounds: unknown = JSON.parse('{"__proto__":{"lower":0},"constructor":{"upper":10}}');
    const { operation } = serializeOperation('outlier', {
      method: 'manual_bounds', columns: ['__proto__', 'constructor'], bounds,
    });
    expect(Object.keys(operation.params.bounds as object)).toEqual(['__proto__', 'constructor']);
    expect(JSON.stringify(operation.params.bounds)).toBe('{"__proto__":{"lower":0},"constructor":{"upper":10}}');
  });

  it('keeps empty selections empty and preserves unresolved selected rules for validation', () => {
    // A cleared selection must not reactivate hidden settings or invent a numeric zero.
    const empty = serializeOperation('outlier', { method: 'manual_bounds', columns: [], bounds: { age: { lower: 18 } } });
    const unresolved = serializeOperation('outlier', { method: 'manual_bounds', columns: ['age'], bounds: {} });
    expect(empty.operation.params).toEqual({ bounds: {} });
    expect(unresolved.operation.params).toEqual({ bounds: { age: {} } });
  });

  it.each(['iqr', 'zscore', 'winsorize', 'elliptic_envelope'])(
    'retains the existing %s route after switching away from manual bounds', method => {
      // Existing outlier choices must keep their backend algorithm and settings.
      const ids = { iqr: 'IQR', zscore: 'ZScore', winsorize: 'Winsorize', elliptic_envelope: 'EllipticEnvelope' };
      const { operation } = serializeOperation('outlier', { method, columns: ['age'], multiplier: 2, bounds: { age: { lower: 18 } } });
      expect(operation.step_type).toBe(ids[method as keyof typeof ids]);
      expect(operation.params).toMatchObject({ method, columns: ['age'], multiplier: 2 });
    },
  );

  it.each([
    ['haversine', 'km', ''], ['haversine', 'mi', 'distance_to_store'],
    ['euclidean', 'km', 'nearby_km'], ['euclidean', 'mi', ''],
  ])('preserves %s/%s distance settings and output %s through saved graphs', (method, unit, output) => {
    // Coordinates, units and output naming must reach the actual Core operation unchanged.
    const config = {
      lat1_col: 'customer_lat', lon1_col: 'customer_lon', lat2_col: 'store_lat', lon2_col: 'store_lon',
      method, unit, output_column: output,
    };
    const { operation, pipeline } = serializeOperation('GeoDistance', { ...config, selected: true });
    expect(operation.step_type).toBe('GeoDistance');
    expect(operation.params).toEqual(config);
    expect(findPreprocessingBeforeSplitIssues(pipeline.nodes)).toEqual([]);
  });
});
