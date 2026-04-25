import { describe, it, expect, beforeAll, beforeEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useUpstreamData } from './useUpstreamData';
import { useGraphStore } from '../store/useGraphStore';
import { initializeRegistry } from '../registry/init';

describe('useUpstreamData', () => {
  beforeAll(() => initializeRegistry());
  beforeEach(() => {
    useGraphStore.setState({ nodes: [], edges: [], executionResult: null });
  });

  it('returns an empty array when the node id is unknown', () => {
    const { result } = renderHook(() => useUpstreamData('missing'));
    expect(result.current).toEqual([]);
  });

  it('returns the data of direct incomers in order', () => {
    useGraphStore.getState().setGraph(
      [
        { id: 'src', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'ds-1' } },
        { id: 'mid', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'imputation_node' } },
      ],
      [{ id: 'src-mid', source: 'src', target: 'mid' }],
    );
    const { result } = renderHook(() => useUpstreamData('mid'));
    expect(result.current).toHaveLength(1);
    expect(result.current[0]).toMatchObject({ datasetId: 'ds-1' });
  });

  it('threads `datasetId` through transitive ancestors when the direct parent has none', () => {
    // src (has datasetId) → mid (no datasetId) → consumer
    // useUpstreamData('consumer') sees `mid` as the direct incomer but
    // surfaces the inherited datasetId from `src`.
    useGraphStore.getState().setGraph(
      [
        { id: 'src', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'ds-7' } },
        { id: 'mid', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'imputation_node' } },
        { id: 'cons', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'encoding' } },
      ],
      [
        { id: 'src-mid', source: 'src', target: 'mid' },
        { id: 'mid-cons', source: 'mid', target: 'cons' },
      ],
    );
    const { result } = renderHook(() => useUpstreamData('cons'));
    expect(result.current[0]).toMatchObject({ datasetId: 'ds-7' });
  });

  it('handles a node with multiple incomers (one entry per incomer)', () => {
    useGraphStore.getState().setGraph(
      [
        { id: 'a', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'ds-a' } },
        { id: 'b', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'ds-b' } },
        { id: 't', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'basic_training' } },
      ],
      [
        { id: 'a-t', source: 'a', target: 't' },
        { id: 'b-t', source: 'b', target: 't' },
      ],
    );
    const { result } = renderHook(() => useUpstreamData('t'));
    expect(result.current).toHaveLength(2);
    const ids = result.current.map((d) => (d as { datasetId?: string }).datasetId).sort();
    expect(ids).toEqual(['ds-a', 'ds-b']);
  });
});
