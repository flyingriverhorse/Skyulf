import { describe, it, expect, beforeAll, beforeEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useRecommendations } from './useRecommendations';
import { useGraphStore } from '../store/useGraphStore';
import { initializeRegistry } from '../registry/init';
import type { PreviewResponse, Recommendation } from '../api/client';

const rec = (over: Partial<Recommendation>): Recommendation => ({
  rule_id: over.rule_id ?? 'r',
  type: over.type ?? 'cleaning',
  target_columns: over.target_columns ?? [],
  description: over.description ?? '',
  suggested_node_type: over.suggested_node_type ?? 'DropMissingColumns',
  suggested_params: over.suggested_params ?? {},
  confidence: over.confidence ?? 1,
  reasoning: over.reasoning ?? '',
});

const seedGraph = () => {
  useGraphStore.getState().setGraph(
    [
      { id: 'src', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'ds-1' } },
      { id: 'tgt', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'imputation_node' } },
    ],
    [{ id: 'e', source: 'src', target: 'tgt' }],
  );
};

const setExecution = (recs: Recommendation[]) => {
  useGraphStore.getState().setExecutionResult({
    pipeline_id: 'p',
    status: 'ok',
    node_results: {},
    preview_data: null,
    recommendations: recs,
  } as unknown as PreviewResponse);
};

describe('useRecommendations', () => {
  beforeAll(() => initializeRegistry());
  beforeEach(() => {
    useGraphStore.setState({ nodes: [], edges: [], executionResult: null });
  });

  it('returns [] when there is no upstream datasetId', () => {
    // No graph → no upstream data.
    setExecution([rec({ rule_id: 'a' })]);
    const { result } = renderHook(() => useRecommendations('tgt'));
    expect(result.current).toEqual([]);
  });

  it('returns [] when executionResult has no recommendations', () => {
    seedGraph();
    const { result } = renderHook(() => useRecommendations('tgt'));
    expect(result.current).toEqual([]);
  });

  it('returns all recommendations when no filters are applied', () => {
    seedGraph();
    setExecution([rec({ rule_id: 'a' }), rec({ rule_id: 'b' })]);
    const { result } = renderHook(() => useRecommendations('tgt'));
    expect(result.current).toHaveLength(2);
  });

  it('filters by `types`', () => {
    seedGraph();
    setExecution([
      rec({ rule_id: 'a', type: 'cleaning' }),
      rec({ rule_id: 'b', type: 'feature_selection' }),
    ]);
    const { result } = renderHook(() =>
      useRecommendations('tgt', { types: ['feature_selection'] }),
    );
    expect(result.current).toHaveLength(1);
    expect(result.current[0]?.rule_id).toBe('b');
  });

  it('filters by `suggestedNodeTypes`', () => {
    seedGraph();
    setExecution([
      rec({ rule_id: 'a', suggested_node_type: 'DropMissingColumns' }),
      rec({ rule_id: 'b', suggested_node_type: 'DropConstantColumns' }),
    ]);
    const { result } = renderHook(() =>
      useRecommendations('tgt', { suggestedNodeTypes: ['DropConstantColumns'] }),
    );
    expect(result.current.map((r) => r.rule_id)).toEqual(['b']);
  });

  it('matches when EITHER `types` OR `suggestedNodeTypes` matches', () => {
    seedGraph();
    setExecution([
      rec({ rule_id: 'a', type: 'cleaning', suggested_node_type: 'X' }),
      rec({ rule_id: 'b', type: 'other', suggested_node_type: 'Y' }),
    ]);
    const { result } = renderHook(() =>
      useRecommendations('tgt', { types: ['cleaning'], suggestedNodeTypes: ['Y'] }),
    );
    expect(result.current.map((r) => r.rule_id).sort()).toEqual(['a', 'b']);
  });

  it('scope=column drops recommendations with no target_columns', () => {
    seedGraph();
    setExecution([
      rec({ rule_id: 'with-cols', target_columns: ['x'] }),
      rec({ rule_id: 'no-cols', target_columns: [] }),
    ]);
    const { result } = renderHook(() => useRecommendations('tgt', { scope: 'column' }));
    expect(result.current.map((r) => r.rule_id)).toEqual(['with-cols']);
  });
});
