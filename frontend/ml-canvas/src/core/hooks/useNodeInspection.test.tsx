import { act, cleanup, renderHook } from '@testing-library/react';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Node, Edge } from '@xyflow/react';
import { runPipelinePreview, type PreviewResponse } from '../api/client';
import { initializeRegistry } from '../registry/init';
import { useGraphStore } from '../store/useGraphStore';
import { useViewStore } from '../store/useViewStore';
import { useNodeInspectionStore } from '../store/useNodeInspectionStore';
import { buildPreviewConfiguration, previewConfigurationKey } from '../utils/previewConfiguration';
import { useNodeInspection } from './useNodeInspection';

vi.mock('../api/client', () => ({ runPipelinePreview: vi.fn() }));

/** Keep validation real so inspection uses the same runnable graph as the toolbar. */
function graph(): { nodes: Node[]; edges: Edge[] } {
  return {
    nodes: [
      { id: 'ds', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'ds1' } },
      { id: 'scale', position: { x: 200, y: 0 }, data: { definitionType: 'scale_numeric_features', method: 'standard', columns: ['x'] } },
    ],
    edges: [{ id: 'edge', source: 'ds', sourceHandle: 'data', target: 'scale', targetHandle: 'in' }],
  };
}

/** Represent one measured execution, independent of the terminal preview table. */
function response(): PreviewResponse {
  const side = {
    status: 'available' as const, reason: null,
    tables: [{ port: 'data', split: null, row_count: 80, column_count: 1,
      columns: [{ name: 'x', dtype: 'float64' }], rows: [{ x: -1 }], truncated: true }],
  };
  return { pipeline_id: 'p', run_id: 'run-one', status: 'success', node_results: {},
    preview_data: [], recommendations: [],
    node_inspections: [
      { node_id: 'ds', branch_id: 'branch-1', branch_label: 'Path A',
        input: { status: 'unavailable', reason: 'Source node', tables: [] },
        output: { ...side, tables: [{ ...side.tables[0]!, rows: [{ x: 10 }] }] } },
      { node_id: 'scale', branch_id: 'branch-1', branch_label: 'Path A', input: side, output: side },
    ],
  };
}

/** Publish a toolbar receipt without adding an inspector execution action. */
function runPreview() {
  const { nodes, edges } = useGraphStore.getState();
  return useNodeInspectionStore.getState().runPreview(buildPreviewConfiguration(nodes, edges));
}

describe('selected-node preview receipts', () => {
  beforeAll(() => initializeRegistry());
  beforeEach(() => {
    vi.clearAllMocks();
    useGraphStore.setState({ ...graph(), predictedSchemas: {}, executionResult: null, lastRunError: null });
    useViewStore.setState({ readOnlyOverride: 'off' });
    useNodeInspectionStore.setState({ receipt: null, isLoading: false, error: null });
    vi.mocked(runPipelinePreview).mockResolvedValue(response());
  });
  afterEach(() => cleanup());

  it('captures all nodes and marks results stale only after semantic changes', async () => {
    // Moving, selecting and renaming must not force users to recompute the same data.
    const { result } = renderHook(() => useNodeInspection('scale'));
    await act(async () => { await runPreview(); });
    expect(runPipelinePreview).toHaveBeenCalledWith(expect.objectContaining({ nodes: expect.any(Array) }), { inspectAll: true });
    expect(result.current.runId).toBe('run-one');
    expect(result.current.branches[0]?.output.tables[0]?.row_count).toBe(80);
    act(() => useGraphStore.setState(state => ({ nodes: state.nodes.map(node => ({ ...node,
      selected: true, position: { x: 900, y: 500 }, data: { ...node.data, label: 'Renamed' },
    })) })));
    expect(result.current.isStale).toBe(false);
    act(() => useGraphStore.getState().updateNodeData('scale', { method: 'minmax' }));
    expect(result.current.isStale).toBe(true);
  });

  it('compares a late response with the configuration submitted before awaiting it', async () => {
    // Edits during execution must never make an old response look current.
    let finish!: (value: PreviewResponse) => void;
    vi.mocked(runPipelinePreview).mockReturnValue(new Promise(resolve => { finish = resolve; }));
    const { result } = renderHook(() => useNodeInspection('scale'));
    let pending!: Promise<PreviewResponse>;
    act(() => { pending = runPreview(); });
    expect(result.current.isLoading).toBe(true);
    act(() => useGraphStore.getState().updateNodeData('ds', { datasetId: 'ds2' }));
    await act(async () => { finish(response()); await pending; });
    expect(result.current.isStale).toBe(true);
    expect(result.current.isLoading).toBe(false);
  });

  it('reuses the same run when selecting another node during or after preview', async () => {
    // Every captured node becomes inspectable without retargeting or repeating execution.
    let finish!: (value: PreviewResponse) => void;
    vi.mocked(runPipelinePreview).mockReturnValue(new Promise(resolve => { finish = resolve; }));
    const { result, rerender } = renderHook(({ id }) => useNodeInspection(id), { initialProps: { id: 'scale' } });
    let pending!: Promise<PreviewResponse>;
    act(() => { pending = runPreview(); });
    rerender({ id: 'ds' });
    expect(runPipelinePreview).toHaveBeenCalledOnce();
    await act(async () => { finish(response()); await pending; });
    expect(result.current.branches[0]?.output.tables[0]?.rows).toEqual([{ x: 10 }]);
    expect(result.current.runId).toBe('run-one');
    rerender({ id: 'scale' });
    expect(result.current.runId).toBe('run-one');
    expect(result.current.branches[0]?.output.tables[0]?.rows).toEqual([{ x: -1 }]);
    rerender({ id: 'not-captured' });
    expect(result.current.branches).toEqual([]);
    expect(result.current.runId).toBeNull();
    expect(runPipelinePreview).toHaveBeenCalledOnce();
  });

  it('shares toolbar requests, loading and failures with the inspection view', async () => {
    // A toolbar receipt must be inspectable without another API request.
    const { result } = renderHook(() => useNodeInspection('scale'));
    const { nodes, edges } = graph();
    await act(async () => { await useNodeInspectionStore.getState().runPreview(buildPreviewConfiguration(nodes, edges)); });
    expect(result.current.branches).toHaveLength(1);
    vi.mocked(runPipelinePreview).mockRejectedValueOnce(new Error('Connection lost'));
    await act(async () => { await expect(runPreview()).rejects.toThrow('Connection lost'); });
    expect(result.current.error).toContain('Connection lost');
    expect(result.current.isLoading).toBe(false);
    expect(result.current.branches).toEqual([]);
  });

  it.each([
    { isResultsPanelDismissed: true, isResultsPanelExpanded: false, isResultsPanelMaximized: false },
    { isResultsPanelDismissed: false, isResultsPanelExpanded: false, isResultsPanelMaximized: false },
    { isResultsPanelDismissed: false, isResultsPanelExpanded: true, isResultsPanelMaximized: false },
  ])('preserves Results content and visibility when browsing inspections: %j', (visibility) => {
    // Switching inspected nodes must not reopen a closed pane or replace a result being read.
    const previous = { ...response(), run_id: 'previous-global-result' };
    useGraphStore.setState({ executionResult: previous, lastRunError: 'Previous global error' });
    useViewStore.setState(visibility);
    useNodeInspectionStore.setState({ receipt: { response: response(),
      configurationKey: previewConfigurationKey(buildPreviewConfiguration(graph().nodes, graph().edges)) } });
    const { result, rerender } = renderHook(({ id }) => useNodeInspection(id), { initialProps: { id: 'scale' } });
    expect(result.current.runId).toBe('run-one');
    rerender({ id: 'ds' });
    expect(result.current.branches[0]?.output.tables[0]?.rows).toEqual([{ x: 10 }]);
    expect(useGraphStore.getState().executionResult).toBe(previous);
    expect(useGraphStore.getState().lastRunError).toBe('Previous global error');
    expect(useViewStore.getState()).toMatchObject(visibility);
    expect(runPipelinePreview).not.toHaveBeenCalled();
  });

  it('explains read-only and invalid graphs without initiating requests', () => {
    // The inspector observes execution eligibility but provides no execution action itself.
    useViewStore.setState({ readOnlyOverride: 'on' });
    const { result } = renderHook(() => useNodeInspection('scale'));
    expect(result.current.blockReason).toMatch(/read.only/i);
    act(() => { useViewStore.setState({ readOnlyOverride: 'off' }); useGraphStore.setState({ edges: [] }); });
    expect(result.current.blockReason).toMatch(/validation|connect/i);
    expect(result.current).not.toHaveProperty('refresh');
    expect(runPipelinePreview).not.toHaveBeenCalled();
  });

  it('keeps predicted output separate until measured data is captured', () => {
    // A schema prediction has no measured rows or input/output pair.
    useGraphStore.setState({ predictedSchemas: { scale: { columns: ['x'], dtypes: { x: 'float64' } } } });
    const { result } = renderHook(() => useNodeInspection('scale'));
    expect(result.current.predictedSchema?.columns).toEqual(['x']);
    expect(result.current.branches).toEqual([]);
    expect(result.current.runId).toBeNull();
  });

  it('ignores generated IDs and object-key order but preserves input order and parameter arrays', () => {
    // Merge precedence and selected columns can change data even when the same nodes remain.
    const { nodes, edges } = graph();
    const config = buildPreviewConfiguration(nodes, edges);
    const equivalent = { ...config, pipeline_id: 'different', nodes: [...config.nodes].reverse() };
    expect(previewConfigurationKey(config)).toBe(previewConfigurationKey(equivalent));
    const changed = { ...config, nodes: config.nodes.map(node => node.node_id === 'scale'
      ? { ...node, inputs: ['left', 'right'], params: { ...node.params, columns: ['y', 'x'] } } : node) };
    expect(previewConfigurationKey(config)).not.toBe(previewConfigurationKey(changed));
    const reversed = { ...changed, nodes: changed.nodes.map(node => ({ ...node, inputs: [...node.inputs].reverse() })) };
    expect(previewConfigurationKey(changed)).not.toBe(previewConfigurationKey(reversed));
  });

  it('ignores feature-operation disclosure state without dropping real column mappings', () => {
    // Expanding a settings row is presentation, but a column named isExpanded is real data.
    const config = { pipeline_id: 'p', nodes: [{ node_id: 'math', step_type: 'FeatureMath', inputs: ['ds'],
      params: { operations: [{ method: 'add', input_columns: ['x'], isExpanded: true }] } }] };
    const collapsed = { ...config, nodes: [{ ...config.nodes[0]!, params: {
      operations: [{ method: 'add', input_columns: ['x'], isExpanded: false }],
    } }] };
    expect(previewConfigurationKey(config)).toBe(previewConfigurationKey(collapsed));
    const mapping = { ...config, nodes: [{ ...config.nodes[0]!, step_type: 'Casting',
      params: { column_types: { isExpanded: 'int64' } } }] };
    const changed = { ...mapping, nodes: [{ ...mapping.nodes[0]!, params: { column_types: { isExpanded: 'str' } } }] };
    expect(previewConfigurationKey(mapping)).not.toBe(previewConfigurationKey(changed));
  });
});
