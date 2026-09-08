import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, renderHook } from '@testing-library/react';
import { useRunControls } from './useRunControls';
import { useGraphStore } from '../../../../core/store/useGraphStore';
import { useNodeInspectionStore } from '../../../../core/store/useNodeInspectionStore';
import { buildPreviewConfiguration } from '../../../../core/utils/previewConfiguration';
import { initializeRegistry } from '../../../../core/registry/init';
import { useJobStore } from '../../../../core/store/useJobStore';
import { useNotificationsStore } from '../../../../core/store/useNotificationsStore';
import { useViewStore } from '../../../../core/store/useViewStore';
import { runPipelinePreview } from '../../../../core/api/client';
import { jobsApi } from '../../../../core/api/jobs';
import { RUN_PREVIEW_EVENT } from '../../../../core/hooks/useKeyboardShortcuts';
import type { Node, Edge } from '@xyflow/react';

const originalStartPolling = useJobStore.getState().startPolling;

/** A valid preprocessing graph keeps submission tests on the real validation path. */
function previewGraph(): { nodes: Node[]; edges: Edge[] } {
  return {
    nodes: [
      { id: 'dataset', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'ds-1' } },
      { id: 'drop', position: { x: 200, y: 0 }, data: { definitionType: 'drop_missing_columns', columns: ['id'], missing_threshold: 0 } },
    ],
    edges: [{ id: 'edge', source: 'dataset', sourceHandle: 'data', target: 'drop', targetHandle: 'in' }],
  };
}

vi.mock('../../../../core/api/client', () => ({
  runPipelinePreview: vi.fn(),
}));

vi.mock('../../../../core/api/jobs', () => ({
  jobsApi: {
    runPipeline: vi.fn(),
    getJobs: vi.fn().mockResolvedValue([]),
    getJob: vi.fn().mockRejectedValue(new Error('Job snapshot is not available yet')),
  },
}));

describe('useRunControls', () => {
  beforeAll(() => initializeRegistry());

  beforeEach(() => {
    vi.clearAllMocks();
    useNodeInspectionStore.setState({ receipt: null, isLoading: false, error: null });
    vi.mocked(runPipelinePreview).mockReset();
    vi.mocked(jobsApi.runPipeline).mockReset();
    useNotificationsStore.getState().clear();
    useViewStore.setState({ readOnlyOverride: 'off', isResultsPanelExpanded: false, leakageNotice: null });
    useGraphStore.setState({
      nodes: [],
      edges: [],
      executionResult: null,
      lastRunError: null,
    });
    useJobStore.setState({
      jobs: [],
      activeParallelRun: null,
      inspectedRun: null,
      isDrawerOpen: false,
      // Polling opens network/WebSocket subscriptions; submission state remains real.
      startPolling: vi.fn(),
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    useJobStore.setState({ startPolling: originalStartPolling });
  });

  it('explains a blocked keyboard preview using the same validation flow as clicking', async () => {
    // A shortcut must explain what to fix instead of silently ignoring an invalid graph.
    const { RUN_PREVIEW_EVENT } = await import('../../../../core/hooks/useKeyboardShortcuts');
    const { runPipelinePreview } = await import('../../../../core/api/client');
    renderHook(() => useRunControls());
    act(() => window.dispatchEvent(new CustomEvent(RUN_PREVIEW_EVENT)));
    expect(useNotificationsStore.getState().items[0]).toMatchObject({ message: expect.stringContaining('Preview blocked'), action: { type: 'preview' } });
    expect(useNotificationsStore.getState().items).toHaveLength(1);
    expect(runPipelinePreview).not.toHaveBeenCalled();
  });

  it('captures the graph even when no node is selected when the toolbar previews data', async () => {
    // One toolbar run should populate Input / Output for any node selected afterward.
    const graph = previewGraph();
    useGraphStore.setState(graph);
    vi.mocked(runPipelinePreview).mockResolvedValueOnce({ pipeline_id: 'p', run_id: 'r', status: 'success', node_results: {}, preview_data: null, recommendations: [] });
    const { result } = renderHook(() => useRunControls());
    await act(async () => { await result.current.handleRun(); });
    expect(runPipelinePreview).toHaveBeenCalledWith(expect.any(Object), { inspectAll: true });
    expect(useNodeInspectionStore.getState().receipt?.response.run_id).toBe('r');
  });

  it('reflects an inspection request already running and prevents another toolbar request', async () => {
    // The panel and toolbar must share one pending preview instead of racing results.
    const graph = previewGraph();
    useGraphStore.setState(graph);
    let finish!: (value: Awaited<ReturnType<typeof runPipelinePreview>>) => void;
    vi.mocked(runPipelinePreview).mockReturnValueOnce(new Promise(resolve => { finish = resolve; }));
    const { result } = renderHook(() => useRunControls());
    let pending!: Promise<unknown>;
    act(() => { pending = useNodeInspectionStore.getState().runPreview(buildPreviewConfiguration(graph.nodes, graph.edges)); });
    const running = result.current.isRunning;
    await act(async () => { await result.current.handleRun(); });
    await act(async () => { finish({ pipeline_id: 'p', status: 'success', node_results: {}, preview_data: null, recommendations: [] }); await pending; });
    expect(running).toBe(true);
    expect(runPipelinePreview).toHaveBeenCalledOnce();
  });

  it('submits the graph and selection present at activation before React rerenders', async () => {
    // A shortcut immediately after an edit must validate and execute the same graph snapshot.
    useGraphStore.setState(previewGraph());
    vi.mocked(runPipelinePreview).mockResolvedValueOnce({ pipeline_id: 'p', status: 'success', node_results: {}, preview_data: null, recommendations: [] });
    const { result } = renderHook(() => useRunControls());
    await act(async () => {
      useGraphStore.setState(state => ({ nodes: state.nodes.map(node => node.id === 'dataset'
        ? { ...node, selected: true, data: { ...node.data, datasetId: 'ds-2' } } : node) }));
      await result.current.handleRun();
    });
    expect(runPipelinePreview).toHaveBeenCalledWith(expect.objectContaining({
      metadata: { dataset_source_id: 'ds-2' },
    }), { inspectAll: true });
  });

  it('blocks preview submission when graph validation finds issues', async () => {
    // Invalid node configuration must never reach the preview API.
    useGraphStore.getState().setGraph(
      [
        {
          id: 'dataset',
          type: 'custom',
          position: { x: 0, y: 0 },
          data: { definitionType: 'dataset_node', datasetId: 'ds-1' },
        },
        {
          id: 'orphan-encoding',
          type: 'custom',
          position: { x: 100, y: 0 },
          data: { definitionType: 'encoding', method: 'label', columns: ['status'] },
        },
      ],
      [],
    );

    const { runPipelinePreview } = await import('../../../../core/api/client');
    const previewSpy = vi.mocked(runPipelinePreview);

    const { result } = renderHook(() => useRunControls());
    await act(async () => {
      await result.current.handleRun();
    });

    expect(previewSpy).not.toHaveBeenCalled();
  });

  it('blocks experiment submission when graph validation finds issues', async () => {
    // Invalid graphs must not create background jobs.
    useGraphStore.getState().setGraph(
      [
        {
          id: 'dataset',
          type: 'custom',
          position: { x: 0, y: 0 },
          data: { definitionType: 'dataset_node', datasetId: 'ds-1' },
        },
        {
          id: 'orphan-imputation',
          type: 'custom',
          position: { x: 100, y: 0 },
          data: {
            definitionType: 'imputation_node',
            columns: ['feature_a'],
            method: 'simple',
            strategy: 'mean',
          },
        },
      ],
      [],
    );

    const { jobsApi } = await import('../../../../core/api/jobs');
    const runSpy = vi.mocked(jobsApi.runPipeline);

    const { result } = renderHook(() => useRunControls());
    await act(async () => {
      await result.current.handleRunAll();
    });

    expect(runSpy).not.toHaveBeenCalled();
  });

  it('keeps preview failure and completion distinct and prevents duplicate requests', async () => {
    // A second activation during a pending preview must not overwrite feedback or submit again.
    useGraphStore.getState().setGraph([
      { id: 'dataset', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'ds-1' } },
      { id: 'drop', position: { x: 200, y: 0 }, data: { definitionType: 'drop_missing_columns', columns: ['id'], missing_threshold: 0 } },
    ], [{ id: 'edge', source: 'dataset', sourceHandle: 'data', target: 'drop', targetHandle: 'in' }]);
    const { runPipelinePreview } = await import('../../../../core/api/client');
    let resolve!: (value: Awaited<ReturnType<typeof runPipelinePreview>>) => void;
    vi.mocked(runPipelinePreview).mockReturnValueOnce(new Promise(done => { resolve = done; }));
    const { result } = renderHook(() => useRunControls());
    let pending!: Promise<void>;
    act(() => {
      pending = result.current.handleRun();
      void result.current.handleRun();
    });
    expect(runPipelinePreview).toHaveBeenCalledTimes(1);
    expect(result.current.isRunning).toBe(true);
    await act(async () => { resolve({ pipeline_id: 'p', status: 'failed', node_results: {}, preview_data: null, recommendations: [] }); await pending; });
    expect(result.current.isRunning).toBe(false);
    expect(useNotificationsStore.getState().items[0]?.message).toContain('Preview failed');
    vi.mocked(runPipelinePreview).mockResolvedValueOnce({ pipeline_id: 'p', status: 'success', node_results: {}, preview_data: null, recommendations: [] });
    await act(async () => { await result.current.handleRun(); });
    expect(useNotificationsStore.getState().items).toHaveLength(0);
    expect(useGraphStore.getState().executionResult?.status).toBe('success');
  });

  it('explains that an empty canvas needs a dataset for both run actions', async () => {
    // Empty graphs have no node validation issues, so the dataset guard must explain the block.
    const { result } = renderHook(() => useRunControls());
    expect(result.current.canRunPreview).toBe(false);
    expect(result.current.experimentBlockReason).toContain('Connect a model');
    await act(async () => {
      await result.current.handleRun();
      await result.current.handleRunAll();
    });
    expect(useNotificationsStore.getState().items).toEqual(expect.arrayContaining([
      expect.objectContaining({ message: 'Preview blocked. Add a dataset node and select a dataset.' }),
      expect.objectContaining({ message: 'Experiments blocked. Add a dataset node and select a dataset.' }),
    ]));
    expect(runPipelinePreview).not.toHaveBeenCalled();
    expect(jobsApi.runPipeline).not.toHaveBeenCalled();
  });

  it('requires a selected, connected dataset before declaring preview available', () => {
    // Preview readiness must follow graph edits instead of remaining stale after selection or wiring.
    const { result } = renderHook(() => useRunControls());
    act(() => useGraphStore.setState({ nodes: [{ id: 'dataset', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node' } }] }));
    expect(result.current.canRunPreview).toBe(false);
    const graph = previewGraph();
    act(() => useGraphStore.setState({ nodes: graph.nodes.slice(0, 1), edges: [] }));
    expect(result.current.canRunPreview).toBe(false);
    act(() => useGraphStore.setState(graph));
    expect(result.current.canRunPreview).toBe(true);
    expect(result.current.experimentBlockReason).toContain('Connect a model');
  });

  it.each([1, 2])('counts %i validation issues and directs blocked runs to results', async count => {
    // The review summary and execution feedback must identify how many fixes are required.
    useGraphStore.setState({
      nodes: Array.from({ length: count }, (_, index) => ({
        id: `unknown-${index}`, position: { x: index * 100, y: 0 },
        data: { definitionType: 'unknown_node', label: `Unknown ${index}` },
      })),
    });
    const { result } = renderHook(() => useRunControls());
    expect(result.current.experimentBlockReason).toContain(`Fix ${count} validation issue${count === 1 ? '' : 's'} first.`);
    await act(async () => { await result.current.handleRun(); });
    expect(useViewStore.getState().isResultsPanelExpanded).toBe(true);
    expect(useGraphStore.getState().executionResult).toBeNull();
    expect(useNotificationsStore.getState().items[0]?.message)
      .toBe(`Preview blocked. Review ${count} validation issue${count === 1 ? '' : 's'}.`);
    await act(async () => { await result.current.handleRunAll(); });
    expect(useNotificationsStore.getState().items.some(item => item.message.startsWith('Experiments blocked.'))).toBe(true);
    expect(jobsApi.runPipeline).not.toHaveBeenCalled();
  });

  it.each([true, false])('lists connected models and detects branches sharing a parent: %s', sharedParent => {
    // Disconnected models stay out of the review while training and tuning retain their own labels.
    const graph = previewGraph();
    graph.nodes.push(
      { id: 'first', position: { x: 400, y: 0 }, data: { definitionType: 'classification', label: 'Classifier', model_type: 'random_forest_classifier' } },
      { id: 'second', position: { x: 400, y: 200 }, data: { definitionType: 'classification', label: 'Classifier', run_mode: 'advanced' } },
      { id: 'unconnected', position: { x: 400, y: 400 }, data: { definitionType: 'classification', label: 'Unused classifier' } },
    );
    graph.edges.push(
      { id: 'first-input', source: 'drop', target: 'first' },
      { id: 'second-input', source: sharedParent ? 'drop' : 'dataset', target: 'second' },
    );
    useGraphStore.setState(graph);
    const { result } = renderHook(() => useRunControls());
    expect(result.current.hasMultipleBranches).toBe(true);
    expect(result.current.experimentModels).toEqual([
      { id: 'first', name: 'Classifier (1)', model: 'random forest classifier', action: 'Train' },
      { id: 'second', name: 'Classifier (2)', model: 'Select a model', action: 'Tune' },
    ]);
  });

  it('filters data-preview sinks and their edges from the submitted preprocessing graph', async () => {
    // Inspection-only nodes must not become executable steps or dangling input references.
    const graph = previewGraph();
    graph.nodes.push({ id: 'inspect', position: { x: 300, y: 200 }, data: { definitionType: 'data_preview' } });
    graph.edges.push(
      { id: 'inspect-input', source: 'drop', target: 'inspect' },
      { id: 'inspect-output', source: 'inspect', target: 'drop' },
    );
    useGraphStore.setState(graph);
    vi.mocked(runPipelinePreview).mockResolvedValueOnce({ pipeline_id: 'p', status: 'success', node_results: {}, preview_data: null, recommendations: [] });
    const { result } = renderHook(() => useRunControls());
    await act(async () => { await result.current.handleRun(); });
    expect(runPipelinePreview).toHaveBeenCalledOnce();
    const submitted = vi.mocked(runPipelinePreview).mock.calls[0]![0];
    expect(submitted.nodes.map(node => node.node_id)).toEqual(['dataset', 'drop']);
    expect(JSON.stringify(submitted)).not.toContain('inspect');
    expect(useGraphStore.getState().executionResult?.status).toBe('success');
  });

  it.each([new Error('Backend unavailable'), 'Connection interrupted'])('keeps rejected preview details visible and allows retry: %s', async failure => {
    // Errors must release the pending guard and preserve the actual failure for diagnostics.
    useGraphStore.setState(previewGraph());
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.mocked(runPipelinePreview).mockRejectedValueOnce(failure);
    const { result } = renderHook(() => useRunControls());
    await act(async () => { await result.current.handleRun(); });
    expect(result.current.isRunning).toBe(false);
    expect(useGraphStore.getState().lastRunError).toBe(failure instanceof Error ? failure.message : failure);
    expect(useViewStore.getState().isResultsPanelExpanded).toBe(true);
    expect(useNotificationsStore.getState().items[0]?.message).toContain('Preview failed');
    vi.mocked(runPipelinePreview).mockResolvedValueOnce({ pipeline_id: 'retry', status: 'success', node_results: {}, preview_data: null, recommendations: [] });
    await act(async () => { await result.current.handleRun(); });
    expect(useGraphStore.getState().lastRunError).toBeNull();
    expect(useGraphStore.getState().executionResult?.pipeline_id).toBe('retry');
  });

  it('blocks direct and shortcut submissions in read-only mode', async () => {
    // Imperative callbacks must honor the same read-only protection as hidden toolbar buttons.
    useGraphStore.setState(previewGraph());
    useViewStore.setState({ readOnlyOverride: 'on' });
    const { result } = renderHook(() => useRunControls());
    await act(async () => {
      await result.current.handleRun();
      await result.current.handleRunAll();
      window.dispatchEvent(new CustomEvent(RUN_PREVIEW_EVENT));
    });
    expect(runPipelinePreview).not.toHaveBeenCalled();
    expect(jobsApi.runPipeline).not.toHaveBeenCalled();
    expect(useNotificationsStore.getState().items).toHaveLength(0);
  });

  it('ignores experiment submission and repeated keyboard shortcuts during a pending preview', async () => {
    // A pending preview must not race background submission or another keyboard activation.
    useGraphStore.setState(previewGraph());
    let resolve!: (value: Awaited<ReturnType<typeof runPipelinePreview>>) => void;
    vi.mocked(runPipelinePreview).mockReturnValueOnce(new Promise(done => { resolve = done; }));
    const { result, unmount } = renderHook(() => useRunControls());
    act(() => window.dispatchEvent(new CustomEvent(RUN_PREVIEW_EVENT)));
    await act(async () => {
      window.dispatchEvent(new CustomEvent(RUN_PREVIEW_EVENT));
      await result.current.handleRunAll();
    });
    expect(runPipelinePreview).toHaveBeenCalledOnce();
    expect(jobsApi.runPipeline).not.toHaveBeenCalled();
    await act(async () => {
      resolve({ pipeline_id: 'p', status: 'success', node_results: {}, preview_data: null, recommendations: [] });
    });
    unmount();
    act(() => window.dispatchEvent(new CustomEvent(RUN_PREVIEW_EVENT)));
    expect(runPipelinePreview).toHaveBeenCalledOnce();
  });

  it.each([
    { jobIds: [], expectedIds: ['job-1'], message: '1 experiment submitted' },
    { jobIds: ['job-1'], expectedIds: ['job-1'], message: '1 experiment submitted' },
    { jobIds: ['job-1', 'job-2'], expectedIds: ['job-1', 'job-2'], message: '2 experiments submitted' },
  ])('selects submitted job scope for job ids $jobIds', async ({ jobIds, expectedIds, message }) => {
    // Both legacy single-job and parallel responses must open the exact submitted run in Jobs.
    useGraphStore.setState(previewGraph());
    vi.mocked(jobsApi.runPipeline).mockResolvedValueOnce({
      message: 'Submitted', pipeline_id: 'pipeline', job_id: 'job-1', job_ids: jobIds,
    });
    const { result } = renderHook(() => useRunControls());
    await act(async () => { await result.current.handleRunAll(); });
    expect(jobsApi.runPipeline).toHaveBeenCalledWith(expect.objectContaining({ job_type: 'training' }));
    expect(useJobStore.getState().inspectedRun).toEqual({ label: 'Experiments', jobIds: expectedIds });
    expect(useJobStore.getState().isDrawerOpen).toBe(true);
    expect(useJobStore.getState().startPolling).toHaveBeenCalledOnce();
    expect(useNotificationsStore.getState().items[0]).toMatchObject({
      message,
      action: { type: 'jobs', run: { label: 'Experiments', jobIds: expectedIds } },
    });
    if (jobIds.length > 1) {
      expect(useJobStore.getState().activeParallelRun).toMatchObject({ jobIds: ['job-1', 'job-2'], startedAt: expect.any(String) });
    } else {
      expect(useJobStore.getState().activeParallelRun).toBeNull();
    }
    expect(runPipelinePreview).not.toHaveBeenCalled();
    expect(result.current.isRunningAll).toBe(false);
  });

  /** Backend leakage corrections must remain visible when Run all is rejected before queuing. */
  it('shows the HTTP 400 leakage detail when experiment submission fails', async () => {
    useGraphStore.setState(previewGraph());
    const detail = "Data leakage risk: node 'scale' (StandardScaler) fits on unsplit data. Move it after the train/test splitter.";
    vi.mocked(jobsApi.runPipeline).mockRejectedValueOnce(Object.assign(new Error(detail), {
      response: { status: 400, data: { detail } },
    }));
    const { result } = renderHook(() => useRunControls());
    await act(async () => { await result.current.handleRunAll(); });
    expect(result.current.isRunningAll).toBe(false);
    expect(useJobStore.getState().isDrawerOpen).toBe(false);
    expect(useJobStore.getState().startPolling).not.toHaveBeenCalled();
    expect(useNotificationsStore.getState().items[0]?.message).toContain(detail);
  });

  /** Unknown submission failures must retain actionable fallback feedback. */
  it('shows fallback feedback when experiment submission rejects without a message', async () => {
    useGraphStore.setState(previewGraph());
    vi.mocked(jobsApi.runPipeline).mockRejectedValueOnce(new Error());
    const { result } = renderHook(() => useRunControls());
    await act(async () => { await result.current.handleRunAll(); });
    expect(result.current.isRunningAll).toBe(false);
    expect(useNotificationsStore.getState().items[0]?.message)
      .toBe('Experiment submission failed. Check your connection and settings, then try again.');
  });

  it('rejects duplicate experiment submission and releases the guard after failure', async () => {
    // Connection failures must be recoverable without opening Jobs for work that was never queued.
    useGraphStore.setState(previewGraph());
    let reject!: (reason: Error) => void;
    vi.mocked(jobsApi.runPipeline).mockReturnValueOnce(new Promise((_, fail) => { reject = fail; }));
    const { result } = renderHook(() => useRunControls());
    let pending!: Promise<void>;
    act(() => {
      pending = result.current.handleRunAll();
      void result.current.handleRunAll();
    });
    expect(result.current.isRunningAll).toBe(true);
    expect(jobsApi.runPipeline).toHaveBeenCalledOnce();
    await act(async () => { reject(new Error('Offline')); await pending; });
    expect(result.current.isRunningAll).toBe(false);
    expect(useJobStore.getState().isDrawerOpen).toBe(false);
    expect(useJobStore.getState().inspectedRun).toBeNull();
    expect(useNotificationsStore.getState().items[0]?.message).toContain('Experiment submission failed');
    vi.mocked(jobsApi.runPipeline).mockResolvedValueOnce({ message: 'Submitted', pipeline_id: 'p', job_id: 'retry-job', job_ids: [] });
    await act(async () => { await result.current.handleRunAll(); });
    expect(useJobStore.getState().inspectedRun?.jobIds).toEqual(['retry-job']);
    expect(useNotificationsStore.getState().items.some(item => item.message.includes('submission failed'))).toBe(false);
  });

  /** Local safety failures must leave earlier preview data accessible without expanding the panel. */
  it.each(['handleRun', 'handleRunAll'] as const)('keeps leakage-only %s feedback on the canvas', async action => {
    useGraphStore.setState({
      nodes: [
        { id: 'dataset', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'ds-1' } },
        { id: 'imputer', position: { x: 100, y: 0 }, data: { definitionType: 'imputation_node', columns: ['value'], strategy: 'mean' } },
        { id: 'split', position: { x: 200, y: 0 }, data: { definitionType: 'TrainTestSplitter', test_size: 0.2, validation_size: 0, random_state: 42, stratify: false, shuffle: true } },
      ],
      edges: [{ id: 'e1', source: 'dataset', target: 'imputer' }, { id: 'e2', source: 'imputer', target: 'split' }],
      executionResult: { pipeline_id: 'earlier', status: 'success', node_results: {}, preview_data: [{ value: 3 }], recommendations: [] },
    });
    const { result } = renderHook(() => useRunControls());
    await act(async () => { await result.current[action](); });
    expect(useViewStore.getState().isResultsPanelExpanded).toBe(false);
    expect(useGraphStore.getState().executionResult?.pipeline_id).toBe('earlier');
    expect(useNotificationsStore.getState().items[0]?.message).toMatch(/marked nodes/i);
    expect(runPipelinePreview).not.toHaveBeenCalled();
    expect(jobsApi.runPipeline).not.toHaveBeenCalled();
  });

  /** A backend safety rejection must preserve prior data and never become a results-panel error. */
  it.each(['handleRun', 'handleRunAll'] as const)('routes rejected %s safety details to the canvas notice', async action => {
    useGraphStore.setState({ ...previewGraph(), executionResult: {
      pipeline_id: 'earlier', status: 'success', node_results: {}, preview_data: [{ value: 3 }], recommendations: [],
    } });
    const detail = 'Per-fold preprocessing refit skipped: payload reconstruction failed; CV/tuning scores may be optimistically biased.';
    vi.mocked(runPipelinePreview).mockRejectedValueOnce({ response: { data: { detail } } });
    vi.mocked(jobsApi.runPipeline).mockRejectedValueOnce({ response: { data: { detail } } });
    const { result } = renderHook(() => useRunControls());
    await act(async () => { await result.current[action](); });
    expect(useViewStore.getState().leakageNotice?.message).toBe(detail);
    expect(useViewStore.getState().isResultsPanelExpanded).toBe(false);
    expect(useGraphStore.getState().lastRunError).toBeNull();
    expect(useGraphStore.getState().executionResult?.pipeline_id).toBe('earlier');
  });

  /** Failed response payloads need the same safety routing as thrown HTTP failures. */
  it('diverts a failed leakage preview response without retaining it as results', async () => {
    useGraphStore.setState(previewGraph());
    const detail = "Data leakage risk: node 'scale' fits on unsplit data.";
    vi.mocked(runPipelinePreview).mockResolvedValueOnce({
      pipeline_id: 'rejected', status: 'failed', node_results: { scale: { status: 'failed', error: detail } },
      preview_data: null, recommendations: [],
    });
    const { result } = renderHook(() => useRunControls());
    await act(async () => { await result.current.handleRun(); });
    expect(useViewStore.getState().leakageNotice?.message).toBe(detail);
    expect(useViewStore.getState().isResultsPanelExpanded).toBe(false);
    expect(useGraphStore.getState().executionResult).toBeNull();
    expect(useNodeInspectionStore.getState().receipt).toBeNull();
  });

  /** A late response must retain its submitted graph signature so edits can invalidate it. */
  it('clears an old notice on retry and tags late errors with the submitted graph', async () => {
    const graph = previewGraph();
    useGraphStore.setState(graph);
    useViewStore.setState({ leakageNotice: { message: 'Previous safety failure', graphSignature: 'old' } });
    let reject!: (reason: Error) => void;
    vi.mocked(runPipelinePreview).mockReturnValueOnce(new Promise((_, fail) => { reject = fail; }));
    const { result } = renderHook(() => useRunControls());
    let pending!: Promise<void>;
    act(() => { pending = result.current.handleRun(); });
    expect(useViewStore.getState().leakageNotice).toBeNull();
    act(() => useGraphStore.setState({ nodes: graph.nodes.map(node => ({ ...node, data: { ...node.data, datasetId: 'edited' } })) }));
    await act(async () => { reject(new Error('Data leakage risk: unsafe preprocessing.')); await pending; });
    expect(useViewStore.getState().leakageNotice?.graphSignature).toContain('ds-1');
    expect(useViewStore.getState().leakageNotice?.graphSignature).not.toContain('edited');
  });
});
