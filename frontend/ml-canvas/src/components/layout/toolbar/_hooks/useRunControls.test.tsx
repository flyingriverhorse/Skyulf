import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useRunControls } from './useRunControls';
import { useGraphStore } from '../../../../core/store/useGraphStore';
import { initializeRegistry } from '../../../../core/registry/init';
import { useJobStore } from '../../../../core/store/useJobStore';
import { useNotificationsStore } from '../../../../core/store/useNotificationsStore';

vi.mock('../../../../core/api/client', () => ({
  runPipelinePreview: vi.fn(),
}));

vi.mock('../../../../core/api/jobs', () => ({
  jobsApi: {
    runPipeline: vi.fn(),
  },
}));

describe('useRunControls', () => {
  beforeAll(() => initializeRegistry());

  beforeEach(() => {
    vi.clearAllMocks();
    useNotificationsStore.getState().clear();
    useGraphStore.setState({
      nodes: [],
      edges: [],
      executionResult: null,
      lastRunError: null,
    });
    useJobStore.setState({
      jobs: [],
      activeParallelRun: null,
    });
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

  it('blocks preview submission when graph validation finds issues', async () => {
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
});
