import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useRunControls } from './useRunControls';
import { useGraphStore } from '../../../../core/store/useGraphStore';
import { initializeRegistry } from '../../../../core/registry/init';
import { useJobStore } from '../../../../core/store/useJobStore';

vi.mock('../../../../core/api/client', () => ({
  runPipelinePreview: vi.fn(),
}));

vi.mock('../../../../core/api/jobs', () => ({
  jobsApi: {
    runPipeline: vi.fn(),
  },
}));

vi.mock('../../../../core/toast', () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
}));

describe('useRunControls', () => {
  beforeAll(() => initializeRegistry());

  beforeEach(() => {
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
});
